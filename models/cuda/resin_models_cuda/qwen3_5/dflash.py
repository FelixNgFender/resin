"""DFlash2 block drafter for Qwen3.8 (z-lab/Qwen3.8-27B-DFlash2).

DFlash2 drafts a whole block in one forward: the committed anchor followed by
mask tokens runs through five non-causal, sliding-window attention layers that
also attend to a context cache built from the target's hidden states at five
of its layers. The target's own head scores the block, and a low-rank pairwise
selector picks one candidate per position from the top-k, conditioned on the
token before it. Drafts only steer verification: the target still samples every
committed token, so draft numerics affect acceptance, never output.

Reference: https://github.com/z-lab/dflash (`DFlash2DraftModel`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import fused

GROUP_SIZE_KEY = "conv_group_size"


def _rms_norm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Standard (Qwen3) RMSNorm with an FP32 reduction."""

    values = hidden.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * weight.float()).to(hidden.dtype)


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _grouped_dynamic_convolve(
    hidden: Tensor, dynamic: Tensor, base: Tensor, group_size: int
) -> Tensor:
    """Causal depthwise convolution along the block with per-token kernels."""

    batch, length, width = hidden.shape
    groups = width // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = (
            blocks
            if offset == 0
            else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        )
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


def _residual_convolve(
    residual: Tensor, hidden: Tensor, dynamic: Tensor, base: Tensor, group_size: int
) -> Tensor:
    return residual + _grouped_dynamic_convolve(hidden, dynamic, base, group_size)


def _norm_rotary(hidden: Tensor, weight: Tensor, cos: Tensor, sin: Tensor, eps: float) -> Tensor:
    """Per-head RMSNorm followed by rotary embedding (query/key path)."""

    normed = _rms_norm(hidden, weight, eps)
    return normed * cos + _rotate_half(normed) * sin


def _silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


def _rotary_tables(positions: Tensor, inv_freq: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    frequencies = positions.float()[..., None] * inv_freq
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    return embeddings.cos().to(dtype)[:, :, None], embeddings.sin().to(dtype)[:, :, None]


def _select_path(
    logits: Tensor,
    projected: Tensor,
    anchors: Tensor,
    predecessor_codebook: Tensor,
    successor_codebook: Tensor,
    top_k: int,
) -> Tensor:
    """Greedy path through the top-k candidates with pairwise scores."""

    unary, candidates = torch.topk(logits.float(), top_k, dim=-1, sorted=False)
    projected = projected.float()
    predecessor = anchors
    path = []
    for position in range(projected.shape[1]):
        left = predecessor_codebook[predecessor].float() * projected[:, position]
        right = successor_codebook[candidates[:, position]].float()
        scores = unary[:, position] + torch.einsum("br,bkr->bk", left, right)
        index = scores.argmax(dim=-1)
        predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
        path.append(predecessor)
    return torch.stack(path, dim=1)


_COMPILED: dict = {}


def _region(function, hidden: Tensor):
    """`function`, compiled with Inductor inside decode graphs (see `fused`)."""

    if not fused.use_fused(hidden):
        return function
    compiled = _COMPILED.get(function)
    if compiled is None:
        compiled = _COMPILED[function] = fused._compile(function)
    return compiled


class _DynamicConv(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, group_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.groups = hidden // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden))
        self.kernel_projection = nn.Linear(
            hidden, 2 * kernel_size * self.groups, bias=False
        )

    def prepare(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, self.groups
        )
        convolved = _region(_grouped_dynamic_convolve, hidden)(
            hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size
        )
        return convolved, dynamic[..., 1, :, :]

    def finish(self, residual: Tensor, hidden: Tensor, dynamic: Tensor) -> Tensor:
        """`residual` plus the second dynamic convolution of `hidden`."""

        return _region(_residual_convolve, hidden)(
            residual, hidden, dynamic, self.base_kernel[1], self.group_size
        )


class _Layer(nn.Module):
    def __init__(self, config: dict, kernel_size: int, group_size: int) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        heads = config["num_attention_heads"]
        kv_heads = config["num_key_value_heads"]
        head_dim = config["head_dim"]
        self.input_layernorm = nn.Parameter(torch.empty(hidden))
        self.post_attention_layernorm = nn.Parameter(torch.empty(hidden))
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = nn.Parameter(torch.empty(head_dim))
        self.k_norm = nn.Parameter(torch.empty(head_dim))
        intermediate = config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.attention_conv = _DynamicConv(hidden, kernel_size, group_size)
        self.mlp_conv = _DynamicConv(hidden, kernel_size, group_size)


class DFlash2Drafter(nn.Module):
    """Block drafter whose KV lives in one Resin cache group of its layers."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        draft = config["dflash_config"]
        self.config = config
        self.block_size = int(draft["block_size"])
        self.mask_token_id = int(draft["mask_token_id"])
        self.target_layer_ids = tuple(int(layer) for layer in draft["target_layer_ids"])
        self.top_k = int(draft["selector_top_k"])
        self.hidden_size = int(config["hidden_size"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.eps = float(config["rms_norm_eps"])
        self.window = int(config.get("sliding_window") or 0)
        # Rotary frequencies are computed on the target device in `load`.
        self.inv_freq: Tensor | None = None
        kernel = int(draft["conv_kernel_size"])
        group = int(draft[GROUP_SIZE_KEY])
        self.layers = nn.ModuleList(
            _Layer(config, kernel, group) for _ in range(config["num_hidden_layers"])
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * self.hidden_size, self.hidden_size, bias=False
        )
        self.hidden_norm = nn.Parameter(torch.empty(self.hidden_size))
        self.norm = nn.Parameter(torch.empty(self.hidden_size))
        rank = int(draft["selector_rank"])
        vocab = int(config["vocab_size"])
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab, rank))
        self.hidden_projection = nn.Linear(self.hidden_size, rank, bias=False)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> DFlash2Drafter:
        from safetensors import safe_open

        config = json.loads((path / "config.json").read_text())
        with torch.device("meta"):
            model = cls(config)
        names = {
            "fc.weight": "fc.weight",
            "hidden_norm.weight": "hidden_norm",
            "norm.weight": "norm",
            "candidate_selector.predecessor_codebook": "predecessor_codebook",
            "candidate_selector.successor_codebook": "successor_codebook",
            "candidate_selector.hidden_projection.weight": "hidden_projection.weight",
        }
        layer_names = {
            "input_layernorm.weight": "input_layernorm",
            "post_attention_layernorm.weight": "post_attention_layernorm",
            "self_attn.q_proj.weight": "q_proj.weight",
            "self_attn.k_proj.weight": "k_proj.weight",
            "self_attn.v_proj.weight": "v_proj.weight",
            "self_attn.o_proj.weight": "o_proj.weight",
            "self_attn.q_norm.weight": "q_norm",
            "self_attn.k_norm.weight": "k_norm",
            "mlp.gate_proj.weight": "gate_proj.weight",
            "mlp.up_proj.weight": "up_proj.weight",
            "mlp.down_proj.weight": "down_proj.weight",
            "attention_conv.base_kernel": "attention_conv.base_kernel",
            "attention_conv.kernel_projection.weight": "attention_conv.kernel_projection.weight",
            "mlp_conv.base_kernel": "mlp_conv.base_kernel",
            "mlp_conv.kernel_projection.weight": "mlp_conv.kernel_projection.weight",
        }
        for index in range(len(model.layers)):
            for source, target in layer_names.items():
                names[f"layers.{index}.{source}"] = f"layers.{index}.{target}"
        state = {}
        with safe_open(str(path / "model.safetensors"), framework="pt") as handle:
            keys = set(handle.keys())
            missing = set(names) - keys
            if missing:
                raise ValueError(f"DFlash2 checkpoint lacks {sorted(missing)[:4]}")
            for source, target in names.items():
                state[target] = handle.get_tensor(source).to(torch.bfloat16)
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to(device).eval()
        theta = float(config["rope_parameters"]["rope_theta"])
        steps = torch.arange(0, model.head_dim, 2, dtype=torch.float32, device=device)
        model.inv_freq = 1.0 / theta ** (steps / model.head_dim)
        return model

    def rotary(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        assert self.inv_freq is not None
        return _region(_rotary_tables, positions[..., None])(positions, self.inv_freq, dtype)

    def _keys(self, layer: _Layer, hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, tokens = hidden.shape[:2]
        keys = layer.k_proj(hidden).view(batch, tokens, self.kv_heads, self.head_dim)
        return _region(_norm_rotary, hidden)(keys, layer.k_norm, cos, sin, self.eps)

    def write_context(
        self,
        cache,
        features: Tensor,
        positions: Tensor,
        slots: tuple[Tensor, Tensor],
    ) -> None:
        """Store context keys/values for target features at `positions`."""

        projected = self.fc(features)
        context = _region(_rms_norm, projected)(projected, self.hidden_norm, self.eps)
        cos, sin = self.rotary(positions, context.dtype)
        pages, offsets = slots
        keys_view, values_view = _draft_views(cache, self.kv_heads, self.head_dim)
        batch, tokens = context.shape[:2]
        for index, layer in enumerate(self.layers):
            keys = self._keys(layer, context, cos, sin)
            values = layer.v_proj(context).view(batch, tokens, self.kv_heads, self.head_dim)
            keys_view[pages + index, offsets] = keys.to(keys_view.dtype)
            values_view[pages + index, offsets] = values.to(values_view.dtype)

    def forward(
        self,
        noise: Tensor,
        positions: Tensor,
        cache,
        attention,
        slots: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Run the block; its own keys/values are written at the block positions."""

        batch, tokens = noise.shape[:2]
        cos, sin = self.rotary(positions, noise.dtype)
        pages, offsets = slots
        keys_view, values_view = _draft_views(cache, self.kv_heads, self.head_dim)
        hidden = noise
        for index, layer in enumerate(self.layers):
            residual = hidden
            normed = _region(_rms_norm, hidden)(hidden, layer.input_layernorm, self.eps)
            normed, finish = layer.attention_conv.prepare(normed)
            queries = layer.q_proj(normed).view(batch, tokens, self.heads, self.head_dim)
            queries = _region(_norm_rotary, hidden)(
                queries, layer.q_norm, cos, sin, self.eps
            )
            keys = self._keys(layer, normed, cos, sin)
            values = layer.v_proj(normed).view(batch, tokens, self.kv_heads, self.head_dim)
            keys_view[pages + index, offsets] = keys.to(keys_view.dtype)
            values_view[pages + index, offsets] = values.to(values_view.dtype)
            attended = attention.run(queries.contiguous(), index)
            output = layer.o_proj(attended.reshape(batch, tokens, -1))
            hidden = layer.attention_conv.finish(residual, output, finish)
            residual = hidden
            normed = _region(_rms_norm, hidden)(
                hidden, layer.post_attention_layernorm, self.eps
            )
            normed, finish = layer.mlp_conv.prepare(normed)
            gated = _region(_silu_mul, normed)(layer.gate_proj(normed), layer.up_proj(normed))
            hidden = layer.mlp_conv.finish(residual, layer.down_proj(gated), finish)
        return _region(_rms_norm, hidden)(hidden, self.norm, self.eps)

    def select(self, hidden: Tensor, logits: Tensor, anchors: Tensor) -> Tensor:
        """Greedy path through the top-k candidates with pairwise scores."""

        return _region(_select_path, hidden)(
            logits,
            self.hidden_projection(hidden),
            anchors,
            self.predecessor_codebook,
            self.successor_codebook,
            self.top_k,
        )


def _draft_views(cache, kv_heads: int, head_dim: int) -> tuple[Tensor, Tensor]:
    """Reinterpret target-shaped KV pages as the drafter's head geometry."""

    keys, values = cache.keys, cache.values
    if keys.shape[-2:] == (kv_heads, head_dim):
        return keys, values
    if keys.shape[-2] * keys.shape[-1] != kv_heads * head_dim:
        raise ValueError("drafter KV rows must match the target's KV row bytes")
    return (
        keys.flatten(-2).unflatten(-1, (kv_heads, head_dim)),
        values.flatten(-2).unflatten(-1, (kv_heads, head_dim)),
    )


__all__ = ["DFlash2Drafter"]
