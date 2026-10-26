"""Compiled pointwise regions for Qwen3.5 decode and speculative verification.

A batch-1 speculative step launches thousands of small kernels; on an RTX PRO
6000 their execution plus graph launch gaps cost as much as the weight reads
of a full forward. Each region below is pure PyTorch, compiled once per decode
graph shape with Inductor so its elementwise chain becomes one or two kernels.
Larger (prefill) inputs keep the eager path, so their kernels and numerics are
unchanged. Inductor evaluates BF16 chains in FP32 and rounds once, so results
may differ from eager in the last BF16 bit.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor

# Decode graphs cover batch x verified-width rows; prefill chunks are larger.
MAX_FUSED_ROWS = 64
ENABLED = os.environ.get("RESIN_QWEN_COMPILE", "1") != "0"


def use_fused(hidden: Tensor) -> bool:
    rows = hidden.numel() // hidden.shape[-1]
    return ENABLED and hidden.is_cuda and rows <= MAX_FUSED_ROWS


def _compile(function):
    import torch._dynamo

    config = torch._dynamo.config
    for name in ("cache_size_limit", "recompile_limit"):
        if hasattr(config, name):
            setattr(config, name, max(getattr(config, name), 256))
    return torch.compile(function, dynamic=False, fullgraph=True)


def _rms_norm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Qwen's one-centered RMSNorm, FP32 internally like FlashInfer's kernel."""

    values = hidden.float()
    scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * scale * (1.0 + weight.float())).to(hidden.dtype)


def _rotary(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    width = cos.shape[-1]
    rotary, passthrough = hidden[..., :width], hidden[..., width:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1)


def _attention_inputs(
    projected_q: Tensor,
    projected_k: Tensor,
    projected_v: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float,
    heads: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: torch.dtype,
):
    batch, tokens = projected_q.shape[:2]
    query, gate = projected_q.reshape(batch, tokens, heads, 2 * head_dim).chunk(
        2, dim=-1
    )
    key = projected_k.reshape(batch, tokens, kv_heads, head_dim)
    value = projected_v.reshape(batch, tokens, kv_heads, head_dim)
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    query = _rotary(_rms_norm(query, q_weight, eps), cos, sin)
    key = _rotary(_rms_norm(key, k_weight, eps), cos, sin)
    return query, gate.contiguous(), key.to(kv_dtype), value.to(kv_dtype)


def _gated_output(output: Tensor, gate: Tensor) -> Tensor:
    batch, tokens = output.shape[:2]
    return output.reshape(batch, tokens, -1) * torch.sigmoid(
        gate.reshape(batch, tokens, -1)
    )


def _delta_gates(
    beta_decay: Tensor, negative_a: Tensor, dt_bias: Tensor, heads: int
) -> tuple[Tensor, Tensor]:
    beta_logits, decay_logits = beta_decay.split(heads, dim=-1)
    beta = beta_logits.sigmoid()
    decay = negative_a * F.softplus(decay_logits.float() + dt_bias)
    return beta, decay


def _short_convolution(
    state: Tensor, projected: Tensor, weight: Tensor, splits: tuple[int, ...]
):
    """Depthwise causal convolution + SiLU, split into contiguous q/k/v.

    Also returns the padded input whose trailing columns are the next state.
    """

    inputs = torch.cat((state, projected.transpose(1, 2)), dim=-1)
    tokens = projected.shape[1]
    taps = weight.shape[-1]
    values = inputs.float()
    kernel = weight[:, 0].float()
    output = sum(
        values[..., tap : tap + tokens] * kernel[:, tap, None] for tap in range(taps)
    )
    mixed = F.silu(output.to(inputs.dtype)).transpose(1, 2)
    return (*(part.contiguous() for part in mixed.split(splits, dim=-1)), inputs)


def _add_rms_norm(
    hidden: Tensor, delta: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """Residual add followed by the next layer's one-centered RMSNorm."""

    total = hidden + delta
    return total, _rms_norm(total, weight, eps)


def _silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


attention_inputs = _compile(_attention_inputs)
gated_output = _compile(_gated_output)
delta_gates = _compile(_delta_gates)
short_convolution = _compile(_short_convolution)
silu_mul = _compile(_silu_mul)
add_rms_norm = _compile(_add_rms_norm)

__all__ = [
    "MAX_FUSED_ROWS",
    "add_rms_norm",
    "attention_inputs",
    "delta_gates",
    "gated_output",
    "short_convolution",
    "silu_mul",
    "use_fused",
]
