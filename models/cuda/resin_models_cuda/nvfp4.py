"""ModelOpt NVFP4 adapters shared by Blackwell model runners."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import Tensor, nn

from resin.backend.loading import prepare_model


def select_nvfp4_backend(
    requested: str, cudnn_version: int | None
) -> Literal["cudnn", "cute-dsl"]:
    if requested == "cudnn" or requested == "cute-dsl":
        return requested
    if requested == "auto":
        return "cudnn" if (cudnn_version or 0) >= 92301 else "cute-dsl"
    raise ValueError("nvfp4_backend must be 'auto', 'cudnn', or 'cute-dsl'")


def swizzle_block_scales(scales: Tensor) -> Tensor:
    """Convert linear ModelOpt scales to the canonical 128x4 layout."""

    if scales.ndim != 2:
        raise ValueError("NVFP4 block scales must have shape [out, in / 16]")
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError("NVFP4 block scales must use float8_e4m3fn")
    rows, columns = scales.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    padded = torch.zeros(
        (1, padded_rows, padded_columns), dtype=scales.dtype, device=scales.device
    )
    padded[:, :rows, :columns] = scales
    return (
        padded.reshape(1, padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(padded_rows, padded_columns)
    )


# Opt-in: prefill-sized NVFP4 GEMMs quantize activations to FP4 (W4A4).
PREFILL_W4A4 = os.environ.get("RESIN_NVFP4_PREFILL_W4A4", "0") == "1"
W4A4_MIN_ROWS = int(os.environ.get("RESIN_NVFP4_W4A4_MIN_ROWS", "256"))


class Nvfp4LinearWeights(nn.Module):
    """ModelOpt checkpoint storage; replaced before serving or graph capture."""

    weight: Tensor
    weight_scale: Tensor
    input_scale: Tensor
    weight_scale_2: Tensor

    def __init__(self, in_features: int, out_features: int, group_size: int = 16):
        super().__init__()
        if in_features % group_size:
            raise ValueError("NVFP4 input width must be divisible by group_size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer(
            "weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                out_features, in_features // group_size, dtype=torch.float8_e4m3fn
            ),
        )
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
        self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32))

    def forward(self, *args, **kwargs):
        raise RuntimeError("NVFP4 checkpoint weights must be lowered before inference")


class Nvfp4Linear(nn.Module):
    """Execution-only W4A16 linear with a fixed kernel and prepared layout."""

    weight: Tensor
    weight_scale: Tensor
    weight_scale_2: Tensor | None

    def __init__(
        self,
        source: Nvfp4LinearWeights,
        backend: Literal["cudnn", "cute-dsl"],
        *,
        prepare: Callable[..., tuple[Tensor, Tensor, Tensor | None]] | None = None,
        mm: Callable[..., Tensor] | None = None,
        w4a4: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.group_size = source.group_size
        if prepare is None or mm is None:
            from flashinfer.gemm import mm_bf16_fp4, prepare_bf16_fp4_weights

            prepare = prepare_bf16_fp4_weights
            mm = mm_bf16_fp4
        scales = swizzle_block_scales(source.weight_scale)
        weight, scales, alpha = prepare(
            source.weight,
            scales,
            source.weight_scale_2.float().reshape(1),
            backend=backend,
            block_size=self.group_size,
        )
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scales)
        self.register_buffer("weight_scale_2", alpha)
        self.backend = backend
        self._mm = mm
        self.w4a4_weight: Tensor | None = None
        if w4a4:
            # Optional FP4-activation path for large (prefill) row counts. It
            # uses the checkpoint's calibrated activation scale and FP4 tensor
            # cores (~4.7x the W4A16 rate on sm_120) at lower precision.
            self.w4a4_weight = source.weight.detach().clone()
            self.w4a4_scales = swizzle_block_scales(source.weight_scale)
            self.w4a4_input = (1.0 / source.input_scale.float()).reshape(1)
            self.w4a4_alpha = (
                source.input_scale.float() * source.weight_scale_2.float()
            ).reshape(1)

    def forward(self, hidden: Tensor, *, out: Tensor | None = None) -> Tensor:
        if hidden.dtype != torch.bfloat16:
            raise TypeError("NVFP4 W4A16 linears require bfloat16 activations")
        shape = (*hidden.shape[:-1], self.out_features)
        flat = hidden.reshape(-1, self.in_features)
        flat_out = None if out is None else out.reshape(-1, self.out_features)
        if self.w4a4_weight is not None and flat.shape[0] >= W4A4_MIN_ROWS:
            from flashinfer import fp4_quantize, mm_fp4

            packed, packed_scales = fp4_quantize(flat, self.w4a4_input, 16, False, True)
            result = mm_fp4(
                packed,
                self.w4a4_weight.T,
                packed_scales,
                self.w4a4_scales.T,
                self.w4a4_alpha,
                torch.bfloat16,
                out=flat_out,
            )
            return result.reshape(shape)
        result = self._mm(
            flat,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            backend=self.backend,
            out_dtype=torch.bfloat16,
            out=flat_out,
            block_size=self.group_size,
        )
        return result.reshape(shape)


def prepare_nvfp4_modules(
    model: nn.Module, backend: Literal["cudnn", "cute-dsl"]
) -> nn.Module:
    """Lower checkpoint containers; already prepared modules pass through."""

    w4a4 = PREFILL_W4A4
    return prepare_model(
        model,
        lambda module: (
            Nvfp4Linear(module, backend, w4a4=w4a4)
            if isinstance(module, Nvfp4LinearWeights)
            else module
        ),
    )


def nvfp4_weights_from_linear(linear: nn.Linear) -> Nvfp4LinearWeights:
    """Quantize a BF16 linear into ModelOpt-layout NVFP4 checkpoint storage.

    One per-tensor FP32 scale maps the weight amax to E4M3 x E2M1 range, and
    each 16-element block carries an E4M3 scale, matching ModelOpt exports.
    """

    from flashinfer import fp4_quantize

    weight = linear.weight.detach().to(torch.bfloat16).contiguous()
    out_features, in_features = weight.shape
    global_scale = (448.0 * 6.0) / weight.abs().amax().float().clamp_min(1e-12)
    packed, scales = fp4_quantize(weight, global_scale.reshape(1), 16, False, False)
    source = Nvfp4LinearWeights(in_features, out_features).to(weight.device)
    source.weight.copy_(packed.view(torch.uint8).reshape(out_features, -1))
    source.weight_scale.copy_(
        scales.view(torch.float8_e4m3fn).reshape(out_features, in_features // 16)
    )
    source.weight_scale_2.copy_(1.0 / global_scale)
    source.input_scale.fill_(1.0)
    return source


def quantize_linears_nvfp4(
    model: nn.Module, backend: Literal["cudnn", "cute-dsl"]
) -> nn.Module:
    """Replace every BF16 `nn.Linear` in `model` with an NVFP4 W4A16 linear."""

    return prepare_model(
        model,
        lambda module: (
            Nvfp4Linear(nvfp4_weights_from_linear(module), backend)
            if type(module) is nn.Linear
            and module.bias is None
            and module.in_features % 16 == 0
            else module
        ),
    )


def autotune_nvfp4_modules(model: nn.Module, rows: Iterable[int]) -> int:
    """Profile FlashInfer W4A16 tactics for every linear shape and row count.

    Without tuning, the cute-DSL dispatch uses its streaming GEMV only at one
    row and an MMA heuristic above it, which on an RTX PRO 6000 reads NVFP4
    weights at ~0.85 TB/s for 2-16 rows versus ~1.37 TB/s for the tuned
    tactics. Speculative verification and small decode batches live in that
    range. Choices are cached in FlashInfer's process-wide autotuner, so this
    must run before CUDA graph capture. Returns the number of tuned shapes.
    """

    from flashinfer.autotuner import autotune

    counts = sorted({int(count) for count in rows if int(count) > 0})
    seen: set[tuple[int, int]] = set()
    with torch.inference_mode(), autotune(True):
        for module in model.modules():
            if not isinstance(module, Nvfp4Linear):
                continue
            shape = (module.in_features, module.out_features)
            if shape in seen:
                continue
            seen.add(shape)
            for count in counts:
                hidden = torch.zeros(
                    count,
                    module.in_features,
                    dtype=torch.bfloat16,
                    device=module.weight.device,
                )
                module(hidden)
    torch.cuda.synchronize()
    return len(seen)


def quantize_nvfp4(
    hidden: Tensor,
    input_scale_quant: Tensor,
    *,
    quantize: Callable[..., tuple[Tensor, Tensor]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize BF16 rows into TRT-LLM's packed NVFP4 activation layout."""

    if hidden.ndim != 2 or hidden.dtype != torch.bfloat16:
        raise TypeError("NVFP4 MoE activations must be a BF16 matrix")
    if hidden.shape[1] % 16:
        raise ValueError("NVFP4 activation width must be divisible by 16")
    if quantize is None:
        from flashinfer import fp4_quantize

        quantize = fp4_quantize
    packed, scales = quantize(hidden, input_scale_quant, 16, False, False)
    return (
        packed.reshape(hidden.shape[0], hidden.shape[1] // 2),
        scales.view(torch.float8_e4m3fn).reshape(
            hidden.shape[0], hidden.shape[1] // 16
        ),
    )


def prepare_nvfp4_moe_weights(
    w13: Tensor,
    w13_scales: Tensor,
    w2: Tensor,
    w2_scales: Tensor,
    *,
    reorder: Callable[[Tensor], Tensor] | None = None,
    shuffle: Callable[[Tensor, int], Tensor] | None = None,
    interleave: Callable[[Tensor], Tensor] | None = None,
) -> None:
    """Convert linear ModelOpt expert weights in place for TRT-LLM MoE."""

    tensors = (w13, w13_scales, w2, w2_scales)
    if all(getattr(tensor, "_resin_nvfp4_moe_layout", False) for tensor in tensors):
        return
    if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("NVFP4 MoE weights must have matching expert dimensions")
    experts, doubled_intermediate, packed_hidden = w13.shape
    hidden = packed_hidden * 2
    if doubled_intermediate % 2 or w2.shape != (
        experts,
        hidden,
        doubled_intermediate // 4,
    ):
        raise ValueError("NVFP4 MoE weights have incompatible hidden dimensions")
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise TypeError("NVFP4 MoE weights must use packed uint8 storage")
    if (
        w13_scales.dtype != torch.float8_e4m3fn
        or w2_scales.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("NVFP4 MoE scales must use float8_e4m3fn")
    if w13_scales.shape != (experts, doubled_intermediate, hidden // 16):
        raise ValueError("NVFP4 W13 scales have the wrong shape")
    if w2_scales.shape != (experts, hidden, doubled_intermediate // 32):
        raise ValueError("NVFP4 W2 scales have the wrong shape")
    if reorder is None and shuffle is None:
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        permutation_cache = {}
        w13_rows = _maybe_get_cached_w3_w1_permute_indices(
            permutation_cache, w13[0], 128
        )
        w13_scale_rows = _maybe_get_cached_w3_w1_permute_indices(
            permutation_cache,
            w13_scales[0].view(torch.uint8),
            128,
            num_elts_per_sf=16,
        )
        w2_rows = get_w2_permute_indices_with_cache(permutation_cache, w2[0], 128)
        w2_scale_rows = get_w2_permute_indices_with_cache(
            permutation_cache,
            w2_scales[0].view(torch.uint8),
            128,
            num_elts_per_sf=16,
        )
    elif reorder is not None and shuffle is not None:

        def permutation(rows: int, *, gated: bool, device: torch.device) -> Tensor:
            probe = torch.arange(rows, dtype=torch.long, device=device).view(rows, 1)
            if gated:
                probe = reorder(probe)
            return shuffle(probe, 128).flatten()

        w13_rows = permutation(doubled_intermediate, gated=True, device=w13.device)
        w13_scale_rows = w13_rows
        w2_rows = permutation(hidden, gated=False, device=w2.device)
        w2_scale_rows = w2_rows
    else:
        raise ValueError("reorder and shuffle must be provided together")
    if interleave is None:
        from flashinfer import nvfp4_block_scale_interleave

        interleave = nvfp4_block_scale_interleave
    with torch.no_grad():
        # Transform one expert at a time to avoid another multi-GB allocation.
        for expert in range(experts):
            w13[expert].copy_(w13[expert].index_select(0, w13_rows))
            w2[expert].copy_(w2[expert].index_select(0, w2_rows))
            w13_scale_bytes = w13_scales[expert].view(torch.uint8)
            w2_scale_bytes = w2_scales[expert].view(torch.uint8)
            w13_scale_bytes.copy_(
                interleave(
                    w13_scales[expert]
                    .index_select(0, w13_scale_rows)
                    .contiguous()
                    .view(torch.uint8)
                ).reshape_as(w13_scale_bytes)
            )
            w2_scale_bytes.copy_(
                interleave(
                    w2_scales[expert]
                    .index_select(0, w2_scale_rows)
                    .contiguous()
                    .view(torch.uint8)
                ).reshape_as(w2_scale_bytes)
            )
    for tensor in tensors:
        setattr(tensor, "_resin_nvfp4_moe_layout", True)


__all__ = [
    "Nvfp4Linear",
    "Nvfp4LinearWeights",
    "prepare_nvfp4_modules",
    "prepare_nvfp4_moe_weights",
    "quantize_nvfp4",
    "swizzle_block_scales",
    "select_nvfp4_backend",
]
