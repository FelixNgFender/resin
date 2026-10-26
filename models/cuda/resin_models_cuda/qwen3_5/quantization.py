"""Qwen3.5 ModelOpt FP8 projection modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ModelOptFp8Linear(nn.Module):
    """Tensor-scaled W8A8 linear matching unified ModelOpt tensor names."""

    weight: Tensor
    input_scale: Tensor
    weight_scale: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32))

    def forward(self, hidden: Tensor) -> Tensor:
        if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
            raise TypeError("ModelOpt FP8 linears require CUDA BF16 activations")
        matrix = hidden.reshape(-1, self.in_features)
        quantized = torch.clamp(
            matrix.float() / self.input_scale,
            min=-448.0,
            max=448.0,
        ).to(torch.float8_e4m3fn)
        output = torch._scaled_mm(
            quantized,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        return output.view(*hidden.shape[:-1], self.out_features)


class FusedFp8Projection(nn.Module):
    """FP8 projections sharing an input and activation scale, as one GEMM.

    Small-row calls (decode, speculative verification) run the Triton W8A8
    kernel once over the concatenated weights with per-column scales. Larger
    calls keep the original per-projection cuBLAS path. Source modules are
    rebound to row views, so the weights are stored once.
    """

    weight: Tensor
    input_scale: Tensor
    column_scales: Tensor

    def __init__(self, sources: list[ModelOptFp8Linear]) -> None:
        super().__init__()
        scale = sources[0].input_scale
        if any(
            source.in_features != sources[0].in_features
            or not torch.equal(source.input_scale, scale)
            for source in sources
        ):
            raise ValueError("fused FP8 projections need one input and scale")
        self.in_features = sources[0].in_features
        self.splits = tuple(source.out_features for source in sources)
        weight = torch.cat([source.weight for source in sources])
        self.register_buffer("weight", weight)
        self.register_buffer("input_scale", scale.detach().clone().float())
        self.register_buffer(
            "column_scales",
            torch.cat(
                [
                    source.weight_scale.detach().float().reshape(1).expand(
                        source.out_features
                    )
                    for source in sources
                ]
            ).contiguous(),
        )
        offset = 0
        for source in sources:
            source.weight = weight[offset : offset + source.out_features]
            offset += source.out_features
        # Not submodules: their owners already register them.
        self._sources = tuple(sources)

    def forward(self, hidden: Tensor) -> tuple[Tensor, ...]:
        from resin_models_cuda.fp8_gemv import MAX_ROWS, w8a8_linear

        if hidden.dtype != torch.bfloat16:
            raise TypeError("ModelOpt FP8 linears require CUDA BF16 activations")
        matrix = hidden.reshape(-1, self.in_features)
        if matrix.shape[0] > MAX_ROWS:
            return tuple(source(hidden) for source in self._sources)
        output = w8a8_linear(matrix, self.weight, self.input_scale, self.column_scales)
        return tuple(
            part.reshape(*hidden.shape[:-1], part.shape[-1])
            for part in output.split(self.splits, dim=-1)
        )


class FusedNvfp4Projection(nn.Module):
    """Opt-in: fused FP8 projections re-quantized to one NVFP4 W4A16 GEMM.

    Dequantizes each FP8 source (weight x per-tensor scale) to BF16 and
    quantizes the concatenation to NVFP4 with one global scale. Weights drop
    from 8 to ~4.5 bits (fewer bytes per decode step) while activations stay
    BF16. This changes the model's numerics, so it is off by default.
    """

    def __init__(self, sources: tuple[ModelOptFp8Linear, ...], backend) -> None:
        super().__init__()
        from resin_models_cuda.nvfp4 import Nvfp4Linear, nvfp4_weights_from_linear

        in_features = sources[0].in_features
        self.splits = tuple(source.out_features for source in sources)
        weight = torch.cat(
            [
                (source.weight.float() * source.weight_scale.float()).to(torch.bfloat16)
                for source in sources
            ]
        )
        linear = nn.Linear(
            in_features, weight.shape[0], bias=False, device=weight.device, dtype=torch.bfloat16
        )
        linear.weight.data = weight
        self.linear = Nvfp4Linear(nvfp4_weights_from_linear(linear), backend)

    def forward(self, hidden: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.linear(hidden).split(self.splits, dim=-1))


def requantize_projections_nvfp4(model: nn.Module, backend) -> int:
    """Replace every fused FP8 projection with an NVFP4 one; returns the count."""

    count = 0
    for module in list(model.modules()):
        for name in ("fused_input", "fused_output", "fused_qkv"):
            fused = getattr(module, name, None)
            if isinstance(fused, FusedFp8Projection):
                setattr(module, name, FusedNvfp4Projection(fused._sources, backend))
                for source in fused._sources:
                    # The FP8 copies are no longer read; free their weights.
                    source.weight = source.weight.new_empty(0)
                count += 1
    return count


def warm_fp8_projections(model: nn.Module, max_rows: int) -> int:
    """Compile and autotune every fused FP8 shape before CUDA graph capture."""

    from resin_models_cuda.fp8_gemv import MAX_ROWS

    seen: set[tuple[int, int]] = set()
    rows = [count for count in (1, 17, 33) if count <= min(max_rows, MAX_ROWS)]
    with torch.inference_mode():
        for module in model.modules():
            if not isinstance(module, FusedFp8Projection):
                continue
            shape = (module.in_features, int(module.weight.shape[0]))
            if shape in seen:
                continue
            seen.add(shape)
            for count in rows:
                module(
                    torch.ones(
                        count,
                        module.in_features,
                        dtype=torch.bfloat16,
                        device=module.weight.device,
                    )
                )
    torch.cuda.synchronize()
    return len(seen)


__all__ = ["FusedFp8Projection", "ModelOptFp8Linear", "warm_fp8_projections"]
