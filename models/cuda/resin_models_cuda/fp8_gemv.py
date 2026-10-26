"""Small-row W8A8 FP8 linear with in-kernel static activation quantization.

ModelOpt FP8 projections compute `fp8(x / s_x) @ fp8(W)^T * s_x * s_w`. For
decode and speculative verification (a handful of rows) the eager form costs
four elementwise launches to quantize `x`, then a cuBLASLt kernel that on
consumer Blackwell (sm_120) selects an Ada tile and reads weights at ~70% of
DRAM bandwidth. This Triton kernel quantizes each activation tile as it loads
it, using the same round-to-nearest-even E4M3 conversion and IEEE division,
and streams weights once per output tile. Accumulation is FP32; only the
summation order differs from cuBLAS (outputs matched bit-for-bit in tests).

Weight scales are per output column, so projections that share an input and
its activation scale (Qwen's DeltaNet qkv+z, attention q/k/v) run as one
launch over their concatenated weights with unchanged per-tensor products.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

MAX_ROWS = 16

_CONFIGS = [
    triton.Config({"BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=s)
    for n in (16, 32, 64)
    for k in (128, 256)
    for w in (4, 8)
    for s in (3, 4)
]


@triton.autotune(configs=_CONFIGS, key=["ROWS", "N", "K"])
@triton.jit
def _w8a8_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    input_scale_ptr,
    column_scale_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, ROWS)
    columns = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    column_mask = columns < N
    input_scale = tl.load(input_scale_ptr)
    accumulator = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + (start + depth)[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        # IEEE division takes a slow subroutine for a zero numerator, which
        # made zero-padded tiles 2.7x slower; 0 / s is 0 exactly anyway.
        zero = x == 0.0
        quotient = tl.math.div_rn(tl.where(zero, 1.0, x), input_scale)
        quantized = tl.clamp(tl.where(zero, 0.0, quotient), -448.0, 448.0).to(
            tl.float8e4nv, fp_downcast_rounding="rtne"
        )
        weight = tl.load(
            w_ptr + columns[None, :] * stride_wn + (start + depth)[:, None],
            mask=column_mask[None, :],
            other=0.0,
        )
        accumulator = tl.dot(quantized, weight, accumulator)
    scale = input_scale * tl.load(column_scale_ptr + columns, mask=column_mask)
    tl.store(
        out_ptr + rows[:, None] * stride_om + columns[None, :],
        (accumulator * scale[None, :]).to(tl.bfloat16),
        mask=row_mask[:, None] & column_mask[None, :],
    )


def _rows_bucket(rows: int) -> int:
    return max(16, triton.next_power_of_2(rows))


def w8a8_linear(
    hidden: Tensor,
    weight: Tensor,
    input_scale: Tensor,
    column_scales: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """`hidden [M, K] bf16 @ weight [N, K] e4m3 ^T` for M <= MAX_ROWS.

    `column_scales` holds one FP32 weight scale per output column.
    """

    rows, depth = hidden.shape
    columns = weight.shape[0]
    if rows > MAX_ROWS:
        raise ValueError(f"small-row FP8 linear supports at most {MAX_ROWS} rows")
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        raise TypeError("expected BF16 activations and E4M3 weights")
    if hidden.stride(1) != 1 or weight.stride(1) != 1:
        raise ValueError("activation and weight rows must be contiguous")
    if depth % 256:
        raise ValueError("small-row FP8 linear needs K divisible by 256")
    if column_scales.shape != (columns,) or column_scales.dtype != torch.float32:
        raise ValueError("column scales must be one FP32 value per output")
    if out is None:
        out = torch.empty(rows, columns, dtype=torch.bfloat16, device=hidden.device)

    def grid(meta):
        return (triton.cdiv(columns, meta["BLOCK_N"]),)

    _w8a8_kernel[grid](
        hidden,
        weight,
        out,
        input_scale,
        column_scales,
        rows,
        columns,
        depth,
        hidden.stride(0),
        weight.stride(0),
        out.stride(0),
        ROWS=_rows_bucket(rows),
    )
    return out


__all__ = ["MAX_ROWS", "w8a8_linear"]
