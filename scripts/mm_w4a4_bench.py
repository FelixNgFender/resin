"""Prefill-size NVFP4 linears: W4A16 (current) versus W4A4 FP4 tensor cores."""

import torch

from resin_models_cuda.nvfp4 import Nvfp4Linear, Nvfp4LinearWeights, swizzle_block_scales

SHAPES = {"gate_up": (5120, 17408), "down": (17408, 5120)}
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def dequant(source):
    w = source.weight.cpu()
    lo, hi = (w & 0xF).long(), (w >> 4).long()
    vals = torch.stack((E2M1[lo], E2M1[hi]), -1).reshape(w.shape[0], -1)
    scales = source.weight_scale.cpu().float().repeat_interleave(16, 1)
    return (vals * scales * source.weight_scale_2.cpu()).cuda()


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


torch.manual_seed(0)
from flashinfer import fp4_quantize, mm_fp4

for name, (k, n) in SHAPES.items():
    linear = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="cuda")
    from resin_models_cuda.nvfp4 import nvfp4_weights_from_linear

    source = nvfp4_weights_from_linear(linear)
    w4a16 = Nvfp4Linear(source, "cute-dsl")
    reference_weight = dequant(source)
    w_sf = swizzle_block_scales(source.weight_scale)
    for m in (512, 4096):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        input_scale = (x.abs().amax().float() / (6 * 448)).reshape(1)
        ref = x.float() @ reference_weight.t()

        def run_w4a4():
            a, a_sf = fp4_quantize(x, (1.0 / input_scale), 16, False, True)
            return mm_fp4(a, source.weight.T, a_sf, w_sf.T, input_scale * source.weight_scale_2, torch.bfloat16)

        out16 = w4a16(x).float()
        out4 = run_w4a4().float()
        e16 = ((out16 - ref).norm() / ref.norm()).item()
        e4 = ((out4 - ref).norm() / ref.norm()).item()
        t16 = timed(lambda: w4a16(x))
        t4 = timed(run_w4a4)
        flops = 2 * m * n * k
        print(f"{name:7} M={m:4d}  W4A16 {t16:8.1f}us {flops / t16 / 1e6:6.1f} TFLOPS err {e16:.4f}   W4A4 {t4:8.1f}us {flops / t4 / 1e6:6.1f} TFLOPS err {e4:.4f}", flush=True)
