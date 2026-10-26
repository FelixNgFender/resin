"""Compare cuBLASLt `_scaled_mm` with the small-row Triton W8A8 linear."""

import torch

from resin_models_cuda.fp8_gemv import w8a8_linear

SHAPES = {
    "qkv": (5120, 10240),
    "z": (5120, 6144),
    "out": (6144, 5120),
    "q": (5120, 12288),
    "kv": (5120, 1024),
}


def scaled_mm(x, w, s_x, s_w):
    q = torch.clamp(x.float() / s_x, -448.0, 448.0).to(torch.float8_e4m3fn)
    return torch._scaled_mm(q, w.t(), scale_a=s_x, scale_b=s_w, out_dtype=torch.bfloat16, use_fast_accum=True)


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


torch.manual_seed(0)
for name, (k, n) in SHAPES.items():
    copies = max(1, 512 * 2**20 // (n * k))
    weights = [(torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn) for _ in range(copies)]
    s_x = torch.tensor(0.02, device="cuda")
    s_w = torch.tensor(0.003, device="cuda")
    for m in (1, 5, 7, 16, 28, 40, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = scaled_mm(x, weights[0], s_x, s_w).float()
        cols = s_w.reshape(1).expand(n).contiguous()
        got = w8a8_linear(x, weights[0], s_x, cols).float()
        rel = ((got - ref).norm() / ref.norm()).item()
        exact = (got == ref).float().mean().item()
        t_ref = timed(lambda: [scaled_mm(x, w, s_x, s_w) for w in weights]) / copies
        t_new = timed(lambda: [w8a8_linear(x, w, s_x, cols) for w in weights]) / copies
        gb = n * k / 1e9
        print(f"{name:4} M={m:2d} cublas {t_ref:6.1f}us ({gb / t_ref * 1e3:4.2f} TB/s)  triton {t_new:6.1f}us ({gb / t_new * 1e3:4.2f} TB/s)  rel_err {rel:.2e} bit-equal {exact:.3f}", flush=True)
