import torch

from resin_models_cuda.fp8_gemv import _w8a8_kernel

k, n = 5120, 1024
w = (torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
s_x = torch.tensor(0.02, device="cuda")
s_w = torch.tensor(0.003, device="cuda")
out = torch.empty(16, n, device="cuda", dtype=torch.bfloat16)


def run(x):
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
    _w8a8_kernel[grid](x, w, out, s_x, s_w, 16, n, k, x.stride(0), w.stride(0), out.stride(0), ROWS=16)


import triton

cases = {
    "random": torch.randn(16, k, device="cuda", dtype=torch.bfloat16),
    "zeros": torch.zeros(16, k, device="cuda", dtype=torch.bfloat16),
    "half-zero": torch.cat((torch.randn(5, k, device="cuda", dtype=torch.bfloat16), torch.zeros(11, k, device="cuda", dtype=torch.bfloat16))),
    "tiny": torch.full((16, k), 1e-3, device="cuda", dtype=torch.bfloat16),
    "ones": torch.ones(16, k, device="cuda", dtype=torch.bfloat16),
}
for name, x in cases.items():
    for _ in range(3):
        run(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(100):
        run(x)
    end.record()
    torch.cuda.synchronize()
    print(f"{name:10} {start.elapsed_time(end) * 10:.1f} us", flush=True)
