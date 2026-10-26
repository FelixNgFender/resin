import torch
from torch.profiler import ProfilerActivity, profile

from resin_models_cuda.fp8_gemv import w8a8_linear

k, n = 5120, 1024
w = (torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
s_x = torch.tensor(0.02, device="cuda")
s_w = torch.tensor(0.003, device="cuda")
from resin_models_cuda import fp8_gemv
for m in (16, 5, 16):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        w8a8_linear(x, w, s_x, s_w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            w8a8_linear(x, w, s_x, s_w)
        torch.cuda.synchronize()
    print(f"M={m}", fp8_gemv._w8a8_kernel.cache)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
