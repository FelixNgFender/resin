"""Practical HBM/GDDR read bandwidth vs. the small-row FP8 GEMV on one GPU.

Streams large buffers with a plain Triton read kernel (the ceiling any weight
GEMV can approach), then times the W8A8 kernel at 8 rows for Qwen3.8-27B's
projection shapes, back to back inside a CUDA graph as in decode.
"""

import torch
import triton
import triton.language as tl

from resin_models_cuda.fp8_gemv import w8a8_linear


@triton.jit
def _read(x, out, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * ITERS
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(ITERS):
        offsets = base + i * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x + offsets, mask=offsets < n, other=0)
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


def timed(fn, iters=20):
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
    return start.elapsed_time(end) / iters * 1e-3


def read_bandwidth(nbytes, block, iters, warps):
    words = nbytes // 4
    x = torch.ones(words, dtype=torch.int32, device="cuda")
    programs = triton.cdiv(words, block * iters)
    out = torch.empty(programs * block, dtype=torch.int32, device="cuda")
    seconds = timed(lambda: _read[(programs,)](x, out, words, BLOCK=block, ITERS=iters, num_warps=warps))
    return nbytes / seconds / 1e12


print(torch.cuda.get_device_name())
best = 0.0
for nbytes in (64 << 20, 1 << 30):
    for block, iters, warps in ((1024, 8, 4), (2048, 8, 8), (4096, 4, 8), (2048, 16, 8), (1024, 32, 4)):
        bw = read_bandwidth(nbytes, block, iters, warps)
        best = max(best, bw)
        print(f"read {nbytes >> 20:5d} MB block={block} iters={iters} warps={warps}: {bw:.3f} TB/s")
print(f"best practical read: {best:.3f} TB/s")

shapes = {"qkvz": (5120, 16384), "out": (6144, 5120), "attn_q": (5120, 14336)}
torch.manual_seed(0)
for name, (k, n) in shapes.items():
    copies = max(1, (1 << 30) // (n * k))
    weights = [(torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn) for _ in range(copies)]
    s_x = torch.tensor(0.02, device="cuda")
    cols = torch.full((n,), 0.003, device="cuda")
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    seconds = timed(lambda: [w8a8_linear(x, w, s_x, cols) for w in weights]) / copies
    print(f"w8a8 {name:7s} {k}x{n} M=8: {seconds * 1e6:7.1f} us  {n * k / seconds / 1e12:.3f} TB/s")
