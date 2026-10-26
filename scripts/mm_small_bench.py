"""Time candidate implementations of Qwen's tiny DeltaNet a/b projections."""

import torch
import torch.nn.functional as F


def timed(fn, iters=200):
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


for m in (1, 5, 9, 40):
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(48, 5120, device="cuda", dtype=torch.bfloat16) * 0.01
    b = torch.randn(48, 5120, device="cuda", dtype=torch.bfloat16) * 0.01
    ab = torch.cat((a, b))
    abt = ab.t().contiguous()
    results = {
        "two F.linear": timed(lambda: (F.linear(x, a), F.linear(x, b))),
        "one F.linear(96)": timed(lambda: F.linear(x, ab)),
        "mm x@WT": timed(lambda: torch.mm(x, abt)),
        "fp32 mm": timed(lambda: torch.mm(x.float(), abt.float())),
        "mul+sum": timed(lambda: (x[:, None, :] * ab[None]).float().sum(-1)),
    }
    print(f"M={m:2d} " + "  ".join(f"{k}={v:6.1f}us" for k, v in results.items()), flush=True)
