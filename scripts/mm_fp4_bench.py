"""Microbenchmark NVFP4 W4A16 linears at small M with and without autotuning."""

import argparse

import torch

from resin_models_cuda.nvfp4 import Nvfp4Linear, Nvfp4LinearWeights

SHAPES = {"gate_up": (5120, 17408), "down": (17408, 5120), "lm_head": (5120, 248320)}


def make(in_features, out_features, backend):
    source = Nvfp4LinearWeights(in_features, out_features).cuda()
    source.weight.random_(0, 256)
    source.weight_scale.copy_(
        (torch.rand_like(source.weight_scale, dtype=torch.float32) + 0.5).to(
            torch.float8_e4m3fn
        )
    )
    source.weight_scale_2.fill_(0.01)
    source.input_scale.fill_(1.0)
    return Nvfp4Linear(source, backend)


def timed(fn, iters=50):
    for _ in range(5):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cute-dsl")
    parser.add_argument("--ms", default="1,2,3,4,5,7,9,16")
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()
    ms = [int(m) for m in args.ms.split(",")]
    for name, (k, n) in SHAPES.items():
        weight_bytes = n * k // 2 + n * k // 16
        # Rotate through enough copies to stream from DRAM, not the 128 MB L2.
        copies = [make(k, n, args.backend) for _ in range(max(1, 512 * 2**20 // weight_bytes))]
        linear = copies[0]
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            if args.autotune:
                from flashinfer.autotuner import autotune

                with autotune(True):
                    linear(x, out=out)
                    for other in copies[1:]:
                        other(x, out=out)
                torch.cuda.synchronize()

            def run():
                for other in copies:
                    other(x, out=out)

            us = timed(run, iters=10) / len(copies)
            print(
                f"{name:8} M={m:2d} {us:8.1f} us  {weight_bytes / us / 1e6:6.2f} TB/s",
                flush=True,
            )


def check_quantized_linear():
    """Relative error of BF16 -> NVFP4 quantization on a random linear."""
    from torch import nn

    from resin_models_cuda.nvfp4 import Nvfp4Linear, nvfp4_weights_from_linear

    torch.manual_seed(0)
    linear = nn.Linear(5120, 17408, bias=False, dtype=torch.bfloat16, device="cuda")
    quantized = Nvfp4Linear(nvfp4_weights_from_linear(linear), "cute-dsl")
    x = torch.randn(4, 5120, dtype=torch.bfloat16, device="cuda")
    ref = linear(x).float()
    got = quantized(x).float()
    rel = (got - ref).norm() / ref.norm()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    print(f"nvfp4 quantized linear: relative error {rel:.4f}, cosine {cos:.5f}")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        check_quantized_linear()
    else:
        main()
