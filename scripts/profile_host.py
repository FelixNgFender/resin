"""Split one speculative decode step into host phases versus graph replay.

Wraps the runner's graph-forward helpers with wall-clock timers (synchronizing
after replay) to show how much of a batch-1 step the GPU spends idle while the
host plans attention, stages inputs and reads results.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import torch

from deploy.models import create_runner, get_model
from resin import Engine, SamplingParams, SchedulerConfig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, default=6)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)
    spec = get_model("qwen3_5")
    runner = create_runner(
        spec,
        args.checkpoint,
        device="cuda:0",
        context_length=8192,
        max_batch_size=1,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=args.draft_tokens,
        gpu_memory_utilization=0.85,
        decode_batch_sizes=[1],
    )
    runner.capture_cuda_graphs()
    totals: collections.Counter[str] = collections.Counter()

    def timed(name, function, sync=False):
        def wrapper(*a, **k):
            start = time.perf_counter()
            result = function(*a, **k)
            if sync:
                torch.cuda.synchronize()
            totals[name] += time.perf_counter() - start
            return result

        return wrapper

    runner._run_cuda_graph = timed("graph replay (synced)", runner._run_cuda_graph, sync=True)
    for attention in runner._graph_attention.values():
        attention.plan = timed("flashinfer plan (host)", attention.plan)
    runner._cache.graph_plan_inputs = timed("cache inputs (host)", runner._cache.graph_plan_inputs)
    forward = runner.forward
    runner.forward = timed("runner.forward total", forward)
    engine = Engine(
        runner,
        max_running_requests=1,
        context_length=8192,
        scheduler_config=SchedulerConfig(max_batch_tokens=4096, prefill_chunk_size=4096),
    )
    prompt = torch.randint(1000, 30000, (2048,)).tolist()
    engine.submit(prompt, sampling_params=SamplingParams(max_output_tokens=4000, stop_token_ids=[], temperature=0.0))
    for _ in range(20):
        engine.step()
    totals.clear()
    start = time.perf_counter()
    steps = 0
    while engine.has_work() and steps < args.steps:
        engine.step()
        steps += 1
    wall = time.perf_counter() - start
    print(f"[host] {steps} steps, wall {wall / steps * 1e3:.2f} ms/step")
    for name, seconds in totals.most_common():
        print(f"[host] {name:28s} {seconds / steps * 1e3:7.3f} ms/step")
    engine.close()


if __name__ == "__main__":
    main()
