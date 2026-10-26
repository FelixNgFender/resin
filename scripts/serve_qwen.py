"""Serve a local Qwen3.8 checkpoint over Resin's OpenAI-compatible HTTP API.

Binds to loopback by default, for same-host benchmarking against other engines.

    uv run scripts/serve_qwen.py --checkpoint CKPT --draft-tokens 8 --port 18090
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deploy.models import create_recipe, create_runner, get_model
from resin import Engine, SchedulerConfig
from resin.server import HttpServer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18090)
    args = parser.parse_args(argv)
    spec = get_model("qwen3_5")
    runner = create_runner(
        spec,
        args.checkpoint,
        device="cuda:0",
        context_length=args.context_length,
        max_batch_size=args.max_batch_size,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=args.draft_tokens,
        gpu_memory_utilization=0.85,
    )
    runner.capture_cuda_graphs()
    torch.cuda.synchronize()
    engine = Engine(
        runner,
        max_running_requests=args.max_batch_size,
        context_length=args.context_length,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=spec.prefill_chunk_size,
            prefill_chunk_size=spec.prefill_chunk_size,
        ),
    )
    recipe = create_recipe(spec, args.checkpoint, runner=runner)
    print(f"[serve] http://{args.host}:{args.port}/v1", flush=True)
    HttpServer(engine, recipe=recipe, model="qwen3.8-27b").serve(
        host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
