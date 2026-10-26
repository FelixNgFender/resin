"""Mean token NLL of fixed text through one local Engine (prompt logprobs).

Used to compare numerics modes (e.g. W4A16 versus W4A4 prefill) on the same
tokens: each document is split into `--chunk`-token prompts and every token
after the first is scored by the target model.

    uv run scripts/score_text.py --checkpoint CKPT --output score.json \
        corpus/wiki.parquet /usr/lib/python3.12/json/decoder.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from deploy.models import create_runner, get_model
from resin import Engine, SamplingParams, SchedulerConfig


def texts(path: Path, limit_chars: int):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        used = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
            for text in batch.column("text").to_pylist():
                if len(text) < 2000:
                    continue
                used += len(text)
                yield text
                if used >= limit_chars:
                    return
        return
    yield path.read_text(errors="ignore")[:limit_chars]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--model", default="qwen3_5")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--limit-chars", type=int, default=400_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    prompts = []
    for source in args.sources:
        tokens = []
        for text in texts(source, args.limit_chars):
            tokens.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        for start in range(0, len(tokens) - args.chunk + 1, args.chunk):
            if sum(1 for name, _ in prompts if name == source.name) >= args.chunks:
                break
            prompts.append((source.name, tokens[start : start + args.chunk]))
    spec = get_model(args.model)
    capacity = args.chunk + 64
    runner = create_runner(
        spec,
        args.checkpoint,
        device="cuda:0",
        context_length=capacity,
        max_batch_size=1,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=0,
        decode_graph="disabled",
        gpu_memory_utilization=0.85,
        decode_batch_sizes=[1],
    )
    runner.capture_cuda_graphs()
    engine = Engine(
        runner,
        max_running_requests=1,
        context_length=capacity,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=spec.prefill_chunk_size,
            prefill_chunk_size=spec.prefill_chunk_size,
        ),
    )
    per_source: dict[str, list[float]] = {}
    started = time.perf_counter()
    for name, tokens in prompts:
        lease = engine.submit(
            tokens,
            sampling_params=SamplingParams(
                max_output_tokens=1,
                stop_token_ids=[],
                temperature=0.0,
                prompt_logprobs=True,
            ),
        )
        while engine.has_work():
            engine.step()
        scores = lease.prompt_logprobs
        lease.take_result()
        per_source.setdefault(name, []).extend(
            -entry.logprob for entry in scores[1:] if entry is not None
        )
    engine.close()
    summary = {
        name: dict(tokens=len(values), nll=sum(values) / len(values), ppl=math.exp(sum(values) / len(values)))
        for name, values in per_source.items()
    }
    summary["elapsed_s"] = time.perf_counter() - started
    print("[score-summary] " + json.dumps(summary), flush=True)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
