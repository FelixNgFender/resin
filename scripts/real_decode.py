"""Decode real chat prompts through one local Engine and report per-request speed.

Unlike `scripts/bench.py`, which times synthetic random-token cohorts, this
measures what a local user sees: chat-templated prompts, recipe stop tokens,
natural speculative acceptance and optional sampling. Each cohort of
`--batch-size` concurrent requests is timed from its first decoded token to its
last, so prefill (TTFT) is reported separately.

    uv run scripts/real_decode.py --checkpoint /path/to/Qwen3.8-27B-NVFP4 \
        --draft-tokens 2 --batch-size 1 --output artifacts/real.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from deploy.models import create_runner, get_model
from resin import Engine, SamplingParams, SchedulerConfig

PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 duration string such "
    "as 'P3DT4H12M' into a datetime.timedelta. Include docstrings, type hints "
    "and a handful of pytest unit tests.",
    "explain": "Explain how a CPU cache hierarchy works, why cache lines matter "
    "for performance, and give two concrete examples of cache-friendly versus "
    "cache-hostile code.",
    "story": "Write a short story (about 400 words) about a lighthouse keeper "
    "who discovers that the light has been signalling to something other than "
    "ships.",
    "math": "A tank fills through pipe A in 6 hours and through pipe B in 9 "
    "hours, while pipe C drains it in 12 hours. If all three pipes are opened "
    "when the tank is one quarter full, how long until it is full? Show your "
    "reasoning step by step.",
    "json": "Produce a JSON array of 12 fictional employees. Each object needs "
    "id, name, department, title, start_date (YYYY-MM-DD), salary and a list "
    "of 2-3 skills. Output only the JSON.",
    "translate": "Translate the following paragraph into French, then into "
    "Spanish, keeping the tone formal: 'The committee has reviewed the "
    "proposal in detail and, while it appreciates the ambition of the plan, "
    "it requests a revised budget and a clearer timeline before the next "
    "meeting.'",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3_5")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--document",
        type=Path,
        help="prepend this text file (truncated to --document-tokens) to every prompt",
    )
    parser.add_argument("--document-tokens", type=int, default=0)
    parser.add_argument("--decode-graph", default="full")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def render(tokenizer, text: str, thinking: bool) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=thinking,
        return_dict=False,
    )


def run_cohort(engine, prompts, params):
    leases = {}
    started = time.perf_counter()
    for name, tokens in prompts:
        lease = engine.submit(tokens, sampling_params=params)
        leases[lease.request_id] = name
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    generated: dict[int, list[int]] = {request: [] for request in leases}
    while engine.has_work():
        events = engine.step()
        now = time.perf_counter()
        for event in events:
            first.setdefault(event.request_id, now)
            last[event.request_id] = now
            generated[event.request_id].append(event.token_id)
    rows = []
    for request, name in leases.items():
        tokens = generated[request]
        span = last[request] - first[request]
        rows.append(
            dict(
                prompt=name,
                output_tokens=len(tokens),
                ttft_s=first[request] - started,
                decode_s=span,
                decode_tok_s=(len(tokens) - 1) / span if span > 0 else None,
                token_ids=tokens,
            )
        )
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    spec = get_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    generation = json.loads((args.checkpoint / "generation_config.json").read_text())
    stops = generation.get("eos_token_id", [])
    stops = [stops] if isinstance(stops, int) else list(stops)
    names = args.prompts.split(",")
    prefix = ""
    if args.document is not None:
        ids = tokenizer(args.document.read_text(errors="ignore"))["input_ids"]
        prefix = tokenizer.decode(ids[: args.document_tokens])
        prefix = "Here is a reference document:\n\n" + prefix + "\n\nNow, a separate task. "
    prompts = [
        (name, render(tokenizer, prefix + PROMPTS[name], args.thinking)) for name in names
    ]
    capacity = args.context_length

    started = time.perf_counter()
    runner = create_runner(
        spec,
        args.checkpoint,
        device="cuda:0",
        context_length=capacity,
        max_batch_size=args.batch_size,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=args.draft_tokens,
        decode_graph=args.decode_graph,
        gpu_memory_utilization=args.gpu_memory_utilization,
        decode_batch_sizes=[args.batch_size],
    )
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    runner.capture_cuda_graphs()
    torch.cuda.synchronize()
    capture_s = time.perf_counter() - started
    print(f"[real] loaded in {load_s:.1f}s, graphs in {capture_s:.1f}s", flush=True)

    params = SamplingParams(
        max_output_tokens=args.max_output_tokens,
        stop_token_ids=stops,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    engine = Engine(
        runner,
        max_running_requests=args.batch_size,
        context_length=capacity,
        enable_overlap=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=args.batch_size * spec.prefill_chunk_size,
            prefill_chunk_size=spec.prefill_chunk_size,
        ),
    )
    # Warm sampling/attention paths outside the measured cohorts.
    run_cohort(engine, prompts[:1] * args.batch_size, params)
    def stats():
        value = getattr(runner, "speculative_stats", None)
        value = value() if callable(value) else value
        return dict(value or {})

    before = stats()
    rows = []
    for repeat in range(args.repeats):
        for index in range(len(prompts)):
            cohort = [prompts[(index + offset) % len(prompts)] for offset in range(args.batch_size)]
            for row in run_cohort(engine, cohort, params):
                row["repeat"] = repeat
                rows.append(row)
                print(
                    f"[real] {row['prompt']:>9} out={row['output_tokens']:4d} "
                    f"ttft={row['ttft_s'] * 1e3:7.1f}ms "
                    f"decode={row['decode_tok_s'] or 0:7.1f} tok/s",
                    flush=True,
                )
    after = stats()
    engine.close()
    total_tokens = sum(row["output_tokens"] - 1 for row in rows)
    total_time = sum(row["decode_s"] for row in rows)
    summary = dict(
        model=args.model,
        checkpoint=str(args.checkpoint),
        hardware=torch.cuda.get_device_name(0),
        draft_tokens=args.draft_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        thinking=args.thinking,
        prompt_tokens=sum(len(tokens) for _, tokens in prompts) / len(prompts),
        decode_graph=args.decode_graph,
        load_s=load_s,
        capture_s=capture_s,
        mean_decode_tok_s=sum(r["decode_tok_s"] or 0 for r in rows) / len(rows),
        pooled_decode_tok_s=total_tokens / total_time if total_time else None,
        speculation={
            key: after.get(key, 0) - before.get(key, 0)
            for key in ("blocks", "drafts", "accepted", "emitted")
            if isinstance(after.get(key), (int, float))
        },
    )
    spec_stats = summary["speculation"]
    if spec_stats.get("drafts"):
        summary["accept_rate"] = spec_stats["accepted"] / spec_stats["drafts"]
        summary["tokens_per_block"] = spec_stats["emitted"] / spec_stats["blocks"]
    print("[real-summary] " + json.dumps(summary), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == "__main__":
    main()
