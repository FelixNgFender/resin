"""Teacher-force greedy outputs through the target model and score agreement.

A speculative or re-batched decode may legitimately diverge from another greedy
run at near-tied logits. A state or cache bug instead produces tokens the target
itself rejects by a wide margin. This replays every generated sequence from
`scripts/real_decode.py` as one target-only prefill with prompt logprobs and
reports, per generated position, whether the emitted token was the target's
argmax and otherwise how far below the argmax it scored.

    uv run scripts/check_greedy.py --checkpoint CKPT artifacts/real.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy.models import create_runner, get_model
from resin import Engine, SamplingParams, SchedulerConfig
from scripts.real_decode import PROMPTS, render


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--model", default="qwen3_5")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tie-nats", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    data = json.loads(args.results.read_text())
    thinking = bool(data["summary"].get("thinking"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    spec = get_model(args.model)
    runner = create_runner(
        spec,
        args.checkpoint,
        device="cuda:0",
        context_length=4096,
        max_batch_size=1,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=0,
        decode_graph="disabled",
        prefill_graph="disabled",
        gpu_memory_utilization=0.85,
        decode_batch_sizes=[1],
    )
    engine = Engine(
        runner,
        max_running_requests=1,
        context_length=4096,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=spec.prefill_chunk_size,
            prefill_chunk_size=spec.prefill_chunk_size,
        ),
    )
    report = []
    for row in data["rows"]:
        prompt = render(tokenizer, PROMPTS[row["prompt"]], thinking)
        generated = row["token_ids"]
        lease = engine.submit(
            prompt + generated,
            sampling_params=SamplingParams(
                max_output_tokens=1,
                stop_token_ids=[],
                temperature=0.0,
                prompt_logprobs=True,
                prompt_top_logprobs=1,
            ),
        )
        while engine.has_work():
            engine.step()
        scores = lease.prompt_logprobs
        lease.take_result()
        gaps = []
        for index, token in enumerate(generated):
            entry = scores[len(prompt) + index]
            (top_token, top_logprob), *_ = entry.top_logprobs
            gaps.append(0.0 if top_token == token else top_logprob - entry.logprob)
        mismatches = [gap for gap in gaps if gap > 0]
        wide = [gap for gap in gaps if gap > args.tie_nats]
        report.append(
            dict(
                prompt=row["prompt"],
                tokens=len(gaps),
                argmax_agreement=1 - len(mismatches) / len(gaps),
                mismatches=len(mismatches),
                wide_mismatches=len(wide),
                max_gap_nats=max(gaps),
            )
        )
        print(json.dumps(report[-1]), flush=True)
    engine.close()
    total = sum(r["tokens"] for r in report)
    summary = dict(
        tokens=total,
        argmax_agreement=1 - sum(r["mismatches"] for r in report) / total,
        wide_mismatches=sum(r["wide_mismatches"] for r in report),
        max_gap_nats=max(r["max_gap_nats"] for r in report),
        tie_nats=args.tie_nats,
    )
    print("[check-summary] " + json.dumps(summary), flush=True)
    if args.output:
        args.output.write_text(json.dumps(dict(summary=summary, rows=report), indent=1))


if __name__ == "__main__":
    main()
