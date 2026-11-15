"""run one pinned model over a prompt file; save raw transcripts, activations, and compute.

usage (from the repo root):
    uv run --locked python -m experiments.sycophancy_diffing.run_prompts \
        --model base --prompts PATH --format raw --out artifacts/sycophancy_diffing/NAME

format, thinking, and the prompt file are protocol choices: there are no defaults for them.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from src.activation_cache import (
    MODELS,
    ComputeClock,
    file_sha256,
    generate,
    load_model,
    load_prompts,
    render,
    residual_stream,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--format", choices=["raw", "chat"], required=True)
    p.add_argument("--thinking", choices=["on", "off"], help="required with --format chat")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--activations", choices=["prompt", "none"], default="prompt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.format == "chat" and args.thinking is None:
        p.error("--format chat needs --thinking on|off")
    if args.format == "raw" and args.thinking is not None:
        p.error("--thinking only applies to --format chat")
    if args.out.exists():
        p.error(f"{args.out} exists; runs are never overwritten")
    return args


def git_commit() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    args = parse_args(argv)
    import nnsight
    import transformers
    from safetensors.torch import save_file

    prompts = load_prompts(args.prompts, args.format)
    thinking = None if args.thinking is None else args.thinking == "on"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True)
    act_dir = args.out / "activations"
    if args.activations == "prompt":
        act_dir.mkdir()

    model_id, revision = MODELS[args.model]
    record = {
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "model_id": model_id,
        "revision": revision,
        "dtype": "bfloat16",
        "format": args.format,
        "thinking": args.thinking,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "seed": args.seed,
        "prompts_file": str(args.prompts),
        "prompts_sha256": file_sha256(args.prompts),
        "n_prompts": len(prompts),
        "activations": args.activations,
        "residual_layout": "[n_layers + 1, prompt_positions, d_model]; row 0 = embeddings, "
        "row i = decoder layer i output before final norm",
        "git_commit": git_commit(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "nnsight": nnsight.__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "status": "running",
    }
    (args.out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    with ComputeClock() as load_clock:
        model, tokenizer = load_model(args.model)
    with (
        ComputeClock() as run_clock,
        open(args.out / "transcripts.jsonl", "w", encoding="utf-8", newline="\n") as transcripts,
    ):
        for i, prompt in enumerate(prompts, start=1):
            rendered = render(tokenizer, prompt, args.format, thinking)
            result = generate(model, tokenizer, rendered, args.max_new_tokens)
            if args.activations == "prompt":
                resid = residual_stream(model, result["input_ids"]).cpu().contiguous()
                save_file({"resid": resid}, act_dir / f"{prompt.id}.safetensors")
            row = {
                "id": prompt.id,
                "meta": prompt.meta,
                "rendered_prompt": rendered,
                "completion": result["completion"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
            }
            transcripts.write(json.dumps(row, ensure_ascii=False) + "\n")
            transcripts.flush()
            print(f"[{i}/{len(prompts)}] {prompt.id}: {result['completion_tokens']} tokens")

    record.update(
        status="done",
        compute={
            "load_seconds": round(load_clock.wall_seconds, 1),
            "run_seconds": round(run_clock.wall_seconds, 1),
            "gpu_hours": round((load_clock.wall_seconds + run_clock.wall_seconds) / 3600, 4),
            "peak_gpu_gib": round(max(load_clock.peak_gib, run_clock.peak_gib), 2),
        },
    )
    (args.out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record["compute"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
