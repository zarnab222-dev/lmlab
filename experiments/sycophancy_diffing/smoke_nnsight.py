"""nnsight hello-world plus an invariant check: nnsight and plain hooks see the same residual.

usage: uv run --locked python -m experiments.sycophancy_diffing.smoke_nnsight --model base
"""

from __future__ import annotations

import argparse
import json

from src.activation_cache import MODELS, ComputeClock, load_model, residual_stream

PROMPT = "The capital of France is"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--layer", type=int, default=16)
    args = p.parse_args()

    from nnsight import LanguageModel

    model_id, _ = MODELS[args.model]
    with ComputeClock() as clock:
        hf_model, tokenizer = load_model(args.model)
        lm = LanguageModel(hf_model, tokenizer=tokenizer)
        ids = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
        with lm.trace(ids.to("cuda")):
            out = lm.model.layers[args.layer].output
            via_nnsight = (out[0] if isinstance(out, tuple) else out).save()
            logits = lm.output.logits.save()

        via_hooks = residual_stream(hf_model, ids.to("cuda"))

    a = via_nnsight[0].float()
    b = via_hooks[args.layer + 1].float()
    report = {
        "model": model_id,
        "layer": args.layer,
        "residual_shape": list(via_hooks.shape),
        "max_abs_diff_nnsight_vs_hooks": (a - b).abs().max().item(),
        "next_token": lm.tokenizer.decode(logits[0, -1].argmax()),
        "wall_seconds": round(clock.wall_seconds, 1),
        "peak_gpu_gib": round(clock.peak_gib, 2),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["max_abs_diff_nnsight_vs_hooks"] == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
