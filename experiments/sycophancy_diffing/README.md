# sycophancy model diffing: plumbing

active item: [#23](https://github.com/zarnab222-dev/lmlab/issues/23), Qwen3.5-4B base vs instruct.
the question, prediction, prompts, metric, and what counts as a result are mine and live in #23
and `docs/research-log.md`. nothing in this folder answers them.

## Codex notes: what exists (Nov 15, 2026)

- `src/activation_cache.py`: pinned model ids and revisions, text-only loading in bf16, greedy
  generation, and `residual_stream(model, ids) -> [n_layers + 1, positions, d_model]` from plain
  forward hooks. row 0 is the token embedding; row i is decoder layer i's output, before the
  final norm.
- `run_prompts.py`: one model over one jsonl prompt file. writes `run.json` (models, revisions,
  versions, git commit, prompt-file hash, decoding settings, compute), `transcripts.jsonl` (the
  exact rendered prompt and the raw completion with special tokens kept), and one
  `activations/<id>.safetensors` per prompt. it refuses to overwrite a run directory.
- `smoke_nnsight.py`: nnsight hello-world, and a check that nnsight and the plain hooks see the
  identical residual at one layer.
- `smoke_prompts.jsonl`: two factual plumbing prompts. they are not research prompts.

## protocol choices the runner forces

`--format raw|chat` and, for chat, `--thinking on|off` have no defaults, because they change
what each model sees. with `--thinking off` the qwen chat template inserts an empty
`<think>\n\n</think>` block before the answer, so the instruct prompt contains tokens the base
model never sees in raw format. decoding is greedy (`do_sample=False`); `--max-new-tokens`
defaults to 128.

## run

from the repo root, models already downloaded:

```bash
uv run --locked python -m experiments.sycophancy_diffing.smoke_nnsight --model base
uv run --locked python -m experiments.sycophancy_diffing.run_prompts --model base --prompts PROMPTS.jsonl --format raw --out artifacts/sycophancy_diffing/RUN_NAME
```

prompt rows: `{"id": "...", "text": "..."}` for raw, `{"id": "...", "messages": [...]}` for
chat, and an optional `"meta"` object that is copied into the transcript.

## measured on the 3080 (Codex, Nov 15, 2026)

native windows, torch 2.14.1+cu130, transformers 5.20.2, nnsight 0.8.2, bf16.

| check | base | instruct |
|---|---|---|
| nnsight vs hooks, layer 16, max abs diff | 0.0 | 0.0 |
| next token after "The capital of France is" | " Paris" | " Paris" |
| model load, wall clock | 10.2 s | 10.2 s |
| 2 prompts × 32 new tokens, wall clock | 8.2 s | 6.6 s |
| peak gpu memory | 7.91 GiB | 7.91 GiB |

single runs, no intervals: these are plumbing timings, not results. the gated-deltanet layers
run on transformers' reference pytorch path because `flash-linear-attention` and
`causal_conv1d` are not installed; that is correct but slower. activations take
33 × positions × 2560 × 2 bytes per prompt (about 170 KB per token).
