"""load a pretrained causal lm, generate, and capture its residual stream with plain hooks.

this is plumbing for zarnab222-dev/lmlab#23. it contains no prompts, metric, or research claim.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

# pinned so a base-vs-instruct comparison always reads the same weights.
MODELS = {
    "base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
    "instruct": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}


@dataclass
class PromptRecord:
    id: str
    text: str | None = None
    messages: list[dict] | None = None
    meta: dict = field(default_factory=dict)


def load_prompts(path: Path, fmt: str) -> list[PromptRecord]:
    """read a jsonl prompt file. raw format needs "text"; chat format needs "messages"."""
    records, seen = [], set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"line {line_no}: every prompt needs a non-empty string id")
        if rid in seen:
            raise ValueError(f"line {line_no}: duplicate id {rid!r}")
        seen.add(rid)
        need = "text" if fmt == "raw" else "messages"
        if need not in row:
            raise ValueError(f"line {line_no}: format {fmt!r} needs a {need!r} field")
        records.append(PromptRecord(rid, row.get("text"), row.get("messages"), row.get("meta", {})))
    if not records:
        raise ValueError(f"{path} has no prompts")
    return records


def render(tokenizer, record: PromptRecord, fmt: str, thinking: bool | None) -> str:
    """turn a prompt record into the exact string the model sees."""
    if fmt == "raw":
        return record.text
    return tokenizer.apply_chat_template(
        record.messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def load_model(name: str, device: str = "cuda", dtype=torch.bfloat16):
    """load the text-only causal lm for a pinned model; the vision tower is not loaded."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, revision = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, device_map=device
    )
    model.eval()
    return model, tokenizer


def decoder_layers(model) -> torch.nn.ModuleList:
    return model.model.layers


@torch.no_grad()
def residual_stream(model, input_ids: torch.Tensor) -> torch.Tensor:
    """residual stream for one sequence: [n_layers + 1, positions, d_model].

    row 0 is the token embedding; row i is the output of decoder layer i, before the final norm.
    """
    captured: list[torch.Tensor] = []

    def keep(_module, _inputs, output):
        captured.append((output[0] if isinstance(output, tuple) else output).detach())

    embed = model.get_input_embeddings()
    handles = [embed.register_forward_hook(keep)]
    handles += [layer.register_forward_hook(keep) for layer in decoder_layers(model)]
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    expected = len(decoder_layers(model)) + 1
    if len(captured) != expected:
        raise RuntimeError(f"captured {len(captured)} tensors, expected {expected}")
    return torch.stack([t[0] for t in captured])


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """greedy completion; returns the raw decoded text with special tokens kept."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, enc.input_ids.shape[1] :]
    return {
        "input_ids": enc.input_ids,
        "completion": tokenizer.decode(new, skip_special_tokens=False),
        "prompt_tokens": int(enc.input_ids.shape[1]),
        "completion_tokens": int(new.shape[0]),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComputeClock:
    """wall clock plus peak gpu memory for one run, for the compute record."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.wall_seconds = time.perf_counter() - self.start
        self.peak_gib = (
            torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        )
        return False
