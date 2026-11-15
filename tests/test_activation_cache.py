import json

import pytest
import torch

from experiments.sycophancy_diffing.run_prompts import parse_args
from src.activation_cache import load_prompts, residual_stream


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_prompts_raw_and_chat(tmp_path):
    f = write_jsonl(tmp_path / "p.jsonl", [{"id": "a", "text": "hi", "messages": []}])
    assert load_prompts(f, "raw")[0].text == "hi"
    assert load_prompts(f, "chat")[0].messages == []


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], "duplicate id"),
        ([{"text": "x"}], "non-empty string id"),
        ([{"id": "a", "messages": []}], "needs a 'text' field"),
    ],
)
def test_load_prompts_rejects_bad_rows(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_prompts(write_jsonl(tmp_path / "p.jsonl", rows), "raw")


def test_protocol_flags_have_no_defaults(tmp_path):
    base = ["--model", "base", "--prompts", "p.jsonl", "--out", str(tmp_path / "run")]
    with pytest.raises(SystemExit):
        parse_args(base)  # --format is required
    with pytest.raises(SystemExit):
        parse_args(base + ["--format", "chat"])  # chat needs --thinking
    assert parse_args(base + ["--format", "chat", "--thinking", "off"]).thinking == "off"


def test_existing_run_dir_is_never_overwritten(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit):
        parse_args(
            ["--model", "base", "--prompts", "p", "--format", "raw", "--out", str(tmp_path / "run")]
        )


class Block(torch.nn.Module):
    """like a hugging face decoder layer: adds its update to the residual and returns a tuple."""

    def __init__(self, d):
        super().__init__()
        self.mix = torch.nn.Linear(d, d)

    def forward(self, h):
        return (h + self.mix(h),)


class TinyLM(torch.nn.Module):
    """two-layer stand-in with the module layout residual_stream expects."""

    def __init__(self, d=4, vocab=10):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(d) for _ in range(2)])

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, use_cache=False):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            (h,) = layer(h)
        return h


@torch.no_grad()
def test_residual_stream_rows_follow_the_forward_pass():
    torch.manual_seed(0)
    model = TinyLM()
    ids = torch.tensor([[1, 2, 3]])
    resid = residual_stream(model, ids)
    assert resid.shape == (3, 3, 4)  # [n_layers + 1, positions, d_model]
    h = model.embed(ids)
    assert torch.equal(resid[0], h[0])
    for i, layer in enumerate(model.model.layers, start=1):
        (h,) = layer(h)
        assert torch.equal(resid[i], h[0])
    assert torch.equal(resid[-1], model(ids)[0])
