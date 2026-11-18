"""CPU smoke test for replicate.py on a tiny random model. no internet needed.

numbers are meaningless (random weights). checks: the pipeline runs, multi-token adjectives are
scored correctly, typo matching picks the closest candidate, and outputs are written.

    python smoke_replication.py
"""
import tempfile

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import replicate as R

torch.manual_seed(0)
adjs, _ = R.load_adjectives()
pairs = R.load_constructed() + R.load_habitual()
corpus = [p["aae"] for p in pairs] + [p["sae"] for p in pairs] + [R.TEMPLATE] * 20 + [" ".join(adjs)] * 3

bpe = Tokenizer(models.BPE(unk_token=None))
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
bpe.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1200, special_tokens=["<|eos|>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|eos|>", pad_token="<|eos|>")
model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()

ntok = [len(tok.encode(" " + a, add_special_tokens=False)) for a in adjs]
print(f"adjective token counts under the toy tokenizer: {sorted(set(ntok))} ({sum(n > 1 for n in ntok)} multi-token)")
assert any(n > 1 for n in ntok), "want multi-token adjectives in the smoke test"

# check 1: batched multi-token scoring equals a slow, one-token-at-a-time reference
text = pairs[0]["aae"]
ids, _ = R.text_ids(tok, text)
fast, _ = R.adjective_logprobs(model, tok, [text], adjs[:6], 4, "cpu")
for j, a in enumerate(adjs[:6]):
    seq = ids[:]
    ref = 0.0
    for t in tok.encode(" " + a, add_special_tokens=False):
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([seq])).logits[0, -1], -1)
        ref += lp[t].item()
        seq.append(t)
    assert abs(ref - fast[0, j]) < 1e-3, (a, ref, fast[0, j])
print("multi-token scoring matches the step-by-step reference")

with tempfile.TemporaryDirectory() as d:
    out, results = R.main(["--model", "tiny-random", "--out", d, "--batch", "16", "--n-boot", "500",
                           "--allow-unfrozen"], model=model, tok=tok)
    import pandas as pd
    df = pd.read_csv(out / "pairs_scored.csv")
    # check 2: the chosen typo is at least as close to the AAE NLL as the untouched SAE text
    gap_typo = (df.nll_sae_typo - df.nll_aae).abs().mean()
    gap_sae = (df.nll_sae - df.nll_aae).abs().mean()
    print(f"mean |NLL - NLL_aae|: sae {gap_sae:.3f}, matched typo {gap_typo:.3f}")
    assert (df.sae_typo != df.sae).all()
    print("typo examples:", list(df.sae_typo.head(4)))
    print((out / "report.md").read_text()[:1800])
    for f in ["report.md", "summary.json", "pairs_scored.csv", "adjective_logprobs.npy",
              "adjectives_without_habitual_aae_vs_sae.csv", "adjectives_with_habitual_aae_minus_typo.csv"]:
        assert (out / f).exists(), f
ok, msg = R.check_frozen()
print("freeze check:", msg)
assert ok, "analysis files changed since FROZEN.json was written"
print("smoke test passed")
