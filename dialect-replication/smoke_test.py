"""CPU smoke test: runs every experiment end to end on a tiny randomly initialized model.

the numbers it prints are meaningless (random weights). it only proves the code paths work:
tokenizing, left padding, hooks, caching, ablation, probes, patching, saving.
needs no internet: the tokenizer is trained here on the pairs file.

    python smoke_test.py
"""
import random
import time

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import dialect_lab as dl

t0 = time.time()
torch.manual_seed(0)
pairs_all = dl.load_pairs()
adj_words = (dl.DATA / "katz_adjectives.txt").read_text().split()
corpus = [p["aae"] for p in pairs_all] + [p["sae"] for p in pairs_all]
corpus += [t.replace("{}", "x") for t in dl.TRAIT_PROMPTS + [dl.CACHE_PROMPT]]
corpus += ["A Black person", "A White person", "African American", "European American"] * 20
corpus += [" ".join(adj_words)] * 50

bpe = Tokenizer(models.BPE(unk_token=None))
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
bpe.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=3000, special_tokens=["<|eos|>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|eos|>", pad_token="<|eos|>")

cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                  tie_word_embeddings=True)
model = Qwen2ForCausalLM(cfg).eval()
runner = dl.Runner(model, tok, batch_size=16)
pairs = dl.load_pairs(n=60, seed=0)
adj = dl.load_adjectives(tok)
print(f"tiny model: {runner.n_layers} layers, {len(pairs)} pairs, {len(adj['words'])} adjectives")

print("typo examples:", [dl.add_typos(p["sae"], 1, random.Random(i)) for i, p in enumerate(pairs[:3])])

beh = dl.experiment_behavior(runner, pairs, adj, prompts=dl.TRAIT_PROMPTS[:3], typo_edits=1)
dl.print_behavior(beh)

cache = dl.experiment_cache(runner, beh["texts"], adj)
print("cache shapes:", {c: {p: tuple(a.shape) for p, a in v["acts"].items()} for c, v in cache.items()})

A = dl.experiment_direction_ablation(runner, pairs, cache, adj)
print(A["per_layer"].round(3).to_string())
print(A["ablation"].round(4).to_string())
# a sanity check that the ablation hook really edits the stream: projecting out the AAE
# direction should leave ~zero component along it in the re-cached activations
te = A["test_idx"]
enc = dl.encode(tok, dl.CACHE_PROMPT, [pairs[i]["aae"] for i in te])
re = runner.run(enc, adj["ids"], cache_layers=[1], positions=("last",),
                edit_fn=dl.ablation_edit({1: A["dirs"]["aae"][1]}))
resid = (re["acts"]["last"][:, 0].float() @ A["dirs"]["aae"][1]).abs().max().item()
print(f"max |component along ablated direction| after ablation at layer 1: {resid:.2e}")
assert resid < 1e-2

B = dl.experiment_race_direction(runner, pairs, cache, adj)
print(B["per_layer"].round(3).to_string())
print("overt gap, label frame:", tuple(round(x, 4) for x in B["overt_gap_label_frame"]))

C = dl.experiment_patching(runner, pairs, adj, n=24)
print(f"patching: {C['n_pairs']} pairs, suffix tokens {C['suffix_tokens']}")
print(C["recovery"].round(2).to_string())
# patching every suffix position at the last layer with clean activations must give recovery 1
# at the final token, since the final token's last-layer state fully determines the logits
assert abs(C["recovery"].iloc[-1, -1] - 1) < 1e-3, C["recovery"].iloc[-1, -1]

out = dl.save_run("runs/smoke-tiny-random", "tiny-random-qwen2", behavior_pairs=beh["pairs"],
                  optA_layers=A["per_layer"], optA_ablation=A["ablation"], optB_layers=B["per_layer"],
                  optC_recovery=C["recovery"], summary=beh["summary"])
print(f"saved to {out}. smoke test passed in {time.time() - t0:.1f}s")
