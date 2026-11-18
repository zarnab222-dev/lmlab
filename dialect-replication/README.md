# fellows-internals: exploratory harness for dialect bias internals

**status: EXPLORATORY, NOT PREREGISTERED.** built Nov 17, 2026 so you can try options A, B and C
from `reading/prior-work-map.md` (section 5) on your 3080 and decide from what you see. nothing
here is a finding.

## start here (Nov 18)

for the base-first replication, go to `replication/README.md`. the rest of this file is the wider exploratory harness for options A, B and C.

## what's here

| file | what it is |
|---|---|
| `dialect_lab.py` | the library: data, forward hooks, metrics, the five experiments |
| `explore.py` / `explore.ipynb` | the same cells two ways; run top to bottom |
| `smoke_test.py` | runs everything on a tiny random model, no internet; proves the plumbing |
| `data/` | hofmann's public habitual-be pairs, katz adjectives, favorability ratings (MIT, see `data/README.md`) |

the five cells:
1. **behavior**: hofmann's matched guise probe (9 trait prompts, 37 katz adjectives) on AAE vs SAE,
   plus a **typo-noised SAE** control for the perplexity confound. reports the favorability gap with
   a cluster bootstrap CI (clusters = verbs), the text surprisal gap per condition, and hofmann's own
   log-ratio table.
2. **cache**: residual stream after every block, at the last prompt token and averaged over the
   quoted text, for one template (`A person who says " {} " tends to be`).
3. **option A**: AAE minus SAE difference-of-means direction per layer (train verbs), cosine with the
   typo direction, then project it out of every layer and re-score held-out verbs. compared against
   ablating the typo direction and a random direction.
4. **option B**: a direction from explicit labels in the same frame
   (`A Black person who says " {SAE text} " tends to be` vs `A White person ...`, plus
   "African American" / "European American"), a logistic probe per layer, and how far AAE text moves
   along it as a fraction of the label gap. also the overt behavioral gap in that same frame.
5. **option C**: patch the SAE run into the AAE run one (layer, suffix token) at a time; recovery 0
   means no effect, 1 means fully SAE-like.

## defaults i picked, and why

- **model: `Qwen/Qwen2.5-3B`**, with `Qwen/Qwen2.5-3B-Instruct` as the base vs instruct twin for option B.
  should be ungated (no Hugging Face login); i couldn't confirm from the cloud, where Hugging Face is
  blocked. about 6 GB in bf16, fits a 3080 with room for caching. fallback if it asks for a login:
  `Qwen/Qwen2.5-1.5B` (Apache 2.0). Gemma 2 2B plus Gemma Scope SAEs is the upgrade path, but it's
  gated, so it needs you to log in and accept the license yourself.
- **plain pytorch hooks, not TransformerLens or nnsight.** one less dependency, works on any HF
  decoder model as is, and the smoke test can run without downloading weights. the cost: no
  ready-made hook names, so attention heads and MLP outputs aren't separately hookable yet.
- **metric: expected favorability** of the next adjective, renormalized over the 37 adjectives,
  averaged over prompts. a per-pair number, so it carries a CI and works as a patching metric. this
  is my derived metric, not hofmann's; his log-ratio and top-5 favorability are printed beside it.
- **stats**: the 699 pairs are 233 verbs x he/she/they, so they aren't independent. the bootstrap
  resamples verbs, and train/test splits keep a verb's three pairs together.

## setup on the PC (Windows, RTX 3080)

```
cd D:\fellowship
mkdir fellows-internals   (copy this folder in, or pull it from the project files)
cd fellows-internals
python -m venv .venv
.venv\Scripts\activate
```

install a CUDA build of pytorch with the command that pytorch.org/get-started/locally gives for
Windows + pip + CUDA, then:

```
pip install transformers tokenizers pandas scikit-learn jupyter
python smoke_test.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

the smoke test should end with `smoke test passed`. then open `explore.ipynb` (or run `explore.py`
cell by cell in VS Code).

## what to try first tonight

1. set `N_PAIRS = 120` and run cells 1 and 2 only. look at two lines: does the AAE gap come out
   negative, and how does the typo surprisal gap compare to the AAE one?
2. raise `TYPO_EDITS` (1, 2, 3) until typo text NLL is about as high as AAE text NLL. the control only
   means something once the two are matched.
3. then all 699 pairs and cells 3 to 5. option C is the cheapest map and tells you which layers to
   look at for A and B.
4. for option B's base vs instruct comparison, rerun with `MODEL = "Qwen/Qwen2.5-3B-Instruct"`;
   results go to separate `runs/` folders.

## things the build itself surfaced

- **the public data is narrower than the paper.** hofmann's repo only ships the habitual-be pairs
  (`he be cracking` / `he's usually cracking`). the 2,019 Groenwold meaning-matched tweet pairs the
  paper's main result uses aren't in the repo, and neither are the other single-feature files
  (g-dropping, ain't, copula, finna). to use them, drop a tab-separated `AAE<TAB>SAE` file in `data/`
  and pass its path to `load_pairs`.
- **(rigor push, flagged)** the habitual pairs differ in more than dialect: the SAE side always adds
  "usually" and a contraction. so an "AAE direction" from these pairs may partly be a "the word
  usually is absent" direction. in the smoke test, held-out pairs separated perfectly along the
  direction even with random weights, which is that lexical difference showing through. worth
  keeping in mind before reading option A's separability column as evidence of anything.
- single-token adjectives only (hofmann's rule). `load_adjectives` prints any it drops for your
  tokenizer.

## what ran in the cloud

`smoke_test.py` passed on CPU with a 4-layer random Qwen2 model and a tokenizer trained on the pairs
file: every experiment ran end to end, ablation leaves ~0 component along the removed direction, and
patching the last layer at the final token gives recovery 1.00. batched and one-at-a-time runs give
identical outputs, so left padding isn't leaking. no real model ran: Hugging Face downloads are blocked
from the cloud.
