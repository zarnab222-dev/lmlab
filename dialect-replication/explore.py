# %% [markdown]
# # dialect internals: exploratory harness
#
# **EXPLORATORY, NOT PREREGISTERED.** this is for poking, so you can decide between options
# A, B and C (reading/prior-work-map.md, section 5) from what you see. nothing here is a result.
#
# each cell runs top to bottom. results land in `runs/<model>/`. in VS Code this file runs as
# cells; `explore.ipynb` is the same thing as a notebook.

# %%
import torch
import dialect_lab as dl

MODEL = "Qwen/Qwen2.5-3B"          # base. twin for option B: "Qwen/Qwen2.5-3B-Instruct"
N_PAIRS = None                      # None = all 699 habitual pairs; try 120 for a fast first pass
TYPO_EDITS = 1                      # raise until typo text NLL is close to AAE text NLL
BATCH = 32                          # drop to 16 if you run out of VRAM
SEED = 0
OUT = f"runs/{MODEL.split('/')[-1]}"

torch.manual_seed(SEED)
model, tok = dl.load_model(MODEL)
runner = dl.Runner(model, tok, batch_size=BATCH)
pairs = dl.load_pairs(n=N_PAIRS, seed=SEED)
adj = dl.load_adjectives(tok)
print(f"{MODEL}: {runner.n_layers} layers, {len(pairs)} pairs, {len(adj['words'])} adjectives")

# %% [markdown]
# ## 1. behavior: does this model show hofmann's covert gap, and does a typo do the same thing?
# 9 prompts x 3 conditions x all pairs. on a 3080 with a 3B model this should take minutes.

# %%
beh = dl.experiment_behavior(runner, pairs, adj, typo_edits=TYPO_EDITS, seed=SEED)
dl.print_behavior(beh)
print(beh["hofmann"].head(8).to_string())

# %% [markdown]
# ## 2. cache: residual stream after every block, last token and text-mean, for one template

# %%
cache = dl.experiment_cache(runner, beh["texts"], adj)
for c, v in cache.items():
    print(c, {p: tuple(a.shape) for p, a in v["acts"].items()})

# %% [markdown]
# ## 3. option A: one AAE direction, and does ablating it remove the gap?
# directions from train verbs, ablation scored on held-out verbs. compare against ablating the
# typo direction and a random direction at the same layers.

# %%
A = dl.experiment_direction_ablation(runner, pairs, cache, adj, position="last")
print(A["per_layer"].round(3).to_string())
print(A["ablation"].round(4).to_string())

# %% [markdown]
# ## 4. option B: does AAE text move along an explicit-label race direction?
# rerun this whole file with the instruct twin to get the base vs instruct comparison.

# %%
B = dl.experiment_race_direction(runner, pairs, cache, adj, position="last")
print(B["per_layer"].round(3).to_string())
m, lo, hi = B["overt_gap_label_frame"]
print(f"overt gap in the same frame (Black label minus White label): {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

# %% [markdown]
# ## 5. option C: patching SAE into AAE by layer and suffix token

# %%
C = dl.experiment_patching(runner, pairs, adj, n=48, seed=SEED)
print(f"{C['n_pairs']} pairs, clean minus corrupt score: {C['clean_minus_corrupt']:+.4f}")
print(C["recovery"].round(2).to_string())

# %%
dl.save_run(OUT, MODEL, behavior_pairs=beh["pairs"], hofmann=beh["hofmann"],
            hofmann_typo=beh["hofmann_typo"], optA_layers=A["per_layer"], optA_ablation=A["ablation"],
            optB_layers=B["per_layer"], optC_recovery=C["recovery"], summary=beh["summary"])
torch.save({c: v["acts"] for c, v in cache.items()}, f"{OUT}/cache.pt")
print("saved to", OUT)
