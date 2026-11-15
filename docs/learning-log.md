# learning log

one entry per new concept, per AGENTS.md. Codex writes the entry; the answer is mine, verbatim.

## 2026-11-15: clustered units and the design effect

- **concept:** when prompts come from shared templates, the template is the independent unit.
  treating correlated prompts as independent overstates the sample size, so standard errors and
  intervals come out too narrow and false positives become more likely. the design effect is
  deff = 1 + (m − 1)·ρ for clusters of size m with within-cluster correlation ρ; effective
  n = n / deff. example: 200 prompts, 20 templates of 10, ρ = 0.3 → deff 3.7, effective n ≈ 54,
  standard error understated by about 1.9×.
- **where it appears:** not in code yet. the paired, clustered interval is exercise B in #15
  (mine to write) and is used for #23.
- **check question:** in a toy factual-recall comparison with 200 questions from 20 templates,
  why does treating the 200 as independent make a false "instruct is better" claim more likely?
- **my answer (verbatim):** "don't you hide information by treating everything as independent?
  you're testing "effect" instead of 20?"
- **Codex: gap seen.** the right instinct is that the real sample size is 20, not 200. the
  direction of the error was reversed: independence does not hide information, it claims more
  independent evidence than exists, which shrinks the interval. the second half points at a
  real distinction worth keeping: whether the 20 templates are the whole population of interest
  (fixed effects) or a sample of possible phrasings (resample templates).
- **my correction (verbatim, same day):** "one* instead of 20 / ohhh that makes sense, a template
  is too homogeneous, i misunderstood the detail there"
- **Codex: after the correction.** "one effect instead of 20" is the random-effects view: each
  template has its own effect, and pooling hides how much they disagree, which is the variation
  the interval must carry. "too homogeneous" names high within-template correlation ρ. no gap
  remaining on this concept.
