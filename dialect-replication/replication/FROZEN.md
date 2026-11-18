# frozen analysis: replication-v1

frozen Nov 18, 2026, **before any real model was scored**. the tiny random model in the smoke test is the only thing that has gone through it.

`replicate.py` checks the sha256 of these files against `FROZEN.json` and refuses to run if any changed:
- replicate.py
- data/constructed_pairs.tsv
- data/habitual_pairs.tsv
- data/favorability.json

line endings are normalized before hashing, so a windows copy still matches.

why: base and instruct have to go through the same analysis, and the analysis shouldn't bend to what the base run shows. the hash check turns that promise into something the code enforces.

if the base run shows a real bug:
1. fix it, bump `VERSION` in replicate.py (replication-v2)
2. add an amendment below: what changed, why, and whether you'd already seen scores
3. rewrite FROZEN.json, rerun base, then instruct

`--allow-unfrozen` runs anyway for scratch work, and every output it writes says UNFROZEN.

if you keep this in git on the PC: commit it and `git tag replication-v1` before the base run. the tag is the same promise, signed.

## amendments
(none)
