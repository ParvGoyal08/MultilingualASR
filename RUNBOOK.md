# Runbook — who owns what

Two notebooks, split by **role** rather than by system.

| | role | writes? |
|---|---|---|
| `main_kaggle.ipynb` | **the deliverable** — every major run, Steps 1–4, all scoring and results | yes, exclusively |
| `main_kaggle_2.ipynb` | **the experiment bench** — probes, hypotheses, approach decisions | **never** |

`main_kaggle_2` runs in memory over a handful of clips and writes no checkpoint.
So it can be edited, re-run or thrown away at any time, including while
`main_kaggle` is mid-sweep, and it can never leave half-written state that
someone later scores by accident.

A finding is **promoted, not executed**: when a probe settles a question, flip
the corresponding flag in `main_kaggle` and sweep there. Never sweep in the
bench.

## The flags in `main_kaggle`

| flag | cell | meaning |
|---|---|---|
| `RUN_ASR` | 3.3 | master switch for the Sarvam sweep |
| `SMOKE_FIRST` | 3.3 | `True` = 3 clips, then stop and print REF vs HYP |
| `RUN_WHISPER` | 3.4 | long-form + per-segment Whisper; set after the bench's ratio test |

Nothing expensive runs unless one is flipped deliberately. Every `else` branch
prints what is already on disk, so a `False` flag still tells you where you are.

## Order of operations

1. `main_kaggle` 1.0 → 3.0. Builds the fusion; the bench asserts it exists.
2. `main_kaggle` 3.3, `SMOKE_FIRST = True`. Read the transcripts. Then `False`
   for the full Sarvam sweep (~70 min, ~₹580) and leave it running.
3. `main_kaggle_2` cells 1–3: long-form vs per-segment on 3 clips.
4. If the ratio lands near 1.0 → `RUN_WHISPER = True` in `main_kaggle` 3.4.
5. `main_kaggle` 3.5 / 3.6 for the current tables. Read-only, safe any time.

## Editing the library while a sweep runs

Both notebooks `git reset --hard origin/main` on their first cell, so a push
changes what the *next* kernel restart loads, never a running one. Any library
change that affects a notebook currently mid-sweep gets called out explicitly —
which notebook, and whether it invalidates checkpoints already written.

Checkpoints carry a `settings_key` (`beam5-cond1-lidlarge-v3-batch0`), so a
decoding change makes stale work visibly stale rather than silently mixed.
