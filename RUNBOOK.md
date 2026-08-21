# Runbook — who owns what

Two Kaggle notebooks run at the same time against **one** working directory,
`/kaggle/working/sarvam_diarization`. That is fine only because their writes are
disjoint. This file is the contract; the notebooks assert it rather than assume it.

## Ownership

| | `main_kaggle.ipynb` | `main_kaggle_2.ipynb` |
|---|---|---|
| Steps 1–2 (audio, diarization) | **owns** | — |
| Step 4 fusion RTTMs | **owns** (3.0 builds) | asserts present, never builds |
| `asr/sarvam-saaras-v3@fusion/` | **owns** | never touches |
| `asr/whisper-*` (long-form and `@fusion`) | reports only | **owns** |
| Scoring, results tables, MD files | **owns** | scratch tables only |

Every checkpoint path is therefore written by exactly one notebook. Both read
everything, so `main_kaggle` 3.6 scores Whisper results the moment `main_kaggle_2`
produces them — no copying, no merge step.

## The three flags

| notebook | flag | meaning |
|---|---|---|
| `main_kaggle` | `RUN_ASR` | master switch for the Sarvam sweep |
| `main_kaggle` | `SMOKE_FIRST` | `True` = 3 clips then stop and print REF vs HYP |
| `main_kaggle_2` | `RUN` | `False` until the long-form/per-segment test shows ratio ≈ 1.0 |

Nothing expensive runs unless one of these is flipped deliberately.

## Order of operations

1. `main_kaggle` cells 1.0 → 3.0. This builds the fusion; `main_kaggle_2` blocks
   without it.
2. `main_kaggle` 3.3 with `SMOKE_FIRST = True`. Read the printed transcripts.
   Then `False` for the full sweep (~70 min, ~₹580) and leave it.
3. `main_kaggle_2` cells 1–2 (test). If ratio ≈ 1.0, set `RUN = True`.
4. `main_kaggle` 3.6 whenever you want the current table. It is read-only.

## Editing the library while a sweep runs

Both notebooks `git reset --hard origin/main` on cell 1, so a push changes what
the *next* kernel restart loads, never a running one. A change that affects a
notebook currently mid-sweep gets said out loud, with which notebook and whether
it invalidates checkpoints already written.

Checkpoints carry a `settings_key` (`beam5-cond1-lidlarge-v3-batch0`), so a
decoding change makes stale work visibly stale instead of silently mixed.
