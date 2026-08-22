# Multilingual Speaker Diarization + ASR on Indic YouTube

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ParvGoyal08/MultilingualASR/blob/main/main.ipynb)

Speaker diarization and speaker-attributed transcription over **99 YouTube clips,
12.26 h, nine Indic scripts, 2–8 speakers per clip, 7.1% overlapped speech**.

**Start here: [`main.ipynb`](main.ipynb).** It runs end to end from committed
checkpoints — no GPU, no API key, no gated model access — and has a `SUBSET = 10`
toggle at the top so the whole pipeline can be verified in a few minutes before
committing to the full corpus. Verified from a fresh clone: it reproduces every
number below. The one network call is a first-run fetch of the assignment's
`youtube_segments.csv`, which is not redistributed in this repo.

## Result

| | baseline | final | |
|---|---|---|---|
| **cpWER** | 0.3957 | **0.3049** | **−23.0%** |
| **WDER** | 0.1128 | **0.0603** | **−46.6%** |
| WER | 0.2728 | 0.2617 | −4.1% |
| DER | 0.2521 | 0.2442 | not significant |
| JER | 0.3834 | 0.3608 | −5.9% |
| speaker-count accuracy | 60.6% | 79.8% | +19.2 pt |

Baseline is Saaras v3 on the single best diarizer (`reverb-v2`). Final is

```
sarvam-saaras-v3 @ DOVER-Lap(community-1, reverb-v2, diarizen-large)
                 + per-clip script correction (Sonnet 4.6, temperature 0)
```

Improvement holds on the held-out half — test cpWER 0.4463 → 0.3364 (−0.1100) —
and on **92 of 99 clips individually**, with 6 regressions. DER is scored at
**collar 0.0 with overlap included**, as the brief requires.

For scale: feeding the reference transcript back as if perfectly recognised, on
the reference diarization, still scores cpWER **0.1242** — one transcript cannot
carry two simultaneous speakers. The final system sits 0.1807 above that floor,
not 0.3049 above zero.

## What produced the gain

Three changes produced it. They decompose cleanly, and the decomposition matters
because the middle term is easy to hide inside the first:

| step | cpWER | Δ |
|---|---|---|
| baseline, Saaras v3 @ `reverb-v2` | 0.3957 | |
| 1. swap the diarizer for the DOVER-Lap fusion | 0.3381 | **−0.0576** |
| 2. fix the provenance bug | 0.3181 | **−0.0200** |
| 3. per-clip script correction | **0.3049** | **−0.0133** |

1. **DOVER-Lap fusion** (Step 4a), with each speaker thresholded independently so
   overlap survives — an argmax vote cannot emit two simultaneous speakers, and
   7.1% of this corpus is overlapped. The largest single gain. **GT-free at
   inference**: it reads only the three constituent RTTMs and the clip duration,
   which comes from the CSV manifest rather than from any annotation. The
   configuration was selected on **dev** and frozen in
   `results/fusion_config.json` before test was scored — the operator is GT-free,
   the selection used dev DER, and `WRITEUP.md` §3 keeps those two apart rather
   than conflating them.
2. **A provenance bug.** 27 clips had been transcribed against a stale 2-system
   fusion, so the measured "fusion" result was partly not the fusion. Catching it
   needed a `segmentation_key` — a hash of the turn boundaries, re-derived from
   each transcript's stored segment spans at audit time and compared against the
   diarization on disk. −0.0200 cpWER and −41% insertions, from no modelling
   change at all.
3. **Per-clip script correction** (Step 4b) — the reference writes code-switched
   English phonetically in the native script; Saaras writes Latin. 3,481 of
   21,357 substitutions were *correctly recognised words scored wrong for their
   script*. Claude Sonnet 4.6 rewrites the script of a token and nothing else,
   seeing **only the clip it is correcting** — no corpus vocabulary, no other
   clip, never the reference. −0.0133 cpWER overall and **−0.0189 on held-out
   test with 29 clips better and 0 worse**; 2,232 helpful edits, **0 harmful**,
   0 cross-script corruptions. Config frozen at `f284e59` before test was
   scored.

   An earlier variant built one lookup table from all 99 clips' hypotheses. It
   scores marginally better (0.3017) but is **transductive**, so its dev/test
   split cannot test generalisation — restricted to a dev-built vocabulary its
   test gain collapses to −0.0042. Both are reported in `WRITEUP.md` §6.1–§6.2;
   the per-clip variant ships because its number survives review.

Three interventions were built, measured end-to-end and **rejected**: a 1-of-3
overlap vote, ConvTasNet source separation, and LLM contextual refinement. Each
had a stated hypothesis, a control and a pre-committed abandon criterion, and
each is written up as a negative result. Two further ideas — an MSDD-style
verifier and an OSD∩constituent intersection — got only as far as *audits* of
how precise their candidate sets would be (11.5% and 23.6%); they were dropped
before implementation and are reported as such, not as experiments.

## Ground truth is never a pipeline input

The brief requires it and the code enforces it structurally: `ClipInput` carries
no reference field, so Steps 2–4 cannot reach ground truth. Step 4b's lookup
tables are built from the **hypothesis** vocabulary — verified cryptographically,
since rebuilding them from the reference instead produces different cache keys.
Raw annotations are never modified; alignment corrections live in a separate
manifest applied to a copy at scoring time.

The 50/49 dev/test split is frozen in `results/split.json`. Tuning happened on
dev; test was scored once.

## Layout

| | |
|---|---|
| [`main.ipynb`](main.ipynb) | **the deliverable** — Steps 1–5 end to end, `SUBSET` toggle |
| [`WRITEUP.md`](WRITEUP.md) | full method, results, failure analysis, limitations |
| [`obs.txt`](obs.txt) | lab notebook — 55 numbered observations, including the wrong turns |
| `sarvam_diar/` | library: data, reference, diarization, refinement, asr, translit, metrics |
| `tools/` | probes and drivers; nothing here writes under `asr/` |
| `checkpoints/` | committed model outputs (26 MB) so nothing needs re-running |
| `results/` | `step2_metrics.csv` (495 rows), `step3_metrics.csv` (638 rows), split, tables |
| [`RUNBOOK.md`](RUNBOOK.md) | how to re-run each stage, and which notebook owns what |
| [`RESULTS_DIARIZATION.md`](RESULTS_DIARIZATION.md), [`RESULTS_ASR.md`](RESULTS_ASR.md) | per-stage detail |
| [`GT_AUDIT.md`](GT_AUDIT.md) | reference-quality audit and the correction manifest |
| [`ERROR_EXPLORER.md`](ERROR_EXPLORER.md) | how to run the local error browser (`sarvam_diar/explorer.py`) |
| [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) | how the GPU sweeps were run |
| `Errors.txt` | raw hand-written notes from listening to clips; unedited, kept as the audit trail behind `GT_AUDIT.md` |

`main_kaggle.ipynb` is the sweep notebook that produced the checkpoints;
`main_kaggle_2.ipynb` is the experiment bench and writes nothing.

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb          # SUBSET = 10 first, then 0
```

**Nothing to configure for the default path** — every model output is committed,
so the notebook re-scores rather than re-runs. No GPU, no API key, no gated
model access.

Running a stage **from scratch** instead needs its credential in a `.env` at the
repo root: `HF_TOKEN` for the pyannote/Reverb diarizers (plus accepting their
licences), `SARVAM_API_KEY` for Step 3, `AWS_BEARER_TOKEN_BEDROCK` for Step 4b.
`.env` is gitignored, Colab/Kaggle Secrets are read under the same names, and no
key value is ever logged. **DiariZen is the one stage that cannot run in the same
environment** — its dependencies conflict with pyannote's, so it has its own
runner (`notebooks/diarizen_runner.ipynb`) and its RTTMs are committed, which is
why it is never required. `main.ipynb`'s second cell and `RUNBOOK.md` have the
full per-stage table.
