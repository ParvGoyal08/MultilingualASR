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
| **cpWER** | 0.3957 | **0.3017** | **−23.8%** |
| **WDER** | 0.1128 | **0.0602** | **−46.6%** |
| WER | 0.2728 | 0.2585 | −5.2% |
| DER | 0.2521 | 0.2442 | not significant |
| JER | 0.3834 | 0.3608 | −5.9% |
| speaker-count accuracy | 60.6% | 79.8% | +19.2 pt |

Baseline is Saaras v3 on the single best diarizer (`reverb-v2`). Final is

```
sarvam-saaras-v3 @ DOVER-Lap(community-1, reverb-v2, diarizen-large)
                 + script/numeral normalisation
```

Improvement holds on the held-out half (test cpWER −0.0216) and on 72 of 99 clips
individually, with 1 regression. DER is scored at **collar 0.0 with overlap
included**, as the brief requires.

For scale: feeding the reference transcript back as if perfectly recognised, on
the reference diarization, still scores cpWER **0.1242** — one transcript cannot
carry two simultaneous speakers. The final system sits 0.1775 above that floor,
not 0.3017 above zero.

## What produced the gain

Three changes produced it, and the largest was not a modelling change at all:

1. **A provenance bug.** 27 clips had been transcribed against a stale 2-system
   fusion. Detecting it needed a `segmentation_key` — a hash of the turn
   boundaries stored beside each transcript — so stale work is *visibly* stale
   rather than silently mixed. Fixing it: −0.0200 cpWER, −41% insertions. Larger
   than any model swap in this repo.
2. **DOVER-Lap fusion** (Step 4a), with each speaker thresholded independently so
   overlap survives an argmax vote. −0.0776 cpWER, −0.0500 WDER.
3. **Script and numeral normalisation** (Step 4b) — the reference writes
   code-switched English phonetically in the native script and numbers as spoken
   words; Saaras writes Latin and digits. 3,481 of 21,357 substitutions were
   *correctly recognised words scored wrong for their spelling convention*.
   −0.0164 cpWER, 72 clips better / 1 worse, 0 dev regressions.

Five interventions were built, measured and **rejected**: a 1-of-3 overlap vote,
an MSDD verifier over constituent disagreement, an OSD∩constituent intersection,
ConvTasNet source separation, and LLM contextual refinement (null across three
models). They are written up as negative results because they were well-posed
experiments, not because they worked.

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
| [`obs.txt`](obs.txt) | lab notebook — 55 dated observations, including the wrong turns |
| `sarvam_diar/` | library: data, reference, diarization, refinement, asr, translit, metrics |
| `tools/` | probes and drivers; nothing here writes under `asr/` |
| `checkpoints/` | committed model outputs (26 MB) so nothing needs re-running |
| `results/` | `step2_metrics.csv` (495 rows), `step3_metrics.csv` (539 rows), split, tables |
| [`RUNBOOK.md`](RUNBOOK.md) | how to re-run each stage, and which notebook owns what |
| [`RESULTS_DIARIZATION.md`](RESULTS_DIARIZATION.md), [`RESULTS_ASR.md`](RESULTS_ASR.md) | per-stage detail |
| [`GT_AUDIT.md`](GT_AUDIT.md) | reference-quality audit and the correction manifest |

`main_kaggle.ipynb` is the sweep notebook that produced the checkpoints;
`main_kaggle_2.ipynb` is the experiment bench and writes nothing.

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb          # SUBSET = 10 first, then 0
```

Re-running a model instead of reading its checkpoint needs the corresponding
credential in `.env` (`SARVAM_API_KEY`, `HF_TOKEN`, `AWS_BEARER_TOKEN_BEDROCK`);
`.env` is gitignored. DiariZen needs its own environment — its RTTMs are
committed so it is never required.
