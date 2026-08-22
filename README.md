# Multilingual Speaker Diarization + ASR on Indic YouTube

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ParvGoyal08/MultilingualASR/blob/main/main.ipynb)

Speaker diarization and speaker-attributed transcription over **99 YouTube clips,
12.26 h, nine Indic scripts, 2–8 speakers per clip, 7.13 % overlapped speech**.

**[`WRITEUP.md`](WRITEUP.md) is the short final report.** This file is the full
documentation: method, per-stage results, metric definitions, failure analysis,
negative results and operational notes. The lab notebook — including the wrong
turns, in the order they were taken — is [`obs.txt`](obs.txt).

**Start with [`main.ipynb`](main.ipynb).** It runs Steps 1–5 end to end from
committed checkpoints — no GPU, no API key, no gated model access — with a
`SUBSET = 10` toggle that verifies the whole pipeline in under a minute. Verified
from a clean clone against a virtualenv holding only six packages.

---

## Contents

[Result](#result) · [Layout](#layout) · [Dataset](#1--dataset) ·
[Metrics and leak discipline](#2--metrics-normalisation-and-leak-discipline) ·
[Step 2 diarization](#3--step-2--baseline-diarization) ·
[Step 3 ASR](#4--step-3--asr-on-diarized-segments) ·
[Whisper and IndicConformer](#5--why-whisper-was-not-selected) ·
[Step 4a fusion](#6--step-4a--dover-lap-fusion) ·
[The provenance fix](#7--the-provenance-fix--not-a-model-improvement) ·
[Step 4b script correction](#8--step-4b--script-correction) ·
[Failure analysis](#9--failure-analysis) ·
[Negative results](#10--negative-results) ·
[Ground-truth audit](#11--ground-truth-audit) ·
[Running it](#12--running-it) · [Error explorer](#13--error-explorer) ·
[Kaggle](#14--running-the-gpu-sweeps-on-kaggle)

---

## Result

| | baseline | final | |
|---|---|---|---|
| **cpWER** | 0.3957 | **0.3049** | **−23.0 %** |
| **WDER** | 0.1128 | **0.0603** | **−46.6 %** |
| WER | 0.2728 | 0.2617 | −4.1 % |
| DER | 0.2521 | 0.2442 | *not significant* |
| JER | 0.3834 | 0.3608 | −5.9 % |
| speaker-count accuracy | 60.6 % | 79.8 % | +19.2 pt |

Baseline is Saaras v3 on the single best diarizer (`reverb-v2`). Final is

```
sarvam-saaras-v3 @ DOVER-Lap(community-1, reverb-v2, diarizen-large)
                 + per-clip script correction (Sonnet 4.6, temperature 0)
```

Improvement holds on the held-out half — test cpWER 0.4463 → 0.3364 (−0.1100) —
and on **92 of 99 clips individually**, with 6 regressions. DER is scored at
**collar 0.0 with overlapping speech included**, as the brief requires.

For scale: feeding the reference transcript back as if perfectly recognised, on
the reference diarization, still scores cpWER **0.1242** — one transcript cannot
carry two simultaneous speakers. The final system sits **0.1807 above that
floor**, not 0.3049 above zero.

### The decomposition

| step | cpWER | Δ |
|---|---|---|
| baseline, Saaras v3 @ `reverb-v2` | 0.3957 | |
| 1 · swap the diarizer for the DOVER-Lap fusion | 0.3381 | **−0.0576** |
| 2 · fix the provenance bug | 0.3181 | **−0.0200** |
| 3 · per-clip script correction | **0.3049** | **−0.0133** |

The middle term matters and is easy to hide inside the first: the fusion figure
usually quoted (0.3957 → 0.3181) already contains the bug fix.

---

## Layout

| | |
|---|---|
| [`main.ipynb`](main.ipynb) | **the deliverable** — Steps 1–5 end to end, `SUBSET` toggle, outputs committed |
| [`WRITEUP.md`](WRITEUP.md) | the short final report |
| `README.md` | this file — full documentation |
| [`obs.txt`](obs.txt) | lab notebook, 56 numbered observations |
| `sarvam_diar/` | library: `data`, `reference`, `diarization`, `refinement`, `asr`, `translit`, `text_metrics`, `llm_refine`, `gt_qc`, `evaluation`, `explorer` |
| `tools/` | probes and drivers; nothing here writes under `asr/` unless it says so |
| `checkpoints/` | committed model outputs (29 MB) so nothing needs re-running |
| `results/` | `step2_metrics.csv` (495 rows), `step3_metrics.csv` (638 rows), split, config, probe records |
| `experiments/` | the frozen Step 4b per-clip experiment: config, audits, edit logs, scores |
| `notebooks/` | `diarizen_runner.ipynb` — DiariZen's separate environment |
| `Errors.txt` | raw hand-written notes from listening to clips, unedited |

`main_kaggle.ipynb` is the sweep notebook that produced the checkpoints;
`main_kaggle_2.ipynb` is the experiment bench and writes nothing.

---

## 1 · Dataset

**Extracted: 99 of 100 clips, 12.26 h.** One clip (`GUVrL5ltiP4__23_86`) failed
extraction and is excluded everywhere. Audio is 16 kHz mono, cut to the exact
`[start_sec, end_sec]` window.

| property | value |
|---|---|
| reference turns / utterances | 9,544 / 9,482 |
| overlapped speech | 3,148 s = **7.13 % of scored time**, 12.8 % of reference words |
| clips with no overlap | 9 |
| speakers per clip | 2–8 (`{2:25, 3:29, 4:28, 5:9, 6:4, 7:2, 8:2}`) |
| scripts | Devanagari 25, Gujarati 12, Telugu 12, Tamil 10, Kannada 9, Oriya 9, Bengali 8, Gurmukhi 7, Malayalam 7 |

The ground truth gives `diarization_segments` and `asr_segments` sharing the same
boundaries in the same order, timestamps relative to `start_sec`.

**Two properties drive every design decision.** Overlap is 7.13 % of time and only
9 clips have none, so overlap handling is the dominant axis rather than an edge
case. And the reference is **code-switched with a dual-form convention** — 20,599
parenthesised Latin-alphabetic glosses (22,022 counting all round brackets), e.g.
`कॉफी(coffee)` — which means *surface form*, not just meaning, is what gets scored.

---

## 2 · Metrics, normalisation and leak discipline

### Metrics

**DER and JER** via `pyannote.metrics`, collar 0, overlap scored. Plus
speaker-count accuracy and MAE.

**cpWER** — concatenated minimum-permutation WER (CHiME-6). The Hungarian
assignment is **exact, not an approximation**: pair costs are independent, so
minimising the sum *is* the linear assignment problem. `verify_assignment_exact`
checks it against brute force.

**WDER** = (S_IS + C_IS)/(S + C) (El Shafey et al. 2019) — insertions and
deletions are excluded from both numerator and denominator.

**DI-cpWER** — diarization-invariant cpWER on flat speaker-agnostic streams. On
this corpus it equals WER in all 638 per-clip rows, so `cpWER − DI-cpWER` is
simply `cpWER − WER`; it is reported because that difference is the share of cpWER
attributable to *attribution* rather than *words*. It is not a clean zero-floor
quantity — the oracle's own DI-cpWER is 0.0818, from cross-speaker word ordering
under overlap.

**Pooling convention.** DER components (miss / FA / confusion) pool as seconds and
are divided once. **JER, speaker-count accuracy and MAE are unweighted per-clip
means** — JER has no additive decomposition, so it cannot be pooled. cpWER / WER /
WDER pool their counts.

### Text normalisation

One normalizer, `reference.normalize_text()`, applied **identically** to reference
and hypothesis. Every step is here because it was measured on this corpus:

| # | step | why, measured |
|---|---|---|
| 1 | Unicode NFC | 154 segments are not NFC (Bengali `ড়` U+09DC decomposes) |
| 2 | strip inline non-speech tags | 236 *speech* segments carry `<unintelligible>` |
| 3 | strip glosses | 24,914 annotation tokens, 16.7 % of the raw reference |
| 4 | strip zero-width | ZWNJ ×193, ZWJ ×1 |
| 5 | punctuation → space | danda `।` ×1,706 plus 22 other marks |
| 6 | case-fold | 3,180 Capitalized + 835 ALLCAPS Latin tokens |
| 7 | whitespace collapse | — |

Step 3 is the only asymmetry and it is **inert on hypotheses**: the dual-form
convention exists only in the reference, so the step is a no-op on system output —
verified as **0 token difference across 361,628 hypothesis tokens**. Tokenization
is whitespace after normalization.

### Leak discipline

**Ground truth is never a pipeline input.** Stages receive `ClipInput`, a frozen
dataclass that structurally carries no reference field — that type boundary, plus
the `isinstance` check in `diarization.py`, is what actually enforces the rule.
(`utils.assert_no_reference_fields` is a DataFrame column-name check and is a
backstop, not the mechanism; its docstring says so.)

- Speaker counts are never passed to a diarizer.
- Language is never passed to a recogniser. Saaras is called with
  `language_code="unknown"`; Whisper detects from audio.
- `ref_lang_script` is derived from the reference and reaches no stage — it is
  used only to *group* results.
- The one deliberate exception is an oracle-language **ablation**, quarantined and
  excluded from every headline table.

The 50/49 dev/test split is frozen in `results/split.json`. Tuning happened on dev;
test was scored once.

---

## 3 · Step 2 — baseline diarization

Four systems, all 99 clips, raw ground truth:

| system | DER | miss | FA | confusion | JER | spk acc | spk MAE |
|---|---|---|---|---|---|---|---|
| `reverb-v2` | **0.2521** | 0.0719 | 0.0681 | 0.1121 | 0.3834 | 60.6 % | 0.54 |
| `diarizen-large` | 0.2625 | 0.1143 | 0.0640 | 0.0842 | 0.3660 | 72.7 % | 0.30 |
| `pyannote community-1` | 0.2634 | 0.1079 | 0.0602 | 0.0954 | 0.3768 | 78.8 % | 0.23 |
| `pyannote 3.1` | 0.2637 | 0.1079 | 0.0602 | 0.0957 | 0.3785 | 71.7 % | 0.35 |

25–26 % DER is unsurprising here: nine-language code-switched YouTube audio, no
collar, full overlap scored, and 20 % of reference segments under 0.5 s.

**The four systems sit within 0.0116 DER of each other** — inside the noise of a
99-clip corpus. They differ far more in *character* than in aggregate: `reverb-v2`
has the lowest miss (0.0719) but the worst confusion (0.1121) and worst speaker
counting (60.6 %), because it emits long loose turns averaging 15.5 s against
DiariZen's 2.6 s. Ranking by DER alone would report "the models are equivalent"
and discard the only interesting result.

**`pyannote 3.1` is excluded from the fusion**: it shares
`pyannote/segmentation-3.0` with `community-1` and differs by **0.320 s** of
miss+FA over 12.4 h. Under majority voting they are one vote counted twice.

### Overlap

| system | overlap predicted | as % of reference overlap |
|---|---|---|
| `reverb-v2` | 8.0 min | 15 % |
| `diarizen-large` | 44.2 min | 84 % |
| `community-1` | 36.9 min | 70 % |
| `pyannote-3.1` | 36.9 min | 70 % |

Note the inversion: `reverb-v2` predicts the least overlap and still has the
lowest miss, because its long turns blanket the timeline.

Fusion overlap **recall is 21.5 %** — it finds 678 s of the reference's 3,148 s. It
marks 1,560 s as ≥2 speakers, so **882 s of the overlap it does claim is in the
wrong place** (precision 43.5 %). Majority voting suppresses second speakers,
because a concurrent speaker needs the same majority as the first and `reverb-v2`
votes against nearly all of them.

---

## 4 · Step 3 — ASR on diarized segments

### How audio reaches the recogniser

**Per-segment** (Saaras). Cut the audio at the diarized turns and transcribe each
separately, so speaker attribution is exact by construction — the segment *is* the
speaker. For Saaras this is not a preference but the only option: it returns
exactly one timestamp span per request (verified across v3/v4, 5 s and 20 s
inputs, every documented mode), so there are no word times to attribute with.

Before cutting, adjacent same-speaker turns separated by ≤ **1.0 s** are merged.
A recogniser handed a 0.3 s fragment has no context. Merged counts differ sharply
by source — `reverb-v2` yields 2,650 segments at median 7.66 s, `community-1`
10,279 at 1.13 s. **The diarization being scored is never modified**; merging is
an ASR segmentation choice applied to a copy.

Two hard rules at the boundary: segments under **0.30 s** are skipped; segments
over **29.0 s** are split with a 0.5 s step-back (a server limit — over it the API
answers `400 Audio duration exceeds the maximum limit of 30 seconds`).

**Long-form** (Whisper only). Transcribe the whole clip and assign each word to
the turn it **overlaps most**; ties and words landing in a gap go to the nearest
turn by midpoint, so no word is silently dropped — a dropped word would present as
a deletion and quietly flatter WDER. Under overlap this is
*dominant-speaker-wins*, a real ceiling rather than an implementation shortcut.

### Results, all 99 clips

| system | ratio | WER | **cpWER** | DI-cpWER | attribution | WDER |
|---|---|---|---|---|---|---|
| `saaras-v3@reverb-v2` | 0.960 | 0.2728 | 0.3957 | 0.2728 | 0.1229 | 0.1128 |
| **`saaras-v3@fusion`** | 0.979 | 0.2787 | **0.3181** | 0.2787 | 0.0394 | **0.0628** |
| **`saaras-v3@fusion+xlitpc`** (shipped) | 0.979 | **0.2617** | **0.3049** | 0.2617 | 0.0431 | **0.0603** |
| `saaras-v3@fusion+xlit+num` (corpus vocab) | 0.981 | 0.2585 | 0.3017 | 0.2585 | 0.0432 | 0.0602 |

`ratio` = hypothesis words ÷ reference words.

**The fusion front-end is worth −19.6 % relative cpWER and −44.3 % relative WDER**
over `reverb-v2`, and cuts the attribution component by two thirds
(0.1229 → 0.0394), winning **88 of 99 clips**. Flat WER is slightly *worse*
(0.2728 → 0.2787) because the fusion preserves overlap, so overlapped audio is
transcribed once per speaker and some words appear twice. cpWER is the metric the
brief asks for and it moves decisively the right way.

### Strategy for overlapping speech

The brief asks this directly.

**What we do: per-speaker segment cutting.** Because the fusion preserves overlap,
an overlapped region belongs to two turns and the same audio is sent to Saaras
**once per speaker**. Each pass hears the mixture; attribution is exact by
construction. 1,693 s of audio is transcribed twice this way.

**What we rejected.** Long-form with post-hoc word assignment was measured and is
strictly worse: attribution cost 0.1229 of cpWER against 0.0394 for per-segment
cutting on the fusion — and it cannot work for Saaras at all.

**What it costs.** Duplicate transcription inflates insertions; per-clip insertion
rate correlates with duplicated-audio *fraction* at r = 0.94 (R² = 0.886). A
deterministic deduplicator helped by −0.0064 cpWER on the pre-fix baseline and
**reverses sign to +0.0027 after the provenance fix**, so it is not applied — what
remains in overlapped regions is mostly genuine simultaneous speech.

**The ceiling.** Oracle ASR on oracle diarization still scores cpWER 0.1242, and
exactly 0 on the 9 zero-overlap clips. Target-speaker ASR or separation before
recognition are the routes past it; both were out of scope.

### Saaras v3 vs v4

Paired on the 9 clips both cover: v3 WER 0.2284 / cpWER 0.3197, v4 0.2606 /
0.3485. v3 wins here, but an earlier 6-clip sample had v4 ahead and a paired
bootstrap put it ahead in only 82 % of resamples. **The comparison is genuinely
unsettled and is not claimed either way.**

---

## 5 · Why Whisper was not selected

Whisper was benchmarked and rejected on evidence, across three independent probes.

1. **Configuration is not the problem.** Eight configurations over ten clips —
   greedy vs beam 5, self-LID vs forced LID, turbo vs `large-v3`, long-form vs
   per-segment — span **0.906 to 0.964 WER**. The entire spread is 0.058.
2. **Language identification is not the problem.** Handing Whisper the correct
   language changes WER by at most 0.007 across three paired configurations, and
   in one pair the oracle is *worse*. `large-v3` is identical to four decimals.
3. **It is an Indic capability gap.** AI4Bharat **IndicConformer-600M** on the
   same ten clips, same segmentation, language supplied to both, scores **WER
   0.4158 (CTC)** against Whisper's best 0.9058 — **2.2× better** from a model
   2.6× smaller (600 M against 1.55 B).

Whisper's failure mode is specific: on **92 of 99 clips it produced a different
script from the reference**, emitting fluent English *translation* rather than
transcription. And **Whisper has no Oriya** — `or` is absent from its 99
languages, so 9 % of this corpus is outside its label set entirely; the pipeline
substitutes `bn` and records the substitution. IndicConformer matched all ten
scripts including Oriya.

`large-v3-turbo` is distilled and its language identification is badly degraded
here — it returned `en` on 75 of 99 clips and then translated. `large-v3` on the
same audio got 34 of 35 right, which is why language ID and decoding are split
across two models.

**Saaras v3 remains best** (0.2876 vs IndicConformer's 0.4158 on the shared ten
clips) — and by more than it appears, since it detects its own language where
IndicConformer had to be given one.

Whisper and IndicConformer are deliberately **not** in the headline table.
Whisper's 99-clip run used a configuration later shown to carry three independent
defects at once, and quoting it as "Whisper's performance" would be wrong.

---

## 6 · Step 4a — DOVER-Lap fusion

**Motivation.** The single systems are statistically indistinguishable in
aggregate but make different errors: `reverb-v2` under-segments and mislabels,
DiariZen over-segments precisely. That is the classic condition for ensembling.

**Method** (Raj et al., SLT 2021, re-implemented in `sarvam_diar/refinement.py`):

1. **Label alignment.** Hungarian mapping of each system's labels onto a growing
   centroid, so `SPEAKER_01` in one system and `SPEAKER_03` in another become the
   same identity before voting.
2. **Overlap-preserving voting.** Per 10 ms frame, each mapped speaker is
   thresholded **independently** rather than taking a single argmax. This is the
   part that matters: an argmax vote cannot emit two simultaneous speakers, and
   7.13 % of this corpus is overlapped.
3. Turns shorter than `min_dur = 0.20 s` are dropped.

**Configuration**: `community-1 + reverb-v2 + diarizen-large`, equal weights,
threshold 0.5 (2-of-3), `min_dur` 0.20 — in `results/fusion_config.json`.

### GT-freeness, stated precisely

**Fusion is GT-free at inference.** `dover_lap` takes exactly two inputs: the
three constituent RTTMs, and the clip duration — and that duration comes from the
CSV manifest's `start_sec`/`end_sec`, not from any annotation. No reference
quantity is reachable from the operator.

**The configuration was selected on dev and frozen before test was scored.**
Candidates — three-vs-four members, equal-vs-rank weighting, `threshold`,
`min_dur` — were compared by **dev** DER and the winner recorded before the test
half was touched. That is legitimate hyperparameter selection on a held-out split,
but it does use reference DER: **the operator is GT-free, the selection is not**,
and conflating the two would be the overclaim. Two qualifications:

- The dev/test boundary is itself stratified by script and speaker-count band,
  both reference-derived. Held out from tuning, but drawing it was not a GT-blind
  act.
- The chosen configuration is **not fitted to test**. Sweeping `threshold` ∈
  {0.34, 0.5, 0.67} × `min_dur` ∈ {0.0, 0.1, 0.2, 0.3, 0.5} and scoring each half
  separately, dev-argmin and test-argmin are the *same* point (0.34 / 0.0), and
  the shipped setting sits within 0.0001 DER of it on both halves (dev 0.2390,
  test 0.2481). A flat plateau, not a peak found by looking at test.

### Result, with significance

Paired bootstrap of the **pooled** Δ DER over clips, 10,000 resamples:

| comparison | Δ DER | 95 % CI | per-clip W/L |
|---|---|---|---|
| `community-1` → fusion | −0.0193 | [−0.0254, −0.0139] | **83 / 16** |
| `pyannote-3.1` → fusion | −0.0195 | [−0.0309, −0.0090] | 80 / 19 |
| `diarizen-large` → fusion | −0.0184 | [−0.0287, −0.0067] | 80 / 19 |
| `reverb-v2` → fusion | −0.0079 | **[−0.0353, +0.0161]** | **42 / 57** |

**The fusion beats three of four constituents decisively and is statistically
indistinguishable from `reverb-v2` on DER.** The pooled −0.0079 is carried by a
few long clips; per clip it loses 57 to 42, and on the dev half it is 6.8 %
*worse*. An earlier draft reported "−3.1 % relative DER over the best single
system" as the Step 4 result; that figure is a point estimate whose interval spans
zero, and it is **withdrawn**.

What survives, and is testable:

| | `reverb-v2` | **FUSION** |
|---|---|---|
| DER | 0.2521 | 0.2442 *(not significant)* |
| **JER** | 0.3834 | **0.3608** |
| **speaker-count accuracy** | 60.6 % | **79.8 %** |
| speaker MAE | 0.54 | **0.21** |
| confusion | 0.1121 | **0.0806** |
| false alarm | 0.0681 | **0.0498** |

The fusion is also the **only system that is 1st or 2nd on both halves** of the
split across 2,000 random 50/49 resamples, 100 % of the time (`reverb-v2` ~40 %,
the other three 0 %). Rank stability, not variance reduction.

**The real payoff is downstream**: as an ASR front-end the fusion is worth
−19.6 % relative cpWER and −44.3 % relative WDER, winning 88 of 99 clips. That is
where Step 4a earns its place, not in DER.

---

## 7 · The provenance fix — not a model improvement

**This is a bug fix, not an algorithmic gain, and is reported separately for that
reason.**

**Symptom.** The Saaras sweep reported 97 of 99 clips. The by-script table summed
to 97, missing one Devanagari and one Gurmukhi clip worth 9,978 reference words.

**Investigation.** The two missing clips were the **longest** (1,822 s, 476
segments) and **fourth-longest** in the corpus. `transcribe_segments` fans
segments out with `pool.map`, which re-raises the first exception — so a single
non-retryable HTTP 400 anywhere in a clip discarded the entire transcript. A clip
with 476 segments has 476 chances to hit one.

**A second, larger defect surfaced during the fix.** For **27 of 99 clips** the
segmentation actually sent to Saaras could not be reproduced from the fusion on
disk. All 27 reproduced exactly from a **two-system** `community-1 + reverb-v2`
fusion: DiariZen had been absent from the session that materialised the fusion the
ASR then consumed. Independently corroborated — that session's `step2_metrics.csv`
has 297 rows over three models with no DiariZen, against 396 over four locally.

So DER described the 3-system fusion while cpWER described a fusion that was
2-system for 27 % of the corpus.

**Root cause.** `settings_key` recorded *decoding* settings so stale work was
visibly stale, but recorded nothing about *segmentation*. A checkpoint built from
a different diarization was indistinguishable from a current one, and `is_done()`
skipped all 27.

**Fix.** `segmentation_key()` fingerprints the turns a transcript was cut on,
**re-derived from the stored segment spans** at audit time rather than read from
the payload, so checkpoints written before the fix are auditable too.
`run_segmented` skips a clip only when that re-derived key matches the
segmentation it would use, so drift self-heals. `audit_segmentation()` reports the
re-run set without transcribing. Per-segment failures are isolated so one bad
segment cannot discard a clip.

**Effect of re-transcribing the 27 clips against the correct fusion:**

| | ratio | WER | cpWER | WDER | insertions |
|---|---|---|---|---|---|
| before | 1.0105 | 0.3066 | 0.3381 | 0.0788 | 8,959 |
| **after** | 0.9786 | 0.2787 | **0.3181** | 0.0628 | **5,261** |
| Δ | −0.032 | −0.028 | **−0.0200 (−5.9 %)** | −20 % rel | **−41 %** |

For scale: the GT-perfect ceiling for overlap deduplication was −0.023 cpWER. **A
provenance fix captured almost the entire prize an LLM refinement layer was being
designed to chase** — with no model, no API and no tuning. It also *removed* that
opportunity: duplicated tokens fell 66 % and the free dedup rule reversed sign.

> **`settings_key` still does not do its job on the per-segment path.** Committed
> values are `beam?-cond1-lidself-batch0` (297 payloads) and `None` (242) — the
> literal `?` is a missing beam width, and it records neither `merge_gap` nor the
> diarization model. `segmentation_key` is what actually caught the bug. Widening
> `settings_key` is listed as future work.

---

## 8 · Step 4b — script correction

### The finding

The reference writes code-switched English **phonetically in the native script**.
Measured: the reference is 99.98 % Indic script (**27 Latin tokens in 123,896**);
the Saaras hypothesis is 3.0 % Latin (3,689 tokens). Of 21,357 substitutions,
**3,481 (16.3 %) are reference-Indic against hypothesis-Latin**:

```
ఐ → i     టు → to    ఓకే → okay   సో → so
એમએલ → ml  एंड → and  યુ → you     బోత్ → both
```

**These are not recognition errors.** Saaras heard the word and wrote it in the
wrong script for this corpus's convention.

### Shipped: per-clip correction (`+xlitpc`)

Each clip is corrected **in isolation**. The model receives exactly two things:
the Latin tokens in *that clip's own hypothesis*, and the dominant script of *that
same hypothesis*. No other clip, no corpus vocabulary, no split membership, never
the reference. Ground truth is opened only by the scorer, after inference.

Only the *script* of an already-recognised token may change; replacement is
whole-token, so token count is invariant and words cannot be added, deleted,
reordered or translated. **Numerals are excluded** — spelling a digit out changes
the token count.

Six guards, each counted: abstain-on-null, single token with no whitespace, no
digits, reject still-Latin, **reject any rendering not in the clip's own script**,
reject identity. The prompt carries an explicit abstention instruction —
*"returning null is always safe and is the correct answer when in doubt."*

`us.anthropic.claude-sonnet-4-6`, temperature 0, prompt `xlit-perclip-v1`, batched
at 60 words. **Model, prompt, temperature and every threshold were frozen and
committed at `f284e59` before the test half was scored**, so the freeze is
checkable from git history rather than asserted.

| | WER | cpWER | WDER |
|---|---|---|---|
| baseline `saaras-v3@fusion` | 0.2787 | 0.3181 | 0.0628 |
| **+ per-clip script correction** | **0.2617** | **0.3049** | **0.0603** |
| dev (50) | 0.2317 → 0.2258 | 0.2676 → **0.2619** | 0.0439 → 0.0424 |
| **test (49), held out** | 0.3132 → **0.2881** | 0.3552 → **0.3364** | 0.0769 → 0.0737 |

| split | mean Δ cpWER | 95 % CI | better | worse | tie |
|---|---|---|---|---|---|
| dev | −0.0067 | [−0.0108, −0.0034] | 26 | **0** | 24 |
| test | −0.0135 | [−0.0241, −0.0051] | 29 | **0** | 20 |

**3,249 of 121,243 tokens changed (2.68 %). 2,232 helpful edits, 0 harmful, 1,017
neutral. Zero cross-script corruptions.** Attribution is per-token: replacement is
whole-token, so before/after streams have equal length and the alignment op at
each changed position says whether that edit turned an error into a match or the
reverse.

**Zero harmful is structural, not luck.** With 27 Latin tokens in 123,896
reference tokens, a Latin hypothesis token is almost never already a match — an
edit can only help or be inert. The downside is bounded at zero.

### The earlier variant, and why it does not ship

An earlier version extracts the Latin and digit vocabulary from **all 99 clips'
hypotheses** — 1,651 unique (script, token) pairs and 382 numerals — and applies a
frozen lookup table. It scores **0.3017**, marginally better.

It is **transductive**: the table's coverage was fixed with the test clips'
transcripts in view, so both halves are in-vocabulary by construction and the
dev/test split cannot detect a generalisation failure.

| split | baseline | corpus vocab | dev-only vocab |
|---|---|---|---|
| dev | 0.2676 | 0.2582 (−0.0094) | 0.2582 (−0.0094) |
| test | 0.3552 | 0.3337 (**−0.0216**) | 0.3511 (**−0.0042**) |
| all | 0.3181 | 0.3017 (−0.0164) | 0.3118 (−0.0064) |

**About 80 % of its measured test gain depends on the table having seen test
vocabulary.** The dev column is unchanged to four decimals — the tell that no
dev-side diagnostic could have caught it. The cause is coverage, not modelling:
dev holds 403 Latin types against test's 1,365, and only **8.6 % of test types
appear anywhere in dev**.

| system | cpWER (99) | test Δ | test W/L | information used |
|---|---|---|---|---|
| `+xlit` | 0.3045 | −0.0197 | 27 / 0 | all 99 hypotheses, **incl. test** |
| `+xlit+num` | **0.3017** | −0.0216 | 38 / 1 | all 99 hypotheses, **incl. test** |
| dev-built vocabulary | — | −0.0042 | — | dev only |
| **per-clip (shipped)** | 0.3049 | **−0.0189** | **29 / 0** | **one clip only** |

**The transductive property was not necessary to the result.** Per-clip scoping
recovers ~90 % of the gain while supporting a clean held-out claim, and 4.5× the
honest inductive estimate. The two are 0.0032 cpWER apart; the per-clip variant
ships because its number survives review.

### Why it is GT-free — verified, not asserted

The target script comes from `data.dominant_script` on the *hypothesis*; the
spellings come from a general model's knowledge. An independent audit rebuilt
every cache key: the hypothesis-built vocabulary produces **43 keys that hit 43 of
43** and account for every cache entry on disk with none unexplained, while
rebuilding from the **reference** produces 124 keys of which **0 hit**.

### Failure modes

**Abstentions** (181 of 1,916 types, 9.4 %) are single letters, contraction
remnants and fragments — `s`, `a`, `t`, `ll`, `idi`, `roo`, `tns`. Correct refusals.

**Neutral edits** are the real limitation: 1,017 of 3,249 (31 %) are correct
transliterations that still do not match, because the annotator chose a different
spelling — `you → యూ`, `sir → सर`. Free, but unrealised headroom.

**Successes** split in two. Code-switched English is expected: `youtube → यूट्यूब`,
`challenges → चॅलेंजेस`. More interesting are **romanised Indic words**, where the
model produced the native word rather than a phonetic rendering of English —
`mhanun → म्हणून`, `aahet → आहेत`, `nahi → नाही`, `thik → ठीक`. A pure
sound-transliterator would have failed these.

### One defect, disclosed

The first frozen run produced 431 spurious "abstentions" on a single test clip.
Not model judgement: the reply hit `stop_reason: max_tokens` at 3,999 of 4,000
output tokens, the JSON never closed, and the code counted an unparseable response
as abstention on every word. The frozen run's test result was −0.0100.

The repair batches at 60 words and treats truncation as a hard error; prompt,
model, temperature and thresholds are byte-identical. It was made **after** test
was scored, which is disclosed rather than hidden — defensible because
`stop_reason` is returned at inference time and needs no reference, so a correct
harness would have caught it before any scoring. Both results are kept in
`experiments/xlit_perclip/`.

**The lesson, twice over in this project: a silent failure that looks like a valid
result is worse than a crash.**

---

## 9 · Failure analysis

### Overlap dominates

Frame-level decomposition against ground truth:

| region | ref sec | miss | FA | confusion | share of all error |
|---|---|---|---|---|---|
| true silence | 0 | 0 | 1,372 | 0 | 12.6 % |
| single speaker | 38,213 | 2,407 | 841 | 3,283 | 60.0 % |
| **overlap** | 6,356 | **2,664** | 5 | 310 | **27.4 %** |

**52.5 % of all miss is in overlapped regions.** Overlapped words are deleted at
**3.9× the clean rate** (18.5 % vs 4.8 %). A perfect separator would be worth
**−0.031 WER** — the largest ceiling measured here.

**Boundary tightness is a substantial effect.** 39.4 % of missed frames lie within
200 ms of a hypothesis boundary against 14.5 % of *all* frames — an enrichment of
**2.71×** (2.99× at 100 ms, 2.14× at 500 ms) over 4.4 M frames. The control
matters: without it the raw 39.4 % means nothing, since boundaries are dense.

**False alarm is mostly spill, not invention.** 1,372 s of FA in true reference
silence across 3,427 runs, median run 0.20 s; only 7.9 % is in runs over 5 s.

### The brief's hint does not hold on this corpus

Testing shallow discourse signals against whether a segment's speaker label is
wrong, over the 6,312 segments of the final system (base rate 33.1 %):

| signal | n | wrong | lift |
|---|---|---|---|
| duration < 0.5 s | 949 | 68.3 % | **2.06×** |
| word count 0 | 599 | 68.6 % | **2.07×** |
| gap to previous < 0.05 s | 2,952 | 41.9 % | 1.27× |
| gap to previous ≥ 1.5 s | 1,084 | 26.0 % | 0.79× |
| **speaker changed at boundary** | 5,296 | 34.0 % | **1.03× (none)** |
| **word repeats across boundary** | **36** | 36.1 % | **1.09× (none)** |

"Repeated text across a speaker boundary suggesting a false split" occurs **36
times in 6,312 segments** and does not predict error. Worse, the signals that *do*
work are duration proxies pointing at the wrong targets:

| duration | segments | error rate | share of wrong-label *time* |
|---|---|---|---|
| < 2 s | 2,713 | 56.0 % | 30.1 % |
| 2–5 s | 1,416 | 30.2 % | 33.9 % |
| ≥ 5 s | 2,109 | **5.6 %** | **36.0 %** |

A detector keyed on short segments finds errors at twice the base rate but reaches
only 30 % of the damage; the 36 % sitting in long segments has a 5.6 % error rate,
so flagging it would be 94 % false positives.

### By script, shipped final system

| script | clips | WER | cpWER | WDER |
|---|---|---|---|---|
| Telugu | 12 | 0.3535 | **0.4123** | 0.1127 |
| Malayalam | 7 | 0.3322 | 0.3900 | 0.0736 |
| Oriya | 9 | 0.3464 | 0.3868 | 0.0535 |
| Bengali | 8 | 0.2467 | 0.3750 | **0.1233** |
| Gujarati | 12 | 0.2950 | 0.3332 | 0.0673 |
| Kannada | 9 | 0.2372 | 0.3281 | 0.0696 |
| Gurmukhi | 7 | 0.2770 | 0.3229 | 0.0548 |
| Tamil | 10 | 0.2471 | 0.2723 | 0.0280 |
| Devanagari | 25 | 0.1906 | **0.2006** | 0.0293 |

**Telugu is 2.06× Devanagari on cpWER.** Language is a far larger axis of
variation than anything else measured. WER and cpWER rank scripts differently:
**Bengali has the third-best WER but the worst WDER** — its words are recognised
well and its speakers attributed badly. Group sizes are 7–25 clips, so read the
ordering as indicative.

### Attribution vs word error

**14.1 % of remaining cpWER is attribution** (0.0431 of 0.3049); the other 85.9 %
is word error.

That gap *widened* after Step 4b, from 0.0394 to 0.0431, and it looks like a
regression but is not one. Step 4b cut DI-cpWER by 0.0170 and cpWER by only
0.0133, and `cpWER − DI-cpWER` is the cost speaker-attributed scoring adds over
speaker-agnostic scoring — not attribution quality. A word that was both
misrecognised *and* misattributed scored wrong under both metrics; once the word
is fixed it scores right under DI-cpWER and still wrong under cpWER. **Correcting
words unmasks misattributions that word errors were hiding.** The direct
attribution metric moved the other way: WDER 0.0628 → 0.0603.

---

## 10 · Negative results

Recorded because a negative result on a well-motivated idea is worth as much as a
positive one. **Grids below were run on the dev half** (`results/split.json`),
where every Step 4 configuration decision was made. None was promoted, so none
affects a reported result.

| intervention | outcome |
|---|---|
| **1-of-3 overlap voting** | With three equal voters there is no threshold between 2-of-3 and 1-of-3. Dropping 0.5 → 0.30 moves DER **0.2442 → 0.3193** and false alarm **0.0498 → 0.2180**, buying a miss reduction of only 0.1138 → 0.0370. |
| **Non-uniform weights** | Moves the quantisation cliffs, does not remove them; no weighting beat equal. |
| **Segmentation transplant** (`reverb-v2` boundaries, `diarizen-large` labels) | Miss improved as predicted, 0.0731 → 0.0581, but confusion worsened 0.0894 → 0.0944 against DiariZen's own 0.0849. Its low confusion is entangled with its own 2.6 s turns and does not survive being painted onto 15.5 s ones. No cell of the 4×4 grid beat `reverb-v2` alone. |
| **Blind boundary padding** | Exploits a measured 137 ms trailing bias for a net **−0.0031 DER** — real and reproducible, and far too small to report as an improvement. |
| **Overlap deduplication** | Helped by −0.0064 cpWER before the provenance fix; **hurts by +0.0027 after**. What remains in overlapped regions is mostly genuine simultaneous speech. |
| **ConvTasNet source separation** | On 100 GT-located overlap runs, word recovery **52.6 % → 50.0 %**. Scoped narrowly: an *out-of-domain 8 kHz* separator, given *oracle* overlap regions, does not help. A 16 kHz in-domain separator is **untested**. |
| **LLM contextual refinement** | Null. All four pre-registered abandon criteria fired: CI includes zero, 20 % win rate against a 60 % bar, insertions did not fall, 29 % revert rate against a 20 % bar. |

### The LLM refinement null, in detail

Architecture in `sarvam_diar/llm_refine.py`. The model sees a speaker-attributed
window and returns **only `{id, text}`** — speakers and timestamps are never in
the response schema, so changing them is **structurally impossible** rather than
merely forbidden. Guards: reject a window whose returned id set differs; revert a
segment on dominant-script change, Latin-fraction jump > 0.10, growth beyond
1.05×, or unlicensed edits exceeding `max(1, ⌈0.25·len⌉)`.

Ten clips, selected **GT-blind** — dominant script *of the hypothesis*, duration,
segment count, predicted overlap from segment spans, predicted duplication from
hypothesis-vs-hypothesis matching. No WER, cpWER, WDER or reference overlap was
used, because selecting clips by measured error is ground truth leaking into
experimental design. Three zero-overlap control clips arose by construction.

The architecture held perfectly — all 20 refined payloads have segmentation
byte-identical to source. **The guard was the wrong shape.** 7 of 24 applied edits
(29 %) introduced a script absent from the original, and all 7 passed the
*dominant*-script rail because one corrupted token in forty cannot move a
majority:

```
Telugu   పార్టీ  →  パーティー   (Japanese katakana)
Bengali  এটাও   →  것도        (Korean hangul)
Marathi  हे     →  হে          (Bengali)
```

The correct rail is **per-token** — reject any output token whose script is absent
from the input. It was deliberately not applied, because the guards were frozen
before the run and changing them after seeing results would invalidate the pilot.
The per-clip Step 4b stage carries that rail from the start, and it never fired.

### Two ideas that were audited, not implemented

An **MSDD-style verifier** over constituent disagreement and an
**OSD∩constituent intersection** got only as far as measuring how precise their
candidate sets would be — 11.5 % and 23.6 %. Both were dropped before
implementation. They are reported as audits, **not** as experiments, and should
not be read alongside the measured results above.

---

## 11 · Ground-truth audit

An alignment audit flagged **39 of 99 clips whose reference is displaced 1.0–5.0 s
relative to the speech** — **38 of the 39 in the same direction** (reference late;
the exception, `8r2Nltl0W4o__259_320` at −4.5 s, is not VAD-corroborated). The
detector uses 10 ms speech-activity rasters, 2-of-3 consensus across *independent*
systems, and IoU across ±5 s lags, with optional energy-VAD corroboration.

**23 met the auto-accept bar.** Applying them (a **diagnostic**, not the headline)
moves fusion DER 0.2442 → **0.1959 (−19.8 %)** — **and inverts the ranking among
the single systems**: `reverb-v2` best → last, `diarizen-large` second → first.
The least temporally precise model looked best *because* the reference was wrong;
long loose turns still overlap a reference displaced by a second.

**All headline numbers use the raw, unmodified ground truth.** The corrections
live in `results/gt_alignment_qc/corrections.json`, applied to a **copy** at
scoring time; the reference on disk is never edited. Nothing in the scoring path
calls them — `evaluation.shift_turns` has no call site anywhere in the repo, and
that absence is what makes the claim checkable rather than merely asserted.

**Two failure families, both worth listening to:**

- **A · unannotated stretches (false alarm).** 39 stretches over 5 s where the
  reference has no speaker at all, totalling 8.6 minutes and carrying 218 s of
  false alarm — 8.1 % of `community-1`'s FA, the system they were measured on.
  The top 10 carry 185.4 s of that 218 s (85 %).
- **B · implausibly long reference turns (miss).** 16 turns carrying 563 s of
  miss. **Row 1 alone is 546 s (97 %)** — one pathological `Speaker_D` label
  rather than a labelling convention.

**One caveat on the evidence.** These figures use `community-1`, which shares
`pyannote/segmentation-3.0` with `pyannote-3.1`, so those two agreeing that speech
exists in a gap is *not* independent corroboration.

**11 of the 15 worst-DER clips are in the 39-clip flag set**, and the tail skews
short — 9 of the 15 are under 90 s, where a 4 s shift is catastrophic.

---

## 12 · Running it

### The default path — nothing to configure

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb          # SUBSET = 10 first, then 0
```

Every model output is committed, so the notebook re-scores rather than re-runs.
No GPU, no API key, no gated model access. The one network call is a first-run
fetch of the assignment's `youtube_segments.csv`, which is not redistributed here.

On Colab, use the badge at the top — it shallow-clones the repo and installs only
what is missing, pinning numpy so `pip` cannot swap it under a kernel that has
already imported it.

### Running a stage from scratch

| stage | needs | cost |
|---|---|---|
| 1 · extract 99 clips | `yt-dlp`, network | ~1.3 GB audio, no credentials |
| 2 · `community-1`, `pyannote-3.1` | GPU + **`HF_TOKEN`** + accepted licences | ~40 min on a T4 |
| 2 · `reverb-v2` | GPU + `HF_TOKEN` | ~25 min |
| 2 · `diarizen-large` | **a separate environment** | `notebooks/diarizen_runner.ipynb` |
| 3 · Saaras v3 ASR | **`SARVAM_API_KEY`** | ~70 min, paid per request |
| 4a · DOVER-Lap fusion | nothing — pure function of Step 2 | seconds |
| 4b · script correction | **`AWS_BEARER_TOKEN_BEDROCK`** | ~70 requests |
| 5 · scoring, tables | nothing | seconds |

**DiariZen is the one hard blocker** — its dependencies conflict with pyannote's,
so it cannot run in the same kernel. Its RTTMs are committed, which is why it is
never required.

Credentials go in a gitignored `.env` at the repo root. Colab and Kaggle Secrets
are read under the same names. **No key value is ever logged** — the idiom
throughout is `f"token present: {bool(token)}"`.

### Notebook roles

| | role | writes? |
|---|---|---|
| `main.ipynb` | **the submission** — Steps 1–5 from checkpoints | no |
| `main_kaggle.ipynb` | the sweep notebook that produced the checkpoints | yes, exclusively |
| `main_kaggle_2.ipynb` | the experiment bench — probes, hypotheses | **never** |

A finding is **promoted, not executed**: when a probe settles a question, flip the
corresponding flag in `main_kaggle` and sweep there. Never sweep in the bench.

### Verify before trusting a sweep

```bash
python3 tools/audit_segmentation.py --root checkpoints   # expect: no mismatches
```

A mismatch means transcripts were produced against a different diarization than
the one on disk — the defect that cost 0.0200 cpWER before it was caught. Nothing
downstream should be scored until this is clean.

### Step 4b commands

```bash
# shipped per-clip stage (needs AWS_BEARER_TOKEN_BEDROCK)
python3 tools/xlit_perclip_experiment.py --split dev
python3 tools/xlit_perclip_score.py --splits dev,test

# earlier corpus-vocabulary variant — applies committed tables, no key needed
python3 tools/run_translit.py --root checkpoints
```

---

## 13 · Error explorer

`sarvam_diar/explorer.py` exports a standalone offline browser for per-clip
errors — no backend, no build step. Every number is computed by the Python
pipeline and serialised; the browser only draws it.

```bash
cd error_explorer
python3 serve.py            # http://localhost:8000
```

**Use `serve.py`, not `python -m http.server`.** The stdlib server does not
implement HTTP `Range`, so a browser cannot seek inside a 30-minute WAV — the
whole file must download before audio plays. `serve.py` is the stdlib handler plus
`206 Partial Content`. Opening `index.html` directly shows an empty page: browsers
block `fetch()` from `file://`.

The model dropdown is populated from whatever the export contains, with a per-clip
delta against the others. Audio is optional and can be symlinked in.

---

## 14 · Running the GPU sweeps on Kaggle

Steps 1–3 were run on Kaggle 2×T4. Upload the extracted audio as a Dataset, set
the accelerator to **GPU T4 ×2** and enable internet, then point cell 1.1 at the
dataset path. `HF_TOKEN` goes in Kaggle Secrets under that exact name.

Two things worth knowing before a long sweep:

- **Kaggle sessions do not share `/kaggle/working`.** A later session sees nothing
  a previous one wrote unless it was saved as a Dataset or a Version. The
  notebooks therefore re-link the dataset, fall back to an attached zip, and
  discover attached diarization *by content* rather than by slug.
- **Language ID runs on the second GPU** (`_lid_device`). Both models otherwise
  default to `cuda:0`, and ~5 GB of weights plus both activation peaks on one card
  is how a 16 GB T4 runs out of memory here.

Both notebooks `git reset --hard origin/main` on their first cell, so a push
changes what the *next* kernel restart loads, never a running one.

---

## Limitations

- **DER parity with `reverb-v2`.** The fusion's case rests on JER, speaker count
  and downstream cpWER, not on DER.
- 99 clips is small; 0.01-level DER differences are not resolvable, and the LLM
  pilot at 402 segments cannot resolve an effect below ~0.005 cpWER.
- Step 4b is script-only and cannot fix a misheard word; 31 % of its edits are
  neutral.
- The dev/test boundary is stratified on reference-derived fields.
- Whisper's 99-clip row is a known-broken configuration; its fixed configuration
  was only evaluated on 10 clips. IndicConformer likewise, and with an oracle
  language since it has no LID of its own. Saaras v4 covers 9 clips.
- The Step 5 pilot's per-edit log was overwritten by a later re-run to the same
  path — the same defect class as §7, on a stage that was measured and not shipped.
