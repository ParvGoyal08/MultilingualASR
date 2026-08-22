# Multilingual Speaker Diarization + ASR on Indic YouTube

Every number below was re-derived from the artefacts in this repository, and
independently re-verified by an adversarial review pass whose corrections are
incorporated. Scoring uses **collar 0.0 with overlapping speech included**, per
the brief.

**Pooling convention:** DER components (miss / FA / confusion) are pooled as
seconds and divided once. **JER, speaker-count accuracy and speaker MAE are
unweighted per-clip means** — JER has no additive decomposition, so it cannot be
pooled. cpWER / WER / WDER pool their counts.

---

## 1. Objective and dataset

Benchmark open-source diarization and multiple STT systems on 100 YouTube clips
of conversational Indic speech, then build a pipeline on top of the best
combination that measurably improves the output.

**Extracted: 99 of 100 clips, 12.26 h.** One clip (`GUVrL5ltiP4__23_86`) failed
extraction and is excluded everywhere. Audio is 16 kHz mono, cut to the exact
`[start_sec, end_sec]` window.

| property | value |
|---|---|
| reference turns / utterances | 9,544 / 9,482 |
| overlapped speech | 3,148 s = **7.13% of scored time**, 12.8% of reference words |
| clips with no overlap | 9 |
| speakers per clip | 2–8 (`{2:25, 3:29, 4:28, 5:9, 6:4, 7:2, 8:2}`) |
| scripts | Devanagari 25, Gujarati 12, Telugu 12, Tamil 10, Kannada 9, Oriya 9, Bengali 8, Gurmukhi 7, Malayalam 7 |

The ground truth gives `diarization_segments` (`Speaker A [start-end] | ...`) and
`asr_segments` sharing the same boundaries in the same order, with timestamps
relative to `start_sec`.

**Two properties of this corpus drive every design decision.** Overlap is 7.13%
of time and only 9 clips have none, so overlap handling is the dominant axis
rather than an edge case. And the reference is **code-switched with a dual-form
convention** — 20,599 parenthesised Latin-alphabetic glosses (22,022 counting all
round brackets), e.g. `कॉफी(coffee)` — which
means surface form, not just meaning, is what gets scored.

**Metrics.** DER and JER (`pyannote.metrics`, collar 0, overlap scored),
speaker-count accuracy and MAE; **cpWER** (concatenated minimum-permutation WER,
Watanabe et al. 2020), **WDER** = (S_IS + C_IS)/(S + C) (El Shafey et al. 2019),
and **DI-cpWER**, computed on flat speaker-agnostic streams. On this corpus
DI-cpWER equals WER in all 539 per-clip rows, so `cpWER − DI-cpWER` is simply
`cpWER − WER`; it is reported because that difference is the share of cpWER
attributable to attribution rather than words. It is not a clean zero-floor
quantity — the oracle's own DI-cpWER is 0.0818, from cross-speaker word ordering
under overlap. The Hungarian assignment in
cpWER is **exact, not an approximation** — pair costs are independent, so
minimising the sum *is* the linear assignment problem; `verify_assignment_exact`
checks it against brute force.

**Leak discipline.** Pipeline stages receive `ClipInput`, a frozen dataclass that
structurally carries no reference field — that type boundary, plus the
`isinstance` check at `diarization.py:287`, is what actually enforces the rule.
(`utils.assert_no_reference_fields` is a DataFrame column-name check with one
call site, on a Step 2 *output* frame; it is a backstop, not the mechanism.)
Speaker
counts are never passed to a diarizer. Language is never passed to an ASR system
from the reference — Saaras is called with `language_code="unknown"`. The one
deliberate exception is an oracle-language *ablation* (§4), quarantined and
excluded from every headline table.

---

## 2. Baseline diarization

Four systems, all 99 clips, raw ground truth:

| system | DER | miss | FA | confusion | JER | spk acc | spk MAE |
|---|---|---|---|---|---|---|---|
| `reverb-v2` | **0.2521** | 0.0719 | 0.0681 | 0.1121 | 0.3834 | 60.6% | 0.54 |
| `diarizen-large` | 0.2625 | 0.1143 | 0.0640 | 0.0842 | 0.3660 | 72.7% | 0.30 |
| `pyannote community-1` | 0.2634 | 0.1079 | 0.0602 | 0.0954 | 0.3768 | 78.8% | 0.23 |
| `pyannote 3.1` | 0.2637 | 0.1079 | 0.0602 | 0.0957 | 0.3785 | 71.7% | 0.35 |

25–26% DER is unsurprising here: nine-language code-switched YouTube audio, no
collar, full overlap scored, and 20% of reference segments under 0.5 s.

The four single systems sit within **0.0116 DER of each other** — inside the
noise of a 99-clip corpus. The models differ far more in *character* than in
aggregate: `reverb-v2` has the lowest miss (0.0719) but the worst confusion
(0.1121) and worst speaker counting (60.6%), because it emits long loose turns
averaging 15.5 s against DiariZen's 2.6 s.

**`pyannote 3.1` was excluded from fusion**: it shares `pyannote/segmentation-3.0`
with `community-1` and differs by 0.320 s of miss+FA over 12.4 h. Under majority
voting they are one vote counted twice.

### Ground-truth reliability

An alignment audit flagged **39 of 99 clips whose reference is displaced
1.0–5.0 s relative to the speech** — **38 of the 39 in the same direction**
(reference late; the exception, `8r2Nltl0W4o__259_320` at −4.5 s, is not
VAD-corroborated). The detector uses 10 ms speech-activity rasters, 2-of-3
consensus across *independent* systems, and IoU across ±5 s lags.

Applying the **23 auto-accepted corrections** (a **diagnostic**, not the
headline) moves fusion DER 0.2442 → **0.1959 (−19.8%)** — **and inverts the
ranking among the single systems**: `reverb-v2` best → last, `diarizen-large`
second → first. The least temporally precise model looked best *because* the
reference was wrong; long loose turns still overlap a reference displaced by a
second.

*(`gt_alignment_qc/README.md` and `BEFORE_AFTER.md` describe a superseded
27-detected / 18-applied run and should be disregarded; `corrections.json` and
`metrics_raw_vs_qc.csv` are current.)*

**All headline numbers in this report use the raw, unmodified ground truth.**
The corrections live in a separate manifest applied to a copy at scoring time;
the reference on disk is never edited.

---

## 3. Step 4 — DOVER-Lap fusion

**Motivation.** The single systems are statistically indistinguishable in
aggregate but make different errors: `reverb-v2` under-segments and mislabels,
DiariZen over-segments precisely. That is the classic condition for ensembling.

**Method** (Raj et al., SLT 2021, re-implemented in `sarvam_diar/refinement.py`):

1. **Label alignment.** Hungarian mapping of each system's labels onto a growing
   centroid, so `SPEAKER_01` in one system and `SPEAKER_03` in another become the
   same identity before voting.
2. **Overlap-preserving voting.** Per 10 ms frame, each mapped speaker is
   thresholded **independently** rather than taking a single argmax. This is the
   part that matters here: an argmax vote cannot emit two simultaneous speakers,
   and 7.13% of this corpus is overlapped.
3. Turns shorter than `min_dur = 0.20 s` are dropped.

**Configuration**: `community-1 + reverb-v2 + diarizen-large`, equal weights,
threshold 0.5 (i.e. 2-of-3), frozen on dev before any Step 4 measurement
(`results/fusion_config.json`).

**The fusion operator is GT-free** — it reads only the three systems' RTTMs. Its
*configuration* was not: equal-vs-rank weighting and three-vs-four members were
both selected by **dev** DER. That is legitimate hyperparameter selection on a
split held out from test, but it is not "GT-free", and the distinction matters.

**Result, with significance.** Paired bootstrap over clips, 10,000 resamples:

| comparison | Δ DER | 95% CI | per-clip W/L |
|---|---|---|---|
| `community-1` → fusion | −0.0193 | [−0.0254, −0.0139] | **83 / 16** |
| `pyannote-3.1` → fusion | −0.0195 | [−0.0309, −0.0090] | 80 / 19 |
| `diarizen-large` → fusion | −0.0184 | [−0.0287, −0.0067] | 80 / 19 |
| `reverb-v2` → fusion | −0.0079 | **[−0.0353, +0.0161]** | **42 / 57** |

**The fusion beats three of four constituents decisively (~7% relative, CIs
excluding zero, 80+ of 99 clips) and is statistically indistinguishable from
`reverb-v2` on DER.** The pooled −0.0079 is carried by a few long clips; per
clip it loses 57 to 42, and on the dev half it is 6.8% *worse* than reverb-v2.
An earlier draft reported "−3.1% relative DER over the best single system" as
the Step 4 result; that figure is a point estimate whose interval spans zero,
and it is withdrawn.

What survives, and is testable:

| | `reverb-v2` | **FUSION** |
|---|---|---|
| DER | 0.2521 | 0.2442 *(not significant)* |
| **JER** | 0.3834 | **0.3608** |
| **speaker-count accuracy** | 60.6% | **79.8%** |
| speaker MAE | 0.54 | **0.21** |
| confusion | 0.1121 | **0.0806** |
| false alarm | 0.0681 | **0.0498** |

The fusion is also the **only system that is 1st or 2nd on both halves of the
split**, across 2,000 random 50/49 resamples, 100% of the time (`reverb-v2`
41.6%, the other three 0.0%). Rank stability, not variance reduction — the
observed split spread is no lower than `community-1`'s.

**The real payoff is downstream** (§4): as an ASR front-end the fusion is worth
**−19.6% relative cpWER** and **−44.3% relative WDER**, winning 88 of 99 clips.
That is where Step 4a earns its place, not in DER.

Three other refinement ideas were implemented and **rejected on measurement**:
segment transplantation, boundary padding, and overlap-region deduplication.
Details in §6 and `obs.txt` [32], [35], [50].

---

## 4. ASR benchmarking

**Strategy.** Saaras is run **per-segment**: audio is cut at the diarized turns
and each is transcribed, so attribution is exact by construction. This is not a
preference — Saaras returns exactly one timestamp span per request (verified
across v3/v4, 5 s and 20 s inputs, every mode), so there are no word times to
attribute with. Adjacent same-speaker turns within 1.0 s are merged first,
because 46% of raw `community-1` turns are under a second. Segments below 0.30 s
are skipped; above 29.0 s they are split (a hard server limit).

**Final ASR results, all 99 clips:**

| system | ratio | WER | **cpWER** | DI-cpWER | attribution | WDER |
|---|---|---|---|---|---|---|
| **`saaras-v3@fusion`** | 0.979 | 0.2787 | **0.3181** | 0.2787 | 0.0394 | **0.0628** |
| `saaras-v3@reverb-v2` | 0.960 | 0.2728 | 0.3957 | 0.2728 | 0.1229 | 0.1128 |
| **`saaras-v3@fusion+xlit+num`** (§6) | 0.981 | **0.2585** | **0.3017** | 0.2585 | 0.0432 | **0.0602** |

`ratio` = hypothesis words ÷ reference words. All rows are the full 99 clips.

Whisper and IndicConformer are deliberately **not** in this table. Whisper's
99-clip run used a configuration later shown to carry three independent defects
at once, and quoting it as "Whisper's performance" would be wrong; both are
reported properly in §4.1 from controlled probes. `saaras-v4` covered only 9
clips and is compared against v3 on exactly those 9 in §8.

**The fusion front-end is worth −19.6% relative cpWER and −44.3% relative WDER**
over `reverb-v2`, and cuts the attribution component by two thirds
(0.1229 → 0.0394). Flat WER is slightly *worse* (0.2728 → 0.2787) because the
fusion preserves overlap, so overlapped audio is transcribed once per speaker and
some words appear twice. cpWER is the metric the brief asks for and it moves
decisively the right way. Fusion wins on **88 of 99 clips** (9 losses, 2 ties).

### Why Whisper was not selected

Whisper was benchmarked and rejected on evidence, across three independent
probes:

1. **Configuration is not the problem.** Nine configurations over ten clips —
   greedy vs beam 5, self-LID vs forced LID, turbo vs `large-v3`, long-form vs
   per-segment — span **0.906 to 0.964 WER**. The entire spread is 0.058.
2. **Language identification is not the problem.** Handing Whisper the correct
   language changes WER by at most 0.007 across three paired configurations, and
   in one pair the oracle is *worse*. `large-v3` is identical to four decimals.
3. **It is an Indic capability gap.** AI4Bharat **IndicConformer-600M** on the
   same ten clips, same segmentation, language supplied to both, scores **WER
   0.4158 (CTC)** against Whisper's best 0.9058 — **2.2× better** from a model
   2.6× smaller (600 M against 1.55 B).

Whisper's failure mode is specific and worth recording: on **92 of 99 clips it
produced a different script from the reference**, emitting fluent English
*translation* rather than transcription. And **Whisper has no Oriya** — `or` is
absent from its 99 languages, so 9% of this corpus is outside its label set
entirely. IndicConformer matched all ten scripts including Oriya.

Saaras v3 remains best (0.2876 vs IndicConformer's 0.4158 on the shared ten
clips) — and by more than it appears, since it detects its own language where
IndicConformer had to be given one.

### Strategy for overlapping speech during ASR

The brief asks this directly, so it is stated explicitly.

**What we do: per-speaker segment cutting.** Because the fusion preserves
overlap, an overlapped region belongs to two turns, and the same audio is sent to
Saaras **once per speaker**. Each pass hears the mixture and returns whatever it
can; attribution is then exact by construction. 1,693 s of audio is transcribed
twice this way.

**What we rejected, and why.** Long-form transcription with post-hoc word
assignment ("dominant speaker wins") was measured and is strictly worse here:
attribution cost 0.1229 of cpWER on `reverb-v2` against 0.0394 for per-segment
cutting on the fusion. It also cannot work for Saaras at all, which returns one
timestamp span per request.

**What it costs.** Duplicate transcription inflates insertions — corpus insertion
rate 0.072 before the §5 fix, and per-clip insertion rate correlates with
duplicated-audio *fraction* at r = 0.94 (R² = 0.886). We built a deterministic
deduplicator (delete a shared ≥3-token verbatim run from the longer segment) and
an LLM variant (§6). On the pre-fix baseline dedup helped by −0.0064 cpWER; on
the corrected baseline it **reverses sign to +0.0027 and is not applied**, because
what remains in overlapped regions is mostly genuine simultaneous speech.

**The ceiling.** Even oracle ASR on oracle diarization scores cpWER 0.1242,
exactly 0 on the 9 zero-overlap clips — one transcript per speaker cannot
represent two people talking at once. Target-speaker ASR or source separation
before recognition are the routes past that; both were out of scope here and are
untested.

**Final ASR baseline: `sarvam-saaras-v3 @ DOVER-Lap fusion`, per-segment.**

---

## 5. Pipeline provenance fix — not a model improvement

**This section describes a bug fix, not an algorithmic gain. It is reported
separately for that reason.**

**Symptom.** The Saaras sweep reported 97 of 99 clips. The by-script table summed
to 97, missing one Devanagari and one Gurmukhi clip worth 9,978 reference words.

**Investigation.** The two missing clips were the **longest** (1,822 s,
**476 segments**) and the **fourth-longest** (1,740 s, 113 segments) in the
corpus — both far above median segment count. `transcribe_segments`
fans segments out with `pool.map`, which re-raises the first exception — so a
single non-retryable HTTP 400 anywhere in a clip discarded the entire transcript.
A clip with 476 segments has 476 chances to hit one.

**A second, larger defect surfaced during the fix.** For **27 of 99 clips** the
segmentation actually sent to Saaras could not be reproduced from the fusion on
disk. All 27 reproduced exactly from a **two-system** `community-1 + reverb-v2`
fusion: DiariZen had been absent from the session that materialised the fusion
the ASR then consumed. Independently corroborated — that session's
`step2_metrics.csv` has 297 rows over three models with no DiariZen, against 396
over four locally.

So DER described the 3-system fusion while cpWER described a fusion that was
2-system for 27% of the corpus.

**Root cause.** `settings_key` recorded *decoding* settings so stale work was
visibly stale, but recorded nothing about *segmentation*. A checkpoint built from
a different diarization was indistinguishable from a current one, and `is_done()`
skipped all 27.

**Fix.** `segmentation_key()` fingerprints the turns a transcript was cut on. The
key is **re-derived from the stored segment spans** at audit time rather than
read from the payload, so checkpoints written before the fix are auditable too;
`run_segmented` skips a clip only when that re-derived key matches the
segmentation it would use, so drift self-heals. (New payloads written by
`run_segmented` do not yet persist the key — the audit does not need them to.)
`audit_segmentation()` reports the re-run set without transcribing. Per-segment
failures are isolated so one bad segment cannot discard a clip, and clip failures
persist to `logs/step3_failures.jsonl` (in the working tree; not committed —
the run completed with no failures, so the file is empty).

**Effect of re-transcribing the 27 clips against the correct fusion:**

| | ratio | WER | cpWER | WDER | insertions |
|---|---|---|---|---|---|
| before | 1.0105 | 0.3066 | 0.3381 | 0.0788 | 8,959 |
| **after** | 0.9786 | 0.2787 | **0.3181** | 0.0628 | **5,261** |
| Δ | −0.032 | −0.028 | **−0.0200 (−5.9%)** | −20% rel | **−41%** |

For scale: the GT-perfect ceiling for overlap deduplication was −0.023 cpWER. **A
provenance fix captured almost the entire prize an LLM refinement layer was being
designed to chase** — with no model, no API and no tuning.

It also removed that opportunity. Duplicated tokens in overlapping pairs fell
66% (4,264 → 1,432), and the free deterministic dedup rule *reversed sign*: it
helped by −0.0064 cpWER before the fix and **hurts by +0.0027 after**. What
remains in overlapping regions is mostly genuine simultaneous speech.

---

## 6. Step 4 refinement — Gemini contextual ASR refinement

**Hypothesis.** With diarization fixed, use conversational context to correct ASR
text — the brief's suggested direction.

**Architecture.** `sarvam_diar/llm_refine.py`.

- The model sees a speaker-attributed window with timestamps and returns
  **only `{id, text}`**. Speakers and timestamps are never in the response
  schema, so changing them is **structurally impossible**, not merely forbidden.
  They are copied from the source payload.
- **Windowing as executed: one window per clip** (largest = 136 segments). A
  pair-atomic scheme was designed and implemented — 12-segment cores extended to
  keep time-overlapping pairs together (cap 24), 8 segments of read-only context
  each side, which puts 97.5% of overlapping pairs in one core against 14.5% for
  naive blocks — but it was **not the configuration that ran**. The Gemini free
  tier caps requests at 2/day/model, so the pilot was forced to 10 calls per arm
  (one window per clip) instead of 37. With one window per clip the no-double-edit
  guarantee is trivially satisfied and the read-only context is the whole clip.
- Overlap is computed deterministically in Python and marked in-band; the model
  never infers it from timestamps.
- Temperature 0, seed fixed, responses SHA-1 cached on
  `(model, config, system instruction, rendered window)` so any prompt change
  invalidates automatically.
- **Guards**, all structural: reject a window whose returned id set differs;
  revert a segment on dominant-script change, on a Latin-fraction jump > 0.10, on
  growth beyond 1.05×, or when unlicensed edits exceed
  `max(1, ⌈0.25·len⌉)` — where "licensed" means inside a ≥3-token run verifiably
  shared with an overlapping segment. Revert both members of a pair that both
  deleted the same shared run. Drop a window if >25% of its segments trip a guard.
- Every edit and revert was logged to `logs/step5_edits.jsonl`. **That file is
  not committed**: it lives in the gitignored working tree, and the Gemini
  pilot's records were later overwritten in place by the Sonnet re-run of §6.4.
  The surviving record of this run is the aggregate in
  `results/step5_llm_refine.md`; the qualitative examples below were taken from
  the log while it existed and cannot now be re-derived. That is a provenance
  failure of exactly the kind §5 is about, and it is the reason the Step 4b
  tables are committed rather than left in a working directory.

**Ground truth never enters refinement or the prompt.** The only `reference`
function imported is `normalize_text`, a pure string function applied identically
to both sides.

**Two arms**, differing only in the instruction block: **A** = word-error and
stutter/hallucination correction; **B** = A plus overlap-duplication removal.

**Development pilot: 10 clips, selected GT-blind.** Selection uses only
hypothesis- and diarization-derived features — dominant script *of the
hypothesis*, duration, segment count, predicted overlap fraction from segment
spans, predicted duplication from hypothesis-vs-hypothesis n-gram matching,
speaker count — stratified by script with a fixed seed. No WER, cpWER, WDER or
reference overlap was used, because selecting clips by measured error is ground
truth leaking into experimental design. The set is drawn from the **dev half
only**: 9 scripts, 41.9 min, 402 segments, predicted overlap 0.0%–12.9%, three
zero-overlap control clips arising by construction (`6ZeRgvDHwcI__6_100`,
`7L4gi7Ncc0s__90_148`, `Iare1Emeueg__16_69`).

**Result** (`gemini-3.5-flash`, temperature 0):

| system | ratio | WER | cpWER | WDER | ins | ΔcpWER |
|---|---|---|---|---|---|---|
| baseline | 0.9734 | 0.2357 | **0.2498** | 0.0346 | 223 | — |
| arm A | 0.9722 | 0.2369 | **0.2493** | 0.0334 | 223 | −0.0006 |
| arm B | 0.9700 | 0.2371 | **0.2498** | 0.0335 | 217 | +0.0000 |

| | arm A | arm B |
|---|---|---|
| mean Δ cpWER | −0.0014 | +0.0013 |
| 95% CI (paired bootstrap over clips) | **[−0.0054, +0.0014]** | [−0.0008, +0.0038] |
| per-clip wins / losses | 2 / 3 | 2 / 3 |
| segments changed / reverted | 10 / 4 | 14 / 5 |
| **window fallbacks** | 0 | **3** |
| clips with no edit at all | 5 | 4 |

**Arm A's aggregate is one clip.** Five of ten clips received no edit; the mean
Δ of −0.0014 is carried almost entirely by `7L4gi7Ncc0s__90_148` (−0.0167), a
4-segment, 120-word clip. That is another reason the CI spans zero.

**Arm B is effectively n = 7.** With one window per clip, a failed call discards
the whole clip, and three arm-B clips (`HZv_WvIr6lE`, `Iare1Emeueg`,
`Ig6szCD8m20`) hit HTTP 429 after all retries. They enter the statistics as forced
zero-deltas, which biases arm B's mean toward zero and narrows its CI. Arm A had
no fallbacks.

**Rejected.** All four pre-registered abandon criteria fired: CI includes zero,
20% win rate against a 60% bar, insertions did not fall, 29% revert rate against
a 20% bar.

**The held-out 49-clip test split was never scored for this stage.** The pilot
was dev-only and nothing cleared its bar, so there was nothing frozen worth
spending the single test evaluation on.

### Failure analysis

The architecture held perfectly: **all 20 refined payloads have a segmentation
byte-identical to source** — same `segmentation_key`, speakers, timestamps, order
and count. The pre-write assertion never fired.

The *guard* was the wrong shape. **7 of 24 applied edits (29%) introduced a
script absent from the original**, and all 7 passed the dominant-script rail
because one corrupted token in forty cannot move a majority:

```
Telugu   పార్టీ  →  パーティー   (Japanese katakana)
Bengali  এটাও   →  것도        (Korean hangul)
Marathi  हे     →  হে          (Bengali)
```

Single-token cross-script lexical substitution. This is why WER got slightly
*worse* (0.2357 → 0.2369) while cpWER barely moved: sparse token-level damage,
never an improvement. The correct rail is **per-token** — reject any output token
whose script is absent from the input. It was deliberately not applied, because
the guards were frozen before the run and changing them after seeing results
would invalidate the pilot.

The model did do something genuinely useful. On `7L4gi7Ncc0s__90_148` it restored
a truncated proper noun from world knowledge — Saaras produced `എച്ച് എസ് പ്ര`
and the model completed it to `എച്ച് എസ് പ്രണോയിക്കും, ഗോൾഡൻ ഗ്ലോബ് റേസിൽ`,
both of which are in the reference. On the same clip it also deleted
`കായിക കായിക`, a repetition the speaker genuinely made. One right, one wrong.

A single manual cross-check, **not logged in the repository and therefore
anecdotal**: the same prompt pasted into ChatGPT returned all 4 segments of
`7L4gi7Ncc0s__90_148` unchanged, taking the "when unsure, preserve" branch
everywhere. n = 1 clip, so it supports no comparison between models — it is
reported only because "edit nothing" is a legitimate strategy that scores exactly
0.0000 with zero variance, which on this evidence is not clearly worse than
editing.

### Why the null was predictable

Arm B was expected to fail: §5 had already removed 66% of the duplication, and
the free dedup rule had reversed sign. Arm A targets word error, but a text-only
refiner cannot hear the audio — and Saaras's residual errors on this corpus are
code-switched surface forms and ambiguous short turns, not linguistic
implausibilities the model could catch.

---

## 6.1 Step 4b — script and numeral normalisation *(the stage that worked)*

**Finding.** The reference writes code-switched English **phonetically in the
native script** — the dual-form convention behind its 20,599 parenthesised
glosses. Measured: the reference is 99.98% Indic script (27 Latin tokens in
123,896); the Saaras hypothesis is 3.0% Latin (3,689 tokens). Of 21,357
substitutions, **3,481 (16.3%) are reference-Indic against hypothesis-Latin**:

```
ఐ → i     టు → to    ఓకే → okay   సో → so
એમએલ → ml  एंड → and  યુ → you     బోత్ → both
```

These are **not recognition errors**. Saaras heard the word and wrote it in the
wrong script for this corpus's convention. A second, smaller mismatch of the
same kind: the corpus writes numbers as spoken words (14 digit-bearing tokens in
123,896) while Saaras writes digits (892).

**Method.** Extract the Latin and digit **vocabulary from the hypothesis only** —
1,651 unique (script, token) pairs and 382 numerals — have `claude-sonnet-4-6`
render each in the clip's language, and apply the result as a **frozen lookup
table**. One batch per script, cached by prompt hash. No per-clip inference, no
re-run of the recogniser.

**Why this is GT-free.** The target script comes from `data.dominant_script` on
the *hypothesis* text; the spellings come from a general model's knowledge of
the language. Deriving the mapping from reference pairs would be a leak and
would inflate the result. (An independent audit confirmed this by rebuilding
every cache key from the hypothesis vocabulary: 32/32 and 11/11 keys hit, while
building from the wrong source gives 5 hits / 6 misses — so the prompts provably
contained the hypothesis vocabulary and nothing else.)

**Result, all 99 clips:**

| stage | ratio | WER | cpWER | WDER |
|---|---|---|---|---|
| `saaras-v3@fusion` | 0.9786 | 0.2787 | 0.3181 | 0.0628 |
| + transliteration | 0.9786 | 0.2614 | 0.3045 | 0.0603 |
| **+ numerals** | 0.9807 | **0.2585** | **0.3017** | **0.0602** |

**WER −7.2% relative, cpWER −5.2%, WDER −4.1%.** Substitutions fall 21,357 →
18,671.

| split | pooled Δ cpWER | 95% CI | better / worse / tie |
|---|---|---|---|
| all | −0.0164 | [−0.0276, −0.0080] | 72 / 1 / 26 |
| dev | −0.0094 | [−0.0147, −0.0056] | **34 / 0 / 16** |
| test | −0.0216 | [−0.0405, −0.0072] | 38 / 1 / 10 |

The intervals above are paired bootstraps of the **pooled** delta over 10,000
clip resamples, matching the pooled point estimate beside them. Resampling the
**per-clip mean** instead gives −0.0131, CI [−0.0188, −0.0084] — a different
estimand with a narrower interval, reported in `obs.txt` [54]. Both exclude zero.

**This stage is transductive, and the dev/test split does not test it.**
`build_table` draws its vocabulary from the hypotheses of **all 99 clips**,
including test. No ground truth is read — the leak audit above is unaffected —
but the table's *coverage* was determined with the test clips' own transcripts in
view, so both halves are in-vocabulary by construction and the split above cannot
detect a generalisation failure. That has to be stated, because the table is
otherwise easy to read as a held-out result.

The inductive counterfactual: restrict the shipped table to keys observed in
**dev** hypotheses only, then apply it everywhere.

| split | baseline | shipped (99-clip vocab) | dev-only vocab |
|---|---|---|---|
| dev | 0.2676 | 0.2582 (−0.0094) | 0.2582 (−0.0094) |
| test | 0.3552 | 0.3337 (**−0.0216**) | 0.3511 (**−0.0042**) |
| all | 0.3181 | 0.3017 (−0.0164) | 0.3118 (−0.0064) |

**About 80% of the measured test gain depends on the table having seen test
vocabulary.** The dev column is unchanged to four decimals, which is itself the
tell: dev is in-vocabulary either way, so no dev-side diagnostic could have
caught this.

The cause is coverage, not modelling. Latin code-switching is extremely
concentrated in this corpus — dev holds 403 Latin types / 582 tokens against
test's 1,365 types / 3,107 tokens, and only **8.6% of test types (18.9% of test
tokens) appear anywhere in dev**. A dev-built table simply has no entry for most
test words. The LLM itself generalises fine: it renders any word handed to it,
and never saw the split. What does not transfer is the *word list*.

Which number is the honest headline depends on the deployment, so both are
reported. As a batch post-processor over a corpus you already hold — the setting
here, and the normal one for offline transcription — you build the table from
that corpus and get **−0.0164**. If new audio must be transcribed against a
frozen table, the right expectation is **−0.0064**, and the fix is to extend the
table at inference time, which costs one LLM call per unseen word type and needs
no ground truth. The stage's other properties are unaffected either way: it can
only rewrite tokens already emitted in Latin or as digits, so its downside stays
bounded at zero.

All three CIs exclude zero; sign test p = 1.6 × 10⁻²⁰. **Not one clip regressed on
dev** — by construction, since the stage only rewrites tokens the recogniser
already emitted in Latin or as digits, so a wrong rendering leaves a
substitution that was already a substitution. The downside is bounded at zero.

**Not a scorer artefact.** Replacing 3,520 Latin tokens changes the hypothesis
token count by **+2** (only 3 of 1,651 table values are multi-word), so the gain
cannot come from token-count manipulation. The numeral stage does add tokens
(+260 on 697 replacements) and pays for it: insertions +81 against substitutions
−254. It costs, it does not flatter.

**What was tested on the residual and rejected.** 46% of remaining substitutions
are within two characters of the reference, but only 423 (2.2%) vanish under
orthographic folding — nukta (`ज़्यादा`/`ज्यादा`) and chandrabindu/anusvara
(`हूँ`/`हूं`) — worth 0.34% of tokens, and that is a *scoring* normalisation
applied to both sides rather than a system change. The rest are genuinely
different words (`તે`/`એ` is "that" vs "this"). Word-boundary convention
accounts for a further 271 cases (0.22%), strikingly one-directional (271
join-cases against 0 split-cases), but the suffix inventory would have to be
learned from the reference. Both rejected.

---

## 7. Final system and results

**`sarvam-saaras-v3 @ DOVER-Lap(community-1, reverb-v2, diarizen-large)`,
per-segment, 1.0 s same-speaker merge.**

### Diarization — all 99 clips, raw GT, collar 0, overlap scored

| | DER | miss | FA | conf | JER | spk acc |
|---|---|---|---|---|---|---|
| `reverb-v2` (best single by DER) | 0.2521 | 0.0719 | 0.0681 | 0.1121 | 0.3834 | 60.6% |
| mean of the four singles | 0.2604 | — | — | — | 0.3762 | 70.9% |
| **final (FUSION)** | **0.2442** | 0.1138 | 0.0498 | 0.0806 | **0.3608** | **79.8%** |

**DER against `reverb-v2` is a statistical tie** (Δ −0.0079, CI [−0.0353,
+0.0161], 42/57 per clip). Against the other three constituents the fusion wins
by ~7% relative with CIs excluding zero and 80+/99 clips. The defensible
diarization claims are **JER −5.9%**, **speaker-count accuracy +19.2 points**,
**confusion −28%**, and rank stability across resampled splits.

### ASR — all 99 clips

| | ratio | WER | cpWER | WDER |
|---|---|---|---|---|
| baseline (`saaras-v3@reverb-v2`) | 0.960 | 0.2728 | 0.3957 | 0.1128 |
| + Step 4a fusion front-end | 0.979 | 0.2787 | 0.3181 | 0.0628 |
| **+ Step 4b normalisation (final)** | 0.981 | **0.2585** | **0.3017** | **0.0602** |
| **total improvement** | — | **−5.2%** | **−23.8%** | **−46.6%** |

Step 4a contributes −19.6% relative cpWER (88/99 clips), Step 4b a further
−5.2% (72/99, CIs excluding zero on both halves — though see §6.1: Step 4b is
**transductive**, so its dev/test split is not a generalisation test, and the
inductive estimate is −0.0064 rather than −0.0164). Flat WER dips at 4a — the
fusion preserves overlap, so overlapped audio is transcribed once per speaker
and some words appear twice — and 4b more than recovers it.

### Split-wise, final system

| split | clips | ratio | WER | cpWER | WDER |
|---|---|---|---|---|---|
| dev | 50 | 0.988 | 0.2219 | 0.2582 | 0.0423 |
| test | 49 | 0.975 | 0.2854 | 0.3337 | 0.0736 |
| all | 99 | 0.981 | 0.2585 | 0.3017 | 0.0602 |

**Dev and test are not equally hard** (final-system cpWER 0.2582 vs 0.3337; the
gap is the same before Step 4b, 0.2676 vs 0.3552). Only deltas
transfer across the split; a dev absolute is not a prediction for test. The split
was frozen before any Step 4 tuning (`results/split.json`), stratified by script
and speaker-count band — both reference-derived, so the dev/test *boundary* is
not itself GT-blind, though no per-clip metric influenced it.

### Against the achievable floor

Feeding the reference transcript back as if recognised, attributed against the
reference diarization, still scores **cpWER 0.1242 / WDER 0.0349** — and exactly
0 on the 9 zero-overlap clips. The final system is **0.1775 above that floor**,
not 0.3017 above zero.

Two components make up the floor, and only one is physical: 0.0424 is
attribution ("one transcript cannot carry two simultaneous speakers"), and the
remaining 0.0818 is flat-stream word **ordering** under overlap, an artefact of
approximating reference word times by even spread within a turn. The second
component would shrink with real word alignments, so the floor is an upper bound
on what is unreachable and it flatters the final system slightly.

**14.3% of remaining cpWER is attribution** (0.0432 of 0.3017); the other 85.7%
is word error.

That gap *widened* after Step 4b, from 0.0394 to 0.0432, and the reason is worth
stating because it looks like a regression and is not one. Step 4b cut DI-cpWER
by 0.0202 but cpWER by only 0.0164, and `cpWER − DI-cpWER` is the cost that
speaker-attributed scoring adds over speaker-agnostic scoring — not a measure of
attribution quality on its own. A word that was both misrecognised and
misattributed used to score wrong under both metrics. Once the word is fixed it
scores right under DI-cpWER and still wrong under cpWER, so correcting words
**unmasks** attribution errors that word errors had been hiding. The direct
attribution metric moved the other way over the same change: WDER 0.0628 →
0.0602. Both readings are in `results/step3_metrics.csv`.

**Per-video tables.** `results/step3_metrics.csv` — **539 rows**, `system × clip`,
seven ASR systems including the final one (pooling reproduces cpWER 0.3017).
`results/step2_metrics.csv` — **495 rows**, `model × clip`, all five diarization
systems **including the fusion** (pooling reproduces DER 0.2442, JER 0.3608).

Both tables are committed and pool exactly to the corpus figures above.
`checkpoints/` additionally carries every diarization RTTM and ASR transcript, so
a reviewer can re-derive every number in this report **without a GPU, without
HF-gated model access, and without any API key** — see `checkpoints/README.md`.

**How to re-derive them.** `main.ipynb` runs the whole pipeline from those
checkpoints and prints the tables in this section. It has a `SUBSET = 10` toggle
so the pipeline can be checked end to end in a few minutes before the full
corpus. It was verified from a clean clone against a virtualenv holding only
pandas, numpy, scipy, rapidfuzz, pyannote.metrics and jinja2, in both modes; the
full run reproduces every figure above, and the fusion regenerates bit-for-bit
from its three constituents on 99/99 clips. The one network call is a first-run
fetch of the assignment's `youtube_segments.csv`, which is not redistributed
here.

---

## 8. Failure analysis and engineering lessons

**Overlap is the dominant diarization weakness.** A frame-level decomposition
against GT:

| region | ref sec | miss | FA | confusion | share of all error |
|---|---|---|---|---|---|
| true silence | 0 | 0 | 1,372 | 0 | 12.6% |
| single speaker | 38,213 | 2,407 | 841 | 3,283 | 60.0% |
| **overlap** | 6,356 | **2,664** | 5 | 310 | **27.4%** |

**52.5% of all miss is in overlapped regions**, and fusion overlap recall is only
**21.5%** (678 s of 3,148 s found). It marks 1,560 s as ≥2 speakers, so 882 s of
the overlap it does claim is in the wrong place.

**Boundary tightness is a substantial effect.** 39.4% of missed frames lie within
200 ms of a hypothesis boundary against 14.5% of *all* frames — an enrichment of
**2.71×** (2.99× at 100 ms, 2.14× at 500 ms), over 4.4 M frames and 9,910 fusion
turns. The control matters: without it the raw 39.4% means nothing, since
boundaries are dense. An earlier draft reported 1.36× from a histogram that had
silently capped its denominator at 800 ms; that was wrong and the corrected
figure roughly doubles the estimated importance of boundary precision.

**False alarm is mostly spill, not invention.** 1,372 s of FA in true reference
silence across 3,427 runs, **median run 0.20 s**; only 7.9% is in runs over 5 s.

**The transcript cannot see speaker errors — the assignment's hint does not hold
on this corpus.** Over the 6,312 segments of the final system, testing shallow
discourse signals against whether a segment's speaker label is wrong (base rate
**33.1%**, Hungarian-mapped):

| signal | n | wrong | lift |
|---|---|---|---|
| duration < 0.5 s | 949 | 68.3% | **2.06×** |
| word count 0 | 599 | 68.6% | **2.07×** |
| gap to previous < 0.05 s | 2,952 | 41.9% | 1.27× |
| gap to previous ≥ 1.5 s | 1,084 | 26.0% | 0.79× |
| **speaker changed at boundary** | 5,296 | 34.0% | **1.03× (none)** |
| **word repeats across boundary** | **36** | 36.1% | **1.09× (none)** |

"Repeated text across a speaker boundary suggesting a false split" occurs **36
times in 6,312 segments** and does not predict error. Worse, the signals that
*do* work are duration proxies and they point at the wrong targets:

| duration | segments | error rate | share of wrong-label *time* |
|---|---|---|---|
| < 2 s | 2,713 | 56.0% | 30.1% |
| 2–5 s | 1,416 | 30.2% | 33.9% |
| ≥ 5 s | 2,109 | **5.6%** | **36.0%** |

A detector keyed on short segments finds errors at twice the base rate but
reaches only 30% of the damage; the 36% sitting in long segments has a 5.6% error
rate, so flagging it would be 94% false positives.

**ASR error is overwhelmingly recognition, not attribution** — 87.6% of remaining
cpWER. By script, on the **final system** (`saaras-v3@fusion+xlit+num`):

| script | clips | WER | cpWER | WDER |
|---|---|---|---|---|
| Telugu | 12 | 0.3521 | **0.4103** | 0.1122 |
| Malayalam | 7 | 0.3299 | 0.3877 | 0.0732 |
| Oriya | 9 | 0.3448 | 0.3852 | 0.0534 |
| Bengali | 8 | 0.2455 | 0.3746 | **0.1233** |
| Gujarati | 12 | 0.2901 | 0.3287 | 0.0672 |
| Kannada | 9 | 0.2276 | 0.3183 | 0.0697 |
| Gurmukhi | 7 | 0.2704 | 0.3169 | 0.0548 |
| Tamil | 10 | 0.2458 | 0.2710 | 0.0282 |
| Devanagari | 25 | 0.1885 | **0.1985** | 0.0292 |

**Telugu is 2.07× Devanagari on cpWER.** Language is a far larger axis of
variation than anything else measured. WER and cpWER also rank scripts
differently: **Bengali has the third-best WER (0.2455) but the worst WDER
(0.1233)** and only the fourth-worst cpWER — its words are recognised well and
its speakers attributed badly. Group sizes are 7–25 clips, so read the ordering
as indicative.

**Ground-truth quality is a measurement floor.** 39 of 99 clips have reference
timestamps displaced 1–5 s. **11 of the 15 worst-DER clips (fusion) are in the
39-clip flag set**, and the tail skews short — 9 of the 15 are under 90 s, where a
4 s shift is catastrophic — though the range runs 53 s to 898 s. Unannotated GT gaps account for 218 s, which is 8.1% of `community-1`'s false
alarm (2,683 s) — the system those gaps were measured on.

### Engineering lessons

1. **Provenance outranks modelling.** The single largest ASR improvement in this
   project (−5.9% cpWER) was a bug fix, not a model. `settings_key` existed
   precisely to catch stale work and still missed this, because it fingerprinted
   decoding and not segmentation. "Is the input what I think it is" should be
   checked before "is the model good enough".
2. **Always measure the control, and check the control's own arithmetic.**
   The boundary-miss enrichment was first computed
   as 1.36× from a histogram whose denominator had been silently truncated; the
   correct value is 2.71×, and a sanity check — 19,820 boundaries × 0.4 s over
   44,130 s caps the null at 18% — would have caught it immediately. Order-of-
   magnitude checks on your own statistics are cheap.
3. **Aggregates hide inverted rankings.** GT correction moved fusion DER by
   **19.8%** and reversed which single model is best. Dev and test invert
   rankings too — `reverb-v2` scores 0.2238 on dev and 0.2738 on test. Any claim
   resting on a 0.01 DER gap is noise, **including one of our own**: the −0.0079
   fusion-vs-`reverb-v2` gap was reported as a result before its interval was
   computed, and it spans zero.
4. **Structural impossibility beats validation.** Omitting speakers and
   timestamps from the LLM's response schema removed a whole class of failure
   that no amount of prompt instruction or post-hoc checking would have
   guaranteed.

---

## 9. Limitations and future work

Only proposals with measured support are listed.

**Limitations.**
- 99 clips is small; 0.01-level DER differences are not resolvable, and the
  Step 5 pilot at 402 segments cannot resolve an effect below ~0.005 cpWER.
- The GT-correction analysis is a diagnostic: 39 clips flagged, 23 auto-accepted,
  the rest held for manual review. Headline numbers use raw, unmodified GT.
- Whisper's 99-clip row is a known-broken configuration; its fixed configuration
  was only evaluated on 10 clips. IndicConformer likewise (10 clips, and with an
  oracle language, since it has no LID of its own).
- Saaras v4 covers 9 clips; v3-vs-v4 is genuinely unsettled.
- The Step 5 pilot used `gemini-3.5-flash` and whole-clip windows because the API
  free tier capped requests at 2/day/model for the intended model, forcing a
  departure from the designed 12-segment windows.

**Future work, evidence-backed.**
1. **Overlap remains the largest opportunity and is currently unaddressed.**
   Overlapped regions carry 52.5% of miss and 27.4% of all DER error, and
   overlapped words are deleted at 17.9% against 4.7% for clean speech — a
   perfect separator would be worth **−0.031 WER** (0.2787 → 0.2479), the
   largest ceiling measured here.

   One intervention was built and measured end-to-end: **source separation of
   overlapped regions**. On 100 GT-located overlap runs from the 10
   most-overlapped clips, word recovery went **52.6% → 50.0%** — it hurts.

   | condition | word recovery |
   |---|---|
   | mixture @ 16 kHz | 52.6% |
   | mixture 16k→8k→16k, no separation | 52.0% |
   | ConvTasNet-Libri2Mix, 2 streams | **50.0%** |

   **Scope this result carefully.** The bandwidth control shows the *recogniser*
   tolerates 8 kHz audio (0.6 points), so the 2.0-point loss is not that. But
   condition C also changes the *separator's* operating bandwidth, and mask
   estimation depends on cues above 4 kHz that an 8 kHz model never sees — the
   control cannot decompose those. And the overlap regions were located from the
   reference, which no shipped system can do (fusion overlap recall is 21.5%).
   So what is established is narrow: **an out-of-domain 8 kHz separator, given
   oracle overlap regions, does not help.** Whether a 16 kHz in-domain separator
   would is **untested**, and is the next thing to try — not a settled question.
2. **Per-token script rail for any future LLM stage.** Would have caught all 7
   corrupting edits in §6; two lines.
3. **IndicConformer at corpus scale, leak-free.** It beat Whisper 2.2× on 10
   clips and covers Oriya, which Whisper structurally cannot. A `--lang lid` run
   over 99 clips would turn a probe into a benchmark.

**Explicitly not claimed.** Boundary padding, segment transplantation, overlap
deduplication, source separation and LLM refinement were all implemented and
measured, and none of them improved the system. They are reported as negative
results. The MSDD-style verifier and the OSD∩constituent intersection were
**not** implemented — they were dropped after a candidate-precision audit — and
are reported as audits, not experiments.

---

## 10. References

**Diarization**
- Raj, Garcia, Huang, Watanabe, Povey, Stolcke, Khudanpur. *DOVER-Lap: A Method
  for Combining Overlap-aware Diarization Outputs.* SLT 2021.
- Bredin. *pyannote.audio 2.1 speaker diarization pipeline.* Interspeech 2023.
  Models: `pyannote/speaker-diarization-community-1`,
  `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`.
- Han, Landini, Rohdin, et al. *DiariZen / Leveraging Self-Supervised Learning
  for Speaker Diarization.* ICASSP 2025. Model: `BUT-FIT/diarizen-wavlm-large-s80-md`.
- Rev.ai. *Reverb Diarization v2.*
- Bredin. *pyannote.metrics: a toolkit for reproducible evaluation of speaker
  diarization systems.* Interspeech 2017.

**ASR and metrics**
- Sarvam AI. *Saaras v3 / v2.5 speech-to-text-translate.* `api.sarvam.ai`.
- Radford, Kim, Xu, Brockman, McLeavey, Sutskever. *Robust Speech Recognition via
  Large-Scale Weak Supervision (Whisper).* ICML 2023. Via `faster-whisper` /
  CTranslate2.
- AI4Bharat. *IndicConformer-600M-Multilingual.*
- Watanabe et al. *CHiME-6 Challenge: Tackling Multispeaker Speech Recognition
  for Unsegmented Recordings.* CHiME 2020. (cpWER)
- El Shafey, Soltau, Shafran. *Joint Speech Recognition and Speaker Diarization
  via Sequence Transduction.* Interspeech 2019. (WDER)

**Step 4 refinement**
- Google DeepMind. *Gemini 3.5 Flash*, Gemini API, structured output with
  `responseSchema`, temperature 0.

**Methods**
- Kuhn. *The Hungarian Method for the Assignment Problem.* 1955. (via
  `scipy.optimize.linear_sum_assignment`)
- Breiman, Friedman, Olshen, Stone. *Classification and Regression Trees.* 1984.
  (one-standard-error rule, used in operating-point selection)
