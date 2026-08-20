# Ground-truth alignment QC

Some clips' annotations sit later in time than the speech they describe. This
stage finds them, ranks the evidence, and produces a shortlist for human
verification. **It never edits the raw annotations.** Confirmed offsets live in
`corrections.json` and are applied to a copy at scoring time.

Regenerate with:

```bash
python3 tools/gt_alignment_qc.py
```

## Method

1. **Consensus speech activity.** Every completed diarizer is rasterised to
   10 ms and a frame counts as speech when **at least 2 of N** models say so.
   Speaker identity is discarded -- a global time shift has nothing to do with
   who is speaking. The consensus is built from **raw model outputs only**; the
   fused system is deliberately excluded, since the fusion is one of the things
   this QC exists to keep honest.
2. **Lag search.** Intersection-over-union between GT activity and consensus
   activity at every lag in ±5 s. IoU rather than raw correlation because it is
   bounded and legible: "the two agree on 96% of the frames either calls
   speech". **DER and JER are never consulted**, so a later DER improvement is
   evidence rather than a tautology.
3. **Evidence.** Best lag, IoU at lag 0, IoU at the peak, the gain between
   them, the peak's margin over everything more than 0.5 s away, and the spread
   of the per-model lags.
4. **Independent corroboration.** A frame-energy VAD computed straight from the
   waveform. It shares no code, no training data and no assumptions with any
   diarizer, so its agreement is a genuine second opinion rather than the same
   system counted twice.
5. **Flagging.** A clip is flagged only when the lag is at least 1.0 s, the IoU
   gain at least 0.05, the peak margin at least 0.02, and at least two models
   contributed.

## What was found

| | |
|---|---|
| clips assessed | 99 |
| flagged as strong candidates | 27 |
| corroborated by the independent energy VAD | 26/27 |
| auto-accepted (VAD agrees **and** per-model spread ≤ 1 s) | 18 |
| held for manual review | 9 |
| offsets observed | +1.0 s to +4.6 s |

Every flagged offset is **positive**: the annotation runs later than the speech.
Not one clip needs shifting the other way.

## Before and after

Raw ground truth against QC-adjusted, applying only the 18 auto-accepted
corrections. Same audio, same model outputs, same scoring code -- only the
reference timestamps differ.

| model | DER raw | DER QC | Δ | JER raw | JER QC |
|---|---|---|---|---|---|
| diarizen-large | 0.2625 | **0.2239** | −14.7% | 0.366 | 0.3065 |
| community-1 | 0.2634 | **0.2343** | −11.0% | 0.3768 | 0.3336 |
| pyannote-3.1 | 0.2637 | **0.2363** | −10.4% | 0.3785 | 0.3373 |
| reverb-v2 | 0.2521 | **0.2454** | −2.7% | 0.3834 | 0.3692 |

The ranking changes, which is the point. On raw GT the four models sit within
0.011 DER and `reverb-v2` leads; QC-adjusted, `diarizen-large` leads by 0.010
and `reverb-v2` is last. Loose, long turns are robust to a displaced reference,
so the least precise model looked best precisely because the reference was
wrong.

`reverb-v2` gains least (−2.7%) and `diarizen-large` most (−14.7%) for the same
reason.

## Files

| file | what it is |
|---|---|
| `candidates.csv` | all 99 clips, ranked by evidence |
| `shortlist.csv` | the 27 flagged, with a proposed offset and blank verification columns |
| `corrections.json` | versioned manifest; only `verified` rows are applied |
| `diagnostics/*.svg` | per flagged clip: GT, GT shifted, consensus, energy VAD, and the IoU-vs-lag curve |
| `metrics_raw_vs_qc.csv` | the benchmark scored both ways |

## Reading a diagnostic

Four activity bands over the clip timeline, then the IoU-vs-lag curve beneath.
If the candidate is real, **GT raw** is visibly displaced from **consensus** and
**energy VAD**, **GT shifted** lines up with both, and the curve shows one sharp
peak away from zero. A flat or multi-peaked curve means the clip needs a human.

## Rules this stage keeps

* **The raw ground truth is immutable.** No file under `data/` or `reference/`
  is written. A correction is a manifest row applied to a copy.
* **No ground truth reaches the pipeline.** Detection is scoring-side; the
  diarizers and the fusion never see a lag or a correction.
* **QC-adjusted numbers are a diagnostic, never the headline.** The brief fixes
  the scoring protocol, so results against raw GT remain the reported result.
  The QC column exists to show how much of the measured error is annotation.
* **`verified` does not mean a human looked.** The 18 auto-accepted rows
  are corroborated by two independent signals and are marked as such in
  `verified_by`; they are still pending a human spot-check. The 9 held
  rows carry the reason they were held.

## Held for manual review
* `7L4gi7Ncc0s__90_148` +3.0s — needs manual review: per-model lags spread 3.02s
* `OXUt5KnlUMo__91_181` +3.0s — needs manual review: per-model lags spread 5.48s
* `0p6cktLGIfY__12_930` +3.0s — needs manual review: per-model lags spread 2.09s
* `JBYGLQJwFDc__103_407` +3.0s — needs manual review: per-model lags spread 2.80s
* `2HGP34TNvjg__84_194` +2.0s — needs manual review: per-model lags spread 2.68s
* `Gq45evw6ytY__597_1464` +2.0s — needs manual review: per-model lags spread 1.49s
* `Kdi-ECuOaKg__2_78` +2.0s — needs manual review: energy VAD disagrees, per-model lags spread 4.10s
* `T3I2T-cfhG4__160_210` +1.5s — needs manual review: per-model lags spread 1.33s
* `0VEwL9XZ0LY__261_557` +1.0s — needs manual review: per-model lags spread 1.19s

## Is this a leak? The honest answer

**No correction reaches the pipeline.** The diarizers ran before this stage
existed, the fusion never sees a lag, and corrections modify the *reference* at
scoring time, never a hypothesis. `grep` confirms no pipeline module imports
`gt_qc`. On the brief's own terms -- "the ground truth is never an input to your
pipeline" -- this is clean.

**But there is a real objection, and it should be stated rather than hidden.**
The lag is chosen to maximise agreement between the ground truth and the models
being evaluated. That is moving the target toward the shooters. Using IoU rather
than DER makes it indirect, not innocent: if every model were wrong the same way
-- all calling the same music speech, say -- the "correction" would shift the
reference to match that shared error, and the QC-adjusted numbers would flatter
them.

Four things bound that risk:

* **A model-free second opinion.** The energy VAD is arithmetic on the waveform,
  sharing no code, training data or assumptions with any diarizer. It
  corroborates 26 of 27.
* **A high floor.** Only offsets of at least 1 s are flagged. Four independent
  segmenters being wrong by more than a second in the same direction on the same
  clip is not a plausible coincidence.
* **Agreement is required, not just a peak.** Auto-acceptance also needs the
  per-model lags to agree within 1 s. Where voting alone carried the peak, the
  clip is held for a human -- that is 9 of the 27.
* **Raw stays the headline.** QC-adjusted is reported beside it as a labelled
  diagnostic, never instead of it.

**What actually closes the objection is a human ear**, and that is what the
worksheet is for. "The models disagree with the annotation" is an argument;
"I listened, and the words are not where the annotation says" is evidence.

## Manual verification

```bash
python3 tools/gt_qc_worksheet.py     # -> verification_worksheet.csv
```

The check uses the **transcript**, not speech activity. For one well-isolated
mid-clip utterance the ground truth claims "these words start at t" and the QC
claims "no, at t - offset". Both are YouTube links in the worksheet. Open both;
the words are audible at exactly one. Write `GT` or `QC` in `which_is_right`.

The transcript check is the right one for three reasons: it works on clips that
open with speech, where comparing first onsets is useless because both sides say
0.01 s; it is **lexical**, so it shares nothing with the acoustic models that
proposed the offset, which is precisely the circularity a reviewer would press
on; and it needs no tooling beyond a browser.

**Controls are included and are the part that makes this rigorous.** Six clips
the detector did NOT flag are in the worksheet with offset 0, so both links point
at the same moment and the words should simply be there. Verifying only flagged
clips can confirm what the detector already believes. Controls are what
establish it is not flagging everything -- and a control whose words are absent
is a detector miss worth more than any confirmation.

Ten flagged plus six controls is enough. If all ten flagged read `QC` and all
six controls read `GT`, the detector has perfect precision and recall on a
16-clip audited sample, and the remaining 17 auto-accepted rows inherit that
credibility. If even one control fails, stop and re-examine the thresholds.

Record outcomes in the worksheet, then set `verified` and `verified_by` in
`corrections.json` for the confirmed rows and re-run
`tools/gt_alignment_qc.py`, which rescores using only verified corrections.

