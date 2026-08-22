# Step 2 & 4 — diarization results

99 clips, 12.26 h, nine Indic scripts. Scored with **collar 0.0 and overlap
included**, per the brief. `FUSION` is the Step 4 system.

## Headline — raw ground truth

| system | DER | miss | FA | confusion | JER | spk acc | spk MAE |
|---|---|---|---|---|---|---|---|
| `FUSION` | **0.2442** | 0.1138 | 0.0498 | 0.0806 | 0.3608 | 79.8% | 0.21 |
| `reverb-v2` | **0.2521** | 0.0719 | 0.0681 | 0.1121 | 0.3834 | 60.6% | 0.54 |
| `diarizen-large` | **0.2625** | 0.1143 | 0.0640 | 0.0842 | 0.3660 | 72.7% | 0.30 |
| `community-1` | **0.2634** | 0.1079 | 0.0602 | 0.0954 | 0.3768 | 78.8% | 0.23 |
| `pyannote-3.1` | **0.2637** | 0.1079 | 0.0602 | 0.0957 | 0.3785 | 71.7% | 0.35 |

**Step 4 result.** The fusion reaches DER 0.2442 against `reverb-v2`'s 0.2521,
and is simultaneously best on false alarm, confusion, JER and speaker-count
accuracy. It is beaten only on **miss**, where `reverb-v2` leads — voting cannot
recover speech that a majority never found.

**That DER gap is not statistically significant, and is not claimed.** Paired
bootstrap over clips: Δ −0.0079, 95% CI [−0.0353, +0.0161], and `reverb-v2` wins
57 of 99 clips head-to-head. An earlier draft of this file reported it as "3.1%
relative DER reduction"; that figure is a point estimate whose interval spans
zero and it is **withdrawn** — see `WRITEUP.md` §3. Against the other three
constituents the fusion wins decisively (Δ −0.0193 / −0.0195 / −0.0184, all CIs
excluding zero, 83/16 and 80/19 per clip). The defensible claims here are the
JER, confusion and speaker-count wins, the variance reduction below, and the
downstream ASR result — cpWER 0.3957 → 0.3181, which is significant.

## The four models are not four versions of the same thing

The DER spread across the single systems is 0.0116, which is nothing.
By component they behave completely differently:

* `reverb-v2` owns **detection** — miss 0.0719, a third below the field —
  but pays for it with the worst confusion (0.1121) and
  speaker-count accuracy (61%).
* `diarizen-large` owns **assignment** — confusion 0.0842 — and is the only
  model with essentially no speaker-count bias (+0.02 against −0.17 to −0.35).
* `community-1` and `pyannote-3.1` are near-identical: they share
  `pyannote/segmentation-3.0` and differ by 0.320 s of miss+FA over 12.4 hours.

Ranking by DER alone would report "the models are equivalent" and discard the
only interesting result.

## Overlap

Reference has **52.5 minutes** of overlapped speech — 7.13% of time but
**12.8% of reference words**, since overlapped speech is denser.

| system | predicted overlap | % of reference |
|---|---|---|
| `FUSION` | 26.0 min | 50% |
| `reverb-v2` | 8.0 min | 15% |
| `diarizen-large` | 44.2 min | 84% |
| `community-1` | 36.9 min | 70% |
| `pyannote-3.1` | 36.9 min | 70% |

Note the inversion: `reverb-v2` predicts **15%** of the reference's overlap and
still has the lowest miss, so its advantage comes entirely from single-speaker
detection. `diarizen-large` recovers **84%**, the closest match, and still has
the *worst* miss. Overlap detection and miss are separable failures, and no
model does both.

Fusion overlap **recall is 21.5%** — it finds 678 s of the reference's 3,148 s of overlap. It marks 1,560 s as ≥2 speakers, so **882 s of the overlap it does claim is in the wrong place** (precision 43.5%). Majority voting suppresses second speakers,
because a concurrent speaker needs the same majority as the first and
`reverb-v2` votes against nearly all of them.

## Held-out split

50 dev / 49 test, stratified by script and speaker count, frozen before any
Step 4 tuning. The fusion's threshold was chosen on dev; test was scored once.

| system | dev | test | corpus |
|---|---|---|---|
| `FUSION` | 0.2390 | 0.2481 | 0.2442 |
| `reverb-v2` | 0.2238 | 0.2738 | 0.2521 |
| `diarizen-large` | 0.2659 | 0.2600 | 0.2625 |
| `community-1` | 0.2556 | 0.2694 | 0.2634 |
| `pyannote-3.1` | 0.2482 | 0.2757 | 0.2637 |

**Read this table carefully — it is the most important one here.** Every single
system inverts between halves: `reverb-v2` is best on dev (0.2238) and worst
on test (0.2738); `diarizen-large` is worst on dev and best on test.
With ~50 clips a half and a duration-pooled DER dominated by a handful of long
clips, **a 0.02–0.04 DER gap is inside split noise.**

The fusion is the only entry that is 1st or 2nd on both halves. Its value is
less that it edges the average and more that it removes the variance —
picking any single model amounts to betting on which half of the corpus you drew.

## Ground-truth alignment (diagnostic, not the headline)

39 of 99 clips were flagged as carrying annotations displaced 1.0–5.0 s later
than the speech, by cross-model consensus with independent energy-VAD
corroboration and visual spot-checks against the waveform; **23** met the
auto-accept bar and are the ones actually applied below. Corrections are applied to a copy at
scoring time; the raw annotations are never modified.

| system | DER | miss | FA | confusion | JER | spk acc | spk bias |
|---|---|---|---|---|---|---|---|
| `FUSION` | **0.1959** | 0.1002 | 0.0369 | 0.0588 | 0.2975 | 81.8% | -0.15 |
| `diarizen-large` | **0.2039** | 0.0973 | 0.0478 | 0.0588 | 0.2910 | 74.7% | +0.04 |
| `community-1` | **0.2166** | 0.0934 | 0.0465 | 0.0766 | 0.3193 | 80.8% | -0.15 |
| `pyannote-3.1` | **0.2190** | 0.0934 | 0.0465 | 0.0790 | 0.3233 | 73.7% | -0.15 |
| `reverb-v2` | **0.2396** | 0.0712 | 0.0681 | 0.1003 | 0.3625 | 62.6% | -0.33 |

### What each component gained

| system | DER | miss | FA | confusion | JER |
|---|---|---|---|---|---|
| `FUSION` | 0.2442 → **0.1959** | 0.1138 → 0.1002 | 0.0498 → 0.0369 | 0.0806 → 0.0588 | 0.3608 → 0.2975 |
| `reverb-v2` | 0.2521 → **0.2396** | 0.0719 → 0.0712 | 0.0681 → 0.0681 | 0.1121 → 0.1003 | 0.3834 → 0.3625 |
| `diarizen-large` | 0.2625 → **0.2039** | 0.1143 → 0.0973 | 0.0640 → 0.0478 | 0.0842 → 0.0588 | 0.3660 → 0.2910 |
| `community-1` | 0.2634 → **0.2166** | 0.1079 → 0.0934 | 0.0602 → 0.0465 | 0.0954 → 0.0766 | 0.3768 → 0.3193 |
| `pyannote-3.1` | 0.2637 → **0.2190** | 0.1079 → 0.0934 | 0.0602 → 0.0465 | 0.0957 → 0.0790 | 0.3785 → 0.3233 |

**Every component improves for every system.** That is the signature a global
time shift should leave: displacing the reference misaligns speech onsets
(miss), speech offsets (false alarm) and speaker identity (confusion) all at
once. A correction that improved only one of them would be evidence of
something other than a shift, and would be a reason to distrust it.

Three details worth reading:

* **`reverb-v2`'s false alarm does not move at all** (0.0681 → 0.0681).
  Its turns average 15.5 s, long enough that a ±2 s displacement rarely pushes
  one off the end of real speech. That insensitivity is exactly what made it
  look best on raw ground truth, and it is why it gains least from the fix.
* **JER improves most in relative terms** — up to 20% — because it is computed
  per speaker, and a displaced reference damages every speaker's segments
  simultaneously rather than concentrating the damage anywhere.
* **Speaker-count accuracy barely moves**, +2.02 points — two clips — for
  every system, and the
  bias is essentially unchanged. That is a consistency check rather than a null
  result: how many speakers a model finds should not depend on *when* the
  reference says they spoke. A correction that changed speaker counts would mean
  it was doing something other than shifting time.

Overlap figures are unchanged by alignment, since predicted overlap is a
property of the hypothesis and the reference's total overlapped duration is
preserved by a shift.

## By script (DER, raw ground truth)

| script | clips | `FUSION` | `reverb-v2` | `diarizen-large` | `community-1` | `pyannote-3.1` |
|---|---|---|---|---|---|---|
| Malayalam | 7 | 0.356 | 0.296 | 0.409 | 0.364 | 0.367 |
| Telugu | 12 | 0.317 | 0.340 | 0.334 | 0.338 | 0.350 |
| Bengali | 8 | 0.317 | 0.359 | 0.331 | 0.335 | 0.315 |
| Gujarati | 12 | 0.266 | 0.264 | 0.290 | 0.286 | 0.278 |
| Devanagari | 25 | 0.232 | 0.210 | 0.266 | 0.251 | 0.249 |
| Oriya | 9 | 0.219 | 0.204 | 0.243 | 0.223 | 0.224 |
| Gurmukhi | 7 | 0.178 | 0.316 | 0.163 | 0.206 | 0.242 |
| Tamil | 10 | 0.169 | 0.175 | 0.183 | 0.193 | 0.185 |
| Kannada | 9 | 0.159 | 0.181 | 0.133 | 0.182 | 0.176 |

Language is a larger axis of variation than model choice — the spread across
scripts is several times the spread across systems. Group sizes are 7–25 clips,
so treat the ordering as indicative rather than significant.

## Step 4 method

DOVER-Lap (Raj et al., SLT 2021): map each system's speaker labels onto a
growing centroid by Hungarian assignment on frame overlap, then vote per 10 ms
frame with **each speaker thresholded independently** so overlap survives.

Three configuration choices, all measured on dev rather than assumed:

* **Equal weights, not DOVER-Lap's 1/rank.** Rank weighting gives the leader 1
  and the runner-up 0.5, so with two systems the leader alone clears any
  threshold below 0.67 and the fusion silently degenerates into "use the best
  system" — it reproduced `reverb-v2`'s dev score to four decimal places. Equal
  weights won 0.2105 to 0.2223.
* **Three systems, dropping `pyannote-3.1`.** Including a near-duplicate lets
  one segmentation lineage carry a majority. Dev improved 0.2181 → 0.2105.
* **Majority (≥2 of 3).** With three equal voters every threshold in (1/3, 2/3]
  is the same rule, so this is a plateau bounded by cliffs rather than a tuned
  value — the fusion has effectively no continuously-fitted parameter.

## Things tried that did not work

**Scope.** The grids below were run on the **dev half only** (`results/split.json`),
which is where every Step 4 configuration decision was made; see `obs.txt` [32]
and [35] for the full grids. They sit next to corpus-scale tables elsewhere in
this file, so read the numbers here as dev, not as 99 clips. None of them was
promoted, so none affects a reported result.

Recorded because a negative result on a well-motivated idea is worth as much as
a positive one.

* **Segmentation transplant** (`reverb-v2` boundaries, `diarizen-large` labels).
  Miss improved exactly as predicted, 0.0731 → 0.0581, but confusion got
  *worse*, 0.0894 → 0.0944, against DiariZen's own 0.0849. Its low confusion is
  entangled with its own 2.6 s turns and does not survive being painted onto
  15.5 s ones. No cell of the full 4×4 grid beat `reverb-v2` alone.
* **Lower threshold for additional speakers**, to recover the overlap majority
  voting suppresses. With three equal voters there is no threshold between
  "2 of 3" and "1 of 3", so it either changed nothing or exploded false alarm.
  Measured on all 99 clips, equal weights, `min_dur` 0.20: dropping the
  threshold from 0.5 to 0.30 moves DER **0.2442 → 0.3193** and false alarm
  **0.0498 → 0.2180**, buying a miss reduction of only 0.1138 → 0.0370. An
  earlier draft of this file reported the false-alarm pair as 0.0441 → 0.1404;
  those figures do not reproduce under any configuration recorded here and are
  replaced by the measured ones. The conclusion is unchanged.
* **Non-uniform weights**, to break that quantisation. Moves the cliffs, does
  not remove them; no weighting beat equal.
* **Blind boundary padding**, to exploit a measured 137 ms trailing bias in the
  reference. Recovered 30% of DiariZen's miss but added false alarm ~1:1, for a
  net −0.0031 DER.

