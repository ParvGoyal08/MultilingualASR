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

**Step 4 result: 3.1% relative DER reduction** over the best single system
(`reverb-v2`, 0.2521 → 0.2442), and the fusion is best on
DER, false alarm, confusion, JER and speaker-count accuracy simultaneously. It is
beaten only on **miss**, where `reverb-v2` leads — voting cannot recover speech
that a majority never found.

## The four models are not four versions of the same thing

The DER spread across the single systems is 0.0116, which is nothing.
By component they behave completely differently:

* `reverb-v2` owns **detection** — miss 0.0719, a third below the field —
  but pays for it with the worst confusion (0.1121) and
  speaker-count accuracy (61%).
* `diarizen-large` owns **assignment** — confusion 0.0842 — and is the only
  model with essentially no speaker-count bias (+0.02 against −0.17 to −0.35).
* `community-1` and `pyannote-3.1` are near-identical: they share
  `pyannote/segmentation-3.0` and differ by 0.44 s of miss+FA over 12.4 hours.

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

The fusion recovers only **50%** — majority voting suppresses second speakers,
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
less that it is 3% better on average and more that it removes the variance —
picking any single model amounts to betting on which half of the corpus you drew.

## Ground-truth alignment (diagnostic, not the headline)

23 of 99 clips carry annotations displaced 1.0–5.0 s later than the speech,
detected by cross-model consensus with independent energy-VAD corroboration and
spot-checked visually against the waveform. Corrections are applied to a copy at
scoring time; the raw annotations are never modified.

| system | DER raw | DER alignment-adjusted | Δ |
|---|---|---|---|
| `FUSION` | 0.2442 | **0.1959** | −19.8% |
| `diarizen-large` | 0.2625 | **0.2039** | −22.3% |
| `community-1` | 0.2634 | **0.2166** | −17.8% |
| `pyannote-3.1` | 0.2637 | **0.2190** | −17.0% |
| `reverb-v2` | 0.2521 | **0.2396** | −4.9% |

The ranking changes: `reverb-v2` leads on raw ground truth and is **last** once
aligned. Long loose turns (15.5 s mean against `diarizen-large`'s 2.6 s) still
overlap a reference displaced by a second or two, so the least temporally precise
model looked best precisely because the reference was wrong. It gains least from
the fix (−4.9%) and the most precise gains most (−22.3%).

`FUSION` leads under **both** protocols, so its advantage does not depend on
which reference version you believe.

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

Recorded because a negative result on a well-motivated idea is worth as much as
a positive one.

* **Segmentation transplant** (`reverb-v2` boundaries, `diarizen-large` labels).
  Miss improved exactly as predicted, 0.0731 → 0.0581, but confusion got
  *worse*, 0.0894 → 0.0944, against DiariZen's own 0.0849. Its low confusion is
  entangled with its own 2.6 s turns and does not survive being painted onto
  15.5 s ones. No cell of the full 4×4 grid beat `reverb-v2` alone.
* **Lower threshold for additional speakers**, to recover the overlap majority
  voting suppresses. With three equal voters there is no threshold between
  "2 of 3" and "1 of 3", so it either changed nothing or exploded false alarm
  (0.0441 → 0.1404).
* **Non-uniform weights**, to break that quantisation. Moves the cliffs, does
  not remove them; no weighting beat equal.
* **Blind boundary padding**, to exploit a measured 137 ms trailing bias in the
  reference. Recovered 30% of DiariZen's miss but added false alarm ~1:1, for a
  net −0.0031 DER.

