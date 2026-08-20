# Ground-truth audit worklist

Every ground-truth artifact found in the corpus, ranked by the error it causes,
so the listening pass has a definite scope instead of "open the worst clips".

**How to check one.** Start the explorer (`python3 serve.py`), select the clip,
type the timestamp target into the transport or click the region chip at that
time. The YouTube link opens the same moment in the source video, offset by the
clip's `start_sec` -- useful when the WAV is ambiguous and you want the picture.

**What you are deciding**, per row, one of:

| verdict | meaning | consequence |
|---|---|---|
| `GT-MISSING` | someone is audibly speaking, the reference does not say so | error is unfixable; report as measurement floor |
| `MODEL-WRONG` | silence, music, or noise; the model invented a speaker | genuine false alarm, a tuning target |
| `AMBIGUOUS` | faint, distant, overlapping, or a non-speech vocalisation | record and exclude from both claims |

Record the verdict in the `verdict` column of `results/gt_unannotated_gaps.csv`
and `results/gt_long_turns.csv`.

These two artifact classes together account for **1.75 of the 26.34 DER points**
measured for community-1 (6.6% of all error). That is the ceiling on what this
audit can reclassify -- worth an hour, not a day.

---

## A. Unannotated stretches (false alarm)

39 stretches over 5 s where the reference has **no speaker at all**, totalling
8.6 minutes and carrying **218 s of false alarm**. `covered` is how much of the
stretch the model fills with predicted speech -- rows near 100% are the ones
where the model is most confidently contradicting the reference, and therefore
the most informative to listen to.

`tail` marks a stretch running to the end of the clip: the annotation simply
stops there, which is a different failure from a gap in the middle.

| # | clip | in-clip | dur | FA | covered | tail | listen |
|---|---|---|---|---|---|---|---|
| 1 | `Cku_X_SL7qU__60_660` | 509.69–600.0s | 90.31s | **88.19s** | 90% | yes | [569.7s](https://youtu.be/Cku_X_SL7qU?t=569) |
| 2 | `QuA_B6IZ6Ls__61_1863` | 24.34–54.03s | 29.69s | **28.63s** | 96% |  | [85.3s](https://youtu.be/QuA_B6IZ6Ls?t=85) |
| 3 | `QuA_B6IZ6Ls__61_1863` | 54.83–76.85s | 22.02s | **21.16s** | 96% |  | [115.8s](https://youtu.be/QuA_B6IZ6Ls?t=115) |
| 4 | `HpYG46tGyYs__103_998` | 866.37–880.19s | 13.82s | **13.7s** | 99% |  | [969.4s](https://youtu.be/HpYG46tGyYs?t=969) |
| 5 | `Kdi-ECuOaKg__2_78` | 38.35–49.97s | 11.62s | **7.87s** | 68% |  | [40.4s](https://youtu.be/Kdi-ECuOaKg?t=40) |
| 6 | `8mvORuRHw2U__0_1197` | 424.05–430.83s | 6.78s | **7.28s** | 93% |  | [424.1s](https://youtu.be/8mvORuRHw2U?t=424) |
| 7 | `OxYCBQKZ3iY__0_1822` | 1809.59–1822.0s | 12.41s | **6.92s** | 54% | yes | [1809.6s](https://youtu.be/OxYCBQKZ3iY?t=1809) |
| 8 | `BYLcUGNB_zM__20_1387` | 490.37–511.79s | 21.42s | **4.02s** | 19% |  | [510.4s](https://youtu.be/BYLcUGNB_zM?t=510) |
| 9 | `BYLcUGNB_zM__20_1387` | 1006.01–1027.55s | 21.54s | **4.02s** | 19% |  | [1026.0s](https://youtu.be/BYLcUGNB_zM?t=1026) |
| 10 | `87Zbup3ohcw__106_445` | 116.05–128.47s | 12.42s | **3.6s** | 29% |  | [222.1s](https://youtu.be/87Zbup3ohcw?t=222) |
| 11 | `L7xRazDdtgw__19_83` | 0.0–6.11s | 6.11s | **3.48s** | 57% |  | [19.0s](https://youtu.be/L7xRazDdtgw?t=19) |
| 12 | `0p6cktLGIfY__12_930` | 0.0–6.51s | 6.51s | **2.9s** | 45% |  | [12.0s](https://youtu.be/0p6cktLGIfY?t=12) |
| 13 | `JBYGLQJwFDc__103_407` | 100.51–106.73s | 6.22s | **2.33s** | 37% |  | [203.5s](https://youtu.be/JBYGLQJwFDc?t=203) |
| 14 | `Kdi-ECuOaKg__2_78` | 11.01–17.49s | 6.48s | **2.32s** | 36% |  | [13.0s](https://youtu.be/Kdi-ECuOaKg?t=13) |
| 15 | `JBYGLQJwFDc__103_407` | 86.13–92.87s | 6.74s | **2.3s** | 34% |  | [189.1s](https://youtu.be/JBYGLQJwFDc?t=189) |
| 16 | `2T4pjueLrsk__677_1290` | 177.37–182.65s | 5.28s | **2.26s** | 43% |  | [854.4s](https://youtu.be/2T4pjueLrsk?t=854) |
| 17 | `JBYGLQJwFDc__103_407` | 123.21–129.47s | 6.26s | **2.24s** | 36% |  | [226.2s](https://youtu.be/JBYGLQJwFDc?t=226) |
| 18 | `L7xRazDdtgw__19_83` | 31.12–40.13s | 9.01s | **2.23s** | 25% |  | [50.1s](https://youtu.be/L7xRazDdtgw?t=50) |
| 19 | `Gq45evw6ytY__597_1464` | 747.73–755.79s | 8.06s | **2.0s** | 25% |  | [1344.7s](https://youtu.be/Gq45evw6ytY?t=1344) |
| 20 | `Kdi-ECuOaKg__2_78` | 21.31–27.41s | 6.1s | **1.98s** | 32% |  | [23.3s](https://youtu.be/Kdi-ECuOaKg?t=23) |
| 21 | `0p6cktLGIfY__12_930` | 311.61–317.55s | 5.94s | **1.72s** | 29% |  | [323.6s](https://youtu.be/0p6cktLGIfY?t=323) |
| 22 | `6ZeRgvDHwcI__6_100` | 2.39–11.39s | 9.0s | **1.4s** | 16% |  | [8.4s](https://youtu.be/6ZeRgvDHwcI?t=8) |
| 23 | `Kdi-ECuOaKg__2_78` | 0.0–9.61s | 9.61s | **1.35s** | 14% |  | [2.0s](https://youtu.be/Kdi-ECuOaKg?t=2) |
| 24 | `1LFl5JEipII__0_597` | 467.85–475.03s | 7.18s | **1.28s** | 18% |  | [467.9s](https://youtu.be/1LFl5JEipII?t=467) |
| 25 | `90uAd6xpTPo__16_408` | 123.69–141.93s | 18.24s | **0.76s** | 4% |  | [139.7s](https://youtu.be/90uAd6xpTPo?t=139) |
| 26 | `GEc_sS2KWo8__96_290` | 168.45–187.8s | 19.35s | **0.64s** | 3% |  | [264.4s](https://youtu.be/GEc_sS2KWo8?t=264) |
| 27 | `FDMf9ApmoNU__16_686` | 160.49–165.55s | 5.06s | **0.62s** | 12% |  | [176.5s](https://youtu.be/FDMf9ApmoNU?t=176) |
| 28 | `GEc_sS2KWo8__96_290` | 58.43–72.25s | 13.82s | **0.62s** | 4% |  | [154.4s](https://youtu.be/GEc_sS2KWo8?t=154) |
| 29 | `CO_8ppdzq9U__0_695` | 688.89–695.0s | 6.11s | **0.08s** | 1% | yes | [688.9s](https://youtu.be/CO_8ppdzq9U?t=688) |
| 30 | `8pkp-lpMJP0__326_926` | 588.25–594.41s | 6.16s | **0.07s** | 1% |  | [914.2s](https://youtu.be/8pkp-lpMJP0?t=914) |
| 31 | `1LFl5JEipII__0_597` | 212.64–220.17s | 7.53s | **0.05s** | 1% |  | [212.6s](https://youtu.be/1LFl5JEipII?t=212) |
| 32 | `CO_8ppdzq9U__0_695` | 89.21–108.89s | 19.68s | **0.02s** | 0% |  | [89.2s](https://youtu.be/CO_8ppdzq9U?t=89) |
| 33 | `OxYCBQKZ3iY__0_1822` | 98.13–103.71s | 5.58s | **0s** | 0% |  | [98.1s](https://youtu.be/OxYCBQKZ3iY?t=98) |
| 34 | `OxYCBQKZ3iY__0_1822` | 111.07–116.75s | 5.68s | **0s** | 0% |  | [111.1s](https://youtu.be/OxYCBQKZ3iY?t=111) |
| 35 | `OxYCBQKZ3iY__0_1822` | 977.65–989.11s | 11.46s | **0s** | 0% |  | [977.6s](https://youtu.be/OxYCBQKZ3iY?t=977) |
| 36 | `OxYCBQKZ3iY__0_1822` | 1130.93–1139.17s | 8.24s | **0s** | 0% |  | [1130.9s](https://youtu.be/OxYCBQKZ3iY?t=1130) |
| 37 | `PEDfT9-Yf2o__0_469` | 70.51–75.67s | 5.16s | **0s** | 0% |  | [70.5s](https://youtu.be/PEDfT9-Yf2o?t=70) |
| 38 | `PEDfT9-Yf2o__0_469` | 144.31–178.39s | 34.08s | **0.0s** | 0% |  | [144.3s](https://youtu.be/PEDfT9-Yf2o?t=144) |
| 39 | `RZUZyvRzi80__0_336` | 0.0–7.37s | 7.37s | **0s** | 0% |  | [0.0s](https://youtu.be/RZUZyvRzi80?t=0) |

## B. Implausibly long reference turns (miss)

16 single reference turns longer than 60 s, totalling 35.6 minutes and carrying
563 s of miss. A turn of this length is one uninterrupted utterance by one
speaker; where the speaker is actually silent inside it, the model correctly
says so and is charged miss for it.

**Row 1 is 97% of this table.** The QuA Speaker_D label carries 546 s of the
563 s; the other fifteen have a median miss of 1% of their length, meaning
those speakers genuinely do talk that long. So this is one pathological label
rather than a labelling convention -- check row 1 carefully and treat rows 2-16
as a spot-check that the rule found nothing else.

`miss%` is the share of the turn the model does not attribute to that speaker.
High values mean the label spans long silences or other speakers.

| # | clip | speaker | in-clip | dur | miss | miss% | listen |
|---|---|---|---|---|---|---|---|
| 1 | `QuA_B6IZ6Ls__61_1863` | Speaker_D | 108.11–1035.13s | 927.02s | **546.07s** | 59% | [169.1s](https://youtu.be/QuA_B6IZ6Ls?t=169) |
| 2 | `83gP2vLH7UY__255_2005` | Speaker_B | 360.83–435.27s | 74.44s | **4.29s** | 6% | [615.8s](https://youtu.be/83gP2vLH7UY?t=615) |
| 3 | `BYLcUGNB_zM__20_1387` | Speaker_D | 925.99–998.83s | 72.84s | **2.7s** | 4% | [946.0s](https://youtu.be/BYLcUGNB_zM?t=946) |
| 4 | `83gP2vLH7UY__255_2005` | Speaker_D | 493.17–559.29s | 66.12s | **2.25s** | 3% | [748.2s](https://youtu.be/83gP2vLH7UY?t=748) |
| 5 | `83gP2vLH7UY__255_2005` | Speaker_C | 989.97–1065.71s | 75.74s | **1.57s** | 2% | [1245.0s](https://youtu.be/83gP2vLH7UY?t=1245) |
| 6 | `83gP2vLH7UY__255_2005` | Speaker_E | 625.79–701.27s | 75.48s | **1.55s** | 2% | [880.8s](https://youtu.be/83gP2vLH7UY?t=880) |
| 7 | `HpYG46tGyYs__103_998` | Speaker_D | 167.21–290.37s | 123.16s | **1.05s** | 1% | [270.2s](https://youtu.be/HpYG46tGyYs?t=270) |
| 8 | `8o7jKmq6HsM__30_1770` | Speaker_D | 1567.06–1641.91s | 74.85s | **0.79s** | 1% | [1597.1s](https://youtu.be/8o7jKmq6HsM?t=1597) |
| 9 | `J4-VpEkBzWw__0_600` | Speaker_F | 369.83–433.25s | 63.42s | **0.66s** | 1% | [369.8s](https://youtu.be/J4-VpEkBzWw?t=369) |
| 10 | `8o7jKmq6HsM__30_1770` | Speaker_F | 621.38–695.49s | 74.11s | **0.63s** | 1% | [651.4s](https://youtu.be/8o7jKmq6HsM?t=651) |
| 11 | `BYLcUGNB_zM__20_1387` | Speaker_A | 0.01–99.21s | 99.2s | **0.41s** | 0% | [20.0s](https://youtu.be/BYLcUGNB_zM?t=20) |
| 12 | `BYLcUGNB_zM__20_1387` | Speaker_C | 1063.23–1143.99s | 80.76s | **0.3s** | 0% | [1083.2s](https://youtu.be/BYLcUGNB_zM?t=1083) |
| 13 | `BYLcUGNB_zM__20_1387` | Speaker_B | 811.67–901.55s | 89.88s | **0.24s** | 0% | [831.7s](https://youtu.be/BYLcUGNB_zM?t=831) |
| 14 | `HpYG46tGyYs__103_998` | Speaker_C | 694.99–755.09s | 60.1s | **0.15s** | 0% | [798.0s](https://youtu.be/HpYG46tGyYs?t=798) |
| 15 | `F2uXav-wriA__0_907` | Speaker_A | 77.61–137.61s | 60.0s | **0.03s** | 0% | [77.6s](https://youtu.be/F2uXav-wriA?t=77) |
| 16 | `BYLcUGNB_zM__20_1387` | Speaker_B | 259.33–378.41s | 119.08s | **0s** | 0% | [279.3s](https://youtu.be/BYLcUGNB_zM?t=279) |

---

## Sampling, if you would rather not do all 55

Rows are ranked by error contribution, so the head is worth more than the tail.
The top 10 of A carry 138 s of the 218 s of false alarm (63%), and the top 5 of
B carry 448 s of the 563 s of miss (80%). Fifteen rows therefore settle roughly
three quarters of what is in question.

## One caveat on the evidence

These figures use community-1. It shares `pyannote/segmentation-3.0` with
pyannote-3.1, so those two agreeing that speech exists in a gap is **not**
independent corroboration -- it is one segmenter counted twice.

Reverb v2 has a different segmenter, so it *is* an independent check. If Reverb
also predicts speech in a gap, that is real evidence the reference is missing
it. Re-export with all three models and this table can carry an agreement
column, which would let the obvious rows be settled without listening at all.

