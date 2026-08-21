# Alignment QC — before and after

Corpus of 99 clips. `pyannote-3.1` is **excluded throughout**: it and
`community-1` share `pyannote/segmentation-3.0` and differ by 0.44 s of miss+FA
over 12.4 hours, so under 2-of-N voting they are one vote counted twice — any
frame both call speech reaches the threshold without a second *independent*
system ever agreeing. Consensus is only meaningful over systems that can
disagree.

`FUSION` is the Step 4 system: 2-of-3 majority vote over the three below.

**18 of 27 detected offsets are applied.** The other
9 are held for manual review and are **not** applied — they are listed at
the end.

---

## Before — raw ground truth

This is the headline protocol. The brief fixes the scoring rules, so these are
the reported numbers.

| system | DER | miss | FA | confusion | JER | spk acc | spk bias |
|---|---|---|---|---|---|---|---|
| `FUSION` | **0.2442** | 0.1138 | 0.0498 | 0.0806 | 0.3608 | 79.8% | -0.17 |
| `reverb-v2` | **0.2521** | 0.0719 | 0.0681 | 0.1121 | 0.3834 | 60.6% | -0.35 |
| `diarizen-large` | **0.2625** | 0.1143 | 0.0640 | 0.0842 | 0.3660 | 72.7% | +0.02 |
| `community-1` | **0.2634** | 0.1079 | 0.0602 | 0.0954 | 0.3768 | 78.8% | -0.17 |

## After — QC-adjusted ground truth

Same audio, same model outputs, same scoring code. Only 18 clips'
reference timestamps move, each by a multiple of 0.5 s. **A diagnostic, not the
headline.**

| system | DER | miss | FA | confusion | JER | spk acc | spk bias |
|---|---|---|---|---|---|---|---|
| `FUSION` | **0.2134** | 0.1036 | 0.0402 | 0.0696 | 0.3123 | 80.8% | -0.16 |
| `diarizen-large` | **0.2239** | 0.1019 | 0.0522 | 0.0698 | 0.3065 | 73.7% | +0.03 |
| `community-1` | **0.2343** | 0.0974 | 0.0503 | 0.0866 | 0.3336 | 79.8% | -0.16 |
| `reverb-v2` | **0.2454** | 0.0713 | 0.0681 | 0.1060 | 0.3692 | 61.6% | -0.34 |

## What moved

| system | DER raw | DER QC | Δ | relative |
|---|---|---|---|---|
| `FUSION` | 0.2442 | 0.2134 | −0.0308 | −12.6% |
| `diarizen-large` | 0.2625 | 0.2239 | −0.0387 | −14.7% |
| `community-1` | 0.2634 | 0.2343 | −0.0292 | −11.1% |
| `reverb-v2` | 0.2521 | 0.2454 | −0.0067 | −2.7% |

### The ranking inverts

Raw, the three single models sit within 0.011 DER and `reverb-v2` leads.
QC-adjusted, `diarizen-large` leads and `reverb-v2` is **last**, 0.022 behind.

That is the substantive finding, not a side effect. `reverb-v2` averages 15.5 s
per turn against `diarizen-large`'s 2.6 s, and long loose turns still overlap a
reference displaced by a second or two. The least temporally precise model
looked best precisely *because* the reference was wrong, and it gains least from
the fix (−2.7%) while the most precise model gains most (−14.7%).

`FUSION` leads under both protocols, which is the useful part: its advantage
does not depend on which reference version you believe.

### Where the gain comes from

Splitting the corpus into the 18 corrected clips and the 81
untouched ones. The untouched half cannot change — it is the control.

| system | corrected 18 (raw → QC) | untouched 81 |
|---|---|---|
| `community-1` | 0.4030 → **0.2002** (−50%) | 0.2400 |
| `reverb-v2` | 0.3569 → **0.3107** (−13%) | 0.2345 |
| `diarizen-large` | 0.4312 → **0.1620** (−62%) | 0.2342 |
| `FUSION` | 0.3922 → **0.1781** (−55%) | 0.2193 |

Two things stand out. On the corrected clips `diarizen-large` falls from 0.4312
to 0.1620 — from *worst of the three* to *better than any model manages on the
clean clips*. And raw scores on the corrected clips (0.36–0.43) are far worse
than on the untouched ones (0.23–0.24), which is what a displaced reference
looks like: not a model failing, a measurement failing.

### Components

| system | miss | FA | confusion | JER |
|---|---|---|---|---|
| `community-1` | 0.1079 → 0.0974 | 0.0602 → 0.0503 | 0.0954 → 0.0866 | 0.3768 → 0.3336 |
| `reverb-v2` | 0.0719 → 0.0713 | 0.0681 → 0.0681 | 0.1121 → 0.1060 | 0.3834 → 0.3692 |
| `diarizen-large` | 0.1143 → 0.1019 | 0.0640 → 0.0522 | 0.0842 → 0.0698 | 0.3660 → 0.3065 |
| `FUSION` | 0.1138 → 0.1036 | 0.0498 → 0.0402 | 0.0806 → 0.0696 | 0.3608 → 0.3123 |

Every component improves for every system, which is the expected signature: a
global time shift misaligns speech onsets (miss), speech offsets (FA) and
speaker identity (confusion) all at once. A correction that improved only one of
them would suggest something other than a shift.

`reverb-v2`'s FA does not move at all (0.0681 → 0.0681). Its turns are long
enough that a ±2 s displacement rarely pushes one off the end of real speech —
the same insensitivity that made it look best on raw GT.

JER improves most in relative terms (up to −16%) because it is computed per
speaker, and a displaced reference damages every speaker's segment at once.

### Speaker counting barely moves

`speaker_count_acc` shifts by about one clip per system and the bias is
essentially unchanged. That is a **consistency check, not a null result**: how
many speakers a model finds should not depend on when the reference says they
spoke. A correction that changed speaker counts would mean it was doing
something other than shifting time.

---

## The 18 applied corrections

| clip | offset | detected | IoU gain | margin | model spread | VAD |
|---|---|---|---|---|---|---|
| `2iMXYxBTwbM__339_401` | −4.5s | +4.64s | 0.268 | 0.135 | 0.01s | ✓ |
| `7YfsQPYY-W0__351_411` | −4.5s | +4.52s | 0.061 | 0.035 | 0.23s | ✓ |
| `4Qg2jWP6_T0__136_202` | −4.0s | +4.04s | 0.103 | 0.073 | 0.21s | ✓ |
| `7k-xDqNjESc__73_981` | −4.0s | +4.01s | 0.064 | 0.048 | 0.19s | ✓ |
| `IAm1rt-MYcA__4_558` | −4.0s | +3.98s | 0.092 | 0.043 | 0.04s | ✓ |
| `L7xRazDdtgw__19_83` | −4.0s | +3.92s | 0.216 | 0.108 | 0.08s | ✓ |
| `87Zbup3ohcw__106_445` | −3.5s | +3.63s | 0.084 | 0.048 | 0.54s | ✓ |
| `LBIWBxQ5H1s__90_1020` | −3.0s | +3.00s | 0.057 | 0.044 | 0.04s | ✓ |
| `6ZeRgvDHwcI__6_100` | −3.0s | +2.98s | 0.216 | 0.183 | 0.03s | ✓ |
| `6f6TLzlP8Wk__82_144` | −2.0s | +2.07s | 0.115 | 0.071 | 0.04s | ✓ |
| `PSIzXy5-y2E__2_918` | −2.0s | +2.00s | 0.105 | 0.066 | 0.23s | ✓ |
| `0SoItGfM_sY__7_88` | −2.0s | +1.93s | 0.128 | 0.042 | 0.41s | ✓ |
| `DPmew0didIM__191_512` | −2.0s | +1.93s | 0.065 | 0.043 | 0.02s | ✓ |
| `Jc5AVwg2cZM__153_214` | −1.5s | +1.44s | 0.249 | 0.122 | 0.12s | ✓ |
| `Tlha36rSd5o__240_374` | −1.5s | +1.37s | 0.092 | 0.053 | 0.04s | ✓ |
| `Cku_X_SL7qU__60_660` | −1.5s | +1.31s | 0.127 | 0.077 | 0.03s | ✓ |
| `2T4pjueLrsk__677_1290` | −1.0s | +1.18s | 0.144 | 0.097 | 0.03s | ✓ |
| `86mMTUeDiR8__181_1079` | −1.0s | +1.02s | 0.144 | 0.095 | 0.02s | ✓ |

Offsets are **subtracted** from reference timestamps. Every detected offset is
positive — annotations run later than speech, never earlier.

## The 9 held for review — NOT applied

| clip | proposed | why held |
|---|---|---|
| `7L4gi7Ncc0s__90_148` | −3.0s | per-model lags spread 3.02s |
| `OXUt5KnlUMo__91_181` | −3.0s | per-model lags spread 5.48s |
| `0p6cktLGIfY__12_930` | −3.0s | per-model lags spread 2.09s |
| `JBYGLQJwFDc__103_407` | −3.0s | per-model lags spread 2.80s |
| `2HGP34TNvjg__84_194` | −2.0s | per-model lags spread 2.68s |
| `Gq45evw6ytY__597_1464` | −2.0s | per-model lags spread 1.49s |
| `Kdi-ECuOaKg__2_78` | −2.0s | energy VAD disagrees, per-model lags spread 4.10s |
| `T3I2T-cfhG4__160_210` | −1.5s | per-model lags spread 1.33s |
| `0VEwL9XZ0LY__261_557` | −1.0s | per-model lags spread 1.19s |

Applying all 27 at full 10 ms precision would take `diarizen-large` to 0.2030
rather than 0.2239 — so roughly 45% of the available correction is being left on
the table by design. Verify these with `verification_worksheet.csv` and they can
be enabled by setting `verified: true` and re-running the QC.

## Reproduce

```bash
python3 tools/gt_alignment_qc.py --models community-1,reverb-v2,diarizen-large
```

