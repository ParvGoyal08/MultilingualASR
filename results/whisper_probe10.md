# Whisper configuration probe — 5 configurations × 10 clips

Run: `main_kaggle_2.ipynb` §1b → `tools/whisper_probe.py`, Kaggle 2×T4,
2026-08-21 15:09–15:35 UTC. Raw per-clip rows: `whisper_probe10.json` in the
Kaggle session's `results/` (not carried into the repo).

**Sample** — 10 clips, 19.4 min of audio, all nine scripts, plus one long clip:

| clip | script | sec |
|---|---|---|
| `GfFLEIAAumk__90_163` | Bengali | 73 |
| `T3I2T-cfhG4__160_210` | Devanagari | 50 |
| `L7xRazDdtgw__19_83` | Gujarati | 64 |
| `8r2Nltl0W4o__259_320` | Gurmukhi | 61 |
| `PRAzUz0GANs__223_283` | Kannada | 60 |
| `7L4gi7Ncc0s__90_148` | Malayalam | 58 |
| `7YfsQPYY-W0__351_411` | Oriya | 60 |
| `HZv_WvIr6lE__21_105` | Tamil | 84 |
| `ARZl7LT0UC0__73_130` | Telugu | 57 |
| `0AEEA8NyVwY__11_609` | Devanagari | 598 |

`0AEEA8NyVwY__11_609` is included deliberately: short clips would miss any
long-audio failure mode and flatter every configuration equally.

## Results

| configuration | ratio | WER | cpWER | WDER | del% | script | rep5 | sec |
|---|---|---|---|---|---|---|---|---|
| **A** old: greedy, self-LID, long-form | 0.65 | 0.9640 | 0.9848 | 0.2816 | 36.1% | 2/10 | 7.3% | 0 |
| **B** beam 5 + `large-v3` LID, long-form | 0.54 | 0.9463 | 0.9532 | 0.1843 | 47.4% | **9/10** | 11.1% | 206 |
| **C** B + `condition_on_previous_text=False` | 0.54 | 0.9241 | 0.9272 | **0.1399** | 46.5% | **9/10** | 9.7% | 202 |
| **D** beam 5 + LID, per-segment on fusion | **0.72** | 0.9362 | 0.9401 | 0.1979 | **29.1%** | **9/10** | **6.8%** | 308 |
| **E** `large-v3`, beam 5, own LID, long-form | 0.55 | **0.9058** | **0.9144** | 0.1538 | 47.5% | 8/10 | 14.8% | 591 |

`script` = clips whose output is in the reference's script. `rep5` = share of
output covered by its most frequent 5-gram (corpus median 3.0%).
A costs no GPU — it scores the existing checkpoints on these same clips, so
every comparison here is **paired within clip**.

**Language detected per clip (B/C/D/E path, one detection per clip):**
`bn` 0.97 · `mr` 0.83 · `gu` 0.78 · `pa` 0.57 · `kn` 0.97 · `ml` 0.98 ·
**`bn` 0.87 (Oriya clip — the one miss)** · `ta` 0.97 · `te` 0.90 · `mr` 0.82

## Findings

**1. The language fix works, and it is worth almost nothing.**
Script match goes 2/10 → 9/10, and the detected languages are right (Marathi as
`mr`, Punjabi as `pa`, not `en`). But WER moves only 0.9640 → 0.9463. Getting
the script right buys **0.018 WER**.

This **refutes obs [41]**, which put script match at ~0.35 WER (0.693 vs 1.040).
That figure compared 7 in-script clips against 92 out-of-script ones — a
*between-clip* comparison confounded by which clips happened to be in-script.
The paired within-clip measurement here is the correct design and it says the
effect is negligible.

**2. Correct-script output is SPARSER, not denser.**
Ratio falls 0.65 → 0.54 and deletions rise 36.1% → 47.4% when Whisper stops
translating. This kills the compression hypothesis in obs [42] — English output
was *more* verbose here, not less.

**3. `condition_on_previous_text=False` helps, consistently.**
C beats B on every metric: WER −0.022, cpWER −0.026, WDER −0.044, rep5 −1.4pp.
Small, but same-direction across all four on paired clips. The setting was
reverted in obs [37] for lack of evidence; there is now evidence, and it points
the other way.

**4. Per-segment buys coverage, not accuracy.**
D has the best ratio (0.72) and the lowest deletions (29.1%) and repetition
(6.8%) — forcing output per turn works exactly as predicted. Its WER (0.9362)
beats long-form turbo but loses to `large-v3` long-form, and it costs 1.5× B.

**5. `large-v3` beats turbo on words, at 2.9× the time.**
E has the best WER (0.9058) and cpWER (0.9144) but is the slowest (591 s vs
206 s), the worst on script (8/10) and the most repetitive (rep5 14.8%).

**6. The decisive result: configuration is not the problem.**
Every configuration lands between **0.906 and 0.964 WER**. The full spread across
greedy-vs-beam, self-LID-vs-forced, turbo-vs-large-v3, and
long-form-vs-per-segment is **0.058 WER**. Saaras v3 scores 0.2728 on the same
corpus — Whisper's best is **3.3× worse**, and no setting closes that.

## Conclusion

**Do not sweep Whisper.** The pre-registered gate was ratio ≈ 1.0; the best
configuration reaches 0.72. A corpus sweep would cost 2–4 h to produce a number
around 0.91 WER, which is not a competitive baseline and is not made competitive
by any knob measured here.

Whisper's failure on this corpus is **recognition**, not decoding: it writes the
right script, in the right language, and still gets the words wrong. That is a
finding about multilingual ASR on code-switched Indic YouTube audio, and it is
worth more written up than re-run.
