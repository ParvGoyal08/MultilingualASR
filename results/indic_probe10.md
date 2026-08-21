# IndicConformer-600M-Multilingual — 10-clip probe

Run 2026-08-21 17:57–18:03 UTC, Kaggle 2×T4, via `main_kaggle_2.ipynb` §1d →
`tools/indic_probe.py`. Per-segment on the DOVER-Lap fusion, 30 s cap. Same ten
clips as the Whisper probe, so every row below is directly comparable.

**Language is the ORACLE** — derived from the reference, disambiguating
Devanagari by Marathi/Hindi function words. IndicConformer has **no language
identification of its own**, so a language must come from somewhere; this row
is therefore an **ablation**, not a reportable pipeline result. The leak-free
variant is `--lang lid`. Languages used:

`bn · mr · gu · pa · kn · ml · or · ta · te · mr`

## Results, same 10 clips

| system | language | ratio | WER | cpWER | WDER | del% | script |
|---|---|---|---|---|---|---|---|
| `sarvam-saaras-v3@fusion` | self-detected | 0.97 | **0.2876** | **0.3119** | 0.0641 | 9.3% | — |
| `sarvam-saaras-v3@reverb-v2` | self-detected | 0.92 | 0.2592 | 0.3697 | 0.0931 | 10.7% | — |
| **IndicConformer CTC** | oracle | 0.82 | 0.4158 | 0.4252 | **0.0332** | 19.6% | **10/10** |
| **IndicConformer RNNT** | oracle | 0.69 | 0.4730 | 0.4865 | 0.0393 | 33.0% | **10/10** |
| Whisper `large-v3` (H) | oracle | 0.43 | 0.9058 | 0.9158 | 0.1743 | 58.5% | 8/10 |
| Whisper turbo (F) | oracle | 0.53 | 0.9428 | 0.9453 | 0.1297 | 47.9% | 8/10 |

## Findings

**1. IndicConformer is a real Indic recogniser; Whisper is not.**
WER 0.4158 against Whisper's best 0.9058 — **2.2× better** on identical audio,
identical segmentation and identically-supplied languages. This is the strongest
single piece of evidence in the project that Whisper's failure here is
Indic-specific rather than a configuration problem.

**2. It transcribes every script, including the one Whisper cannot.**
10/10 script match versus 8/10. Whisper has no `or` in its 99 languages
(obs [45]) and 9% of this corpus is Oriya; IndicConformer covers it natively.

**3. Saaras v3 still wins, and the margin is understated.**
0.2876 against 0.4158 on WER, 0.3119 against 0.4252 on cpWER — Saaras is ~1.4×
better **while detecting its own language**, where IndicConformer was handed the
right answer. Removing the oracle can only widen this gap.

**4. CTC beats RNNT here, and is 1.7× faster.**
0.4158 vs 0.4730 WER, 138 s vs 231 s, deletions 19.6% vs 33.0%, repetition 3.9%
vs 6.8%. RNNT normally leads on published Indic benchmarks, so this inverts the
expectation. **Inferred, not established:** RNNT's higher deletion and repetition
rates look like a decoder more prone to early termination on short diarized
segments than on the full utterances it was trained for.

**5. The WDER result is confounded — do not quote it.**
IndicConformer's 0.0332 looks far better than Saaras's 0.0641, but both run
per-segment on the *same* fusion turns, so attribution is identical by
construction. What differs is coverage: WDER's denominator counts only aligned
words (S + C), and IndicConformer emits ratio 0.82 against Saaras's 0.97. Fewer
words align, so the denominator shrinks. This is a coverage artefact, not better
speaker attribution.

## Verdict

**Saaras v3 remains the ASR system of record.** IndicConformer is a legitimate
second Indic STT benchmark that genuinely works, and it makes the Step 3
comparison a three-way one with a clear story: a commercial Indic model, an
open-source Indic model, and a general multilingual model that fails on this
material.

Its remaining value is (a) the second-system requirement in the brief, (b) the
Oriya capability gap, and (c) as the control that proves Whisper's failure is
about Indic coverage rather than decoding configuration.

Not run: the leak-free `--lang lid` variant, and any corpus-scale sweep.
