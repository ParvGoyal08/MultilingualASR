# Multilingual Speaker Diarization + ASR on Indic YouTube

**Final report.** Full method, per-stage results, failure analysis and operational
notes are in [`README.md`](README.md); the lab notebook is [`obs.txt`](obs.txt).
Every number here is re-derived from committed artefacts and reproduced by
[`main.ipynb`](main.ipynb).

---

```
Saaras v3  @  DOVER-Lap(community-1, reverb-v2, diarizen-large)
           +  per-clip script correction (Claude Sonnet 4.6, temperature 0)
```

| | baseline | final | |
|---|---|---|---|
| **cpWER** | 0.3957 | **0.3049** | **−23.0 %** |
| **WDER** | 0.1128 | **0.0603** | **−46.6 %** |
| WER | 0.2728 | 0.2617 | −4.1 % |
| DER | 0.2521 | 0.2442 | *not significant* |
| JER | 0.3834 | 0.3608 | −5.9 % |
| speaker-count accuracy | 60.6 % | 79.8 % | +19.2 pt |

Baseline is Saaras v3 on the best single diarizer (`reverb-v2`). The final system
wins on **92 of 99 clips** individually.

**Against the achievable floor.** Feeding the reference transcript back as if
perfectly recognised, on the reference diarization, still scores cpWER **0.1242** —
one transcript cannot carry two simultaneous speakers. The final system sits
**0.1807 above that floor**, not 0.3049 above zero.

## 1 · Data and discipline

Benchmark open-source diarization and multiple STT systems on conversational
Indic YouTube audio, then build a pipeline on the best combination that
measurably improves the output.

**99 of 100 clips extracted, 12.26 h, nine Indic scripts, 2–8 speakers per clip,
7.13 % overlapped speech**, 9 clips with no overlap.

Two properties of this corpus drive every decision. Overlap is pervasive rather
than an edge case, so overlap handling is the dominant axis. And the reference is
**code-switched with a dual-form convention** — 20,599 parenthesised Latin glosses,
e.g. `कॉफी(coffee)` — which means *surface form*, not just meaning, is scored.

DER and JER at **collar 0.0 with overlapping speech included**, per the brief;
**cpWER** (CHiME-6, exact Hungarian assignment) and **WDER**. **Ground truth is
never a pipeline input** — stages receive `ClipInput`, a frozen dataclass that
structurally carries no reference field.

## 2 · What produced the gain

| step | cpWER | Δ |
|---|---|---|
| baseline, Saaras v3 @ `reverb-v2` | 0.3957 | |
| 1 · swap diarizer for the DOVER-Lap fusion | 0.3381 | **−0.0576** |
| 2 · fix a provenance bug | 0.3181 | **−0.0200** |
| 3 · per-clip script correction | **0.3049** | **−0.0133** |

**1 · DOVER-Lap fusion.** Hungarian label alignment onto a growing centroid, then
per-frame voting with **each speaker thresholded independently** so overlap
survives — an argmax vote cannot emit two simultaneous speakers, and 7.1 % of this
corpus is overlapped. `pyannote-3.1` is excluded — it shares `segmentation-3.0`
with `community-1` and differs by 0.320 s of miss+FA over 12.4 h, so under
majority voting they would be one vote counted twice.

*GT-free at inference:* `dover_lap` reads only the three constituent RTTMs and the
clip duration, which comes from the CSV manifest, not from any annotation. The
configuration was selected on **dev** and frozen in `results/fusion_config.json`
before test evaluation — the operator is GT-free, the selection used dev DER, and
those are kept apart deliberately.

**2 · A provenance bug — the largest lesson here.** 27 of 99 clips had been
transcribed against a *stale two-system fusion*, so the measured "fusion" result
was partly not the fusion. `settings_key` existed precisely to catch stale work
and missed it, because it fingerprinted **decoding** and not **segmentation**.
The fix was a `segmentation_key` — a hash of the turn boundaries, re-derived from
each transcript's stored spans at audit time and compared against the diarization
on disk. Worth **−0.0200 cpWER and −41 % insertions**, from no modelling change.

For scale: the GT-perfect ceiling for overlap deduplication was −0.023 cpWER. A
provenance fix captured almost the entire prize an LLM refinement layer was being
designed to chase — and then removed it, since duplicated tokens fell 66 %.

**3 · Per-clip script correction.** The reference writes code-switched English
*phonetically in the native script*; Saaras writes Latin. **3,481 of 21,357
substitutions are correctly-recognised words scored wrong for their script.**
Sonnet 4.6 rewrites the script of a token and nothing else, seeing only the clip
it is correcting — no corpus vocabulary, no other clip, never the reference.
Replacement is whole-token, so token count is invariant and words cannot be added,
deleted, reordered or translated.

## 3 · Held-out validation

`results/split.json` is a frozen 50/49 dev/test split. Test was scored **once**,
with model, prompt, temperature and every guard frozen at commit `f284e59`.

| split | cpWER before → after | Δ | 95 % CI | better | worse |
|---|---|---|---|---|---|
| dev (50) | 0.2676 → 0.2619 | −0.0056 | [−0.0108, −0.0034] | 26 | **0** |
| **test (49)** | 0.3552 → **0.3364** | **−0.0189** | [−0.0241, −0.0051] | **29** | **0** |

3,249 of 121,243 tokens changed (2.68 %): **2,232 helpful edits, 0 harmful, 0
cross-script corruptions.** Zero harmful is structural — the reference holds 27
Latin tokens in 123,896, so a Latin hypothesis token is almost never already a
match and an edit can only help or be inert. The downside is bounded at zero.

## 4 · The diarization number, stated plainly

The fusion beats `community-1`, `pyannote-3.1` and `diarizen-large` decisively
(Δ −0.0193 / −0.0195 / −0.0184, all CIs excluding zero, 80+/99 clips). Against `reverb-v2` the DER difference is **not
significant** — Δ −0.0079, CI [−0.0353, +0.0161], `reverb-v2` winning 57 of 99
clips. An earlier draft reported "−3.1 % relative DER" as the Step 4 result; that
is **withdrawn**. What the fusion does win is JER, confusion, speaker-count
accuracy (60.6 % → 79.8 %) and, decisively, the downstream ASR result — cpWER
0.3957 → 0.3181, which *is* significant.

## 5 · Error by language

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

**Telugu is 2.06× Devanagari on cpWER** — language is a far larger axis of
variation than anything else measured. Bengali has the third-best WER but the
worst WDER: words recognised well, speakers attributed badly. Groups are 7–25
clips, so read the ordering as indicative.

## 6 · What did not work

| intervention | outcome |
|---|---|
| 1-of-3 overlap voting | DER 0.2442 → 0.3193; false alarm 0.0498 → 0.2180 |
| ConvTasNet source separation | word recovery 52.6 % → 50.0 % |
| LLM contextual refinement (Gemini, Sonnet) | null; every pre-registered abandon criterion fired |
| segmentation transplant | miss improved, confusion worsened; no grid cell beat `reverb-v2` |
| blind boundary padding | −0.0031 DER — real, reproducible, far too small to ship |
| overlap deduplication | helped before the provenance fix, **hurts after** (+0.0027) |

Two further ideas — an MSDD-style verifier and an OSD∩constituent intersection —
got only as far as candidate-precision audits (11.5 % and 23.6 %) and were dropped
before implementation. They are reported as audits, **not** experiments.

The separation result is scoped narrowly: an *out-of-domain 8 kHz* separator,
given *oracle* overlap regions, does not help. Whether a 16 kHz in-domain
separator would is **untested**.

**Whisper was benchmarked and rejected on evidence**, across three probes:
configuration is not the problem (eight configurations span 0.906–0.964 WER),
language ID is not the problem (oracle language moves WER by ≤ 0.007), and it is
an Indic capability gap — IndicConformer-600M scores **WER 0.4158 against
Whisper's best 0.9058**, 2.2× better from a model 2.6× smaller. On 92 of 99 clips
Whisper emitted fluent English *translation* rather than transcription, and it has
**no Oriya** at all.

## 7 · Where the error still is

**Overlap is the dominant unfixed weakness.** It carries 52.5 % of all miss and
27.4 % of DER error, and overlapped words are deleted at **3.9× the clean rate**.
Fusion overlap recall is 21.5 % (678 s of 3,148 s), and 882 s of the 1,560 s it
does mark is in the wrong place. A perfect separator is worth **−0.031 WER** — the
largest ceiling measured here.

**The brief's hint does not hold on this corpus.** "Repeated text across a speaker
boundary suggesting a false split" occurs **36 times in 6,312 segments** and does
not predict error (lift 1.09×). The signals that do work are duration proxies
pointing at the wrong targets: segments under 2 s have a 56 % error rate but hold
only 30 % of the wrong-labelled time, while the 36 % in long segments has a 5.6 %
error rate — flagging it would be 94 % false positives.

**Ground-truth quality is a measurement floor**, and the audit is *recorded, not
fully resolved*. An alignment audit flagged 39 of 99 clips whose reference is
displaced 1.0–5.0 s, 38 in the same direction. Applying the **23** that met the
auto-accept bar is a **diagnostic, not the headline** — every reported metric uses
raw ground truth — but it moves fusion DER 0.2442 → 0.1959 and *inverts the
ranking among single systems*: the least temporally precise model looked best
because the reference was wrong. **The other 16 remain open**, held for manual
review because the constituent systems disagreed on the lag. Two further defect
families are catalogued and likewise unfixed: 39 unannotated stretches over 5 s
carrying 218 s of false alarm, and 16 implausibly long reference turns carrying
563 s of miss — 97 % of it one pathological speaker label.

## 8 · Engineering lessons

1. **Provenance outranks modelling.** The largest single ASR gain after the
   front-end swap was a bug fix. "Is the input what I think it is" should be
   checked before "is the model good enough".
2. **A silent failure that looks like a valid result is worse than a crash.** Twice:
   a stale fusion that `is_done()` happily skipped, and a truncated LLM response
   that parsed to `{}` and was counted as 431 abstentions.
3. **Aggregates hide inverted rankings.** GT correction moved fusion DER by 19.8 %
   and reversed which single model is best. Any claim resting on a 0.01 DER gap is
   noise — **including one of our own**, now withdrawn.
4. **Structural impossibility beats validation.** Omitting speakers and timestamps
   from the LLM's response schema removed a whole class of failure that no amount
   of prompt instruction would have guaranteed.

## 9 · Limitations

- **DER parity with `reverb-v2`.** The fusion's case rests on JER, speaker count
  and downstream cpWER, not on DER.
- **99 clips is small.** 0.01-level DER differences are not resolvable, and the
  dev/test boundary is stratified on reference-derived fields — held out from
  tuning, but drawing it was not a GT-blind act.
- **Step 4b is script-only.** It cannot fix a misheard word, and 31 % of its edits
  are correct transliterations the annotator spelled differently — unrealised
  headroom.
- **An earlier Step 4b variant** builds one lookup table from all 99 clips'
  hypotheses. It scores 0.0032 better (0.3017) but is **transductive** — with a
  dev-built vocabulary its test gain collapses from −0.0216 to −0.0042. The
  per-clip variant ships because its number survives review.
- **Whisper's 99-clip row is a known-broken configuration** and is excluded from
  the headline; IndicConformer and Saaras v4 cover only 10 and 9 clips.
- **The GT audit is a diagnostic, not a correction pass.** 39 clips flagged, 23
  auto-accepted, **16 held for manual review**; two further defect families
  catalogued and unfixed. Visual spot-checks covered 15–20 of the 39 and per-clip
  verdicts were not recorded, so the attestation is at that granularity. Headline
  numbers use raw, unmodified ground truth throughout.

## 10 · Reproducing

[`main.ipynb`](main.ipynb) runs Steps 1–5 end to end from committed checkpoints —
**no GPU, no API key, no gated model access.** A `SUBSET = 10` toggle verifies the
whole pipeline in under a minute; verified from a clean clone. Per-video tables:
`results/step2_metrics.csv` (495 rows) and `results/step3_metrics.csv` (638 rows),
both pooling exactly to every figure above.

---

**References.** Raj et al., *DOVER-Lap*, SLT 2021 · Bredin, *pyannote.audio*,
Interspeech 2023 · Han et al., *DiariZen*, ICASSP 2025 · Rev.ai *Reverb v2* ·
Bredin, *pyannote.metrics*, 2017 · Sarvam AI *Saaras v3* · Radford et al.,
*Whisper*, ICML 2023 · AI4Bharat *IndicConformer-600M* · Watanabe et al.,
*CHiME-6*, 2020 (cpWER) · El Shafey et al., 2019 (WDER) · Kuhn, *Hungarian
Method*, 1955 · Anthropic *Claude Sonnet 4.6*.
