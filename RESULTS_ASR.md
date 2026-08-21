# Step 3 — ASR on diarized segments

Speaker-attributed transcription over the 99 extracted clips (12.26 h,
123,896 reference words after normalization), scored with **cpWER** as the
headline and **WER**, **DI-cpWER** and **WDER** as the decomposition.

Everything below states the algorithm actually executed, with the constants and
the code path, so a number can be re-derived rather than taken on trust. Where a
result is partial or a claim untested, it says so in place rather than in a
footnote.

---

## 1. Methodology

### 1.1 Text normalization

One normalizer, `reference.normalize_text()`, applied **identically** to
reference and hypothesis. Ordered, and every step is here because it was
measured on this corpus, not because it is conventional:

| # | step | why, measured |
|---|---|---|
| 1 | Unicode NFC | 154 segments are not NFC (Bengali `ড়` U+09DC decomposes) |
| 2 | strip inline non-speech tags | 236 *speech* segments carry `<unintelligible>` |
| 3 | strip glosses | 24,914 annotation tokens, 16.7% of the raw reference |
| 4 | strip zero-width | ZWNJ ×193, ZWJ ×1 |
| 5 | punctuation → space | danda `।` ×1,706 plus 22 other marks |
| 6 | case-fold | 3,180 Capitalized + 835 ALLCAPS Latin tokens |
| 7 | whitespace collapse | — |

Step 3 is the only asymmetry and it is **inert on hypotheses**: the dual-form
convention `कॉफी(coffee)` exists only in the reference, so the step is a no-op on
system output. This is asserted in the notebook's verification cell rather than
assumed. Tokenization is whitespace after normalization.

### 1.2 Segmentation — how audio reaches the recogniser

Two strategies, and which one is available is a property of the system:

**Per-segment** (Saaras, and Whisper when compared against it). Cut the audio at
the diarized turns and transcribe each turn separately. Speaker attribution is
then exact by construction — the segment *is* the speaker.

For Saaras this is not a preference but the only option: Saaras returns exactly
one timestamp span per request (verified across v3/v4, 5 s and 20 s inputs, and
every documented mode), so there are no word times to attribute with.

Before cutting, adjacent turns of the same speaker separated by ≤ **1.0 s** are
merged (`asr.merge_same_speaker`). Diarizers emit utterance-level fragments and a
recogniser handed a 0.3 s fragment has no context: 46% of `community-1`'s raw
turns are under one second. Merged counts differ sharply by source —
`reverb-v2` yields 2,650 segments at median 7.66 s, `community-1` 10,279 at
1.13 s. **The diarization being scored is never modified**; merging is an ASR
segmentation choice applied to a copy.

Two hard rules at the boundary:

* Segments shorter than **0.30 s** are skipped rather than sent.
* Segments longer than **29.0 s** are split into ≤29 s pieces with a 0.5 s
  step-back. This is a server limit, not a guideline: over it the API answers
  `400 Audio duration exceeds the maximum limit of 30 seconds`.

**Long-form** (Whisper only). Transcribe the whole clip, take Whisper's word
timestamps, and assign each word to a turn afterwards with `asr.assign_words`:
each word goes to the turn it **overlaps most**; ties and words landing in a gap
go to the nearest turn by midpoint, so no word is silently dropped. A dropped
word would present as an ASR deletion and quietly flatter WDER, whose
denominator counts only aligned words.

Under overlap this is **dominant-speaker-wins**. That is a real ceiling, not an
implementation shortcut — see §3.

### 1.3 Leak discipline

No language is ever hinted. Saaras is called with `language_code="unknown"`;
Whisper detects from audio. `ref_lang_script` is derived from the reference
transcript and reaches no pipeline stage — it is used only to *group* results in
§6. Pipeline stages receive `ClipInput`, which structurally carries no reference
field, checked by `utils.assert_no_reference_fields()`.

---

## 2. Metric definitions

All four are easy to state almost-correctly, so each is pinned to its source and
to the code that implements it (`sarvam_diar/text_metrics.py`). Word alignment
throughout is Levenshtein via `rapidfuzz`, expanded from block opcodes to
per-word `(op, ref_index, hyp_index)` triples.

### 2.1 WER

    WER = (S + D + I) / N

over a single concatenated, speaker-blind stream. **A diagnostic only**: a
system that transcribes perfectly and attributes every word to one speaker
scores 0. Reported because it separates *word* errors from *attribution* errors,
never as a headline.

### 2.2 cpWER — concatenated minimum-permutation WER

Watanabe et al., CHiME-6, 2020. Concatenate each speaker's words in time order
on both sides, then choose the one-to-one assignment of hypothesis speakers to
reference speakers minimising total errors:

    cpWER = min over assignments π of  Σ_r (S + D + I)(ref_r, hyp_π(r))  /  N

The cost matrix is **padded square** with empty sequences, so an unmatched
reference speaker becomes all deletions and an unmatched hypothesis speaker all
insertions — which is precisely what cpWER should charge for a missed or an
invented speaker.

**On the assignment being exact.** cpWER is often described as requiring a search
over all permutations, with Hungarian offered as an approximation. It is not an
approximation here. The error count for a `(reference speaker, hypothesis
speaker)` pair does not depend on which other pairs are chosen, so the total for
a permutation is a sum of independent cell costs and minimising it *is* the
linear assignment problem. Solved with `scipy.optimize.linear_sum_assignment`,
and `verify_assignment_exact()` checks it against brute force over every
permutation; the notebook runs that check.

### 2.3 DI-cpWER — diarization-invariant cpWER

The same word error with attribution discarded entirely, computed on **flat**
streams — every word on each side in one sequence, no speaker grouping.

    cpWER − DI-cpWER  =  the part of the error that exists only because
                         speakers were assigned wrongly

An earlier implementation concatenated per speaker and then joined the groups.
That is wrong: grouping by speaker is exactly what makes a metric
diarization-*dependent*, so the "invariant" figure still moved when attribution
changed and the subtraction measured nothing. Caught because oracle ASR plus
oracle diarization scored 0.08 instead of 0.

Caveat inherent to the metric: the reference stream is in utterance order while a
recogniser emits in time order, and under overlap those orders genuinely differ.
That costs a little even for a perfect system, which is one more reason the
headline is cpWER.

### 2.4 WDER — word diarization error rate

El Shafey, Soltau & Shafran, *Joint Speech Recognition and Speaker Diarization
via Sequence Transduction*, Interspeech 2019.

    WDER = (S_IS + C_IS) / (S + C)

Of the reference words **aligned** to a hypothesis word — substitutions and
correct words — the fraction carrying the wrong speaker. Hypothesis labels are
first renamed through the mapping cpWER chose, so both metrics describe the same
speaker alignment.

Two traps, both handled explicitly:

* **Substitutions count.** A misrecognised word still has an attribution and it
  can still be wrong. Excluding them measures something else.
* **The denominator is S + C, not C.** Insertions and deletions are excluded from
  *both* numerator and denominator, because a word present on only one side has
  no pair of speaker labels to disagree about.

One caveat belongs to the metric rather than to this implementation: when several
minimal-cost alignments exist they split the same edit distance differently
across S/D/I. `a b c d e` against `a x c e f` costs 3 either as three
substitutions or as one substitution plus a deletion and an insertion — WER is
0.6 either way, but `S + C` differs, so WDER's denominator moves. Every WDER
implementation inherits this; ours is whatever rapidfuzz's backtrace picks,
applied identically to every system so the comparison stays fair.

### 2.5 Pooling

Corpus figures **sum counts and divide once**; they never average per-clip rates.
A 50 s clip and a 30 min clip must not carry equal weight. Same rule as
`evaluation.pool()` for DER.

---

## 3. The floor is not zero

Feeding the **reference transcript back** as if it had been recognised, and
attributing it against the **reference diarization**, still scores:

    cpWER 0.1240      WDER 0.0348      (exactly 0 on the 9 zero-overlap clips)

One transcript cannot carry two people talking at once, so under overlap only the
dominant speaker's words are recoverable and the other's are structurally lost.
**12.8% of reference words** lie in overlapped speech — against 7.13% of *time*,
because overlapped speech is denser.

Every cpWER below should be read against 0.1240, not against zero.

---

## 4. Configuration provenance

Read back from the checkpoint sidecars rather than from memory.

| system | strategy | beam | language ID | batched | VAD |
|---|---|---|---|---|---|
| `sarvam-saaras-v3` | per-segment, merge 1.0 s | — | model-internal (`unknown`) | — | — |
| `sarvam-saaras-v4` | per-segment, merge 1.0 s | — | model-internal (`unknown`) | — | — |
| `whisper-large-v3-turbo` | long-form | **1 (greedy)** | **self-detected** | no | no |
| `whisper-large-v3` | long-form | **1 (greedy)** | **self-detected** | no | no |

The two Whisper rows carry **all three diagnosed defects at once** — greedy
decoding, long-form, and self-detected language. All three have since been
changed in code and none has been re-run at corpus scale. They are recorded as
the starting point of a diagnosis and **must not be quoted as Whisper's
performance on this corpus**.

---

## 5. Results

Attribution for the long-form rows uses the **fusion** diarization.

| system | clips | ratio | WER | cpWER | DI-cpWER | attribution | WDER |
|---|---|---|---|---|---|---|---|
| `sarvam-saaras-v3@reverb-v2` | 99 | 0.96 | 0.2728 | **0.3957** | 0.2728 | 0.1229 | 0.1128 |
| `sarvam-saaras-v3@fusion` | **32 (partial)** | 1.06 | 0.3549 | **0.3811** | — | — | 0.1197 |
| `sarvam-saaras-v4@reverb-v2` | 9 | 0.96 | 0.2606 | **0.3485** | 0.2606 | 0.0879 | 0.0718 |
| `whisper-large-v3-turbo` | 99 | 0.75 | 0.9827 | **0.9957** | 0.9827 | 0.0130 | 0.5471 |
| `whisper-large-v3` | 35 | 0.31 | 0.9296 | **0.9340** | 0.9296 | 0.0044 | 0.2664 |

`ratio` = hypothesis words ÷ reference words; a system producing about as many
words as were spoken sits near 1.0.

> **The `@fusion` row is not comparable to the `@reverb-v2` row.** 32 clips
> against 99, and the 32 are whichever the sweep had finished. It is shown
> because the sweep is still running, and it will be replaced, not because the
> comparison is currently valid.

The Whisper cpWER/WDER figures differ slightly from earlier revisions of this
file (1.0082 → 0.9957, 0.5551 → 0.5471) for one reason: attribution moved from
`reverb-v2` to the fusion. The word streams are byte-identical. This is
independent corroboration that the fusion is the better ASR front-end.

### 5.1 Decomposition of the headline

**cpWER 0.3957 against a floor of 0.1240** — Saaras v3 is **0.2717 above the
achievable floor**, not 0.3957 above zero.

**31% of the cpWER is attribution, not words.** DI-cpWER discards attribution and
lands at 0.2728, so `cpWER − DI-cpWER = 0.1229` is what getting speakers wrong
costs. The oracle's equivalent is 0.0423, so about **0.0806 is avoidable** and
belongs to the diarization rather than the recogniser.

**WDER 0.1128** — of reference words aligned to a hypothesis word, 11.3% carry
the wrong speaker.

---

## 6. By script

`sarvam-saaras-v3@reverb-v2`, grouped by the reference's script. Grouping only —
script never enters the pipeline.

| script | clips | ref words | WER | cpWER | WDER |
|---|---|---|---|---|---|
| Telugu | 12 | 15,848 | 0.3867 | 0.5643 | 0.2088 |
| Bengali | 8 | 8,858 | 0.2317 | 0.5273 | 0.2025 |
| Malayalam | 7 | 5,156 | 0.3311 | 0.4961 | 0.1389 |
| Gurmukhi | 7 | 10,817 | 0.2749 | 0.4794 | 0.1621 |
| Oriya | 9 | 8,469 | 0.3540 | 0.4685 | 0.1127 |
| Kannada | 9 | 9,509 | 0.2607 | 0.3927 | 0.0953 |
| Gujarati | 12 | 16,685 | 0.3078 | 0.3923 | 0.1023 |
| Tamil | 10 | 9,993 | 0.2457 | 0.3373 | 0.0737 |
| Devanagari | 25 | 38,561 | 0.2040 | 0.2607 | 0.0551 |

Telugu is **2.2× Devanagari** on cpWER — language is a far larger axis of
variation than anything else measured here. Group sizes are 7–25 clips, so read
the ordering as indicative rather than significant.

WER and cpWER do **not** rank the scripts the same way: Bengali has the
second-best WER (0.2317) and the second-worst cpWER (0.5273). The words are
recognised; the speakers are not. That is a diarization failure surfacing in an
ASR metric, and it is exactly what cpWER exists to expose.

---

## 7. Whisper: what actually goes wrong

### 7.1 It translates rather than transcribes

Examining all 99 turbo outputs by Unicode script — not by WER, which hides this:

| GT script → HYP script | clips |
|---|---|
| Devanagari → Latin | 19 |
| Telugu → Latin | 11 |
| Tamil → Latin | 10 |
| Kannada → Latin | 9 |
| Gujarati → Latin | 9 |
| Oriya → Latin | 6 |
| Malayalam → Latin | 6 |
| Bengali → Latin | 5 |
| Gurmukhi → Latin | 4 |
| Devanagari → Devanagari | **6 (match)** |
| Gujarati → Gujarati | **1 (match)** |
| other cross-script (incl. Gurmukhi → Arabic) | 13 |

**92 of 99 clips are in a different script from their ground truth.** The Latin
output is **English prose, not romanised Indic**, so no transliteration step can
recover it.

Quality is bimodal, and the distinction matters. Some are accurate translations —
the Gujarati clip renders `હી વોઝ સ્લેટેડ ફોર ટ્વેલ્થ ઓફ મે` as *"it was slated
for 12th of May"* and recovers the names Aditya and Manan; a Tamil clip produces
*"let's welcome Savenia and Vicky"* for `லெட்ஸ் வெல்கம் சவண்யா அண்ட் விக்கி`.
Others are degenerate — a Kannada clip emits *"I have the best opportunity to use
my work"* five times.

So on many clips Whisper **heard the speech correctly and answered in English**.

What the language costs, measured on those same 99 outputs:

| | clips | ratio | WER |
|---|---|---|---|
| script matches GT | 7 | 0.74 | **0.693** |
| script differs | 92 | 0.73 | **1.040** |

The best in-script clips reach WER 0.435–0.540. Note the ratio is **0.73 in both
groups**: the deletion problem is independent of the language problem, so fixing
the language fixes at most one of the two.

**Cause.** `large-v3-turbo` returned `en` on 75 of 99 clips. `large-v3` on the
same audio got 34 of 35 right, so this is specific to the distilled model: turbo
keeps large-v3's encoder but cuts the decoder from 32 layers to 4, and language
ID degraded with it. Fixed in code by delegating detection to `large-v3`.

**Why an LLM cannot repair this.** Back-translating the English output would not
recover the reference's surface forms. The reference contains **19,948
parenthesised English glosses** — code-switched words written phonetically in
native script, e.g. Kannada `ಇನ್ಫಾರ್ಮೇಷನ್` for *information*. An LLM translating
"information" back writes the native word `ಮಾಹಿತಿ`, not the phonetic loan the
reference uses, so WER stays high with perfect semantics. It would also report an
LLM's translation quality as an ASR result. As a **diagnostic** it is legitimate —
judging semantic adequacy separates "heard it, answered in English" from
"hallucinated" — but never as a headline number.

### 7.2 It deletes about half the words

Both variants produce roughly half the reference words, with ~50% of the error
being deletions rather than substitutions. Part of it is **repetition collapse**:
on `0AEEA8NyVwY__11_609` Whisper transcribes 47 s correctly then repeats one
five-word phrase for the remaining 550 s; on `83gP2vLH7UY__255_2005` a single
5-gram occupies 37% of the output.

Sarvam reaches ratio 0.96 on the same audio, so the reference word counts are
right and the audio is transcribable.

### 7.3 The probe — what has been tested, exactly

Three shortest clips, 405 reference words, **beam 5 and `large-v3` language ID**
(i.e. the *fixed* configuration, unlike §4):

| strategy | hyp words | ratio | WER | del% | sec |
|---|---|---|---|---|---|
| long-form | 210 | 0.52 | **0.7481** | 49.9% | 45 |
| long-form, `no_speech_threshold=None` | 210 | 0.52 | **0.7481** | 49.9% | 34 |
| per-segment on fusion | 313 | 0.77 | **0.8000** | 24.9% | 54 |

Three findings:

1. **The language fix works.** Detected `mr` (p=0.83), `hi` (p=0.95), `te`
   (p=0.90) — one detection per clip, propagated to every segment, no `en`
   fallback.
2. **`no_speech_threshold` is a genuine null result.** Identical word count and
   WER to four decimals. The parameter is plumbed through to faster-whisper
   (`asr.py:227`), so the no-speech discard is *not* the deletion mechanism.
   Hypothesis eliminated.
3. **Per-segment trades accuracy for coverage.** Deletions halve (49.9% → 24.9%)
   and ratio rises 0.52 → 0.77, so forcing output per turn does work as
   predicted — but **WER gets worse**, 0.7481 → 0.8000. The ~103 recovered words
   are largely wrong ones. The pre-registered gate was "ratio near 1.0"; 0.77
   against Saaras's 0.96 does not clear it.

405 words on the three shortest clips is a probe, not a benchmark. It is reported
as one.

### 7.4 Not yet benchmarked at corpus scale

beam 5 · forced language · per-segment · batched inference · Whisper's own VAD ·
`large-v3` beyond the 35 clips it reached.

An earlier revision of this file claimed beam size and
`condition_on_previous_text` "were A/B tested and change nothing". That rested on
`tools/whisper_ab.py --clips 3`, printed to stdout and never saved. **Retracted**:
treat those two settings as untested. The tool now writes
`results/whisper_ab.json`.

---

## 8. Saaras v3 vs v4

Paired on the 9 clips both cover:

| system | WER | cpWER | WDER |
|---|---|---|---|
| `sarvam-saaras-v3` | 0.2284 | 0.3197 | 0.0737 |
| `sarvam-saaras-v4` | 0.2606 | 0.3485 | 0.0718 |

v3 wins on WER and cpWER here. An earlier 6-clip sample had v4 ahead by 8%
relative, and a paired bootstrap over those clips put it ahead in only 82% of
resamples — so that result was noise, and 9 clips is not much better evidence.
**The comparison is genuinely unsettled and is not claimed either way.**

---

## 9. Reproducing

| what | where |
|---|---|
| metric implementations | `sarvam_diar/text_metrics.py` |
| segmentation, attribution, backends | `sarvam_diar/asr.py` |
| normalization | `sarvam_diar/reference.normalize_text` |
| exactness check for cpWER's assignment | `text_metrics.verify_assignment_exact` |
| corpus sweeps | `main_kaggle.ipynb` §3 |
| hypothesis probes | `main_kaggle_2.ipynb` |
| decoding A/B | `tools/whisper_ab.py` → `results/whisper_ab.json` |
