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
| `sarvam-saaras-v3@fusion` | **81 (partial)** | 1.01 | 0.3179 | **0.3446** | — | — | 0.0846 |
| `sarvam-saaras-v4@reverb-v2` | 9 | 0.96 | 0.2606 | **0.3485** | 0.2606 | 0.0879 | 0.0718 |
| `whisper-large-v3-turbo` | 99 | 0.75 | 0.9827 | **0.9957** | 0.9827 | 0.0130 | 0.5471 |
| `whisper-large-v3` | 35 | 0.31 | 0.9296 | **0.9340** | 0.9296 | 0.0044 | 0.2664 |

`ratio` = hypothesis words ÷ reference words; a system producing about as many
words as were spoken sits near 1.0.

> **The `@fusion` row is not comparable to the `@reverb-v2` row.** 81 clips
> against 99, and the 81 are whichever the sweep had finished. It is shown
> because the sweep is still running, and it will be replaced, not because the
> comparison is currently valid.
>
> The trend across sweep progress is nonetheless consistent: at 32 clips the row
> read cpWER 0.3811 / WDER 0.1197, at 81 it reads 0.3446 / 0.0846. If it holds to
> 99, the fusion front-end is worth roughly 13% relative on cpWER and 25% on WDER
> against `reverb-v2` — which would be the Step 4 result.

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

An earlier revision priced this at ~0.35 WER, from a **between-clip**
comparison of the 7 clips that came out in-script against the 92 that did not
(0.693 vs 1.040). That comparison is confounded — which clips land in-script is
not random — and the paired experiment in §7.3 **refutes it**: forcing the
language moves script match from 2/10 to 9/10 and WER by 0.018.

The ratio was 0.73 in both of those groups, and that part survives: the deletion
problem is independent of the language problem.

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

### 7.2 It deletes about half the words — but not by truncating

Both variants produce roughly half the reference words, with ~50% of the error
being deletions rather than substitutions. Two mechanisms have been proposed and
both are now **measured and rejected** as the explanation.

Over all 99 turbo outputs:

| | |
|---|---|
| transcripts spanning <50% of their clip | **0 / 99** |
| `rep5` > 50% (one 5-gram dominates) | 1 / 99 |
| `rep5` > 20% | 3 / 99 |
| median `rep5` | 3.0% |

`rep5` is the share of output tokens covered by occurrences of the most frequent
5-gram, counted by **position** — consecutive repeats produce overlapping
n-grams, so counting occurrences × n yields figures above 100%.

**Not one transcript stops early.** Every one emits words across the full clip
duration, so the deficit is word *density*, not early termination. An earlier
revision of this file said the cause was greedy decoding truncating output; that
is **retracted**.

**Repetition collapse is real but rare.** `Kdi-ECuOaKg__2_78` reaches `rep5`
79.8% (272 words for 76 s) and `83gP2vLH7UY__255_2005` produces 273 words for
1750 s at `rep5` 32.6%. Three clips out of 99 cannot move a corpus ratio from
1.0 to 0.75.

The clip previously cited here as the canonical collapse case,
`0AEEA8NyVwY__11_609`, **is not one**: 896 words, transcript spanning 100% of its
598 s, top 5-gram occurring once, `rep5` 0.6%. Its actual failure is
mis-recognition — garbled Devanagari, Marathi content detected as `hi`.

Sarvam reaches ratio 0.96 on the same audio, so the reference word counts are
right and the audio is transcribable. The density deficit remains **unexplained**;
§7.1 gives the likeliest candidate (English translation is more compact than the
code-switched Indic source), but the 7 in-script clips show the same 0.74 ratio
as the 92 out-of-script ones, which that story does not predict.

### 7.3 The configuration probe — five configurations, ten clips

Ten clips, 19.4 min, all nine scripts plus one long clip (`0AEEA8NyVwY__11_609`,
598 s — included deliberately, since a sample of short clips would miss any
long-audio failure and flatter every configuration equally). **Every comparison
is paired within clip**: row A costs no GPU because it scores the existing
checkpoints on these same ten.

| configuration | ratio | WER | cpWER | WDER | del% | script | rep5 | sec |
|---|---|---|---|---|---|---|---|---|
| **A** old: greedy, self-LID, long-form | 0.65 | 0.9640 | 0.9848 | 0.2816 | 36.1% | 2/10 | 7.3% | 0 |
| **B** beam 5 + `large-v3` LID, long-form | 0.54 | 0.9463 | 0.9532 | 0.1843 | 47.4% | **9/10** | 11.1% | 206 |
| **C** B + `condition_on_previous_text=False` | 0.54 | 0.9241 | 0.9272 | **0.1399** | 46.5% | **9/10** | 9.7% | 202 |
| **D** beam **1** + LID, per-segment on fusion | **0.72** | 0.9362 | 0.9401 | 0.1979 | **29.1%** | **9/10** | **6.8%** | 308 |
| **E** `large-v3`, beam 5, own LID, long-form | 0.55 | **0.9058** | **0.9144** | 0.1538 | 47.5% | 8/10 | 14.8% | 591 |

Full write-up and per-clip detail: `results/whisper_probe10.md`.

**The language fix works and is worth almost nothing.** Script match goes
2/10 → 9/10 with correct languages detected (Marathi as `mr`, Punjabi as `pa`,
no `en` fallback), yet WER moves only 0.9640 → 0.9463 — **0.018**. This is what
refutes the §7.1 estimate.

**Correct-script output is sparser, not denser.** Ratio *falls* 0.65 → 0.54 and
deletions *rise* 36.1% → 47.4% once Whisper stops translating.

**`condition_on_previous_text=False` helps, consistently.** C beats B on WER
(−0.022), cpWER (−0.026), WDER (−0.044) and rep5 (−1.4 pp) — same direction on
all four, paired.

**Per-segment buys coverage, not accuracy.** D has the best ratio and the lowest
deletions and repetition, but its WER loses to `large-v3` long-form at 1.5× the
cost of B.

**Configuration is not the problem.** Every configuration lands between **0.906
and 0.964 WER**. The whole spread across greedy-vs-beam, self-vs-forced LID,
turbo-vs-`large-v3` and long-form-vs-per-segment is **0.058**. Saaras v3 scores
0.2728 on this corpus, so Whisper's best is **3.3× worse** and no knob measured
here closes it.

**One language resisted entirely.** `7YfsQPYY-W0__351_411` is Oriya and every
configuration detected it as `bn` (p = 0.87). Oriya is the thinnest Indic
language in Whisper's training data and is the single script miss.

### 7.4 Decision, and what remains untested

**Whisper is not swept.** The pre-registered gate was ratio ≈ 1.0; the best
configuration reaches 0.72. A corpus sweep would cost 2–4 h to produce roughly
0.91 WER — not a competitive baseline, and not made competitive by any setting
measured. The failure is **recognition, not decoding**: Whisper writes the right
script, in the right language, and still gets the words wrong. That is the
finding worth reporting.

Untested and now unlikely to change the conclusion, given a 0.058 spread across
everything that was tested: batched inference, Whisper's own VAD, temperature
fallback settings, and `large-v3` beyond the 35 clips it reached in §5.

An earlier revision claimed beam size and `condition_on_previous_text` "were A/B
tested and change nothing", on the basis of `tools/whisper_ab.py --clips 3`
printed to stdout and never saved. **Retracted**: §7.3 measures both, and
`condition_on_previous_text` does change something.


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
