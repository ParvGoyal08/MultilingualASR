# Step 3 — ASR on diarized segments

Speaker-attributed transcripts over **99 clips / 123,896 reference words**,
scored with cpWER and WDER plus WER and DI-cpWER as diagnostics.

Transcription is **per-segment**: the audio is cut at `reverb-v2`'s diarized
turns and each turn transcribed, so speaker attribution is exact by
construction. That is not a preference — Saaras returns exactly one timestamp
span per request (measured across v3/v4, 5 s/20 s inputs and every mode), so
there are no word times to attribute with and the long-form strategy is
unavailable to it.

Adjacent same-speaker turns are merged first, since 46% of `community-1`'s raw
turns are under a second and a recogniser handed a 0.3 s fragment has no context.
`reverb-v2` needs it least: 2,650 merged segments at a median 7.66 s against
`community-1`'s 10,279 at 1.13 s.

**No language is ever hinted.** Sarvam is called with `language_code="unknown"`
and detects it from audio; `ref_lang_script` is derived from the reference
transcript and never reaches any pipeline stage.

## Results

| system | clips | ref words | ratio | WER | cpWER | DI-cpWER | attribution | WDER |
|---|---|---|---|---|---|---|---|---|
| `sarvam-saaras-v3@reverb-v2` | 99 | 123,896 | 0.96 | 0.2728 | **0.3957** | 0.2728 | 0.1229 | 0.1128 |
| `sarvam-saaras-v4@reverb-v2` | 9 | 12,532 | 0.96 | 0.2606 | **0.3485** | 0.2606 | 0.0879 | 0.0718 |
| `whisper-large-v3-turbo` | 99 | 123,896 | 0.75 | 0.9827 | **1.0082** | 0.9827 | 0.0255 | 0.5551 |
| `whisper-large-v3` | 35 | 45,046 | 0.31 | 0.9296 | **0.9367** | 0.9296 | 0.0071 | 0.3286 |

`ratio` is hypothesis words ÷ reference words — a system producing about as many
words as were spoken sits near 1.0.

**Configuration behind each row**, read back from the checkpoint sidecars rather
than from memory:

| system | strategy | beam | language ID | batched | VAD |
|---|---|---|---|---|---|
| `sarvam-saaras-v3` | per-segment | — | model-internal | — | — |
| `whisper-large-v3-turbo` | long-form | **1 (greedy)** | **self-detected** | no | no |
| `whisper-large-v3` | long-form | **1 (greedy)** | **self-detected** | no | no |

**Only Saaras v3 is a usable result.** The two Whisper rows are not a measurement
of Whisper. They were produced in a configuration carrying all three defects
diagnosed below at once — greedy decoding, long-form, and self-detected language
— and every one of those has since been changed in code but not re-run. They are
recorded as the starting point of the diagnosis, and must not be quoted as
Whisper's performance on this corpus.

**Variations not yet benchmarked at corpus scale:** beam 5, forced language via
`large-v3`, per-segment on the fusion, batched inference, Whisper's own VAD, and
`large-v3` beyond the 35 clips it reached.

## Reading the numbers

**cpWER 0.3957 against a floor of 0.1240.** That floor is not zero: feeding the
reference transcript back as if it had been recognised, attributed against the
reference diarization, still scores cpWER 0.1240 and WDER 0.0348. One transcript
cannot carry two people talking at once, so in overlapped speech only the
dominant speaker's words are recoverable. Saaras is **0.2717 above the achievable
floor**, not 0.3957 above zero — and 12.8% of reference words lie in overlapped
speech against 7.13% of time, because overlapped speech is denser.

**31% of the cpWER is speaker attribution, not words.** DI-cpWER discards
attribution entirely and lands at 0.2728; cpWER − DI-cpWER = 0.1229 is what
getting the speakers wrong costs. The oracle's equivalent is 0.0423, so roughly
0.0806 of it is avoidable and belongs to the diarization rather than
to the recogniser.

**WDER 0.1128** — of the reference words that aligned to a hypothesis word,
that fraction carries the wrong speaker. Definition is (S_IS + C_IS) / (S + C):
substitutions count on both sides, since a misrecognised word still has an
attribution that can be wrong, and insertions and deletions are excluded from
both because a word present on only one side has no speaker pair to compare.

## By script

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

Telugu is 2.2× Devanagari on cpWER — language is a far larger axis of
variation than anything else measured here. Group sizes are 7–25 clips, so read
the ordering as indicative rather than significant.

Note that WER and cpWER do not rank the scripts the same way: Bengali has the
second-best WER (0.2317) and the second-worst cpWER (0.5273). The words are
recognised; the speakers are not. That is a diarization failure surfacing in an
ASR metric, and it is exactly what cpWER is for.

## The Whisper failures

Two independent faults, both diagnosed, neither yet fixed.

**Language identification.** `large-v3-turbo` returned `en` on 75 of 99 clips and
then *translated* to English rather than transcribing. `large-v3` on the same
audio got 34 of 35 right, so this is specific to the distilled model — it keeps
large-v3's encoder but cuts the decoder from 32 layers to 4, and language ID
degraded with it. Fixed in code by delegating detection to `large-v3`, which is
an encoder-only pass.

**Massive deletion.** Both variants produce roughly half the reference words,
with ~50% of the error being deletions rather than substitutions. Beam size and
`condition_on_previous_text` were compared on **three short clips** by
`tools/whisper_ab.py` and appeared to change nothing — a probe, not a benchmark,
and its output was never persisted, so treat it as untested. At least part
of it is repetition collapse: on `0AEEA8NyVwY__11_609` Whisper transcribes 47 s
correctly and then repeats one five-word phrase for the remaining 550 s, and on
`83gP2vLH7UY__255_2005` a single 5-gram occupies 37% of the output.

Sarvam reaches ratio 0.96 on the same audio, so the reference word counts are
right and the audio is transcribable. `main_kaggle_2.ipynb` carries the
diagnosis: Whisper's own per-segment `no_speech_prob` / `avg_logprob` /
`compression_ratio` metadata, the effect of disabling each discard threshold,
and the per-segment path that removes Whisper's freedom to skip a window.

## Saaras v3 vs v4

Paired on the 9 clips both covered:

| system | WER | cpWER | WDER |
|---|---|---|---|
| `sarvam-saaras-v3` | 0.2284 | 0.3197 | 0.0737 |
| `sarvam-saaras-v4` | 0.2606 | 0.3485 | 0.0718 |

v3 wins on WER and cpWER here. An earlier 6-clip sample had v4 ahead by 8%
relative, and a paired bootstrap over those clips put it ahead in only 82% of
resamples — so that result was noise, and this one on 9 clips is not much better
evidence. The comparison is genuinely unsettled and is not claimed either way.

