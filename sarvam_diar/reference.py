"""Build the scoring reference from ground truth.

The brief is explicit: *"The ground truth is never an input to your pipeline.
`diarization_segments` and `asr_segments` are only for computing the final
scores."* Everything in this module is therefore scoring-side only, and the
types it produces (`ClipReference`) are deliberately separate from the types the
pipeline receives (`data.ClipInput`).

Why the reference has to be built rather than scored raw:

* ground truth runs past the audio window on 85/100 clips (median +1.43 s), so
  it must be cropped to the UEM or the hypothesis is scored against speech that
  is not in the file;
* 140 same-speaker interval pairs overlap themselves (340 merges once
  exactly-adjacent intervals are folded in too), which double-counts speaker
  time unless unioned;
* **16.7% of reference tokens (24,914 of 149,121) are code-switch gloss
  annotation, not spoken words** -- scoring raw imposes a ~17% cpWER floor that
  says nothing about ASR quality.

What is deliberately NOT done, because the brief forbids it or it would tune
toward a system: no overlap removal, no forgiveness collar as the headline
metric, no minimum-duration filter, no per-system normalization.

The normalizer is legitimate exactly because it is a pure function of one text
string, applied identically to reference and hypothesis, fixed before any system
output was seen, and versioned (`config.NORMALIZER_VERSION`).
"""

from __future__ import annotations

import collections
import json
import re
import unicodedata
from dataclasses import asdict
from typing import Any, Iterable

import pandas as pd

from .config import NORMALIZER_VERSION, Config
from .data import Clip, ClipReference, Segment, Turn, Utterance
from .utils import LOG, apply_selection, atomic_publish, write_json_atomic

# --------------------------------------------------------------------- regexes

NONSPEECH_TAG_RE = re.compile(r"<[^>]{0,40}>")

# A code-switch gloss: a Latin/digit run inside round, square or curly brackets.
# The leading `\s?` absorbs the optional space in `ஹாய் (hi)` so the native head
# word and its gloss are removed as one unit.
# Same body as data.GLOSS_RE, deliberately compiled differently: data's has a
# CAPTURE GROUP and is used with .findall() to count/extract glosses, this one
# has no group and a leading \s? so .sub() removes the gloss AND the space that
# preceded it. Two names, two jobs -- do not "deduplicate" them into one.
_GLOSS_BODY = r"[\(\[{]\s*[A-Za-z0-9][^)\]}]{0,40}?\s*[\)\]}]"
GLOSS_STRIP_RE = re.compile(r"\s?" + _GLOSS_BODY)
GLOSS_RE = GLOSS_STRIP_RE          # backwards-compatible alias

# `head(gloss)-suffix`, where a native grammatical suffix is re-attached after
# the gloss (1,827 occurrences). The `(?=\s|$)` is load-bearing: it stops the
# suffix group before a following gloss, so chained numerals like
# `आठ(8)-नऊ(9)` (13 cases) fall through to plain gloss removal and stay two
# words instead of being fused into one.
GLOSS_SUFFIX_RE = re.compile(
    r"(?P<head>[^\s\(\[{\)\]}]*)\s?" + _GLOSS_BODY + r"-(?P<suffix>[^\s\(\[{]+)(?=\s|$)"
)

ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")


def rttm_safe(label: str) -> str:
    """RTTM is whitespace-delimited, so a speaker label cannot contain spaces.

    Labels are normalised at build time rather than encoded on write and decoded
    on read: a decode-on-read mangles model labels that legitimately contain
    underscores (`SPEAKER_00` -> `SPEAKER 00`), so write-then-read would not be
    the identity. Scoring is label-agnostic, but Steps 3-4 join transcripts on
    these labels, so they must survive a checkpoint round trip exactly.
    """
    return "_".join(str(label).split())


def _is_punct(ch: str) -> bool:
    """Unicode punctuation/symbol, but never a combining mark.

    Indic vowel signs and viramas are category `M*`, so they survive; the danda
    `।`, ASCII punctuation, `₹` and the `<>` of tag markup are `P*`/`S*` and go.

    This is why the obvious `re.sub(r"[^\\w\\s]", " ", text)` is wrong for this
    corpus: Python's `\\w` is Unicode-aware but keys off `str.isalnum()`, and
    combining marks are not alphanumeric. That expression deletes every vowel
    sign and virama -- `नमस्कार` becomes `नमस क र` -- silently inflating the
    reference by ~14% in shattered tokens and destroying the very characters
    that distinguish Indic words.
    """
    return unicodedata.category(ch)[0] in "PS"


def _strip_trailing_punct(s: str) -> str:
    return s.rstrip("".join(c for c in set(s) if _is_punct(c))) if s else s


# ------------------------------------------------------------- the normalizer


def _resolve_gloss_suffix(match: re.Match) -> str:
    """`head(gloss)-suffix` -> the spoken single word.

    Two cases, both present in the corpus and verified over all 1,827 hits:
      * the head already carries the suffix (385 cases, mostly Tamil) --
        `ஷோக்கு(show)-க்கு` is spoken `ஷோக்கு`, so the repeat is dropped;
      * it does not (1,442 cases, mostly Marathi/Hindi) --
        `चॅनल(channel)-वर` is spoken `चॅनलवर`, so the suffix is joined on.
    Naively always joining produces the doubled `ସେକ୍ସନରେରେ`.
    """
    head, suffix = match.group("head"), match.group("suffix")
    clean_suffix = _strip_trailing_punct(suffix)
    if not head:
        return suffix
    if clean_suffix and head.endswith(clean_suffix):
        return head
    return head + suffix


def strip_glosses(text: str) -> str:
    """Remove dual-form code-switch glosses, keeping the native spoken form."""
    # Suffix-bearing glosses first, so the head and its suffix are reunited
    # before the plain pass would have separated them.
    previous = None
    while previous != text:
        previous = text
        text = GLOSS_SUFFIX_RE.sub(_resolve_gloss_suffix, text)
    # Then plain glosses. Repeated because `((Stardom))` and the 29 unbalanced
    # segments need more than one pass to unwind; leftover stray brackets are
    # removed by punctuation stripping.
    previous = None
    while previous != text:
        previous = text
        text = GLOSS_STRIP_RE.sub("", text)
    return text


def normalize_text(text: str, strip_gloss: bool = True) -> str:
    """The one text normalizer, applied identically to reference and hypothesis.

    `strip_gloss` is the only asymmetry and it is inert: hypotheses contain no
    glosses, so the step is a no-op on them (asserted in the verification cell).
    Ordered, and each step is here because it was measured on this corpus:

    1. NFC          -- 154 segments are not NFC (Bengali `ড়` U+09DC decomposes)
    2. tags         -- 236 *speech* segments carry an inline `<unintelligible>`
    3. glosses      -- 24,914 tokens of annotation, 16.7% of the raw reference
    4. zero-width   -- ZWNJ x193, ZWJ x1
    5. punctuation  -- danda `।` x1,706 plus 22 other marks
    6. case-fold    -- 3,180 Capitalized + 835 ALLCAPS Latin tokens
    7. whitespace
    """
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = NONSPEECH_TAG_RE.sub(" ", text)
    if strip_gloss:
        text = strip_glosses(text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = "".join(" " if _is_punct(ch) else ch for ch in text)
    return " ".join(text.lower().split())


def tokenize(text: str) -> list[str]:
    return text.split()


# --------------------------------------------------- diarization reference


def crop_and_union(segments: Iterable[Segment], uem_end: float) -> tuple[list[Turn], dict]:
    """Crop turns to [0, uem_end] and union each speaker's overlapping intervals.

    Cross-speaker overlap is preserved exactly -- the brief requires it to be
    scored. Only a speaker overlapping (or exactly abutting) *itself* is merged,
    which is bookkeeping, not forgiveness: without it the same second of one
    speaker counts twice in the DER denominator. 340 merges corpus-wide.
    """
    dropped = truncated = 0
    per_speaker: dict[str, list[list[float]]] = collections.defaultdict(list)

    for seg in segments:
        if seg.start >= uem_end:
            dropped += 1
            continue
        end = seg.end
        if end > uem_end:
            truncated += 1
            end = uem_end
        start = max(0.0, seg.start)
        if end > start:
            per_speaker[seg.speaker].append([start, end])

    turns: list[Turn] = []
    merged_pairs = 0
    for speaker, intervals in per_speaker.items():
        speaker = rttm_safe(speaker)
        intervals.sort()
        for lo, hi in intervals:
            speaker = rttm_safe(speaker)
            if turns and turns[-1].speaker == speaker and lo <= turns[-1].end + 1e-9:
                if hi > turns[-1].end:
                    turns[-1] = Turn(speaker, turns[-1].start, hi)
                merged_pairs += 1
            else:
                turns.append(Turn(speaker, lo, hi))
    turns.sort(key=lambda t: (t.start, t.end, t.speaker))
    return turns, {
        "n_segments_dropped_by_uem": dropped,
        "n_segments_truncated_by_uem": truncated,
        "n_same_speaker_merges": merged_pairs,
    }


def overlap_seconds(turns: list[Turn]) -> float:
    """Time with >= 2 distinct speakers active (sweep line, not pairwise sums)."""
    if not turns:
        return 0.0
    bounds = sorted({b for t in turns for b in (t.start, t.end)})
    total = 0.0
    for lo, hi in zip(bounds, bounds[1:]):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2.0
        if len({t.speaker for t in turns if t.start <= mid < t.end}) >= 2:
            total += hi - lo
    return total


# ---------------------------------------------------------- build one clip


def build_reference(clip: Clip, drop_nonspeech: bool = False) -> ClipReference:
    """Assemble the scoring reference for one clip.

    `drop_nonspeech` is the sensitivity variant only. It is False by default
    because tag-only segments are annotated speaker activity, and dropping them
    erases Speaker D entirely from `0p6cktLGIfY__12_930`, changing that clip's
    reference speaker count.
    """
    uem_end = clip.duration  # from start_sec/end_sec -- input columns, not GT
    segments = [s for s in clip.segments if s.is_speech] if drop_nonspeech else clip.segments

    turns, crop_stats = crop_and_union(segments, uem_end)

    utterances: list[Utterance] = []
    for seg in clip.segments:
        if not seg.is_speech:          # no words to score
            continue
        if seg.start >= uem_end:
            continue
        norm = normalize_text(seg.text, strip_gloss=True)
        if not norm:
            continue
        utterances.append(
            Utterance(
                speaker=rttm_safe(seg.speaker),
                start=seg.start,
                end=min(seg.end, uem_end),
                text_raw=seg.text,
                text_norm=norm,
            )
        )

    speech = sum(t.duration for t in turns)
    overlap = overlap_seconds(turns)
    stats = {
        "clip_id": clip.clip_id,
        "uem_end": round(uem_end, 3),
        "n_turns": len(turns),
        "n_speakers": len({t.speaker for t in turns}),
        "speakers": ",".join(sorted({t.speaker for t in turns})),
        "speaker_time_sec": round(speech, 3),
        "overlap_sec": round(overlap, 3),
        "overlap_frac": round(overlap / uem_end, 5) if uem_end else 0.0,
        "n_utterances": len(utterances),
        "n_ref_tokens": sum(len(tokenize(u.text_norm)) for u in utterances),
        "n_bad_gt_segments": len(clip.bad_segments),
        # Dominant Unicode script of the reference transcript, so the explorer
        # can group metrics by language. Scoring-side only: ClipReference is
        # never handed to a model, so this is not the lang_hint leak.
        "lang_script": clip.stats.get("lang_script"),
        "lang_hint": clip.stats.get("lang_hint"),
        "drop_nonspeech": drop_nonspeech,
        **crop_stats,
    }
    return ClipReference(
        clip_id=clip.clip_id,
        uem=(0.0, uem_end),
        turns=turns,
        utterances=utterances,
        stats=stats,
        normalizer_version=NORMALIZER_VERSION,
    )


def speaker_texts(ref: ClipReference) -> dict[str, str]:
    """Per-speaker concatenated normalized text, in time order -- cpWER input."""
    by_speaker: dict[str, list[str]] = collections.defaultdict(list)
    for utt in sorted(ref.utterances, key=lambda u: u.start):
        by_speaker[utt.speaker].append(utt.text_norm)
    return {spk: " ".join(parts) for spk, parts in by_speaker.items()}


def word_stream(ref: ClipReference) -> list[tuple[str, str]]:
    """Flat (word, speaker) stream in time order -- WDER input."""
    out = []
    for utt in sorted(ref.utterances, key=lambda u: u.start):
        out.extend((w, utt.speaker) for w in tokenize(utt.text_norm))
    return out


# ------------------------------------------------------------------- RTTM


def to_rttm(ref: ClipReference) -> str:
    return "".join(
        f"SPEAKER {ref.clip_id} 1 {t.start:.3f} {t.duration:.3f} "
        f"<NA> <NA> {rttm_safe(t.speaker)} <NA> <NA>\n"
        for t in ref.turns
    )


def parse_rttm(text: str) -> list[Turn]:
    turns = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur = float(parts[3]), float(parts[4])
        # RTTM stores onset+duration at 3 dp, so reconstructing the end
        # accumulates float dust (5.79 + 5.08 -> 10.870000000000001). Round it
        # away so a written-then-read reference compares equal to the original.
        # Verbatim: see rttm_safe(). Decoding underscores here would corrupt
        # model labels like SPEAKER_00.
        turns.append(Turn(parts[7], round(start, 6), round(start + dur, 6)))
    return turns


def to_annotation(ref: ClipReference):
    """pyannote Annotation, imported lazily so Step 2 owns that dependency."""
    from pyannote.core import Annotation, Segment as PSegment

    ann = Annotation(uri=ref.clip_id)
    for t in ref.turns:
        ann[PSegment(t.start, t.end)] = t.speaker
    return ann


# --------------------------------------------------------------- the runner


def run(cfg: Config, clips: list[Clip], drop_nonspeech: bool = False,
        force: bool = False, flags=None) -> pd.DataFrame:
    """Build and checkpoint the reference for every clip.

    Cheap compared to extraction, but checkpointed the same way so a bumped
    NORMALIZER_VERSION invalidates references without touching the audio.
    """
    # Accept a StageFlags so callers do not have to remember that this stage
    # spells force_redo differently from the others.
    if flags is not None:
        force = force or bool(getattr(flags, "force_redo", False))
        clips = apply_selection(clips, flags)

    rows, rebuilt, skipped = [], 0, 0
    for clip in clips:
        rttm_path, asr_path = cfg.rttm_path(clip.clip_id), cfg.ref_asr_path(clip.clip_id)
        if not force and rttm_path.exists() and asr_path.exists():
            existing = json.loads(asr_path.read_text(encoding="utf-8"))
            if (existing.get("normalizer_version") == NORMALIZER_VERSION
                    and existing.get("stats", {}).get("drop_nonspeech") == drop_nonspeech):
                rows.append(existing["stats"])
                skipped += 1
                continue

        ref = build_reference(clip, drop_nonspeech=drop_nonspeech)
        tmp = cfg.work_dir / f"{clip.clip_id}.rttm"
        tmp.write_text(to_rttm(ref), encoding="utf-8")
        atomic_publish(tmp, rttm_path)
        tmp.unlink(missing_ok=True)
        write_json_atomic(asr_path, {
            "clip_id": ref.clip_id,
            "uem": list(ref.uem),
            "normalizer_version": ref.normalizer_version,
            "stats": ref.stats,
            "utterances": [asdict(u) for u in ref.utterances],
        })
        rows.append(ref.stats)
        rebuilt += 1

    df = pd.DataFrame(rows)
    tmp = cfg.work_dir / "reference_manifest.csv"
    df.to_csv(tmp, index=False)
    atomic_publish(tmp, cfg.reference_manifest)
    tmp.unlink(missing_ok=True)
    LOG.info("reference: %d rebuilt, %d skipped (normalizer %s)",
             rebuilt, skipped, NORMALIZER_VERSION)
    return df


def load_reference(cfg: Config, clip_id: str) -> ClipReference:
    payload = json.loads(cfg.ref_asr_path(clip_id).read_text(encoding="utf-8"))
    return ClipReference(
        clip_id=payload["clip_id"],
        uem=tuple(payload["uem"]),
        turns=parse_rttm(cfg.rttm_path(clip_id).read_text(encoding="utf-8")),
        utterances=[Utterance(**u) for u in payload["utterances"]],
        stats=payload["stats"],
        normalizer_version=payload["normalizer_version"],
    )


# ------------------------------------------------------------------ report


def normalization_report(clips: list[Clip], cfg: Config | None = None) -> dict[str, Any]:
    """Corpus-level before/after, so every transformation is accountable."""
    segs = [s for c in clips for s in c.segments]
    speech = [s for s in segs if s.is_speech]

    raw_tokens = sum(len(s.text.split()) for s in speech)
    kept_gloss = sum(len(normalize_text(s.text, strip_gloss=False).split()) for s in speech)
    final_tokens = sum(len(normalize_text(s.text).split()) for s in speech)
    normalized = " ".join(normalize_text(s.text) for s in speech)
    residual_latin = re.findall(r"[a-z]+", normalized)

    report = {
        "normalizer_version": NORMALIZER_VERSION,
        "segments": {
            "total": len(segs),
            "speech": len(speech),
            "tag_only_or_empty": len(segs) - len(speech),
            "dropped_malformed": sum(len(c.bad_segments) for c in clips),
        },
        "tokens": {
            "raw_whitespace": raw_tokens,
            "gloss_kept": kept_gloss,
            "gloss_stripped_final": final_tokens,
            "removed_by_gloss_strip": kept_gloss - final_tokens,
            "gloss_share_of_reference": round((kept_gloss - final_tokens) / kept_gloss, 4)
            if kept_gloss else 0.0,
        },
        "residual_latin_tokens": len(residual_latin),
        "residual_latin_share": round(len(residual_latin) / max(final_tokens, 1), 6),
        "residual_latin_sample": sorted(set(residual_latin))[:20],
        "known_uncorrected": {
            "numeral_glosses": "1,298 numeral glosses keep the spoken word form "
                               "(एक(1) -> एक, ~1% of tokens); an ASR emitting '1' mismatches. "
                               "Correcting needs number->word conversion in nine languages.",
            "script_convention": "Whisper tends to emit English for code-switched words and "
                                 "Saaras the native form. Against a native-script reference "
                                 "Whisper is penalised -- reported as a finding, not normalised "
                                 "away.",
        },
    }
    if cfg is not None:
        write_json_atomic(cfg.normalization_report, report)
    return report


GOLDEN_CASES = [
    "आणि तुमचं सगळ्यांचं कॉफी(coffee) क्रिकेट(cricket) आणि बरंच काही",
    "चॅनल(channel)-वरचा पहिला आपला पाहुणा",
    "ସେକ୍ସନରେ(section)-ରେ ଯାଉଛି",
    "आठ(8)-नऊ(9) वर्षांनी",
    "एक(1) सीसीबीके(CCBK) स्पेशल(special)-चा भाग",
    "ஹாய் (hi) ஹலோ (hello) அண்ட் (and) வெல்கம் (welcome)",
    "தேங்க்யூ [thankyou] அண்ட்[and] உங்களோட சப்ஸ்கிரைபர்ஸ்[subscribers]-ஓட",
    "ये लैपटॉप {laptop} में कुछ प्रॉब्लम {problem} है।",
    "ଆଦିତ୍ୟ ସେ ଷ୍ଟାରଡମ୍ ((Stardom)) ଫିଲିଙ୍ଗ୍ (Feeling) ହେଲାଣି",
    "ସବୁ ପ୍ରକାର ଆପଣଙ୍କର ଫାସିଲିଟିଜ((facilities) ଅଛି",
    "நான் சென்னையை வந்தேன் <laughter> ரொம்ப நல்லா",
    "ஷோக்கு(show)-க்கு போயிட்டு",
    "<unintelligible>",
    "ਮੇਲੇ ਵਾਲੇ ਦਿਨ ਪਕੋੜੇ ਬਣਾਏ।",
    "વિઘે 15 મણ જીરું ની ફોરમુલ્યા?",
]


