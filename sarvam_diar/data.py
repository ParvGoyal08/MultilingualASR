"""Dataset loading, ground-truth parsing and profiling.

Parsing notes that the regexes below depend on (all verified against the real
file, 100 rows / 9,942 segments):

* Both label columns are `|`-separated. Split on `|` FIRST and anchor the
  timestamp regex at the start of each piece -- 76 transcript segments contain
  literal square brackets as code-switch glosses (``டவுட் [doubt]``), so a
  global `findall` over `\\[...\\]` mis-parses the file.
* `diarization_segments[n]` and `asr_segments[n]` carry byte-identical
  timestamps for every segment, so index-join and time-join are equivalent.
* Two segments in the corpus are malformed and are dropped here with a flag
  rather than silently: ``QuA_B6IZ6Ls`` seg 93 has end < start, and
  ``6ZeRgvDHwcI`` seg 0 has zero duration.
"""

from __future__ import annotations

import collections
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import SEGMENTS_CSV_FILE_ID, Config
from .utils import LOG, read_json, write_json_atomic

EXPECTED_COLUMNS = [
    "video_id",
    "start_sec",
    "end_sec",
    "youtube_link",
    "diarization_segments",
    "asr_segments",
]
EXPECTED_ROWS = 100

# Anchored at both ends: the speaker name is non-greedy so it stops at the
# final bracketed range even if a name were to contain brackets.
DIAR_RE = re.compile(
    r"^(?P<speaker>.+?)\s*\[\s*(?P<start>-?\d+(?:\.\d+)?)\s*-\s*(?P<end>-?\d+(?:\.\d+)?)\s*\]$"
)
# Anchored at the start: this is what keeps `[doubt]` inside the text from
# being read as a timestamp.
ASR_RE = re.compile(
    r"^\[\s*(?P<start>-?\d+(?:\.\d+)?)\s*-\s*(?P<end>-?\d+(?:\.\d+)?)\s*\]\s*(?P<text>.*)$",
    re.S,
)

NONSPEECH_TAG_RE = re.compile(r"<[^>]{0,40}>")
TAG_ONLY_RE = re.compile(r"^(?:\s*<[^>]{0,40}>\s*)+$")
# Code-switch glosses appear with and without a space, in round, square OR
# curly brackets, and are sometimes followed by a re-attached native suffix.
# Curly braces are rare (10 occurrences, all in Iare1Emeueg) but 16 segments
# across 8 videos mix more than one bracket style, so all three are matched.
GLOSS_RE = re.compile(r"[\(\[{]\s*([A-Za-z0-9][^)\]}]{0,40}?)\s*[\)\]}]")

# Unicode block -> (script, ISO-ish language hint). Devanagari is genuinely
# ambiguous in this corpus (both Hindi and Marathi videos), so it is labelled
# as such rather than guessed.
SCRIPT_BLOCKS: list[tuple[int, int, str, str]] = [
    (0x0900, 0x097F, "Devanagari", "hi_or_mr"),
    (0x0980, 0x09FF, "Bengali", "bn"),
    (0x0A00, 0x0A7F, "Gurmukhi", "pa"),
    (0x0A80, 0x0AFF, "Gujarati", "gu"),
    (0x0B00, 0x0B7F, "Oriya", "or"),
    (0x0B80, 0x0BFF, "Tamil", "ta"),
    (0x0C00, 0x0C7F, "Telugu", "te"),
    (0x0C80, 0x0CFF, "Kannada", "kn"),
    (0x0D00, 0x0D7F, "Malayalam", "ml"),
    (0x0600, 0x06FF, "Arabic", "ur"),
]


def clip_id_of(video_id: str, start_sec: float, end_sec: float) -> str:
    """Composite key.

    `video_id` alone is NOT unique: `RL2fhIEEbZg` appears twice with the
    disjoint windows 0-65 and 66-1034.
    """
    return f"{video_id}__{int(round(start_sec))}_{int(round(end_sec))}"


# ------------------------------------------------------------------ containers


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_speech(self) -> bool:
        """Tag-only and empty segments are real turns but carry no words."""
        return bool(self.text.strip()) and not TAG_ONLY_RE.match(self.text)


@dataclass
class Clip:
    """Parsed CSV row: audio window PLUS ground truth.

    This is the *authoring* type, used by Step 1 and by reference building. It
    is never handed to a pipeline stage -- see `split_reference()`.
    """

    clip_id: str
    video_id: str
    youtube_link: str
    start_sec: float
    end_sec: float
    segments: list[Segment] = field(default_factory=list)
    bad_segments: list[dict] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})


# ----------------------------------------------------- the leakage boundary
# "The ground truth is never an input to your pipeline. diarization_segments and
# asr_segments are only for computing the final scores."
#
# Steps 2-4 accept ClipInput and nothing else, so the classic leaks -- passing
# the true speaker count to a diarizer, or a ground-truth-derived language code
# to an ASR -- are type errors rather than matters of discipline.


@dataclass(frozen=True)
class ClipInput:
    """Everything a pipeline stage is allowed to see. No ground truth."""

    clip_id: str
    video_id: str
    youtube_link: str
    start_sec: float
    end_sec: float
    duration: float
    wav_path: Path | None = None
    n_samples: int | None = None

    # Deliberately absent, and asserted absent by the verification cell:
    # segments, speaker counts, speaker labels, language hints, overlap stats.


@dataclass(frozen=True)
class Turn:
    """A reference speaker turn (DER / JER)."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Utterance:
    """A reference utterance with normalized text (cpWER / WDER)."""

    speaker: str
    start: float
    end: float
    text_raw: str
    text_norm: str


@dataclass(frozen=True)
class ClipReference:
    """Scoring-side ground truth. Never passed to a pipeline stage."""

    clip_id: str
    uem: tuple[float, float]
    turns: list[Turn]
    utterances: list[Utterance]
    stats: dict[str, Any]
    normalizer_version: str

    @property
    def n_speakers(self) -> int:
        return len({t.speaker for t in self.turns})


def split_reference(
    clips: Iterable[Clip], results: pd.DataFrame | None = None, cfg=None
) -> tuple[dict[str, ClipInput], list[Clip]]:
    """Separate what the pipeline may see from what only the scorer may see.

    Returns pipeline-safe `ClipInput`s keyed by clip_id, plus the `Clip`s
    themselves for `reference.run()` to build the scoring reference from. Pass
    the Step 1 results frame to attach `wav_path` / `n_samples` and to restrict
    the inputs to clips that actually extracted.
    """
    audio: dict[str, dict] = {}
    if results is not None and len(results):
        ok = results[results.status == "ok"]
        audio = {
            r["clip_id"]: {"wav_path": r.get("wav_path"), "n_samples": r.get("n_samples")}
            for r in ok.to_dict("records")
        }

    inputs: dict[str, ClipInput] = {}
    for clip in clips:
        if results is not None and clip.clip_id not in audio:
            continue
        extra = audio.get(clip.clip_id, {})
        n = extra.get("n_samples")
        # Prefer this run's audio directory over the path recorded at extraction
        # time: Step 1 may have run on a different machine (it did here), so the
        # recorded absolute path is meaningless once the WAVs are copied across.
        wav = extra.get("wav_path")
        if cfg is not None:
            local = cfg.wav_path(clip.clip_id)
            if local.exists():
                wav = str(local)
        inputs[clip.clip_id] = ClipInput(
            clip_id=clip.clip_id,
            video_id=clip.video_id,
            youtube_link=clip.youtube_link,
            start_sec=clip.start_sec,
            end_sec=clip.end_sec,
            duration=clip.duration,
            wav_path=Path(wav) if isinstance(wav, str) and wav else None,
            n_samples=int(n) if n == n and n is not None else None,  # NaN-safe
        )
    return inputs, list(clips)


# -------------------------------------------------------------------- loading


def load_segments_csv(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Fetch youtube_segments.csv to Drive once, then reuse it (skip-if-exists)."""
    path = cfg.segments_csv
    if force or not path.exists() or path.stat().st_size == 0:
        url = f"https://drive.google.com/uc?export=download&id={SEGMENTS_CSV_FILE_ID}"
        LOG.info("downloading segments CSV -> %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import gdown  # available on Colab

            gdown.download(url, str(path), quiet=True)
        except ImportError:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
    else:
        LOG.info("segments CSV already cached (%s)", path)

    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"segments CSV missing columns: {missing}")
    if len(df) != EXPECTED_ROWS:
        LOG.warning("expected %d rows, got %d", EXPECTED_ROWS, len(df))
    # Column order in the file differs from the brief -- always index by name.
    df = df[EXPECTED_COLUMNS].copy()
    df["clip_id"] = [
        clip_id_of(v, s, e) for v, s, e in zip(df.video_id, df.start_sec, df.end_sec)
    ]
    if df.clip_id.duplicated().any():
        dupes = df.loc[df.clip_id.duplicated(keep=False), "clip_id"].tolist()
        raise ValueError(f"clip_id is not unique: {dupes}")
    return df


# -------------------------------------------------------------------- parsing


def parse_diarization_field(raw: str) -> tuple[list[tuple[str, float, float]], list[str]]:
    parsed, unparsed = [], []
    for piece in str(raw).split("|"):
        piece = piece.strip()
        if not piece:
            continue
        m = DIAR_RE.match(piece)
        if not m:
            unparsed.append(piece)
            continue
        parsed.append((m.group("speaker").strip(), float(m.group("start")), float(m.group("end"))))
    return parsed, unparsed


def parse_asr_field(raw: str) -> tuple[list[tuple[float, float, str]], list[str]]:
    parsed, unparsed = [], []
    for piece in str(raw).split("|"):
        piece = piece.strip()
        if not piece:
            continue
        m = ASR_RE.match(piece)
        if not m:
            unparsed.append(piece)
            continue
        parsed.append((float(m.group("start")), float(m.group("end")), m.group("text").strip()))
    return parsed, unparsed


def _sweep(segments: list[Segment]) -> tuple[float, float, float]:
    """Sweep-line over segment boundaries.

    Returns (speech_union_sec, overlap_sec, speaker_time_sec) where overlap is
    time with >= 2 *distinct* speakers active. Pairwise summation would
    double-count 3-way overlaps, so the elementary-interval sweep is used
    instead -- these numbers feed the DER denominator in Step 2.
    """
    if not segments:
        return 0.0, 0.0, 0.0
    bounds = sorted({b for s in segments for b in (s.start, s.end)})
    union = overlap = 0.0
    for lo, hi in zip(bounds, bounds[1:]):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2.0
        active = {s.speaker for s in segments if s.start <= mid < s.end}
        if active:
            union += hi - lo
            if len(active) >= 2:
                overlap += hi - lo
    speaker_time = sum(s.duration for s in segments)
    return union, overlap, speaker_time


def dominant_script(text: str) -> tuple[str, str, float]:
    """(script, language hint, latin character fraction) for a transcript blob."""
    counts: collections.Counter = collections.Counter()
    latin = total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        cp = ord(ch)
        if cp < 128:
            latin += 1
            continue
        for lo, hi, script, lang in SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                counts[(script, lang)] += 1
                break
    if not counts:
        return "Latin", "en", 1.0 if total else 0.0
    (script, lang), _ = counts.most_common(1)[0]
    return script, lang, (latin / total if total else 0.0)


def compute_gt_stats(clip: Clip) -> dict[str, Any]:
    segments = clip.segments
    union, overlap, speaker_time = _sweep(segments)
    max_end = max((s.end for s in segments), default=0.0)
    min_start = min((s.start for s in segments), default=0.0)
    joined = " ".join(s.text for s in segments)
    script, lang, latin_frac = dominant_script(joined)
    tags = collections.Counter(NONSPEECH_TAG_RE.findall(joined))
    return {
        "n_gt_segments": len(segments),
        "n_gt_speakers": len(clip.speakers),
        "gt_speakers": ",".join(clip.speakers),
        "gt_min_start": round(min_start, 3),
        "gt_max_end": round(max_end, 3),
        # Positive => ground truth annotates speech beyond the hard clip
        # boundary. 85/100 rows do (median +1.43 s, max +6.41 s). Step 2 must
        # crop the reference to the UEM rather than extend the audio.
        "gt_overrun_sec": round(max_end - clip.duration, 3),
        "gt_speech_sec": round(union, 3),
        "gt_speaker_time_sec": round(speaker_time, 3),
        "gt_overlap_sec": round(overlap, 3),
        "gt_overlap_frac": round(overlap / clip.duration, 5) if clip.duration else 0.0,
        "n_bad_gt_segments": len(clip.bad_segments),
        "n_nonspeech_segments": sum(1 for s in segments if not s.is_speech),
        "n_nonspeech_tags": int(sum(tags.values())),
        "lang_script": script,
        "lang_hint": lang,
        "latin_char_frac": round(latin_frac, 4),
    }


def parse_ground_truth(df: pd.DataFrame, strict: bool = True) -> list[Clip]:
    """Turn the two label columns into Clip objects with per-clip statistics."""
    clips: list[Clip] = []
    total_unparsed = 0

    for _, row in df.iterrows():
        diar, diar_bad = parse_diarization_field(row.diarization_segments)
        asr, asr_bad = parse_asr_field(row.asr_segments)
        total_unparsed += len(diar_bad) + len(asr_bad)
        if diar_bad or asr_bad:
            LOG.warning("%s: %d unparsed entries", row.clip_id, len(diar_bad) + len(asr_bad))
        if len(diar) != len(asr):
            msg = f"{row.clip_id}: {len(diar)} diarization vs {len(asr)} asr entries"
            if strict:
                raise ValueError(msg)
            LOG.warning(msg)

        clip = Clip(
            clip_id=row.clip_id,
            video_id=row.video_id,
            youtube_link=row.youtube_link,
            start_sec=float(row.start_sec),
            end_sec=float(row.end_sec),
        )
        for idx, ((speaker, d_start, d_end), (a_start, a_end, text)) in enumerate(zip(diar, asr)):
            if abs(d_start - a_start) > 1e-6 or abs(d_end - a_end) > 1e-6:
                LOG.warning(
                    "%s seg %d: timestamp mismatch %s vs %s",
                    row.clip_id, idx, (d_start, d_end), (a_start, a_end),
                )
            if d_end <= d_start:
                # Real corruption in the corpus -- drop, but keep a record so
                # the writeup can state exactly what was excluded.
                clip.bad_segments.append(
                    {
                        "index": idx,
                        "speaker": speaker,
                        "start": d_start,
                        "end": d_end,
                        "text": text,
                        "reason": "end <= start",
                    }
                )
                continue
            clip.segments.append(Segment(speaker, d_start, d_end, text))

        clip.stats = compute_gt_stats(clip)
        clips.append(clip)

    LOG.info(
        "parsed %d clips, %d segments, %d dropped as malformed, %d unparsable entries",
        len(clips),
        sum(len(c.segments) for c in clips),
        sum(len(c.bad_segments) for c in clips),
        total_unparsed,
    )
    return clips


def clips_to_frame(clips: Iterable[Clip]) -> pd.DataFrame:
    rows = []
    for clip in clips:
        rows.append(
            {
                "clip_id": clip.clip_id,
                "video_id": clip.video_id,
                "youtube_link": clip.youtube_link,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "requested_dur_sec": clip.duration,
                **clip.stats,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ profiling


def profile_dataset(clips: list[Clip], cfg: Config | None = None) -> dict[str, Any]:
    """Corpus-level profile; written to results/dataset_profile.json."""
    segments = [s for c in clips for s in c.segments]
    durations = [s.duration for s in segments]
    all_text = " ".join(s.text for s in segments)
    total_clip = sum(c.duration for c in clips)

    glosses = GLOSS_RE.findall(all_text)
    profile = {
        "n_rows": len(clips),
        "n_unique_video_ids": len({c.video_id for c in clips}),
        "n_segments": len(segments),
        "n_dropped_segments": sum(len(c.bad_segments) for c in clips),
        "dropped_segments": [
            {"clip_id": c.clip_id, **b} for c in clips for b in c.bad_segments
        ],
        "clip_duration_sec": {
            "min": min((c.duration for c in clips), default=0),
            "median": float(pd.Series([c.duration for c in clips]).median()),
            "max": max((c.duration for c in clips), default=0),
            "total_hours": round(total_clip / 3600, 3),
        },
        "segment_duration_sec": {
            "min": round(min(durations, default=0), 3),
            "median": round(float(pd.Series(durations).median()), 3),
            "mean": round(float(pd.Series(durations).mean()), 3),
            "max": round(max(durations, default=0), 3),
            "n_under_0.5s": sum(1 for d in durations if d < 0.5),
            "n_under_1s": sum(1 for d in durations if d < 1.0),
        },
        "speakers": {
            "label_vocabulary": dict(
                collections.Counter(s.speaker for s in segments).most_common()
            ),
            "per_clip_distribution": dict(
                sorted(collections.Counter(c.stats["n_gt_speakers"] for c in clips).items())
            ),
        },
        "overlap": {
            "total_overlap_sec": round(sum(c.stats["gt_overlap_sec"] for c in clips), 2),
            "overlap_frac_of_corpus": round(
                sum(c.stats["gt_overlap_sec"] for c in clips) / total_clip, 5
            )
            if total_clip
            else 0.0,
            "n_clips_without_overlap": sum(
                1 for c in clips if c.stats["gt_overlap_sec"] < 1e-6
            ),
            "most_overlapped": sorted(
                [(c.clip_id, c.stats["gt_overlap_frac"]) for c in clips],
                key=lambda t: -t[1],
            )[:5],
        },
        "gt_boundary": {
            "n_clips_gt_beyond_end_sec": sum(1 for c in clips if c.stats["gt_overrun_sec"] > 1e-3),
            "overrun_median_sec": round(
                float(pd.Series([c.stats["gt_overrun_sec"] for c in clips]).median()), 3
            ),
            "overrun_max_sec": round(max(c.stats["gt_overrun_sec"] for c in clips), 3),
            "worst_undershoot": sorted(
                [(c.clip_id, c.stats["gt_overrun_sec"]) for c in clips], key=lambda t: t[1]
            )[:3],
        },
        "language": {
            "script_distribution": dict(
                collections.Counter(c.stats["lang_script"] for c in clips).most_common()
            ),
            "lang_hint_distribution": dict(
                collections.Counter(c.stats["lang_hint"] for c in clips).most_common()
            ),
            "median_latin_char_frac": round(
                float(pd.Series([c.stats["latin_char_frac"] for c in clips]).median()), 4
            ),
        },
        "transcript": {
            "nonspeech_tags": dict(
                collections.Counter(NONSPEECH_TAG_RE.findall(all_text)).most_common()
            ),
            "n_tag_only_or_empty_segments": sum(1 for s in segments if not s.is_speech),
            "n_code_switch_glosses": len(glosses),
            "n_segments_with_gloss": sum(1 for s in segments if GLOSS_RE.search(s.text)),
            "gloss_bracket_styles": {
                "round": len(re.findall(r"\([^)]{0,40}\)", all_text)),
                "square": len(re.findall(r"\[[^\]]{0,40}\]", all_text)),
                "curly": len(re.findall(r"\{[^}]{0,40}\}", all_text)),
            },
            "n_unbalanced_paren_segments": sum(
                1 for s in segments if s.text.count("(") != s.text.count(")")
            ),
            "n_gloss_with_reattached_suffix": len(re.findall(r"[)\]]-\S", all_text)),
        },
    }
    if cfg is not None:
        write_json_atomic(cfg.dataset_profile, profile)
        LOG.info("dataset profile -> %s", cfg.dataset_profile)
    return profile


def load_dataset_profile(cfg: Config) -> dict | None:
    return read_json(cfg.dataset_profile)
