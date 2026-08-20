"""Step 3 -- ASR over diarized segments, producing speaker-attributed transcripts.

Two ways to turn audio plus a diarization into speaker-attributed text, and the
difference between them is a result rather than an implementation detail:

* **long-form** (`strategy="longform"`, the default) -- transcribe the whole
  clip once, then assign each recognised word to whichever diarized turn covers
  it. The ASR sees full context, and the output is a word stream with times,
  which is what WDER is defined over and what Step 4 needs in order to reason
  about the transcript.

* **per-segment** (`strategy="segment"`) -- cut the audio at the diarized turn
  boundaries and transcribe each turn separately. This is the literal reading of
  the brief and speaker attribution is free, but 33% of reference utterances in
  this corpus are under a second, so the recogniser is handed sub-second clips
  with no context. Run on a subset for comparison.

**No language hint ever reaches a recogniser.** The corpus spans nine scripts
and `ref_lang_script` / `ref_lang_hint` are derived from the ground-truth
transcript, so passing either would hand the system free language ID that the
brief reserves for scoring. Sarvam is called with `language_code="unknown"`,
Whisper with `language=None`; both detect it themselves, and what they detect is
recorded so it can be scored as a byproduct.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from .config import Config, StageFlags
from .data import ClipInput, Turn
from .utils import (LOG, apply_selection, human_time, now_utc_iso, read_json,
                    write_json_atomic)

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saaras:v3"
# The REST endpoint is documented for "quick responses under 30 seconds", so
# long clips are split. 28 s leaves headroom for the boundary padding below.
SARVAM_CHUNK_SEC = 28.0
# Chunks are cut on a fixed grid, which will land mid-word. Each request gets a
# little extra audio on both sides and the words recovered from the padding are
# dropped, so a word straddling a cut is transcribed in at least one chunk with
# its context intact.
CHUNK_PAD_SEC = 1.0


@dataclass(frozen=True)
class Word:
    """One recognised word with its clip-relative time span."""

    text: str
    start: float
    end: float


# ------------------------------------------------------------------ backends


def _load_audio(path: str | os.PathLike, sr: int = 16_000):
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != sr:
        raise ValueError(f"{path} is {rate} Hz, expected {sr}")
    return audio, rate


def transcribe_whisper(cfg: Config, wav: Path, model_size: str = "large-v3") -> tuple[list[Word], dict]:
    """faster-whisper with word timestamps.

    Deliberately not WhisperX: it pins an older pyannote and would fight the
    environment Step 2 needs, while the only part of it we want -- assigning
    words to speakers -- is `assign_words()` below.
    """
    from faster_whisper import WhisperModel

    key = ("whisper", model_size)
    if key not in _MODEL_CACHE:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        LOG.info("loading faster-whisper %s on %s (%s)", model_size, device, compute)
        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute)
    model = _MODEL_CACHE[key]

    segments, info = model.transcribe(
        str(wav),
        language=None,          # detect; never hinted from the reference
        word_timestamps=True,
        vad_filter=False,       # diarization already decides what is speech
    )
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if text:
                words.append(Word(text, float(w.start), float(w.end)))
    meta = {"detected_language": info.language,
            "language_probability": round(float(info.language_probability), 4)}
    return words, meta


def transcribe_sarvam(cfg: Config, wav: Path, model: str = SARVAM_MODEL) -> tuple[list[Word], dict]:
    """Sarvam Saaras over the REST endpoint, chunked to the documented limit."""
    import soundfile as sf

    key = resolve_sarvam_key(cfg)
    audio, sr = _load_audio(wav)
    total = len(audio) / sr
    words: list[Word] = []
    languages: list[str] = []
    granularity = "none"

    starts = [s for s in _frange(0.0, total, SARVAM_CHUNK_SEC)]
    for c_start in starts:
        c_end = min(total, c_start + SARVAM_CHUNK_SEC)
        a = max(0.0, c_start - CHUNK_PAD_SEC)
        b = min(total, c_end + CHUNK_PAD_SEC)
        chunk = audio[int(a * sr):int(b * sr)]
        if not len(chunk):
            continue
        buf = cfg.work_dir / f"sarvam_{wav.stem}_{int(a)}.wav"
        buf.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(buf), chunk, sr, subtype="PCM_16")
        try:
            payload = _sarvam_post(buf, key, model)
        finally:
            buf.unlink(missing_ok=True)

        if payload.get("language_code"):
            languages.append(payload["language_code"])
        got, gran = _sarvam_words(payload, offset=a)
        granularity = gran if granularity == "none" else granularity
        # Keep only what falls inside the unpadded window, so the overlap added
        # for context does not produce the same word twice.
        words.extend(w for w in got if c_start - 1e-6 <= w.start < c_end)

    words.sort(key=lambda w: (w.start, w.end))
    meta = {"detected_language": max(set(languages), key=languages.count) if languages else None,
            "languages_seen": sorted(set(languages)),
            "n_chunks": len(starts), "timestamp_granularity": granularity}
    return words, meta


def _sarvam_post(path: Path, key: str, model: str, retries: int = 4) -> dict:
    import requests

    last = None
    for attempt in range(retries):
        try:
            with open(path, "rb") as fh:
                r = requests.post(
                    SARVAM_URL,
                    headers={"api-subscription-key": key},
                    files={"file": (path.name, fh, "audio/wav")},
                    # "unknown" asks the API to detect the language. Passing a
                    # real code here would be a ground-truth leak.
                    data={"model": model, "language_code": "unknown",
                          "with_timestamps": "true"},
                    timeout=180,
                )
            if r.status_code == 200:
                return r.json()
            # 429 and 5xx are worth retrying; 4xx otherwise is not.
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"sarvam {r.status_code}: {r.text[:300]}")
            last = f"{r.status_code}: {r.text[:200]}"
        except Exception as exc:  # noqa: BLE001 - network layer
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"sarvam failed after {retries} attempts -- {last}")


def _sarvam_words(payload: dict, offset: float) -> tuple[list[Word], str]:
    """Words from a Saaras response, whichever granularity it returned.

    The documentation says chunk-level while the response schema names the field
    `words`, so this reads whatever is actually there and records which it was:
    if the entries are multi-word the timings are phrase-level, and word->speaker
    assignment is correspondingly coarse. That distinction is reported rather
    than smoothed over, because it changes what WDER means for this system.
    """
    ts = payload.get("timestamps") or {}
    toks = ts.get("words") or []
    starts = ts.get("start_time_seconds") or []
    ends = ts.get("end_time_seconds") or []
    if toks and len(toks) == len(starts) == len(ends):
        gran = "word" if all(len(str(t).split()) == 1 for t in toks[:50]) else "phrase"
        out = []
        for t, s, e in zip(toks, starts, ends):
            t = str(t).strip()
            if not t:
                continue
            out.append(Word(t, float(s) + offset, float(e) + offset))
        return out, gran
    # No timings at all: fall back to spreading the transcript across the chunk,
    # which is honest but useless for WDER -- flagged so it cannot be mistaken.
    text = (payload.get("transcript") or "").strip()
    if not text:
        return [], "none"
    toks = text.split()
    return [Word(t, offset, offset) for t in toks], "transcript-only"


_MODEL_CACHE: dict = {}

BACKENDS: dict[str, Callable[..., tuple[list[Word], dict]]] = {
    "whisper-large-v3": lambda cfg, wav: transcribe_whisper(cfg, wav, "large-v3"),
    "sarvam-saaras-v3": lambda cfg, wav: transcribe_sarvam(cfg, wav, SARVAM_MODEL),
}


def resolve_sarvam_key(cfg: Config) -> str:
    """Same ladder as the HF token: explicit, .env, environment, platform vault."""
    from .utils import load_dotenv

    load_dotenv(cfg=cfg)
    key = os.environ.get("SARVAM_API_KEY")
    if key:
        return key
    try:  # Kaggle
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("SARVAM_API_KEY")
    except Exception:  # noqa: BLE001
        pass
    try:  # Colab
        from google.colab import userdata  # type: ignore

        return userdata.get("SARVAM_API_KEY")
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(
        "SARVAM_API_KEY not found. Put it in .env at the Drive/working root, "
        "or add it to Kaggle Secrets / Colab Secrets."
    )


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    x = start
    while x < stop - 1e-9:
        yield x
        x += step


# ------------------------------------------------- words -> speaker attribution


def assign_words(words: Sequence[Word], turns: Sequence[Turn]) -> list[tuple[str, str]]:
    """Attach each recognised word to the diarized turn it overlaps most.

    Ties and words falling in a gap go to the nearest turn by midpoint, so no
    word is silently dropped -- a dropped word would look like an ASR deletion
    and quietly flatter WDER, whose denominator counts only aligned words.

    Where two speakers overlap, the word goes to whichever covers more of it.
    That is the "dominant speaker wins" strategy: in overlapped speech only one
    speaker's words can be recovered from a single transcript, so the other's
    are structurally lost. 12.8% of reference words in this corpus lie in
    overlapped speech, which is the size of the ceiling this imposes.
    """
    if not turns:
        return [(w.text, "SPEAKER_UNKNOWN") for w in words]
    out: list[tuple[str, str]] = []
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(w.end, t.end) - max(w.start, t.start)
            if ov > best_ov:
                best, best_ov = t.speaker, ov
        if best is None:
            mid = (w.start + w.end) / 2.0
            best = min(turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid))).speaker
        out.append((w.text, best))
    return out


def speaker_texts_from_words(pairs: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """Per-speaker word lists in time order -- the cpWER input, hypothesis side."""
    out: dict[str, list[str]] = {}
    for word, spk in pairs:
        out.setdefault(spk, []).append(word)
    return out


# ------------------------------------------------------------------- runner


def is_done(cfg: Config, system: str, clip_id: str) -> bool:
    """A transcript counts as complete only if its sidecar agrees with it."""
    p = asr_path(cfg, system, clip_id)
    if not p.exists():
        return False
    try:
        payload = read_json(p)
    except Exception:  # noqa: BLE001
        return False
    return payload.get("status") == "ok" and isinstance(payload.get("words"), list)


def asr_path(cfg: Config, system: str, clip_id: str) -> Path:
    return cfg.root / "asr" / system / f"{clip_id}.json"


def run(cfg: Config, inputs: Sequence[ClipInput], flags: StageFlags | None = None,
        systems: Sequence[str] | None = None) -> pd.DataFrame:
    """Transcribe every clip with every system. ClipInput only -- no reference."""
    flags = flags or StageFlags()
    systems = list(systems or BACKENDS)
    clips = apply_selection(list(inputs), flags)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for system in systems:
        if system not in BACKENDS:
            raise KeyError(f"unknown ASR system {system!r}; have {sorted(BACKENDS)}")
        done = skipped = failed = 0
        for i, clip in enumerate(clips, 1):
            prefix = f"[{system} {i}/{len(clips)}] {clip.clip_id}"
            if is_done(cfg, system, clip.clip_id) and not flags.force_redo:
                skipped += 1
                rows.append(read_json(asr_path(cfg, system, clip.clip_id)) | {"status": "skipped"})
                continue
            wav = Path(clip.wav_path or cfg.wav_path(clip.clip_id))
            if not wav.exists():
                LOG.warning("%s  no audio at %s", prefix, wav)
                failed += 1
                continue
            try:
                started = time.perf_counter()
                words, meta = BACKENDS[system](cfg, wav)
                elapsed = time.perf_counter() - started
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s  FAILED %s: %s", prefix, type(exc).__name__, exc)
                failed += 1
                continue

            payload = {
                "clip_id": clip.clip_id, "system": system, "status": "ok",
                "words": [{"t": w.text, "s": round(w.start, 3), "e": round(w.end, 3)}
                          for w in words],
                "n_words": len(words),
                "elapsed_sec": round(elapsed, 2),
                "rtf": round(elapsed / clip.duration, 4) if clip.duration else None,
                "clip_dur_sec": clip.duration,
                "transcribed_at_utc": now_utc_iso(),
                **meta,
            }
            write_json_atomic(asr_path(cfg, system, clip.clip_id), payload)
            rows.append(payload)
            done += 1
            LOG.info("%s  %d words  %.1fs  rtf %.3f  lang=%s", prefix, len(words),
                     elapsed, payload["rtf"] or 0, meta.get("detected_language"))
        LOG.info("%s: %d ok, %d skipped, %d failed", system, done, skipped, failed)

    LOG.info("step 3 done in %s", human_time(time.perf_counter() - t0))
    return pd.DataFrame(rows)


def load_words(cfg: Config, system: str, clip_id: str) -> list[Word]:
    payload = read_json(asr_path(cfg, system, clip_id))
    return [Word(w["t"], w["s"], w["e"]) for w in payload.get("words", [])]


# ------------------------------------------------------------------ scoring


def score_all(cfg: Config, references: dict, systems: Sequence[str],
              diar_models: Sequence[str], clip_ids: Sequence[str] | None = None,
              normalize=None) -> pd.DataFrame:
    """Score every (ASR system x diarization model) pairing on every clip.

    The pairing is the point: cpWER and WDER measure the joint system, so an ASR
    is only as good as the diarization it is attributed with, and the table has
    to show both axes rather than collapsing one.

    `normalize` defaults to reference.normalize_text with gloss stripping OFF.
    It is the SAME function the reference went through, which is what makes the
    comparison fair; gloss stripping is disabled because hypotheses contain no
    code-switch glosses, so applying it would be a no-op that only risks
    diverging the two paths.
    """
    from . import diarization, reference as refmod
    from . import text_metrics as tm

    if normalize is None:
        def normalize(text: str) -> str:
            return refmod.normalize_text(text, strip_gloss=False)

    rows: list[dict] = []
    ids = list(clip_ids or references)
    for system in systems:
        for model in diar_models:
            for cid in ids:
                ref = references.get(cid)
                if ref is None or not is_done(cfg, system, cid):
                    continue
                if not diarization.is_done(cfg, model, cid):
                    continue
                words = load_words(cfg, system, cid)
                # Normalise per word, then re-split: a normaliser can turn one
                # token into several (punctuation inside a word) and the word
                # stream must stay flat for WDER.
                turns = diarization.load_hypothesis(cfg, model, cid)
                pairs: list[tuple[str, str]] = []
                for word, spk in assign_words(words, turns):
                    for tok in normalize(word).split():
                        pairs.append((tok, spk))

                hyp_texts = speaker_texts_from_words(pairs)
                ref_texts = {k: v.split() for k, v in refmod.speaker_texts(ref).items()}
                ref_stream = refmod.word_stream(ref)
                row = tm.score_transcript(ref_texts, hyp_texts, ref_stream, pairs)
                row.update({"clip_id": cid, "asr_system": system, "diar_model": model,
                            "n_hyp_words": len(pairs),
                            "n_hyp_speakers": len(hyp_texts),
                            "n_ref_speakers": len(ref_texts)})
                rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(cfg.step3_metrics_csv, index=False)
    return df


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    """Corpus figures per (ASR system, diarization model), pooled not averaged."""
    from . import text_metrics as tm

    if not len(metrics):
        return pd.DataFrame()
    out = []
    for (system, model), grp in metrics.groupby(["asr_system", "diar_model"]):
        out.append({"asr_system": system, "diar_model": model,
                    **tm.summarise(grp.to_dict("records"))})
    return pd.DataFrame(out).sort_values("cpwer").reset_index(drop=True)


def oracle_ceiling(cfg: Config, references: dict, clip_ids: Sequence[str] | None = None) -> dict:
    """What a PERFECT ASR and a PERFECT diarization still score, and why.

    Feeds the reference transcript back as if it were recognised, with word
    times spread evenly across each utterance, and attributes it against the
    reference diarization. Every word is correct and every turn is correct, so
    plain WER is 0 by construction -- but cpWER and WDER are not, because
    `assign_words` must give an overlapped instant to ONE speaker and a single
    transcript cannot carry two people talking at once.

    The number this returns is therefore the floor of the long-form strategy on
    this corpus: no ASR and no diarizer can beat it without separating
    overlapped speech. Reporting a cpWER without it invites reading a structural
    limit as a system failure.

    Uses the reference, so it is a scoring-side diagnostic and never part of the
    pipeline -- the same standing as the oracle speaker-count ablation.
    """
    from . import reference as refmod
    from . import text_metrics as tm

    rows = []
    for cid in (clip_ids or references):
        ref = references[cid]
        pairs: list[tuple[str, str]] = []
        words: list[Word] = []
        for utt in sorted(ref.utterances, key=lambda u: u.start):
            toks = refmod.tokenize(utt.text_norm)
            if not toks:
                continue
            step = (utt.end - utt.start) / len(toks)
            words.extend(Word(w, utt.start + i * step, utt.start + (i + 1) * step)
                         for i, w in enumerate(toks))
        words.sort(key=lambda w: w.start)
        pairs = assign_words(words, ref.turns)
        row = tm.score_transcript(
            {k: v.split() for k, v in refmod.speaker_texts(ref).items()},
            speaker_texts_from_words(pairs),
            refmod.word_stream(ref), pairs)
        row["clip_id"] = cid
        row["overlap_frac"] = ref.stats.get("overlap_frac", 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    zero = df[df.overlap_frac < 1e-9]
    out = {"n_clips": len(df), **tm.summarise(df.to_dict("records"))}
    # Not zero even for a perfect system: see the ordering caveat in
    # text_metrics.di_cpwer. Named for what it is, not for a hope.
    out["wer_speaker_agnostic"] = out["wer"]
    out["n_clips_no_overlap"] = len(zero)
    out["cpwer_no_overlap_clips"] = (
        zero["cperrors"].sum() / zero["cpn_ref_words"].sum() if len(zero) else float("nan"))
    return out
