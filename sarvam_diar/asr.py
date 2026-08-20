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


def transcribe_whisper(cfg: Config, wav: Path, model_size: str = "large-v3",
                       word_timestamps: bool = True, beam_size: int = 5,
                       condition_on_previous_text: bool = True,
                       batch_size: int = 0) -> tuple[list[Word], dict]:
    """faster-whisper with word timestamps.

    Deliberately not WhisperX: it pins an older pyannote and would fight the
    environment Step 2 needs, while the only part of it we want -- assigning
    words to speakers -- is `assign_words()` below.
    """
    from faster_whisper import WhisperModel

    # Default: the diarizer decides what is speech, and Whisper transcribes
    # everything it is given. Only the batched path overrides this.
    vad_filter = False

    key = ("whisper", model_size)
    if key not in _MODEL_CACHE:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        t0 = time.perf_counter()
        LOG.info("loading faster-whisper %s on %s (%s) -- the first call also "
                 "downloads ~3 GB", model_size, device, compute)
        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute)
        LOG.info("model ready in %.0fs", time.perf_counter() - t0)
    model = _MODEL_CACHE[key]

    if batch_size and batch_size > 1:
        # BatchedInferencePipeline refuses to run without either vad_filter=True
        # or explicit clip_timestamps (in SAMPLES), and a single whole-file span
        # would collapse to one chunk and defeat the batching anyway. So
        # batching here necessarily means letting Whisper's own Silero VAD
        # decide where speech is.
        #
        # That is a second voice-activity decision on top of the diarizer's, and
        # anything it drops is speech no downstream stage can recover -- it
        # becomes a miss attributable to the ASR rather than the diarization we
        # are trying to measure. It is therefore opt-in, never the default, and
        # the cost is recorded in the meta so a run using it is identifiable.
        from faster_whisper import BatchedInferencePipeline

        bkey = ("whisper-batched", model_size)
        if bkey not in _MODEL_CACHE:
            _MODEL_CACHE[bkey] = BatchedInferencePipeline(model=model)
        model = _MODEL_CACHE[bkey]
        vad_filter = True

    # Whisper pads every input to a fixed 30 s window, so a 1 s segment costs
    # the same encoder pass as a 30 s one -- per-segment work is dominated by
    # call count, not audio length. beam_size=5 then multiplies the decoder work
    # on top for no benefit on short fragments, and conditioning on previous
    # text is meaningless when each call is an independent segment (and is a
    # known cause of repetition loops). Both are dropped for that path.
    segments, info = model.transcribe(
        str(wav),
        language=None,          # detect; never hinted from the reference
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        beam_size=beam_size,
        **({"batch_size": batch_size} if batch_size and batch_size > 1
           else {"condition_on_previous_text": condition_on_previous_text}),
    )
    words: list[Word] = []
    for seg in segments:
        if word_timestamps and seg.words:
            for w in seg.words:
                text = w.word.strip()
                if text:
                    words.append(Word(text, float(w.start), float(w.end)))
        else:
            # Per-segment work needs only the text, and forcing the extra
            # alignment pass on every one of thousands of short segments is
            # pure cost. Words carry the segment span so the shape is uniform.
            for tok in (seg.text or "").split():
                words.append(Word(tok, float(seg.start), float(seg.end)))
    meta = {"detected_language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "beam_size": beam_size, "batched": bool(batch_size and batch_size > 1),
            "whisper_vad": vad_filter}
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


def _sarvam_post(path: Path, key: str, model: str, retries: int = 7) -> dict:
    """POST one file, backing off on rate limits.

    429 is the normal steady state at any useful concurrency, not an
    exceptional event, so it gets a real backoff: the server's Retry-After when
    offered, otherwise exponential with jitter. The jitter matters because
    without it every worker in the pool retries on the same schedule and they
    collide again on each round.
    """
    import random
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
            # 429 and 5xx are transient; any other 4xx is a request the server
            # will refuse identically forever. Raising it inside the try meant
            # the except below caught it and retried a 400 four times, turning a
            # fast failure into a slow one.
            if r.status_code != 429 and r.status_code < 500:
                raise _Fatal(f"sarvam {r.status_code}: {r.text[:300]}")
            last = f"{r.status_code}: {r.text[:200]}"
            wait = float(r.headers.get("Retry-After") or 0) or None
        except _Fatal:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer
            last = f"{type(exc).__name__}: {exc}"
            wait = None
        # 1.5, 3, 6, 12, 24, 48 s plus jitter, unless the server named a delay.
        time.sleep(wait if wait else min(60.0, 1.5 * (2 ** attempt)) * (0.5 + random.random()))
    raise RuntimeError(f"sarvam failed after {retries} attempts -- {last}")


class _Fatal(RuntimeError):
    """A response the server will refuse identically no matter how often asked."""


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
    # Long-form, greedy, unbatched: 99 model calls rather than one per segment,
    # and no second VAD deciding what counts as speech. Batching would need
    # Whisper's own VAD (see transcribe_whisper), so it stays opt-in.
    #
    # large-v3 was measured at RTF ~0.5 on a T4 -- about 6.6 h for this corpus,
    # which does not fit. large-v3-turbo is the practical default: distilled
    # from large-v3 with 4 decoder layers instead of 32, and decoding is where
    # long-form time goes. Crucially it stays MULTILINGUAL, unlike the distil-*
    # models, which are English-only and therefore useless on a corpus spanning
    # nine Indic scripts. The cost is a small accuracy loss against large-v3;
    # both are kept so the trade can be measured rather than assumed.
    "whisper-large-v3-turbo": lambda cfg, wav: transcribe_whisper(
        cfg, wav, "large-v3-turbo", word_timestamps=True, beam_size=1),
    "whisper-large-v3": lambda cfg, wav: transcribe_whisper(
        cfg, wav, "large-v3", word_timestamps=True, beam_size=1),
    "whisper-large-v3-batched": lambda cfg, wav: transcribe_whisper(
        cfg, wav, "large-v3", word_timestamps=True, beam_size=1, batch_size=8),
    "sarvam-saaras-v3": lambda cfg, wav: transcribe_sarvam(cfg, wav, "saaras:v3"),
    "sarvam-saaras-v4": lambda cfg, wav: transcribe_sarvam(cfg, wav, "saaras:v4"),
}


def resolve_sarvam_key(cfg: Config) -> str:
    """Same ladder as the HF token: explicit, .env, environment, platform vault."""
    from .utils import load_dotenv

    # `extra` is where a Drive/Kaggle root .env lives -- the repo clone cannot
    # carry one, since .env is gitignored so the key stays out of a public repo.
    load_dotenv(extra=[cfg.dotenv_path, cfg.root / ".env"])
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


# ------------------------------------------------- per-segment (strategy "A")
#
# Forced rather than chosen for Sarvam. Saaras returns exactly ONE timestamp
# span per request -- measured across v3 and v4, 5 s and 20 s inputs, and every
# `mode` -- so there are no word timings to attribute with, and the long-form
# path is unavailable. Cutting the audio at the diarized turn boundaries instead
# makes attribution exact by construction: every word of a request belongs to
# that turn's speaker.
#
# The cost is context. On this corpus 46% of community-1's turns are under a
# second, which is close to useless to a recogniser, so adjacent same-speaker
# turns are merged first. reverb-v2 needs it least -- 2,833 segments at a median
# of 6.46 s against community-1's 14,380 at 1.21 s.


def sarvam_model_for(system: str) -> str:
    """Map a system key to the API model id, so v3 and v4 are separate systems.

    Keeping them as distinct system keys rather than a parameter means their
    checkpoints, metrics rows and explorer columns never collide, and a sweep
    of one cannot be silently attributed to the other.
    """
    if system.endswith("v4") or ":v4" in system:
        return "saaras:v4"
    return SARVAM_MODEL


def merge_same_speaker(turns: Sequence[Turn], gap: float = 1.0) -> list[Turn]:
    """Join consecutive turns of one speaker separated by at most `gap`.

    Purely an ASR segmentation choice; the diarization being scored is
    untouched. Diarizers emit utterance-level fragments, and handing a
    recogniser a 0.3 s fragment throws away the context it needs.
    """
    out: list[Turn] = []
    for t in sorted(turns, key=lambda x: x.start):
        if out and out[-1].speaker == t.speaker and t.start - out[-1].end <= gap:
            out[-1] = Turn(start=out[-1].start, end=max(out[-1].end, t.end),
                           speaker=t.speaker)
        else:
            out.append(Turn(start=t.start, end=t.end, speaker=t.speaker))
    return out


# Hard server limit, not a guideline: over this the API answers 400 with
# "Audio duration exceeds the maximum limit of 30 seconds."
SARVAM_MAX_SEC = 29.0


def _sarvam_text(cfg: Config, key: str, wav: Path, model: str) -> tuple[str, str | None]:
    """Transcribe one segment, splitting it if it exceeds the server's limit.

    Splitting is safe here in a way it would not be for the long-form path: the
    whole segment belongs to one diarized speaker, so every sub-chunk does too,
    and concatenating their transcripts cannot mix speakers. Sub-chunks overlap
    slightly and the pieces are joined in order, which risks a duplicated word
    at a seam -- accepted, because the alternative is dropping one.
    """
    import soundfile as sf

    info = sf.info(str(wav))
    if info.duration <= SARVAM_MAX_SEC:
        payload = _sarvam_post(wav, key, model)
        return (payload.get("transcript") or "").strip(), payload.get("language_code")

    audio, sr = sf.read(str(wav), dtype="float32")
    texts: list[str] = []
    langs: list[str] = []
    step = SARVAM_MAX_SEC - 0.5
    n = 0
    while n * step < info.duration:
        a = n * step
        b = min(info.duration, a + SARVAM_MAX_SEC)
        piece = audio[int(a * sr):int(b * sr)]
        n += 1
        if len(piece) < int(0.3 * sr):
            break
        tmp = wav.with_name(f"{wav.stem}_p{n}.wav")
        sf.write(str(tmp), piece, sr, subtype="PCM_16")
        try:
            payload = _sarvam_post(tmp, key, model)
        finally:
            tmp.unlink(missing_ok=True)
        t = (payload.get("transcript") or "").strip()
        if t:
            texts.append(t)
        if payload.get("language_code"):
            langs.append(payload["language_code"])
        if b >= info.duration:
            break
    lang = max(set(langs), key=langs.count) if langs else None
    return " ".join(texts), lang


def transcribe_segments(cfg: Config, system: str, wav: Path, turns: Sequence[Turn],
                        min_dur: float = 0.30, workers: int = 4) -> tuple[list[dict], dict]:
    """Transcribe each turn separately. Returns per-segment rows plus meta.

    Segments shorter than `min_dur` are skipped rather than sent: they were
    measured to come back empty anyway, and each one still costs a request and a
    spurious language guess. They are counted so the skip is visible.
    """
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor

    audio, sr = _load_audio(wav)
    key = resolve_sarvam_key(cfg) if system.startswith("sarvam") else None
    model = sarvam_model_for(system)

    def one(idx_turn):
        idx, t = idx_turn
        dur = t.end - t.start
        if dur < min_dur:
            return {"i": idx, "start": t.start, "end": t.end, "speaker": t.speaker,
                    "text": "", "skipped": "too_short", "lang": None}
        seg = audio[int(t.start * sr):int(t.end * sr)]
        if not len(seg):
            return {"i": idx, "start": t.start, "end": t.end, "speaker": t.speaker,
                    "text": "", "skipped": "empty", "lang": None}
        buf = cfg.work_dir / f"seg_{wav.stem}_{idx}.wav"
        buf.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(buf), seg, sr, subtype="PCM_16")
        try:
            if system.startswith("sarvam"):
                text, lang = _sarvam_text(cfg, key, buf, model)
            else:
                size = "large-v3-turbo" if "turbo" in system else "large-v3"
                words, meta = transcribe_whisper(
                    cfg, buf, model_size=size, word_timestamps=False, beam_size=1,
                    condition_on_previous_text=False)
                text, lang = " ".join(w.text for w in words), meta.get("detected_language")
            return {"i": idx, "start": t.start, "end": t.end, "speaker": t.speaker,
                    "text": text, "skipped": None, "lang": lang}
        finally:
            buf.unlink(missing_ok=True)

    items = list(enumerate(turns))
    # Only the API backend benefits from concurrency; a local GPU model is
    # already saturated by one worker and threads would just contend.
    n_workers = workers if system.startswith("sarvam") else 1
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        rows = list(pool.map(one, items))
    rows.sort(key=lambda r: r["i"])

    langs = [r["lang"] for r in rows if r["lang"]]
    meta = {"n_segments": len(rows),
            "n_skipped_short": sum(1 for r in rows if r["skipped"] == "too_short"),
            "detected_language": max(set(langs), key=langs.count) if langs else None,
            "languages_seen": sorted(set(langs))}
    return rows, meta


def run_segmented(cfg: Config, inputs: Sequence[ClipInput], diar_model: str,
                  systems: Sequence[str], flags: StageFlags | None = None,
                  merge_gap: float = 1.0, workers: int = 4) -> pd.DataFrame:
    """Per-segment ASR over one diarization's turns, checkpointed per clip."""
    from . import diarization

    flags = flags or StageFlags()
    clips = apply_selection(list(inputs), flags)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for system in systems:
        tag = f"{system}@{diar_model}"
        done = skipped = failed = 0
        for i, clip in enumerate(clips, 1):
            prefix = f"[{tag} {i}/{len(clips)}] {clip.clip_id}"
            if is_done(cfg, tag, clip.clip_id) and not flags.force_redo:
                skipped += 1
                continue
            if not diarization.is_done(cfg, diar_model, clip.clip_id):
                continue
            wav = Path(clip.wav_path or cfg.wav_path(clip.clip_id))
            if not wav.exists():
                failed += 1
                continue
            turns = merge_same_speaker(
                diarization.load_hypothesis(cfg, diar_model, clip.clip_id), merge_gap)
            try:
                started = time.perf_counter()
                segs, meta = transcribe_segments(cfg, system, wav, turns, workers=workers)
                elapsed = time.perf_counter() - started
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s  FAILED %s: %s", prefix, type(exc).__name__, exc)
                failed += 1
                continue
            payload = {"clip_id": clip.clip_id, "system": tag, "status": "ok",
                       "strategy": "segment", "diar_model": diar_model,
                       "merge_gap": merge_gap, "segments": segs,
                       "words": [],  # keeps the is_done() contract uniform
                       "n_words": sum(len(s["text"].split()) for s in segs),
                       "elapsed_sec": round(elapsed, 2),
                       "rtf": round(elapsed / clip.duration, 4) if clip.duration else None,
                       "clip_dur_sec": clip.duration,
                       "transcribed_at_utc": now_utc_iso(), **meta}
            write_json_atomic(asr_path(cfg, tag, clip.clip_id), payload)
            rows.append(payload)
            done += 1
            LOG.info("%s  %d segs, %d words  %.1fs  rtf %.3f  lang=%s", prefix,
                     len(segs), payload["n_words"], elapsed, payload["rtf"] or 0,
                     meta.get("detected_language"))
        LOG.info("%s: %d ok, %d skipped, %d failed", tag, done, skipped, failed)

    LOG.info("step 3 (segmented) done in %s", human_time(time.perf_counter() - t0))
    return pd.DataFrame(rows)


def load_pairs(cfg: Config, system: str, clip_id: str, normalize=None) -> list[tuple[str, str]]:
    """(word, speaker) stream from either strategy, so scoring is one code path.

    Per-segment runs already carry the speaker on every segment, so attribution
    is exact and no time-overlap assignment is involved.
    """
    from . import reference as refmod

    if normalize is None:
        def normalize(text):
            return refmod.normalize_text(text, strip_gloss=False)

    payload = read_json(asr_path(cfg, system, clip_id))
    if payload.get("strategy") == "segment":
        out: list[tuple[str, str]] = []
        for seg in sorted(payload["segments"], key=lambda s: s["start"]):
            for tok in normalize(seg["text"]).split():
                out.append((tok, seg["speaker"]))
        return out
    raise ValueError(f"{clip_id}: not a segmented run; use load_words + assign_words")
