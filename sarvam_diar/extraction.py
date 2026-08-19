"""Step 1 - YouTube audio -> exact-length 16 kHz mono WAV, checkpointed to Drive.

Two things this module is strict about:

1. **Boundaries.** `[start_sec, end_sec]` is a hard requirement. Every published
   WAV holds exactly ``round((end_sec - start_sec) * 16000)`` samples, so each
   clip's timeline is exactly ``[0, duration]`` and ground-truth timestamps
   (which are relative to `start_sec`) index into it with no offset arithmetic
   anywhere downstream. `DUR_TOL_SEC` is a *diagnostic* that decides whether a
   fetch is trustworthy enough to keep -- it never defines the output boundary.

2. **Checkpoint truth.** A file existing on Drive proves nothing; a Colab
   runtime can die mid-copy. A clip counts as done only when its sidecar exists
   (written last, so it is the commit marker) and the WAV probes as 16 kHz mono
   pcm_s16le with the exact expected sample count.
"""

from __future__ import annotations

import os
import random
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BACKOFF_BASE_SEC,
    CHANNELS,
    KNOWN_UNAVAILABLE,
    SAMPLE_RATE,
    SUBPROCESS_TIMEOUT_SEC,
    CLIENT_LADDER,
    Config,
    StageFlags,
)
from .data import Clip
from .utils import (
    LOG,
    append_jsonl,
    atomic_publish,
    clear_dir,
    ffprobe_audio,
    human_bytes,
    human_time,
    now_utc_iso,
    read_json,
    run_cmd,
    tail,
    tool_version,
    write_json_atomic,
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
FIELD_SEP = "|||"

# Distinguishes "caller passed None on purpose" from "caller passed nothing".
_UNSET = object()

RESULT_COLUMNS = [
    "clip_id", "video_id", "youtube_link", "start_sec", "end_sec", "requested_dur_sec",
    "status", "permanent", "wav_path", "n_samples", "n_expected_samples",
    "raw_delta_sec", "pad_samples", "trim_samples", "sample_rate", "channels",
    "file_size_bytes", "source_dur_sec", "source_title", "ytdlp_format_id",
    "download_tier", "player_client", "attempts", "elapsed_sec",
    "n_gt_segments", "n_gt_speakers", "gt_min_start", "gt_max_end", "gt_overrun_sec",
    "gt_speech_sec", "gt_speaker_time_sec", "gt_overlap_sec", "gt_overlap_frac",
    "n_bad_gt_segments", "ref_lang_script", "ref_lang_hint",
    "error_class", "error_msg", "extracted_at_utc",
]

# stderr substring -> error class. Ordered: first match wins.
_ERROR_PATTERNS: list[tuple[tuple[str, ...], str, bool]] = [
    (("video unavailable", "has been removed", "account associated", "terminated",
      "private video", "video has been removed", "this video is unavailable",
      "not available on this app"), "video_unavailable", True),
    (("sign in to confirm your age", "age-restricted", "inappropriate for some users"),
     "age_restricted", False),
    (("confirm you're not a bot", "confirm you are not a bot", "sign in to confirm",
      "cookies", "captcha"), "bot_check", False),
    (("not available in your country", "geo restricted", "geo-restricted",
      "blocked it in your country"), "geo_blocked", True),
    (("requested format is not available", "no video formats", "only images are available"),
     "no_audio_format", False),
    (("members-only", "join this channel", "paid", "purchase"), "members_only", True),
    (("live event will begin", "premieres in", "is live"), "live_or_upcoming", False),
    # A 403 on the media URL means the chosen player client's format needs a PO
    # token -- another rung of the client ladder usually fixes it, so it is
    # retryable and deliberately classified apart from generic network errors.
    (("403", "forbidden", "ffmpeg exited with code 8", "page needs to be reloaded"),
     "format_forbidden", False),
    (("http error 4", "http error 5", "unable to download", "connection", "timed out",
      "timeout", "temporary failure", "resolve host", "ssl", "network"), "network", False),
]


# ------------------------------------------------------------------- geometry


def expected_samples(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> int:
    """The one definition of clip length. Used for writing AND for validation."""
    return int(round(duration_sec * sample_rate))


def classify_error(message: str | None) -> tuple[str, bool]:
    """Map a yt-dlp/ffmpeg stderr blob to (error_class, is_permanent)."""
    if not message:
        return "unknown", False
    low = message.lower()
    for needles, name, permanent in _ERROR_PATTERNS:
        if any(n in low for n in needles):
            return name, permanent
    return "unknown", False


# ------------------------------------------------------------------ wav layer
# stdlib `wave` is used rather than soundfile/librosa: the format is entirely
# under our control (mono pcm_s16le), and this keeps Step 1 dependency-free
# apart from yt-dlp and ffmpeg.


def read_wav_mono16(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit")
        rate, channels = wf.getframerate(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype("<i2")
    return audio, rate


def write_wav_mono16(path: str | os.PathLike, audio: np.ndarray, rate: int = SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.astype("<i2", copy=False).tobytes())
    return path


def force_exact_length(path: str | os.PathLike, n_expected: int) -> dict[str, Any]:
    """Pad with silence or truncate so the file holds exactly n_expected samples.

    Returns the diagnostics recorded in the results CSV. In a healthy run
    pad/trim are 0 or a handful of samples (one AAC frame is ~368 samples at
    16 kHz); a large value is a signal that the fetch drifted.
    """
    audio, rate = read_wav_mono16(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {rate}")
    raw_samples = int(audio.shape[0])
    pad = trim = 0
    if raw_samples < n_expected:
        pad = n_expected - raw_samples
        audio = np.concatenate([audio, np.zeros(pad, dtype="<i2")])
    elif raw_samples > n_expected:
        trim = raw_samples - n_expected
        audio = audio[:n_expected]
    write_wav_mono16(path, audio, SAMPLE_RATE)
    return {
        "raw_samples": raw_samples,
        "raw_delta_sec": round((raw_samples - n_expected) / SAMPLE_RATE, 4),
        "pad_samples": pad,
        "trim_samples": trim,
        "n_samples": int(audio.shape[0]),
    }


def read_wav_window(path: str | os.PathLike, start: float, end: float) -> np.ndarray:
    """Slice [start, end) seconds out of a published clip (alignment spot-checks)."""
    audio, rate = read_wav_mono16(path)
    lo = max(0, int(round(start * rate)))
    hi = min(audio.shape[0], int(round(end * rate)))
    return audio[lo:hi] if hi > lo else audio[:0]


# ---------------------------------------------------------------- checkpoints


def probe_checkpoint(cfg: Config, clip: Clip) -> tuple[bool, dict[str, Any]]:
    """Is this clip genuinely finished? Existence alone is never enough."""
    wav, meta_path = cfg.wav_path(clip.clip_id), cfg.meta_path(clip.clip_id)
    n_expected = expected_samples(clip.duration, cfg.sample_rate)

    if not meta_path.exists():
        return False, {"reason": "no sidecar"}
    if not wav.exists() or wav.stat().st_size == 0:
        return False, {"reason": "wav missing or empty"}

    probe = ffprobe_audio(wav)
    if probe is None:
        return False, {"reason": "unprobeable wav"}
    if probe["sample_rate"] != cfg.sample_rate:
        return False, {"reason": f"sample_rate {probe['sample_rate']}"}
    if probe["channels"] != CHANNELS:
        return False, {"reason": f"channels {probe['channels']}"}
    if probe["codec_name"] != "pcm_s16le":
        return False, {"reason": f"codec {probe['codec_name']}"}

    # Byte-exact sample count, derived from the header rather than the duration
    # float, so a truncated copy can never pass.
    n_samples = (wav.stat().st_size - 44) // 2
    if n_samples != n_expected:
        return False, {"reason": f"{n_samples} samples != expected {n_expected}"}

    meta = read_json(meta_path, {}) or {}
    return True, {"reason": "ok", "meta": meta, "n_samples": n_samples}


# --------------------------------------------------------------------- yt-dlp


def _ytdlp_base(cfg: Config, clients: str | None = None) -> list[str]:
    """Common yt-dlp flags.

    `clients` is left unset on normal attempts: yt-dlp's own client rotation
    exposes the audio-only DASH formats, and pinning it hides them.
    """
    argv = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--ignore-config",
        "--retries", "3",
        "--socket-timeout", "30",
        "--sleep-requests", "1",
        "--user-agent", UA,
    ]
    if clients:
        argv += ["--extractor-args", f"youtube:player_client={clients}"]
    cookies = cfg.resolve_cookies()
    if cookies:
        argv += ["--cookies", str(cookies)]
    return argv


def resolve_media_url(cfg: Config, clip: Clip,
                      clients: str | None = None) -> tuple[dict[str, Any] | None, str]:
    """Resolve the direct bestaudio URL plus source metadata in one call.

    The URL is short-lived (~6 h) so it is resolved immediately before use and
    never cached to Drive.
    """
    argv = _ytdlp_base(cfg, clients) + [
        "-f", "bestaudio/best",
        "--skip-download",
        "--print",
        FIELD_SEP.join(["%(urls)s", "%(format_id)s", "%(duration)s", "%(title)s"]),
        clip.youtube_link,
    ]
    res = run_cmd(argv, timeout=300)
    if not res.ok or not res.stdout.strip():
        return None, res.stderr or "yt-dlp produced no output"

    line = res.stdout.strip().splitlines()[0]
    parts = line.split(FIELD_SEP)
    if len(parts) < 4:
        return None, f"unexpected --print output: {line[:200]}"

    url, format_id, duration_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
    title = FIELD_SEP.join(parts[3:]).strip()
    if not url or url == "NA":
        return None, f"no media url resolved: {line[:200]}"
    try:
        source_dur = float(duration_raw)
    except ValueError:
        source_dur = None
    return (
        {"url": url, "format_id": format_id, "source_dur_sec": source_dur, "title": title},
        "",
    )


# --------------------------------------------------------------------- ffmpeg


def _ffmpeg_trim(source: str, start: float, duration: float, out_path: Path,
                 is_url: bool) -> Any:
    """Decode [start, start+duration) to 16 kHz mono pcm_s16le.

    `-ss` before `-i` seeks the input. On an audio-only stream that is accurate
    to roughly one AAC frame (~23 ms), and over HTTP it turns into byte-range
    requests so only the needed part of the file transfers.
    """
    argv = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    if is_url:
        argv += [
            "-user_agent", UA,
            "-headers", "Referer: https://www.youtube.com/\r\n",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "10",
        ]
    argv += [
        "-ss", f"{start:.6f}",
        "-i", source,
        "-t", f"{duration:.6f}",
        "-vn", "-sn", "-dn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(out_path),
    ]
    return run_cmd(argv, timeout=SUBPROCESS_TIMEOUT_SEC)


def fetch_tier_a(cfg: Config, clip: Clip, out_path: Path,
                 clients: str | None = None) -> tuple[bool, dict[str, Any], str]:
    """Range-fetch only the requested window from the resolved media URL."""
    info, err = resolve_media_url(cfg, clip, clients)
    if info is None:
        return False, {}, err
    res = _ffmpeg_trim(info["url"], clip.start_sec, clip.duration, out_path, is_url=True)
    if not res.ok or not out_path.exists() or out_path.stat().st_size <= 44:
        return False, info, res.stderr or "ffmpeg produced no audio"
    return True, info, ""


def fetch_tier_b(cfg: Config, clip: Clip, out_path: Path,
                 clients: str | None = None) -> tuple[bool, dict[str, Any], str]:
    """Download the whole audio track, then trim locally. Slower but reliable."""
    stem = cfg.work_dir / f"{clip.clip_id}_src"
    clear_dir(cfg.work_dir, [f"{clip.clip_id}_src.*"])
    argv = _ytdlp_base(cfg, clients) + [
        "-f", "bestaudio/best",
        "--no-part",
        "-o", f"{stem}.%(ext)s",
        # --print implies --simulate; without --no-simulate nothing downloads.
        "--no-simulate",
        "--print", FIELD_SEP.join(["%(format_id)s", "%(duration)s", "%(title)s"]),
        clip.youtube_link,
    ]
    res = run_cmd(argv, timeout=SUBPROCESS_TIMEOUT_SEC)

    info: dict[str, Any] = {}
    if res.stdout.strip():
        parts = res.stdout.strip().splitlines()[0].split(FIELD_SEP)
        if len(parts) >= 3:
            try:
                source_dur = float(parts[1].strip())
            except ValueError:
                source_dur = None
            info = {
                "format_id": parts[0].strip(),
                "source_dur_sec": source_dur,
                "title": FIELD_SEP.join(parts[2:]).strip(),
            }

    downloaded = sorted(cfg.work_dir.glob(f"{clip.clip_id}_src.*"))
    if not res.ok or not downloaded:
        return False, info, res.stderr or "yt-dlp download produced no file"

    src = downloaded[0]
    try:
        trim = _ffmpeg_trim(str(src), clip.start_sec, clip.duration, out_path, is_url=False)
        if not trim.ok or not out_path.exists() or out_path.stat().st_size <= 44:
            return False, info, trim.stderr or "ffmpeg produced no audio"
    finally:
        for f in downloaded:
            f.unlink(missing_ok=True)
    return True, info, ""


# ------------------------------------------------------------- per-clip driver


def attempt_plan(cfg: Config, preferred: str | None | object = _UNSET) -> list[tuple[str, str | None]]:
    """(tier, player_client) pairs to try, cheapest and most likely first.

    Which YouTube player client yields a *downloadable* format is not stable --
    it varies by network, video and yt-dlp version, and the best-format client
    is often not a working one. Measured from this machine: the default client
    rotation offers opus audio-only (itag 251) but the media URL 403s without a
    PO token, while `android` offers only itag 18 (muxed 360p) and downloads
    fine. On another network the reverse is common.

    So the ladder is walked rather than guessed, and `run()` feeds back whichever
    client last worked so only the first clip pays for the search.
    """
    ladder = list(CLIENT_LADDER)
    if preferred is not _UNSET and preferred in ladder:
        ladder.remove(preferred)          # type: ignore[arg-type]
        ladder.insert(0, preferred)       # type: ignore[arg-type]
    plan = [("A", c) for c in ladder] + [("B", c) for c in ladder]
    return plan[: max(1, cfg.max_attempts)]


def extract_clip(cfg: Config, clip: Clip,
                 preferred_client: str | None | object = _UNSET) -> dict[str, Any]:
    """Fetch one clip, walking the tier/client ladder. Returns a results row."""
    n_expected = expected_samples(clip.duration, cfg.sample_rate)
    local_wav = cfg.work_dir / f"{clip.clip_id}.wav"
    started = time.perf_counter()

    known = KNOWN_UNAVAILABLE.get(clip.video_id)
    if known:
        LOG.warning("%s: known-unavailable source (%s)", clip.clip_id, known)

    plan = attempt_plan(cfg, preferred_client)
    last_error, last_class, permanent, info = "", "unknown", False, {}
    attempt = 0

    for attempt, (tier, clients) in enumerate(plan, start=1):
        local_wav.unlink(missing_ok=True)
        fetch = fetch_tier_a if tier == "A" else fetch_tier_b
        ok, tier_info, err = fetch(cfg, clip, local_wav, clients)
        info = {**info, **(tier_info or {})}

        if ok:
            try:
                geom = force_exact_length(local_wav, n_expected)
            except Exception as exc:  # malformed wav from ffmpeg
                ok, err = False, f"wav normalisation failed: {exc}"
            else:
                # Diagnostic gate: a large pre-normalisation delta means the
                # fetch cannot be trusted even though it produced audio.
                if abs(geom["raw_delta_sec"]) > cfg.dur_tol_sec and tier == "A":
                    ok = False
                    err = (
                        f"duration mismatch: got {geom['raw_samples'] / SAMPLE_RATE:.3f}s, "
                        f"expected {clip.duration:.3f}s (tolerance {cfg.dur_tol_sec}s)"
                    )
                    last_class, permanent = "duration_mismatch", False
                else:
                    if abs(geom["raw_delta_sec"]) > cfg.dur_tol_sec:
                        LOG.warning(
                            "%s: tier B still off by %.3fs; padding/trimming to exact length",
                            clip.clip_id, geom["raw_delta_sec"],
                        )
                    return _publish(cfg, clip, local_wav, geom, info, tier, clients,
                                    attempt, time.perf_counter() - started, n_expected)

        last_error = err
        if last_class != "duration_mismatch":
            last_class, permanent = classify_error(err)
        append_jsonl(
            cfg.failures_jsonl,
            {
                "ts_utc": now_utc_iso(),
                "clip_id": clip.clip_id,
                "video_id": clip.video_id,
                "tier": tier,
                "player_client": clients or "default",
                "attempt": attempt,
                "error_class": last_class,
                "error_msg": tail(err, 200),
                "stderr_tail": tail(err, 800),
            },
        )
        LOG.warning(
            "%s attempt %d (tier %s, client %s) failed [%s]",
            clip.clip_id, attempt, tier, clients or "default", last_class,
        )

        if permanent:
            LOG.error("%s: permanent failure, not retrying", clip.clip_id)
            break
        if attempt < len(plan):
            time.sleep(BACKOFF_BASE_SEC * (2 ** (attempt - 1)) + random.uniform(0, 2))

    local_wav.unlink(missing_ok=True)
    return {
        **_gt_columns(clip),
        "status": "failed",
        "permanent": permanent,
        "wav_path": "",
        "n_expected_samples": n_expected,
        "source_dur_sec": info.get("source_dur_sec"),
        "source_title": info.get("title", ""),
        "ytdlp_format_id": info.get("format_id", ""),
        "download_tier": "",
        "player_client": "",
        "attempts": attempt,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "error_class": last_class,
        "error_msg": tail(last_error, 300),
        "extracted_at_utc": now_utc_iso(),
    }


def _publish(cfg: Config, clip: Clip, local_wav: Path, geom: dict, info: dict,
             tier: str, clients: str | None, attempt: int, elapsed: float,
             n_expected: int) -> dict[str, Any]:
    """Copy to Drive atomically, then write the sidecar as the commit marker."""
    dest = atomic_publish(local_wav, cfg.wav_path(clip.clip_id))
    size = dest.stat().st_size
    local_wav.unlink(missing_ok=True)

    row = {
        **_gt_columns(clip),
        "status": "ok",
        "permanent": False,
        "wav_path": str(dest),
        "n_samples": geom["n_samples"],
        "n_expected_samples": n_expected,
        "raw_delta_sec": geom["raw_delta_sec"],
        "pad_samples": geom["pad_samples"],
        "trim_samples": geom["trim_samples"],
        "sample_rate": cfg.sample_rate,
        "channels": CHANNELS,
        "file_size_bytes": size,
        "source_dur_sec": info.get("source_dur_sec"),
        "source_title": info.get("title", ""),
        "ytdlp_format_id": info.get("format_id", ""),
        "download_tier": tier,
        "player_client": clients or "default",
        "attempts": attempt,
        "elapsed_sec": round(elapsed, 2),
        "error_class": "",
        "error_msg": "",
        "extracted_at_utc": now_utc_iso(),
    }
    write_json_atomic(cfg.meta_path(clip.clip_id), row)  # written LAST
    LOG.info(
        "%s ok  tier %s/%s  fmt %s  %s  %.1fs  (pad %d / trim %d)",
        clip.clip_id, tier, clients or "default", info.get("format_id", "?"),
        human_bytes(size), elapsed, geom["pad_samples"], geom["trim_samples"],
    )
    return row


def _gt_columns(clip: Clip) -> dict[str, Any]:
    """Ground-truth stats travel with the audio so Step 2 never re-parses the CSV."""
    stats = clip.stats
    return {
        "clip_id": clip.clip_id,
        "video_id": clip.video_id,
        "youtube_link": clip.youtube_link,
        "start_sec": clip.start_sec,
        "end_sec": clip.end_sec,
        "requested_dur_sec": clip.duration,
        "n_gt_segments": stats.get("n_gt_segments"),
        "n_gt_speakers": stats.get("n_gt_speakers"),
        "gt_min_start": stats.get("gt_min_start"),
        "gt_max_end": stats.get("gt_max_end"),
        "gt_overrun_sec": stats.get("gt_overrun_sec"),
        "gt_speech_sec": stats.get("gt_speech_sec"),
        "gt_speaker_time_sec": stats.get("gt_speaker_time_sec"),
        "gt_overlap_sec": stats.get("gt_overlap_sec"),
        "gt_overlap_frac": stats.get("gt_overlap_frac"),
        "n_bad_gt_segments": stats.get("n_bad_gt_segments"),
        # ref_ prefix marks these as ground-truth-derived: they are computed
        # from the reference transcript's script, so they are for reporting and
        # error analysis only and must never reach the ASR as a language code.
        "ref_lang_script": stats.get("lang_script"),
        "ref_lang_hint": stats.get("lang_hint"),
    }


# ------------------------------------------------------------------- the loop


def load_results(cfg: Config) -> dict[str, dict]:
    if not cfg.extraction_csv.exists():
        return {}
    df = pd.read_csv(cfg.extraction_csv)
    return {r["clip_id"]: dict(r) for r in df.to_dict("records")}


def _flush(cfg: Config, results: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(results.values()))
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[RESULT_COLUMNS]
    tmp = cfg.work_dir / "step1_extraction.csv"
    df.to_csv(tmp, index=False)
    atomic_publish(tmp, cfg.extraction_csv)
    return df


def select_clips(clips: list[Clip], flags: StageFlags, results: dict[str, dict]) -> list[Clip]:
    selected = list(clips)
    if flags.only_clip_ids:
        wanted = set(flags.only_clip_ids)
        selected = [c for c in selected if c.clip_id in wanted or c.video_id in wanted]
    if flags.retry_failed_only:
        selected = [
            c for c in selected
            if results.get(c.clip_id, {}).get("status") == "failed"
            and not bool(results.get(c.clip_id, {}).get("permanent"))
        ]
    if flags.limit is not None:
        selected = selected[: flags.limit]
    return selected


def run(cfg: Config, clips: list[Clip], flags: StageFlags | None = None) -> pd.DataFrame:
    """Extract every selected clip, checkpointing after each one."""
    flags = flags or StageFlags()
    run_started = time.perf_counter()

    results = load_results(cfg)
    for clip in clips:  # seed so the CSV always covers all 100 input rows
        results.setdefault(
            clip.clip_id, {**_gt_columns(clip), "status": "pending", "permanent": False}
        )

    selected = select_clips(clips, flags, results)
    LOG.info("step 1: %d clip(s) selected of %d", len(selected), len(clips))

    # Whichever player client last worked is tried first on the next clip, so
    # only the first clip pays for walking the ladder. A pinned force_client
    # skips the search altogether.
    preferred: str | None | object = cfg.force_client if cfg.force_client else _UNSET

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for i, clip in enumerate(selected, start=1):
        prefix = f"[{i}/{len(selected)}] {clip.clip_id}"

        if not flags.force_redo:
            valid, detail = probe_checkpoint(cfg, clip)
            if valid:
                counts["skipped"] += 1
                cached = detail.get("meta") or {}
                results[clip.clip_id] = {**results[clip.clip_id], **cached, "status": "ok"}
                LOG.info("%s skip (checkpoint valid)", prefix)
                continue
            LOG.info("%s extracting (%s)", prefix, detail["reason"])
        else:
            LOG.info("%s extracting (force_redo)", prefix)

        row = extract_clip(cfg, clip, preferred)
        results[clip.clip_id] = row
        counts["ok" if row["status"] == "ok" else "failed"] += 1
        if row["status"] == "ok":
            won = row.get("player_client") or "default"
            preferred = None if won == "default" else won
        _flush(cfg, results)

    df = _flush(cfg, results)
    clear_dir(cfg.work_dir, ["*.wav", "*_src.*"])
    summary = build_summary(cfg, df, counts, time.perf_counter() - run_started, flags)
    LOG.info(
        "step 1 done in %s: %d ok, %d skipped, %d failed",
        human_time(summary["wall_clock_sec"]), counts["ok"], counts["skipped"], counts["failed"],
    )
    return df


def build_summary(cfg: Config, df: pd.DataFrame, counts: dict, wall_clock: float,
                  flags: StageFlags) -> dict[str, Any]:
    ok = df[df.status == "ok"]
    failed = df[df.status == "failed"]
    total_sec = float(pd.to_numeric(ok.requested_dur_sec, errors="coerce").sum())
    total_bytes = float(pd.to_numeric(ok.file_size_bytes, errors="coerce").fillna(0).sum())
    elapsed = pd.to_numeric(ok.elapsed_sec, errors="coerce").dropna()

    summary = {
        "step": 1,
        "generated_at_utc": now_utc_iso(),
        "wall_clock_sec": round(wall_clock, 1),
        "wall_clock_human": human_time(wall_clock),
        "session_counts": counts,
        "flags": {k: v for k, v in vars(flags).items()},
        "totals": {
            "n_input_rows": int(len(df)),
            "n_ok": int(len(ok)),
            "n_failed": int(len(failed)),
            "n_pending": int((df.status == "pending").sum()),
            "audio_hours": round(total_sec / 3600, 3),
            "bytes": int(total_bytes),
            "bytes_human": human_bytes(total_bytes),
            "mean_sec_per_clip": round(float(elapsed.mean()), 2) if len(elapsed) else None,
        },
        "audio_contract": {
            "sample_rate": cfg.sample_rate,
            "channels": CHANNELS,
            "codec": "pcm_s16le",
            "exact_length": "n_samples == round((end_sec - start_sec) * 16000)",
            "dur_tol_sec_role": "diagnostic / tier-B fallback trigger only",
            "n_clips_padded": int((pd.to_numeric(ok.pad_samples, errors="coerce") > 0).sum()),
            "n_clips_trimmed": int((pd.to_numeric(ok.trim_samples, errors="coerce") > 0).sum()),
            "max_abs_raw_delta_sec": round(
                float(pd.to_numeric(ok.raw_delta_sec, errors="coerce").abs().max()), 4
            ) if len(ok) else None,
            "all_exact": bool(
                len(ok) == 0
                or (
                    pd.to_numeric(ok.n_samples, errors="coerce")
                    == pd.to_numeric(ok.n_expected_samples, errors="coerce")
                ).all()
            ),
        },
        "download_tiers": ok.download_tier.value_counts().to_dict() if len(ok) else {},
        "player_clients": ok.player_client.value_counts().to_dict() if len(ok) else {},
        "formats": ok.ytdlp_format_id.value_counts().to_dict() if len(ok) else {},
        "error_classes": failed.error_class.value_counts().to_dict() if len(failed) else {},
        "failed_clip_ids": failed.clip_id.tolist(),
        "permanent_failure_clip_ids": failed[failed.permanent == True].clip_id.tolist(),  # noqa: E712
        "known_unavailable_sources": KNOWN_UNAVAILABLE,
        "tool_versions": {
            "yt_dlp": tool_version("yt-dlp"),
            "ffmpeg": tool_version("ffmpeg", "-version"),
        },
        "paths": {
            "root": str(cfg.root),
            "audio_dir": str(cfg.audio_dir),
            "extraction_csv": str(cfg.extraction_csv),
            "failures_jsonl": str(cfg.failures_jsonl),
        },
    }
    write_json_atomic(cfg.extraction_summary, summary)
    return summary


def audit(cfg: Config, clips: list[Clip]) -> pd.DataFrame:
    """Re-probe every published WAV against the exact-length contract."""
    rows = []
    for clip in clips:
        valid, detail = probe_checkpoint(cfg, clip)
        rows.append(
            {
                "clip_id": clip.clip_id,
                "valid": valid,
                "reason": detail["reason"],
                "expected_samples": expected_samples(clip.duration, cfg.sample_rate),
                "n_samples": detail.get("n_samples"),
            }
        )
    return pd.DataFrame(rows)
