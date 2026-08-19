"""Shared plumbing: logging, subprocess, ffprobe, atomic writes to Drive.

The atomicity helpers exist because Google Drive's FUSE mount gives no write
atomicity and a Colab runtime can die mid-write. Anything that lands on Drive
goes local-file -> `.part` copy -> rename, so a half-written file can never be
mistaken for a finished one.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "sarvam_diar", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


LOG = get_logger()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tail(text: str | None, n: int = 800) -> str:
    """Last n chars of a stderr blob, for compact failure logs."""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= n else "..." + text[-n:]


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


@contextmanager
def timer():
    """`with timer() as t: ...` then `t()` gives elapsed seconds."""
    start = time.perf_counter()
    elapsed = lambda: time.perf_counter() - start  # noqa: E731
    yield elapsed


# --------------------------------------------------------------- subprocesses


class CmdResult:
    __slots__ = ("argv", "returncode", "stdout", "stderr", "elapsed_sec", "timed_out")

    def __init__(self, argv, returncode, stdout, stderr, elapsed_sec, timed_out=False):
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed_sec = elapsed_sec
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def __repr__(self) -> str:
        return f"<CmdResult rc={self.returncode} timed_out={self.timed_out} {self.argv[0]}>"


def run_cmd(argv: list[str], timeout: float = 1800.0) -> CmdResult:
    """Run a command as an argv list (never a shell string) and capture output."""
    argv = [str(a) for a in argv]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return CmdResult(
            argv, proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - start
        )
    except subprocess.TimeoutExpired as exc:
        return CmdResult(
            argv,
            -1,
            exc.stdout or "",
            (exc.stderr or "") + f"\nTimeoutExpired after {timeout}s",
            time.perf_counter() - start,
            timed_out=True,
        )
    except FileNotFoundError as exc:
        return CmdResult(argv, -2, "", f"executable not found: {exc}", 0.0)


def tool_version(executable: str, *args: str) -> str:
    """Best-effort version string, used to stamp the run summary."""
    res = run_cmd([executable, *(args or ("--version",))], timeout=60)
    if not res.ok:
        return "unavailable"
    first = (res.stdout or res.stderr).strip().splitlines()
    return first[0][:120] if first else "unknown"


def have(executable: str) -> bool:
    return shutil.which(executable) is not None


# -------------------------------------------------------------------- ffprobe


def ffprobe_audio(path: str | os.PathLike) -> dict[str, Any] | None:
    """Probe the first audio stream. Returns None if the file is unreadable."""
    res = run_cmd(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,codec_name,duration",
            "-show_entries", "format=duration,size",
            "-of", "json",
            str(path),
        ],
        timeout=120,
    )
    if not res.ok:
        return None
    try:
        payload = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream, fmt = streams[0], payload.get("format", {})

    def _f(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "codec_name": stream.get("codec_name"),
        "duration_sec": _f(stream.get("duration")) or _f(fmt.get("duration")),
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
    }


# ------------------------------------------------------------- atomic writes


def atomic_publish(local_path: str | os.PathLike, dest_path: str | os.PathLike) -> Path:
    """Copy a finished local file onto Drive without ever exposing a partial file.

    Copies to `<dest>.part` then renames. os.replace is atomic within the same
    directory even on the Drive FUSE mount, so a reader either sees the previous
    file or the complete new one -- never a truncated one.
    """
    local_path, dest_path = Path(local_path), Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        shutil.copyfile(local_path, part)
        os.replace(part, dest_path)
    finally:
        if part.exists():
            part.unlink(missing_ok=True)
    return dest_path


def write_json_atomic(path: str | os.PathLike, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(part, path)
    return path


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    """Append one record. Opened/closed per call so Drive flushes each line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ------------------------------------------------------------------ dotenv
# Secrets live in a .env file rather than in the notebook, because main.ipynb is
# published to a public repo. .env itself is gitignored and never committed.


def parse_dotenv(text: str) -> dict[str, str]:
    """Minimal .env parser -- no dependency, and the format is trivial.

    Handles `KEY=value`, `KEY="value"`, `KEY='value'`, `export KEY=value`,
    comments and blank lines. A trailing ` # comment` is stripped from unquoted
    values only, since a quoted value may legitimately contain a '#'.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        out[key] = value
    return out


def find_dotenv(extra: Iterable[str | os.PathLike] = ()) -> Path | None:
    """First existing .env among the candidates, nearest-first.

    On Colab the repo clone will NOT contain .env (it is gitignored), so the
    Drive root is the location that actually works there.
    """
    candidates = [Path(p) for p in extra]
    candidates += [Path.cwd() / ".env", Path.cwd().parent / ".env",
                   Path.home() / ".env"]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_dotenv(extra: Iterable[str | os.PathLike] = (), export: bool = True) -> dict[str, str]:
    """Read the nearest .env. With `export`, values that are not already set in
    the environment are placed there, so later stages (e.g. SARVAM_API_KEY in
    Step 3) can just read os.environ."""
    path = find_dotenv(extra)
    if path is None:
        return {}
    values = parse_dotenv(path.read_text(encoding="utf-8"))
    if export:
        for k, v in values.items():
            os.environ.setdefault(k, v)
    LOG.info("loaded %d key(s) from %s", len(values), path)
    return values


# ------------------------------------------------------------- leakage guard


def reference_fields(columns: Iterable[str]) -> list[str]:
    """Column names that carry ground-truth-derived information."""
    from .config import REFERENCE_FIELD_PREFIXES

    return sorted(c for c in columns if str(c).startswith(REFERENCE_FIELD_PREFIXES))


def assert_no_reference_fields(frame, where: str = "pipeline stage") -> None:
    """Fail loudly if ground truth reaches a pipeline stage.

    `data.ClipInput` covers the object path; this covers the DataFrame path,
    where a stray `n_gt_speakers` column is one `.get()` away from being handed
    to a diarizer as `num_speakers`.
    """
    leaked = reference_fields(getattr(frame, "columns", frame))
    if leaked:
        raise AssertionError(
            f"ground-truth fields reached {where}: {leaked}. "
            "The brief allows ground truth only for computing final scores -- "
            "drop these columns or use data.split_reference()."
        )


def clear_dir(path: str | os.PathLike, patterns: Iterable[str] = ("*",)) -> int:
    """Remove scratch files. Used to keep /content/work from filling the disk."""
    path = Path(path)
    removed = 0
    for pattern in patterns:
        for item in path.glob(pattern):
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
                removed += 1
            except OSError:
                pass
    return removed
