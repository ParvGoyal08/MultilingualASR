"""AI4Bharat IndicConformer-600M-Multilingual as a third ASR backend.

A new module rather than an addition to asr.py, so nothing an already-running
notebook imports changes shape.

Three things about this model shape the code:

* **It has no language identification.** `model(wav, lang, decoding)` requires a
  language code and offers no auto mode, unlike Saaras (`language_code=
  "unknown"`) or Whisper (built-in LID). A language must therefore be supplied
  from outside, which means any number reported for it is a PIPELINE number --
  language source plus recogniser -- and its errors include language errors.
* **It is an utterance-level Conformer.** Attention cost grows quadratically, so
  long-form input is not an option; segments are capped at MAX_SEC.
* **It loads with `trust_remote_code=True`**, i.e. it executes code fetched from
  the Hub. Standard for this model, worth stating.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

from .config import Config
from .data import Turn

LOG = logging.getLogger("sarvam_diar")

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16_000
MAX_SEC = 30.0
MIN_DUR = 0.30

# The 22 scheduled languages the model card lists. Checked before a call rather
# than after, because an unsupported code is a silent wrong answer otherwise.
LANGS = frozenset({"as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks",
                   "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat",
                   "sd", "ta", "te", "ur"})

_CACHE: dict = {}


def preflight() -> dict:
    """What is installed, and is torch intact? Run BEFORE installing anything.

    The IndicConformer card lists `pip install transformers torchaudio
    onnxruntime-gpu`. On Kaggle that is actively harmful: torch and torchaudio
    are preinstalled and pinned to the image's CUDA build, and letting pip
    resolve them upgrades torch underneath a running kernel. The symptom is
    `AttributeError: module 'torch' has no attribute '_utils'` raised from deep
    inside an unrelated import -- a corrupted install, not a missing package.
    """
    import importlib.util as iu

    out: dict = {}
    try:
        from .diarization import resolve_token

        out["hf_token"] = "present" if resolve_token(Config.create()) else "MISSING"
    except Exception as exc:  # noqa: BLE001
        out["hf_token"] = f"unresolved: {type(exc).__name__}"
    for mod in ("torch", "torchaudio", "transformers", "onnxruntime", "nemo"):
        spec = iu.find_spec(mod)
        if spec is None:
            out[mod] = None
            continue
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"BROKEN: {type(exc).__name__}: {str(exc)[:80]}"
    return out


def install_hint(state: dict) -> list[str]:
    """The packages that are genuinely missing -- never torch or torchaudio."""
    return [m for m in ("transformers", "onnxruntime") if state.get(m) is None]


def load(model_id: str = MODEL_ID, device: str | None = None, cfg: Config | None = None):
    """Load once and cache. Returns (model, device_str).

    The repo is GATED. Two separate things are needed and the 401 does not
    distinguish them: access must be granted to your account on the model page,
    AND a token must be presented. The token is resolved the same way the
    diarization models resolve theirs -- .env, then the process environment,
    then the host secret store -- never from a notebook cell, because the
    notebooks go to a public repo.
    """
    import torch

    if not hasattr(torch, "_utils"):
        raise RuntimeError(
            "torch is corrupted (no torch._utils). Something pip-installed a "
            "different torch underneath this kernel. A kernel restart is NOT "
            "enough because the replaced files persist for the session -- use "
            "Session options > Factory reset, then install nothing that pulls "
            "torch. See asr_indic.preflight().")
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from .diarization import resolve_token

    token = resolve_token(cfg) if cfg is not None else resolve_token(Config.create())
    key = (model_id, device)
    if key in _CACHE:
        return _CACHE[key]

    t0 = time.perf_counter()
    LOG.info("loading %s (trust_remote_code) -- first call also downloads ~2.4 GB",
             model_id)
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                          token=token)
    except OSError as exc:
        raise RuntimeError(
            f"cannot load {model_id}: {str(exc)[:200]}\n"
            f"  token present: {bool(token)}\n"
            "  This repo is gated, so BOTH of these must be true:\n"
            "   1. your HF account has been granted access -- open\n"
            f"      https://huggingface.co/{model_id} while logged in and\n"
            "      accept the terms (approval is usually immediate)\n"
            "   2. that account's token reaches this process -- Kaggle\n"
            "      Add-ons > Secrets > HF_TOKEN, or .env locally\n"
            "  A token from an account WITHOUT access still returns 401."
        ) from exc
    try:
        model = model.to(device)
    except Exception as exc:  # noqa: BLE001
        # The card lists onnxruntime-gpu, so inference may not be a torch graph
        # at all and .to() may be meaningless. Report rather than pretend.
        LOG.warning("could not move model to %s (%s); continuing on default device",
                    device, type(exc).__name__)
        device = f"{device}?unmoved"
    if hasattr(model, "eval"):
        model.eval()
    LOG.info("model ready in %.0fs on %s", time.perf_counter() - t0, device)
    _CACHE[key] = (model, device)
    return _CACHE[key]


def load_wav(path: Path):
    """16 kHz mono float tensor shaped [1, N], as the model card expects."""
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    return wav


def _text_of(out) -> str:
    """The model returns a string or a one-element sequence depending on path."""
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        return " ".join(_text_of(o) for o in out)
    return str(out)


def transcribe_tensor(model, wav, lang: str, decoding: str = "rnnt") -> str:
    if lang not in LANGS:
        raise ValueError(f"{lang!r} is not one of IndicConformer's languages: "
                         f"{sorted(LANGS)}")
    if decoding not in ("rnnt", "ctc"):
        raise ValueError(f"decoding must be rnnt or ctc, got {decoding!r}")
    import torch

    with torch.no_grad():
        return _text_of(model(wav, lang, decoding)).strip()


def smoke(cfg: Config, clip_id: str, lang: str, decoding: str = "rnnt",
          seconds: float = 20.0) -> dict:
    """Prove download, device, 16 kHz input and one transcription end to end."""
    import torch

    model, device = load(cfg=cfg)
    wav_path = cfg.wav_path(clip_id)
    wav = load_wav(wav_path)
    clipped = wav[:, : int(seconds * SAMPLE_RATE)]
    t0 = time.perf_counter()
    text = transcribe_tensor(model, clipped, lang, decoding)
    elapsed = time.perf_counter() - t0
    dur = clipped.shape[1] / SAMPLE_RATE
    mem = (torch.cuda.max_memory_allocated() / 2 ** 30
           if torch.cuda.is_available() else 0.0)
    return {"clip_id": clip_id, "lang": lang, "decoding": decoding,
            "device": device, "wav_shape": tuple(wav.shape), "sample_rate": SAMPLE_RATE,
            "audio_sec": round(dur, 1), "elapsed_sec": round(elapsed, 2),
            "rtf": round(elapsed / max(dur, 1e-9), 4),
            "gpu_gb_peak": round(mem, 2), "n_words": len(text.split()), "text": text}


def transcribe_segments(cfg: Config, wav: Path, turns: Sequence[Turn], lang: str,
                        decoding: str = "rnnt", min_dur: float = MIN_DUR,
                        max_sec: float = MAX_SEC) -> tuple[list[dict], dict]:
    """One diarized turn at a time, so attribution is exact by construction.

    Mirrors asr.transcribe_segments: turns under `min_dur` are skipped and
    counted, and turns over `max_sec` are split, since a Conformer's attention
    cost is quadratic in input length.
    """
    import math

    model, device = load(cfg=cfg)
    audio = load_wav(Path(wav))
    n = audio.shape[1]
    rows: list[dict] = []
    skipped = 0
    t0 = time.perf_counter()

    for idx, t in enumerate(sorted(turns, key=lambda x: x.start)):
        dur = t.end - t.start
        if dur < min_dur:
            skipped += 1
            rows.append({"i": idx, "start": t.start, "end": t.end,
                         "speaker": t.speaker, "text": "", "skipped": "too_short"})
            continue
        parts = []
        for k in range(max(1, math.ceil(dur / max_sec))):
            a = t.start + k * max_sec
            b = min(t.end, a + max_sec)
            seg = audio[:, max(0, int(a * SAMPLE_RATE)):min(n, int(b * SAMPLE_RATE))]
            if seg.shape[1] < int(min_dur * SAMPLE_RATE):
                continue
            parts.append(transcribe_tensor(model, seg, lang, decoding))
        rows.append({"i": idx, "start": t.start, "end": t.end, "speaker": t.speaker,
                     "text": " ".join(p for p in parts if p), "skipped": None})

    meta = {"model": MODEL_ID, "device": device, "clip_language": lang,
            "decoding": decoding, "n_segments": len(rows), "n_skipped": skipped,
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "n_words": sum(len(r["text"].split()) for r in rows)}
    return rows, meta
