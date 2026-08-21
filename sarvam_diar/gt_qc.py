"""Clip-level quality control on ground-truth temporal alignment.

Some clips' annotations are displaced in time against the audio. This stage
detects that, ranks the evidence, and produces a shortlist for human
verification. It never edits the raw annotations: confirmed corrections live in
a separate, versioned manifest, and scoring can be run against raw GT or against
QC-adjusted GT with the latter always labelled as a diagnostic.

Design rules, in order of importance:

* **The raw ground truth is immutable.** Nothing here writes to it. A correction
  is a row in a manifest with a provenance field, not an edit.
* **No ground truth reaches the pipeline.** Everything in this module is
  scoring-side. The consensus signal is built only from model outputs, and the
  fusion in `refinement` never sees a lag.
* **Detection must not optimise the metric it will be judged by.** The lag is
  chosen by speech-activity agreement, never by DER or JER, so a later DER
  improvement is evidence rather than a tautology.
* **Consensus, not a single model.** A lag from one diarizer could be that
  diarizer being wrong. 2-of-N voting across independent segmenters cannot be
  wrong the same way, and an energy VAD that shares no code with any of them
  provides a further, model-free check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import Turn

HOP = 0.01
MAX_LAG = 5.0
# A correction is only worth recording when it is large enough to matter and
# sharp enough to trust. Both thresholds are deliberately conservative: the cost
# of missing a defect is a slightly pessimistic score, the cost of inventing one
# is a corrupted benchmark.
MIN_FLAG_LAG = 1.0          # seconds
# Fraction of the AVAILABLE headroom the shift must recover, i.e.
# (peak - zero) / (1 - zero). An absolute gain threshold is the wrong test: a
# clip whose speech is near-continuous already scores ~0.95 IoU at lag zero, so
# it can gain at most 0.05 no matter how badly misaligned it is, and a fixed
# 0.05 bar makes it structurally unflaggable. Measured against headroom, a real
# shift on such a clip recovers 60-90% of what is left, while an aligned clip
# recovers ~10%.
MIN_HEADROOM_RECOVERED = 0.35
MIN_IMPROVEMENT = 0.01      # small absolute floor, to reject pure noise
MIN_PEAK_MARGIN = 0.02      # peak must beat everything outside +-0.5 s of it
ROUND_TO = 0.5              # confirmed corrections are recorded on this grid


@dataclass
class LagCandidate:
    clip_id: str
    best_lag: float
    zero_lag_iou: float
    peak_iou: float
    improvement: float
    headroom_recovered: float
    peak_margin: float
    n_models: int
    model_spread: float
    vad_lag: float | None
    vad_agrees: bool
    duration: float
    flagged: bool
    reason: str

    def as_row(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- signals


def activity(turns: Sequence[Turn], n_frames: int) -> np.ndarray:
    """Binary speech/non-speech at HOP resolution. Speaker identity discarded."""
    a = np.zeros(n_frames, dtype=np.int8)
    for t in turns:
        lo = int(max(0.0, t.start) / HOP)
        hi = int(min(n_frames * HOP, t.end) / HOP)
        if hi > lo:
            a[lo:hi] = 1
    return a


def consensus(per_model: dict[str, Sequence[Turn]], n_frames: int,
              min_votes: int = 2) -> np.ndarray:
    """Frames where at least `min_votes` models call speech.

    Deliberately built from RAW model outputs, never from the fused result:
    the fusion is the thing this QC is meant to keep honest, so letting it feed
    the detector would make the two agree by construction.
    """
    if not per_model:
        return np.zeros(n_frames, dtype=np.int8)
    votes = np.zeros(n_frames, dtype=np.int16)
    for turns in per_model.values():
        votes += activity(turns, n_frames)
    return (votes >= min(min_votes, len(per_model))).astype(np.int8)


def energy_vad(wav_path: Path, n_frames: int, percentile: float = 35.0) -> np.ndarray | None:
    """A crude frame-energy VAD, as a model-free second opinion.

    Shares no code, no training data and no assumptions with any diarizer, so
    when it agrees on a lag it is genuine corroboration rather than the same
    system counted twice. Too blunt to diarize with; entirely adequate to say
    where speech starts.
    """
    try:
        import soundfile as sf
    except ImportError:      # pragma: no cover
        return None
    if not Path(wav_path).exists():
        return None
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    step = int(HOP * sr)
    usable = min(n_frames, len(audio) // step)
    if usable < 50:
        return None
    frames = audio[:usable * step].reshape(usable, step)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    thr = np.percentile(rms, percentile)
    out = np.zeros(n_frames, dtype=np.int8)
    out[:usable] = (rms > thr).astype(np.int8)
    return out


# ------------------------------------------------------------------- search


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def lag_profile(ref_act: np.ndarray, hyp_act: np.ndarray,
                max_lag: float = MAX_LAG) -> tuple[np.ndarray, np.ndarray]:
    """IoU between the two activity tracks at every candidate lag.

    A positive lag means the reference must move EARLIER to match -- i.e. the
    annotation sits later than the speech. IoU is used rather than a raw
    correlation because it is bounded and interpretable: "the two agree on 68%
    of the frames either calls speech" is a statement a human can check.
    """
    lim = int(max_lag / HOP)
    lags = np.arange(-lim, lim + 1)
    n = len(ref_act)
    out = np.zeros(len(lags), dtype=np.float32)
    for i, lag in enumerate(lags):
        if lag == 0:
            out[i] = _iou(ref_act, hyp_act)
        elif lag > 0:
            if lag >= n:
                continue
            out[i] = _iou(ref_act[lag:], hyp_act[:n - lag])
        else:
            out[i] = _iou(ref_act[:lag], hyp_act[-lag:])
    return lags * HOP, out


def assess_clip(clip_id: str, ref_turns: Sequence[Turn], duration: float,
                per_model: dict[str, Sequence[Turn]],
                wav_path: Path | None = None) -> tuple[LagCandidate, dict]:
    """Evidence for a global temporal offset on one clip."""
    n = max(1, int(duration / HOP))
    ref_act = activity(ref_turns, n)
    cons = consensus(per_model, n)

    lags, prof = lag_profile(ref_act, cons)
    zero_i = int(np.argmin(np.abs(lags)))
    peak_i = int(np.argmax(prof))
    best_lag, peak, zero = float(lags[peak_i]), float(prof[peak_i]), float(prof[zero_i])

    # Sharpness: how far the peak stands above everything more than 0.5 s away.
    far = np.abs(lags - best_lag) > 0.5
    margin = float(peak - prof[far].max()) if far.any() else float(peak)

    # per-model lags, to report spread as a second consistency signal
    spread = 0.0
    if len(per_model) > 1:
        each = []
        for turns in per_model.values():
            _, p = lag_profile(ref_act, activity(turns, n))
            each.append(float(lags[int(np.argmax(p))]))
        spread = float(np.ptp(each))

    vad_lag, vad_ok = None, False
    if wav_path is not None:
        v = energy_vad(Path(wav_path), n)
        if v is not None:
            _, vp = lag_profile(ref_act, v)
            vad_lag = float(lags[int(np.argmax(vp))])
            vad_ok = abs(vad_lag - best_lag) <= 0.5

    headroom = (peak - zero) / (1.0 - zero) if zero < 1.0 else 0.0

    reasons = []
    if abs(best_lag) < MIN_FLAG_LAG:
        reasons.append(f"lag {best_lag:+.2f}s below {MIN_FLAG_LAG}s")
    if peak - zero < MIN_IMPROVEMENT:
        reasons.append(f"IoU gain {peak - zero:.3f} below {MIN_IMPROVEMENT}")
    if headroom < MIN_HEADROOM_RECOVERED:
        reasons.append(f"recovers {headroom:.2f} of headroom, below "
                       f"{MIN_HEADROOM_RECOVERED}")
    if margin < MIN_PEAK_MARGIN:
        reasons.append(f"peak margin {margin:.3f} below {MIN_PEAK_MARGIN}")
    if len(per_model) < 2:
        reasons.append("fewer than 2 models")
    flagged = not reasons

    cand = LagCandidate(
        clip_id=clip_id, best_lag=best_lag, zero_lag_iou=zero, peak_iou=peak,
        improvement=peak - zero, headroom_recovered=headroom,
        peak_margin=margin, n_models=len(per_model),
        model_spread=spread, vad_lag=vad_lag, vad_agrees=vad_ok,
        duration=duration, flagged=flagged,
        reason="strong candidate" if flagged else "; ".join(reasons))
    detail = {"lags": lags.tolist(), "profile": prof.tolist(),
              "ref_act": ref_act, "consensus": cons}
    return cand, detail


# ------------------------------------------------------------- corrections


def round_correction(lag: float, grid: float = ROUND_TO) -> float:
    """Confirmed offsets are recorded on a coarse grid.

    The detector resolves to 10 ms, but claiming that precision for a human
    judgement would be false. Half a second is about what a person can confirm
    by ear against a waveform.
    """
    return round(lag / grid) * grid


def load_corrections(path: str | Path) -> dict[str, float]:
    """clip_id -> seconds to SUBTRACT from every reference timestamp.

    Only rows marked verified are applied. Candidates sit in the same file
    unverified so the shortlist and its outcome stay in one place.
    """
    p = Path(path)
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    return {r["clip_id"]: float(r["offset_sec"])
            for r in doc.get("corrections", []) if r.get("verified")}


def apply_correction(turns: Sequence[Turn], offset: float,
                     lo: float, hi: float) -> list[Turn]:
    """Move reference turns EARLIER by `offset` and clip to the UEM.

    Applied to a COPY for scoring. The stored annotation is never touched.
    """
    out = []
    for t in turns:
        a, b = max(lo, t.start - offset), min(hi, t.end - offset)
        if b > a:
            out.append(Turn(start=a, end=b, speaker=t.speaker))
    return out
