"""Export a standalone, offline error explorer for Step 2.

Everything the UI shows is computed HERE and serialised. The browser does no
metric arithmetic at all, so what you see is exactly what the benchmark counted
-- there is no second implementation to drift from the first.

The interesting part is `error_regions()`, which turns a (reference, hypothesis,
optimal mapping) triple into time-aligned MISS / FA / CONFUSION / OVERLAP spans.
It delegates the per-interval counting to pyannote's own `LabelMatcher`, which
is the same object `DiarizationErrorRate` uses, rather than reimplementing the
decomposition.

That matters. The obvious set-based formula --

    correct = |ref_speakers & mapped_hyp_speakers|
    miss    = max(0, n_ref - n_hyp)          # WRONG
    fa      = max(0, n_hyp - n_ref)          # WRONG

-- disagrees with pyannote, because pyannote treats the active labels at an
instant as a MULTISET (`get_labels(..., unique=False)`) and solves a Hungarian
assignment over it. When a hypothesis puts the same speaker on two overlapping
tracks, that speaker counts twice. Reference ['A','B'] against hypothesis
['A','A'] scores correct=1 / confusion=1 under pyannote, but the set formula
says miss=1 / confusion=0. Measured over 80 clip/model pairs, the set formula
disagreed on 55 of them, by up to 14 seconds.

`verify_against_metric()` asserts that integrating the per-interval counts
reproduces pyannote's own components to within a microsecond. If it ever fails,
the visualisation is lying and should not be trusted.
"""

from __future__ import annotations


import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .data import ClipReference, Turn
from .evaluation import (
    KEY_CONF,
    KEY_FA,
    KEY_MISS,
    to_annotation,
    turns_to_annotation,
    uem_timeline,
)
from .utils import LOG, write_json_atomic

ASSETS = Path(__file__).parent / "explorer_assets"


# --------------------------------------------------------------- error regions


def optimal_mapping(ref: ClipReference, hyp: list[Turn]) -> dict[str, str]:
    """Hypothesis label -> reference label, under the Hungarian mapping.

    This is what makes speaker colours consistent across the GT and prediction
    rows in the UI: a predicted speaker is drawn in the colour of the reference
    speaker it was matched to, so a colour mismatch IS the confusion error.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    m = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    mapping = m.optimal_mapping(to_annotation(ref),
                                turns_to_annotation(hyp, ref.clip_id),
                                uem=uem_timeline(ref))
    return {str(k): str(v) for k, v in mapping.items()}


def _boundaries(ref: ClipReference, hyp: list[Turn]) -> list[float]:
    lo, hi = ref.uem
    pts = {lo, hi}
    for t in list(ref.turns) + list(hyp):
        for v in (t.start, t.end):
            if lo < v < hi:
                pts.add(v)
    return sorted(pts)


def error_regions(ref: ClipReference, hyp: list[Turn],
                  mapping: dict[str, str] | None = None) -> tuple[list[dict], dict]:
    """Per-interval error decomposition over the whole UEM.

    Returns (regions, totals). Adjacent intervals with an identical error
    signature are merged, so the UI draws a handful of spans instead of
    thousands of one-frame slivers.
    """
    if mapping is None:
        mapping = optimal_mapping(ref, hyp)
    from pyannote.metrics.matcher import LabelMatcher

    matcher = LabelMatcher()
    lo, hi = ref.uem
    bounds = _boundaries(ref, hyp)

    raw: list[dict] = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        # LISTS, not sets: a speaker active on two overlapping tracks counts
        # twice, exactly as pyannote's get_labels(unique=False) reports it.
        rlabels = [t.speaker for t in ref.turns if t.start <= mid < t.end]
        hraw = [t.speaker for t in hyp if t.start <= mid < t.end]
        hlabels = [mapping.get(s, s) for s in hraw]

        counts, _ = matcher(rlabels, hlabels)
        raw.append({
            "start": a, "end": b,
            "ref": sorted(set(rlabels)), "hyp": sorted(set(hlabels)),
            "hyp_raw": sorted(set(hraw)),
            "miss": counts["missed detection"], "fa": counts["false alarm"],
            "confusion": counts["confusion"], "correct": counts["correct"],
            "overlap": int(len(set(rlabels)) >= 2),
        })

    # Merge neighbours whose error signature and active-speaker sets agree.
    merged: list[dict] = []
    for seg in raw:
        if merged:
            p = merged[-1]
            same = (abs(p["end"] - seg["start"]) < 1e-9
                    and p["ref"] == seg["ref"] and p["hyp"] == seg["hyp"]
                    and p["miss"] == seg["miss"] and p["fa"] == seg["fa"]
                    and p["confusion"] == seg["confusion"])
            if same:
                p["end"] = seg["end"]
                continue
        merged.append(dict(seg))

    totals = {
        "miss_sec": sum((s["end"] - s["start"]) * s["miss"] for s in merged),
        "fa_sec": sum((s["end"] - s["start"]) * s["fa"] for s in merged),
        "confusion_sec": sum((s["end"] - s["start"]) * s["confusion"] for s in merged),
        "correct_sec": sum((s["end"] - s["start"]) * s["correct"] for s in merged),
        "overlap_sec": sum((s["end"] - s["start"]) for s in merged if s["overlap"]),
        "uem": [lo, hi],
    }

    # Only the intervals that actually carry an error need to reach the browser.
    regions = [
        {"start": round(s["start"], 3), "end": round(s["end"], 3),
         "types": [t for t, k in (("MISS", "miss"), ("FA", "fa"),
                                  ("CONFUSION", "confusion")) if s[k]],
         "overlap": bool(s["overlap"]),
         "ref": s["ref"], "hyp": s["hyp"]}
        for s in merged
        if s["miss"] or s["fa"] or s["confusion"] or s["overlap"]
    ]
    return regions, totals


def verify_against_metric(ref: ClipReference, hyp: list[Turn],
                          tol: float = 1e-6) -> dict[str, Any]:
    """Assert the exported regions reproduce pyannote's own DER components.

    The UI must not become a second, subtly different implementation of the
    metric. If this disagrees, the picture is wrong, not the number.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    m = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    comp = m.compute_components(to_annotation(ref),
                                turns_to_annotation(hyp, ref.clip_id),
                                uem=uem_timeline(ref))
    _, totals = error_regions(ref, hyp)
    deltas = {
        "miss": abs(totals["miss_sec"] - comp[KEY_MISS]),
        "fa": abs(totals["fa_sec"] - comp[KEY_FA]),
        "confusion": abs(totals["confusion_sec"] - comp[KEY_CONF]),
    }
    return {"ok": all(v < tol for v in deltas.values()), "deltas": deltas,
            "pyannote": {k: comp[k] for k in (KEY_MISS, KEY_FA, KEY_CONF)},
            "exported": {k: totals[f"{k}_sec"] for k in ("miss", "fa", "confusion")}}


# ------------------------------------------------------------------- export


def short_turn_fraction(ref: ClipReference, threshold: float = 0.5) -> float:
    if not ref.turns:
        return 0.0
    return sum(1 for t in ref.turns if t.duration < threshold) / len(ref.turns)


def export_clip(ref: ClipReference, hyps: dict[str, list[Turn]],
                metrics_row: dict[str, dict]) -> dict[str, Any]:
    """One clip, every model, everything the UI needs."""
    payload = {
        "clip_id": ref.clip_id,
        "uem": list(ref.uem),
        "reference": [{"speaker": t.speaker, "start": round(t.start, 3),
                       "end": round(t.end, 3)} for t in ref.turns],
        "ref_speakers": sorted({t.speaker for t in ref.turns}),
        "utterances": [{"speaker": u.speaker, "start": round(u.start, 3),
                        "end": round(u.end, 3), "text": u.text_norm}
                       for u in ref.utterances],
        "models": {},
    }
    for model, hyp in hyps.items():
        mapping = optimal_mapping(ref, hyp)
        regions, totals = error_regions(ref, hyp, mapping)
        payload["models"][model] = {
            "hypothesis": [{"speaker": t.speaker,
                            "mapped": mapping.get(t.speaker, t.speaker),
                            "start": round(t.start, 3), "end": round(t.end, 3)}
                           for t in hyp],
            "mapping": mapping,
            "hyp_speakers": sorted({t.speaker for t in hyp}),
            "regions": regions,
            "totals": {k: round(v, 3) for k, v in totals.items() if k != "uem"},
            "metrics": metrics_row.get(model, {}),
        }
    return payload


def export(cfg: Config, metrics: pd.DataFrame, references: dict[str, ClipReference],
           hypotheses: dict[tuple[str, str], list[Turn]],
           out_dir: str | Path | None = None, copy_audio: bool = False,
           verify: bool = True) -> Path:
    """Write a self-contained error_explorer/ directory.

    `copy_audio=False` by default: the corpus is 1.3 GB, which does not belong in
    a git repository. The UI looks for audio/<clip_id>.wav and degrades to a
    timeline-only view when it is absent, so the export stays committable.
    """
    out = Path(out_dir) if out_dir else (cfg.root / "error_explorer")
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    by_clip: dict[str, dict[str, list[Turn]]] = {}
    for (model, clip_id), hyp in hypotheses.items():
        by_clip.setdefault(clip_id, {})[model] = hyp

    mrows: dict[str, dict[str, dict]] = {}
    for row in metrics.to_dict("records"):
        mrows.setdefault(row["clip_id"], {})[row["model"]] = {
            k: (None if pd.isna(v) else v) for k, v in row.items()
            if k not in ("clip_id", "model")
        }

    index, problems = [], []
    for clip_id, hyps in sorted(by_clip.items()):
        ref = references.get(clip_id)
        if ref is None:
            continue
        if verify:
            for model, hyp in hyps.items():
                v = verify_against_metric(ref, hyp)
                if not v["ok"]:
                    problems.append({"clip_id": clip_id, "model": model, **v})

        payload = export_clip(ref, hyps, mrows.get(clip_id, {}))
        write_json_atomic(out / "data" / f"{clip_id}.json", payload)

        entry = {
            "clip_id": clip_id,
            "duration": round(ref.uem[1], 2),
            "n_speakers_ref": len(payload["ref_speakers"]),
            "overlap_frac": round(ref.stats.get("gt_overlap_frac", 0.0), 4),
            "short_turn_frac": round(short_turn_fraction(ref), 4),
            "n_turns_ref": len(ref.turns),
            # Whether audio EXISTS, not whether we intended to copy it: with
            # copy_audio=True and a missing source, promising audio makes the UI
            # show a load error instead of its "no audio" message.
            "has_audio": cfg.wav_path(clip_id).exists(),
            "models": {},
        }
        for model, block in payload["models"].items():
            m = block["metrics"]
            entry["models"][model] = {
                "der": m.get("der"), "jer": m.get("jer"),
                "miss_sec": block["totals"]["miss_sec"],
                "fa_sec": block["totals"]["fa_sec"],
                "confusion_sec": block["totals"]["confusion_sec"],
                "error_sec": round(block["totals"]["miss_sec"]
                                   + block["totals"]["fa_sec"]
                                   + block["totals"]["confusion_sec"], 3),
                "overlap_der": m.get("overlap_der"),
                "n_speakers_hyp": len(block["hyp_speakers"]),
                "speaker_count_error": len(block["hyp_speakers"]) - entry["n_speakers_ref"],
            }
        index.append(entry)

        if copy_audio:
            src = cfg.wav_path(clip_id)
            if src.exists():
                shutil.copyfile(src, out / "audio" / f"{clip_id}.wav")

    models = sorted({m for (m, _) in hypotheses})
    write_json_atomic(out / "data" / "clips.json", {
        "generated_from": str(cfg.root),
        "models": models,
        "n_clips": len(index),
        "audio_bundled": copy_audio,
        "clips": index,
        "verification": {"checked": verify, "mismatches": problems},
    })

    src_html = ASSETS / "index.html"
    if src_html.exists():
        shutil.copyfile(src_html, out / "index.html")
    (out / "audio" / "README.md").write_text(AUDIO_README, encoding="utf-8")
    (out / "README.md").write_text(EXPORT_README, encoding="utf-8")

    if problems:
        LOG.error("%d clip/model pairs FAILED region verification -- the "
                  "visualisation would not match the metric", len(problems))
    LOG.info("error explorer -> %s (%d clips, models=%s, audio_bundled=%s)",
             out, len(index), models, copy_audio)
    return out


AUDIO_README = """\
# audio/

Empty by default, and deliberately so: the corpus is ~1.3 GB of WAV, which does
not belong in a git repository.

The UI looks for `audio/<clip_id>.wav`. Without it everything still works except
playback -- the timeline, error regions, filtering and speaker rows are all
driven by the JSON in `../data/`.

To add audio, either:

    # copy the whole corpus (~1.3 GB)
    cp /path/to/sarvam_diarization/audio_16k/*.wav audio/

    # or just the clips you are inspecting
    cp /path/to/audio_16k/Cku_X_SL7qU__60_660.wav audio/

or re-export with `explorer.export(..., copy_audio=True)`.
"""

EXPORT_README = """\
# Diarization error explorer

Standalone and offline. No Colab, no backend, no build step -- every number was
computed by the Python pipeline and serialised; the browser only draws it.

    python -m http.server 8000
    # then open http://localhost:8000

## Layout

    index.html          the UI, single file
    data/clips.json     index: one row per clip, all sortable/filterable metrics
    data/<clip_id>.json GT turns, hypothesis turns, optimal speaker mapping,
                        DER/JER components, and time-aligned MISS/FA/CONFUSION/
                        OVERLAP regions
    audio/              optional, see audio/README.md

## Why the UI does no arithmetic

The exported error regions are verified at export time to reproduce
pyannote.metrics' own DER components to within a microsecond
(`explorer.verify_against_metric`). Recomputing anything in JavaScript would
create a second implementation free to drift from the one that produced the
benchmark. `data/clips.json` records the verification result.
"""
