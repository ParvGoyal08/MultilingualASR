#!/usr/bin/env python3
"""Run ground-truth alignment QC over every clip and write results/gt_alignment_qc/.

    python3 tools/gt_alignment_qc.py [--root local_out] [--corrections FILE]

Produces:
  candidates.csv            every clip, ranked by evidence for a global offset
  shortlist.csv             the flagged subset, for manual verification
  corrections.json          versioned manifest; only `verified` rows are applied
  diagnostics/<clip>.svg    GT vs consensus vs energy-VAD, before and after
  metrics_raw_vs_qc.csv     the benchmark scored both ways
  README.md                 what was found and what it changes

The raw annotations are never written to. A correction is a manifest row.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sarvam_diar import (config, data, diarization, evaluation,  # noqa: E402
                         gt_qc, reference)
from sarvam_diar.config import Config  # noqa: E402

OUT = Path("results/gt_alignment_qc")


def waveform_envelope(wav_path, n_frames):
    """Per-frame RMS, normalised. The point of drawing this is that a person can
    SEE a misalignment without understanding a word of the language -- speech
    against silence is visible, and nine Indic languages are not something a
    reviewer can be assumed to speak."""
    try:
        import soundfile as sf
    except ImportError:
        return None
    if wav_path is None or not Path(wav_path).exists():
        return None
    a, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    step = int(0.01 * sr)
    usable = min(n_frames, len(a) // step)
    if usable < 50:
        return None
    rms = np.sqrt((a[:usable * step].reshape(usable, step) ** 2).mean(axis=1) + 1e-12)
    out = np.zeros(n_frames, dtype=np.float32)
    out[:usable] = rms / (np.percentile(rms, 99) + 1e-9)
    return np.clip(out, 0, 1)


def svg_diagnostic(path: Path, clip_id: str, cand, detail, vad, corrected_act, wave=None):
    """One self-contained SVG per flagged clip.

    Deliberately not matplotlib: this has to be readable from a repo checkout
    with no plotting stack, and an SVG opens in any browser.
    """
    W, H, pad = 1100, 380, 60
    n = len(detail["ref_act"])
    dur = n * gt_qc.HOP
    x = lambda f: pad + f / max(n, 1) * (W - 2 * pad)

    def band(track, y, h, colour, label):
        parts = [f'<text x="4" y="{y + h - 2}" font-size="11" fill="#555">{label}</text>']
        if track is None:
            parts.append(f'<text x="{pad}" y="{y + h - 2}" font-size="11" '
                         f'fill="#999">unavailable</text>')
            return "".join(parts)
        d = np.diff(np.concatenate(([0], np.asarray(track, dtype=np.int8), [0])))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            parts.append(f'<rect x="{x(s):.1f}" y="{y}" width="{max(1.0, x(e) - x(s)):.1f}" '
                         f'height="{h}" fill="{colour}"/>')
        return "".join(parts)

    # The waveform is drawn FIRST and largest: it is the ground truth about the
    # audio, and every band below is a claim about it.
    if wave is None:
        wavepath = ('<text x="4" y="80" font-size="11" fill="#999">'
                    'waveform unavailable</text>')
    else:
        stepf = max(1, len(wave) // 1400)
        pts = []
        for i in range(0, len(wave), stepf):
            pts.append(f"{x(i):.1f},{80 - float(wave[i]) * 38:.1f}")
        pts += [f"{x(len(wave) - 1):.1f},80", f"{x(0):.1f},80"]
        wavepath = (f'<text x="4" y="48" font-size="11" fill="#555">waveform</text>'
                    f'<polygon points="{" ".join(pts)}" fill="#cfd8e3" stroke="none"/>')

    lags, prof = np.array(detail["lags"]), np.array(detail["profile"])
    px = lambda l: pad + (l - lags[0]) / (lags[-1] - lags[0]) * (W - 2 * pad)
    py = lambda v: 340 - v / max(prof.max(), 1e-6) * 60
    poly = " ".join(f"{px(l):.1f},{py(v):.1f}" for l, v in zip(lags, prof))

    ticks = "".join(
        f'<line x1="{x(int(t / gt_qc.HOP)):.1f}" y1="30" x2="{x(int(t / gt_qc.HOP)):.1f}" '
        f'y2="186" stroke="#eee"/><text x="{x(int(t / gt_qc.HOP)):.1f}" y="200" '
        f'font-size="10" fill="#999" text-anchor="middle">{t:.0f}s</text>'
        for t in np.linspace(0, dur, 9))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="ui-sans-serif,system-ui,sans-serif">
<rect width="{W}" height="{H}" fill="#fff"/>
<text x="8" y="18" font-size="14" font-weight="600">{clip_id}</text>
<text x="8" y="34" font-size="11" fill="#666">candidate offset {cand.best_lag:+.2f}s
 · IoU {cand.zero_lag_iou:.3f} at lag 0 -&gt; {cand.peak_iou:.3f} at peak
 (+{cand.improvement:.3f}) · margin {cand.peak_margin:.3f}
 · {cand.n_models} models, spread {cand.model_spread:.2f}s
 · VAD {"agrees" if cand.vad_agrees else "differs"}{f" ({cand.vad_lag:+.2f}s)" if cand.vad_lag is not None else ""}</text>
{ticks}
{wavepath}
{band(detail["ref_act"], 92, 18, "#c0392b", "GT raw")}
{band(corrected_act, 116, 18, "#e67e22", "GT shifted")}
{band(detail["consensus"], 140, 18, "#2c6fbb", "consensus")}
{band(vad, 164, 18, "#7f8c8d", "energy VAD")}
<text x="8" y="230" font-size="11" fill="#555">IoU vs lag</text>
<line x1="{pad}" y1="340" x2="{W - pad}" y2="340" stroke="#ddd"/>
<line x1="{px(0):.1f}" y1="280" x2="{px(0):.1f}" y2="340" stroke="#bbb" stroke-dasharray="3,3"/>
<text x="{px(0):.1f}" y="355" font-size="10" fill="#999" text-anchor="middle">0</text>
<polyline points="{poly}" fill="none" stroke="#2c6fbb" stroke-width="1.5"/>
<line x1="{px(cand.best_lag):.1f}" y1="280" x2="{px(cand.best_lag):.1f}" y2="340"
      stroke="#c0392b" stroke-width="1.5"/>
<text x="{px(cand.best_lag):.1f}" y="372" font-size="10" fill="#c0392b"
      text-anchor="middle">{cand.best_lag:+.2f}s</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="local_out")
    ap.add_argument("--corrections", default=str(OUT / "corrections.json"))
    # pyannote-3.1 is excluded by default. It and community-1 share
    # pyannote/segmentation-3.0 and differ by 0.44 s of miss+FA over 12.4 h
    # (obs [18]), so under 2-of-N voting they are effectively one vote counted
    # twice: any frame both call speech reaches the threshold on its own,
    # without a second INDEPENDENT system ever agreeing. Consensus is only
    # meaningful over systems that can disagree.
    ap.add_argument("--models", default="community-1,reverb-v2,diarizen-large",
                    help="comma-separated; consensus is only built from these")
    args = ap.parse_args()

    cfg = Config.create(root=args.root, work_dir=f"{args.root}/.work")
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)

    clips = data.parse_ground_truth(data.load_segments_csv(cfg))
    inputs, _ = data.split_reference(clips, None, cfg=cfg)
    refs = {c.clip_id: reference.build_reference(c) for c in clips if c.clip_id in inputs}
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    models = [m for m in wanted if any(diarization.is_done(cfg, m, cid) for cid in refs)]
    missing = [m for m in wanted if m not in models]
    if missing:
        print(f"  requested but absent on disk: {missing}")
    print(f"  consensus from: {models}")
    # Scoring still covers every model that ran, including any excluded from the
    # consensus -- a model that did not vote should still be measured.
    scored = [m for m in config.scored_models()
              if any(diarization.is_done(cfg, m, cid) for cid in refs)]

    cands, details = [], {}
    for cid, ref in refs.items():
        per_model = {m: diarization.load_hypothesis(cfg, m, cid)
                     for m in models if diarization.is_done(cfg, m, cid)}
        if not per_model:
            continue
        wav = cfg.wav_path(cid)
        cand, detail = gt_qc.assess_clip(cid, ref.turns, ref.uem[1], per_model,
                                         wav if wav.exists() else None)
        cands.append(cand)
        details[cid] = detail

    cands.sort(key=lambda c: (-int(c.flagged), -abs(c.best_lag), -c.improvement))
    fields = list(cands[0].as_row())
    with open(OUT / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for c in cands:
            w.writerow(c.as_row())
    flagged = [c for c in cands if c.flagged]
    with open(OUT / "shortlist.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["proposed_offset_sec", "verified", "note"])
        w.writeheader()
        for c in flagged:
            w.writerow({**c.as_row(),
                        "proposed_offset_sec": gt_qc.round_correction(c.best_lag),
                        "verified": "", "note": ""})

    # diagnostics for the flagged clips only
    for c in flagged:
        d = details[c.clip_id]
        ref = refs[c.clip_id]
        n = len(d["ref_act"])
        corrected = gt_qc.activity(
            gt_qc.apply_correction(ref.turns, gt_qc.round_correction(c.best_lag),
                                   ref.uem[0], ref.uem[1]), n)
        wav = cfg.wav_path(c.clip_id)
        vad = gt_qc.energy_vad(wav, n) if wav.exists() else None
        wave = waveform_envelope(wav if wav.exists() else None, n)
        svg_diagnostic(OUT / "diagnostics" / f"{c.clip_id}.svg",
                       c.clip_id, c, d, vad, corrected, wave)

    # manifest: pre-populated as UNVERIFIED, except where two independent
    # signals agree, which is recorded as auto-corroborated but still flagged
    # for a human to confirm.
    manifest_path = Path(args.corrections)
    existing = {}
    if manifest_path.exists():
        existing = {r["clip_id"]: r for r in json.loads(manifest_path.read_text())
                    .get("corrections", [])}
    rows = []
    for c in flagged:
        prev = existing.get(c.clip_id, {})
        rows.append({
            "clip_id": c.clip_id,
            "offset_sec": prev.get("offset_sec", gt_qc.round_correction(c.best_lag)),
            "detected_lag_sec": round(c.best_lag, 3),
            "iou_gain": round(c.improvement, 4),
            "headroom_recovered": round(c.headroom_recovered, 3),
            "peak_margin": round(c.peak_margin, 4),
            "model_spread_sec": round(c.model_spread, 3),
            "vad_corroborates": bool(c.vad_agrees),
            "verified": prev.get("verified", False),
            "verified_by": prev.get("verified_by", ""),
            "note": prev.get("note", ""),
        })
    write = {"version": 1, "grid_sec": gt_qc.ROUND_TO,
             "convention": "offset_sec is SUBTRACTED from every reference timestamp",
             "detector": {"hop_sec": gt_qc.HOP, "max_lag_sec": gt_qc.MAX_LAG,
                          "min_votes": 2, "consensus_models": models,
                          "excluded_from_consensus":
                              [m for m in config.scored_models() if m not in models],
                          "min_flag_lag": gt_qc.MIN_FLAG_LAG,
                          "min_improvement": gt_qc.MIN_IMPROVEMENT,
                          "min_headroom_recovered": gt_qc.MIN_HEADROOM_RECOVERED,
                          "min_peak_margin": gt_qc.MIN_PEAK_MARGIN},
             "raw_gt": "immutable; corrections are applied to a copy at scoring time",
             "corrections": rows}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(write, indent=1), encoding="utf-8")

    # scoring, raw vs QC-adjusted (verified rows only)
    verified = gt_qc.load_corrections(manifest_path)
    out_rows = []
    for m in scored:
        for label, corr in (("raw", {}), ("qc_adjusted", verified)):
            rows_m = []
            for cid, ref in refs.items():
                if not diarization.is_done(cfg, m, cid):
                    continue
                turns = ref.turns
                off = corr.get(cid, 0.0)
                if off:
                    turns = gt_qc.apply_correction(turns, off, ref.uem[0], ref.uem[1])
                import dataclasses
                r2 = dataclasses.replace(ref, turns=turns)
                rows_m.append(evaluation.score_clip(r2, diarization.load_hypothesis(cfg, m, cid)))
            p = evaluation.pool(rows_m)
            out_rows.append({"model": m, "gt_version": label, "n_clips": p["n_clips"],
                             "der": round(p["der"], 4),
                             "miss": round(p["der_miss_frac"], 4),
                             "fa": round(p["der_fa_frac"], 4),
                             "confusion": round(p["der_confusion_frac"], 4),
                             "jer_mean": round(p["jer_mean"], 4),
                             "speaker_count_acc": round(p["speaker_count_accuracy"], 4),
                             "speaker_count_mae": round(p["speaker_count_mae"], 4),
                             "speaker_count_bias": round(p["speaker_count_bias"], 4),
                             "in_consensus": m in models,
                             "n_corrected": len(corr)})
    with open(OUT / "metrics_raw_vs_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader()
        w.writerows(out_rows)

    print(f"  {len(cands)} clips assessed, {len(flagged)} flagged")
    print(f"  corroborated by the energy VAD: "
          f"{sum(1 for c in flagged if c.vad_agrees)}/{len(flagged)}")
    print(f"  verified corrections applied: {len(verified)}")
    print(f"  -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
