#!/usr/bin/env python3
"""Build (and read back) the hand-labelling worksheet for GT alignment.

    python3 tools/gt_hand_label.py            # build the worksheet
    python3 tools/gt_hand_label.py --report   # read verdicts, score, summarise

Framing. The automatic detector is a TRIAGE tool: it decides which clips are
worth listening to, nothing more. What gets applied to any reported number is
what a human confirmed by ear. That distinction is the whole point -- an
automatic correction derived from the models being scored is open to the
objection that it moves the target toward the shooters, while a hand-checked
label is not.

Two consequences follow, and both are designed for here:

* **Rank by impact, accept by ear.** Rows are ordered by how much correcting
  them would move JER, because listening time is finite and should go where it
  matters. That ordering must never become the acceptance test -- selecting
  clips by how much they flatter the metric and then accepting them because
  they flatter the metric is circular. The `verdict` column is the only thing
  that decides.

* **Audit the negatives too.** The sheet includes low-impact flagged clips and
  clips the detector did NOT flag. Checking only high-impact candidates can
  measure precision at best; it cannot notice a defect the detector missed, and
  it cannot notice the detector crying wolf on clips that are actually fine.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sarvam_diar import data, diarization, evaluation, gt_qc, reference  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402

OUT = Path("results/gt_alignment_qc")
SHEET = OUT / "hand_labels.csv"
MODELS = ["community-1", "reverb-v2", "diarizen-large"]
N_LOW_IMPACT, N_CONTROL = 5, 8


def impact(ref, cid, cfg, shift):
    """Mean DER and JER across models, before and after a candidate shift.

    Averaged over the three distinct systems rather than taken from one, so the
    ranking is not steered by a single model's quirks -- and in particular so a
    clip is not promoted just because it happens to trouble the weakest system.
    """
    hyps = {m: diarization.load_hypothesis(cfg, m, cid)
            for m in MODELS if diarization.is_done(cfg, m, cid)}
    if not hyps:
        return None
    shifted = dataclasses.replace(
        ref, turns=gt_qc.apply_correction(ref.turns, shift, ref.uem[0], ref.uem[1]))
    raw = [evaluation.score_clip(ref, h) for h in hyps.values()]
    fix = [evaluation.score_clip(shifted, h) for h in hyps.values()]
    n = len(raw)
    return {
        "der_raw": sum(r["der"] for r in raw) / n,
        "der_fixed": sum(r["der"] for r in fix) / n,
        "jer_raw": sum(r["jer"] for r in raw) / n,
        "jer_fixed": sum(r["jer"] for r in fix) / n,
    }


def pick_utterance(ref, offset):
    """A long, isolated, mid-clip utterance -- the thing the human listens for."""
    lo, hi = ref.uem[1] * 0.15, ref.uem[1] * 0.85
    utts = sorted(ref.utterances, key=lambda u: u.start)
    best, score = None, -1.0
    for i, u in enumerate(utts):
        if not (lo <= u.start <= hi) or u.start - offset < 1.0:
            continue
        words = len((u.text_norm or "").split())
        if words < 6:
            continue
        gb = u.start - utts[i - 1].end if i else 5.0
        ga = utts[i + 1].start - u.end if i + 1 < len(utts) else 5.0
        s = min(words, 25) + 3.0 * min(gb, 1.5) + 3.0 * min(ga, 1.5)
        if s > score:
            best, score = u, s
    return best


def build(cfg, refs, clips):
    cand = {r["clip_id"]: r for r in csv.DictReader(open(OUT / "candidates.csv"))}
    flagged = [r for r in cand.values() if r["flagged"] == "True"]
    unflagged = [r for r in cand.values() if r["flagged"] != "True"]

    rows = []
    for r in flagged:
        cid = r["clip_id"]
        shift = gt_qc.round_correction(float(r["best_lag"]))
        imp = impact(refs[cid], cid, cfg, shift)
        if imp:
            rows.append((cid, "flagged", shift, r, imp))
    rows.sort(key=lambda x: -(x[4]["jer_raw"] - x[4]["jer_fixed"]))

    # low-impact flagged rows the ranking would otherwise bury
    tail = rows[-N_LOW_IMPACT:] if len(rows) > N_LOW_IMPACT else []
    head = rows[:-N_LOW_IMPACT] if len(rows) > N_LOW_IMPACT else rows

    # controls: clips NOT flagged, scored as if shifted by their own best lag,
    # so a real defect the detector missed would show a large gain here
    ctrl = []
    unflagged.sort(key=lambda r: -abs(float(r["best_lag"])))
    for r in unflagged[:N_CONTROL]:
        cid = r["clip_id"]
        shift = gt_qc.round_correction(float(r["best_lag"]))
        imp = impact(refs[cid], cid, cfg, shift)
        if imp:
            ctrl.append((cid, "control", shift, r, imp))

    out = []
    for rank, (cid, kind, shift, r, imp) in enumerate(head + tail + ctrl, 1):
        clip = clips[cid]
        ref = refs[cid]
        u = pick_utterance(ref, shift)
        gt_t = clip.start_sec + (u.start if u else 0.0)
        qc_t = clip.start_sec + max(0.0, (u.start if u else 0.0) - shift)
        out.append({
            "rank": rank if kind == "flagged" else "",
            "kind": kind,
            "clip_id": cid,
            "proposed_shift_sec": shift,
            "jer_raw": round(imp["jer_raw"], 4),
            "jer_if_shifted": round(imp["jer_fixed"], 4),
            "jer_gain": round(imp["jer_raw"] - imp["jer_fixed"], 4),
            "der_raw": round(imp["der_raw"], 4),
            "der_if_shifted": round(imp["der_fixed"], 4),
            "der_gain": round(imp["der_raw"] - imp["der_fixed"], 4),
            "headroom_recovered": r.get("headroom_recovered", ""),
            "model_spread_sec": round(float(r["model_spread"]), 2),
            "vad_agrees": r["vad_agrees"],
            "gt_says_at_sec": round(u.start, 2) if u else "",
            "shift_says_at_sec": round(max(0.0, u.start - shift), 2) if u else "",
            "transcript": (u.text_raw or "")[:150] if u else "",
            "listen_gt": f"https://youtu.be/{clip.video_id}?t={int(gt_t)}",
            "listen_shifted": f"https://youtu.be/{clip.video_id}?t={int(qc_t)}",
            "diagnostic": (f"diagnostics/{cid}.svg" if kind == "flagged" else ""),
            # ---- filled in by hand ----
            "verdict": "",              # CONFIRM | REJECT | UNSURE
            "confirmed_shift_sec": "",  # blank = accept proposed_shift_sec
            "checked_by": "",
            "note": "",
        })

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(SHEET, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    tot = sum(r["jer_gain"] for r in out if r["kind"] == "flagged")
    run = 0.0
    n80 = 0
    for r in out:
        if r["kind"] != "flagged":
            continue
        run += r["jer_gain"]
        n80 += 1
        if run >= 0.8 * tot:
            break
    print(f"  {sum(1 for r in out if r['kind']=='flagged')} flagged "
          f"+ {sum(1 for r in out if r['kind']=='control')} controls -> {SHEET}")
    print(f"  total JER gain available: {tot:.3f}")
    print(f"  the top {n80} rows carry 80% of it -- checking those settles most of the question\n")
    print(f"  {'#':>3} {'kind':<8}{'clip':<26}{'shift':>7}{'JER':>8}{'->':>8}{'gain':>8}{'DER gain':>10}")
    for r in out[:14]:
        print(f"  {str(r['rank']):>3} {r['kind']:<8}{r['clip_id']:<26}"
              f"{r['proposed_shift_sec']:>+7.1f}{r['jer_raw']:>8.3f}{r['jer_if_shifted']:>8.3f}"
              f"{r['jer_gain']:>+8.3f}{r['der_gain']:>+10.3f}")
    return 0


def report(cfg, refs):
    if not SHEET.exists():
        print(f"  {SHEET} not found -- build it first")
        return 1
    rows = list(csv.DictReader(open(SHEET)))
    done = [r for r in rows if r["verdict"].strip()]
    if not done:
        print(f"  no verdicts recorded yet in {SHEET}")
        return 0

    conf = [r for r in done if r["verdict"].strip().upper() == "CONFIRM"]
    rej = [r for r in done if r["verdict"].strip().upper() == "REJECT"]
    unsure = [r for r in done if r["verdict"].strip().upper() == "UNSURE"]
    fl = [r for r in done if r["kind"] == "flagged"]
    ct = [r for r in done if r["kind"] == "control"]
    fl_conf = [r for r in fl if r["verdict"].strip().upper() == "CONFIRM"]
    ct_conf = [r for r in ct if r["verdict"].strip().upper() == "CONFIRM"]

    print(f"  checked {len(done)} of {len(rows)} rows: "
          f"{len(conf)} CONFIRM, {len(rej)} REJECT, {len(unsure)} UNSURE")
    if fl:
        print(f"  detector precision on audited flags: {len(fl_conf)}/{len(fl)} "
              f"= {len(fl_conf)/len(fl):.0%}")
    if ct:
        print(f"  audited controls confirmed as ALSO misaligned: {len(ct_conf)}/{len(ct)}"
              + ("   <- detector misses" if ct_conf else "   (none: no misses found)"))

    labels = {}
    for r in conf:
        s = r["confirmed_shift_sec"].strip() or r["proposed_shift_sec"]
        labels[r["clip_id"]] = float(s)
    path = OUT / "hand_labels.json"
    path.write_text(json.dumps({
        "version": 1, "grid_sec": gt_qc.ROUND_TO,
        "provenance": "hand-verified by ear against the source video",
        "convention": "shift_sec is SUBTRACTED from every reference timestamp",
        "audited": len(done), "confirmed": len(conf), "rejected": len(rej),
        "labels": [{"clip_id": k, "shift_sec": v} for k, v in sorted(labels.items())],
    }, indent=1))
    print(f"\n  {len(labels)} hand-verified labels -> {path}")

    for m in MODELS:
        rows_raw, rows_fix = [], []
        for cid, ref in refs.items():
            if not diarization.is_done(cfg, m, cid):
                continue
            h = diarization.load_hypothesis(cfg, m, cid)
            rows_raw.append(evaluation.score_clip(ref, h))
            s = labels.get(cid, 0.0)
            r2 = (dataclasses.replace(ref, turns=gt_qc.apply_correction(
                ref.turns, s, ref.uem[0], ref.uem[1])) if s else ref)
            rows_fix.append(evaluation.score_clip(r2, h))
        a, b = evaluation.pool(rows_raw), evaluation.pool(rows_fix)
        print(f"    {m:<16} DER {a['der']:.4f} -> {b['der']:.4f}   "
              f"JER {a['jer_mean']:.4f} -> {b['jer_mean']:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="local_out")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = Config.create(root=args.root, work_dir=f"{args.root}/.work")
    clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(cfg))}
    inputs, _ = data.split_reference(list(clips.values()), None, cfg=cfg)
    refs = {cid: reference.build_reference(clips[cid]) for cid in inputs}
    return report(cfg, refs) if args.report else build(cfg, refs, clips)


if __name__ == "__main__":
    raise SystemExit(main())
