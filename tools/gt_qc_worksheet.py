#!/usr/bin/env python3
"""Manual-verification worksheet for the GT alignment QC.

The check uses the TRANSCRIPT, not speech activity. For a chosen utterance the
ground truth makes a falsifiable claim -- "these words are spoken starting at
t" -- and the QC makes a competing one -- "no, they are spoken at t - offset".
A person jumps to both timestamps and says which one has those words in it.

This is the right check for three reasons. It works on clips that open with
speech, where a first-onset comparison is useless because both sides say 0.01 s.
It is lexical, so it shares nothing with the acoustic models that proposed the
offset -- which is exactly the circularity a reviewer would press on. And it
needs no tooling: a browser and a pair of ears settle it.

CONTROL clips that were NOT flagged are included. Verifying only flagged clips
can confirm what the detector already believes; controls are what show it is not
simply flagging everything.
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sarvam_diar import data, reference  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402

OUT = Path("results/gt_alignment_qc")
N_FLAGGED, N_CONTROL = 10, 6


def pick_utterance(ref, offset: float):
    """A long, well-isolated utterance from the middle of the clip.

    Long so there is enough text to recognise; isolated so a neighbouring
    speaker cannot be mistaken for it; mid-clip because the opening seconds are
    where intros and music make listening ambiguous. It must also stay inside
    the audio once the offset is removed, or the check has nothing to play.
    """
    lo, hi = ref.uem[1] * 0.15, ref.uem[1] * 0.85
    best, score = None, -1.0
    utts = sorted(ref.utterances, key=lambda u: u.start)
    for i, u in enumerate(utts):
        if not (lo <= u.start <= hi) or u.start - offset < 1.0:
            continue
        words = len((u.text_norm or "").split())
        if words < 6:
            continue
        gap_before = u.start - utts[i - 1].end if i else 5.0
        gap_after = utts[i + 1].start - u.end if i + 1 < len(utts) else 5.0
        s = min(words, 25) + 3.0 * min(gap_before, 1.5) + 3.0 * min(gap_after, 1.5)
        if s > score:
            best, score = u, s
    return best


def main() -> int:
    cfg = Config.create(root="local_out", work_dir="local_out/.work")
    clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(cfg))}
    cand = {r["clip_id"]: r for r in csv.DictReader(open(OUT / "candidates.csv"))}
    flagged = {r["clip_id"]: r
               for r in json.loads((OUT / "corrections.json").read_text())["corrections"]}

    fl = sorted(flagged.values(), key=lambda r: -abs(r["detected_lag_sec"]))
    step = max(1, len(fl) // N_FLAGGED)
    picks = [("flagged", r) for r in fl[::step][:N_FLAGGED]]
    unfl = sorted((r for cid, r in cand.items() if cid not in flagged),
                  key=lambda r: -abs(float(r["best_lag"])))
    picks += [("control", r) for r in unfl[:3] + unfl[-3:]]

    rows = []
    for kind, r in picks:
        cid = r["clip_id"]
        clip = clips[cid]
        ref = reference.build_reference(clip)
        off = float(r.get("offset_sec", 0) or 0) if kind == "flagged" else 0.0
        u = pick_utterance(ref, off)
        if u is None:
            continue
        abs_gt = clip.start_sec + u.start
        abs_qc = clip.start_sec + max(0.0, u.start - off)
        rows.append({
            "kind": kind, "clip_id": cid,
            "detected_lag_sec": round(float(r.get("detected_lag_sec",
                                                  r.get("best_lag", 0))), 2),
            "proposed_offset_sec": off,
            "speaker": u.speaker,
            "gt_says_spoken_at_sec": round(u.start, 2),
            "qc_says_spoken_at_sec": round(max(0.0, u.start - off), 2),
            "transcript": (u.text_raw or u.text_norm or "")[:160],
            "listen_gt_claim": f"https://youtu.be/{clip.video_id}?t={int(abs_gt)}",
            "listen_qc_claim": f"https://youtu.be/{clip.video_id}?t={int(abs_qc)}",
            "diagnostic": (f"results/gt_alignment_qc/diagnostics/{cid}.svg"
                           if kind == "flagged" else ""),
            # fill these in by hand
            "which_is_right": "", "heard_at_sec": "", "checked_by": "", "note": "",
        })

    path = OUT / "verification_worksheet.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"  {sum(1 for r in rows if r['kind']=='flagged')} flagged + "
          f"{sum(1 for r in rows if r['kind']=='control')} controls -> {path}\n")
    print(f"  {'kind':<9}{'clip':<24}{'off':>6}{'GT@':>8}{'QC@':>8}  transcript")
    for r in rows:
        print(f"  {r['kind']:<9}{r['clip_id']:<24}{r['proposed_offset_sec']:>+6.1f}"
              f"{r['gt_says_spoken_at_sec']:>8.2f}{r['qc_says_spoken_at_sec']:>8.2f}"
              f"  {r['transcript'][:46]}")
    print("\n  For each row: open both links. The words in `transcript` are audible")
    print("  at exactly one of them. Put 'GT' or 'QC' in which_is_right.")
    print("  Controls have offset 0, so both links are the same moment: the words")
    print("  should simply be there. A control where they are NOT is a detector miss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
