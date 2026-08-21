#!/usr/bin/env python3
"""Does each stored per-segment transcript match the segmentation we would use?

Reports the re-run set without transcribing anything. Run this BEFORE a sweep:
run_segmented now refuses to skip a mismatched clip, so an unchecked re-run
silently spends money on however many clips drifted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import asr, data, diarization  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--system", default="sarvam-saaras-v3")
    ap.add_argument("--diar", default="fusion")
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = a.root or next((p for p in ("/kaggle/working/sarvam_diarization",
                                       "/content/sarvam_diarization", "local_out")
                           if Path(p).exists()), "local_out")
    cfg = Config.create(root=Path(root), work_dir=Path(root) / "tmp")
    tag = f"{a.system}@{a.diar}"
    ids = [c.clip_id for c in data.parse_ground_truth(data.load_segments_csv(cfg))
           if cfg.wav_path(c.clip_id).exists()]
    rows = asr.audit_segmentation(cfg, tag, a.diar, ids, merge_gap=a.merge_gap)

    by = {}
    for r in rows:
        by.setdefault(r["status"], []).append(r)
    print(f"{tag}  merge_gap={a.merge_gap}  root={root}\n")
    for k in sorted(by):
        print(f"  {k:<16}{len(by[k]):>4}")

    bad = by.get("MISMATCH", [])
    if bad:
        print(f"\n{len(bad)} clips must be re-transcribed:")
        print(f"  {'clip':<28}{'stored':>14}{'expected':>14}{'longest stored':>16}")
        for r in sorted(bad, key=lambda r: -(r["stored_sec"] - r["expected_sec"])):
            print(f"  {r['clip_id']:<28}"
                  f"{r['stored_segments']:>5}/{r['stored_sec']:>7.0f}s"
                  f"{r['expected_segments']:>6}/{r['expected_sec']:>7.0f}s"
                  f"{r['longest_stored_sec']:>15.0f}s")
        d = sum(r["stored_sec"] for r in bad)
        print(f"\n  {d/3600:.2f} h of audio to re-send "
              f"(~Rs{45*d/3600:.0f} at Saaras's ~Rs45/h)")
    else:
        print("\nevery stored transcript matches the current segmentation")

    legacy = [r for r in rows if r.get("status") == "ok" and not r.get("stored_key")]
    if legacy:
        print(f"\n{len(legacy)} clips match but predate segmentation_key -- they are "
              "correct, and will gain the fingerprint next time they are written")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
