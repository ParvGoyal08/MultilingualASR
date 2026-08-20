#!/usr/bin/env python3
"""Bring an error-explorer export produced elsewhere up to the current schema.

A sweep runs on Kaggle or Colab against whatever commit that session pulled, so
its export can predate fields the UI now uses. Rather than re-run the sweep to
regenerate them, this recomputes locally everything that does not require the
hypotheses, links the audio, and installs the current UI.

    python3 tools/adopt_export.py <export_dir> [--audio local_out/audio_16k]

What it fills in, and why each is safe to compute here:

  script / lang   dominant Unicode script of the reference transcript; a pure
                  function of the ground truth, which is local.
  total_sec       the DER denominator (summed per-speaker reference duration
                  inside the UEM). Verified, not assumed: it must reproduce
                  every exported DER as error_sec / total_sec, and the script
                  refuses to write if it does not.
  overlap_frac    fraction of reference time with 2+ speakers active.
  has_audio       recorded at export time, so stale the moment audio is linked.

Region counts (`n`) are NOT backfilled -- they need pyannote's matcher over the
hypotheses. Their absence is safe: the UI's residual() returns 0 without them,
and the check below reports whether any model could actually need them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sarvam_diar import data, reference  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "sarvam_diar" / "explorer_assets"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=pathlib.Path)
    ap.add_argument("--audio", type=pathlib.Path, default=pathlib.Path("local_out/audio_16k"))
    ap.add_argument("--root", default="local_out", help="Config root for the ground truth")
    ap.add_argument("--copy-audio", action="store_true",
                    help="copy instead of symlink (for moving the export to another machine)")
    args = ap.parse_args()

    exp = args.export_dir
    idx_path = exp / "data" / "clips.json"
    if not idx_path.exists():
        print(f"no export at {idx_path}", file=sys.stderr)
        return 1

    cfg = Config.create(root=args.root, work_dir=f"{args.root}/.work")
    refs = {c.clip_id: reference.build_reference(c)
            for c in data.parse_ground_truth(data.load_segments_csv(cfg))}

    idx = json.loads(idx_path.read_text())
    checked = bad = 0
    for e in idx["clips"]:
        r = refs.get(e["clip_id"])
        if r is None:
            e.setdefault("script", "unknown")
            continue
        e["script"] = r.stats.get("lang_script") or "unknown"
        e["lang"] = r.stats.get("lang_hint") or "unknown"
        e["overlap_frac"] = round(r.stats.get("overlap_frac", 0.0), 4)
        total = sum(t.end - t.start for t in r.turns)
        for m in e["models"].values():
            m["total_sec"] = round(total, 3)
            if m.get("der") is not None and total > 0:
                checked += 1
                if abs(m["error_sec"] / total - m["der"]) > 2e-3:
                    bad += 1
    if bad:
        print(f"REFUSING TO WRITE: computed total_sec fails to reproduce DER on "
              f"{bad}/{checked} clip-model pairs", file=sys.stderr)
        return 2
    print(f"  total_sec reproduces DER on {checked}/{checked} clip-model pairs")

    # audio
    (exp / "audio").mkdir(exist_ok=True)
    n = 0
    for wav in sorted(args.audio.glob("*.wav")):
        dst = exp / "audio" / wav.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy_audio:
            shutil.copyfile(wav, dst)
        else:
            dst.symlink_to(wav.resolve())
        n += 1
    for e in idx["clips"]:
        e["has_audio"] = (exp / "audio" / f"{e['clip_id']}.wav").exists()
    print(f"  audio {'copied' if args.copy_audio else 'linked'}: {n} files, "
          f"{sum(e['has_audio'] for e in idx['clips'])}/{len(idx['clips'])} clips playable")

    idx_path.write_text(json.dumps(idx, ensure_ascii=False))

    # would any model actually need the region counts we cannot backfill?
    worst = 0
    for e in idx["clips"]:
        j = json.loads((exp / "data" / f"{e['clip_id']}.json").read_text())
        for block in j["models"].values():
            h = sorted(block["hypothesis"], key=lambda t: t["start"])
            for i, a in enumerate(h):
                for b in h[i + 1:]:
                    if b["start"] >= a["end"]:
                        break
                    if a["speaker"] == b["speaker"]:
                        worst += 1
    print(f"  self-overlapping same-speaker turns across all models: {worst}"
          f"{'  (region counts unnecessary)' if worst == 0 else '  <- RE-EXPORT to get region counts'}")

    # explorer_assets is itself a valid export target during development, and
    # SameFileError is not a failure -- the UI is already the current one.
    if ASSETS.resolve() != exp.resolve():
        for name in ("index.html", "serve.py"):
            shutil.copyfile(ASSETS / name, exp / name)
        (exp / "serve.py").chmod(0o755)
    scripts = sorted({e.get("script", "unknown") for e in idx["clips"]})
    print(f"  UI installed · models={idx['models']} · scripts={scripts}")
    print(f"\n  cd {exp} && python3 serve.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
