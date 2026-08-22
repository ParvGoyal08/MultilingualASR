#!/usr/bin/env python3
"""Step 5 pilot: Gemini transcript refinement, arms A and B, before vs after.

Clip selection is GT-BLIND. Nothing derived from the reference -- not WER, not
cpWER, not WDER, not reference overlap -- is used to choose clips, because
picking clips by their measured error is ground truth leaking into the
experimental design and biases the pilot toward clips where the stage flatters
itself. Selection uses only the hypothesis and the diarization.

The set drawn here is a DEVELOPMENT/PILOT set, not a test set. It is drawn from
the dev half of results/split.json only; the held-out 49 are never touched
during development and are scored once, at the end, with everything frozen.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import asr, data, llm_refine as lr, reference, text_metrics as tm  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402
from sarvam_diar.utils import append_jsonl, read_json, write_json_atomic  # noqa: E402

import os
SOURCE = os.environ.get("XLIT_SOURCE", "sarvam-saaras-v3@fusion")
SEED = 20260822


# --------------------------------------------------------------- GT-blind pick


def clip_features(cfg: Config, clip_id: str) -> dict | None:
    """Every feature here comes from the hypothesis or the diarization."""
    p = asr.asr_path(cfg, SOURCE, clip_id)
    if not p.exists():
        return None
    d = read_json(p)
    segs = d.get("segments") or []
    if not segs:
        return None
    dur = d.get("clip_dur_sec") or max(s["end"] for s in segs)
    ov = sum(o for _, _, o in lr.overlap_pairs(segs))
    dup = sum(len(v) for v in lr.dup_positions(segs).values())
    text = " ".join((s.get("text") or "") for s in segs)
    script, _, _ = data.dominant_script(text)          # hypothesis script, not reference
    return {"clip_id": clip_id, "script": script, "dur": float(dur),
            "n_seg": len(segs), "ovl_frac": ov / max(dur, 1e-9),
            "dup_tokens": dup, "n_spk": len({s["speaker"] for s in segs})}


def select_pilot(cfg: Config, n: int = 10, seed: int = SEED,
                 max_sec: float = 2700.0) -> list[dict]:
    """Script-stratified, seeded, spanning predicted overlap and size."""
    split = json.loads(Path("results/split.json").read_text())
    dev = set(split["dev"])
    ids = [c.clip_id for c in data.parse_ground_truth(data.load_segments_csv(cfg))
           if c.clip_id in dev]
    audit = {r["clip_id"]: r["status"]
             for r in asr.audit_segmentation(cfg, SOURCE, "fusion", ids)}
    feats = [f for cid in ids if audit.get(cid) == "ok"
             for f in [clip_features(cfg, cid)] if f]

    by_script: dict[str, list[dict]] = defaultdict(list)
    for f in feats:
        by_script[f["script"]].append(f)
    rng = random.Random(seed)

    # one per script, rotating the target overlap percentile so the set spans
    # the range instead of clustering at the top
    targets = [0.9, 0.5, 0.1]
    picked: list[dict] = []
    for k, script in enumerate(sorted(by_script)):
        rows = sorted(by_script[script], key=lambda f: f["ovl_frac"])
        q = targets[k % len(targets)]
        picked.append(rows[min(len(rows) - 1, int(q * (len(rows) - 1)))])

    # fill to n by widening duration / segment-count coverage
    rest = [f for f in feats if f["clip_id"] not in {p["clip_id"] for p in picked}]
    rng.shuffle(rest)
    while len(picked) < n and rest:
        have = [p["n_seg"] for p in picked]
        rest.sort(key=lambda f: -min(abs(f["n_seg"] - h) for h in have))
        picked.append(rest.pop(0))

    picked.sort(key=lambda f: -f["dur"])
    while sum(p["dur"] for p in picked) > max_sec and len(picked) > 1:
        drop = picked.pop(0)
        pool = [f for f in feats
                if f["script"] == drop["script"]
                and f["clip_id"] not in {p["clip_id"] for p in picked}]
        if pool:
            picked.append(min(pool, key=lambda f: f["dur"]))
        picked.sort(key=lambda f: -f["dur"])
    return sorted(picked, key=lambda f: f["clip_id"])


# --------------------------------------------------------------- scoring


def score(cfg: Config, system: str, clip_ids, refs) -> list[dict]:
    rows = []
    for cid in clip_ids:
        if not asr.is_done(cfg, system, cid):
            continue
        pr = asr.load_pairs(cfg, system, cid)
        ref = refs[cid]
        r = tm.score_transcript(
            {k: v.split() for k, v in reference.speaker_texts(ref).items()},
            asr.speaker_texts_from_words(pr), reference.word_stream(ref), pr)
        r.update(clip_id=cid, n_hyp=len(pr))
        rows.append(r)
    return rows


def line(label: str, rows: list[dict]) -> str:
    g = tm.summarise(rows)
    ratio = sum(r["n_hyp"] for r in rows) / max(g["n_ref_words"], 1)
    ins = sum(r["ins"] for r in rows)
    return (f"{label:<34}{g['n_clips']:>4}{ratio:>8.4f}{g['wer']:>9.4f}"
            f"{g['cpwer']:>9.4f}{g['wder']:>9.4f}{ins:>8,}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--model", default=lr.DEFAULT_MODEL)
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--k", type=int, default=lr.K_CORE,
                    help="core segments per window; large = one window per clip")
    ap.add_argument("--k-max", type=int, default=lr.K_MAX)
    ap.add_argument("--clip-workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="render windows, no API calls")
    a = ap.parse_args()

    root = a.root or next((p for p in ("local_out/step4_input",
                                       "/kaggle/working/sarvam_diarization", "local_out")
                           if Path(p).exists()), "local_out")
    cfg = Config.create(root=Path(root), work_dir=Path(root) / "tmp")
    meta_cfg = Config.create(root=Path("local_out"), work_dir=Path("./.work"))
    clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(meta_cfg))}

    pilot = select_pilot(cfg, a.clips)
    print(f"root={root}  model={a.model}  prompt={lr.PROMPT_VERSION}\n")
    print("GT-BLIND pilot set (dev only; features from hypothesis + diarization):")
    print(f"  {'clip':<26}{'script':<12}{'sec':>6}{'segs':>6}{'ovl%':>7}{'dup tok':>9}{'spk':>5}")
    for f in pilot:
        print(f"  {f['clip_id']:<26}{f['script']:<12}{f['dur']:>6.0f}{f['n_seg']:>6}"
              f"{f['ovl_frac']:>7.1%}{f['dup_tokens']:>9}{f['n_spk']:>5}")
    ids = [f["clip_id"] for f in pilot]
    print(f"  total {sum(f['dur'] for f in pilot)/60:.1f} min, "
          f"{sum(f['n_seg'] for f in pilot)} segments")

    if a.dry_run:
        tot = 0
        for cid in ids:
            src = read_json(asr.asr_path(cfg, SOURCE, cid))
            w = lr.build_windows(src["segments"], k=a.k, k_max=max(a.k, a.k_max))
            tot += len(w)
        print(f"\ndry run: {tot} windows, no API calls made")
        return 0

    refs = {cid: reference.build_reference(clips[cid]) for cid in ids}
    base = score(cfg, SOURCE, ids, refs)
    print(f"\n{'system':<34}{'n':>4}{'ratio':>8}{'WER':>9}{'cpWER':>9}{'WDER':>9}{'ins':>8}")
    print(line("baseline  saaras-v3@fusion", base))

    out: dict = {"root": root, "model": a.model, "prompt_version": lr.PROMPT_VERSION,
                 "pilot": pilot, "rows": {"baseline": base}}
    key = lr.resolve_gemini_key(cfg)
    for arm in [x.strip().upper() for x in a.arms.split(",") if x.strip()]:
        tag = f"{SOURCE}+llm{arm.lower()}"
        n_edit = n_rev = n_fb = 0
        def one(cid):
            return cid, lr.refine_clip(cfg, SOURCE, cid, arm, a.model, key,
                                       use_cache=not a.no_cache, workers=a.workers,
                                       k=a.k, k_max=max(a.k, a.k_max))
        with ThreadPoolExecutor(max_workers=a.clip_workers) as pool:
            for cid, (payload, edits) in pool.map(one, ids):
                write_json_atomic(asr.asr_path(cfg, tag, cid), payload)
                for e in edits:
                    append_jsonl(cfg.root / "logs" / "step5_edits.jsonl", e)
                n_edit += payload["n_segments_changed"]
                n_rev += payload["n_segments_reverted"]
                n_fb += payload["n_window_fallbacks"]
                print(f"    {cid:<28}changed {payload['n_segments_changed']:>3}  "
                      f"reverted {payload['n_segments_reverted']:>3}  "
                      f"fallbacks {payload['n_window_fallbacks']}", flush=True)
        rows = score(cfg, tag, ids, refs)
        out["rows"][f"arm_{arm}"] = rows
        out[f"counts_{arm}"] = {"changed": n_edit, "reverted": n_rev, "fallbacks": n_fb}
        print(line(f"arm {arm}     +llm{arm.lower()}", rows)
              + f"   changed {n_edit}, reverted {n_rev}, fallbacks {n_fb}")

    dest = cfg.root / "results" / "step5_llm_refine.json"
    write_json_atomic(dest, out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
