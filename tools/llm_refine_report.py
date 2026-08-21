#!/usr/bin/env python3
"""Report the Step 5 pilot: aggregate table, per-clip deltas, paired bootstrap,
and qualitative wins/losses/reverts drawn from the edit log by a stated rule.

The selection rule for the qualitative sections is fixed here so it cannot be
cherry-picking: wins are the largest per-segment error reductions, losses the
largest increases, and every revert_reason that actually fired gets one example.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import reference, text_metrics as tm  # noqa: E402
from sarvam_diar.utils import read_jsonl  # noqa: E402


def pooled(rows: list[dict]) -> dict:
    g = tm.summarise(rows)
    g["ratio"] = sum(r["n_hyp"] for r in rows) / max(g["n_ref_words"], 1)
    g["ins"] = sum(r["ins"] for r in rows)
    return g


def boot_ci(deltas: list[float], n: int = 10000, seed: int = 20260822):
    """Paired bootstrap over clips on the per-clip cpWER delta."""
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [rng.choice(deltas) for _ in deltas]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="local_out/step4_input")
    a = ap.parse_args()
    root = Path(a.root)
    res = json.loads((root / "results" / "step5_llm_refine.json").read_text())
    rows = res["rows"]

    print(f"model {res['model']}   prompt {res['prompt_version']}   "
          f"pilot {len(res['pilot'])} clips (GT-blind, dev only)\n")
    print(f"{'system':<30}{'n':>4}{'ratio':>8}{'WER':>9}{'cpWER':>9}{'WDER':>9}"
          f"{'ins':>7}{'ΔcpWER':>9}")
    base = pooled(rows["baseline"])
    print(f"{'baseline':<30}{base['n_clips']:>4}{base['ratio']:>8.4f}{base['wer']:>9.4f}"
          f"{base['cpwer']:>9.4f}{base['wder']:>9.4f}{base['ins']:>7,}{'':>9}")
    b_by = {r["clip_id"]: r for r in rows["baseline"]}

    for name in [k for k in rows if k.startswith("arm_")]:
        g = pooled(rows[name])
        d = g["cpwer"] - base["cpwer"]
        print(f"{name:<30}{g['n_clips']:>4}{g['ratio']:>8.4f}{g['wer']:>9.4f}"
              f"{g['cpwer']:>9.4f}{g['wder']:>9.4f}{g['ins']:>7,}{d:>+9.4f}")

    for name in [k for k in rows if k.startswith("arm_")]:
        arm = name.split("_")[1]
        deltas, wins, losses = [], 0, 0
        print(f"\nper-clip cpWER, arm {arm}")
        print(f"  {'clip':<28}{'base':>9}{'after':>9}{'Δ':>9}")
        for r in sorted(rows[name], key=lambda r: r["clip_id"]):
            b = b_by[r["clip_id"]]
            d = r["cpwer"] - b["cpwer"]
            deltas.append(d)
            wins += d < -1e-9
            losses += d > 1e-9
            flag = "" if abs(d) < 1e-9 else ("  better" if d < 0 else "  WORSE")
            print(f"  {r['clip_id']:<28}{b['cpwer']:>9.4f}{r['cpwer']:>9.4f}{d:>+9.4f}{flag}")
        lo, hi = boot_ci(deltas)
        mean = sum(deltas) / len(deltas)
        print(f"  mean Δ {mean:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"wins {wins}/{len(deltas)}, losses {losses}")
        print(f"  CI excludes zero: {'yes' if hi < 0 else 'NO'}")
        c = res.get(f"counts_{arm}", {})
        print(f"  segments changed {c.get('changed')}, reverted {c.get('reverted')}, "
              f"window fallbacks {c.get('fallbacks')}")

    # ---- qualitative, from the edit log
    log = root / "logs" / "step5_edits.jsonl"
    if not log.exists():
        return 0
    edits = read_jsonl(log)
    print(f"\nedit log: {len(edits)} records")
    for arm in sorted({e.get("arm") for e in edits if e.get("arm")}):
        sub = [e for e in edits if e.get("arm") == arm]
        reasons = Counter(e.get("reason") for e in sub if e.get("action") == "reverted")
        acts = Counter(e.get("action") for e in sub)
        print(f"  arm {arm}: {dict(acts)}")
        if reasons:
            print(f"    revert reasons: {dict(reasons)}")

    def show(e, tag):
        print(f"    [{tag}] {e['clip_id']} seg {e.get('seg_i')} {e.get('speaker','')}")
        print(f"      old: {(e.get('old_text') or '')[:150]}")
        print(f"      new: {(e.get('new_text') or '')[:150]}")
        if e.get("reason"):
            print(f"      reverted: {e['reason']}")

    applied = [e for e in edits if e.get("action") == "changed" and e.get("old_text")]
    print(f"\n  sample applied edits ({len(applied)} total):")
    for e in applied[:6]:
        show(e, "changed")
    seen = set()
    print("\n  one example per revert reason that fired:")
    for e in edits:
        r = e.get("reason")
        if e.get("action") == "reverted" and r and r not in seen and e.get("old_text"):
            seen.add(r)
            show(e, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
