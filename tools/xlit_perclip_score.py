#!/usr/bin/env python3
"""Score the per-clip script-correction experiment. GT is read HERE ONLY.

Inference finished before this script ran. Nothing computed here fed back into
the prompt, the vocabulary, the guards or the configuration.

Edit-level attribution: the correction is a whole-token substitution, so the
before/after hypothesis streams for a speaker have identical length and differ
only at changed positions. Aligning each against the reference and comparing the
alignment op at a changed position tells us whether that specific edit turned an
error into a match (helpful), a match into an error (harmful), or neither.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import asr, data, reference, text_metrics as tm  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402
from sarvam_diar.utils import read_json  # noqa: E402

SOURCE = "sarvam-saaras-v3@fusion"
EXP = Path("local_out/experiments/xlit_perclip")
SEED = 20260822


def token_script(tok: str) -> str:
    c = Counter()
    for ch in tok:
        if not ch.isalpha():
            continue
        try:
            c[unicodedata.name(ch).split()[0]] += 1
        except ValueError:
            pass
    return c.most_common(1)[0][0].title() if c else ""


def pairs_from(payload: dict) -> list[tuple[str, str]]:
    return [(t, s["speaker"])
            for s in sorted(payload["segments"], key=lambda s: s["start"])
            for t in reference.normalize_text(s["text"], strip_gloss=False).split()]


def score_one(ref, prs):
    return tm.score_transcript(
        {k: v.split() for k, v in reference.speaker_texts(ref).items()},
        asr.speaker_texts_from_words(prs), reference.word_stream(ref), prs)


def boot(deltas, n=10000, seed=SEED):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--root", default="checkpoints")
    a = ap.parse_args()

    cfg = Config.create(root=Path(a.root), work_dir=Path("/tmp/xs"))
    meta = Config.create(root=Path("local_out"), work_dir=Path("/tmp/xs"))
    clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(meta))}
    split = json.loads(Path("results/split.json").read_text())
    want = [s.strip() for s in a.splits.split(",")]
    ids = sorted({c for s in want for c in split[s]
                  if (EXP / "asr" / f"{SOURCE}+xlitpc" / f"{c}.json").exists()})
    refs = {c: reference.build_reference(clips[c]) for c in ids}

    before_rows, after_rows, per_clip = {}, {}, {}
    helpful = harmful = neutral = 0
    n_changed = n_tokens = 0
    corrupt = 0
    by_script = defaultdict(lambda: {"clips": 0, "b": [], "a": [], "edits": 0,
                                     "help": 0, "harm": 0})
    good, bad = [], []

    for c in ids:
        src = read_json(asr.asr_path(cfg, SOURCE, c))
        exp = read_json(EXP / "asr" / f"{SOURCE}+xlitpc" / f"{c}.json")
        pb, pa = pairs_from(src), pairs_from(exp)
        assert len(pb) == len(pa), f"{c}: token count changed {len(pb)} -> {len(pa)}"
        n_tokens += len(pb)
        before_rows[c] = score_one(refs[c], pb)
        after_rows[c] = score_one(refs[c], pa)
        sc = refs[c].stats.get("lang_script")
        by_script[sc]["clips"] += 1
        by_script[sc]["b"].append(before_rows[c])
        by_script[sc]["a"].append(after_rows[c])

        # ---- per-edit attribution, per speaker
        rt = {k: v.split() for k, v in reference.speaker_texts(refs[c]).items()}
        hb, ha = defaultdict(list), defaultdict(list)
        for (t, s) in pb:
            hb[s].append(t)
        for (t, s) in pa:
            ha[s].append(t)
        # speaker label mapping is what cpWER solves; use the agnostic stream for
        # attribution so a mapping change cannot be mistaken for an edit effect
        refflat = [w for w, _ in reference.word_stream(refs[c])] if refs[c] else []
        RB = [t for t, _ in pb]
        RA = [t for t, _ in pa]
        opb = {}
        for op, i, j in tm.align(refflat, RB):
            if j >= 0:
                opb[j] = op
        opa = {}
        for op, i, j in tm.align(refflat, RA):
            if j >= 0:
                opa[j] = op
        for j, (tb, ta) in enumerate(zip(RB, RA)):
            if tb == ta:
                continue
            n_changed += 1
            b_ok = opb.get(j) == "equal"
            a_ok = opa.get(j) == "equal"
            if a_ok and not b_ok:
                helpful += 1
                by_script[sc]["help"] += 1
                if len(good) < 400:
                    good.append((sc, tb, ta, c))
            elif b_ok and not a_ok:
                harmful += 1
                by_script[sc]["harm"] += 1
                if len(bad) < 400:
                    bad.append((sc, tb, ta, c))
            else:
                neutral += 1
            by_script[sc]["edits"] += 1
        # cross-script corruption: rendered script != clip's own dominant script
        dom = data.dominant_script(" ".join(
            (s.get("text") or "") for s in src["segments"]))[0]
        for tb, ta in zip(RB, RA):
            if tb != ta and token_script(ta) and token_script(ta) != dom:
                corrupt += 1

    print(f"\n{'='*74}\nSPLIT(S): {','.join(want)}   clips {len(ids)}\n{'='*74}")
    gb, ga = tm.summarise(list(before_rows.values())), tm.summarise(list(after_rows.values()))
    print(f"{'system':<34}{'WER':>9}{'cpWER':>9}{'WDER':>9}{'DI-cpWER':>10}")
    print(f"{'baseline  saaras-v3@fusion':<34}{gb['wer']:>9.4f}{gb['cpwer']:>9.4f}"
          f"{gb['wder']:>9.4f}{gb['di_cpwer']:>10.4f}")
    print(f"{'  + per-clip script correction':<34}{ga['wer']:>9.4f}{ga['cpwer']:>9.4f}"
          f"{ga['wder']:>9.4f}{ga['di_cpwer']:>10.4f}")
    print(f"{'  delta':<34}{ga['wer']-gb['wer']:>+9.4f}{ga['cpwer']-gb['cpwer']:>+9.4f}"
          f"{ga['wder']-gb['wder']:>+9.4f}{ga['di_cpwer']-gb['di_cpwer']:>+10.4f}")

    for s in want:
        sub = [c for c in ids if c in set(split[s])]
        if not sub:
            continue
        b = tm.summarise([before_rows[c] for c in sub])
        aa = tm.summarise([after_rows[c] for c in sub])
        d = [after_rows[c]["cpwer"] - before_rows[c]["cpwer"] for c in sub]
        lo, hi = boot(d)
        w = sum(x < -1e-9 for x in d); l = sum(x > 1e-9 for x in d)
        print(f"\n  {s}: n={len(sub)}  pooled cpWER {b['cpwer']:.4f} -> {aa['cpwer']:.4f} "
              f"({aa['cpwer']-b['cpwer']:+.4f})")
        print(f"       per-clip mean delta {sum(d)/len(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  {'excludes 0' if hi < 0 or lo > 0 else 'INCLUDES 0'}")
        print(f"       wins {w}  losses {l}  ties {len(d)-w-l}")

    print(f"\n  tokens changed {n_changed:,} of {n_tokens:,} = {n_changed/n_tokens:.2%}")
    print(f"  helpful {helpful}  harmful {harmful}  neutral {neutral}"
          f"   net {helpful-harmful:+d}  ratio {helpful/max(harmful,1):.1f}:1")
    print(f"  cross-script corruption (rendered script != clip script): {corrupt}")

    print(f"\n  {'script':<12}{'clips':>6}{'edits':>7}{'help':>6}{'harm':>6}"
          f"{'cpWER b':>9}{'cpWER a':>9}{'delta':>9}")
    for sc, d in sorted(by_script.items(), key=lambda x: -x[1]["edits"]):
        b, aa = tm.summarise(d["b"]), tm.summarise(d["a"])
        print(f"  {sc:<12}{d['clips']:>6}{d['edits']:>7}{d['help']:>6}{d['harm']:>6}"
              f"{b['cpwer']:>9.4f}{aa['cpwer']:>9.4f}{aa['cpwer']-b['cpwer']:>+9.4f}")

    print("\n  representative HELPFUL edits:")
    seen = set()
    for sc, tb, ta, c in good:
        if (sc, tb) in seen:
            continue
        seen.add((sc, tb))
        print(f"    {sc:<11} {tb:<18} -> {ta}")
        if len(seen) >= 12:
            break
    print("\n  representative HARMFUL edits:")
    seen = set()
    for sc, tb, ta, c in bad:
        if (sc, tb) in seen:
            continue
        seen.add((sc, tb))
        print(f"    {sc:<11} {tb:<18} -> {ta}")
        if len(seen) >= 12:
            break

    out = {"splits": want, "n_clips": len(ids),
           "pooled": {"before": {k: gb[k] for k in ("wer", "cpwer", "wder", "di_cpwer")},
                      "after": {k: ga[k] for k in ("wer", "cpwer", "wder", "di_cpwer")}},
           "tokens_changed": n_changed, "tokens_total": n_tokens,
           "helpful": helpful, "harmful": harmful, "neutral": neutral,
           "cross_script_corruption": corrupt,
           "per_clip": {c: {"before": before_rows[c]["cpwer"],
                            "after": after_rows[c]["cpwer"]} for c in ids}}
    (EXP / f"score_{'_'.join(want)}.json").write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {EXP}/score_{'_'.join(want)}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
