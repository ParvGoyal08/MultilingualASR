#!/usr/bin/env python3
"""IndicConformer-600M on the same ten clips as the Whisper probe.

Reuses whisper_probe's clip selection, scoring and table format, so every row is
directly comparable to the Whisper rows rather than merely similar.

Language is supplied from the reference (`oracle_language`). IndicConformer has
no language identification of its own, so SOME source is mandatory; the oracle
bounds what the model can do when the language is right, and is an ablation
rather than a reportable result. A leak-free run with large-v3 LID is a
one-argument change (`--lang lid`) and is the number that could be published.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import asr, asr_indic, data, diarization, reference, text_metrics as tm  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402
from whisper_probe import _norm, _row, oracle_language, select_clips  # noqa: E402

# Whisper's ISO codes and IndicConformer's mostly agree; this covers the gaps.
LID_TO_INDIC = {"pa": "pa", "or": "or", "bn": "bn", "gu": "gu", "kn": "kn",
                "ml": "ml", "mr": "mr", "hi": "hi", "ta": "ta", "te": "te",
                "ur": "ur", "ne": "ne", "sa": "sa", "as": "as", "sd": "sd"}


def probe(cfg: Config, clip_ids: list[str], diar: str = "fusion",
          lang_source: str = "oracle", decodings=("rnnt", "ctc")) -> dict:
    clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(cfg))}
    refs = {i: reference.build_reference(clips[i]) for i in clip_ids}
    gt_script = {i: clips[i].stats.get("lang_script", "?") for i in clip_ids}

    langs: dict[str, str] = {}
    for cid in clip_ids:
        if lang_source == "oracle":
            langs[cid] = oracle_language(refs[cid], clips[cid].stats)
        else:
            code, p = asr.detect_language(cfg, cfg.wav_path(cid))
            langs[cid] = LID_TO_INDIC.get(code, "hi")
            print(f"  LID {cid}: {code} (p={p:.2f}) -> {langs[cid]}", flush=True)
    print(f"languages ({lang_source}): "
          + ", ".join(f"{c.split('__')[0][:8]}={l}" for c, l in langs.items()), flush=True)

    out: dict = {"model": asr_indic.MODEL_ID, "diar": diar,
                 "lang_source": lang_source, "clips": clip_ids, "langs": langs,
                 "runs": {}}
    print(f"\n{'configuration':<40}{'n':>4}{'ratio':>7}{'WER':>8}{'cpWER':>8}"
          f"{'WDER':>8}{'del%':>7}{'script':>8}{'rep5':>8}{'sec':>7}", flush=True)

    for dec in decodings:
        rows = []
        for cid in clip_ids:
            t0 = time.time()
            turns = asr.merge_same_speaker(
                diarization.load_hypothesis(cfg, diar, cid), 1.0)
            segs, meta = asr_indic.transcribe_segments(
                cfg, cfg.wav_path(cid), turns, langs[cid], decoding=dec)
            ordered = sorted(segs, key=lambda s: s["start"])
            pairs = [(t, s["speaker"]) for s in ordered for t in _norm(s["text"])]
            rows.append({"clip_id": cid,
                         **_row(refs[cid], pairs, " ".join(s["text"] for s in ordered),
                                langs[cid], gt_script[cid], time.time() - t0)})
        label = f"IndicConformer {dec} [{lang_source} lang], per-segment/{diar}"
        out["runs"][label] = rows
        g = tm.summarise(rows)
        nref = sum(r["n_ref_words"] for r in rows) or 1
        n = sum(r["hits"] + r["sub"] + r["del"] for r in rows) or 1
        print(f"{label:<40}{len(rows):>4}{sum(r['n_hyp'] for r in rows)/nref:>7.2f}"
              f"{g['wer']:>8.4f}{g['cpwer']:>8.4f}{g['wder']:>8.4f}"
              f"{sum(r['del'] for r in rows)/n:>7.1%}"
              f"{sum(r['script_ok'] for r in rows):>4}/{len(rows):<3}"
              f"{sum(r['rep5'] for r in rows)/len(rows):>8.1%}"
              f"{sum(r['elapsed_sec'] for r in rows):>7.0f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--diar", default="fusion")
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--lang", default="oracle", choices=("oracle", "lid"))
    ap.add_argument("--decoding", default="rnnt,ctc")
    ap.add_argument("--smoke", action="store_true", help="one clip, then stop")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = a.root or next((p for p in ("/kaggle/working/sarvam_diarization",
                                       "/content/sarvam_diarization", "local_out")
                           if Path(p).exists()), "local_out")
    cfg = Config.create(root=Path(root), work_dir=Path(root) / "tmp")
    ids = select_clips(cfg, a.clips)
    missing = [c for c in ids if not diarization.is_done(cfg, a.diar, c)]
    if missing:
        raise SystemExit(f"{a.diar} turns missing for {missing}")

    if a.smoke:
        clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(cfg))}
        cid = ids[0]
        lang = oracle_language(reference.build_reference(clips[cid]), clips[cid].stats)
        r = asr_indic.smoke(cfg, cid, lang)
        for k, v in r.items():
            print(f"  {k:<14}{v}")
        return 0

    out = probe(cfg, ids, diar=a.diar, lang_source=a.lang,
                decodings=tuple(a.decoding.split(",")))
    dest = Path(a.out) if a.out else Path(cfg.root) / "results" / "indic_probe10.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
