#!/usr/bin/env python3
"""Whisper configuration probe: five configurations, ten clips, one table.

Answers "which Whisper configuration, if any, is worth a corpus sweep" without
paying for one. Writes NO checkpoint under asr/ -- it calls transcribe_whisper
and transcribe_segments directly, which are side-effect free apart from the
temporary segment buffers they cut into work_dir. Nothing here can collide with
a sweep running in another notebook.

The configurations, and the question each exists to answer:

  A  greedy, self-LID, long-form     the config the 99-clip checkpoints were
                                     produced in. Scored FROM DISK on the same
                                     ten clips, so the comparison is paired and
                                     costs no GPU time.
  B  beam 5, large-v3 LID            does the corrected configuration hold up
                                     beyond the three-clip probe?
  C  B + condition_on_previous_text  is repetition collapse the deletion
     = False                         mechanism? Turning off the carried prompt
                                     is the direct test.
  D  beam 5, LID, PER-SEGMENT        does forcing output per diarized turn beat
                                     long-form on more than three clips?
  E  large-v3 (not turbo), beam 5    is turbo inherently worse, or only worse at
                                     language ID? Decides which model a sweep
                                     would use.

Sampling. Nine clips, one per script, shortest in each -- plus ONE long clip
known to collapse. That inclusion is deliberate: repetition collapse is observed
on long audio, so a sample of short clips would miss the failure mode this probe
exists to measure and would flatter every configuration equally. Selecting by
script is a reporting choice made from the reference; it never reaches a model.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import asr, data, diarization, reference, text_metrics as tm  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402

# 598 s, Devanagari: transcribes 47 s correctly then repeats one five-word
# phrase for the remaining 550 s. The canonical collapse case.
COLLAPSE_CLIP = "0AEEA8NyVwY__11_609"
NGRAM = 5


def max_ngram_coverage(tokens: list[str], n: int = NGRAM) -> float:
    """Fraction of output covered by the single most frequent n-gram.

    Turns "repetition collapse" from an anecdote into a number. Healthy speech
    sits near zero; a decoder stuck in a loop approaches 1.0.
    """
    if len(tokens) < n:
        return 0.0
    pos: dict[tuple, list[int]] = collections.defaultdict(list)
    for i in range(len(tokens) - n + 1):
        pos[tuple(tokens[i:i + n])].append(i)
    best = max(pos.values(), key=len)
    # Positions, not occurrences: consecutive repeats of one token produce
    # overlapping n-grams, and counting occurrences x n then exceeds the length
    # of the output. Coverage is by definition bounded by 1.0.
    covered = {j for i in best for j in range(i, i + n)}
    return len(covered) / len(tokens)


def select_clips(cfg: Config, n: int = 10) -> list[str]:
    """One clip per script (shortest), plus the known collapse clip."""
    clips = [c for c in data.parse_ground_truth(data.load_segments_csv(cfg))
             if cfg.wav_path(c.clip_id).exists()]
    by: dict[str, list] = collections.defaultdict(list)
    for c in clips:
        by[c.stats.get("lang_script", "?")].append((c.end_sec - c.start_sec, c.clip_id))
    picked = [min(v)[1] for _, v in sorted(by.items())]
    if COLLAPSE_CLIP in {c.clip_id for c in clips} and COLLAPSE_CLIP not in picked:
        picked.append(COLLAPSE_CLIP)
    rest = sorted((d, i) for v in by.values() for d, i in v if i not in picked)
    for _, i in rest:
        if len(picked) >= n:
            break
        picked.append(i)
    return picked[:max(n, len(picked))]


def _norm(t: str) -> list[str]:
    return reference.normalize_text(t, strip_gloss=False).split()


def _row(ref, pairs, raw_text: str, lang, gt_script: str, elapsed: float) -> dict:
    r = tm.score_transcript(
        {k: v.split() for k, v in reference.speaker_texts(ref).items()},
        asr.speaker_texts_from_words(pairs),
        reference.word_stream(ref), pairs)
    hyp_script, _hint, _latin = data.dominant_script(raw_text)
    r.update(n_hyp=len(pairs), lang=lang, hyp_script=hyp_script,
             script_ok=bool(hyp_script == gt_script),
             rep5=max_ngram_coverage([w for w, _ in pairs]),
             elapsed_sec=round(elapsed, 1))
    return r


def probe(cfg: Config, clip_ids: list[str], diar: str = "fusion",
          model: str = "large-v3-turbo", only: set[str] | None = None) -> dict:
    clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(cfg))}
    refs = {i: reference.build_reference(clips[i]) for i in clip_ids}
    gt_script = {i: clips[i].stats.get("lang_script", "?") for i in clip_ids}
    out: dict = {"model": model, "diar": diar, "clips": clip_ids, "runs": {}}

    def emit(label: str, rows: list[dict]) -> None:
        out["runs"][label] = rows
        g = tm.summarise(rows)
        nref = sum(r["n_ref_words"] for r in rows) or 1
        nhyp = sum(r["n_hyp"] for r in rows)
        n = sum(r["hits"] + r["sub"] + r["del"] for r in rows) or 1
        dele = sum(r["del"] for r in rows)
        print(f"{label:<40}{len(rows):>4}{nhyp/nref:>7.2f}{g['wer']:>8.4f}"
              f"{g['cpwer']:>8.4f}{g['wder']:>8.4f}{dele/n:>7.1%}"
              f"{sum(r['script_ok'] for r in rows):>4}/{len(rows):<3}"
              f"{sum(r['rep5'] for r in rows)/len(rows):>8.1%}"
              f"{sum(r['elapsed_sec'] for r in rows):>7.0f}", flush=True)

    print(f"{'configuration':<40}{'n':>4}{'ratio':>7}{'WER':>8}{'cpWER':>8}"
          f"{'WDER':>8}{'del%':>7}{'script':>8}{'rep5':>8}{'sec':>7}", flush=True)

    # --- A: the existing checkpoints, scored on these same clips. No GPU.
    if not only or "A" in only:
        rows = []
        for cid in clip_ids:
            sysname = f"whisper-{model}"
            if not asr.is_done(cfg, sysname, cid):
                continue
            words = asr.load_words(cfg, sysname, cid)
            turns = diarization.load_hypothesis(cfg, diar, cid)
            pairs = [(t, s) for w, s in asr.assign_words(words, turns) for t in _norm(w)]
            meta = json.loads(asr.asr_path(cfg, sysname, cid).read_text())
            rows.append({"clip_id": cid,
                         **_row(refs[cid], pairs, " ".join(w.text for w in words),
                                meta.get("detected_language"), gt_script[cid], 0.0)})
        if rows:
            emit("A old: greedy, self-LID, long-form", rows)
        else:
            print("A skipped: no existing whisper checkpoints on these clips", flush=True)

    LONGFORM = [
        ("B", "B beam5 + large-v3 LID, long-form",
         dict(model_size=model, beam_size=5, lid_model=asr.LID_MODEL)),
        ("C", "C   + condition_on_previous_text=False",
         dict(model_size=model, beam_size=5, lid_model=asr.LID_MODEL,
              condition_on_previous_text=False)),
        ("E", "E large-v3, beam5, own LID, long-form",
         dict(model_size="large-v3", beam_size=5, lid_model=None)),
    ]
    for tag, label, kw in LONGFORM:
        if only and tag not in only:
            continue
        rows = []
        for cid in clip_ids:
            t0 = time.time()
            words, meta = asr.transcribe_whisper(cfg, cfg.wav_path(cid),
                                                 word_timestamps=True, **kw)
            turns = diarization.load_hypothesis(cfg, diar, cid)
            pairs = [(t, s) for w, s in asr.assign_words(words, turns) for t in _norm(w)]
            rows.append({"clip_id": cid,
                         **_row(refs[cid], pairs, " ".join(w.text for w in words),
                                meta.get("language"), gt_script[cid], time.time() - t0)})
        emit(label, rows)

    # --- D: per-segment. Speaker attribution is exact by construction.
    if not only or "D" in only:
        rows = []
        for cid in clip_ids:
            t0 = time.time()
            turns = asr.merge_same_speaker(diarization.load_hypothesis(cfg, diar, cid), 1.0)
            segs, meta = asr.transcribe_segments(cfg, f"whisper-{model}",
                                                 cfg.wav_path(cid), turns)
            ordered = sorted(segs, key=lambda s: s["start"])
            pairs = [(t, s["speaker"]) for s in ordered for t in _norm(s["text"])]
            rows.append({"clip_id": cid,
                         **_row(refs[cid], pairs, " ".join(s["text"] for s in ordered),
                                meta.get("clip_language"), gt_script[cid], time.time() - t0)})
        emit(f"D beam5 + LID, per-segment on {diar}", rows)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--diar", default="fusion")
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--only", default=None, help="comma-separated subset, e.g. B,D")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = a.root or next((p for p in ("/kaggle/working/sarvam_diarization",
                                       "/content/sarvam_diarization", "local_out")
                           if Path(p).exists()), "local_out")
    cfg = Config.create(root=Path(root), work_dir=Path(root) / "tmp")
    ids = select_clips(cfg, a.clips)
    if not ids:
        raise SystemExit(f"no clips with audio under {cfg.audio_dir}")
    missing = [c for c in ids if not diarization.is_done(cfg, a.diar, c)]
    if missing:
        raise SystemExit(f"{a.diar} turns missing for {missing}")
    print(f"root={root}  model={a.model}  diar={a.diar}  clips={len(ids)}\n")
    out = probe(cfg, ids, diar=a.diar, model=a.model,
                only=set(a.only.split(",")) if a.only else None)
    dest = Path(a.out) if a.out else Path(cfg.root) / "results" / "whisper_probe10.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
