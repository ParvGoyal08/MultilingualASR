#!/usr/bin/env python3
"""A/B Whisper decoding settings on a few clips before committing to a sweep.

The stored 35-clip run lost 68.9% of reference words to deletions with
beam_size=1. Rather than assume beam search fixes it, transcribe the same clips
under several settings and count words against the reference -- word COUNT is
the diagnostic here, since the failure is omission rather than mis-recognition.

    python3 tools/whisper_ab.py --model large-v3-turbo --clips 3
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sarvam_diar import asr, data, reference, text_metrics as tm  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402

def default_root() -> str:
    """Where the pipeline lives on whichever platform this is.

    Config.create() knows about Colab but not Kaggle, and defaulting to
    "local_out" on Kaggle silently produces an empty run: the segments CSV is
    downloaded fresh into a new directory, no audio is beside it, and every
    measurement comes back zero.
    """
    for cand in ("/kaggle/working/sarvam_diarization",
                 "/content/drive/MyDrive/sarvam_diarization"):
        if Path(cand, "audio_16k").exists():
            return cand
    return "local_out"


SETTINGS = [
    ("greedy, conditioned   (what ran)", dict(beam_size=1, condition_on_previous_text=True)),
    ("greedy, unconditioned", dict(beam_size=1, condition_on_previous_text=False)),
    ("beam 5, unconditioned", dict(beam_size=5, condition_on_previous_text=False)),
    ("beam 5, conditioned", dict(beam_size=5, condition_on_previous_text=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--lid", default="large-v3", help="'' to let the model detect")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--root", default=None,
                    help="pipeline root; auto-detects Kaggle/Colab when omitted")
    args = ap.parse_args()

    root = args.root or default_root()
    cfg = Config.create(root=root, work_dir=f"{root}/.work")
    clips = data.parse_ground_truth(data.load_segments_csv(cfg))
    # short clips: the question is words-per-second-of-speech, and a short clip
    # answers it as well as a long one for a fraction of the wall clock
    picked = sorted((c for c in clips if cfg.wav_path(c.clip_id).exists()),
                    key=lambda c: c.end_sec - c.start_sec)[:args.clips]

    if not picked:
        # Without this the loop below runs zero times and prints a table of
        # zeros, which reads like a result rather than a missing input.
        raise SystemExit(
            f"no audio found under {cfg.audio_dir}\n"
            f"  root resolved to: {root}\n"
            f"  audio dir exists: {Path(cfg.audio_dir).exists()}\n"
            f"  pass --root explicitly, e.g. "
            f"--root /kaggle/working/sarvam_diarization")

    print(f"  model={args.model}  lid={args.lid or 'self'}  root={root}  "
          f"clips={len(picked)}\n")
    print(f"  {'setting':<36}{'ref w':>7}{'hyp w':>7}{'ratio':>7}{'WER':>8}{'del%':>7}{'sec':>7}")
    for label, kw in SETTINGS:
        ref_w = hyp_w = 0
        dels = subs = ins = hits = 0
        t0 = time.time()
        for c in picked:
            ref = reference.build_reference(c)
            words, _ = asr.transcribe_whisper(
                cfg, cfg.wav_path(c.clip_id), model_size=args.model,
                word_timestamps=False, lid_model=args.lid or None, **kw)
            rw = [t for u in ref.utterances for t in reference.tokenize(u.text_norm)]
            hw = [t for w in words
                  for t in reference.normalize_text(w.text, strip_gloss=False).split()]
            cnt = tm.wer_counts(rw, hw)
            ref_w += len(rw); hyp_w += len(hw)
            dels += cnt.deletions; subs += cnt.substitutions
            ins += cnt.insertions; hits += cnt.hits
        n = hits + subs + dels
        print(f"  {label:<36}{ref_w:>7}{hyp_w:>7}{hyp_w/max(ref_w,1):>7.2f}"
              f"{(subs+dels+ins)/max(n,1):>8.4f}{dels/max(n,1):>7.1%}{time.time()-t0:>7.0f}")
    print("\n  ratio near 1.0 means the recogniser is producing about as many words as")
    print("  were spoken. A low ratio with a high del% is truncation, not mis-hearing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
