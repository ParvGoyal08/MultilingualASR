"""Step 4 -- combine the benchmarked diarizations into a better one.

Why fusion rather than tuning one system. On the aligned reference the four
models land within 0.044 DER of each other but disagree about WHERE they are
wrong: diarizen-large has the lowest confusion (0.0590) and recovers 84% of
overlapped speech, reverb-v2 has the lowest miss (0.0717) but the highest false
alarm, and the pyannote pair sit between. Recomposing the best component from
each gives 0.1708 against the best single system's 0.1919 -- an 11% headroom
that exists only because the errors are not the same errors.

The method is DOVER-Lap (Raj, Garcia-Perera, Huang, Watanabe, Povey, Stolcke,
Khudanpur, "DOVER-Lap: A Method for Combining Overlap-aware Diarization
Outputs", SLT 2021), which extends ROVER's voting idea to diarization. Two
problems have to be solved in order:

1. **Label mapping.** "Speaker 2" in one system is unrelated to "Speaker 2" in
   another, so the outputs cannot be voted on until the labels are put in a
   common space. Systems are mapped one at a time onto a growing centroid by
   maximising total overlap -- the same assignment problem cpWER solves, and
   exact under Hungarian for the same reason.

2. **Voting.** Once labels agree, each frame is a weighted vote per speaker.
   Unlike ROVER this must stay overlap-aware: a frame may legitimately carry
   two speakers, so speakers are thresholded independently rather than being
   forced to compete for one slot.

Nothing here sees the ground truth. The only fitted quantity is the vote
threshold, chosen on the dev split and applied unchanged to test.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np

from .config import Config
from .data import Turn
from .utils import LOG

HOP = 0.01


# ------------------------------------------------------------------ rasters


def rasterise(turns: Sequence[Turn], n_frames: int,
              labels: Sequence[str]) -> np.ndarray:
    """(n_labels, n_frames) activity matrix."""
    idx = {l: i for i, l in enumerate(labels)}
    a = np.zeros((len(labels), n_frames), dtype=np.float32)
    for t in turns:
        i = idx.get(t.speaker)
        if i is None:
            continue
        lo, hi = int(max(0.0, t.start) / HOP), int(min(n_frames * HOP, t.end) / HOP)
        if hi > lo:
            a[i, lo:hi] = 1.0
    return a


def to_turns(active: np.ndarray, labels: Sequence[str],
             min_dur: float = 0.20) -> list[Turn]:
    """Contiguous runs of an activity matrix back into turns.

    `min_dur` drops slivers the voting can leave at boundaries where systems
    disagree by a frame or two. They are not real turns and they inflate the
    turn count without moving any metric.
    """
    out: list[Turn] = []
    for i, lab in enumerate(labels):
        row = active[i]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if (e - s) * HOP >= min_dur:
                out.append(Turn(start=s * HOP, end=e * HOP, speaker=lab))
    out.sort(key=lambda t: (t.start, t.speaker))
    return out


# ------------------------------------------------------------ label mapping


def _overlap_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Frames where row i of `a` and row j of `b` are both active."""
    return a.astype(np.float32) @ b.T.astype(np.float32)


def map_labels(centroid: np.ndarray, cent_labels: list[str],
               other: np.ndarray, other_labels: Sequence[str]) -> dict[str, str]:
    """Rename `other`'s speakers onto the centroid's, maximising overlap.

    Hungarian on the overlap matrix. A speaker with no useful counterpart is
    given a fresh label rather than forced onto the best available one: two
    systems can legitimately find a different NUMBER of speakers, and coercing
    them to match would invent agreement that is not there.
    """
    from scipy.optimize import linear_sum_assignment

    if not len(cent_labels):
        return {l: l for l in other_labels}
    ov = _overlap_matrix(other, centroid)
    rows, cols = linear_sum_assignment(-ov)
    mapping: dict[str, str] = {}
    used = set()
    for r, c in zip(rows, cols):
        if ov[r, c] > 0:
            mapping[other_labels[r]] = cent_labels[c]
            used.add(other_labels[r])
    spare = itertools.count(len(cent_labels))
    for l in other_labels:
        if l not in used:
            mapping[l] = f"spk{next(spare)}"
    return mapping


# ---------------------------------------------------------------- the fusion


def dover_lap(systems: dict[str, Sequence[Turn]], duration: float,
              weights: dict[str, float] | None = None,
              threshold: float = 0.5, min_dur: float = 0.20,
              rank_weighted: bool = False,
              overlap_threshold: float | None = None) -> list[Turn]:
    """Combine several diarizations of one clip into one.

    `threshold` is the share of total weight a speaker needs in a frame to be
    emitted. 0.5 is majority; lower recovers more speech at the cost of false
    alarm, which is exactly the miss/FA trade the components table describes.

    `rank_weighted` defaults OFF, having been measured to hurt. DOVER-Lap's
    1/rank weighting gives the top system weight 1 and the second 0.5, so with
    two systems the leader alone clears any threshold below 0.67 and the fusion
    silently degenerates to "use the best system" -- which is exactly what it
    did here, returning reverb-v2's score to four decimal places. Equal weights
    beat it on dev (0.2105 against 0.2223 on three systems).
    """
    names = [n for n in systems if systems[n]]
    if not names:
        return []
    if len(names) == 1:
        return list(systems[names[0]])

    n_frames = max(1, int(duration / HOP))
    w = dict(weights or {})
    if rank_weighted and not w:
        # DOVER-Lap's rank weighting: systems ordered by how much the others
        # agree with them, weighted 1/rank. A system that agrees with nobody
        # should not get the same say as one that agrees with everybody.
        agree = {}
        for n in names:
            labs = sorted({t.speaker for t in systems[n]})
            a = rasterise(systems[n], n_frames, labs)
            tot = 0.0
            for m in names:
                if m == n:
                    continue
                mlabs = sorted({t.speaker for t in systems[m]})
                b = rasterise(systems[m], n_frames, mlabs)
                tot += float(np.minimum(a.sum(0), b.sum(0)).sum())
            agree[n] = tot
        order = sorted(names, key=lambda n: -agree[n])
        w = {n: 1.0 / (i + 1) for i, n in enumerate(order)}
    for n in names:
        w.setdefault(n, 1.0)

    # 1. map every system onto a growing centroid, strongest-weighted first
    order = sorted(names, key=lambda n: -w[n])
    cent_labels: list[str] = sorted({t.speaker for t in systems[order[0]]})
    centroid = rasterise(systems[order[0]], n_frames, cent_labels) * w[order[0]]
    mapped: dict[str, dict[str, str]] = {order[0]: {l: l for l in cent_labels}}

    for name in order[1:]:
        labs = sorted({t.speaker for t in systems[name]})
        a = rasterise(systems[name], n_frames, labs)
        m = map_labels(centroid, cent_labels, a, labs)
        mapped[name] = m
        for new in sorted(set(m.values()) - set(cent_labels)):
            cent_labels.append(new)
            centroid = np.vstack([centroid, np.zeros((1, n_frames), dtype=np.float32)])
        idx = {l: i for i, l in enumerate(cent_labels)}
        for j, l in enumerate(labs):
            centroid[idx[m[l]]] += a[j] * w[name]

    # 2. vote, per speaker, independently -- overlap must survive
    total_w = sum(w[n] for n in names)
    votes = np.zeros((len(cent_labels), n_frames), dtype=np.float32)
    idx = {l: i for i, l in enumerate(cent_labels)}
    for name in names:
        labs = sorted({t.speaker for t in systems[name]})
        a = rasterise(systems[name], n_frames, labs)
        for j, l in enumerate(labs):
            votes[idx[mapped[name][l]]] += a[j] * w[name]

    active = votes >= threshold * total_w - 1e-9

    # A plain majority suppresses overlap. A second concurrent speaker needs the
    # same majority as the first, but the systems disagree most about exactly
    # those frames -- reverb-v2 predicts 15% of the reference's overlapped
    # speech against diarizen-large's 84%, so it votes against nearly every
    # second speaker. Measured, the majority fusion recovers 50% of overlap
    # where its best member recovers 84%.
    #
    # `overlap_threshold` lowers the bar for ADDITIONAL speakers only, in frames
    # where a first speaker already cleared the full majority. The leading
    # speaker still needs a majority, so this cannot invent speech in silence --
    # it only decides whether someone else is talking at the same time.
    if overlap_threshold is not None and overlap_threshold < threshold:
        lead = active.any(axis=0)
        extra = (votes >= overlap_threshold * total_w - 1e-9) & lead
        active = active | extra

    return to_turns(active, cent_labels, min_dur)


def fuse_corpus(cfg: Config, references: dict, models: Sequence[str],
                clip_ids: Iterable[str], threshold: float = 0.5,
                min_dur: float = 0.20) -> dict[str, list[Turn]]:
    """Run the fusion over a set of clips. Returns clip_id -> fused turns."""
    from . import diarization

    out: dict[str, list[Turn]] = {}
    for cid in clip_ids:
        ref = references.get(cid)
        if ref is None:
            continue
        systems = {m: diarization.load_hypothesis(cfg, m, cid)
                   for m in models if diarization.is_done(cfg, m, cid)}
        if not systems:
            continue
        out[cid] = dover_lap(systems, ref.uem[1], threshold=threshold, min_dur=min_dur)
    LOG.info("fused %d clips from %d systems at threshold %.2f",
             len(out), len(models), threshold)
    return out


# ------------------------------------------------- segmentation transplant


def transplant(seg_turns: Sequence[Turn], lab_turns: Sequence[Turn],
               duration: float, min_dur: float = 0.20) -> list[Turn]:
    """Take WHERE speech is from one system and WHO is speaking from another.

    The components table says one model owns detection and a different one owns
    assignment, which invites keeping each model's strength. A true hybrid would
    re-cluster inside the first system's boundaries using the second's speaker
    embeddings; those are not recoverable from an RTTM, so this transplants the
    LABELS instead: every frame the segmenter calls speech is given the speaker
    the labeller assigns there.

    That approximation is the honest limitation of this experiment. It inherits
    the labeller's clustering decisions exactly as the labeller made them, on
    the labeller's own boundaries -- so if the labeller's low confusion depended
    on its own tight segmentation, the transplant will not reproduce it. The
    result is therefore a test of the hypothesis, not an upper bound on it.

    Frames the segmenter calls speech but the labeller calls silence keep the
    segmenter's own label: dropping them would hand back the miss advantage the
    transplant exists to keep.
    """
    n = max(1, int(duration / HOP))
    seg_labels = sorted({t.speaker for t in seg_turns})
    lab_labels = sorted({t.speaker for t in lab_turns})
    if not seg_labels:
        return []
    if not lab_labels:
        return list(seg_turns)

    seg = rasterise(seg_turns, n, seg_labels)
    lab = rasterise(lab_turns, n, lab_labels)
    speech = seg.max(axis=0) > 0                      # where the segmenter hears speech

    # Put the segmenter's labels into the labeller's space, so a fallback frame
    # is not a speaker nobody else has heard of.
    m = map_labels(lab, lab_labels, seg, seg_labels)
    out_labels = list(lab_labels)
    for v in m.values():
        if v not in out_labels:
            out_labels.append(v)
    idx = {l: i for i, l in enumerate(out_labels)}

    active = np.zeros((len(out_labels), n), dtype=bool)
    for j, l in enumerate(lab_labels):
        active[idx[l]] = (lab[j] > 0) & speech        # labeller's view, gated by segmenter

    covered = active.any(axis=0)
    gap = speech & ~covered                           # segmenter says speech, labeller silent
    if gap.any():
        for j, l in enumerate(seg_labels):
            active[idx[m[l]]] |= (seg[j] > 0) & gap

    return to_turns(active, out_labels, min_dur)


# ------------------------------------------------------------- materialise


def materialise(cfg: Config, references: dict, models: Sequence[str],
                clip_ids: Iterable[str] | None = None, name: str = "fusion",
                threshold: float = 0.5, min_dur: float = 0.20,
                force: bool = False):
    """Write the fusion to disk as an ordinary diarization model.

    Everything downstream -- scoring, the error explorer, and above all the
    per-segment ASR -- loads hypotheses by model name from
    `hypotheses/<model>/<clip>.rttm`. A fusion computed on the fly is invisible
    to all of it. Writing the same RTTM + sidecar pair a real run writes makes
    `name` usable anywhere a model name is accepted, including as the ASR
    segmentation source.

    Deterministic given its inputs, so re-running is a no-op unless `force`.
    """
    import pandas as pd

    from . import diarization
    from .utils import atomic_publish, now_utc_iso, write_json_atomic

    rows = []
    for cid in (clip_ids if clip_ids is not None else references):
        ref = references.get(cid)
        if ref is None:
            continue
        if diarization.is_done(cfg, name, cid) and not force:
            rows.append({"model": name, "clip_id": cid, "status": "skipped"})
            continue
        systems = {m: diarization.load_hypothesis(cfg, m, cid)
                   for m in models if diarization.is_done(cfg, m, cid)}
        if not systems:
            continue
        turns = dover_lap(systems, ref.uem[1], weights={m: 1.0 for m in systems},
                          threshold=threshold, min_dur=min_dur)
        tmp = cfg.work_dir / f"fuse_{name}_{cid}.rttm"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(diarization.to_rttm(cid, turns), encoding="utf-8")
        atomic_publish(tmp, cfg.hyp_rttm_path(name, cid))
        tmp.unlink(missing_ok=True)
        record = {
            "model": name, "clip_id": cid, "status": "ok", "oracle": False,
            "n_turns": len(turns),
            "n_speakers_hyp": len({t.speaker for t in turns}),
            "clip_dur_sec": ref.uem[1],
            "max_end": round(max((t.end for t in turns), default=0.0), 3),
            "elapsed_sec": None, "rtf": None,
            "fused_from": list(systems), "threshold": threshold,
            "min_dur": min_dur, "weights": "equal",
            "diarized_at_utc": now_utc_iso(),
        }
        write_json_atomic(cfg.hyp_meta_path(name, cid), record)
        rows.append(record)
    LOG.info("materialised '%s' for %d clips from %s", name, len(rows), list(models))
    return pd.DataFrame(rows)
