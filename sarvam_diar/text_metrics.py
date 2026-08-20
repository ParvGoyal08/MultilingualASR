"""Speaker-attributed ASR metrics: WER, cpWER, DI-cpWER, WDER.

Every definition here is pinned to its source rather than reconstructed from
memory, because all four are easy to state almost-correctly:

* **WER** -- (S + D + I) / N over a single concatenated stream. Speaker-blind.
  A diagnostic only: it says nothing about attribution, so a system that
  transcribes perfectly and attributes everything to one speaker scores 0.

* **cpWER** -- concatenated minimum-permutation WER (Watanabe et al., CHiME-6,
  2020). Concatenate each speaker's words in time order on both sides, try every
  one-to-one assignment of hypothesis speakers to reference speakers, and take
  the assignment minimising total errors.

* **DI-cpWER** -- diarization-invariant cpWER: the same alignment with speaker
  attribution ignored. `cpWER - DI_cpWER` is the part of the error attributable
  to getting the speakers wrong rather than the words.

* **WDER** -- (S_IS + C_IS) / (S + C), from El Shafey, Soltau & Shafran,
  "Joint Speech Recognition and Speaker Diarization via Sequence Transduction",
  Interspeech 2019. Of the reference words that were *aligned* to a hypothesis
  word -- substitutions and correct words -- the fraction carrying the wrong
  speaker. Insertions and deletions are excluded from BOTH numerator and
  denominator, because a word present on only one side has no speaker pair to
  compare. Note the two traps: substitutions count (a misrecognised word still
  has an attribution, and it can still be wrong), and the denominator is S + C
  rather than C alone.

  One caveat inherent to the metric rather than to this implementation: when
  several minimal-cost alignments exist they can split the same edit distance
  differently across S/D/I. `a b c d e` against `a x c e f` costs 3 either as
  three substitutions, or as one substitution plus a deletion and an insertion.
  WER is 0.6 either way, but S + C differs, so WDER's denominator moves. Every
  WDER implementation inherits this; ours is whatever rapidfuzz's backtrace
  picks, applied identically to every system so the comparison stays fair.

On the exactness of the assignment. cpWER is often described as needing a search
over all permutations, with Hungarian offered as an approximation. It is not an
approximation: the error count for a (reference speaker, hypothesis speaker)
pair does not depend on which other pairs are chosen, so the total for a
permutation is just the sum of independent cell costs and minimising it IS the
linear assignment problem. `verify_assignment_exact()` checks this against brute
force, and the notebook runs it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    from rapidfuzz.distance import Levenshtein
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "text_metrics needs rapidfuzz for word alignment: pip install rapidfuzz"
    ) from exc


@dataclass(frozen=True)
class WerCounts:
    """Edit counts against a reference. `n` is the reference length."""

    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def n(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.n if self.n else (float("inf") if self.insertions else 0.0)

    def __add__(self, other: "WerCounts") -> "WerCounts":
        return WerCounts(self.hits + other.hits,
                         self.substitutions + other.substitutions,
                         self.deletions + other.deletions,
                         self.insertions + other.insertions)

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}wer": self.wer, f"{prefix}hits": self.hits,
                f"{prefix}sub": self.substitutions, f"{prefix}del": self.deletions,
                f"{prefix}ins": self.insertions, f"{prefix}n_ref_words": self.n,
                f"{prefix}errors": self.errors}


# --------------------------------------------------------------------- WER


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int, int]]:
    """Word-level alignment as (op, ref_index, hyp_index) triples.

    `op` is one of equal / replace / delete / insert. Index is -1 on the side
    that has no word. Expanded from rapidfuzz's block opcodes so that callers
    (WDER) can look at individual word pairs.
    """
    out: list[tuple[str, int, int]] = []
    for tag, i0, i1, j0, j1 in Levenshtein.opcodes(list(ref), list(hyp)).as_list():
        if tag == "equal":
            out.extend(("equal", i, j) for i, j in zip(range(i0, i1), range(j0, j1)))
        elif tag == "replace":
            # rapidfuzz emits replace blocks of equal length on both sides.
            out.extend(("replace", i, j) for i, j in zip(range(i0, i1), range(j0, j1)))
        elif tag == "delete":
            out.extend(("delete", i, -1) for i in range(i0, i1))
        elif tag == "insert":
            out.extend(("insert", -1, j) for j in range(j0, j1))
    return out


def wer_counts(ref: Sequence[str], hyp: Sequence[str]) -> WerCounts:
    """Edit counts for one reference/hypothesis word sequence pair."""
    h = s = d = i = 0
    for tag, i0, i1, j0, j1 in Levenshtein.opcodes(list(ref), list(hyp)).as_list():
        if tag == "equal":
            h += i1 - i0
        elif tag == "replace":
            s += max(i1 - i0, j1 - j0)
        elif tag == "delete":
            d += i1 - i0
        elif tag == "insert":
            i += j1 - j0
    return WerCounts(h, s, d, i)


def wer(ref_words: Sequence[str], hyp_words: Sequence[str]) -> float:
    return wer_counts(ref_words, hyp_words).wer


# ------------------------------------------------------------------ cpWER


def _pair_costs(ref_texts: dict[str, Sequence[str]],
                hyp_texts: dict[str, Sequence[str]]) -> tuple[list, list, list[list[WerCounts]]]:
    """Error counts for every (reference speaker, hypothesis speaker) pair.

    Padded to a square matrix with empty sequences, so an unmatched reference
    speaker becomes all deletions and an unmatched hypothesis speaker all
    insertions -- which is what cpWER charges for a missed or invented speaker.
    """
    r_keys = sorted(ref_texts)
    h_keys = sorted(hyp_texts)
    size = max(len(r_keys), len(h_keys), 1)
    r_pad = r_keys + [None] * (size - len(r_keys))
    h_pad = h_keys + [None] * (size - len(h_keys))
    cost = [[wer_counts(list(ref_texts.get(r, ())), list(hyp_texts.get(h, ())))
             for h in h_pad] for r in r_pad]
    return r_pad, h_pad, cost


def _hungarian(cost: list[list[int]]) -> list[int]:
    from scipy.optimize import linear_sum_assignment

    import numpy as np

    _, col = linear_sum_assignment(np.array(cost, dtype=float))
    return list(col)


def cpwer(ref_texts: dict[str, Sequence[str]],
          hyp_texts: dict[str, Sequence[str]]) -> dict:
    """Concatenated minimum-permutation WER, plus the assignment it chose."""
    r_pad, h_pad, cost = _pair_costs(ref_texts, hyp_texts)
    errs = [[c.errors for c in row] for row in cost]
    col = _hungarian(errs)

    total = WerCounts()
    mapping: dict[str, str] = {}
    for i, j in enumerate(col):
        total = total + cost[i][j]
        if r_pad[i] is not None and h_pad[j] is not None:
            mapping[h_pad[j]] = r_pad[i]
    out = total.as_dict("cp")
    out["cp_mapping"] = mapping
    return out


def di_cpwer(ref_words: Sequence[str], hyp_words: Sequence[str]) -> dict:
    """Diarization-invariant cpWER: the words alone, attribution discarded.

    Takes FLAT sequences -- every word on each side in a single stream -- so no
    speaker grouping survives and the result cannot be affected by attribution.
    `cpwer - di_cpwer` is then the part of the word error that exists only
    because speakers were assigned wrongly.

    An earlier version concatenated per speaker and joined the groups, which is
    wrong: grouping by speaker is exactly what makes a metric diarization
    DEPENDENT, so the "invariant" figure still moved when attribution changed
    and the subtraction measured nothing. Caught because a perfect ASR plus a
    perfect diarization scored 0.08 instead of 0.

    Caveat under overlap: the reference stream is in utterance order while a
    recogniser emits in time order, and where two people talk at once those two
    orders genuinely differ. That costs a little even for a perfect system, and
    is one more reason the headline number is cpWER rather than this.
    """
    return wer_counts(list(ref_words), list(hyp_words)).as_dict("di_cp")


def verify_assignment_exact(ref_texts: dict[str, Sequence[str]],
                            hyp_texts: dict[str, Sequence[str]],
                            max_speakers: int = 8) -> dict:
    """Check Hungarian against brute force over every permutation.

    Cheap because the pairwise costs are computed once and each permutation is
    then just a sum of cells -- 8! is 40,320 additions of eight integers, not
    40,320 alignments.
    """
    r_pad, h_pad, cost = _pair_costs(ref_texts, hyp_texts)
    size = len(r_pad)
    if size > max_speakers:
        return {"checked": False, "reason": f"{size} speakers exceeds {max_speakers}"}
    errs = [[c.errors for c in row] for row in cost]
    brute = min(sum(errs[i][p[i]] for i in range(size))
                for p in itertools.permutations(range(size)))
    hung = sum(errs[i][j] for i, j in enumerate(_hungarian(errs)))
    return {"checked": True, "hungarian": hung, "brute_force": brute,
            "agree": hung == brute, "n_permutations": _factorial(size)}


def _factorial(n: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


# ------------------------------------------------------------------- WDER


def wder(ref_stream: Sequence[tuple[str, str]],
         hyp_stream: Sequence[tuple[str, str]],
         mapping: dict[str, str] | None = None) -> dict:
    """WDER = (S_IS + C_IS) / (S + C), El Shafey et al., Interspeech 2019.

    `ref_stream` and `hyp_stream` are flat (word, speaker) sequences in time
    order. `mapping` renames hypothesis speakers to reference speakers; pass the
    one cpWER chose so both metrics describe the same alignment of speakers.
    Without it, hypothesis labels are compared to reference labels literally,
    which is almost never what is wanted.

    Insertions and deletions are skipped: a word on only one side has no pair of
    speaker labels to disagree about.
    """
    mapping = mapping or {}
    ref_w = [w for w, _ in ref_stream]
    hyp_w = [w for w, _ in hyp_stream]
    ref_s = [s for _, s in ref_stream]
    hyp_s = [mapping.get(s, s) for _, s in hyp_stream]

    correct = subs = correct_wrong_spk = subs_wrong_spk = 0
    for tag, i, j in align(ref_w, hyp_w):
        if tag == "equal":
            correct += 1
            if ref_s[i] != hyp_s[j]:
                correct_wrong_spk += 1
        elif tag == "replace":
            subs += 1
            if ref_s[i] != hyp_s[j]:
                subs_wrong_spk += 1
    denom = subs + correct
    return {
        "wder": (subs_wrong_spk + correct_wrong_spk) / denom if denom else float("nan"),
        "wder_num": subs_wrong_spk + correct_wrong_spk,
        "wder_denom": denom,
        "wder_correct_wrong_spk": correct_wrong_spk,
        "wder_sub_wrong_spk": subs_wrong_spk,
    }


def score_transcript(ref_texts: dict[str, Sequence[str]],
                     hyp_texts: dict[str, Sequence[str]],
                     ref_stream: Sequence[tuple[str, str]],
                     hyp_stream: Sequence[tuple[str, str]]) -> dict:
    """Every text metric for one clip, sharing one speaker assignment."""
    ref_words = [w for w, _ in ref_stream]
    hyp_words = [w for w, _ in hyp_stream]
    cp = cpwer(ref_texts, hyp_texts)
    di = di_cpwer(ref_words, hyp_words)
    wd = wder(ref_stream, hyp_stream, mapping=cp["cp_mapping"])
    # Plain WER and DI-cpWER are the same computation on this corpus -- both are
    # the speaker-agnostic word error -- so it is reported once under each name
    # rather than presented as two independent findings.
    flat = wer_counts(ref_words, hyp_words).as_dict("")
    row = {**flat, **cp, **di, **wd}
    # The headline decomposition: how much of the word error is attribution.
    row["cp_minus_di_cp"] = cp["cpwer"] - di["di_cpwer"]
    return row


def summarise(rows: Iterable[dict]) -> dict:
    """Pool across clips by summing counts, never by averaging rates.

    Same reasoning as evaluation.pool(): a 50 s clip and a 30 min clip must not
    carry equal weight in a corpus figure.
    """
    rows = list(rows)
    if not rows:
        return {}
    out: dict = {"n_clips": len(rows)}
    for prefix in ("", "cp", "di_cp"):
        n = sum(r.get(f"{prefix}n_ref_words", 0) for r in rows)
        e = sum(r.get(f"{prefix}errors", 0) for r in rows)
        out[f"{prefix}wer" if prefix else "wer"] = e / n if n else float("nan")
        out[f"{prefix}n_ref_words" if prefix else "n_ref_words"] = n
    num = sum(r.get("wder_num", 0) for r in rows)
    den = sum(r.get("wder_denom", 0) for r in rows)
    out["wder"] = num / den if den else float("nan")
    out["cp_minus_di_cp"] = out["cpwer"] - out["di_cpwer"]
    return out
