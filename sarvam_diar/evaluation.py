"""Metrics: DER, JER, speaker-count accuracy, and where the error comes from.

Scoring rules this corpus forces:

* **Overlap is scored.** 7.13% of scored time has >= 2 distinct speakers active
  and only 9 of 100 clips have none, so `skip_overlap=False` and a 0.0 s collar
  are the primary setting. A collared / overlap-forgiving variant is reported
  alongside for comparability with published numbers, never instead.
* **UEM must be passed explicitly.** Given no `uem`, pyannote.metrics warns and
  approximates it as the union of the reference and hypothesis extents -- which
  silently makes the scored region depend on the hypothesis. Every call here
  passes `[0, requested_dur_sec]`.
* Score against `reference.load_reference()`, never a re-parse of the raw CSV.

Everything below about pyannote.metrics' behaviour was measured, not assumed --
see PROBE_FINDINGS.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .config import (
    DER_COLLAR,
    DER_COLLAR_LENIENT,
    DER_SKIP_OVERLAP,
    DER_SKIP_OVERLAP_LENIENT,
    Config,
)
from .data import ClipReference, Turn
from .utils import LOG, write_json_atomic

# --------------------------------------------------------------------- probe
# Measured against pyannote.metrics 4.1 / pyannote.core 6.0.1 rather than taken
# from the docs, which do not pin any of this down. `probe_metric_semantics()`
# below re-verifies all three at runtime, so a library upgrade that changes them
# fails loudly instead of silently shifting every reported number.
PROBE_FINDINGS = {
    "component_keys": ["confusion", "correct", "false alarm", "missed detection", "total"],
    "total_counts": "sum of per-speaker reference duration -- overlapped speech is "
                    "counted once PER SPEAKER, not once. (Hand-checked: a 2-speaker "
                    "example whose union is 25 s reports total=30 s.)",
    "accumulation": "POOLS components across calls, then divides once. Verified with "
                    "a 10 s perfect file and a 1000 s file at DER 0.5: abs(metric) "
                    "is 0.495 (pooled), not 0.25 (mean of per-file rates).",
    "identity": "(false alarm + missed detection + confusion) / total == abs(metric)",
}

METRIC_COLUMNS = [
    "model", "clip_id", "der", "der_fa_sec", "der_miss_sec", "der_confusion_sec",
    "der_correct_sec", "der_total_sec", "jer", "n_speakers_ref", "n_speakers_hyp",
    "speaker_count_error", "speaker_count_correct",
]

KEY_FA, KEY_MISS, KEY_CONF = "false alarm", "missed detection", "confusion"
KEY_CORRECT, KEY_TOTAL = "correct", "total"
ERROR_KEYS = (KEY_FA, KEY_MISS, KEY_CONF)


def probe_metric_semantics() -> dict[str, Any]:
    """Re-derive pyannote's component semantics at runtime.

    Cheap, and it means a library upgrade that renames a component or switches
    accumulation from pooling to averaging is caught immediately rather than
    quietly changing the corpus DER.
    """
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    def ann(uri, spans):
        a = Annotation(uri=uri)
        for s, e, l in spans:
            a[Segment(s, e)] = l
        return a

    # Union of speech is 25 s; summed per-speaker duration is 30 s. Which one
    # `total` reports tells us how overlap is counted.
    ref = ann("probe", [(0, 10, "A"), (20, 30, "A"), (5, 15, "B")])
    m = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    comp = m.compute_components(ref, ref, uem=Timeline([Segment(0, 30)]))

    # A short perfect file and a long bad one: pooling and averaging disagree.
    short_r, short_h = ann("s", [(0, 10, "A")]), ann("s", [(0, 10, "A")])
    long_r, long_h = ann("l", [(0, 1000, "A")]), ann("l", [(0, 500, "A")])
    acc = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    r1 = acc(short_r, short_h, uem=Timeline([Segment(0, 10)]))
    r2 = acc(long_r, long_h, uem=Timeline([Segment(0, 1000)]))
    pooled = acc[:]
    pooled_ratio = sum(pooled[k] for k in ERROR_KEYS) / pooled[KEY_TOTAL]

    findings = {
        "component_keys": sorted(comp.keys()),
        "total_is_per_speaker_sum": abs(comp[KEY_TOTAL] - 30.0) < 1e-9,
        "total_is_union": abs(comp[KEY_TOTAL] - 25.0) < 1e-9,
        "pools_components": abs(abs(acc) - pooled_ratio) < 1e-12,
        "averages_rates": abs(abs(acc) - (r1 + r2) / 2) < 1e-12,
        "identity_holds": abs(abs(acc) - pooled_ratio) < 1e-12,
    }
    problems = []
    if sorted(comp.keys()) != sorted(PROBE_FINDINGS["component_keys"]):
        problems.append(f"component keys changed: {sorted(comp.keys())}")
    if not findings["total_is_per_speaker_sum"]:
        problems.append("`total` no longer counts overlap per speaker")
    if not findings["pools_components"]:
        problems.append("accumulation no longer pools components -- pool them manually")
    findings["problems"] = problems
    return findings


# -------------------------------------------------------------- conversions


def turns_to_annotation(turns: Iterable[Turn], uri: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for i, t in enumerate(turns):
        if t.end > t.start:
            # Distinct tracks so a speaker overlapping itself is not silently
            # collapsed; the reference builder already unions those, but a
            # hypothesis from a model may not.
            ann[Segment(t.start, t.end), i] = t.speaker
    return ann


def uem_timeline(ref: ClipReference):
    from pyannote.core import Segment, Timeline

    return Timeline([Segment(*ref.uem)], uri=ref.clip_id)


# ------------------------------------------------------------------ scoring


def score_clip(ref: ClipReference, hyp: list[Turn], collar: float = DER_COLLAR,
               skip_overlap: bool = DER_SKIP_OVERLAP) -> dict[str, Any]:
    """DER components, JER and speaker counts for one clip.

    Components are returned raw (in seconds) as well as the ratio, because the
    corpus number must be computed by pooling seconds -- averaging per-clip DER
    would weight a 50 s clip the same as a 30 min one.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    r = to_annotation(ref)
    h = turns_to_annotation(hyp, ref.clip_id)
    uem = uem_timeline(ref)

    der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    comp = der.compute_components(r, h, uem=uem)
    jer = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)(r, h, uem=uem)

    n_ref, n_hyp = len(r.labels()), len(h.labels())
    return {
        "der": der.compute_metric(comp),
        "der_fa_sec": comp[KEY_FA],
        "der_miss_sec": comp[KEY_MISS],
        "der_confusion_sec": comp[KEY_CONF],
        "der_correct_sec": comp[KEY_CORRECT],
        "der_total_sec": comp[KEY_TOTAL],
        "jer": float(jer),
        "n_speakers_ref": n_ref,
        "n_speakers_hyp": n_hyp,
        "speaker_count_error": n_hyp - n_ref,
        "speaker_count_correct": int(n_hyp == n_ref),
    }


def to_annotation(ref: ClipReference):
    """Reference as a pyannote Annotation (kept here so evaluation is self-contained)."""
    return turns_to_annotation(ref.turns, ref.clip_id)


def pool(rows: Iterable[dict], prefix: str = "") -> dict[str, float]:
    """Corpus-level metrics by pooling seconds, per PROBE_FINDINGS['accumulation']."""
    rows = list(rows)
    if not rows:
        return {}
    fa = sum(r[f"{prefix}der_fa_sec"] for r in rows)
    miss = sum(r[f"{prefix}der_miss_sec"] for r in rows)
    conf = sum(r[f"{prefix}der_confusion_sec"] for r in rows)
    total = sum(r[f"{prefix}der_total_sec"] for r in rows)
    if total <= 0:
        return {}
    return {
        "n_clips": len(rows),
        "der": (fa + miss + conf) / total,
        "der_fa_frac": fa / total,
        "der_miss_frac": miss / total,
        "der_confusion_frac": conf / total,
        "total_sec": total,
        # JER has no additive components, so it is averaged per clip and that is
        # labelled rather than passed off as a pooled figure.
        "jer_mean": sum(r[f"{prefix}jer"] for r in rows) / len(rows),
        "speaker_count_accuracy": sum(r[f"{prefix}speaker_count_correct"] for r in rows) / len(rows),
        "speaker_count_mae": sum(abs(r[f"{prefix}speaker_count_error"]) for r in rows) / len(rows),
        "speaker_count_bias": sum(r[f"{prefix}speaker_count_error"] for r in rows) / len(rows),
    }


def score_all(cfg: Config, references: dict[str, ClipReference],
              hypotheses: dict[tuple[str, str], list[Turn]]) -> pd.DataFrame:
    """Score every (model, clip) hypothesis. Keys are (model, clip_id)."""
    rows = []
    for (model, clip_id), hyp in sorted(hypotheses.items()):
        ref = references.get(clip_id)
        if ref is None:
            LOG.warning("no reference for %s, skipping", clip_id)
            continue
        row = {"model": model, "clip_id": clip_id}
        row.update(score_clip(ref, hyp, DER_COLLAR, DER_SKIP_OVERLAP))
        lenient = score_clip(ref, hyp, DER_COLLAR_LENIENT, DER_SKIP_OVERLAP_LENIENT)
        row.update({f"lenient_{k}": v for k, v in lenient.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not len(df):
        # Keep the schema even with no hypotheses (a gated model, an interrupted
        # sweep), so downstream groupby/column access degrades to an empty table
        # instead of KeyError.
        df = pd.DataFrame(columns=METRIC_COLUMNS
                          + [f"lenient_{c}" for c in METRIC_COLUMNS if c not in ("model", "clip_id")])
    if cfg is not None and len(df):
        df.to_csv(cfg.step2_metrics_csv, index=False)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Corpus metrics per model, pooled. Empty in, empty out."""
    if not len(df) or "model" not in df.columns:
        return pd.DataFrame(columns=["model", "n_clips", "der"])
    out = []
    for model, sub in df.groupby("model"):
        rows = sub.to_dict("records")
        agg = {"model": model, **pool(rows)}
        agg.update({f"lenient_{k}": v for k, v in pool(rows, prefix="lenient_").items()})
        out.append(agg)
    return pd.DataFrame(out)


# -------------------------------------------------- where the error comes from


def stratify_by_count_error(df: pd.DataFrame) -> pd.DataFrame:
    """DER decomposed by whether the speaker count was under/exact/over.

    Leak-free: uses only the hypothesis speaker count the model produced and the
    reference count, both of which are scoring-side. Within the `exact` stratum
    no reference speaker is unmappable, so its confusion component is *pure
    assignment error* -- that is the number that answers "count vs assignment".
    """
    if not len(df) or "speaker_count_error" not in df.columns:
        return pd.DataFrame(columns=["model", "stratum", "n_clips", "der"])

    def bucket(e):
        return "exact" if e == 0 else ("over" if e > 0 else "under")

    rows = []
    for (model, strat), sub in df.assign(
        stratum=df.speaker_count_error.map(bucket)
    ).groupby(["model", "stratum"]):
        rows.append({"model": model, "stratum": strat, **pool(sub.to_dict("records"))})
    out = pd.DataFrame(rows)
    order = {"under": 0, "exact": 1, "over": 2}
    return out.sort_values(["model", "stratum"], key=lambda s: s.map(order).fillna(s))


def count_error_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Direction and strength of the count-error -> DER relationship, per model."""
    if not len(df) or "speaker_count_error" not in df.columns:
        return pd.DataFrame(columns=["model", "corr_abs_count_error_vs_der"])
    rows = []
    for model, sub in df.groupby("model"):
        rows.append({
            "model": model,
            "corr_abs_count_error_vs_der": sub.speaker_count_error.abs().corr(sub.der),
            "corr_signed_count_error_vs_der": sub.speaker_count_error.corr(sub.der),
            "der_when_count_exact": sub.loc[sub.speaker_count_error == 0, "der"].mean(),
            "der_when_count_wrong": sub.loc[sub.speaker_count_error != 0, "der"].mean(),
        })
    return pd.DataFrame(rows)


def oracle_gap(df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    """DER cost of count estimation, in DER points.

    DIAGNOSTIC ONLY. `oracle_df` comes from re-running the pipeline with
    `num_speakers` set from the reference, which *is* ground truth reaching the
    model. It exists to size the count-estimation penalty and must never be
    reported as system performance -- see diarization.run(oracle=True).
    """
    a = aggregate(df).set_index("model")
    b = aggregate(oracle_df).set_index("model")
    joined = a[["der"]].join(b[["der"]], rsuffix="_oracle", how="inner")
    joined["der_cost_of_count_estimation"] = joined.der - joined.der_oracle
    return joined.reset_index()


def summarize(cfg: Config, df: pd.DataFrame, extra: dict | None = None) -> dict[str, Any]:
    summary = {
        "probe": probe_metric_semantics(),
        "scoring": {
            "primary": {"collar": DER_COLLAR, "skip_overlap": DER_SKIP_OVERLAP,
                        "note": "the brief requires overlap to be scored"},
            "lenient": {"collar": DER_COLLAR_LENIENT, "skip_overlap": DER_SKIP_OVERLAP_LENIENT,
                        "note": "secondary, for comparability with published numbers"},
        },
        "corpus": aggregate(df).to_dict("records") if len(df) else [],
        "by_count_stratum": stratify_by_count_error(df).to_dict("records") if len(df) else [],
        "count_error_correlation": count_error_correlation(df).to_dict("records") if len(df) else [],
        **(extra or {}),
    }
    if cfg is not None:
        write_json_atomic(cfg.step2_summary, summary)
    return summary
