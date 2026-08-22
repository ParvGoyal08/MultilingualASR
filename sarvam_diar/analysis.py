"""Rankings and interactive tables over the Step 2 metrics.

The point of this module is to make the corpus DER *legible*: which clips it
comes from, which error type dominates, and where the two models disagree.

One idea runs through it. **Ranking by error rate and ranking by error
contribution answer different questions.** A 50 s clip at DER 0.9 has a terrible
rate but contributes almost nothing to the corpus number; a 30 min clip at DER
0.3 may be a fifth of all the error in the benchmark. Fixing the first changes
nothing. So every ranking here carries both the rate and the share of total
corpus error seconds, and `worst_by_contribution()` is the one to act on.
"""

from __future__ import annotations

import pandas as pd

from .utils import LOG

ERROR_SEC_COLS = ["der_fa_sec", "der_miss_sec", "der_confusion_sec"]

# --------------------------------------------------------------- derived cols

def enrich(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add contribution and composition columns used by every ranking."""
    if not len(metrics):
        return metrics
    df = metrics.copy()
    df["error_sec"] = df[ERROR_SEC_COLS].sum(axis=1)

    # Share of the model's total corpus error that this clip accounts for.
    df["error_share"] = df.groupby("model").error_sec.transform(lambda s: s / s.sum())
    # Share of the model's scored seconds -- so error_share > dur_share means the
    # clip is punching above its weight.
    df["dur_share"] = df.groupby("model").der_total_sec.transform(lambda s: s / s.sum())
    df["over_representation"] = df.error_share / df.dur_share.replace(0, pd.NA)

    # Which error type dominates this clip.
    for c in ERROR_SEC_COLS:
        df[c.replace("_sec", "_frac")] = df[c] / df.error_sec.replace(0, pd.NA)
    df["dominant_error"] = (
        df[ERROR_SEC_COLS].idxmax(axis=1)
        .str.replace("der_", "", regex=False).str.replace("_sec", "", regex=False)
        .where(df.error_sec > 0, "none")
    )
    df["abs_count_error"] = df.speaker_count_error.abs()
    return df

# ------------------------------------------------------------------ rankings

def _rank(df, by, ascending=False, n=None, cols=None):
    if not len(df):
        return df
    out = df.sort_values(by, ascending=ascending, na_position="last")
    if cols:
        out = out[[c for c in cols if c in out.columns]]
    return out.head(n) if n else out

BASE = ["model", "clip_id", "clip_dur_sec", "der", "jer", "error_sec", "error_share",
        "dominant_error", "n_speakers_ref", "n_speakers_hyp"]

def worst_by_der(df, n=20):
    """Worst error RATE. Small clips dominate this -- see worst_by_contribution."""
    return _rank(df, "der", False, n, BASE)

def worst_by_jer(df, n=20):
    """JER weights every speaker equally, so it punishes missing a rare speaker
    far more than DER does. Divergence between the two rankings is the signal."""
    return _rank(df, "jer", False, n, BASE)

def worst_by_confusion(df, n=20):
    """Speech found but attributed to the wrong speaker -- a clustering failure."""
    return _rank(df, "der_confusion_sec", False, n,
                 BASE[:5] + ["der_confusion_sec", "der_confusion_frac", "n_speakers_ref",
                             "n_speakers_hyp", "speaker_count_error"])

def worst_by_miss(df, n=20):
    """Reference speech the system never detected -- a VAD/segmentation failure,
    and where nested and overlapped speech lands."""
    return _rank(df, "der_miss_sec", False, n,
                 BASE[:5] + ["der_miss_sec", "der_miss_frac", "overlap_sec",
                             "overlap_der", "n_speakers_ref"])

def worst_by_false_alarm(df, n=20):
    """Speech emitted where the reference has none. Watch the three clips with
    long unannotated tails -- their false alarm may be annotation, not the model."""
    return _rank(df, "der_fa_sec", False, n,
                 BASE[:5] + ["der_fa_sec", "der_fa_frac"])

def worst_by_overlap_der(df, n=20):
    """DER scored ONLY where >= 2 reference speakers are active.

    NaN for the 9 clips with no overlap. This is the hardest condition in the
    corpus and is invisible in the corpus DER, where it is ~7.13% of scored time.
    """
    sub = df[df.overlap_sec > 0] if "overlap_sec" in df.columns else df
    return _rank(sub, "overlap_der", False, n,
                 ["model", "clip_id", "overlap_der", "overlap_sec", "der",
                  "single_speaker_der", "n_speakers_ref"])

def worst_by_speaker_count(df, n=20):
    """Largest speaker-count errors, signed so over- and under-estimation are
    distinguishable. Note some clips are unwinnable: a reference speaker holding
    0.04 s cannot be detected by any clustering diarizer."""
    return _rank(df, "abs_count_error", False, n,
                 ["model", "clip_id", "clip_dur_sec", "n_speakers_ref", "n_speakers_hyp",
                  "speaker_count_error", "der", "jer"])

def worst_by_contribution(df, n=20):
    """Ranked by share of the model's total corpus error. THE actionable list:
    these are the clips whose improvement would actually move the headline."""
    return _rank(df, "error_share", False, n,
                 ["model", "clip_id", "clip_dur_sec", "der", "error_sec", "error_share",
                  "dur_share", "over_representation", "dominant_error"])

# ---------------------------------------------------------- model comparison

def model_comparison(df: pd.DataFrame, metric: str = "der",
                     a: str | None = None, b: str | None = None) -> pd.DataFrame:
    """Per-clip head-to-head. `delta` is positive where the first model is worse.

    This compares exactly TWO systems. With more than two present and no
    explicit `a`/`b`, it used to silently take the two alphabetically-first and
    drop the rest under a name that promised a head-to-head; now it says which
    pair it picked and what it ignored.
    """
    if not len(df) or df.model.nunique() < 2:
        return pd.DataFrame()
    wide = df.pivot_table(index="clip_id", columns="model", values=metric)
    available = list(wide.columns)
    if a is None or b is None:
        a, b = available[0], available[1]
        if len(available) > 2:
            LOG.warning("model_comparison: %d systems present, comparing %s vs %s "
                        "and ignoring %s -- pass a=/b= to choose",
                        len(available), a, b,
                        ", ".join(m for m in available if m not in (a, b)))
    missing = [m for m in (a, b) if m not in available]
    if missing:
        raise KeyError(f"model_comparison: {missing} not in {available}")
    models = [a, b]
    wide["delta"] = wide[a] - wide[b]
    def _better(r):
        # A clip only one model ran has a NaN delta. Calling that a tie would
        # inflate the tie count with clips that were never compared.
        if pd.isna(r["delta"]):
            return "n/a"
        return models[1] if r["delta"] > 0 else (models[0] if r["delta"] < 0 else "tie")

    wide["better"] = wide.apply(_better, axis=1)
    meta = df.drop_duplicates("clip_id").set_index("clip_id")[
        ["clip_dur_sec", "n_speakers_ref"]]
    return wide.join(meta).sort_values("delta", ascending=False).reset_index()

def head_to_head_summary(df: pd.DataFrame, metric: str = "der") -> pd.DataFrame:
    cmp = model_comparison(df, metric)
    if not len(cmp):
        return pd.DataFrame()
    models = [c for c in cmp.columns if c not in
              ("clip_id", "delta", "better", "clip_dur_sec", "n_speakers_ref")]
    return pd.DataFrame([{
        "metric": metric,
        f"{models[0]}_wins": int((cmp.better == models[0]).sum()),
        f"{models[1]}_wins": int((cmp.better == models[1]).sum()),
        "ties": int((cmp.better == "tie").sum()),
        "not_compared": int((cmp.better == "n/a").sum()),
        "mean_abs_delta": float(cmp.delta.abs().mean()),
        "max_delta_clip": cmp.iloc[0].clip_id if len(cmp) else None,
        "max_delta": float(cmp.delta.iloc[0]) if len(cmp) else None,
    }])

def error_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Where each model's total error seconds go, and how it splits by condition."""
    rows = []
    for model, sub in df.groupby("model"):
        err = sub[ERROR_SEC_COLS].sum()
        tot = sub.der_total_sec.sum()
        ov = sub[sub.overlap_sec > 0]
        rows.append({
            "model": model,
            "der": err.sum() / tot if tot else float("nan"),
            "false_alarm_%_of_error": 100 * err["der_fa_sec"] / err.sum(),
            "missed_%_of_error": 100 * err["der_miss_sec"] / err.sum(),
            "confusion_%_of_error": 100 * err["der_confusion_sec"] / err.sum(),
            "der_on_overlap": ov.overlap_der.mean() if len(ov) else float("nan"),
            "der_on_single_speaker": sub.single_speaker_der.mean(),
            "clips_dominated_by_miss": int((sub.dominant_error == "miss").sum()),
            "clips_dominated_by_confusion": int((sub.dominant_error == "confusion").sum()),
            "clips_dominated_by_fa": int((sub.dominant_error == "fa").sum()),
        })
    return pd.DataFrame(rows)

def pareto(df: pd.DataFrame, model: str | None = None, n: int = 20) -> pd.DataFrame:
    """How few clips account for how much of the error."""
    sub = df[df.model == model] if model else df
    s = sub.sort_values("error_sec", ascending=False)
    out = s[["clip_id", "error_sec", "error_share"]].head(n).copy()
    out["cumulative_share"] = out.error_share.cumsum()
    return out.reset_index(drop=True)

# -------------------------------------------------------------- presentation

def show(df: pd.DataFrame, caption: str = "", precision: int = 4, max_rows: int = 25):
    """Interactive table if itables is available, styled static table otherwise.

    itables gives sortable/searchable/paginated tables in Colab; the fallback
    keeps the colour scale so the notebook is still readable without it.
    """
    if not len(df):
        print(f"{caption}: (empty)")
        return
    try:
        from itables import show as ishow

        if caption:
            print(f"\n=== {caption} ===")
        ishow(df.round(precision), maxBytes=0, classes="display compact",
              scrollX=True, lengthMenu=[10, 25, 50, 100])
    except ImportError:
        from IPython.display import display

        if caption:
            print(f"\n=== {caption} ===")
        display(style(df.head(max_rows), precision))

_HEAT = {"der": "Reds", "jer": "Reds", "overlap_der": "Reds", "single_speaker_der": "Reds",
         "error_sec": "Oranges", "error_share": "Oranges", "der_miss_sec": "Blues",
         "der_confusion_sec": "Purples", "der_fa_sec": "Greens",
         "abs_count_error": "Reds", "over_representation": "Oranges"}

def style(df: pd.DataFrame, precision: int = 4):
    """Colour-scale the numeric columns so error type is visible at a glance."""
    st = df.style.format(precision=precision, na_rep="-")
    for col, cmap in _HEAT.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            st = st.background_gradient(cmap=cmap, subset=[col])
    if "delta" in df.columns:
        st = st.background_gradient(cmap="RdBu_r", subset=["delta"])
    return st

def all_rankings(metrics: pd.DataFrame, n: int = 20) -> dict[str, pd.DataFrame]:
    """Every ranking in one call, for the notebook to iterate over."""
    df = enrich(metrics)
    return {
        "Worst by DER (rate)": worst_by_der(df, n),
        "Worst by JER": worst_by_jer(df, n),
        "Worst by confusion (wrong speaker)": worst_by_confusion(df, n),
        "Worst by missed speech": worst_by_miss(df, n),
        "Worst by false alarm": worst_by_false_alarm(df, n),
        "Worst on OVERLAPPED speech only": worst_by_overlap_der(df, n),
        "Largest speaker-count errors": worst_by_speaker_count(df, n),
        "Biggest contributors to corpus DER": worst_by_contribution(df, n),
    }
