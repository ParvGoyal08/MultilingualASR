"""Step 2 - run open-source diarizers over the extracted clips.

Roster: pyannote `speaker-diarization-community-1` (primary) and
`speaker-diarization-3.1` (the baseline it replaced). Both are pyannote.audio
pipelines, so one code path covers both and the delta between them is the
version-over-version comparison.

Two things this module is strict about:

1. **No speaker-count hints.** `num_speakers` / `min_speakers` / `max_speakers`
   are never passed on the normal path. The pipeline's own estimate is exactly
   what speaker-count accuracy measures, and the brief rules out feeding the
   reference count to the model. `run(oracle_counts=...)` is the single, fenced
   exception -- a diagnostic ablation that writes to a separate tree.

2. **The overlapping output, not the exclusive one.** community-1 also exposes
   `exclusive_speaker_diarization`, an overlap-free view meant to make
   transcript reconciliation easy. Using it would silently discard the overlap
   the brief requires us to score (7.13% of scored time here).

Checkpointing mirrors Step 1 exactly: RTTM first, sidecar JSON last as the
commit marker, and a clip counts as done only when both exist.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import DIARIZATION_MODELS, Config, StageFlags
from .data import ClipInput, Turn
from .reference import parse_rttm, rttm_safe
from .utils import (
    LOG,
    append_jsonl,
    apply_selection,
    load_dotenv,
    atomic_publish,
    human_time,
    now_utc_iso,
    read_json,
    tail,
    write_json_atomic,
)

# Kwargs that would leak the reference speaker count into the model.
SPEAKER_HINT_KWARGS = ("num_speakers", "min_speakers", "max_speakers")

_PIPELINE_CACHE: dict[str, Any] = {}


# ------------------------------------------------------------------ loading


TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")


def resolve_token(cfg: Config) -> str | None:
    """HF token, resolved from .env first.

    Order: explicit Config value, then .env (the config root, then the usual
    local spots), then the process environment, then the host secret store
    (Colab Secrets, or Kaggle Add-ons > Secrets).
    Never read from a notebook cell -- the notebooks go to a public repo.
    """
    import os

    if cfg.hf_token:
        return cfg.hf_token

    values = load_dotenv([cfg.dotenv_path], export=True)
    for key in TOKEN_KEYS:
        if values.get(key):
            return values[key]
        if os.environ.get(key):
            return os.environ[key]

    # Platform secret stores, as a last resort. Same role as .env, just managed
    # by the host -- Colab's Secrets panel and Kaggle's Add-ons > Secrets.
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        for key in TOKEN_KEYS:
            try:
                if tok := userdata.get(key):
                    return tok
            except Exception:
                continue
    except ImportError:
        pass

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        client = UserSecretsClient()
        for key in TOKEN_KEYS:
            try:
                if tok := client.get_secret(key):
                    return tok
            except Exception:
                continue
    except ImportError:
        pass
    return None


def load_pipeline(cfg: Config, model_key: str, device: str | None = None):
    """Load (and cache) a pyannote pipeline, moved to GPU when one is available."""
    if model_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_key]

    import torch
    from pyannote.audio import Pipeline

    repo = DIARIZATION_MODELS.get(model_key, model_key)
    token = resolve_token(cfg)
    if not token:
        raise RuntimeError(
            f"{repo} is gated and no HuggingFace token was found.\n"
            f"Add a line  HF_TOKEN=hf_...  to {cfg.dotenv_path}\n"
            "(on Colab that path is in Drive; .env is gitignored so it never "
            "reaches the public repo).\n"
            "You must also accept the model conditions on its HuggingFace page."
        )

    try:
        pipeline = Pipeline.from_pretrained(repo, token=token)
    except TypeError:
        # pyannote.audio < 4 spells it differently.
        pipeline = Pipeline.from_pretrained(repo, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained returned None for {repo}. That means the "
            "token is valid but the model conditions have not been accepted -- "
            f"visit https://hf.co/{repo} and accept them."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    LOG.info("loaded %s on %s", repo, device)
    _PIPELINE_CACHE[model_key] = pipeline
    return pipeline


def device_report() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "cuda": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
    }


# --------------------------------------------------------------- conversion


def annotation_to_turns(annotation) -> list[Turn]:
    turns = [
        Turn(str(label), float(seg.start), float(seg.end))
        for seg, _, label in annotation.itertracks(yield_label=True)
        if seg.end > seg.start
    ]
    turns.sort(key=lambda t: (t.start, t.end, t.speaker))
    return turns


def _extract_annotation(output):
    """Get the OVERLAPPING diarization out of whatever the pipeline returned.

    pyannote 3.x returns an Annotation directly. community-1 (4.x) returns a
    result object carrying both `speaker_diarization` and the overlap-free
    `exclusive_speaker_diarization`; we deliberately take the former.
    """
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization
    if hasattr(output, "itertracks"):
        return output
    raise TypeError(f"unrecognised pipeline output: {type(output)}")


def to_rttm(clip_id: str, turns: Iterable[Turn]) -> str:
    return "".join(
        f"SPEAKER {clip_id} 1 {t.start:.3f} {t.end - t.start:.3f} "
        f"<NA> <NA> {rttm_safe(t.speaker)} <NA> <NA>\n"
        for t in turns
    )


# ---------------------------------------------------------------- one clip


def diarize_clip(cfg: Config, pipeline, clip: ClipInput,
                 num_speakers: int | None = None) -> tuple[list[Turn], dict]:
    """Diarize one clip. `num_speakers` is ONLY set by the oracle ablation."""
    import shutil

    src = Path(clip.wav_path)
    # Copy off the Drive FUSE mount first: reading 1.3 GB across it during GPU
    # work is slow, and this mirrors what extraction already does.
    local = cfg.work_dir / f"{clip.clip_id}.wav"
    local.parent.mkdir(parents=True, exist_ok=True)
    if str(src) != str(local):
        shutil.copyfile(src, local)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = int(num_speakers)

    started = time.perf_counter()
    try:
        output = pipeline(str(local), **kwargs)
    finally:
        if str(src) != str(local):
            local.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started

    turns = annotation_to_turns(_extract_annotation(output))
    return turns, {
        "elapsed_sec": round(elapsed, 2),
        "rtf": round(elapsed / clip.duration, 5) if clip.duration else None,
        "n_turns": len(turns),
        "n_speakers_hyp": len({t.speaker for t in turns}),
        "max_end": round(max((t.end for t in turns), default=0.0), 3),
    }


def is_done(cfg: Config, model: str, clip_id: str, oracle: bool = False) -> bool:
    """Both artifacts present AND the RTTM holds the turns the sidecar claims.

    Existence alone is not enough, for the same reason Step 1 validates sample
    counts: a run interrupted mid-copy leaves a short but plausible file. The
    sidecar records n_turns, so the RTTM is checked against it -- an empty
    hypothesis is legitimate only when the sidecar also says zero turns.
    """
    rttm, meta_path = cfg.hyp_rttm_path(model, clip_id, oracle), cfg.hyp_meta_path(
        model, clip_id, oracle)
    if not rttm.exists() or not meta_path.exists():
        return False
    meta = read_json(meta_path, None)
    if not meta or meta.get("status") != "ok":
        return False
    expected = meta.get("n_turns")
    if expected is None:
        # Every sidecar this module writes records n_turns. One without it was
        # not written by a completed run, so fail safe and re-diarize rather
        # than trust a file of unknown provenance.
        return False
    lines = [ln for ln in rttm.read_text(encoding="utf-8").splitlines()
             if ln.startswith("SPEAKER ")]
    return len(lines) == expected


def load_hypothesis(cfg: Config, model: str, clip_id: str,
                    oracle: bool = False) -> list[Turn]:
    return parse_rttm(cfg.hyp_rttm_path(model, clip_id, oracle).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- runner


def run(cfg: Config, inputs: list[ClipInput], flags: StageFlags | None = None,
        models: list[str] | None = None,
        oracle_counts: dict[str, int] | None = None) -> pd.DataFrame:
    """Diarize every clip with every model, checkpointing after each.

    `inputs` is `list[ClipInput]` by type, so the reference speaker count and
    language are not merely discouraged here -- they are absent.

    `oracle_counts` is the fenced diagnostic: when given, each clip is run with
    `num_speakers` from the reference and results go to `hypotheses_oracle/`.
    Never use its output as a system result.
    """
    flags = flags or StageFlags()
    models = models or list(DIARIZATION_MODELS)
    oracle = oracle_counts is not None
    if oracle:
        LOG.warning("ORACLE ABLATION: feeding reference speaker counts to the model. "
                    "Diagnostic only -- never report this as system performance.")

    for clip in inputs:
        if not isinstance(clip, ClipInput):
            raise TypeError(f"run() takes ClipInput only, got {type(clip)}")

    run_started = time.perf_counter()
    rows: list[dict] = []

    for model in models:
        todo = apply_selection(inputs, flags)
        pipeline = None
        counts = {"ok": 0, "skipped": 0, "failed": 0}

        for i, clip in enumerate(todo, start=1):
            prefix = f"[{model} {i}/{len(todo)}] {clip.clip_id}"

            if not flags.force_redo and is_done(cfg, model, clip.clip_id, oracle):
                meta = read_json(cfg.hyp_meta_path(model, clip.clip_id, oracle), {}) or {}
                rows.append({"model": model, "clip_id": clip.clip_id, "status": "ok", **meta})
                counts["skipped"] += 1
                continue

            if pipeline is None:                       # load lazily: a fully
                pipeline = load_pipeline(cfg, model)   # cached model costs nothing
            try:
                if oracle:
                    if clip.clip_id not in oracle_counts:
                        # Falling through with num_speakers=None would write a
                        # NORMAL result into hypotheses_oracle/ and silently
                        # contaminate the ablation.
                        raise KeyError(
                            f"oracle_counts has no entry for {clip.clip_id}; refusing "
                            "to write a non-oracle result into the oracle tree")
                    n_spk = oracle_counts[clip.clip_id]
                else:
                    n_spk = None
                turns, meta = diarize_clip(cfg, pipeline, clip, num_speakers=n_spk)
            except Exception as exc:
                counts["failed"] += 1
                append_jsonl(cfg.step2_failures_jsonl, {
                    "ts_utc": now_utc_iso(), "model": model, "clip_id": clip.clip_id,
                    "oracle": oracle, "error_class": type(exc).__name__,
                    "error_msg": tail(str(exc), 500),
                })
                LOG.error("%s FAILED [%s] %s", prefix, type(exc).__name__, str(exc)[:200])
                rows.append({"model": model, "clip_id": clip.clip_id, "status": "failed",
                             "error_class": type(exc).__name__})
                continue

            tmp = cfg.work_dir / f"{model}_{clip.clip_id}.rttm"
            tmp.write_text(to_rttm(clip.clip_id, turns), encoding="utf-8")
            atomic_publish(tmp, cfg.hyp_rttm_path(model, clip.clip_id, oracle))
            tmp.unlink(missing_ok=True)

            record = {"model": model, "clip_id": clip.clip_id, "status": "ok",
                      "oracle": oracle, "clip_dur_sec": clip.duration,
                      "diarized_at_utc": now_utc_iso(), **meta}
            write_json_atomic(cfg.hyp_meta_path(model, clip.clip_id, oracle), record)
            rows.append(record)
            counts["ok"] += 1
            LOG.info("%s  %d turns / %d spk  %.1fs  rtf %.4f", prefix,
                     meta["n_turns"], meta["n_speakers_hyp"], meta["elapsed_sec"], meta["rtf"] or 0)

        LOG.info("%s: %d ok, %d skipped, %d failed", model,
                 counts["ok"], counts["skipped"], counts["failed"])

    LOG.info("step 2 done in %s", human_time(time.perf_counter() - run_started))
    return pd.DataFrame(rows)


def import_external_rttm(cfg: Config, src_dir, model: str,
                         clip_durations: dict[str, float] | None = None) -> pd.DataFrame:
    """Adopt RTTMs produced OUTSIDE this environment as a model's hypotheses.

    The escape hatch for systems that cannot share an environment with
    pyannote.audio 4.x -- DiariZen pins numpy==1.26.4, NeMo pins
    lightning<=2.4.0 -- but whose output is still just speaker turns. Run them in
    their own notebook, drop the RTTMs in a folder named <clip_id>.rttm, and
    point this at it. Scoring, ranking and the error explorer then treat the
    model exactly like the ones run in-process.

    Writes the same RTTM + sidecar pair as a normal run, so is_done() and every
    downstream stage behave identically.
    """
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise NotADirectoryError(f"{src_dir} is not a directory")

    rows = []
    for rttm in sorted(src_dir.glob("*.rttm")):
        clip_id = rttm.stem
        turns = parse_rttm(rttm.read_text(encoding="utf-8"))
        # Normalised through our own writer so labels and precision match the
        # in-process models exactly -- an imported model must not differ from a
        # native one in anything but its origin.
        tmp = cfg.work_dir / f"import_{model}_{clip_id}.rttm"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(to_rttm(clip_id, turns), encoding="utf-8")
        atomic_publish(tmp, cfg.hyp_rttm_path(model, clip_id))
        tmp.unlink(missing_ok=True)

        dur = (clip_durations or {}).get(clip_id)
        record = {
            "model": model, "clip_id": clip_id, "status": "ok", "oracle": False,
            "clip_dur_sec": dur,
            "elapsed_sec": None, "rtf": None,       # unknown: run elsewhere
            "n_turns": len(turns),
            "n_speakers_hyp": len({t.speaker for t in turns}),
            "max_end": round(max((t.end for t in turns), default=0.0), 3),
            "imported_from": str(rttm),
            "diarized_at_utc": now_utc_iso(),
        }
        write_json_atomic(cfg.hyp_meta_path(model, clip_id), record)
        rows.append(record)

    LOG.info("imported %d RTTMs as model '%s' from %s", len(rows), model, src_dir)
    return pd.DataFrame(rows)


def throughput_report(df: pd.DataFrame, corpus_sec: float = 44193.0) -> pd.DataFrame:
    """Measured RTF and the projected full-sweep time. No estimates, only data."""
    rows = []
    for model, sub in df[df.status == "ok"].groupby("model"):
        rtf = pd.to_numeric(sub.rtf, errors="coerce").dropna()
        if not len(rtf):
            continue
        rows.append({
            "model": model,
            "n_clips_measured": len(rtf),
            "rtf_mean": rtf.mean(),
            "rtf_min": rtf.min(),
            "rtf_max": rtf.max(),
            "measured_sec": pd.to_numeric(sub.elapsed_sec, errors="coerce").sum(),
            "projected_full_sweep_sec": rtf.mean() * corpus_sec,
            "projected_full_sweep": human_time(rtf.mean() * corpus_sec),
        })
    return pd.DataFrame(rows)
