"""Step 2 - baseline diarization benchmarking. NOT IMPLEMENTED YET.

Intended shape, kept here so Step 1's checkpoint contract is already the input
this step expects:

* Input  : `results/step1_extraction.csv` rows with `status == "ok"`, whose WAVs
           are exactly `[0, requested_dur_sec]` at 16 kHz mono.
* Models : pyannote speaker-diarization-3.1, NeMo (MSDD / Sortformer),
           optionally diart / WeSpeaker for a third opinion.
* Output : `results/step2_diarization/{model}/{clip_id}.rttm` plus a per-clip
           metrics CSV, checkpointed the same way (sidecar written last).
* Score  : against `reference.load_reference(cfg, clip_id)`, which is already
           cropped to the UEM and same-speaker-unioned. Overlap must be scored
           (7.16% of corpus time), so metrics run with `skip_overlap=False` and
           no forgiveness collar by default.
"""

from __future__ import annotations

import pandas as pd

from .config import Config, StageFlags
from .data import ClipInput


def run(cfg: Config, inputs: list[ClipInput], flags: StageFlags | None = None) -> pd.DataFrame:
    """Diarize each clip. Receives ClipInput only -- no ground truth.

    In particular the true speaker count is NOT available here, so pyannote must
    be called without `num_speakers` / `min_speakers` / `max_speakers`. Passing
    the reference count is the classic leak and the brief rules it out.
    """
    raise NotImplementedError("Step 2 not implemented yet")
