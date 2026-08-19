"""Metrics: DER, JER, speaker-count accuracy, cpWER, WDER. NOT IMPLEMENTED YET.

Scoring rules this corpus forces:

* **Overlap is scored.** 7.16% of corpus time has >= 2 distinct speakers active
  and only 9 of 100 clips have none, so `skip_overlap=False` and a 0.0 s collar
  are the defaults. A collared / overlap-forgiving variant may be reported
  alongside, never instead.
* **UEM is `[0, requested_dur_sec]`** for every clip, and `reference.py` has
  already cropped the reference to it and unioned each speaker's self-overlaps.
  Score against `reference.load_reference()`, never against a re-parse of the
  raw CSV.
* **Text normalization is done** -- `Utterance.text_norm` is the scoring form.
  Apply `reference.normalize_text(hyp, strip_gloss=False)` to hypotheses so both
  sides go through the identical function.
"""

from __future__ import annotations

from typing import Any

from .data import ClipReference


def der(reference: Any, hypothesis: Any, uem: Any = None, collar: float = 0.0,
        skip_overlap: bool = False) -> dict[str, float]:
    raise NotImplementedError("evaluation not implemented yet")


def jer(reference: Any, hypothesis: Any, uem: Any = None) -> float:
    raise NotImplementedError("evaluation not implemented yet")


def cp_wer(ref: ClipReference, hypothesis_turns: list) -> dict[str, float]:
    raise NotImplementedError("evaluation not implemented yet")


def wder(ref: ClipReference, hypothesis_turns: list) -> float:
    raise NotImplementedError("evaluation not implemented yet")
