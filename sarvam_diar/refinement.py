"""Step 4 - improving the best baseline. NOT IMPLEMENTED YET.

Intended shape: take the best (diarization, ASR) pair from Steps 2-3 and correct
its output, then re-score on the same metrics. Ground truth is never an input.

Candidate signals, from the dataset profile:
* 20% of reference segments are under 0.5 s (backchannels). Baselines tend to
  either miss them or emit spurious short turns -- a duration prior plus
  transcript content is a cheap discriminator.
* Repeated / stuttered text spanning a speaker boundary suggests a false split;
  incoherent speaker alternation suggests speaker confusion.
* Overlap is 7.13% of corpus time, so an overlap-aware second pass has real
  headroom against a baseline that emits one speaker per frame.
"""

from __future__ import annotations

import pandas as pd

from .config import Config, StageFlags
from .data import ClipInput


def run(cfg: Config, inputs: list[ClipInput], flags: StageFlags | None = None) -> pd.DataFrame:
    """Improve the best baseline. Receives ClipInput only -- no ground truth."""
    raise NotImplementedError("Step 4 not implemented yet")
