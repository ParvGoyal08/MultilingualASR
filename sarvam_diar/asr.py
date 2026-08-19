"""Step 3 - ASR over diarized segments. NOT IMPLEMENTED YET.

Intended shape:

* Input  : Step 2 RTTMs + the Step 1 WAVs.
* Systems: Sarvam Saaras, Whisper large-v3 (and/or faster-whisper), IndicWhisper.
* Output : speaker-attributed transcripts + cpWER / WDER per model per clip.

Dataset facts from Step 1's profile that shape this step:

* The corpus spans 9 Indic scripts (Devanagari 25, Gujarati 12, Telugu 12,
  Tamil 10, Kannada 9, Odia 9, Bengali 8, Gurmukhi 8, Malayalam 7), so a single
  fixed language code will not do.

  **The language code must be derived from the audio** -- Whisper's own language
  ID, or left unset so the model detects it -- and never from `ref_lang_script` /
  `ref_lang_hint`. Those columns are computed from the ground-truth transcript's
  script, so feeding them to the ASR would hand the pipeline free language ID
  that the brief reserves for scoring. They exist for reporting and error
  analysis only. (`data.ClipInput`, which is what `run()` receives, does not
  carry them at all.)

* Reference text normalization lives in `reference.normalize_text` and is
  already done -- call it on hypotheses with `strip_gloss=False`, which is the
  same function and therefore symmetric. Do not write a second normalizer here.
"""

from __future__ import annotations

import pandas as pd

from .config import Config, StageFlags
from .data import ClipInput


def run(cfg: Config, inputs: list[ClipInput], flags: StageFlags | None = None) -> pd.DataFrame:
    """Transcribe diarized segments. Receives ClipInput only -- no ground truth."""
    raise NotImplementedError("Step 3 not implemented yet")
