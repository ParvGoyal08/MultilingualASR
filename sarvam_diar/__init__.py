"""Sarvam ASR assignment: diarization + ASR benchmarking pipeline.

Modules
-------
config      paths, constants, stage flags
utils       logging, subprocess, ffprobe, dotenv, atomic Drive writes
data        CSV fetch/cache, ground-truth parsing, ClipInput/ClipReference split
reference   scoring reference: UEM cropping, RTTM, text normalization
extraction  Step 1 - YouTube audio -> exact-length 16 kHz mono WAV
diarization Step 2 - pyannote community-1 / 3.1 inference
evaluation  DER / JER / speaker-count, region-restricted scoring
analysis    Step 2 rankings, error contribution, model comparison
explorer    standalone offline error-explorer export
asr         Step 3 - ASR on diarized segments (stub)
refinement  Step 4 - improvement pipeline (stub)
"""

__version__ = "0.2.0"
