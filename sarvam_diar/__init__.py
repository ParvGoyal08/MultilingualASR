"""Sarvam ASR assignment: diarization + ASR benchmarking pipeline.

Modules
-------
config      paths, constants, stage flags
utils       logging, subprocess, ffprobe, atomic Drive writes
data        CSV fetch/cache, ground-truth parsing, dataset profiling
extraction  Step 1 - YouTube audio -> exact-length 16 kHz mono WAV
diarization Step 2 - baseline diarization benchmarking (stub)
asr         Step 3 - ASR on diarized segments (stub)
evaluation  DER / JER / cpWER / WDER (stub)
refinement  Step 4 - improvement pipeline (stub)
"""

__version__ = "0.1.0"
