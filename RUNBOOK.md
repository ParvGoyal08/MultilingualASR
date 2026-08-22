# Runbook — who owns what

Three notebooks, split by **role** rather than by system.

| | role | writes? |
|---|---|---|
| `main.ipynb` | **the submission** — Steps 1–5 end to end from committed checkpoints, `SUBSET = 10 / 0` toggle. No GPU, no API key. | no |
| `main_kaggle.ipynb` | **the sweep notebook** — every major run that produced the checkpoints | yes, exclusively |
| `main_kaggle_2.ipynb` | **the experiment bench** — probes, hypotheses, approach decisions | **never** |

`main.ipynb` is what a reviewer runs. The two Kaggle notebooks are what produced
the artifacts it reads, and are kept for provenance.

`main_kaggle_2` runs in memory over a handful of clips and writes no checkpoint.
So it can be edited, re-run or thrown away at any time, including while
`main_kaggle` is mid-sweep, and it can never leave half-written state that
someone later scores by accident.

A finding is **promoted, not executed**: when a probe settles a question, flip
the corresponding flag in `main_kaggle` and sweep there. Never sweep in the
bench.

## The flags in `main_kaggle`

| flag | cell | meaning |
|---|---|---|
| `RUN_ASR` | 3.3 | master switch for the Sarvam sweep |
| `SMOKE_FIRST` | 3.3 | `True` = 3 clips, then stop and print REF vs HYP |
| `RUN_WHISPER` | 3.4 | long-form + per-segment Whisper; set after the bench's ratio test |

Nothing expensive runs unless one is flipped deliberately. Every `else` branch
prints what is already on disk, so a `False` flag still tells you where you are.

## Order of operations

1. `main_kaggle` 1.0 → 3.0. Builds the fusion; the bench asserts it exists.
2. `main_kaggle` 3.3, `SMOKE_FIRST = True`. Read the transcripts. Then `False`
   for the full Sarvam sweep (~70 min, ~₹580) and leave it running.
3. `main_kaggle_2` cells 1–3: long-form vs per-segment on 3 clips.
4. If the ratio lands near 1.0 → `RUN_WHISPER = True` in `main_kaggle` 3.4.
5. `main_kaggle` 3.5 / 3.6 for the current tables. Read-only, safe any time.

## Step 4b — script and numeral normalisation

Runs on the laptop, not Kaggle: it is text-only, so no audio and no GPU.

```bash
# applies the committed lookup tables -- no API key needed
./.venv/bin/python tools/run_translit.py --root local_out/step4_input

# rebuilds the tables from the hypothesis vocabulary (needs AWS_BEARER_TOKEN_BEDROCK)
./.venv/bin/python tools/run_translit.py --root local_out/step4_input --rebuild-tables
```

`table_path()` prefers `checkpoints/results/{translit,numeral}_table.json` over
the run root, so the default path reproduces the shipped result — 3,520 Latin
tokens and 697 numerals replaced — with no network access at all. Only
`--rebuild-tables` calls the model, and responses are cached by prompt hash, so
the prompt and model ID are part of the key and a stale response can never be
silently reused.

Writes `asr/<source>+xlit/` and `asr/<source>+xlit+num/`. Segment boundaries,
speakers and timestamps are copied through untouched, so `segmentation_key` still
matches the source and `audit_segmentation` stays green.

## Verifying the pipeline before trusting a sweep

```bash
./.venv/bin/python tools/audit_segmentation.py --root local_out   # expect: ok 99, MISMATCH 0
```

A mismatch means transcripts were produced against a different diarization than
the one now on disk — the defect that cost 0.0200 cpWER before it was caught.
Nothing downstream should be scored until this is clean.

## Editing the library while a sweep runs

Both notebooks `git reset --hard origin/main` on their first cell, so a push
changes what the *next* kernel restart loads, never a running one. Any library
change that affects a notebook currently mid-sweep gets called out explicitly —
which notebook, and whether it invalidates checkpoints already written.

Checkpoints carry a `settings_key` so a decoding change makes stale work visibly
stale rather than silently mixed. **On the per-segment path it does not yet do
that job.** The committed values are `beam?-cond1-lidself-batch0` (297 payloads)
and `None` (242): `run_segmented` builds the key from a meta dict that carries
none of `beam_size`, `condition_on_previous_text`, `lid_model` or `batched` — the
literal `?` is the missing beam width — and it records neither `merge_gap` nor
the diarization model. So for per-segment runs the key is effectively a constant
and fingerprints nothing.

What actually caught the stale-fusion bug was `segmentation_key`, which is
re-derived from each transcript's stored segment spans at audit time and compared
against the diarization on disk — see `tools/audit_segmentation.py` and
`WRITEUP.md` §5. Widening `settings_key` to cover the decoding parameters is
listed as future work.
