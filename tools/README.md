# tools/

Probes and drivers. **Nothing here writes under `asr/`, `hypotheses/` or
`results/` unless it says so** — probes run in memory and print, so they can be
re-run or thrown away at any time without invalidating a sweep.

Every script takes `--root` (the data root) and works against `checkpoints/`
unless told otherwise.

## Step 4b — script correction

| script | what it does | writes |
|---|---|---|
| `run_translit.py` | applies the committed corpus-vocabulary lookup tables (§6.1). Prefers `checkpoints/results/*_table.json`, so it needs **no API key**; `--rebuild-tables` is the only path that calls a model. | `asr/<src>+xlit/`, `+xlit+num/` |
| `xlit_perclip_experiment.py` | the **shipped** per-clip stage (§6.2). One clip at a time, Sonnet 4.6, temperature 0, config frozen at `f284e59`. | `local_out/experiments/xlit_perclip/` |
| `xlit_perclip_score.py` | scores that experiment. **The only place it opens ground truth is here, after inference.** | `experiments/.../score_*.json` |

## Provenance and audit

| script | what it does |
|---|---|
| `audit_segmentation.py` | checks every stored transcript against the diarization it was cut on. This is what caught the stale-fusion defect in §5. Run it before trusting any sweep. |
| `gt_alignment_qc.py` | reference-alignment detector; writes `results/gt_alignment_qc/`. Corrections are applied to a **copy** — the raw annotations are never modified, and no reported metric uses them. |
| `gt_hand_label.py`, `gt_qc_worksheet.py` | manual verification worksheets for that audit. |

## ASR probes

| script | what it does |
|---|---|
| `whisper_probe.py` | Whisper configuration sweep (8 configurations × 10 clips). Concluded: configuration is not the problem, do not sweep. |
| `whisper_ab.py` | decoding A/B on 3 clips, in memory. No result file was kept. |
| `indic_probe.py` | AI4Bharat IndicConformer-600M. `--lang` defaults to `lid`; `--lang oracle` feeds reference-derived language and is a diagnostic only. |
| `llm_refine_probe.py` | Step 5 LLM contextual refinement, arms A and B, GT-blind clip selection. Measured null; not shipped. |
| `llm_refine_report.py` | reports that pilot. Note its raw artifacts were overwritten — see `results/step5_llm_refine.md`. |

## Explorer regression harnesses

`explorer_smoke.mjs` and `explorer_verify.mjs` are the error explorer's test
suite — it is a single HTML file with no build step and no test framework. Both
take an export directory as their only argument and need nothing installed.

`adopt_export.py` re-points an export at a different data root.
