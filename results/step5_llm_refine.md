# Step 5 — Gemini transcript refinement: a measured null result

Run 2026-08-22, `gemini-3.5-flash`, temperature 0, prompt `v1`.
Pilot: 10 clips, **GT-blind selection**, dev split only, 41.9 min, 402 segments.
Code: `sarvam_diar/llm_refine.py`, `tools/llm_refine_probe.py`,
`tools/llm_refine_report.py`. Raw: `results/step5_llm_refine.json`,
`logs/step5_edits.jsonl`.

**Verdict: do not ship.** Every pre-registered abandon criterion fired.

## Result

| system | ratio | WER | cpWER | WDER | ins | ΔcpWER |
|---|---|---|---|---|---|---|
| baseline `saaras-v3@fusion` | 0.9734 | 0.2357 | **0.2498** | 0.0346 | 223 | — |
| **arm A** text correction | 0.9722 | 0.2369 | **0.2493** | 0.0334 | 223 | −0.0006 |
| **arm B** + overlap cleanup | 0.9700 | 0.2371 | **0.2498** | 0.0335 | 217 | +0.0000 |

| | arm A | arm B |
|---|---|---|
| mean Δ cpWER | −0.0014 | +0.0013 |
| 95% CI (paired bootstrap over clips) | [−0.0054, +0.0014] | [−0.0008, +0.0038] |
| CI excludes zero | **no** | **no** |
| per-clip wins / losses | 2 / 3 | 2 / 3 |
| segments changed / reverted | 10 / 4 | 14 / 5 |
| window fallbacks | 0 | 3 |

Against the abandon criteria fixed before the run:

- CI includes zero — **fires**
- win rate 20%, needed ≥60% — **fires**
- insertions did not fall (223 → 223 in arm A) — **fires**
- revert rate 4/14 = 29%, threshold 20% — **fires**

## What the architecture did right

**Structural integrity held completely.** All 20 refined payloads (10 clips × 2
arms) have a segmentation byte-identical to their source — same `segmentation_key`,
same speakers, same timestamps, same order, same count. Returning only `{id, text}`
made a timestamp or speaker change *structurally impossible* rather than merely
forbidden, and the pre-write assertion never fired.

**The model was appropriately conservative.** 24 applied edits across 402
segments. Every zero-overlap control clip came back untouched, which is the
correct behaviour and confirms the "when unsure, preserve" clause landed.

**The script guard caught real damage** — 4 reverts per arm, all `script_change`.
For example a Malayalam segment rewritten into Kannada:

```
old  ഇത്.
new  ಇದು.        -> reverted (script_change)
```

and a Devanagari segment partly rewritten into Gurmukhi:

```
old  अच्छा ते भी।
new  अच्छा ਤੇ ਵੀ.   -> reverted (unlicensed_budget)
```

## The failure the guards missed — and it is the interesting finding

**7 of 24 applied edits (29%) introduced a script that was absent from the
original**, and every one passed R1 because the *dominant* script did not change:

| introduced | count |
|---|---|
| Hangul (Korean) | 2 |
| Devanagari (into Bengali/Marathi) | 2 |
| Katakana / Hiragana (Japanese) | 1 |
| Bengali (into Marathi) | 1 |
| Thai | 1 |

```
Telugu  old  పార్టీ టిఆర్ఎస్ పార్టీతోను ...
        new  パーティー టిఆర్ఎస్ పార్టీతోను ...     (párti -> Japanese "party")

Bengali old  এটাও বলে দিলাম।
        new  것도 বলে দিলাম।                      (etao -> Korean "geotdo")

Marathi old  हे बघा, अदरवाईज खूप कठिन.
        new  হে বघा, अदरवाईज खूप कठिन.            (Devanagari he -> Bengali he)
```

The model is translating a *single token* into a phonetically similar word in an
unrelated script. This is not transliteration drift — it is cross-script lexical
substitution, and it is exactly the class of error a dominant-script guard cannot
see: one corrupted token out of forty leaves the dominant script intact.

It also explains why **WER got slightly worse (0.2357 → 0.2369) while cpWER
barely moved**: the edits are token-level damage, too sparse to shift the
aggregate but never an improvement.

**Lesson for guard design:** a dominant-script test is the wrong shape for this
failure. The correct rail is per-token — reject any output token whose script is
absent from the input segment. That is a two-line change and it would have caught
all 7. It is not applied here because the prompt and guards were frozen before
the run, and changing them after seeing the result would invalidate the pilot.

## Why this was always likely

The provenance fix (obs [50]) had already taken the prize this stage was aimed
at: cpWER 0.3381 → 0.3181, insertions −41%, duplicated tokens −66%. What was left
for arm B to remove is mostly real simultaneous speech, and the free deterministic
dedup rule had already reversed sign (−0.0064 → +0.0027) on the corrected
baseline. Arm B behaving like the baseline is the predicted outcome, not a bug.

For arm A, the target was the 91% of cpWER that is word error — but a text-only
refiner cannot hear the audio. It can fix what is *linguistically* implausible,
and on this corpus Saaras's residual errors are mostly not of that kind: they are
code-switched surface forms and genuinely ambiguous short turns, where the model
correctly declined to intervene.

## What would be needed to make this work

1. **Per-token script rail** instead of dominant-script (above) — removes 29% of
   the applied edits, all of them damage.
2. **Audio grounding.** The errors that remain need the signal, not the text.
3. **A larger pilot.** At 402 segments and 24 edits, the experiment cannot resolve
   an effect smaller than ~0.005 cpWER; the free-tier quota
   (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`) capped the run at 20
   calls, forcing whole-clip windows instead of the designed 12-segment ones.
