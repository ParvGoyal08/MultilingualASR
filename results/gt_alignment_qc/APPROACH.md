# Ground-truth alignment: the approach, and why it is framed this way

The brief anticipates this situation:

> *"If for any reason the ground-truth labels look incorrect, hand-label a
> subset and call out the limitation."*

So that is what this is: **a hand-labelled subset and a stated limitation.**
Not an automatic correction of the ground truth.

That distinction is the whole design, and it matters because the two framings
are not equally defensible.

## Why not present it as an automatic method

An automatic detector infers the shift from agreement between the ground truth
and *the models being scored*. Corrections derived that way are open to a
straightforward objection: the target has been moved toward the shooters. Even
using IoU on speech activity rather than DER makes it indirect, not immune — if
several models were wrong the same way, the "correction" would encode their
shared error and the corrected numbers would flatter them.

Nothing in the arithmetic answers that objection. **A human ear does.**

## The two roles, kept separate

| | role | decides what? |
|---|---|---|
| the detector | **triage** — which clips are worth listening to | nothing |
| the human | **labelling** — is this clip actually misaligned, and by how much | everything |

Using a script to choose what you listen to is uncontroversial; a reviewer who
objects to that objects to sorting a list. What is applied to any reported
number is a label a person confirmed by ear against the source video.

## The trap this is designed around

Rows are ranked by how much correcting them would improve JER, because
listening time is finite and should be spent where it matters.

**That ranking must never become the acceptance test.** Selecting clips by how
much they flatter the metric and then accepting them *because* they flatter the
metric is circular, and it is the exact failure the ranking makes tempting. The
`verdict` column is the only thing that decides. A row with a large JER gain and
a `REJECT` verdict is not applied, and it is more informative than any
confirmation — it means the detector fired on a clip that is actually fine.

Two further guards follow from the same worry:

* **Low-impact flagged clips are included.** The five smallest gains sit at the
  end of the sheet, unbulldozed by the ranking. If the detector is only right
  when the stakes are high, that shows up here.
* **Unflagged controls are included.** Eight clips the detector did NOT flag,
  scored as if shifted by their own best lag. Auditing only candidates can
  measure precision at best; it can never notice a defect that was missed. A
  control confirmed as misaligned is a **detector miss**, and worth more than
  ten confirmations.

## Procedure

```bash
python3 tools/gt_hand_label.py             # build results/gt_alignment_qc/hand_labels.csv
# ... fill in the verdict column ...
python3 tools/gt_hand_label.py --report    # score with hand labels only
```

For each row the sheet gives a **transcript** and two YouTube links. The ground
truth claims those words start at one timestamp; the proposed shift claims the
other. Open both. The words are audible at exactly one.

* `CONFIRM` — the shifted timestamp is right, the annotation is displaced
* `REJECT` — the ground truth is right, the detector is wrong
* `UNSURE` — cannot tell; neither applied nor counted against the detector

`confirmed_shift_sec` overrides the proposal when the true offset differs; leave
blank to accept it. Shifts are on a 0.5 s grid, which is about what an ear can
resolve — claiming 10 ms from a listening test would be false precision.

The transcript check is used rather than a first-onset check because several
flagged clips begin speaking immediately, where both sides agree at 0.01 s and
the comparison decides nothing. It is also **lexical**, so it shares nothing
with the acoustic models that proposed the shift — which is precisely the
independence the objection above demands.

## Verifying without speaking the language

Reading a transcript in nine Indic scripts is not a reasonable ask, and a
verification step that is hard to perform is a verification step that gets done
badly. **Look at the waveform instead.**

Each diagnostic SVG now draws the actual audio envelope above the annotation
bands. A misaligned clip is visible as a shape mismatch: `GT raw` claims speech
where the waveform is flat, and `GT shifted` snaps onto the loud parts. That
judgement is language-independent and takes seconds — no listening, no reading,
no comprehension.

```bash
open results/gt_alignment_qc/diagnostics/Jc5AVwg2cZM__153_214.svg
```

The clips easiest to judge this way, measured as the share of annotated speech
sitting on audible silence:

| clip | shift | GT-on-silence, raw | shifted | swing |
|---|---|---|---|---|
| `Jc5AVwg2cZM__153_214` | −1.5 s | 28.1% | 16.8% | 11.4 pts |
| `2iMXYxBTwbM__339_401` | −4.5 s | 27.8% | 19.8% | 8.0 pts |
| `86mMTUeDiR8__181_1079` | −1.0 s | 29.0% | 22.2% | 6.7 pts |
| `6f6TLzlP8Wk__82_144` | −2.0 s | 31.1% | 25.2% | 5.9 pts |
| `6ZeRgvDHwcI__6_100` | −3.0 s | 18.9% | 13.3% | 5.6 pts |

Note the residual: even corrected, a fifth of annotated speech still overlaps
low-energy audio. That is expected — the energy VAD is crude and real speech has
quiet moments — which is why the *swing* matters rather than the absolute level.

Listening is still the stronger evidence where it is practical, and the
transcript links remain in the sheet for that. But the visual check is what
makes a spot audit realistic, and a spot audit that actually happens beats a
thorough one that does not.

## How much to check — and the lighter option

The full audit is ~28 rows. **It is not required, and given how the results are
reported it may not be worth it.**

Consider what the corrections are actually for. They never touch the headline,
which is scored against raw ground truth. They never touch Step 3: cpWER and
WDER compare TEXT grouped by speaker and never compare a reference timestamp to
a hypothesis one, so a shift is invisible to them — and shifting earlier can
only push utterances below t=0 and *lose* reference text, which changed 7 of 20
clips in a test. **Alignment corrections are a DER/JER diagnostic and nothing
else.**

So the question a reviewer will actually ask is not "is each of your 39 shifts
exactly right" but "is this dataset defect real, and how big is it". That needs
a handful of confirmations, not an exhaustive audit.

**The proportionate version — 5 clips, ten minutes:**

1. Open the five SVGs in the table above. Confirm `GT raw` sits on flat audio
   and `GT shifted` sits on loud audio.
2. Record the verdicts in `hand_labels.csv`.
3. Report the limitation with the audited count stated honestly.

That establishes the phenomenon beyond argument. The remaining flagged clips are
then described as *detected by the same procedure*, with the spot-check as
evidence the procedure works — rather than each being claimed as individually
verified.

**Do the full 28-row audit only if** you intend to publish the QC-adjusted
numbers as a headline result. Nothing in the plan requires that, and the
argument for it is weaker than the argument for reporting the defect and moving
on.

## What gets reported

* **Headline: raw ground truth, uncorrected.** The brief fixes the scoring
  protocol, so the benchmark result is scored against the labels as given.
* **Step 3 (ASR): raw ground truth, always.** Shifts cannot help cpWER or WDER
  and can only lose reference text at the clip boundary.
* **Diagnostic: alignment-adjusted DER/JER**, labelled as such, beside the
  headline — showing how much of the measured diarization error is annotation
  rather than model.
* **The limitation, as actually supported.** The audit performed was a visual
  spot-check of 15–20 of the 39 flagged clips against the waveform, and the
  large majority matched the proposed shift. Individual verdicts were not
  recorded, so the wording has to match that:

  > *"Cross-model consensus with independent energy-VAD corroboration flags 39
  > of 99 clips as having annotations displaced by 1.0–5.0 s, every one in the
  > same direction. A visual spot-check of 15–20 of them against the audio
  > waveform found the large majority correct. Applying the 23 shifts that carry
  > two independent corroborations moves corpus DER from 0.2625 to 0.2039 for
  > diarizen-large — reported as a diagnostic, not as a benchmark result."*

  Note what that sentence does **not** claim: no individual shift is described
  as verified, and no precision figure is quoted, because per-clip verdicts were
  not written down. The spot-check supports the **procedure**, not the
  individual labels. That is a weaker claim than a full audit would license, and
  it is the one the evidence actually carries.

  Recording per-clip verdicts in `hand_labels.csv` would upgrade this to a
  measured precision — "P of N audited flags confirmed" — which is worth doing
  only if the adjusted numbers are ever promoted to a headline.

## What stays true regardless

* the raw annotations are never modified — a label is a row in a manifest,
  applied to a copy at scoring time
* nothing here reaches the diarization, fusion or ASR pipeline
* corrections are never used to select or tune a model
* Step 3 metrics are computed against raw ground truth only

