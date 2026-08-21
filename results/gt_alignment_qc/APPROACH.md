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

## How much to check

39 flagged clips carry a total JER gain of 7.68. **The top 24 rows carry 80% of
it**; the top 15 carry roughly two thirds.

A defensible audit is about 28 rows:

* the **top 15 flagged** — settles most of the measurable effect
* the **5 low-impact flagged** — tests the detector where the stakes are low
* the **8 controls** — the only way to find misses

That is roughly an hour, and it yields three numbers worth reporting: precision
on high-impact flags, precision on low-impact flags, and whether any miss was
found.

## What gets reported

* **Headline: raw ground truth, uncorrected.** The brief fixes the scoring
  protocol, so the benchmark result is scored against the labels as given.
* **Diagnostic: hand-corrected ground truth**, applying only `CONFIRM` rows,
  labelled as such, reported beside the headline.
* **The limitation, stated plainly:** *N of 99 clips were hand-verified as
  having annotations displaced by 1.0–5.0 s. Correcting only those moves
  corpus DER by X and JER by Y. The detector that shortlisted them had
  precision P on the audited sample, so the unaudited remainder carries a
  corresponding uncertainty.*

The last sentence is what makes the unaudited clips honest rather than assumed.

## What stays true regardless

* the raw annotations are never modified — a label is a row in a manifest,
  applied to a copy at scoring time
* nothing here reaches the diarization or fusion pipeline
* corrections are never used to select or tune a model
