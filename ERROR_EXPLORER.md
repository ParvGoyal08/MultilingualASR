# Running the error explorer

A local, offline UI for finding and labelling diarization failures. No server-side
code, no internet, no build step — everything it shows was computed by the Python
pipeline and serialised; the browser only draws it.

## 1. Get the export

Run notebook cell **2.7**, which writes `error_explorer/` and zips it. Then:

* **Colab** — the folder is in Drive under `sarvam_diarization/error_explorer/`.
  Download it, or grab `error_explorer.zip`.
* **Kaggle** — download `error_explorer.zip` from the **Output** tab (right
  panel) after the run, or **Save Version** first so it persists.

Unzip it anywhere.

## 2. Serve it

The UI fetches JSON, and browsers block `fetch()` from `file://`, so it needs a
static server. Use the one in the export:

```bash
cd error_explorer
python3 serve.py            # http://localhost:8000
```

Open <http://localhost:8000>. That is the whole setup.

> **Use `serve.py`, not `python -m http.server`.** The stdlib server does not
> implement HTTP `Range`: it answers every request with the entire file. A
> browser cannot seek in media served that way, so on our longest clip (30 min,
> 55 MB) dragging the scrubber appears to do nothing until the whole file has
> downloaded. `serve.py` is the stdlib handler plus `206 Partial Content`.

> Opening `index.html` by double-clicking will show an empty page. That is the
> `file://` restriction, not a broken export.

## 3. Add audio (optional)

Audio is **not** bundled — the corpus is 1.3 GB and does not belong in a repo or
a zip. The timeline, error regions, filtering and speaker rows all work without
it; only playback is missing, and the UI says so per clip.

Symlink rather than copy — the export is a view of the corpus, not a second
copy, and `serve.py` follows symlinks:

```bash
ln -sf /path/to/sarvam_diarization/audio_16k/*.wav audio/
```

Or copy, if the export has to move to another machine:

```bash
# just the clips you are inspecting
cp /path/to/audio_16k/Cku_X_SL7qU__60_660.wav error_explorer/audio/

# or all of them
cp /path/to/audio_16k/*.wav error_explorer/audio/
```

File names must match the clip id exactly: `audio/<clip_id>.wav`.

## 4. Using it

**Left panel** — every clip, sortable and filterable. Click a column header to
sort, or use the controls: min DER, min overlap %, min speakers, and a clip-id
search. **Worst failures** jumps to the biggest contributor.

**Right panel** — the selected clip:

| Row | Shows |
|---|---|
| error strip (top) | MISS / FA / CONFUSION stacked, with an OVERLAP bar above |
| `GT <speaker>` | one row per reference speaker |
| `PRED <speaker>` | one row per predicted speaker, **drawn in the colour of the reference speaker it was mapped to** |

That colour rule is the point: after optimal (Hungarian) mapping, a predicted
turn in the *wrong* colour **is** the confusion error, visible without reading a
number.

**Controls**

* click the timeline → seek, and the inspector shows the active GT speakers,
  predicted speakers, error type and reference text at that instant
* scroll → zoom, drag → pan, or the fit / 2× / 5× / 10× / 25× buttons
* space → play/pause
* the model dropdown switches between community-1, pyannote-3.1, reverb-v2 and
  any imported model, with a per-clip delta against the others
* the chips at the bottom list the longest error regions — click one to jump

## 5. What to look for

The rankings in notebook 2.5 tell you *which* clips to open; the explorer tells
you *why*. Some patterns worth checking:

* **A `PRED` row in the wrong colour for a long stretch** — speaker confusion,
  usually two reference speakers merged into one cluster.
* **MISS concentrated under an OVERLAP bar** — the model is emitting one speaker
  where the reference has two. 2,088 reference segments are fully nested inside
  another speaker's turn, and a one-speaker-per-frame model cannot represent
  those at all.
* **FA in the last stretch of `Cku_X_SL7qU`, `OxYCBQKZ3iY` or `CO_8ppdzq9U`** —
  those clips have 90 s / 12 s / 6 s of unannotated tail, so false alarm there is
  the reference's fault, not the model's.
* **Many short MISS slivers** — 20% of reference segments are under 0.5 s;
  backchannels most systems never fire on.

## Why the UI does no arithmetic

Every MISS / FA / CONFUSION region is computed in Python and verified at export
time to reproduce `pyannote.metrics`' own DER components to within a microsecond.
Recomputing anything in JavaScript would create a second implementation free to
drift from the one that produced the benchmark.

`data/clips.json` records that verification result, and the UI shows a red banner
if it ever fails — so a disagreement surfaces as a warning rather than a quietly
wrong picture.

## Filtering by error type and by speaker

The four coloured tags above the timeline are toggles.

* **Click one** to isolate it — click `FALSE ALARM` and only false-alarm regions
  are drawn and listed. Click it again to clear.
* **Shift-click** to combine, for questions like "miss and confusion, but not
  false alarm".

The speaker row below does the same for one speaker. Clicking `Speaker_D`
(or clicking that speaker's row label on the timeline) restricts everything to
regions where **that speaker is implicated**, and the panel underneath splits
their confusion by direction:

* *their speech credited to* — Speaker_D spoke, the model said someone else;
* *their row actually spoken by* — the model said Speaker_D, someone else spoke.

Those are different failures with different fixes, so they are never summed.

The region chips are relabelled under focus: instead of `CONFUSION` you get
`Speaker_D → Speaker_C`, naming the counterpart.

### These seconds are not DER seconds

Chip totals are **region seconds** — how long this speaker is implicated in an
error — and they deliberately do not partition DER:

* a confusion implicates *both* the speaker who spoke and the one credited, so
  the same second counts on both;
* where more reference speakers are lost than there are predicted speakers to
  have taken their turn, pyannote calls one of them *missed* rather than
  *confused*, and which one is arbitrary. Per-speaker miss therefore runs a
  little below the scored figure (about 3% of miss corpus-wide).

Use the chips to find **who**, and **with whom**; use the summary panel above
the timeline for **how much**. That panel is the scored figure.
