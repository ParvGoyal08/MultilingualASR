# Running on Kaggle

`main_kaggle.ipynb` is `main.ipynb` with only the platform glue changed — same
package, same pipeline, same metrics. Four things differ on Kaggle and all of
them are settings, not code.

## 1. Upload the audio as a Dataset

Kaggle has no Drive mount; large inputs arrive as Datasets.

1. **Datasets → New Dataset**
2. Upload the **contents** of your local `local_out/`, keeping the folder names:

   ```
   audio_16k/    99 .wav   (~1.3 GB)  -- the pipeline input
   meta/         99 .json  (396 KB)   -- commit markers; the CSV rebuilds from these
   ```

   Those two are sufficient. `data/`, `results/` and `reference/` all regenerate.
3. Title it something like **Sarvam Diarization Audio**. Kaggle slugifies the
   title into the path, e.g. `/kaggle/input/sarvam-diarization-audio`.
4. Wait for it to finish processing — 1.3 GB takes a while.

## 2. Notebook settings (right-hand panel)

| Setting | Value | Why |
|---|---|---|
| **Internet** | **ON** | `pip install` and the `git clone` both need it. **Off by default** — this is the most common first failure. |
| **Accelerator** | **GPU T4 x2** or P100 | Step 2 inference |
| **Input** | your dataset | *Add Input* → your dataset |
| **Add-ons → Secrets** | `HF_TOKEN` | gated pyannote models (see 4) |

## 3. Point cell 1.1 at your dataset

```python
DATASET = Path("/kaggle/input/sarvam-diarization-audio")
```

Change it to match the Input panel. If it is wrong the cell fails immediately and
lists the datasets that *are* attached, rather than failing later.

Cell 1.1 then **symlinks** `audio_16k/` and `meta/` from the read-only dataset
into `/kaggle/working/sarvam_diarization/`. Nothing is copied, so no 1.3 GB
duplication, and every stage still writes its checkpoints normally.

## 4. HuggingFace token

Either:

* **Add-ons → Secrets** → add `HF_TOKEN`, tick it for this notebook. Persists
  across sessions, never touches disk. Skip cell 1.1b.
* Or run **cell 1.1b**, which writes `.env` under `/kaggle/working` via
  `getpass`. Convenient, but `/kaggle/working` is wiped between sessions.

Separately, accept the model conditions on all three pages, logged in as the
account that owns the token:

* https://huggingface.co/pyannote/speaker-diarization-community-1
* https://huggingface.co/pyannote/speaker-diarization-3.1
* https://huggingface.co/pyannote/segmentation-3.0

The third is easy to miss: 3.1 pulls it in, so accepting only the pipeline still
gives `GatedRepoError: 403`.

## 5. Run

1. Cell **1.0** — installs, then likely prints `RESTART THE KERNEL`. That is
   expected: pyannote upgrades numpy, and a live kernel holding the old one dies
   on the next import. **Run → Restart & clear cell outputs**, then run 1.0 again.
2. Cells **1.1 → 1.8** — config, reference, verification. Seconds.
3. Cell **2.1** — smoke run on 3 clips; prints measured RTF and a projected
   sweep time for **your** GPU. That projection is the decision point.
4. Cell **2.2** — full sweep. Checkpointed, safe to interrupt.
5. Cells **2.3 → 2.7** — scoring, rankings, error-explorer export.

## Differences from the Colab notebook

| | Colab | Kaggle |
|---|---|---|
| storage | Drive mount, persistent | `/kaggle/input` read-only + `/kaggle/working` **wiped between sessions** |
| internet | always on | **off by default** |
| secrets | Secrets panel | Add-ons → Secrets |
| restart | Runtime → Restart session | Run → Restart & clear cell outputs |
| download | `files.download()` | Output tab, or *Save Version* |

## Persisting results

`/kaggle/working` does not survive a session. Before you close the tab, either
**Save Version** (commits the notebook and its outputs) or download
`error_explorer.zip` from the Output tab. Otherwise the sweep has to re-run.

`run_extraction` stays **False**: Kaggle's IPs are bot-gated by YouTube exactly
as Colab's are, which is why the audio is uploaded rather than fetched.
