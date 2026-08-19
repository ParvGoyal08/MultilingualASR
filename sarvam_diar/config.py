"""Paths, constants and stage toggles for the whole pipeline.

Everything that a run depends on is reachable from a single `Config` instance so
the notebook has exactly one place to look, and so a stage can be re-entered
after a Colab disconnect with identical paths.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- audio format
# The published WAV contract. These are hard requirements, not preferences:
# Step 2/3 index into the audio with ground-truth timestamps that are relative
# to start_sec, so the clip timeline must be exactly [0, end_sec - start_sec].
SAMPLE_RATE = 16_000
CHANNELS = 1
SF_SUBTYPE = "PCM_16"  # soundfile subtype -> pcm_s16le

# --------------------------------------------------------------- extraction
# Diagnostic threshold ONLY. If a fetched clip is off by more than this we treat
# the fetch as untrustworthy and fall back to a full download. It never defines
# the boundary of the published WAV -- that is always exactly n_expected samples.
DUR_TOL_SEC = 0.25
# Enough attempts to walk the whole client ladder on the cheap ranged fetch and
# still fall back to one full download. See extraction.attempt_plan().
MAX_ATTEMPTS = 4
BACKOFF_BASE_SEC = 4.0
SUBPROCESS_TIMEOUT_SEC = 1800

# YouTube player clients to walk, best-first. `None` means "let yt-dlp rotate
# clients itself", which exposes the audio-only DASH formats (itags 139/140/251)
# and is the cheapest option when it works. It often does not: those media URLs
# increasingly require a PO token and answer 403, while the `android` client
# offers only itag 18 (muxed 360p) but downloads without one. Which rung works
# depends on network, video and yt-dlp version, so the ladder is walked at run
# time rather than guessed -- see extraction.attempt_plan().
CLIENT_LADDER: list[str | None] = [
    None,
    "android",
    "web_safari,tv,ios,mweb",
]

# Google Drive file id of youtube_segments.csv (from the assignment brief).
SEGMENTS_CSV_FILE_ID = "1Ijs1IWypIY2GAjpNUKV2XNZY6o7dFvSm"
SEGMENTS_CSV_NAME = "youtube_segments.csv"

# ------------------------------------------------------------- diarization
# Step 2 roster. Both are pyannote.audio pipelines, so one code path covers
# both and the delta between them is the version-over-version comparison.
DIARIZATION_MODELS = {
    "community-1": "pyannote/speaker-diarization-community-1",
    "pyannote-3.1": "pyannote/speaker-diarization-3.1",
}

# Primary scoring, per the brief: "Do NOT ignore overlapping speech regions when
# computing metrics." Score everything, forgive nothing.
DER_COLLAR = 0.0
DER_SKIP_OVERLAP = False

# Secondary, reported alongside so the numbers can be compared with published
# results (which almost always use a collar and drop overlap). Never the headline.
DER_COLLAR_LENIENT = 0.25
DER_SKIP_OVERLAP_LENIENT = True


# ----------------------------------------------------------- scoring reference
# Stamped into every reference artifact. Bump it whenever reference.normalize_text
# or the turn-building rules change: it invalidates the (cheap) reference
# checkpoints without touching the (expensive) audio ones.
NORMALIZER_VERSION = "1.1.0"

# Column-name prefixes for anything derived from the ground truth. The brief is
# explicit that ground truth is only for scoring, so these must never reach a
# pipeline stage -- utils.assert_no_reference_fields() enforces it on DataFrames
# and data.ClipInput enforces it at the type level.
REFERENCE_FIELD_PREFIXES = ("gt_", "n_gt_", "ref_")

# Sources verified dead by an actual yt-dlp fetch (not just an oEmbed probe).
# Recorded so the report can distinguish "our pipeline failed" from "the source
# no longer exists" -- the ceiling for Step 1 is 99/100 rows, not 100.
KNOWN_UNAVAILABLE = {
    "GUVrL5ltiP4": "Video unavailable - deleted",
}

# Probing oEmbed returns HTTP 401 for 2HGP34TNvjg, but that only means embedding
# is disabled; yt-dlp downloads it normally. Kept as a note so nobody "fixes"
# it back into the unavailable list from an oEmbed check alone.
EMBED_DISABLED = {"2HGP34TNvjg"}


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


@dataclass
class Config:
    """Resolved filesystem layout for one pipeline run."""

    # Durable storage. On Colab this lives under the mounted Drive so it
    # survives a runtime reset; locally it is just a folder.
    root: Path = Path("/content/drive/MyDrive/sarvam_diarization")
    # Fast scratch on a real filesystem. ffmpeg must never write straight into
    # the Drive FUSE mount -- it is slow and gives no atomicity.
    work_dir: Path = Path("/content/work")

    sample_rate: int = SAMPLE_RATE
    dur_tol_sec: float = DUR_TOL_SEC
    max_attempts: int = MAX_ATTEMPTS
    # Pin a player client to skip the ladder search entirely, e.g. "android".
    force_client: str | None = None
    # HuggingFace token for the gated pyannote pipelines. Resolved from Colab
    # Secrets or the environment -- never written into the notebook, which is
    # published to a public repo.
    hf_token: str | None = None

    # Optional Netscape-format cookie jar for age-gated / bot-gated videos.
    cookies_file: Path | None = None

    @classmethod
    def create(cls, root: str | os.PathLike | None = None, **kwargs) -> "Config":
        """Build a Config, defaulting to Drive on Colab and ./pipeline_out locally."""
        if root is None:
            root = (
                Path("/content/drive/MyDrive/sarvam_diarization")
                if in_colab()
                else Path.cwd() / "pipeline_out"
            )
        work = kwargs.pop("work_dir", None)
        if work is None:
            work = Path("/content/work") if in_colab() else Path.cwd() / ".work"
        cfg = cls(root=Path(root), work_dir=Path(work), **kwargs)
        cfg.mkdirs()
        return cfg

    # ------------------------------------------------------------ derived paths
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio_16k"

    @property
    def meta_dir(self) -> Path:
        """Per-clip sidecars. A sidecar is the commit marker: it is written last."""
        return self.root / "meta"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def segments_csv(self) -> Path:
        return self.data_dir / SEGMENTS_CSV_NAME

    @property
    def failures_jsonl(self) -> Path:
        return self.logs_dir / "step1_failures.jsonl"

    @property
    def extraction_csv(self) -> Path:
        return self.results_dir / "step1_extraction.csv"

    @property
    def extraction_summary(self) -> Path:
        return self.results_dir / "step1_summary.json"

    @property
    def dotenv_path(self) -> Path:
        """Where .env is expected to live on Colab.

        The repo clone cannot carry it (.env is gitignored so secrets stay out of
        the public repo), so the Drive root is the location that persists.
        """
        return self.root / ".env"

    @property
    def dataset_profile(self) -> Path:
        return self.results_dir / "dataset_profile.json"

    # -- scoring reference (built from ground truth; never seen by the pipeline)
    @property
    def reference_dir(self) -> Path:
        return self.root / "reference"

    @property
    def rttm_dir(self) -> Path:
        return self.reference_dir / "rttm"

    # -- step 2: model hypotheses --------------------------------------------
    def hyp_dir(self, model: str, oracle: bool = False) -> Path:
        """Hypotheses for one model. `oracle` output is kept in a separate tree
        so a ground-truth-informed ablation can never be mistaken for a result."""
        root = self.root / ("hypotheses_oracle" if oracle else "hypotheses")
        return root / model

    def hyp_rttm_path(self, model: str, clip_id: str, oracle: bool = False) -> Path:
        return self.hyp_dir(model, oracle) / f"{clip_id}.rttm"

    def hyp_meta_path(self, model: str, clip_id: str, oracle: bool = False) -> Path:
        return self.hyp_dir(model, oracle) / f"{clip_id}.json"

    @property
    def step2_metrics_csv(self) -> Path:
        return self.results_dir / "step2_metrics.csv"

    @property
    def step2_summary(self) -> Path:
        return self.results_dir / "step2_summary.json"

    @property
    def step2_failures_jsonl(self) -> Path:
        return self.logs_dir / "step2_failures.jsonl"

    @property
    def ref_asr_dir(self) -> Path:
        return self.reference_dir / "asr"

    @property
    def reference_manifest(self) -> Path:
        return self.results_dir / "reference_manifest.csv"

    @property
    def normalization_report(self) -> Path:
        return self.results_dir / "normalization_report.json"

    def wav_path(self, clip_id: str) -> Path:
        return self.audio_dir / f"{clip_id}.wav"

    def meta_path(self, clip_id: str) -> Path:
        return self.meta_dir / f"{clip_id}.json"

    def rttm_path(self, clip_id: str) -> Path:
        return self.rttm_dir / f"{clip_id}.rttm"

    def ref_asr_path(self, clip_id: str) -> Path:
        return self.ref_asr_dir / f"{clip_id}.json"

    # ------------------------------------------------------------------ helpers
    def mkdirs(self) -> "Config":
        for d in (
            self.root,
            self.data_dir,
            self.audio_dir,
            self.rttm_dir,
            self.ref_asr_dir,
            self.meta_dir,
            self.logs_dir,
            self.results_dir,
            self.work_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def resolve_cookies(self) -> Path | None:
        """Use an explicit cookie jar, else pick one up from the Drive root."""
        if self.cookies_file and Path(self.cookies_file).exists():
            return Path(self.cookies_file)
        default = self.data_dir / "cookies.txt"
        return default if default.exists() else None

    def describe(self) -> str:
        return json.dumps(
            {k: str(v) for k, v in asdict(self).items()}, indent=2, ensure_ascii=False
        )


@dataclass
class StageFlags:
    """Which expensive stages actually run this session.

    Every stage checkpoints to Drive, so the normal workflow is to enable one
    stage, let it finish, flip it off, and move on. Re-running the notebook
    top-to-bottom then costs nothing but a few file reads.
    """

    run_extraction: bool = True      # Step 1
    build_reference: bool = True     # scoring reference (cheap, always safe)
    run_diarization: bool = False    # Step 2
    # DIAGNOSTIC ONLY: re-run diarization with the reference speaker count to
    # size what count estimation costs. This feeds ground truth to the model, so
    # it is off by default, written to a separate tree, and never a headline.
    run_oracle_count_ablation: bool = False
    run_asr: bool = False            # Step 3
    run_refinement: bool = False     # Step 4

    force_redo: bool = False         # ignore checkpoints for enabled stages
    retry_failed_only: bool = False  # only re-attempt non-permanent failures
    limit: int | None = None         # first N clips (smoke runs)
    only_clip_ids: list[str] | None = field(default=None)

    def describe(self) -> str:
        return json.dumps(asdict(self), indent=2)
