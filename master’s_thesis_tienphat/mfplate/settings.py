"""Paths, limits and serving choices of MF-PlateReader.

Every value can be overridden with an environment variable (MFPLATE_*), so the same code
runs from the repository, from Docker and from the tests.
"""
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
THESIS_DIR = PACKAGE_DIR.parent


def _find_repo_root() -> Path:
    """The training repository `restran` (model code + checkpoints).

    Order: MFPLATE_REPO_ROOT; a sibling folder `restran/` (this thesis folder sits next to
    the repository); the parent folder (the older layout, and the Docker image, where the
    app lives in /app/masters_thesis_tienphat and the model code in /app).
    """
    env = os.environ.get("MFPLATE_REPO_ROOT")
    if env:
        return Path(env).resolve()
    for candidate in (THESIS_DIR.parent / "restran", THESIS_DIR.parent):
        if (candidate / "src" / "models" / "restran.py").is_file():
            return candidate.resolve()
    return (THESIS_DIR.parent / "restran").resolve()


REPO_ROOT = _find_repo_root()
ARTIFACTS_DIR = THESIS_DIR / "artifacts"
# The served checkpoint: a copy of the frozen file (same name, same SHA-256) kept here so the
# app does not read restran/results_new/. Not in git (135 MB); see model/README.md.
MODEL_DIR = THESIS_DIR / "model"

# The model code (src/, configs/) lives in the training repository and is imported, never
# copied, so the served network is the one that was trained and frozen. The weights come
# from MODEL_DIR and are checked against the frozen SHA-256 at start-up.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Plate syntax of the LRLPR-26 data (checked on all 20,000 training labels: only these
# two occur; there are no motorcycle plates).
PLATE_PATTERNS = {"LLLNLNN": "Mercosur", "LLLNNNN": "Brazilian"}
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Settings:
    serving_config: Path = field(default_factory=lambda: _env_path(
        "MFPLATE_SERVING_CONFIG", ARTIFACTS_DIR / "serving_config.json"))
    decoder_file: Path = field(default_factory=lambda: _env_path(
        "MFPLATE_DECODER_FILE", ARTIFACTS_DIR / "decoder_decision.json"))
    threshold_file: Path = field(default_factory=lambda: _env_path(
        "MFPLATE_THRESHOLD_FILE", ARTIFACTS_DIR / "threshold_calibration.json"))
    model_path_override: str = field(default_factory=lambda: os.environ.get("MFPLATE_MODEL", ""))
    db_path: Path = field(default_factory=lambda: _env_path(
        "MFPLATE_DB", THESIS_DIR / "var" / "mfplate.sqlite3"))
    device: str = field(default_factory=lambda: os.environ.get("MFPLATE_DEVICE", "cpu"))
    torch_threads: int = field(default_factory=lambda: int(os.environ.get("MFPLATE_THREADS", "0")))
    verify_checksum: bool = field(default_factory=lambda: os.environ.get(
        "MFPLATE_VERIFY_SHA256", "1") != "0")

    # Upload limits (roadmap: jpg/png, 1-5 images, at most 2 MB per image)
    max_frames: int = 5
    max_file_bytes: int = 2 * 1024 * 1024
    max_zip_bytes: int = 12 * 1024 * 1024
    max_side_px: int = 2048
    min_side_px: int = 4

    def load(self) -> "ServingSpec":
        base = _load_json(self.serving_config)
        if not base:
            raise FileNotFoundError(
                f"{self.serving_config} not found: run scripts/freeze_serving_model.py first")
        # serving_config.json records where the frozen file came from (restran/results_new/);
        # the app serves the copy with the same name in model/, unless MFPLATE_MODEL is set.
        model_path = Path(self.model_path_override).resolve() if self.model_path_override else (
            MODEL_DIR / Path(base["checkpoint"]["path"]).name)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"checkpoint not found: {model_path}. Copy it from "
                f"restran/{base['checkpoint']['path']} into model/ (see model/README.md), "
                "or set MFPLATE_MODEL=<path to the .pth file>")
        return ServingSpec(base=base, decoder=_load_json(self.decoder_file),
                           triage=_load_json(self.threshold_file), model_path=model_path)


@dataclass
class ServingSpec:
    """The frozen serving configuration, read from artifacts/ (written by scripts/)."""
    base: dict
    decoder: dict
    triage: dict
    model_path: Path

    @property
    def checkpoint_label(self) -> str:
        """Where the served weights are, as shown by GET /model (no absolute server path)."""
        try:
            return self.model_path.resolve().relative_to(THESIS_DIR).as_posix()
        except ValueError:
            return self.model_path.name

    @property
    def expected_sha256(self) -> str:
        return self.base["checkpoint"]["sha256"]

    @property
    def model_version(self) -> str:
        return self.base["model_version"]

    @property
    def default_decoder(self) -> str:
        # Until scripts/eval_decoders.py has decided, serve greedy: it is the decoder
        # behind every accuracy number in the paper and in the lock-box report.
        return self.decoder.get("default_decoder", "greedy")

    @property
    def score_name(self) -> str:
        return self.triage.get("score", "mean_char")

    def threshold(self, n_frames: int):
        """Confidence threshold for a track with n_frames frames (None = not calibrated)."""
        table = self.triage.get("thresholds_by_frames", {})
        value = table.get(str(n_frames))
        return None if value is None else float(value)

    def expected_accuracy(self, n_frames: int):
        table = self.base.get("expected_accuracy_by_frames", {})
        value = table.get(str(n_frames))
        return None if value is None else float(value)
