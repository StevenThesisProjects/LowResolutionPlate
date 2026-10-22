"""Loads the S1 checkpoint once and turns uploaded frames into plate readings.

Preprocessing is exactly the validation path of `MultiFrameDataset`
(cv2 decode -> RGB -> `get_val_transforms`), so API output matches the offline
`tools/predict_test.py` numbers for the same checkpoint.
"""

from __future__ import annotations

import base64
import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from configs.config import Config
from src.data.transforms import get_val_transforms
from src.models.crnn import MultiFrameCRNN
from src.utils.postprocess import PlateLayout, decode_batch
from tools.eval_decode import infer_architecture

from backend.app.core.config import Settings
from backend.app.core.errors import APIError
from backend.app.services.decoding import char_confidences, topk_constrained


@dataclass
class TrackResult:
    plate: str
    track_conf: float
    char_conf: list[float]
    suggestions: list[tuple[str, float]]
    sr_frames: list[np.ndarray] | None  # RGB uint8, one per *input* frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image(data: bytes, name: str) -> np.ndarray:
    """Bytes -> RGB uint8, or 422 if the file is not a readable image."""

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR) if array.size else None
    if image is None:
        raise APIError(422, "invalid_image", f"'{name}' không đọc được như ảnh JPG/PNG.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def png_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


class PlateRecognizer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config = Config()
        self.config.WIDTH_DOWNSAMPLE = settings.width_downsample
        self.layout = PlateLayout.from_spec(settings.plate_layouts)
        self.transform = get_val_transforms(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        self.device = torch.device(settings.device)
        # One forward at a time: the model is shared across request threads and
        # torch already parallelises a single forward over `torch_threads`.
        self._lock = threading.Lock()

        path = Path(settings.checkpoint_path)
        torch.set_num_threads(settings.torch_threads)
        state = torch.load(path, map_location="cpu")
        self.architecture = infer_architecture(state, self.config)
        self.model = MultiFrameCRNN(
            num_classes=self.config.NUM_CLASSES,
            hidden_size=self.config.HIDDEN_SIZE,
            backbone_channels=self.config.BACKBONE_CHANNELS,
            backbone_base_channels=self.config.BACKBONE_BASE_CHANNELS,
            backbone_blocks=self.config.BACKBONE_STAGE_BLOCKS,
            backbone_stage_channels=self.config.BACKBONE_STAGE_CHANNELS,
            residual_scale=self.config.BACKBONE_RES_SCALE,
            fusion_mode=self.config.FUSION_MODE,
            width_downsample=self.config.WIDTH_DOWNSAMPLE,
            **self.architecture,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.checkpoint_name = path.name
        self.checkpoint_sha256 = _sha256(path)
        self.version = f"{path.stem}@{self.checkpoint_sha256[:12]}"
        self.parameters = sum(p.numel() for p in self.model.parameters())

    # ── preprocessing ──────────────────────────────────────────────────────

    def to_tensor(self, frames: list[np.ndarray]) -> torch.Tensor:
        """1..max_frames RGB frames -> `[NUM_FRAMES, 3, H, W]`.

        The model was trained on 5 frames; shorter tracks repeat the last frame
        (roadmap §2 flow, step 1).
        """

        tensors = [self.transform(image=frame)["image"] for frame in frames]
        while len(tensors) < self.config.NUM_FRAMES:
            tensors.append(tensors[-1])
        return torch.stack(tensors[: self.config.NUM_FRAMES], dim=0)

    # ── inference ──────────────────────────────────────────────────────────

    def predict(
        self,
        tracks: list[list[np.ndarray]],
        decode: str,
        include_sr: bool = False,
        top_k: int = 5,
    ) -> tuple[list[TrackResult], float]:
        """Run a batch of tracks. Returns results and the wall time in ms."""

        start = time.perf_counter()
        batch = torch.stack([self.to_tensor(frames) for frames in tracks]).to(self.device)
        want_sr = include_sr and self.architecture.get("use_sr", False)

        with self._lock, torch.inference_mode():
            if want_sr:
                log_probs, sr_output, _, _ = self.model(batch, return_sr=True)
            else:
                log_probs = self.model(batch)

        decoded = decode_batch(
            log_probs, self.config.IDX2CHAR, mode=decode,
            layout=self.layout, beam_width=self.settings.beam_width,
        )
        lp_all = log_probs.float().cpu().numpy()

        sr_images = None
        if want_sr:
            # SR lives in the normalised (-1..1) space of the input.
            sr = ((sr_output.clamp(-1, 1) * 0.5 + 0.5) * 255).round().byte()
            sr = sr.permute(0, 2, 3, 1).cpu().numpy()
            frames = self.config.NUM_FRAMES
            sr_images = [sr[i * frames:(i + 1) * frames] for i in range(len(tracks))]

        results = []
        for i, (text, conf) in enumerate(decoded):
            results.append(TrackResult(
                plate=text,
                track_conf=float(conf),
                char_conf=char_confidences(lp_all[i], text, self.config.CHAR2IDX),
                suggestions=topk_constrained(
                    lp_all[i], self.config.IDX2CHAR, self.layout,
                    beam_width=max(self.settings.beam_width, top_k), top_k=top_k,
                ),
                # Only the uploaded frames, not the padded repeats.
                sr_frames=list(sr_images[i][: len(tracks[i])]) if sr_images is not None else None,
            ))
        return results, (time.perf_counter() - start) * 1000


class ModelRegistry:
    """Holds the recognizer; lets the app start (and report `degraded`) without one."""

    def __init__(self) -> None:
        self.recognizer: PlateRecognizer | None = None
        self.load_error: str | None = None

    def load(self, settings: Settings) -> None:
        try:
            self.recognizer = PlateRecognizer(settings)
        except Exception as exc:  # noqa: BLE001 — surfaced via /health
            self.recognizer = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def get(self) -> PlateRecognizer:
        if self.recognizer is None:
            raise APIError(503, "model_unavailable", f"Model chưa sẵn sàng: {self.load_error}")
        return self.recognizer


registry = ModelRegistry()
