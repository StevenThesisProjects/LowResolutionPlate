"""Loads the frozen model once and reads plates from 1-5 frames.

The network is src/models/restran.py from the repository, unchanged, built with the
same arguments as tools/eval_lockbox.py (which scored the lock-box). Nothing here
changes what the model computes; the engine only adds what the app needs to show:
per-frame attention weights, per-frame sharpness, top-k strings and the triage flag.
"""
import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import settings as _settings  # noqa: F401  (puts the repository on sys.path)
from .decoding import Decoded, decode, layout_of
from .preprocess import prepare_track

try:
    from src.models.restran import ResTranOCR  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name not in ("src", "src.models", "src.models.restran"):
        raise
    raise ModuleNotFoundError(
        f"Model code not found: no src/models/restran.py under {_settings.REPO_ROOT}. "
        "Put the training repository restran/ next to masters_thesis_tienphat/, "
        "or set MFPLATE_REPO_ROOT=<path to restran>.", name=exc.name) from exc

NUM_CLASSES = 37


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ChecksumError(RuntimeError):
    pass


def build_model(checkpoint: Path, device: str = "cpu") -> ResTranOCR:
    model = ResTranOCR(num_classes=NUM_CLASSES)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@dataclass
class TrackOutput:
    log_probs: np.ndarray          # [T, C], float32
    frame_weights: List[float]     # attention weight of each frame, mean over positions
    frame_weights_pos: np.ndarray  # [F, W'] weight of each frame at each feature column
    sharpness: List[float]
    sizes: List[tuple]
    model_ms: float


class PlateReader:
    """Thread-safe wrapper around the frozen ResTran model."""

    def __init__(self, checkpoint: Path, expected_sha256: Optional[str] = None,
                 device: str = "cpu", threads: int = 0, grayscale: bool = True):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.checkpoint = checkpoint
        self.sha256 = sha256_file(checkpoint) if expected_sha256 else None
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ChecksumError(
                f"SHA-256 of {checkpoint.name} is {self.sha256}, expected {expected_sha256} "
                "(the file is not the frozen checkpoint)")
        if threads > 0:
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.grayscale = grayscale
        self.model = build_model(checkpoint, device)
        # One forward at a time: the hook below stores the fusion scores of the call in
        # progress, and CPU inference already uses every core through intra-op threads.
        self._lock = threading.Lock()
        self._scores = None
        self.model.fusion.score_net.register_forward_hook(self._keep_scores)

    def _keep_scores(self, module, inputs, output):
        self._scores = output.detach()

    @torch.no_grad()
    def forward(self, frames: List[np.ndarray]) -> TrackOutput:
        """Raw model output for RGB uint8 frames (any number from 1 to 5)."""
        prepared = prepare_track(frames, grayscale=self.grayscale)
        x = prepared.tensor.to(self.device)
        n = x.shape[1]
        with self._lock:
            t0 = time.perf_counter()
            lp = self.model(x)[0].float().cpu().numpy()   # .cpu() waits for the device
            scores = self._scores.float().cpu()
            model_ms = (time.perf_counter() - t0) * 1000.0
        # AttentionFusion: softmax over frames of the score map [B*F, 1, H', W']
        w = torch.softmax(scores.view(1, n, -1), dim=1)[0]                # [F, H'*W']
        return TrackOutput(
            log_probs=lp,
            frame_weights=[float(v) for v in w.mean(dim=1)],
            frame_weights_pos=w.cpu().numpy(),
            sharpness=prepared.sharpness,
            sizes=prepared.sizes,
            model_ms=model_ms,
        )

    def read(self, frames: List[np.ndarray], decoder: str = "greedy",
             top_k: int = 3) -> "Reading":
        out = self.forward(frames)
        t0 = time.perf_counter()
        dec = decode(out.log_probs, decoder=decoder, top_k=top_k)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        return Reading(output=out, decoded=dec, layout=layout_of(dec.text), decode_ms=decode_ms)


@dataclass
class Reading:
    output: TrackOutput
    decoded: Decoded
    layout: Optional[str]
    decode_ms: float
