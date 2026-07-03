"""Configuration for the upgraded CRNN + STN pipeline.

The config keeps the original project shape, but exposes the backbone and
training knobs needed for a stronger OCR recipe: deeper ResBlock stages,
residual scaling, optional SE, warmup/min-LR, and gradient accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch


@dataclass
class Config:
    """Experiment defaults used by `train.py` and `Trainer`."""

    # Experiment metadata and filesystem locations.
    EXPERIMENT_NAME: str = "crnn_stn_resblock"
    DATA_ROOT: str = "dataset/data/train"
    TEST_DATA_ROOT: str = "dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public"
    VAL_SPLIT_FILE: str = "dataset/val_tracks.json"
    OUTPUT_DIR: str = "results"

    # Input resolution used across the whole OCR pipeline.
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Training defaults are intentionally conservative so the stronger model
    # can run for longer without exploding gradients or overfitting too fast.
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 8e-4
    EPOCHS: int = 80
    SEED: int = 42
    NUM_WORKERS: int = 4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 2.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = True
    AUGMENTATION_LEVEL: str = "full"

    # Model knobs.
    USE_STN: bool = True
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25
    BACKBONE_BASE_CHANNELS: int = 64
    BACKBONE_CHANNELS: int = 512
    BACKBONE_STAGE_BLOCKS: Tuple[int, ...] = (2, 2, 2, 2, 2)
    BACKBONE_STAGE_CHANNELS: Tuple[int, ...] = (64, 128, 256, 256, 512)
    BACKBONE_USE_SE: bool = True
    BACKBONE_RES_SCALE: float = 0.1
    FRAME_DROPOUT: float = 0.05
    FUSION_DROPOUT: float = 0.05
    LABEL_SMOOTHING: float = 0.0

    # Stability helpers for long training runs.
    WARMUP_RATIO: float = 0.05
    MIN_LR_RATIO: float = 0.05
    EARLY_STOPPING_PATIENCE: int = 18
    GRAD_ACCUM_STEPS: int = 1
    USE_AMP: bool = True
    CLIP_INPUT_TAILS: bool = False

    # Device is computed dynamically so the same config works on CPU/GPU.
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Derived CTC metadata.
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # CTC uses index 0 as the blank token, so characters start from 1.
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1


def get_default_config() -> Config:
    """Factory helper kept for backwards compatibility with older scripts."""

    return Config()
