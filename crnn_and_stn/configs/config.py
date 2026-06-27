"""Configuration for Baseline 1: Multi-Frame CRNN + STN + optional SR."""
from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class Config:
    EXPERIMENT_NAME: str = "crnn_stn_baseline"
    DATA_ROOT: str = "dataset/data/train"
    TEST_DATA_ROOT: str = "dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public"
    VAL_SPLIT_FILE: str = "dataset/val_tracks.json"
    OUTPUT_DIR: str = "results"

    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30
    SEED: int = 42
    NUM_WORKERS: int = 4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False
    AUGMENTATION_LEVEL: str = "full"

    USE_STN: bool = True
    USE_SR: bool = False
    SR_SCALE: int = 2
    SR_NUM_BLOCKS: int = 8
    SR_NUM_FEATURES: int = 64
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25

    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1


def get_default_config() -> Config:
    return Config()
