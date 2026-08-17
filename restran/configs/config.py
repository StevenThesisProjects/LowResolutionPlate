"""Configuration dataclass for the ResTranOCR training pipeline."""
from dataclasses import dataclass, field
from typing import Dict
import torch


@dataclass
class Config:
    """Training configuration with all hyperparameters."""

    # Experiment tracking
    EXPERIMENT_NAME: str = "restran"
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True  # Enable Spatial Transformer Network

    # Data paths (relative to the project root)
    DATA_ROOT: str = "dataset/data/train"
    TEST_DATA_ROOT: str = "dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public"
    VAL_SPLIT_FILE: str = "splits/val_tracks.json"

    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128

    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False

    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1

    DEVICE: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    OUTPUT_DIR: str = "results"

    # Logging: per-epoch metrics CSV is always written; per-batch losses are opt-in
    LOG_BATCH_LOSS: bool = False

    # Mixed precision: "fp16" (with GradScaler) or "bf16" (no scaler, no overflow)
    AMP_DTYPE: str = "fp16"
    # Skip the optimizer step when the pre-clip grad norm exceeds this (0 = disabled).
    # Guards against a single pathological batch destroying a healthy model.
    SKIP_GRAD_NORM: float = 0.0
    # Apply identical augmentation parameters to all 5 frames of a track. Independent
    # per-frame geometry breaks the spatial correspondence STN + AttentionFusion rely on.
    CONSISTENT_AUG: bool = False
    # ChannelShuffle permutes colour channels; plate background colour separates the
    # two layouts, so this destroys a real signal.
    CHANNEL_SHUFFLE: bool = True

    # Multi-frame SR front-end. Expects the pipeline to feed 16x64 so the module's
    # learned 2x upscale replaces the bilinear resize to 32x128.
    USE_SR: bool = False
    SR_FEATURES: int = 32
    SR_BLOCKS: int = 8
    # 2 = learned 2x upscale (needs 16x64 input); 1 = same-resolution cross-frame mixing
    SR_SCALE: int = 2
    # 'stacked' = channel-concat early fusion (the slide's design);
    # 'edvr' = pyramidal deformable alignment + temporal-spatial attention
    SR_ARCH: str = "stacked"

    # Aspect-preserving resize + padding instead of a stretch to the fixed canvas.
    # Scenario-A (aspect 2.14) and the test set (2.74) otherwise reach the model under
    # different geometric distortions: 1.87x versus 1.46x horizontal stretch.
    LETTERBOX: bool = False

    # Ablation: train/eval on a single frame (replicated 5x) to measure how much
    # the multi-frame fusion is actually worth.
    SINGLE_FRAME: bool = False

    # Pixel-wise SR loss against ECC-registered HR frames (0 = off), and the minimum
    # registration NCC a pair must reach before it is used as a target.
    SR_PIXEL_WEIGHT: float = 0.0
    SR_NCC_MIN: float = 0.8
    REGISTRATION_FILE: str = "splits/registration.npz"

    # Weight on the feature-distillation term (0 = off). Requires --teacher.
    DISTILL_WEIGHT: float = 0.0

    # Train AND validate on the undegraded HR frames. Not a submittable model —
    # it measures the accuracy ceiling that image enhancement could ever reach.
    USE_HR: bool = False

    # Initialise ResNet34 from ImageNet weights instead of from scratch.
    # Needs internet on first run to download the checkpoint.
    PRETRAINED_BACKBONE: bool = False

    # Sampling weight for Brazilian-layout plates (1.0 = no rebalancing). They are
    # 36.6% of training tracks but score ~29 points lower than Mercosur; ~1.73
    # equalises the two layouts in the training stream.
    LAYOUT_BALANCE: float = 1.0
    # Circuit breaker: once more than this fraction of an epoch's batches are skipped,
    # the model itself is broken, not one batch — every later epoch would update
    # nothing. Stop and keep the best checkpoint instead of burning GPU. (0 = off)
    SKIP_ABORT_RATIO: float = 0.2

    # Derived attributes (computed in __post_init__)
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self):
        """Compute derived attributes after initialization."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()
