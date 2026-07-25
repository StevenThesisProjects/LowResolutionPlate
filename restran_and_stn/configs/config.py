"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict
import torch


@dataclass
class Config:
    """Training configuration with all hyperparameters."""
    
    # Experiment tracking
    EXPERIMENT_NAME: str = "restran34_with_stn"
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True  # Enable Spatial Transformer Network
    USE_SR: bool = True  # Enable lightweight super-resolution preprocessing
    USE_LEARNABLE_SR: bool = True  # Enable learnable SR block before STN
    SR_SCALE: int = 2  # Upsampling factor for very small crops in preprocessing
    USE_PRETRAINED_BACKBONE: bool = True  # ImageNet weights for ResNet34

    # Supervised SR module
    SR_LOSS_WEIGHT: float = 0.3      # λ_sr initial weight for auxiliary L1 loss
    SR_LOSS_WEIGHT_MIN: float = 0.05 # λ_sr minimum (decayed over training)
    SR_HIDDEN_CHANNELS: int = 64     # SuperResolutionBlock hidden channels
    SR_NUM_BLOCKS: int = 4           # Number of residual blocks in SR module
    
    # Data paths
    DATA_ROOT: str = "dataset/data/train"
    TEST_DATA_ROOT: str = "dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public"
    VAL_SPLIT_FILE: str = "dataset/data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"
    
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    
    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 3.5e-4
    BACKBONE_LR_RATIO: float = 0.5
    EPOCHS: int = 1
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 1.5e-4
    GRAD_CLIP: float = 3.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False

    # Mixed precision (V100: use float16)
    USE_AMP: bool = True
    AMP_DTYPE: str = "float16"

    # DataLoader throughput
    PERSISTENT_WORKERS: bool = False
    PREFETCH_FACTOR: int = 2

    # Scheduler & stability
    SCHEDULER_TYPE: str = "cosine"  # "cosine", "cosine_restarts", or "onecycle"
    WARMUP_EPOCHS: int = 2
    ONECYCLE_PCT_START: float = 0.2
    MIN_LR: float = 1e-6
    COSINE_T0: int = 15       # CosineAnnealingWarmRestarts: first cycle length
    COSINE_T_MULT: int = 2    # CosineAnnealingWarmRestarts: cycle multiplier
    USE_EMA: bool = True
    EMA_DECAY: float = 0.999
    EARLY_STOP_PATIENCE: int = 7
    MAX_TRAIN_LOSS: float = 50.0
    LABEL_SMOOTHING: float = 0.0  # CTC label smoothing (uniform prior reg)

    # Synthetic data
    USE_SYNTHETIC_LR: bool = True
    SYNTHETIC_SAMPLE_PROB: float = 0.5
    SYNTHETIC_FROM_LR: bool = True
    DEGRADATION_CURRICULUM: bool = True

    # Inference boosts (use only at predict/submission — not during training val)
    USE_VAL_TTA: bool = False
    USE_BEAM_DECODE: bool = False
    USE_INFERENCE_TTA: bool = True
    USE_INFERENCE_BEAM: bool = False
    BEAM_WIDTH: int = 10
    # Fixed-length plate prior: every plate is exactly 7 chars in this dataset.
    # 0 disables the length constraint during beam decoding.
    PLATE_LENGTH: int = 7

    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1
    HEAD_DROPOUT: float = 0.05
    
    DEVICE: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    OUTPUT_DIR: str = "results"
    
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


def apply_v100_preset(config: Config) -> Config:
    """V100 preset with Supervised SR module.

    Uses best regularization from Run 1 (75% val) plus new supervised SR
    with auxiliary L1 loss from HR targets.
    """
    config.BATCH_SIZE = 64
    config.LEARNING_RATE = 3e-4
    config.BACKBONE_LR_RATIO = 0.7   # was 0.5 — pretrained backbone was undertrained
    config.NUM_WORKERS = 8
    config.WEIGHT_DECAY = 3e-4
    config.GRAD_CLIP = 3.0

    # Supervised SR: learnable block with L1 loss from HR targets.
    # NOTE: only ~52% of tracks have matched HR, so the SR signal covers half
    # the data. Keep a steady (non-vanishing) min weight so it acts as a
    # front-end deblur regularizer throughout training rather than fading out.
    config.USE_SR = False             # No OpenCV preprocessing SR
    config.USE_LEARNABLE_SR = True    # Enable supervised SR block
    config.SR_LOSS_WEIGHT = 0.3
    config.SR_LOSS_WEIGHT_MIN = 0.1   # was 0.05
    config.SR_HIDDEN_CHANNELS = 64
    config.SR_NUM_BLOCKS = 4

    config.USE_CUDNN_BENCHMARK = True
    config.USE_AMP = True
    config.AMP_DTYPE = "float16"
    config.PERSISTENT_WORKERS = True
    config.PREFETCH_FACTOR = 2

    config.SCHEDULER_TYPE = "cosine"
    config.WARMUP_EPOCHS = 2
    config.MIN_LR = 1e-6
    config.EARLY_STOP_PATIENCE = 12

    config.USE_EMA = True
    config.EMA_DECAY = 0.999

    # Synthetic degraded copies are the main domain-randomization signal for
    # blurry plates. Reducing this to 0.4 removed most of that signal and let
    # the model overfit; restore near-author strength while keeping curriculum.
    config.SYNTHETIC_SAMPLE_PROB = 0.85   # was 0.4
    config.DEGRADATION_CURRICULUM = True
    config.HEAD_DROPOUT = 0.08
    config.TRANSFORMER_DROPOUT = 0.15
    config.LABEL_SMOOTHING = 0.04

    config.USE_VAL_TTA = True          # Enable Test-Time Augmentation for Val
    config.USE_BEAM_DECODE = True      # Enable Beam Search for Val
    config.USE_INFERENCE_TTA = True
    config.USE_INFERENCE_BEAM = True
    config.BEAM_WIDTH = 16             # was 10 — wider beam so a length-7
                                       # candidate survives for the length prior
    config.PLATE_LENGTH = 7            # fixed-length plate decoding prior

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
    return config


def apply_author_preset(config: Config) -> Config:
    """Align hyperparameters with the original ResTranOCR + STN recipe."""
    config.SCHEDULER_TYPE = "onecycle"
    config.LEARNING_RATE = 4e-4
    config.BACKBONE_LR_RATIO = 1.0
    config.WEIGHT_DECAY = 1e-4
    config.GRAD_CLIP = 5.0
    config.ONECYCLE_PCT_START = 0.2
    config.SYNTHETIC_SAMPLE_PROB = 1.0
    config.DEGRADATION_CURRICULUM = False
    config.HEAD_DROPOUT = 0.0
    config.TRANSFORMER_DROPOUT = 0.1
    config.EARLY_STOP_PATIENCE = 7
    return config
