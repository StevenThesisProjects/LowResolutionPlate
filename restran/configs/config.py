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

    # Mixed precision: "fp16" (with GradScaler), "bf16" (no scaler) or "fp32" (AMP off)
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

    # Exponential moving average of the weights (0 = off). Measured motivation: over
    # the last 6 epochs, when the LR is ~0 and the weights barely move, accuracy still
    # wobbles by 0.4-0.8 points and the reported max sits ~0.7 above the mean. Averaging
    # those near-identical weights recovers that spread deterministically.
    EMA_DECAY: float = 0.0

    # One shared STN affine per track instead of one per frame.
    STN_SHARED: bool = False
    # Keep height 2 into AttentionFusion so its scoring is genuinely 2D, not
    # horizontal-only; the height is pooled away after fusion instead of before.
    FUSION_2D: bool = False
    # Drop the HR-degraded "synthetic LR" half of the training samples.
    NO_SYNTHETIC: bool = False
    # Feed each frame's log Laplacian variance into AttentionFusion as a bias.
    SHARPNESS_PRIOR: bool = False

    # Data enrichment: confidence-filtered test tracks folded into training, and a
    # harsher HR->LR degradation that matches Scenario-B's JPEG artefacts.
    PSEUDO_LABELS: str = ""
    PSEUDO_ROOT: str = ""
    PSEUDO_MIN_CONF: float = 0.99
    DEGRADE_STRENGTH: str = "normal"
    # Blur/noise/JPEG applied to EVERY training image, real LR included.
    AUG_DEGRADE: str = "none"

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

    # Synthetic branch recipe: 'legacy' (degrade HR at its own size, then augment) or
    # 'native' (augment HR on the target canvas, shrink it to the paired real LR's
    # size and degrade there). Measured: the legacy synthetic copy reads at 98.3% vs
    # 78.3% for real LR; native brings it to 93.2%.
    SYN_PIPELINE: str = "legacy"
    # L1 weight between the SR output and the HR target of native synthetic samples
    # (pixel-aligned by construction). 0 = off. Needs USE_SR and SYN_PIPELINE='native'.
    SR_SYN_WEIGHT: float = 0.0
    # Auxiliary CTC on each frame read on its own by the shared Transformer + head
    # (0 = off). Deep supervision for the per-frame features that fusion combines.
    AUX_FRAME_CTC: float = 0.0

    # LR schedule: 'onecycle' (original) or 'cosine' (linear warmup over WARMUP_EPOCHS,
    # cosine decay to MIN_LR_RATIO * peak). PATIENCE > 0 stops after that many epochs
    # without a new best val accuracy.
    SCHEDULER: str = "onecycle"
    WARMUP_EPOCHS: float = 3.0
    MIN_LR_RATIO: float = 0.05
    PATIENCE: int = 0

    # Also report val accuracy on the track ids in this JSON ("" = off). Point it at
    # splits/val_tracks.json to read the 999-track split inside the 1999-track val.
    VAL_SUBSET_FILE: str = ""
    # CoarseDropout holes sized relative to the input height (1/8-1/4) instead of a
    # fixed 4-8 px. Same at 32x128; keeps 16x64 SR runs from losing 4x more area.
    RELATIVE_DROPOUT: bool = False

    # Multi-frame fusion: 'attention' (original per-frame quality weights) or
    # 'temporal' (cross-frame Transformer over all frame x position tokens, then a
    # per-position softmax over frames; starts as an exact mean over frames).
    FUSION: str = "attention"
    FUSION_LAYERS: int = 2

    # Training-only regularisers for multi-frame fusion (0 = off): replace a frame's
    # features by the mean of the other frames, and erase one band per frame with an
    # occlusion drawn independently for every frame.
    FRAME_DROPOUT: float = 0.0
    FRAME_MASK: float = 0.0
    # Luma-only input (train/val/test) — Scenario-B frames are 4:2:0 JPEGs.
    GRAYSCALE: bool = False
    # Geometric augmentation with bicubic resampling + replicated borders.
    CLEAN_GEO_AUG: bool = False

    # Lock-box: JSON list of track ids excluded from training and validation ("" = off),
    # and how many final-epoch checkpoints to keep (0 = only the best one).
    EXCLUDE_TRACKS_FILE: str = ""
    SAVE_LAST_K: int = 0

    # Experiment N1: how the per-frame masks of one track relate across frames
    # ('independent' = frozen configuration, 'tube', 'complementary'), and band
    # orientation ('mixed' = vertical or horizontal, 'vertical' only).
    FRAME_MASK_MODE: str = "independent"
    FRAME_MASK_ORIENT: str = "mixed"
    # Experiment E4: replace the learned STN by one fixed affine "a,b,tx,c,d,ty" ("" = off).
    FIXED_STN: str = ""

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
