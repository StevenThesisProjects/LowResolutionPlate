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
    # Predict ONE affine per track (shared across its 5 frames) instead of one
    # per frame, so frames stay mutually aligned going into fusion.
    STN_SHARED: bool = False
    USE_SR: bool = True  # Enable lightweight super-resolution preprocessing
    USE_LEARNABLE_SR: bool = True  # Enable learnable SR block before STN
    SR_SCALE: int = 2  # Upsampling factor for very small crops in preprocessing
    USE_PRETRAINED_BACKBONE: bool = True  # ImageNet weights for ResNet34
    BACKBONE_NORM: str = "batch"  # "batch" or "group" (GroupNorm: fp16-stable)

    # Supervised SR module
    SR_LOSS_WEIGHT: float = 0.3      # λ_sr initial weight for auxiliary L1 loss
    SR_LOSS_WEIGHT_MIN: float = 0.05 # λ_sr minimum (decayed over training)
    SR_HIDDEN_CHANNELS: int = 64     # SuperResolutionBlock hidden channels
    SR_NUM_BLOCKS: int = 4           # Number of residual blocks in SR module
    # Learnable-SR upsampling factor. 1 = same-resolution deblur (old behavior);
    # 2 = genuine 2x super-resolution: SR block outputs 2x, supervised by real
    # HR at native (2x) resolution, and the backbone recognizes at 2x.
    SR_UPSAMPLE: int = 1
    # RCAN-style channel attention inside the SR residual blocks (reweights the
    # most informative channels for stroke reconstruction).
    SR_CHANNEL_ATTENTION: bool = False
    # Multi-frame SR: reconstruct each frame using a track-level aggregate of
    # all 5 frames instead of treating them independently. The sibling project
    # measured per-frame SR 77.18% -> multi-frame SR 79.78% (+26 tracks), the
    # only change there that cleared the +/-13-track noise band.
    SR_MULTI_FRAME: bool = False
    # HR privileged-information distillation: a detached forward on the clean HR
    # frames (downscaled to model input size) provides a "teacher" target; the
    # LR student is pulled toward it via KL. Exploits 100% HR coverage.
    USE_HR_DISTILL: bool = False
    HR_DISTILL_WEIGHT: float = 0.5

    # Fixed-length positional recognition head (multi-task with CTC).
    # Directly models "position i is char c" for the rigid 7-char plate layout;
    # its per-position posteriors are masked by PLATE_POS_CLASSES at decode and
    # fused with the CTC beam result.
    USE_POSITIONAL_HEAD: bool = False
    NUM_POSITIONS: int = 7
    POS_LOSS_WEIGHT: float = 0.5     # λ_pos weight for the positional CE loss
    POS_FUSE_AT_INFERENCE: bool = True  # fuse positional head with CTC at decode

    # Cross-frame attention fusion (vs quality-weighted-sum AttentionFusion)
    USE_CROSS_FRAME_FUSION: bool = False
    
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
    # Micro-batches accumulated before one optimizer step. Effective batch =
    # BATCH_SIZE x GRAD_ACCUM_STEPS. Larger effective batches markedly reduce
    # gradient noise, which was driving the repeated late-training collapses.
    GRAD_ACCUM_STEPS: int = 1
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
    # Stochastic Weight Averaging: average weights over the final epochs to land
    # in a flatter, better-generalizing minimum (fits the oscillating val curve).
    USE_SWA: bool = False
    SWA_START_FRAC: float = 0.6   # start averaging after this fraction of epochs
    SWA_BN_BATCHES: int = 200     # train batches used to recompute BN stats
    SWA_ACC_MARGIN: float = 3.0   # only average epochs whose val acc is within
                                  # this many points of the best (skip transient
                                  # divergence epochs that would poison the mean)
    SWA_LR: float = 5e-5          # constant LR held during the SWA phase — proper
                                  # SWA needs a steady LR (not cosine→0) so the
                                  # collected snapshots are diverse enough to help
    # Lookahead optimizer (Zhang et al. 2019) wrapping AdamW: "slow" weights are
    # nudged toward the "fast" weights every k steps, giving smoother, better-
    # generalizing updates at almost no cost. Produces a single model.
    USE_LOOKAHEAD: bool = False
    LOOKAHEAD_K: int = 5
    LOOKAHEAD_ALPHA: float = 0.5
    EARLY_STOP_PATIENCE: int = 7
    # Divergence guard: a batch is skipped when its loss is non-finite, or when
    # it exceeds max(MAX_TRAIN_LOSS, LOSS_SPIKE_MULT x running median). The
    # relative term keeps the guard valid across input sizes, since CTC loss
    # grows with the number of timesteps.
    MAX_TRAIN_LOSS: float = 50.0
    LOSS_SPIKE_MULT: float = 10.0
    # Raw-vs-EMA val gap (points) above which the raw weights are considered
    # collapsed and restored from EMA, with the LR scaled by COLLAPSE_LR_DECAY.
    COLLAPSE_ROLLBACK_MARGIN: float = 15.0
    COLLAPSE_LR_DECAY: float = 0.5
    LABEL_SMOOTHING: float = 0.0  # CTC label smoothing (uniform prior reg)

    # Domain adaptation: Scenario-A is stored as lossless PNG while the
    # val/test domain (Scenario-B) is JPEG. Inject JPEG artifacts into
    # PNG-domain training samples to close that gap.
    JPEG_DOMAIN_AUG: bool = False
    # Oversample the minority plate layout (Brazilian: 35% of train but 28
    # points worse on val) so both layouts are equally represented.
    LAYOUT_BALANCE: bool = False

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
    # Per-position character-class template (layout-agnostic union of the two
    # plate layouts present in the data — Brazilian 'LLLDDDD' and Mercosur
    # 'LLLDLDD'). 'L'=letter, 'D'=digit, 'LD'=either. 6/7 positions are fixed
    # across both layouts; only position 4 is free. Enforced during beam decode
    # to prune impossible letter/digit confusions (0/O, 1/I, 5/S, 8/B, 2/Z).
    # Set to None to disable.
    USE_POS_CLASS_DECODE: bool = True
    PLATE_POS_CLASSES: tuple = ('L', 'L', 'L', 'D', 'LD', 'D', 'D')

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
    # 2x SR front-end (PixelShuffle + residual) is far more sensitive to a hot
    # LR than the plain 1x model; 3e-4 + loose clip diverged at ~epoch 9. Use a
    # cooler LR, tight grad clip, longer warmup, and a low divergence-skip cap.
    config.LEARNING_RATE = 2e-4      # was 3e-4 — stabilize the 2x SR path
    config.BACKBONE_LR_RATIO = 0.7   # was 0.5 — pretrained backbone was undertrained
    config.NUM_WORKERS = 8
    # Reduced from 3e-4: the model was over-regularized (stuck ~77% vs author's
    # 78.70% with lighter reg). SR-2x + SWA now provide their own generalization.
    config.WEIGHT_DECAY = 2e-4
    config.GRAD_CLIP = 1.0           # was 3.0 — clamp gradient spikes
    # Divergence floor. The guard is now RELATIVE (max(floor, 10x running
    # median)), so the floor no longer has to clear the initial CTC loss — that
    # is handled by the warmup rule. 8.0 catches a real collapse (loss ~10)
    # while never firing during normal training (converged loss ~0.3-0.5).
    config.MAX_TRAIN_LOSS = 8.0
    config.WARMUP_EPOCHS = 3         # was 2 — gentler ramp into full LR
    # Keep fp16 (bf16 autocast produced NaN on Volta). The real anti-divergence
    # fix is the SR output clamp(±6) in SuperResolutionBlock, which bounds the
    # backbone's input so fp16 activations at 64x256 can't overflow — the prior
    # fp16 divergences all happened BEFORE that clamp existed.
    config.USE_AMP = True
    config.AMP_DTYPE = "float16"

    # Supervised SR: learnable block with L1 loss from HR targets.
    # NOTE: ALL 20k tracks have 5 matched HR frames (100% coverage), so the SR
    # block is fully supervised on every sample. Keep a steady (non-vanishing)
    # min weight so it acts as a front-end deblur regularizer throughout.
    config.USE_SR = False             # No OpenCV preprocessing SR
    config.USE_LEARNABLE_SR = True    # Enable supervised SR block
    # Constant lambda_SR = 0.1, matching the sibling project's best measured
    # setting (their lambda=0.5 ablation scored 3 tracks worse, i.e. no gain).
    # A decaying weight also weakens the SR anchor exactly when it is needed.
    config.SR_LOSS_WEIGHT = 0.1
    config.SR_LOSS_WEIGHT_MIN = 0.1
    config.SR_HIDDEN_CHANNELS = 64
    config.SR_NUM_BLOCKS = 4
    # Genuine 2x multi-frame super-resolution: recognize at 64x256 with real HR
    # supervision at native resolution (the paper's SR contribution). This ~4x's
    # backbone activations, so drop the batch size to fit 32 GB V100 memory.
    # SR x1, not x2. Our HR frames are only ~82x38 px, so a 64x256 target is
    # 81% bicubic-interpolated filler and the SR loss mostly teaches the block
    # to imitate interpolation. At x1 the 32x128 target is ~76% real pixels, so
    # the objective carries actual high-frequency information to recover.
    # (The sibling project reached the same conclusion for their larger ~115x42
    # HR, where x2 was still 55% interpolation.)
    config.SR_UPSAMPLE = 1
    # Multi-frame SR measured 76.18% vs 76.28% per-frame — no gain for ~extra
    # compute, so keep the simpler per-frame block while re-testing SR scale.
    config.SR_MULTI_FRAME = False
    config.BATCH_SIZE = 32            # was 64 — 2x SR quadruples activation memory
    config.GRAD_ACCUM_STEPS = 2       # effective batch 64 (matches the author's)
    # Channel attention + HR distill OFF to isolate the GroupNorm experiment
    # (both were confounded in the unstable 74.97% run). Kept as ablation flags.
    config.SR_CHANNEL_ATTENTION = False
    config.USE_HR_DISTILL = False
    config.HR_DISTILL_WEIGHT = 0.5

    # GroupNorm + train-from-scratch is the only configuration that trains
    # STABLY here. Measured: BatchNorm+pretrained collapsed at epoch 17-22 in
    # every run (3 rollbacks, mean raw-vs-EMA gap 13.5 points), whereas
    # GroupNorm+scratch ran 37 epochs with 1 rollback and a 6.7-point gap while
    # still improving. Converting a PRETRAINED BatchNorm backbone to GroupNorm
    # fails (conv weights are co-adapted to BN) — GroupNorm must be paired with
    # training from scratch, which is also what the original author does.
    config.BACKBONE_NORM = "group"
    config.USE_PRETRAINED_BACKBONE = False
    config.LEARNING_RATE = 2e-4

    # cudnn.benchmark picks algorithms non-deterministically; a sibling project
    # measured up to 6.5 points of run-to-run spread at a FIXED seed because of
    # it. With a 999-sample val set the statistical noise band is already
    # +/-1.3 points, so non-determinism on top makes small deltas unreadable.
    # Reproducibility matters more here than the few percent of speed.
    config.USE_CUDNN_BENCHMARK = False
    # AMP dtype is set above (fp16 for V100; the 4090 preset overrides to bf16).
    config.PERSISTENT_WORKERS = True
    config.PREFETCH_FACTOR = 2

    config.SCHEDULER_TYPE = "cosine"
    config.MIN_LR = 1e-6
    config.EARLY_STOP_PATIENCE = 12

    config.USE_EMA = True
    config.EMA_DECAY = 0.999
    # SWA + Lookahead REGRESSED (Lookahead −3% at epoch 21; SWA_LR=5e-5 too low).
    # Optimizer tricks are not the lever here — back to the clean 77.58 recipe.
    config.USE_SWA = False
    config.USE_LOOKAHEAD = False

    # Cross-frame fusion tested at 76.98% < 77.58% (old AttentionFusion) — it
    # did NOT help, so the default stays on the better weighted-sum fusion. The
    # module remains available (set USE_CROSS_FRAME_FUSION=True) as an ablation.
    config.USE_CROSS_FRAME_FUSION = False

    # Synthetic degraded copies are the main domain-randomization signal for
    # blurry plates. Reducing this to 0.4 removed most of that signal and let
    # the model overfit; restore near-author strength while keeping curriculum.
    # Measured: Scenario-A = PNG (lossless), Scenario-B = JPEG (val/test domain);
    # Brazilian = 55.2% vs Mercosur = 83.5% val accuracy at equal resolution.
    config.JPEG_DOMAIN_AUG = True
    # MEASURED net-negative on this val split: oversampling Brazilian moved it
    # 55.24% -> 59.52% (+4.28) but pushed Mercosur 83.52% -> 81.37% (-2.15).
    # Val is 79% Mercosur, so the weighted result was -0.80. Off by default;
    # re-enable with --layout-balance if the test split is more balanced.
    config.LAYOUT_BALANCE = False
    config.SYNTHETIC_SAMPLE_PROB = 1.0    # was 0.85 → author strength (more data)
    config.DEGRADATION_CURRICULUM = True
    # Lighter head/sequence regularization — the previous values (0.08/0.15/0.04)
    # capped the peak at ~77%. Move toward the author's lighter recipe.
    config.HEAD_DROPOUT = 0.03            # was 0.08
    config.TRANSFORMER_DROPOUT = 0.10     # was 0.15
    config.LABEL_SMOOTHING = 0.02         # was 0.04

    # Val decode: keep beam (needed for the layout pos-class prior) but DROP TTA
    # during training — TTA runs the model 3x AND validate_best_variant already
    # evaluates raw+EMA, so TTA tripled an already-slow val for ~+0.3%. Use full
    # TTA+beam only at final inference (eval_ablation / predict).
    config.USE_VAL_TTA = False         # was True — main cause of slow validation
    config.USE_BEAM_DECODE = True      # keep beam so pos-class decode applies
    # MEASURED on identical logits (sr2x_ar2, 999 val samples):
    #   greedy = beam16 = beam16+len7 = beam16+len7+posclass = 775/999 (77.58%)
    #   +TTA                                                 = 773/999 (77.38%)
    # The model already satisfies the 7-char / letter-digit layout on its own,
    # so the structural priors have nothing to correct (exactly +0.00%), and TTA
    # costs 3 forwards for no gain. Beam is kept only because it is free to
    # leave on and needed if a future model does violate the layout.
    config.USE_INFERENCE_TTA = False
    config.USE_INFERENCE_BEAM = True
    config.BEAM_WIDTH = 16             # was 10 — wider beam so a length-7
                                       # candidate survives for the length prior
    config.PLATE_LENGTH = 7            # fixed-length plate decoding prior

    # Positional head measured 76.68% vs 77.58% without it — it did NOT help, so
    # it is OFF by default (previously ON, which silently required passing
    # --no-positional-head on every run). Enable with --positional-head to
    # reproduce that ablation.
    config.USE_POSITIONAL_HEAD = False
    config.POS_LOSS_WEIGHT = 0.5
    config.POS_FUSE_AT_INFERENCE = True

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
    return config


def apply_rtx4090_preset(config: Config) -> Config:
    """RTX 4090 (Ada Lovelace) preset.

    Same recipe as the V100 preset, but with native **bfloat16** — Ada has
    hardware bf16 (Volta/V100 does not), so bf16 runs fast AND removes the fp16
    overflow that repeatedly caused divergence. GradScaler auto-disables for
    bf16. Batch is trimmed for the 4090's 24 GB (vs the V100's 32 GB).
    """
    apply_v100_preset(config)
    config.AMP_DTYPE = "bfloat16"   # Ada native bf16 → no fp16 overflow/divergence
    config.BATCH_SIZE = 24          # 24 GB VRAM (V100 had 32 GB → batch 32)
    # 64-core EPYC + 120 GB RAM: augmentation is CPU-bound, so feed the fast
    # 4090 with many workers + deep prefetch to avoid data-loading stalls.
    config.NUM_WORKERS = 16
    config.PERSISTENT_WORKERS = True
    config.PREFETCH_FACTOR = 4
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
