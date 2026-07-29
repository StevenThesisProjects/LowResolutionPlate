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
    # "none" giữ đúng backbone ResBlock cũ (mốc 76.68%); "group" thêm GroupNorm
    # thay cho BatchNorm đã bỏ — cần khi bật SR, nếu không gradient bùng nổ -> NaN.
    BACKBONE_NORM: str = "none"
    FRAME_DROPOUT: float = 0.05
    FUSION_DROPOUT: float = 0.05
    # STN localization pool. (1,1) là global-average — vector 64-D sau đó không
    # còn thông tin không gian nào, mà xoay/dịch lại đúng là đại lượng không
    # gian, nên STN gần như chỉ học được zoom. (4,8) giữ lại bố cục để head còn
    # suy ra được ma trận affine.
    STN_POOL: Tuple[int, int] = (4, 8)
    # Số bước thời gian cho CTC = IMG_WIDTH / WIDTH_DOWNSAMPLE.
    # 8 -> T=16 (2.3 bước/ký tự, khá chật); 4 -> T=32 mà không cần bật SR.
    WIDTH_DOWNSAMPLE: int = 8

    # SR knobs — module SR đa frame, chạy sau STN/DCN (fix cho lỗi PR #7:
    # stacked-input SR gộp 5 frame -> 1 -> nhân bản, chỉ học qua gradient CTC).
    USE_SR: bool = False
    # False = quay lại bản SR độc lập từng frame, cho Ablation 2.
    SR_MULTI_FRAME: bool = True
    SR_SCALE: int = 2
    SR_HIDDEN_CHANNELS: int = 32
    SR_NUM_BLOCKS: int = 4
    SR_RES_SCALE: float = 0.1
    # Công thức của issue #9: L_Total = L_CTC + λ_SR · L_SR, với
    # L_SR = (1/N)Σ‖I_SR − I_HR‖₁ + α · L_Perceptual, λ_SR trong khoảng 0.1–0.5.
    LAMBDA_SR: float = 0.1
    # α của L_Perceptual (VGG16 relu2_2). >0 sẽ tải VGG16 và tốn thêm VRAM.
    SR_PERCEPTUAL_WEIGHT: float = 0.0
    # Số hạng Sobel-edge KHÔNG có trong công thức issue #9 — đây là phần mở rộng
    # riêng của project. Để mặc định 0 nên L_SR đúng bằng công thức review; bật
    # lên qua --sr-edge-weight để đo nó như một ablation riêng thay vì mặc định.
    SR_EDGE_WEIGHT: float = 0.0

    # DCNv2 alignment (Step 2 pipeline issue #9) — căn chỉnh cục bộ giữa các frame
    # sau STN, bắt phần chuyển động mà 1 affine toàn cục của STN không biểu diễn được.
    USE_DCN: bool = False
    DCN_HIDDEN_CHANNELS: int = 32

    # Ablation 2: cơ chế gộp frame.
    FUSION_MODE: str = "attention"  # attention | avg | max
    # Ablation 3: số frame dùng mỗi track (5 = toàn bộ).
    NUM_FRAMES: int = 5
    # Nguyên nhân #4: degrade HR ở đúng cỡ LR gốc trước khi resize về IMG_SIZE.
    LR_DOMAIN_MATCH: bool = False

    # Decoding. Toàn bộ 20.000 nhãn dài đúng 7 ký tự và khớp 1 trong 2 layout
    # Brazil (Mercosur LLLNLNN, bản cũ LLLNNNN) => 6/7 vị trí bị khoá cứng lớp
    # chữ/số. Ở ~6.6 px/ký tự gần như mọi nhầm lẫn (0/O, 1/I, 8/B, 5/S, 2/Z) đều
    # vượt đúng ranh giới đó, nên ràng buộc lúc decode xoá được cả họ lỗi này mà
    # không đụng tới model. Mặc định vẫn "greedy" để số liệu cũ giữ nguyên ý
    # nghĩa; validate() luôn log cả hai cột.
    DECODE_MODE: str = "greedy"  # greedy | constrained
    PLATE_LAYOUTS: str = "LLLNLNN,LLLNNNN"
    BEAM_WIDTH: int = 16

    # Trung bình trượt trọng số — làm phẳng dao động ~1 điểm giữa các epoch liền
    # nhau, thứ mà tập val 999 sample không đủ sức phân giải.
    USE_EMA: bool = False
    EMA_DECAY: float = 0.999

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
