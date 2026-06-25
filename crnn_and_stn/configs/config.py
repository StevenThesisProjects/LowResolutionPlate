"""
Cấu hình cho Baseline 1: Multi-Frame CRNN + STN.

Trích xuất từ MultiFrame-LPR-main, chỉ giữ hyperparameters liên quan CRNN.
Tham chiếu report ICPR 2026 LRLPR — Trang 48 (Training Config).
"""
from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class Config:
    """Toàn bộ hyperparameters cho pipeline train/eval CRNN+STN."""

    # --- Thông tin experiment ---
    EXPERIMENT_NAME: str = "crnn_stn_baseline"

    # --- Đường dẫn dữ liệu (tương đối thư mục crnn_and_stn/) ---
    # Train: 20,000 tracks trong dataset/data/train/
    DATA_ROOT: str = "dataset/data/train"
    # Test public: 1,000 tracks — dùng khi --submission-mode
    TEST_DATA_ROOT: str = "dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public"
    # File JSON lưu danh sách track_id dùng cho validation (chỉ lấy từ Scenario-B)
    VAL_SPLIT_FILE: str = "dataset/val_tracks.json"
    # Thư mục lưu checkpoint và file submission
    OUTPUT_DIR: str = "results"

    # --- Tiền xử lý ảnh ---
    # Ảnh LR gốc ~46x19 px được resize về kích thước cố định trước khi vào CNN
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128

    # --- Bảng ký tự CTC (37 classes = 36 ký tự + 1 blank) ---
    # Index 0 = blank (CTC), index 1..36 = ký tự trong CHARS
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # --- Hyperparameters training (report Trang 48) ---
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30
    SEED: int = 42
    NUM_WORKERS: int = 4          # Giảm mặc định cho máy không có GPU mạnh
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9        # 90% Scenario-B → train, 10% → val
    USE_CUDNN_BENCHMARK: bool = False

    # --- Augmentation ---
    # "full" = đủ augmentation (report Trang 46), "light" = chỉ resize + normalize
    AUGMENTATION_LEVEL: str = "full"

    # --- Model: CRNN + STN ---
    USE_STN: bool = True            # Baseline 1 mặc định BẬT STN
    HIDDEN_SIZE: int = 256          # Hidden size BiLSTM
    RNN_DROPOUT: float = 0.25       # Dropout giữa 2 layer LSTM

    # --- Thiết bị ---
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Thuộc tính tự tính (không truyền khi khởi tạo) ---
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Tạo mapping ký tự ↔ index cho CTC loss và decoding."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 cho blank token


def get_default_config() -> Config:
    return Config()
