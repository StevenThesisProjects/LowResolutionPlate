"""
Baseline 1: Multi-Frame CRNN + STN.

Pipeline đầy đủ (report Trang 39):
  5 LR frames → STN → CNN (weight sharing) → Attention Fusion → BiLSTM → FC → CTC

Nguồn gốc: MultiFrame-LPR-main/src/models/crnn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import AttentionFusion, CNNBackbone, STNBlock


class MultiFrameCRNN(nn.Module):
    """
    Multi-Frame CRNN với STN tùy chọn.

    Input : [Batch, 5, 3, 32, 128]  — 5 frame RGB đã resize
    Output: [Batch, SeqLen, NumClasses] log-probabilities cho CTC loss
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        rnn_dropout: float = 0.25,
        use_stn: bool = True,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn

        # Bước 1: STN — căn chỉnh hình học từng frame (nếu bật)
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # Bước 2: CNN backbone — trích feature (weight sharing qua 5 frame)
        self.backbone = CNNBackbone(out_channels=self.cnn_channels)

        # Bước 3: Attention Fusion — gộp 5 feature map thành 1
        self.fusion = AttentionFusion(channels=self.cnn_channels)

        # Bước 4: BiLSTM — mô hình hóa ngữ cảnh theo chiều ngang biển số
        self.rnn = nn.LSTM(
            input_size=self.cnn_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout,
        )

        # Bước 5: FC head — chiếu sang 37 classes (36 ký tự + blank)
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, Frames=5, C=3, H, W]
        Returns:
            log_probs: [B, SeqLen, NumClasses]
        """
        b, f, c, h, w = x.size()

        # Gộp batch và frame để xử lý song song qua STN + CNN
        x_flat = x.view(b * f, c, h, w)  # [B*5, 3, H, W]

        # --- STN: warp từng frame ---
        if self.use_stn:
            theta = self.stn(x_flat)                              # [B*5, 2, 3]
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat

        # --- CNN: trích feature (cùng weight cho cả 5 frame) ---
        features = self.backbone(x_aligned)   # [B*5, 512, 1, W']

        # --- Attention Fusion: 5 frame → 1 feature map ---
        fused = self.fusion(features)         # [B, 512, 1, W']

        # --- BiLSTM: [B, C, 1, W'] → [B, W', C] ---
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(seq_input)      # [B, W', hidden*2]

        # --- FC + log_softmax cho CTC ---
        out = self.head(rnn_out)              # [B, W', num_classes]
        return out.log_softmax(2)
