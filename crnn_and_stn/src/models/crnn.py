"""Multi-frame CRNN with optional STN and stacked-input SR."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components import AttentionFusion, CNNBackbone, STNBlock, StackedSRNet


class MultiFrameCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        rnn_dropout: float = 0.25,
        use_stn: bool = True,
        use_sr: bool = False,
        sr_scale: int = 2,
        num_frames: int = 5,
    ) -> None:
        super().__init__()
        self.use_stn = use_stn
        self.use_sr = use_sr
        self.num_frames = num_frames

        self.sr = StackedSRNet(num_frames=num_frames, target_size=(32, 128)) if use_sr else None
        self.stn = STNBlock(in_channels=3) if use_stn else nn.Identity()
        self.backbone = CNNBackbone(in_channels=3, out_channels=128)
        self.fusion = AttentionFusion(channels=128)

        self.rnn = nn.LSTM(
            input_size=128 * 8,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, C, H, W]
        if self.sr is not None:
            x = self.sr(x)
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)

        feats = []
        for i in range(x.size(1)):
            frame = x[:, i]
            frame = self.stn(frame)
            feats.append(self.backbone(frame))

        feats = torch.stack(feats, dim=1)
        fused = self.fusion(feats)
        b, c, h, w = fused.shape
        seq = fused.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        seq, _ = self.rnn(seq)
        logits = self.classifier(seq)
        # CTC expects log-probabilities over classes at each time step.
        return torch.log_softmax(logits, dim=2)
