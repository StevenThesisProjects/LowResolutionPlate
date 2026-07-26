"""Multi-frame CRNN with STN and a residual backbone.

This module keeps the original OCR flow intact:
5 frames -> STN -> shared CNN backbone -> attention fusion -> BiLSTM -> CTC.
The main change is that the backbone is now a deeper ResBlock-style encoder
that is more stable for low-resolution plates and longer training schedules.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import AttentionFusion, CNNBackbone, FrameSR, STNBlock


def _normalize_stage_channels(
    backbone_channels: int,
    stage_channels: Optional[Sequence[int]],
) -> tuple[int, ...]:
    """Normalize channel presets so the model can accept CLI overrides cleanly."""

    if stage_channels is None:
        return (64, 128, 256, 256, backbone_channels)
    channels = tuple(int(v) for v in stage_channels)
    if len(channels) == 0:
        raise ValueError("stage_channels must not be empty")
    return channels


def _normalize_stage_blocks(stage_blocks: Optional[Sequence[int]]) -> tuple[int, ...]:
    """Normalize block presets; a tuple is easier to serialize and reuse."""

    if stage_blocks is None:
        return (2, 2, 2, 2, 2)
    blocks = tuple(int(v) for v in stage_blocks)
    if len(blocks) == 0:
        raise ValueError("stage_blocks must not be empty")
    return blocks


class MultiFrameCRNN(nn.Module):
    """Multi-frame CRNN with optional STN alignment."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        rnn_dropout: float = 0.25,
        use_stn: bool = True,
        backbone_channels: int = 512,
        backbone_base_channels: int = 64,
        backbone_blocks: Optional[Sequence[int]] = None,
        backbone_stage_channels: Optional[Sequence[int]] = None,
        use_se: bool = False,
        residual_scale: float = 0.1,
        frame_dropout: float = 0.05,
        fusion_dropout: float = 0.05,
        use_sr: bool = False,
        sr_scale: int = 2,
        sr_hidden_channels: int = 32,
        sr_num_blocks: int = 4,
        sr_res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_stn = use_stn
        self.frame_dropout = frame_dropout
        self.use_sr = use_sr

        stage_blocks = _normalize_stage_blocks(backbone_blocks)
        stage_channels = _normalize_stage_channels(backbone_channels, backbone_stage_channels)
        self.cnn_channels = stage_channels[-1]

        if self.use_stn:
            # STN performs a light geometric normalization per frame before the
            # shared backbone extracts OCR features.
            self.stn = STNBlock(in_channels=3)

        if self.use_sr:
            # SR runs after STN (aligned frames are easier to upscale cleanly)
            # and per-frame on the flattened B*F batch, so it never collapses
            # the 5 frames into one image the way the old stacked-input SR did.
            self.sr = FrameSR(
                in_channels=3,
                hidden_channels=sr_hidden_channels,
                num_blocks=sr_num_blocks,
                scale=sr_scale,
                res_scale=sr_res_scale,
            )

        self.backbone = CNNBackbone(
            in_channels=3,
            base_channels=backbone_base_channels,
            stage_blocks=stage_blocks,
            stage_channels=stage_channels,
            res_scale=residual_scale,
            use_se=use_se,
        )

        self.fusion = AttentionFusion(channels=self.cnn_channels, dropout=fusion_dropout)
        self.rnn = nn.LSTM(
            input_size=self.cnn_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(rnn_dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def _apply_stn(self, frames: torch.Tensor) -> torch.Tensor:
        theta = self.stn(frames)
        grid = F.affine_grid(theta, frames.size(), align_corners=False)
        return F.grid_sample(frames, grid, align_corners=False, padding_mode="border")

    def forward(self, x: torch.Tensor, return_sr: bool = False):
        """Return log-probabilities for CTC loss.

        Args:
            x: Tensor with shape [B, Frames, C, H, W]
            return_sr: also return the per-frame SR output (flattened
                [B*F, C, H*scale, W*scale]) so the trainer can compute an
                auxiliary pixel-level loss against HR ground truth.
        """

        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B, F, C, H, W], got shape {tuple(x.shape)}")

        batch_size, num_frames, channels, height, width = x.size()
        x_flat = x.view(batch_size * num_frames, channels, height, width)

        if self.use_stn:
            x_flat = self._apply_stn(x_flat)

        sr_output = None
        if self.use_sr:
            x_flat = self.sr(x_flat)
            if return_sr:
                sr_output = x_flat

        features = self.backbone(x_flat)
        features = features.view(batch_size, num_frames, self.cnn_channels, features.size(2), features.size(3))
        if self.frame_dropout > 0 and self.training:
            # Drop entire frame features with a small probability so the model
            # does not over-rely on a single clear frame in the 5-frame stack.
            frame_mask = torch.rand(batch_size, num_frames, 1, 1, 1, device=x.device)
            keep_mask = (frame_mask > self.frame_dropout).float()
            features = features * keep_mask + features.mean(dim=1, keepdim=True) * (1.0 - keep_mask)

        fused = self.fusion(features)
        # The fused feature map has height 1, so width becomes the sequence axis.
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(seq_input)
        logits = self.head(rnn_out)
        log_probs = logits.log_softmax(2)
        if return_sr:
            return log_probs, sr_output
        return log_probs
