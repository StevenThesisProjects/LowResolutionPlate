"""Core building blocks for the CRNN + STN pipeline.

The backbone is upgraded toward an OCR-friendly ResBlock design:
- no BatchNorm inside residual blocks,
- small residual scaling for stable long runs,
- optional squeeze-and-excitation for ablation,
- a staged downsampling pattern that preserves width for CTC decoding.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel re-weighting used as an optional refinement module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The squeeze path summarizes each channel, then the excitation path
        # learns how much each channel should contribute back to the feature map.
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualBlock(nn.Module):
    """Residual block without BatchNorm for stable OCR feature learning.

    BatchNorm is often less suitable for OCR with small batches and repeated
    frame fusion, so this block uses plain convolutions + PReLU and a small
    residual scale to keep updates well-behaved.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 1,
        res_scale: float = 0.1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        mid_channels = channels * expansion
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.PReLU(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=3, padding=1, bias=True)
        self.res_scale = res_scale
        self.attn = SqueezeExcitation(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.attn(out)
        # Residual scaling lowers the risk of feature explosion when the model
        # is trained for longer schedules or with a stronger backbone preset.
        return residual + out * self.res_scale


class ResBackbone(nn.Module):
    """Residual CNN backbone for multi-frame OCR features.

    The stage layout is intentionally simple: a light stem followed by residual
    stages and a few downsampling points. The network keeps the height small and
    converts the width dimension into a sequence for the BiLSTM head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        stage_blocks: Sequence[int] = (2, 2, 2, 2, 2),
        stage_channels: Sequence[int] = (64, 128, 256, 256, 512),
        res_scale: float = 0.1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        if len(stage_blocks) != len(stage_channels):
            raise ValueError("stage_blocks and stage_channels must have the same length")

        # Stem keeps the first feature projection small and stable before the
        # deeper residual stages start refining OCR-specific details.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(base_channels),
        )

        # The width-preserving downsampling pattern keeps enough temporal steps
        # for CTC while still reducing height aggressively.
        downsample_strides = [(2, 2), (2, 2), (2, 2), (2, 1), (2, 1)]
        stages: list[nn.Sequential] = []
        in_ch = base_channels
        for idx, (num_blocks, out_ch) in enumerate(zip(stage_blocks, stage_channels)):
            stage_layers: list[nn.Module] = []
            if in_ch != out_ch:
                # A 1x1 projection lets each stage change its channel width
                # without adding extra spatial distortion.
                stage_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True))
                stage_layers.append(nn.PReLU(out_ch))
            stage_layers.extend(
                ResidualBlock(out_ch, res_scale=res_scale, use_se=use_se)
                for _ in range(num_blocks)
            )
            if idx < len(downsample_strides):
                stride = downsample_strides[idx]
                stage_layers.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)
                )
                stage_layers.append(nn.PReLU(out_ch))
            in_ch = out_ch
            stages.append(nn.Sequential(*stage_layers))

        self.stages = nn.ModuleList(stages)
        self.output_channels = int(stage_channels[-1])

        # Final refinement gives the head a cleaner and more expressive feature
        # map before it is converted into a sequence.
        self.output_refine = nn.Sequential(
            ResidualBlock(self.output_channels, res_scale=res_scale, use_se=use_se),
            nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(self.output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.output_refine(out)
        if out.size(2) != 1:
            # If the height is still larger than 1, adaptively pool it to a
            # single row so CTC can read the width dimension as a sequence.
            out = F.adaptive_avg_pool2d(out, (1, out.size(3)))
        return out


class AttentionFusion(nn.Module):
    """Fuse multiple frame features with learned frame attention."""

    def __init__(self, channels: int, hidden_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(channels // hidden_ratio, 16)
        self.scorer = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Accept a single feature map or a stack of frame features.
        if features.dim() == 4:
            features = features.unsqueeze(1)
        if features.dim() != 5:
            raise ValueError(f"Expected 4D/5D tensor, got shape {tuple(features.shape)}")

        batch_size, num_frames, channels, height, width = features.shape
        # Global pooling produces one descriptor per frame; the scorer then
        # learns which frames are most reliable for OCR decoding.
        pooled = features.mean(dim=(3, 4))
        scores = self.scorer(pooled).squeeze(-1)
        weights = torch.softmax(scores, dim=1).view(batch_size, num_frames, 1, 1, 1)
        return torch.sum(features * weights, dim=1)


class STNBlock(nn.Module):
    """Spatial transformer that predicts an affine warp for each frame."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        # Start from the identity transform so the STN behaves like a no-op at
        # the beginning of training and learns to correct distortion gradually.
        nn.init.zeros_(self.fc[-1].weight)
        self.fc[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.localization(x).flatten(1)
        theta = self.fc(features).view(-1, 2, 3)
        return theta


# Backwards-compatible alias used by older files in the project.
CNNBackbone = ResBackbone
