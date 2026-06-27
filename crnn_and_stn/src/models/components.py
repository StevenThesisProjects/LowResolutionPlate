"""Model components for Baseline 1: CRNN + STN + stacked-input SR."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class STNBlock(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 8 * 32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )
        nn.init.zeros_(self.fc_loc[2].weight)
        self.fc_loc[2].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class AttentionFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, F, C, H, W]
        b, f, c, h, w = feats.shape
        scores = self.score(feats.view(b * f, c, h, w)).view(b, f, 1, h, w)
        weights = torch.softmax(scores, dim=1)
        return (weights * feats).sum(dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StackedSRNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        res_scale: float = 1.0,
        num_frames: int = 5,
        target_size: Tuple[int, int] = (32, 128),
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.target_size = target_size
        self.res_scale = res_scale

        stacked_channels = in_channels * num_frames
        self.head = nn.Conv2d(stacked_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(num_features, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, C, H, W]
        b, f, c, h, w = x.shape
        x = x.view(b, f * c, h, w)
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = self.tail(x)
        x = x + residual[:, : x.size(1), :, :] if residual.size(1) == x.size(1) else x
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return x
