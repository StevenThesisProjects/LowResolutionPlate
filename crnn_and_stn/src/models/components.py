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


def make_norm(norm: str, channels: int, groups: int = 8) -> nn.Module:
    """Normalization factory for the backbone.

    EDSR drops BatchNorm because BN clamps the dynamic range an SR decoder
    needs to reconstruct RGB. That argument does not carry over to an OCR
    backbone, whose output is a high-dimensional feature sequence feeding a
    BiLSTM: with no normalization at all the feature scale drifts and the
    gradients blow up (observed as NaN once the SR head doubles the spatial
    size). GroupNorm restores stability without BatchNorm's small-batch
    problems, since it normalizes per-sample.
    """
    if norm == "none":
        return nn.Identity()
    if norm == "group":
        return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
    raise ValueError(f"Unknown norm type: {norm!r} (expected 'none' or 'group')")


class ResidualBlock(nn.Module):
    """Residual block with optional GroupNorm for stable OCR feature learning.

    BatchNorm is often less suitable for OCR with small batches and repeated
    frame fusion, so this block uses plain convolutions + PReLU and a small
    residual scale to keep updates well-behaved. `norm="group"` adds GroupNorm
    back as the normalization that BatchNorm's removal left missing.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 1,
        res_scale: float = 0.1,
        use_se: bool = False,
        norm: str = "none",
    ) -> None:
        super().__init__()
        mid_channels = channels * expansion
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = make_norm(norm, mid_channels)
        self.act = nn.PReLU(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = make_norm(norm, channels)
        self.res_scale = res_scale
        self.attn = SqueezeExcitation(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
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
        norm: str = "none",
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
                ResidualBlock(out_ch, res_scale=res_scale, use_se=use_se, norm=norm)
                for _ in range(num_blocks)
            )
            if idx < len(downsample_strides):
                stride = downsample_strides[idx]
                stage_layers.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)
                )
                stage_layers.append(make_norm(norm, out_ch))
                stage_layers.append(nn.PReLU(out_ch))
            in_ch = out_ch
            stages.append(nn.Sequential(*stage_layers))

        self.stages = nn.ModuleList(stages)
        self.output_channels = int(stage_channels[-1])

        # Final refinement gives the head a cleaner and more expressive feature
        # map before it is converted into a sequence.
        self.output_refine = nn.Sequential(
            ResidualBlock(self.output_channels, res_scale=res_scale, use_se=use_se, norm=norm),
            nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1, bias=True),
            make_norm(norm, self.output_channels),
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
    """Fuse multiple frame features.

    `mode` selects the fusion rule so Ablation 2 can compare them on equal
    footing: "attention" learns per-frame reliability weights, while "avg" and
    "max" are parameter-free baselines that ignore frame quality.
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: int = 4,
        dropout: float = 0.0,
        mode: str = "attention",
    ) -> None:
        super().__init__()
        if mode not in {"attention", "avg", "max"}:
            raise ValueError(f"Unknown fusion mode: {mode!r} (expected attention/avg/max)")
        self.mode = mode
        if mode == "attention":
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

        if self.mode == "avg":
            return features.mean(dim=1)
        if self.mode == "max":
            return features.max(dim=1).values

        batch_size, num_frames, channels, height, width = features.shape
        # Global pooling produces one descriptor per frame; the scorer then
        # learns which frames are most reliable for OCR decoding.
        pooled = features.mean(dim=(3, 4))
        scores = self.scorer(pooled).squeeze(-1)
        weights = torch.softmax(scores, dim=1).view(batch_size, num_frames, 1, 1, 1)
        return torch.sum(features * weights, dim=1)

    def last_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Per-frame weights for visualization; uniform for the parameter-free modes."""
        if features.dim() == 4:
            features = features.unsqueeze(1)
        batch_size, num_frames = features.shape[:2]
        if self.mode != "attention":
            return features.new_full((batch_size, num_frames), 1.0 / num_frames)
        scores = self.scorer(features.mean(dim=(3, 4))).squeeze(-1)
        return torch.softmax(scores, dim=1)


class DCNAlignment(nn.Module):
    """Deformable-conv alignment across frames (DCNv2, Step 2 of the pipeline).

    STN can only apply one global affine warp per frame, so residual local
    misalignment (motion blur, rolling shutter, plate bending) survives it.
    Here each frame is aligned toward the middle reference frame: the offsets
    are predicted from the concatenated (frame, reference) pair, then applied
    with a modulated deformable convolution. Offsets start at zero so the
    module begins as a near-identity op, the same way STN starts at identity.
    """

    def __init__(self, channels: int = 3, hidden_channels: int = 32, deform_groups: int = 1) -> None:
        super().__init__()
        from torchvision.ops import DeformConv2d

        self.kernel_size = 3
        self.deform_groups = deform_groups
        offset_channels = deform_groups * 2 * self.kernel_size * self.kernel_size
        mask_channels = deform_groups * self.kernel_size * self.kernel_size

        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(hidden_channels),
        )
        self.offset_head = nn.Conv2d(hidden_channels, offset_channels, kernel_size=3, padding=1, bias=True)
        self.mask_head = nn.Conv2d(hidden_channels, mask_channels, kernel_size=3, padding=1, bias=True)
        self.deform = DeformConv2d(
            channels, channels, kernel_size=self.kernel_size, padding=1, groups=1, bias=True
        )

        # Zero-init the offset/mask heads so the very first forward is a plain
        # 3x3 conv with uniform modulation - no random warping early on.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, F, C, H, W] -> aligned [B, F, C, H, W]."""
        batch_size, num_frames, channels, height, width = frames.shape
        reference = frames[:, num_frames // 2]  # middle frame as alignment target

        aligned = []
        for idx in range(num_frames):
            current = frames[:, idx]
            pair = torch.cat([current, reference], dim=1)
            feat = self.offset_net(pair)
            offset = self.offset_head(feat)
            # Sigmoid*2 keeps modulation in [0,2] and equals 1.0 at zero-init.
            mask = torch.sigmoid(self.mask_head(feat)) * 2.0
            aligned.append(self.deform(current, offset, mask))
        return torch.stack(aligned, dim=1)


class FrameSR(nn.Module):
    """Lightweight per-frame super-resolution head.

    Applied independently to each of the B*F flattened frames (never
    channel-stacked across frames), so every frame keeps its own detail for
    the attention fusion step later. The learned path is added on top of a
    plain bilinear upscale (residual-style), which keeps early training
    well-behaved the same way STN starts from an identity transform.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        scale: int = 2,
        res_scale: float = 0.1,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.head_act = nn.PReLU(hidden_channels)
        self.body = nn.Sequential(
            *[ResidualBlock(hidden_channels, res_scale=res_scale, norm=norm) for _ in range(num_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * scale * scale, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            nn.PReLU(hidden_channels),
        )
        self.tail = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        feat = self.head_act(self.head(x))
        feat = feat + self.body(feat)
        feat = self.upsample(feat)
        return self.tail(feat) + base


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
