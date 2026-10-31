"""Multi-frame super-resolution front-end (EDSR + channel attention + PixelShuffle).

Motivation, measured on this dataset: reading a single LR frame scores 56.5% while
fusing 5 frames scores 77.2%. The 5 frames therefore carry genuinely complementary
information. ResTranOCR only combines them *after* a per-frame backbone, so it never
compares frames at the pixel level — which is exactly what sub-pixel reconstruction
needs. This module stacks the 5 frames on the channel axis and learns a joint 2x
upsample, then hands 5 frames back so `AttentionFusion` (which hard-codes 5) still works.

No pixel-wise loss is used anywhere: `lr-00i` and `hr-00i` are different captures and
cannot be registered to each other, so the module is trained end-to-end through CTC.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    """Channel attention (RCAN): rescale channels by their global context.

    High-frequency channels carry the character strokes that matter for OCR; this
    lets the network weight them above low-frequency background channels.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class ResBlock(nn.Module):
    """EDSR residual block: no BatchNorm, scaled residual, plus channel attention.

    BatchNorm normalises features to mean 0 / std 1, which caps the dynamic range
    reconstruction needs — EDSR removes it for both quality and memory.
    """

    def __init__(self, num_features: int, res_scale: float = 0.1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.calayer = CALayer(num_features, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.calayer(out)
        return out * self.res_scale + x


class Upsampler(nn.Module):
    """2x upsample via PixelShuffle: work in LR space, learn the upscale."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StackedSRNet(nn.Module):
    """5 LR frames -> 5 frames at 2x resolution.

    Returning 5 frames (not the 1 frame the original design produced) is what keeps
    `AttentionFusion`'s `num_frames = 5` reshape valid downstream.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 5,
        num_features: int = 32,
        num_blocks: int = 8,
        res_scale: float = 0.1,
        scale: int = 2,
    ):
        super().__init__()
        if scale not in (1, 2):
            raise ValueError(f"scale must be 1 or 2, got {scale}")
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.scale = scale
        stacked = in_channels * num_frames

        self.head = nn.Conv2d(stacked, num_features, 3, padding=1)
        body = [ResBlock(num_features, res_scale) for _ in range(num_blocks)]
        body.append(nn.Conv2d(num_features, num_features, 3, padding=1))
        self.body = nn.Sequential(*body)
        # scale=1 keeps the input resolution: the module then only does cross-frame
        # pixel-level mixing, which isolates that effect from any resolution change.
        self.upsampler = Upsampler(num_features) if scale == 2 else nn.Identity()
        self.upsample_conv = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.tail = nn.Conv2d(num_features, stacked, 3, padding=1)

        # Start as an exact identity: the module outputs interpolate(x) + 0, so the
        # network behaves like the plain baseline at step 0 and only has to learn the
        # improvement. Without this, a random `tail` emits noise and the whole
        # backbone must first recover from a corrupted image — which is what the
        # scale=1 control measured as -2.85 points. Same trick as STNBlock's
        # identity-initialised affine head.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, F, C, H, W] -> [B, F, C, H*scale, W*scale]"""
        b, f, c, h, w = x.size()
        stacked = x.reshape(b, f * c, h, w)

        feat = self.head(stacked)
        feat = self.body(feat) + feat          # global residual in feature space
        feat = self.upsample_conv(self.upsampler(feat))
        residual = self.tail(feat)             # [B, F*C, H*scale, W*scale]

        # SR = learned residual on top of interpolation, the standard formulation.
        base = stacked if self.scale == 1 else F.interpolate(
            stacked, scale_factor=2, mode='bilinear', align_corners=False
        )
        out = base + residual
        return out.reshape(b, f, c, h * self.scale, w * self.scale)
