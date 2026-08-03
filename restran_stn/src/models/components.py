"""Reusable model components for multi-frame OCR."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class LearnableDeblurBlock(nn.Module):
    """
    Lightweight learnable deblur/SR block trained end-to-end with CTC loss.
    Placed before STN so alignment sees sharper character boundaries.
    DEPRECATED: Use SuperResolutionBlock with auxiliary L1 loss instead.
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _ResidualConvBlock(nn.Module):
    """Single residual block: Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention (RCAN-style)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class _RCAB(nn.Module):
    """Residual Channel Attention Block (RCAN, Zhang et al. 2018).

    Conv-ReLU-Conv -> channel attention -> + skip. Following SR best practice
    (EDSR/RCAN) BatchNorm is dropped inside the SR body, letting the block
    reweight the most informative feature channels for stroke reconstruction.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.ca = _ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.body(x))


class SuperResolutionBlock(nn.Module):
    """Supervised (optionally upsampling) SR block with residual learning.

    Trained with an auxiliary L1 loss against real HR targets. When
    ``scale > 1`` the block performs genuine super-resolution: the body
    features are upsampled by ``scale`` with a sub-pixel (PixelShuffle)
    convolution and the reconstruction is added to a bicubic-upsampled base,
    so the network only has to learn the high-frequency residual. The HR
    target is supervised at its native (``scale``×) resolution rather than
    being downscaled to the LR grid, so the recognition backbone then reads a
    higher-resolution image (more pixels per character).

    output = up_bicubic(x) + tail(upsample(body(head(x))))
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 64,
        num_blocks: int = 4,
        scale: int = 1,
        channel_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        block = _RCAB if channel_attention else _ResidualConvBlock
        self.body = nn.Sequential(
            *[block(hidden) for _ in range(num_blocks)]
        )
        if scale > 1:
            # Sub-pixel upsampling by `scale` (ESPCN/EDSR-style).
            self.upsample = nn.Sequential(
                nn.Conv2d(hidden, hidden * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.ReLU(inplace=True),
            )
        else:
            self.upsample = nn.Identity()
        self.tail = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.upsample(self.body(self.head(x)))
        residual = self.tail(feat)
        if self.scale > 1:
            base = F.interpolate(
                x, scale_factor=self.scale, mode="bicubic", align_corners=False
            )
        else:
            base = x
        # Bound the output well outside the valid normalized image range
        # ([-1, 1]) so an under-supervised SR block (only ~half the samples
        # have HR targets) can never explode and destabilize the backbone.
        return torch.clamp(base + residual, -6.0, 6.0)


class PositionalHead(nn.Module):
    """Fixed-length positional recognition head (parallel-attention style).

    License plates in this dataset are always exactly ``num_positions`` (7)
    characters with a rigid per-position character-class layout. Instead of
    relying only on order-agnostic CTC, this head uses ``num_positions``
    learnable position queries that attend over the transformer sequence to
    extract one feature per output slot, then classifies each slot over the
    real character set (no blank). Trained jointly with CTC (multi-task); at
    inference its per-position posteriors are directly maskable by the layout
    template, which resolves letter/digit confusions structurally.

    Output: ``[B, num_positions, num_chars]`` raw logits (no softmax).
    """

    def __init__(
        self,
        d_model: int,
        num_positions: int = 7,
        num_chars: int = 36,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.num_chars = num_chars
        self.queries = nn.Parameter(torch.randn(num_positions, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_chars)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Args: seq [B, T, d_model]. Returns: [B, num_positions, num_chars]."""
        b = seq.size(0)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)  # [B, P, C]
        attended, _ = self.attn(q, seq, seq)             # [B, P, C]
        feats = self.norm(attended + q)
        return self.classifier(self.dropout(feats))      # [B, P, num_chars]


class MultiFrameSuperResolutionBlock(nn.Module):
    """Super-resolution that reconstructs each frame using ALL frames of a track.

    The per-frame block treats the 5 frames as 5 unrelated images, discarding
    the single most useful property of this dataset: the frames show the SAME
    plate, so their independent sensor noise averages out while the underlying
    strokes reinforce. This block therefore builds a track-level aggregate of
    the head features and concatenates it back onto every frame before the
    residual body, giving each reconstruction access to evidence the frame
    alone does not contain.

    Output stays per-frame — ``[B*F, 3, sH, sW]`` — so the downstream attention
    fusion and the per-frame HR supervision both keep working unchanged.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 64,
        num_blocks: int = 4,
        scale: int = 1,
        num_frames: int = 5,
        channel_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.num_frames = num_frames
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # Merge "this frame" with "the whole track" (mean + max capture the
        # consensus and the sharpest evidence across frames respectively).
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1),
            nn.ReLU(inplace=True),
        )
        block = _RCAB if channel_attention else _ResidualConvBlock
        self.body = nn.Sequential(*[block(hidden) for _ in range(num_blocks)])
        if scale > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(hidden, hidden * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.ReLU(inplace=True),
            )
        else:
            self.upsample = nn.Identity()
        self.tail = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bf = x.size(0)
        f = self.num_frames
        b = bf // f

        feat = self.head(x)                       # [B*F, C, H, W]
        c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
        fv = feat.view(b, f, c, h, w)
        mean_agg = fv.mean(dim=1, keepdim=True).expand(-1, f, -1, -1, -1)
        max_agg = fv.max(dim=1, keepdim=True).values.expand(-1, f, -1, -1, -1)
        merged = torch.cat([fv, mean_agg, max_agg], dim=2).reshape(bf, c * 3, h, w)
        feat = self.fuse(merged)

        residual = self.tail(self.upsample(self.body(feat)))
        if self.scale > 1:
            base = F.interpolate(
                x, scale_factor=self.scale, mode="bicubic", align_corners=False
            )
        else:
            base = x
        return torch.clamp(base + residual, -6.0, 6.0)


class STNBlock(nn.Module):
    """
    Spatial Transformer Network (STN) for image alignment.
    Learns to crop and rectify images before feeding them to the backbone.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Localization network: Predicts transformation parameters
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)) # Output fixed size for FC
        )
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor, num_frames: int = None) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch*Frames, C, H, W]
            num_frames: If given, predict ONE shared affine per track instead of
                one per frame. The localization features are averaged over the
                track's frames before regressing theta, so every frame of a
                track is warped identically. This keeps the 5 frames pixel-
                aligned with each other going into fusion (independent per-frame
                warps can misalign them and blur the fused features).
        Returns:
            theta: Affine transformation matrix [Batch*Frames, 2, 3]
        """
        xs = self.localization(x)
        if num_frames is not None and num_frames > 1:
            bf = xs.size(0)
            b = bf // num_frames
            xs = xs.view(b, num_frames, *xs.shape[1:]).mean(dim=1)
            theta = self.fc_loc(xs).view(b, 2, 3)
            # Broadcast the track-level transform back to every frame.
            theta = theta.unsqueeze(1).expand(b, num_frames, 2, 3).reshape(bf, 2, 3)
        else:
            theta = self.fc_loc(xs).view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining multi-frame features.
    Combines spatial quality maps with global per-frame scores for blur-aware fusion.
    """
    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()
        self.num_frames = num_frames
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )
        self.frame_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        num_frames = self.num_frames
        batch_size = total_frames // num_frames

        x_view = x.view(batch_size, num_frames, c, h, w)

        spatial_scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        global_scores = self.frame_quality(x).view(batch_size, num_frames, 1, 1, 1)
        scores = spatial_scores + global_scores
        weights = F.softmax(scores, dim=1)

        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class CrossFrameFusion(nn.Module):
    """Content-based cross-frame fusion via attention pooling.

    The old ``AttentionFusion`` computes a softmax quality weight per frame and
    returns a weighted sum — it can only *select* the best frame at each
    position, not *combine* complementary detail from several frames. Here, for
    every width position we treat the ``num_frames`` per-frame feature vectors
    as a short sequence and let a learned query attend over them (attention
    pooling). This aggregates sharp strokes from frame A with clean background
    from frame B at the same location — the core multi-frame SR benefit.

    Input:  [B*F, C, 1, W']   (backbone features, height already collapsed)
    Output: [B, C, 1, W']
    """

    def __init__(self, channels: int, num_frames: int = 5, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_frames = num_frames
        self.query = nn.Parameter(torch.randn(1, 1, channels) * 0.02)
        self.attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_frames, c, h, w = x.size()          # h == 1
        f = self.num_frames
        b = total_frames // f
        # [B*F, C, 1, W'] -> [B, F, C, W'] -> [B, W', F, C] -> [B*W', F, C]
        seq = x.squeeze(2).view(b, f, c, w).permute(0, 3, 1, 2).reshape(b * w, f, c)
        q = self.query.expand(b * w, 1, c)         # [B*W', 1, C]
        fused, _ = self.attn(q, seq, seq)          # [B*W', 1, C]
        fused = self.norm(fused.squeeze(1))        # [B*W', C]
        # [B*W', C] -> [B, W', C] -> [B, C, 1, W']
        return fused.view(b, w, c).permute(0, 2, 1).unsqueeze(2)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based backbone customized for OCR.
    Uses ResNet34 with modified strides to preserve width (sequence length) while reducing height.
    """
    @staticmethod
    def _bn_to_gn(module: nn.Module, num_groups: int = 32) -> None:
        """Recursively replace BatchNorm2d with GroupNorm in place.

        Conv weights (the valuable pretrained part) are untouched; only the
        normalization is swapped. GN affine params are seeded from the BN ones.
        GroupNorm is batch-size independent and avoids fp16 BatchNorm blow-ups,
        so it stays stable at the higher LR the author's recipe uses.
        """
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                c = child.num_features
                ng = num_groups
                while c % ng != 0 and ng > 1:
                    ng //= 2
                gn = nn.GroupNorm(ng, c)
                with torch.no_grad():
                    gn.weight.copy_(child.weight.detach())
                    gn.bias.copy_(child.bias.detach())
                setattr(module, name, gn)
            else:
                ResNetFeatureExtractor._bn_to_gn(child, num_groups)

    def __init__(self, pretrained: bool = False, norm: str = "batch"):
        super().__init__()

        # Load ResNet34 from torchvision
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        # --- OCR Customization ---
        # We need to keep the standard first layer (stride 2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify strides in layer3 and layer4 to (2, 1)
        # This reduces height but preserves width for sequence modeling
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

        # Swap BatchNorm -> GroupNorm (keeps pretrained conv weights) if asked.
        if norm == "group":
            self._bn_to_gn(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, 3, H, W]
        Returns:
            Features [Batch, 512, H // 16, W // 2] (approx)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Ensure height is 1 for sequence modeling (Height collapsing)
        # Output shape: [Batch, 512, 1, W']
        x = F.adaptive_avg_pool2d(x, (1, None))
        return x


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Standard Sinusoidal implementation from 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [Batch, Seq_Len, Dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)