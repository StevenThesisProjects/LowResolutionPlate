"""Reusable model components for multi-frame OCR."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, C, H, W]
        Returns:
            theta: Affine transformation matrix [Batch, 2, 3]
        """
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining multi-frame features.
    Computes a weighted sum of features from multiple frames based on their 'quality' scores.
    """
    def __init__(self, channels: int, sharpness_prior: bool = False):
        super().__init__()
        # Laplacian sharpness as an inductive bias on the frame weights. Measured on
        # this data: the blurriest quartile scores 54.2% and the sharpest 89.6%, so how
        # sharp a frame is predicts its usefulness better than anything the score net
        # can infer from ResNet features alone. gamma starts at 0, so the module is
        # identical to the plain version until training moves it.
        self.sharpness_prior = sharpness_prior
        self.gamma = nn.Parameter(torch.zeros(1)) if sharpness_prior else None
        # A small CNN to predict attention scores (quality map) from features
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, log_sharpness: torch.Tensor = None,
                num_frames: int = 5) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
            log_sharpness: [Batch, Frames] log Laplacian variance of the input frames
            num_frames: frames per track (5 in the dataset; fewer when measuring
                accuracy as a function of the number of frames)
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        batch_size = total_frames // num_frames

        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, num_frames, c, h, w)

        # Calculate attention scores: [Batch, Frames, 1, H, W]
        scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        if self.gamma is not None and log_sharpness is not None:
            scores = scores + self.gamma * log_sharpness.view(batch_size, num_frames, 1, 1, 1)
        weights = F.softmax(scores, dim=1)  # Normalize scores across frames

        # Weighted sum fusion
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class TemporalFusion(nn.Module):
    """Cross-frame Transformer fusion: every (frame, position) token attends to all others.

    AttentionFusion scores each frame from that frame's own features, so frames are
    never compared with each other — yet fusing 5 frames is worth +20.7 points over
    reading one (run #18), and the frames disagree exactly on the blurred characters
    that make up the errors (8/6, V/Y, 3/1). Here the 5 x W' feature columns of a
    track form one token sequence; a pre-LN Transformer lets each column look at the
    same and neighbouring columns of the other frames (tolerating small residual
    misalignment), and a per-position softmax over frames then pools the
    contextualised tokens.

    Initialised as an exact mean over frames: the residual branches' output layers and
    the pooling scorer start at zero, and the positional encoding added for attention
    is removed again afterwards. Same idea as the identity-initialised STN.
    """

    def __init__(self, channels: int = 512, num_layers: int = 2, heads: int = 8,
                 ff_dim: int = 1024, dropout: float = 0.1, num_frames: int = 5,
                 max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, max_len, C]
        # Which frame a token came from; zero-init so it starts as a no-op
        self.frame_emb = nn.Parameter(torch.zeros(1, num_frames, 1, channels))

        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        for blk in self.encoder.layers:
            # Pre-LN residual branches that output zero: each layer is the identity
            nn.init.zeros_(blk.self_attn.out_proj.weight)
            nn.init.zeros_(blk.self_attn.out_proj.bias)
            nn.init.zeros_(blk.linear2.weight)
            nn.init.zeros_(blk.linear2.bias)

        self.score = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: Per-frame feature maps [Batch * Frames, C, H', W']
        Returns:
            Fused feature map [Batch, C, 1, W'] (same contract as AttentionFusion)
        """
        x = F.adaptive_avg_pool2d(x, (1, None)).squeeze(2)          # [B*F, C, W']
        bf, c, w = x.shape
        b = bf // num_frames
        tok = x.permute(0, 2, 1).reshape(b, num_frames, w, c) + self.frame_emb[:, :num_frames]
        pe = self.pe[:, :w].unsqueeze(1)                               # [1, 1, W', C]
        out = self.encoder((tok + pe).reshape(b, num_frames * w, c))
        out = out.reshape(b, num_frames, w, c) - pe
        weights = torch.softmax(self.score(out), dim=1)                # [B, F, W', 1]
        fused = (out * weights).sum(dim=1)                             # [B, W', C]
        return fused.permute(0, 2, 1).unsqueeze(2)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based backbone customized for OCR.
    Uses ResNet34 with modified strides to preserve width (sequence length) while reducing height.
    """
    def __init__(self, pretrained: bool = False, collapse_height: bool = True):
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

        # (2,1) collapses height 2 -> 1 before fusion, which leaves AttentionFusion
        # scoring a 1x16 map: horizontal-only attention. Keeping (1,1) preserves a
        # height of 2 so fusion can weight top and bottom of a glyph separately.
        h_stride = 2 if collapse_height else 1
        self.layer4[0].conv1.stride = (h_stride, 1)
        self.layer4[0].downsample[0].stride = (h_stride, 1)
        self.collapse_height = collapse_height

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

        # Same op as the original code. The height is already 1, so the values do not
        # change, but the pooled tensor's memory layout does, and keeping the original
        # op sequence keeps training identical to it down to float rounding (verified
        # by tools/regression_check.py).
        if self.collapse_height:
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
