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


class SuperResolutionBlock(nn.Module):
    """Supervised SR block with residual learning.

    Trained with auxiliary L1 loss against HR targets (resized to model
    resolution).  4 residual blocks at 64 channels give much more capacity
    than the old 3-conv LearnableDeblurBlock.

    output = input + reconstruct(input)   (residual learning)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[_ResidualConvBlock(hidden) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.tail(self.body(self.head(x)))
        return x + residual


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


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based backbone customized for OCR.
    Uses ResNet34 with modified strides to preserve width (sequence length) while reducing height.
    """
    def __init__(self, pretrained: bool = False):
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