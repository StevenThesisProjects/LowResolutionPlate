"""Auxiliary pixel-level loss for the SR head.

The stacked-input SR module (PR #7) was only ever trained through CTC
gradients, so it had no incentive to reconstruct real detail and instead
degraded into a high-frequency noise generator. This loss gives the SR
output a direct, pixel-level target (L1 + Sobel edge) against the matching
HR frame, so the network learns to sharpen edges instead of hallucinating
texture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss on frozen VGG16 features (the L_Perceptual of issue #9).

    Loaded lazily and only when the weight is non-zero, so the default recipe
    never pays the download/VRAM cost. Features are compared at relu2_2, which
    is shallow enough to stay about local structure (strokes, edges) rather
    than object semantics that mean nothing for a license plate.
    """

    def __init__(self, layer_index: int = 9) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[: layer_index + 1].eval()
        for param in self.features.parameters():
            param.requires_grad = False
        # ImageNet statistics; inputs arrive normalized to [-1, 1].
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1,1] -> [0,1]
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.features(self._prepare(pred)), self.features(self._prepare(target)))


class SRPixelLoss(nn.Module):
    """Pixel loss between the SR output and the matching HR frame.

    Implements the issue #9 formula `L_SR = L1 + alpha * L_perceptual`, which is
    what the default configuration runs. `edge_weight` adds a Sobel-edge L1 term
    that the issue does not specify - plates are read from stroke geometry, so it
    is plausible it helps, but it stays off by default so it gets measured as its
    own ablation instead of being folded into the proposed method.
    """

    def __init__(self, edge_weight: float = 0.0, perceptual_weight: float = 0.0) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)  # [2, 1, 3, 3]
        self.register_buffer("sobel_kernel", kernel)
        self.perceptual = VGGPerceptualLoss() if perceptual_weight > 0 else None

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        # Grayscale first: OCR cares about edge geometry, not per-channel color.
        gray = x.mean(dim=1, keepdim=True)
        return F.conv2d(gray, self.sobel_kernel.to(dtype=gray.dtype), padding=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred/target: [N, 3, H, W], N = flattened (selected samples)*frame.
                Only samples with a pixel-aligned HR target reach this point;
                the selection happens in the collate function so real LR frames
                are never materialized as zeros in the first place.
        """
        if pred.numel() == 0:
            return pred.new_zeros(())

        target = target.to(dtype=pred.dtype)
        loss = F.l1_loss(pred, target)
        if self.edge_weight > 0:
            loss = loss + self.edge_weight * F.l1_loss(self._edges(pred), self._edges(target))
        if self.perceptual is not None:
            loss = loss + self.perceptual_weight * self.perceptual(pred, target)
        return loss
