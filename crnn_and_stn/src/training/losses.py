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


class SRPixelLoss(nn.Module):
    """L1 + Sobel-edge L1 between SR output and HR ground truth."""

    def __init__(self, edge_weight: float = 0.5) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)  # [2, 1, 3, 3]
        self.register_buffer("sobel_kernel", kernel)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        # Grayscale first: OCR cares about edge geometry, not per-channel color.
        gray = x.mean(dim=1, keepdim=True)
        return F.conv2d(gray, self.sobel_kernel.to(dtype=gray.dtype), padding=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred/target: [N, 3, H, W], N = flattened batch*frame.
            mask: [N] bool — only samples with a pixel-aligned HR target
                (synthetic LR degraded from HR) contribute to the loss; real
                LR frames have no matching HR pixels and are excluded.
        """
        if not torch.any(mask):
            return pred.new_zeros(())

        pred_m = pred[mask]
        target_m = target[mask].to(dtype=pred_m.dtype)
        l1 = F.l1_loss(pred_m, target_m)
        edge_l1 = F.l1_loss(self._edges(pred_m), self._edges(target_m))
        return l1 + self.edge_weight * edge_l1
