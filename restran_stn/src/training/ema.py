"""Exponential Moving Average of model weights for stabler validation."""
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """Track EMA weights and optionally swap them into the live model."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = deepcopy(model.state_dict())
        for tensor in self.shadow.values():
            tensor.detach_()

    @staticmethod
    def _is_ema_tensor(tensor: torch.Tensor) -> bool:
        return tensor.is_floating_point()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if not self._is_ema_tensor(value):
                # BatchNorm counters and other integer buffers are copied as-is.
                self.shadow[key].copy_(value)
                continue
            self.shadow[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_into(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
