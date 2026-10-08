"""Utility helpers shared across the OCR training pipeline."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, benchmark: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable OCR experiments.

    The optional `benchmark` flag keeps the previous project behavior: enabling
    cuDNN benchmark can improve speed, while disabling it makes runs more
    deterministic.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if benchmark:
        # Faster convolutions, but slightly less reproducible across runs.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        # Better reproducibility, useful for ablation and debugging.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
