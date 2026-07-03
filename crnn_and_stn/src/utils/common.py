"""Tiện ích dùng chung."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, benchmark: bool = False) -> None:
    """Đặt seed cho reproducibility (report: seed=42)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if benchmark:
        print("⚡ CUDNN benchmark ON (nhanh hơn, ít reproducible hơn).")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("🔒 Deterministic mode ON (reproducible, seed=42).")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
