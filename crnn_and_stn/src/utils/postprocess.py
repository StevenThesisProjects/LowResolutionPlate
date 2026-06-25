"""
CTC Decoding — chuyển output model thành chuỗi biển số + confidence.

Quy trình (report Trang 31, 49):
  1. Greedy decode: lấy ký tự xác suất cao nhất tại mỗi time step
  2. Merge repeats: gộp ký tự lặp liên tiếp
  3. Remove blank: loại index 0
  4. Confidence = trung bình xác suất max của từng ký tự giữ lại

Nguồn gốc: MultiFrame-LPR-main/src/utils/postprocess.py
"""
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
) -> List[Tuple[str, float]]:
    """
    Args:
        preds    : [Batch, TimeSteps, NumClasses] log-softmax từ model
        idx2char : {1: '0', 2: '1', ..., 36: 'Z'}
    Returns:
        List of (predicted_text, confidence)
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()

    results: List[Tuple[str, float]] = []

    for batch_idx in range(indices_np.shape[0]):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]

        pred_chars = []
        confidences = []
        time_idx = 0

        # groupby: gom các time step liên tiếp có cùng ký tự
        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)

            if char_idx != 0:  # bỏ qua blank (index 0)
                pred_chars.append(idx2char.get(char_idx, ""))
                group_probs = probs_b[time_idx : time_idx + group_size]
                confidences.append(float(np.max(group_probs)))

            time_idx += group_size

        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))

    return results
