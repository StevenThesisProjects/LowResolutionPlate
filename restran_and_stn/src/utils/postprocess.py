"""Post-processing utilities for OCR decoding."""
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 1,
    expected_length: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """CTC decode with greedy or beam search.

    Args:
        expected_length: If set (e.g. 7 for fixed-length plates), the beam
            decoder prefers the highest-scoring candidate whose collapsed
            length equals this value, falling back to the global best when no
            such candidate exists. Ignored for greedy decoding.
    """
    if beam_width <= 1:
        return _greedy_decode_batch(preds, idx2char)
    return _beam_decode_batch(
        preds, idx2char, beam_width=beam_width, expected_length=expected_length
    )


def _greedy_decode_batch(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
) -> List[Tuple[str, float]]:
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()

    results: List[Tuple[str, float]] = []
    for batch_idx in range(indices_np.shape[0]):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]

        pred_chars: List[str] = []
        confidences: List[float] = []
        time_idx = 0

        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)

            if char_idx != 0:
                pred_chars.append(idx2char.get(char_idx, ''))
                confidences.append(float(np.max(probs_b[time_idx:time_idx + group_size])))

            time_idx += group_size

        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))

    return results


def _beam_decode_batch(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 10,
    blank: int = 0,
    expected_length: Optional[int] = None,
) -> List[Tuple[str, float]]:
    log_probs = preds.detach().cpu().numpy()
    results: List[Tuple[str, float]] = []

    for batch_idx in range(log_probs.shape[0]):
        seq_log_probs = log_probs[batch_idx]
        best_text, best_score = _beam_search_single(
            seq_log_probs, idx2char, beam_width, blank,
            expected_length=expected_length,
        )
        confidence = float(np.exp(best_score / max(len(best_text), 1)))
        results.append((best_text, min(confidence, 1.0)))

    return results


def _beam_search_single(
    log_probs: np.ndarray,
    idx2char: Dict[int, str],
    beam_width: int,
    blank: int = 0,
    expected_length: Optional[int] = None,
) -> Tuple[str, float]:
    """CTC prefix beam search following Graves / Distill.pub.

    When ``expected_length`` is given, the final beams are reranked to prefer
    the highest-scoring labeling of exactly that length (fixed-length plate
    prior), falling back to the global best if none is found.
    """
    # Each beam: (labeling, log_prob_blank_end, log_prob_non_blank_end)
    beams: List[Tuple[Tuple[int, ...], float, float]] = [((), 0.0, -np.inf)]

    for t in range(log_probs.shape[0]):
        next_map: Dict[Tuple[int, ...], List[float]] = {}

        def add_state(
            labeling: Tuple[int, ...],
            log_pb: float,
            log_pnb: float,
        ) -> None:
            if labeling not in next_map:
                next_map[labeling] = [-np.inf, -np.inf]
            next_map[labeling][0] = np.logaddexp(next_map[labeling][0], log_pb)
            next_map[labeling][1] = np.logaddexp(next_map[labeling][1], log_pnb)

        for labeling, pb, pnb in beams:
            logp_blank = float(log_probs[t, blank])
            # 1. Extend by blank: new path ends in blank
            add_state(labeling, np.logaddexp(pb, pnb) + logp_blank, -np.inf)

            # 2. Extend by non-blank
            for char_idx in range(log_probs.shape[1]):
                if char_idx == blank:
                    continue
                logp_char = float(log_probs[t, char_idx])

                if labeling and char_idx == labeling[-1]:
                    # Repeat char:
                    # If previous ended in blank, it extends the sequence
                    add_state(labeling + (char_idx,), -np.inf, pb + logp_char)
                    # If previous ended in non-blank, it merges (collapses)
                    add_state(labeling, -np.inf, pnb + logp_char)
                else:
                    # New char: always extends the sequence
                    add_state(labeling + (char_idx,), -np.inf, np.logaddexp(pb, pnb) + logp_char)

        beams = sorted(
            [
                (labeling, scores[0], scores[1])
                for labeling, scores in next_map.items()
            ],
            key=lambda item: np.logaddexp(item[1], item[2]),
            reverse=True,
        )[:beam_width]

    if not beams:
        return "", -np.inf

    def _beam_score(item: Tuple[Tuple[int, ...], float, float]) -> float:
        return float(np.logaddexp(item[1], item[2]))

    # Fixed-length prior: prefer the best-scoring beam of the expected length.
    if expected_length is not None:
        length_matched = [b for b in beams if len(b[0]) == expected_length]
        if length_matched:
            best = max(length_matched, key=_beam_score)
            text = "".join(idx2char.get(idx, '') for idx in best[0])
            return text, _beam_score(best)

    best = max(beams, key=_beam_score)
    best_score = _beam_score(best)
    text = "".join(idx2char.get(idx, '') for idx in best[0])
    return text, best_score


@torch.no_grad()
def forward_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    brightness_scales: Optional[List[float]] = None,
) -> torch.Tensor:
    """Average predictions over mild brightness variants (probability mean)."""
    if brightness_scales is None:
        brightness_scales = [1.0, 1.06, 0.94]

    prob_sum = None
    for scale in brightness_scales:
        scaled = images if scale == 1.0 else torch.clamp(images * scale, -1.0, 1.0)
        log_preds = model(scaled)
        probs = log_preds.exp()
        prob_sum = probs if prob_sum is None else prob_sum + probs

    assert prob_sum is not None
    avg_probs = prob_sum / len(brightness_scales)
    return avg_probs.clamp(min=1e-8).log()


def decode_batch(
    model: nn.Module,
    images: torch.Tensor,
    idx2char: Dict[int, str],
    use_tta: bool = False,
    beam_width: int = 1,
    expected_length: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Run forward (optionally with TTA) and CTC decode."""
    if use_tta:
        preds = forward_with_tta(model, images)
    else:
        preds = model(images)
    return decode_with_confidence(
        preds, idx2char, beam_width=beam_width, expected_length=expected_length
    )
