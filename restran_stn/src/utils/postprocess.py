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
    pos_classes: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """CTC decode with greedy or beam search.

    Args:
        expected_length: If set (e.g. 7 for fixed-length plates), the beam
            decoder prefers the highest-scoring candidate whose collapsed
            length equals this value, falling back to the global best when no
            such candidate exists. Ignored for greedy decoding.
        pos_classes: Optional per-position character-class template
            (e.g. ``['L','L','L','D','LD','D','D']``). Enforced during beam
            expansion. Ignored for greedy decoding.
    """
    if beam_width <= 1:
        return _greedy_decode_batch(preds, idx2char)
    return _beam_decode_batch(
        preds, idx2char, beam_width=beam_width,
        expected_length=expected_length, pos_classes=pos_classes,
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
    pos_classes: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    log_probs = preds.detach().cpu().numpy()
    results: List[Tuple[str, float]] = []

    for batch_idx in range(log_probs.shape[0]):
        seq_log_probs = log_probs[batch_idx]
        best_text, best_score = _beam_search_single(
            seq_log_probs, idx2char, beam_width, blank,
            expected_length=expected_length, pos_classes=pos_classes,
        )
        confidence = float(np.exp(best_score / max(len(best_text), 1)))
        results.append((best_text, min(confidence, 1.0)))

    return results


def _char_class(ch: str) -> str:
    """'D' for a digit, 'L' for a letter, '' otherwise."""
    if ch.isdigit():
        return 'D'
    if ch.isalpha():
        return 'L'
    return ''


def _beam_search_single(
    log_probs: np.ndarray,
    idx2char: Dict[int, str],
    beam_width: int,
    blank: int = 0,
    expected_length: Optional[int] = None,
    pos_classes: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """CTC prefix beam search following Graves / Distill.pub.

    When ``expected_length`` is given, the final beams are reranked to prefer
    the highest-scoring labeling of exactly that length (fixed-length plate
    prior), falling back to the global best if none is found.

    When ``pos_classes`` is given (e.g. ``['L','L','L','D','LD','D','D']`` for
    Brazilian/Mercosur plates), the search only lets a labeling grow into
    position ``k`` with a character whose class (letter 'L' / digit 'D') is
    allowed at that position, and never grow past ``len(pos_classes)``. Because
    the plate format is 100%% consistent, this prunes impossible letter/digit
    confusions without ever removing the correct answer.
    """
    # Precompute allowed vocab indices per position when a template is given.
    allowed_idx_by_pos = None
    if pos_classes is not None:
        idx_class = {i: _char_class(c) for i, c in idx2char.items()}
        allowed_idx_by_pos = []
        for allowed in pos_classes:
            allowed_set = set(allowed)  # e.g. 'LD' -> {'L','D'}
            allowed_idx_by_pos.append(
                {i for i, cl in idx_class.items() if cl in allowed_set}
            )

    def _can_grow_to(labeling: Tuple[int, ...], char_idx: int) -> bool:
        """Whether appending char_idx as a new collapsed position is allowed."""
        if allowed_idx_by_pos is None:
            return True
        pos = len(labeling)  # 0-based index of the position being created
        if pos >= len(allowed_idx_by_pos):
            return False  # would exceed the fixed plate length
        return char_idx in allowed_idx_by_pos[pos]

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
                    if _can_grow_to(labeling, char_idx):
                        add_state(labeling + (char_idx,), -np.inf, pb + logp_char)
                    # If previous ended in non-blank, it merges (collapses):
                    # no new position, always allowed.
                    add_state(labeling, -np.inf, pnb + logp_char)
                else:
                    # New char: always extends the sequence
                    if _can_grow_to(labeling, char_idx):
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


def decode_positional(
    pos_logits: torch.Tensor,
    pos_idx2char: Dict[int, str],
    pos_classes: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Decode the fixed-length positional head.

    Args:
        pos_logits: [B, P, num_chars] raw logits (no blank).
        pos_idx2char: class-index (0-based) -> character.
        pos_classes: optional per-position class template ('L'/'D'/'LD'); the
            posterior mass of disallowed classes at each position is zeroed
            before taking the argmax, so structurally impossible characters
            can never win.
    Returns:
        List of (text, mean_confidence).
    """
    probs = torch.softmax(pos_logits.float(), dim=-1)  # [B, P, num_chars]
    b, p, n = probs.shape

    if pos_classes is not None:
        idx_class = {i: _char_class(pos_idx2char.get(i, '')) for i in range(n)}
        mask = torch.zeros(p, n, device=probs.device)
        for pos in range(p):
            allowed = set(pos_classes[pos]) if pos < len(pos_classes) else {'L', 'D'}
            for i in range(n):
                if idx_class[i] in allowed:
                    mask[pos, i] = 1.0
        probs = probs * mask.unsqueeze(0)

    conf, idx = probs.max(dim=-1)  # [B, P]
    idx_np = idx.detach().cpu().numpy()
    conf_np = conf.detach().cpu().numpy()

    results: List[Tuple[str, float]] = []
    for bi in range(b):
        text = "".join(pos_idx2char.get(int(j), '') for j in idx_np[bi])
        results.append((text, float(conf_np[bi].mean())))
    return results


def fuse_ctc_positional(
    ctc_results: List[Tuple[str, float]],
    pos_results: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """Sequence-level fusion of the CTC beam and positional-head predictions.

    When both heads agree, keep the label with the higher confidence. When they
    disagree, choose the more confident head. Ties favor the positional head
    because it is structurally constrained to the plate layout.
    """
    fused: List[Tuple[str, float]] = []
    for (ct, cc), (pt, pc) in zip(ctc_results, pos_results):
        if ct == pt:
            fused.append((ct, max(cc, pc)))
        elif pc >= cc:
            fused.append((pt, pc))
        else:
            fused.append((ct, cc))
    return fused


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
    pos_classes: Optional[List[str]] = None,
    fuse_positional: bool = False,
    pos_idx2char: Optional[Dict[int, str]] = None,
) -> List[Tuple[str, float]]:
    """Run forward (optionally with TTA) and CTC decode.

    When ``fuse_positional`` is set and the model has a positional head, also
    decode the positional head (with the same layout mask) and fuse it with the
    CTC beam result.
    """
    want_pos = fuse_positional and getattr(model, "use_positional_head", False)
    pos_logits = None
    if use_tta:
        preds = forward_with_tta(model, images)
        if want_pos:
            _p, pos_logits = model(images, return_pos=True)
    elif want_pos:
        preds, pos_logits = model(images, return_pos=True)
    else:
        preds = model(images)

    ctc_results = decode_with_confidence(
        preds, idx2char, beam_width=beam_width,
        expected_length=expected_length, pos_classes=pos_classes,
    )
    if pos_logits is not None and pos_idx2char is not None:
        pos_results = decode_positional(
            pos_logits, pos_idx2char, pos_classes=pos_classes
        )
        return fuse_ctc_positional(ctc_results, pos_results)
    return ctc_results
