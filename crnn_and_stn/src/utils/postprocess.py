"""Post-processing helpers for CTC OCR decoding.

The original project only needed greedy decoding with a confidence estimate.
This version keeps that behavior and also adds reusable text utilities so
training/validation code can measure exact-match and edit-distance style
metrics consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch

DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize_text(text: str) -> str:
    """Normalize OCR strings so comparisons are case/space insensitive."""

    return "".join(text.strip().upper().split())


@dataclass(slots=True)
class TextTokenizer:
    """Simple CTC tokenizer used for local decoding and metric utilities."""

    charset: str = DEFAULT_CHARSET
    blank_index: int = 0

    def __post_init__(self) -> None:
        if self.blank_index != 0:
            raise ValueError("blank_index must be 0 for CTC decoding.")
        if len(set(self.charset)) != len(self.charset):
            raise ValueError("charset contains duplicate characters.")
        self._idx2char = ["<blank>"] + list(self.charset)
        self._char2idx = {ch: idx for idx, ch in enumerate(self._idx2char)}

    @property
    def num_classes(self) -> int:
        return len(self._idx2char)

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        encoded: list[int] = []
        for ch in normalized:
            idx = self._char2idx.get(ch)
            if idx is not None and idx != self.blank_index:
                encoded.append(idx)
        return encoded

    def decode_indices(
        self,
        indices: Sequence[int],
        *,
        collapse_repeats: bool = True,
        remove_blank: bool = True,
    ) -> str:
        decoded: list[str] = []
        previous = None
        for raw_idx in indices:
            idx = int(raw_idx)
            if collapse_repeats and previous == idx:
                continue
            previous = idx
            if remove_blank and idx == self.blank_index:
                continue
            if 0 <= idx < len(self._idx2char):
                decoded.append(self._idx2char[idx])
        return "".join(decoded)

    def decode_logits(self, logits: torch.Tensor) -> list[str]:
        if logits.ndim != 3:
            raise ValueError("Expected logits with shape [B, T, C].")
        predictions = logits.argmax(dim=-1)
        return [self.decode_indices(row.tolist()) for row in predictions]

    def batch_decode(self, predictions: torch.Tensor | Sequence[Sequence[int]]) -> list[str]:
        if isinstance(predictions, torch.Tensor):
            if predictions.ndim == 3:
                return self.decode_logits(predictions)
            if predictions.ndim != 2:
                raise ValueError("Expected tensor with shape [B, T] or [B, T, C].")
            rows = predictions.tolist()
        else:
            rows = predictions
        return [self.decode_indices(row) for row in rows]


@lru_cache(maxsize=4096)
def _edit_distance_cached(ref: str, hyp: str) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    prev = list(range(len(hyp) + 1))
    for i, r_ch in enumerate(ref, start=1):
        curr = [i]
        for j, h_ch in enumerate(hyp, start=1):
            cost = 0 if r_ch == h_ch else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def edit_distance(ref: str, hyp: str) -> int:
    """Return the character-level edit distance between two strings."""

    return _edit_distance_cached(normalize_text(ref), normalize_text(hyp))


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute CER, a useful companion metric for exact-match accuracy."""

    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / max(len(reference), 1)


def exact_match(reference: str, hypothesis: str) -> float:
    """Return 1.0 only when two OCR strings are identical after normalization."""

    return float(normalize_text(reference) == normalize_text(hypothesis))


def batch_character_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length.")
    if not references:
        return 0.0
    total = 0.0
    for ref, hyp in zip(references, hypotheses):
        total += character_error_rate(ref, hyp)
    return total / len(references)


def batch_exact_match(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length.")
    if not references:
        return 0.0
    total = 0.0
    for ref, hyp in zip(references, hypotheses):
        total += exact_match(ref, hyp)
    return total / len(references)


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: dict[int, str],
) -> list[tuple[str, float]]:
    """Greedy CTC decoding with a simple confidence estimate.

    The confidence is computed from the maximum per-step probability of the
    surviving characters after collapsing repeats and removing blanks.
    """

    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()

    results: list[tuple[str, float]] = []

    for batch_idx in range(indices_np.shape[0]):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]

        pred_chars = []
        confidences = []
        time_idx = 0

        # groupby keeps consecutive repeats together so CTC blanks/repeats can be removed.
        from itertools import groupby

        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)

            if char_idx != 0:
                pred_chars.append(idx2char.get(char_idx, ""))
                group_probs = probs_b[time_idx : time_idx + group_size]
                confidences.append(float(group_probs.max()))

            time_idx += group_size

        pred_str = "".join(pred_chars)
        confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))

    return results
