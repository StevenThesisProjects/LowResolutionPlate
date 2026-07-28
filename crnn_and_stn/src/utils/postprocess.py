"""Post-processing helpers for CTC OCR decoding.

The original project only needed greedy decoding with a confidence estimate.
This version keeps that behavior and also adds reusable text utilities so
training/validation code can measure exact-match and edit-distance style
metrics consistently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch

DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

# Every one of the 20,000 training labels is exactly 7 characters and matches one
# of these two Brazilian layouts (Mercosur `LLLNLNN` and the older `LLLNNNN`).
# `L` = letter, `N` = digit. Six of the seven positions are therefore locked to a
# character class, which is what makes constrained decoding worth doing here.
DEFAULT_PLATE_LAYOUTS = ("LLLNLNN", "LLLNNNN")


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


@dataclass(frozen=True)
class PlateLayout:
    """Per-position character classes shared by every plate layout in the data.

    Multiple layouts are merged into one position-wise union, so a single decode
    pass covers them all: `LLLNLNN` and `LLLNNNN` differ only at position 5, so
    the union is `[A-Z]{3}[0-9][A-Z0-9][0-9]{2}`. Decoding once against the union
    is equivalent to decoding once per layout and keeping the better score, but
    costs a single pass.
    """

    patterns: tuple[str, ...] = DEFAULT_PLATE_LAYOUTS

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("PlateLayout needs at least one pattern.")
        lengths = {len(p) for p in self.patterns}
        if len(lengths) != 1:
            raise ValueError(f"All layouts must have the same length, got {sorted(lengths)}.")
        for pattern in self.patterns:
            unknown = set(pattern) - {"L", "N"}
            if unknown:
                raise ValueError(f"Layout {pattern!r} has unknown class symbols {sorted(unknown)}.")

    @property
    def length(self) -> int:
        return len(self.patterns[0])

    def allowed_chars(self, position: int) -> str:
        """Characters permitted at `position`, unioned across every layout."""
        classes = {pattern[position] for pattern in self.patterns}
        chars = ""
        if "L" in classes:
            chars += LETTERS
        if "N" in classes:
            chars += DIGITS
        return chars

    def allowed_indices(self, char2idx: dict[str, int]) -> list[np.ndarray]:
        """Vocabulary indices permitted at each position, for fast lookup."""
        return [
            np.array(
                sorted(char2idx[c] for c in self.allowed_chars(pos) if c in char2idx),
                dtype=np.int64,
            )
            for pos in range(self.length)
        ]

    @classmethod
    def from_spec(cls, spec: str) -> "PlateLayout":
        """Build from a comma-separated spec such as ``"LLLNLNN,LLLNNNN"``."""
        patterns = tuple(p.strip().upper() for p in spec.split(",") if p.strip())
        return cls(patterns=patterns)


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def constrained_beam_decode(
    log_probs: torch.Tensor,
    idx2char: dict[int, str],
    layout: PlateLayout | None = None,
    beam_width: int = 16,
    char_topk: int = 8,
) -> list[tuple[str, float]]:
    """CTC prefix beam search restricted to the dataset's plate layouts.

    Greedy decoding is free to emit the wrong length or a digit where the layout
    guarantees a letter, and at ~6.6 pixels per character almost every confusion
    (0/O, 1/I, 8/B, 5/S, 2/Z) crosses exactly that letter/digit boundary. Fixing
    the length to 7 and locking six of the seven positions to a character class
    removes that entire error family without touching the model.

    Args:
        log_probs: `[B, T, C]` log-softmax output (the model's native format).
        idx2char: CTC index -> character, blank at index 0.
        layout: position constraints; defaults to the two Brazilian layouts.
        beam_width: prefixes kept per timestep.
        char_topk: candidate characters considered per timestep and position.
            Pruning to the top-k allowed characters keeps the search near-exact
            while bounding the cost, since the tail of the distribution never
            wins a 7-way product.

    Returns:
        `(text, confidence)` per sample. Confidence is the per-character
        geometric mean probability of the winning sequence, which keeps it on
        the same 0-1 scale as `decode_with_confidence`.
    """

    layout = layout or PlateLayout()
    char2idx = {char: idx for idx, char in idx2char.items()}
    allowed = layout.allowed_indices(char2idx)
    target_len = layout.length
    blank = 0

    lp_all = log_probs.detach().float().cpu().numpy()
    greedy_fallback = decode_with_confidence(log_probs, idx2char)
    results: list[tuple[str, float]] = []

    for sample_idx in range(lp_all.shape[0]):
        lp = lp_all[sample_idx]  # [T, C]
        num_steps = lp.shape[0]

        # Candidate characters per (timestep, position): the top-k highest-scoring
        # entries among the ones the layout permits at that position.
        candidates: list[list[np.ndarray]] = []
        for t in range(num_steps):
            row = lp[t]
            per_position = []
            for pos in range(target_len):
                pool = allowed[pos]
                if 0 < char_topk < pool.size:
                    order = np.argpartition(-row[pool], char_topk - 1)[:char_topk]
                    per_position.append(pool[order])
                else:
                    per_position.append(pool)
            candidates.append(per_position)

        # prefix -> [log P(prefix, ends in blank), log P(prefix, ends in a char)]
        beams: dict[tuple[int, ...], list[float]] = {(): [0.0, -math.inf]}

        for t in range(num_steps):
            row = lp[t]
            steps_left = num_steps - t
            nxt: dict[tuple[int, ...], list[float]] = {}

            for prefix, (p_blank, p_nonblank) in beams.items():
                p_total = _logaddexp(p_blank, p_nonblank)
                missing = target_len - len(prefix)
                # A prefix that can no longer reach the required length is dead;
                # dropping it early keeps the beam full of viable candidates.
                if missing > steps_left:
                    continue

                entry = nxt.setdefault(prefix, [-math.inf, -math.inf])
                entry[0] = _logaddexp(entry[0], p_total + row[blank])
                if prefix:
                    # Repeating the last character collapses back onto the same
                    # prefix under CTC, so it extends the non-blank mass only.
                    entry[1] = _logaddexp(entry[1], p_nonblank + row[prefix[-1]])

                if missing <= 0:
                    continue
                last = prefix[-1] if prefix else -1
                for char_idx in candidates[t][len(prefix)]:
                    char_idx = int(char_idx)
                    extended = prefix + (char_idx,)
                    ext_entry = nxt.setdefault(extended, [-math.inf, -math.inf])
                    # A repeat needs an intervening blank, so it may only grow
                    # from the blank-terminated mass.
                    source = p_blank if char_idx == last else p_total
                    ext_entry[1] = _logaddexp(ext_entry[1], source + row[char_idx])

            if not nxt:
                break
            beams = dict(
                sorted(
                    nxt.items(),
                    key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                    reverse=True,
                )[:beam_width]
            )

        complete = [
            (prefix, _logaddexp(scores[0], scores[1]))
            for prefix, scores in beams.items()
            if len(prefix) == target_len
        ]
        if not complete:
            # Nothing reachable under the constraint (e.g. T too short); the
            # unconstrained prediction is still better than an empty string.
            results.append(greedy_fallback[sample_idx])
            continue

        best_prefix, best_score = max(complete, key=lambda item: item[1])
        text = "".join(idx2char.get(idx, "") for idx in best_prefix)
        confidence = float(math.exp(best_score / target_len))
        results.append((text, min(confidence, 1.0)))

    return results


def decode_batch(
    log_probs: torch.Tensor,
    idx2char: dict[int, str],
    mode: str = "greedy",
    layout: PlateLayout | None = None,
    beam_width: int = 16,
) -> list[tuple[str, float]]:
    """Dispatch to greedy or layout-constrained decoding."""

    if mode == "greedy":
        return decode_with_confidence(log_probs, idx2char)
    if mode == "constrained":
        return constrained_beam_decode(
            log_probs, idx2char, layout=layout, beam_width=beam_width
        )
    raise ValueError(f"Unknown decode mode: {mode!r} (expected 'greedy' or 'constrained')")
