"""Decoding helpers the API needs on top of `src.utils.postprocess`.

`src/` is training code and stays untouched, so what the API adds lives here:

* `topk_constrained` — the same constrained CTC prefix beam search as
  `constrained_beam_decode`, but keeping the runners-up. They are the
  suggestions a reviewer picks from. Rank 1 is identical to
  `constrained_beam_decode` (checked in tests/test_decoding.py).
* `char_confidences` — per-character confidence via CTC forced alignment,
  which works for both greedy and constrained output.
"""

from __future__ import annotations

import math

import numpy as np

from src.utils.postprocess import DIGITS, LETTERS, PlateLayout


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def topk_constrained(
    lp: np.ndarray,
    idx2char: dict[int, str],
    layout: PlateLayout,
    beam_width: int = 16,
    char_topk: int = 8,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Top-k full-length plates for one sample. `lp` is `[T, C]` log-probs.

    Returns `(text, confidence)` sorted best-first; confidence is the
    per-character geometric mean probability, as in `constrained_beam_decode`.
    May be empty when no prefix can reach the layout length.
    """

    char2idx = {char: idx for idx, char in idx2char.items()}
    allowed = layout.allowed_indices(char2idx)
    target_len = layout.length
    blank = 0
    num_steps = lp.shape[0]

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
            if missing > steps_left:
                continue
            entry = nxt.setdefault(prefix, [-math.inf, -math.inf])
            entry[0] = _logaddexp(entry[0], p_total + row[blank])
            if prefix:
                entry[1] = _logaddexp(entry[1], p_nonblank + row[prefix[-1]])
            if missing <= 0:
                continue
            last = prefix[-1] if prefix else -1
            for char_idx in candidates[t][len(prefix)]:
                char_idx = int(char_idx)
                extended = prefix + (char_idx,)
                ext_entry = nxt.setdefault(extended, [-math.inf, -math.inf])
                source = p_blank if char_idx == last else p_total
                ext_entry[1] = _logaddexp(ext_entry[1], source + row[char_idx])
        if not nxt:
            break
        beams = dict(
            sorted(nxt.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]), reverse=True)[:beam_width]
        )

    complete = sorted(
        (
            (prefix, _logaddexp(scores[0], scores[1]))
            for prefix, scores in beams.items()
            if len(prefix) == target_len
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        ("".join(idx2char.get(idx, "") for idx in prefix), min(math.exp(score / target_len), 1.0))
        for prefix, score in complete[:top_k]
    ]


def char_confidences(lp: np.ndarray, text: str, char2idx: dict[str, int]) -> list[float]:
    """Per-character probability of `text` under the CTC output `lp` `[T, C]`.

    Viterbi-aligns `text` to the timesteps (standard CTC topology with blanks
    between characters), then takes, for each character, the highest
    probability among the frames aligned to it. Falls back to zeros when the
    text cannot be aligned (longer than T allows).
    """

    labels = [char2idx[c] for c in text if c in char2idx]
    if not labels:
        return []
    num_steps = lp.shape[0]
    states = [0]
    for label in labels:
        states += [label, 0]
    num_states = len(states)
    neg = -math.inf

    score = np.full((num_steps, num_states), neg)
    back = np.zeros((num_steps, num_states), dtype=np.int64)
    score[0, 0] = lp[0, states[0]]
    if num_states > 1:
        score[0, 1] = lp[0, states[1]]
    for t in range(1, num_steps):
        for s in range(num_states):
            best, arg = score[t - 1, s], s
            if s >= 1 and score[t - 1, s - 1] > best:
                best, arg = score[t - 1, s - 1], s - 1
            # Skip over a blank only between two different characters.
            if s >= 2 and states[s] != 0 and states[s] != states[s - 2] and score[t - 1, s - 2] > best:
                best, arg = score[t - 1, s - 2], s - 2
            if best > neg:
                score[t, s] = best + lp[t, states[s]]
                back[t, s] = arg

    end = num_states - 1 if score[-1, -1] >= score[-1, -2] else num_states - 2
    if score[-1, end] == neg:
        return [0.0] * len(labels)

    path = [end]
    for t in range(num_steps - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()

    conf = [0.0] * len(labels)
    for t, s in enumerate(path):
        if s % 2 == 1:
            conf[s // 2] = max(conf[s // 2], float(math.exp(lp[t, states[s]])))
    return conf


def match_layout(text: str, layout: PlateLayout) -> str | None:
    """First layout pattern (`L` letter / `N` digit) that `text` satisfies."""

    for pattern in layout.patterns:
        if len(pattern) == len(text) and all(
            (c in LETTERS) if p == "L" else (c in DIGITS) for p, c in zip(pattern, text)
        ):
            return pattern
    return None
