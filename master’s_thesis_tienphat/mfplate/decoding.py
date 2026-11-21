"""CTC decoding for one track: greedy, prefix beam search, and syntax-constrained beam search.

All functions take the log-probabilities of ONE track, shape [T, C] (T = 16 time steps,
C = 37 classes, class 0 = CTC blank, classes 1..36 = "0-9A-Z"), as a NumPy array.

* greedy      best path, collapsed. Same output as src/utils/postprocess.py, which produced
              every accuracy number of the paper and the lock-box report.
* beam        CTC prefix beam search; returns the most probable strings.
* constrained prefix beam search that only grows prefixes of the plate patterns
              LLLNLNN (Mercosur) / LLLNNNN (Brazilian) and only accepts complete plates.
              This is the "SC-CTC" decoder of the roadmap.

Candidate strings are rescored with the exact CTC forward algorithm, so the probability
attached to every candidate is p(string | track), not a beam approximation.
"""
from dataclasses import dataclass, field
from itertools import groupby
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .settings import CHARS, PLATE_PATTERNS

BLANK = 0
IDX2CHAR: Dict[int, str] = {i + 1: c for i, c in enumerate(CHARS)}
CHAR2IDX: Dict[str, int] = {c: i for i, c in IDX2CHAR.items()}
DIGITS = [CHAR2IDX[c] for c in "0123456789"]
LETTERS = [CHAR2IDX[c] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
_IS_LETTER = np.zeros(len(CHARS) + 1, dtype=bool)
_IS_LETTER[LETTERS] = True
DECODERS = ("greedy", "beam", "constrained")
NEG_INF = -np.inf


def pattern_of(text: str) -> str:
    return "".join("L" if c.isalpha() else "N" if c.isdigit() else "?" for c in text)


def layout_of(text: str) -> Optional[str]:
    """'Mercosur' / 'Brazilian' if the string follows one of the plate patterns, else None."""
    return PLATE_PATTERNS.get(pattern_of(text)) if len(text) == 7 else None


def ids_of(text: str) -> List[int]:
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def text_of(ids: Sequence[int]) -> str:
    return "".join(IDX2CHAR.get(int(i), "") for i in ids)


# ---------------------------------------------------------------- greedy
def greedy(lp: np.ndarray) -> Tuple[str, List[float]]:
    """Best path decode. Returns (text, per-character confidence).

    A character's confidence is the highest probability among the time steps of its
    run in the best path, exactly as decode_with_confidence() computes it.
    """
    probs = np.exp(lp)
    path = probs.argmax(axis=1)
    best = probs.max(axis=1)
    chars, conf, t = [], [], 0
    for idx, run in groupby(path):
        n = len(list(run))
        if idx != BLANK:
            chars.append(IDX2CHAR.get(int(idx), ""))
            conf.append(float(np.max(best[t:t + n])))
        t += n
    return "".join(chars), conf


# ---------------------------------------------------------------- exact CTC scoring
def _extended(ids: Sequence[int]) -> List[int]:
    ext = [BLANK]
    for i in ids:
        ext += [int(i), BLANK]
    return ext


def ctc_log_prob(lp: np.ndarray, ids: Sequence[int]) -> float:
    """log p(ids | track) by the CTC forward algorithm (sum over all alignments)."""
    lp = np.asarray(lp, dtype=np.float64)
    T = lp.shape[0]
    ext = _extended(ids)
    S = len(ext)
    if len(ids) > T:
        return float(NEG_INF)
    skip = np.array([s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2] for s in range(S)])
    alpha = np.full(S, NEG_INF)
    alpha[0] = lp[0, ext[0]]
    if S > 1:
        alpha[1] = lp[0, ext[1]]
    for t in range(1, T):
        new = alpha.copy()
        new[1:] = np.logaddexp(new[1:], alpha[:-1])
        if S > 2:
            new[2:] = np.where(skip[2:], np.logaddexp(new[2:], alpha[:-2]), new[2:])
        alpha = new + lp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2]) if S > 1 else alpha[-1])


def aligned_char_conf(lp: np.ndarray, ids: Sequence[int]) -> Optional[List[float]]:
    """Per-character confidence of a given string through its best (Viterbi) alignment.

    For the greedy string this equals greedy()'s confidences, because the best path is
    also the best alignment of the string it collapses to. Returns None if the string
    cannot be aligned to T steps.
    """
    lp = np.asarray(lp, dtype=np.float64)
    T = lp.shape[0]
    ext = _extended(ids)
    S = len(ext)
    if not ids:
        return []
    delta = np.full((T, S), NEG_INF)
    back = np.zeros((T, S), dtype=np.int64)
    delta[0, 0] = lp[0, ext[0]]
    delta[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(delta[t - 1, s], s)]
            if s >= 1:
                cands.append((delta[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                cands.append((delta[t - 1, s - 2], s - 2))
            val, arg = max(cands, key=lambda c: c[0])
            delta[t, s] = val + lp[t, ext[s]]
            back[t, s] = arg
    end = S - 1 if delta[T - 1, S - 1] >= delta[T - 1, S - 2] else S - 2
    if not np.isfinite(delta[T - 1, end]):
        return None
    states = [end]
    for t in range(T - 1, 0, -1):
        states.append(int(back[t, states[-1]]))
    states.reverse()
    probs = np.exp(lp)
    conf = [0.0] * len(ids)
    for t, s in enumerate(states):
        if s % 2 == 1:
            k = s // 2
            conf[k] = max(conf[k], float(probs[t, ext[s]]))
    return conf


# ---------------------------------------------------------------- prefix beam search
def _prefix_ok(prefix: Tuple[int, ...]) -> bool:
    """Can this prefix still grow into one of the plate patterns?"""
    n = len(prefix)
    if n > 7:
        return False
    kinds = "".join("L" if _IS_LETTER[i] else "N" for i in prefix)
    return any(p.startswith(kinds) for p in PLATE_PATTERNS)


def _complete_ok(prefix: Tuple[int, ...]) -> bool:
    return len(prefix) == 7 and _prefix_ok(prefix)


def prefix_beam_search(lp: np.ndarray, beam: int = 10, top_k: int = 3,
                       constrained: bool = False, per_step: int = 8) -> List[Tuple[str, float]]:
    """CTC prefix beam search. Returns up to top_k (text, log p(text | track)), best first.

    per_step: characters tried at each time step (the most probable ones). With the
    constraint, the most probable letters AND digits are both tried, so a pattern can
    always be completed even when the model puts low mass on the required class.
    """
    lp = np.asarray(lp, dtype=np.float64)
    T, C = lp.shape
    beams: Dict[Tuple[int, ...], List[float]] = {(): [0.0, NEG_INF]}  # prefix -> [pb, pnb]
    for t in range(T):
        row = lp[t]
        order = np.argsort(-row[1:]) + 1
        if constrained:
            letters = [c for c in order if _IS_LETTER[c]][:per_step]
            digits = [c for c in order if not _IS_LETTER[c]][:per_step]
            cand = letters + digits
        else:
            cand = list(order[:per_step])
        nxt: Dict[Tuple[int, ...], List[float]] = {}

        def acc(prefix, which, value):
            entry = nxt.setdefault(prefix, [NEG_INF, NEG_INF])
            entry[which] = np.logaddexp(entry[which], value)

        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            acc(prefix, 0, total + row[BLANK])                 # blank: prefix unchanged
            last = prefix[-1] if prefix else None
            if last is not None:
                acc(prefix, 1, pnb + row[last])                # repeated char collapses
            for c in cand:
                c = int(c)
                new = prefix + (c,)
                if constrained and not _prefix_ok(new):
                    continue
                if c == last:
                    acc(new, 1, pb + row[c])                   # needs a blank in between
                else:
                    acc(new, 1, total + row[c])
        ranked = sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))
        if constrained:
            # Keep the best `beam` prefixes of EACH length. Otherwise, when the model reads
            # only 6 characters with confidence, variants of the 6-character reading fill
            # the beam and no complete plate survives to the last step.
            per_len: Dict[int, int] = {}
            kept = []
            for prefix, v in ranked:
                if per_len.get(len(prefix), 0) < beam:
                    per_len[len(prefix)] = per_len.get(len(prefix), 0) + 1
                    kept.append((prefix, v))
            beams = dict(kept)
        else:
            beams = dict(ranked[:beam])
    finals = [(p, float(np.logaddexp(*v))) for p, v in beams.items()]
    if constrained:
        finals = [f for f in finals if _complete_ok(f[0])]
    # Exact rescoring: beam scores are lower bounds once prefixes were pruned
    rescored = sorted(((text_of(p), ctc_log_prob(lp, p)) for p, _ in finals),
                      key=lambda x: -x[1])
    out, seen = [], set()
    for text, score in rescored:
        if text not in seen:
            seen.add(text)
            out.append((text, score))
    return out[:top_k]


# ---------------------------------------------------------------- one entry point
@dataclass
class Decoded:
    text: str
    char_conf: List[float]
    seq_logprob: float
    candidates: List[Tuple[str, float]] = field(default_factory=list)  # (text, probability)
    decoder: str = "greedy"

    @property
    def mean_char(self) -> float:
        return float(np.mean(self.char_conf)) if self.char_conf else 0.0

    @property
    def min_char(self) -> float:
        return float(np.min(self.char_conf)) if self.char_conf else 0.0

    @property
    def seq_prob(self) -> float:
        return float(np.exp(self.seq_logprob))

    def score(self, name: str) -> float:
        """Confidence score used for triage: 'mean_char', 'min_char' or 'seq_prob'."""
        return {"mean_char": self.mean_char, "min_char": self.min_char,
                "seq_prob": self.seq_prob}[name]


def decode(lp: np.ndarray, decoder: str = "greedy", top_k: int = 3, beam: int = 10) -> Decoded:
    """Decode one track. The first candidate is always the returned text."""
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
    lp = np.asarray(lp, dtype=np.float64)
    if decoder == "greedy":
        text, conf = greedy(lp)
        alternatives = prefix_beam_search(lp, beam=beam, top_k=top_k + 1)
    else:
        found = prefix_beam_search(lp, beam=beam, top_k=top_k + 1,
                                   constrained=decoder == "constrained")
        if not found:  # keep a defined answer anyway
            found = prefix_beam_search(lp, beam=beam, top_k=top_k + 1)
        text = found[0][0] if found else greedy(lp)[0]
        conf = aligned_char_conf(lp, ids_of(text)) or []
        alternatives = found
    seq_lp = ctc_log_prob(lp, ids_of(text)) if text else float(
        np.sum(lp[:, BLANK]))
    cands = [(text, float(np.exp(seq_lp)))]
    cands += [(t, float(np.exp(s))) for t, s in alternatives if t != text]
    return Decoded(text=text, char_conf=[float(c) for c in conf], seq_logprob=float(seq_lp),
                   candidates=cands[:top_k], decoder=decoder)
