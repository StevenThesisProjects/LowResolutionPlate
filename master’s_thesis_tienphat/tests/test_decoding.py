"""Decoder unit tests (roadmap week 1: tests/test_decoder.py, at least 10 cases)."""
import itertools

import numpy as np
import pytest
import torch

from mfplate.decoding import (BLANK, CHAR2IDX, aligned_char_conf, ctc_log_prob, decode, greedy,
                              ids_of, layout_of, pattern_of, prefix_beam_search)

C = 37


def peaked(path, p=0.9, T=None):
    """Log-probs whose argmax follows `path` (class ids, 0 = blank)."""
    T = T or len(path)
    lp = np.full((T, C), np.log((1 - p) / (C - 1)))
    for t, k in enumerate(path):
        lp[t, k] = np.log(p)
    return lp


def chars(text):
    return [CHAR2IDX[c] for c in text]


def path_for(text, T=16):
    """A best path for `text`: each char once, blanks between and after."""
    path = []
    for c in chars(text):
        path += [c, BLANK]
    return (path + [BLANK] * T)[:T]


# --- greedy -------------------------------------------------------------------------
def test_greedy_collapses_repeats_and_drops_blanks():
    a, b = CHAR2IDX["A"], CHAR2IDX["B"]
    text, conf = greedy(peaked([a, a, BLANK, b, b, BLANK]))
    assert text == "AB" and len(conf) == 2


def test_greedy_keeps_double_letter_separated_by_blank():
    a = CHAR2IDX["A"]
    assert greedy(peaked([a, BLANK, a, BLANK]))[0] == "AA"
    assert greedy(peaked([a, a, a, BLANK]))[0] == "A"


def test_greedy_all_blank_is_empty():
    text, conf = greedy(peaked([BLANK] * 16))
    assert text == "" and conf == []


def test_greedy_matches_training_decoder():
    from src.utils.postprocess import decode_with_confidence
    from mfplate.decoding import IDX2CHAR
    rng = np.random.default_rng(0)
    for _ in range(50):
        logits = rng.normal(size=(16, C)) * 3
        lp = torch.log_softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
        ref_text, ref_conf = decode_with_confidence(lp.unsqueeze(0), IDX2CHAR)[0]
        text, conf = greedy(lp.numpy())
        assert text == ref_text
        assert abs((np.mean(conf) if conf else 0.0) - ref_conf) < 1e-6


# --- exact CTC probability -----------------------------------------------------------
def _collapse(path):
    return tuple(k for k, _ in itertools.groupby(path) if k != BLANK)


def test_ctc_log_prob_equals_brute_force():
    rng = np.random.default_rng(1)
    K, T = 3, 5                                    # tiny alphabet: blank + 2 symbols
    lp = np.log(rng.dirichlet(np.ones(K), size=T))
    totals = {}
    for path in itertools.product(range(K), repeat=T):
        p = np.exp(sum(lp[t, k] for t, k in enumerate(path)))
        totals[_collapse(path)] = totals.get(_collapse(path), 0.0) + p
    for label, p in totals.items():
        assert np.isclose(np.exp(ctc_log_prob(lp, label)), p, rtol=1e-9)
    assert np.isclose(sum(totals.values()), 1.0)


def test_ctc_log_prob_impossible_label():
    lp = peaked([BLANK] * 4)
    assert ctc_log_prob(lp, chars("ABCDEFG")) == -np.inf   # 7 labels cannot fit in 4 steps


# --- beam search -----------------------------------------------------------------------
def test_beam_top1_equals_greedy_when_peaked():
    lp = peaked(path_for("ABC1D23"), p=0.95)
    out = prefix_beam_search(lp, beam=10, top_k=3)
    assert out[0][0] == "ABC1D23"
    assert [s for _, s in out] == sorted([s for _, s in out], reverse=True)
    assert len({t for t, _ in out}) == len(out)


def test_beam_finds_string_whose_paths_add_up():
    # The best single path reads "" (blank everywhere at 0.4) but "A" collects mass from
    # many paths (0.6^... spread): beam search, unlike greedy, sums over alignments.
    a = CHAR2IDX["A"]
    lp = np.full((3, C), -30.0)
    for t in range(3):
        lp[t, BLANK] = np.log(0.4)
        lp[t, a] = np.log(0.35)
        lp[t, CHAR2IDX["B"]] = np.log(0.25)
    lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    assert greedy(lp)[0] == ""
    best = prefix_beam_search(lp, beam=16, top_k=1)[0]
    assert best[0] == "A"


def test_constrained_returns_only_plate_patterns():
    rng = np.random.default_rng(2)
    for _ in range(20):
        lp = np.log(rng.dirichlet(np.ones(C) * 0.3, size=16))
        for text, _ in prefix_beam_search(lp, beam=10, top_k=3, constrained=True):
            assert layout_of(text) is not None


def test_constrained_repairs_a_digit_in_a_letter_slot():
    # Greedy reads "AB8" + "1C23" (8 where a letter must be). The constraint must pick the
    # most probable letter for that slot, here "B" (0.3) over others.
    lp = peaked(path_for("AB81C23"), p=0.6)
    slot = 4                                        # time step of the 3rd character
    lp[slot] = np.log(np.full(C, 0.1 / (C - 3)))
    lp[slot, CHAR2IDX["8"]] = np.log(0.6)
    lp[slot, CHAR2IDX["B"]] = np.log(0.3)
    lp[slot] = lp[slot] - np.logaddexp.reduce(lp[slot])
    assert greedy(lp)[0] == "AB81C23"
    assert decode(lp, "constrained").text == "ABB1C23"
    assert layout_of("ABB1C23") == "Mercosur"


def test_constrained_completes_a_six_character_reading():
    # The model reads 6 characters with confidence ("ABC123"); a 7th character only has
    # small mass. A narrow beam keeps only 6-character prefixes; decode() must widen it
    # and still return a complete plate (found on val 999, seed 7, track 643).
    lp = peaked(path_for("ABC123"), p=0.97)
    out = decode(lp, "constrained", top_k=3, beam=2)
    assert layout_of(out.text) is not None and len(out.text) == 7


# --- per-character confidence ------------------------------------------------------------
def test_aligned_conf_equals_greedy_conf_for_greedy_string():
    rng = np.random.default_rng(3)
    for _ in range(30):
        lp = np.log(rng.dirichlet(np.ones(C) * 0.2, size=16))
        text, conf = greedy(lp)
        if not text:
            continue
        assert np.allclose(aligned_char_conf(lp, ids_of(text)), conf)


def test_decode_first_candidate_is_the_answer():
    rng = np.random.default_rng(4)
    for name in ("greedy", "beam", "constrained"):
        lp = np.log(rng.dirichlet(np.ones(C) * 0.3, size=16))
        d = decode(lp, name, top_k=3)
        assert d.candidates[0][0] == d.text
        assert np.isclose(d.candidates[0][1], d.seq_prob)
        assert len(d.char_conf) == len(d.text)
        assert 0.0 <= d.seq_prob <= 1.0


def test_decode_rejects_unknown_decoder():
    with pytest.raises(ValueError):
        decode(peaked([BLANK] * 16), "viterbi")


# --- plate syntax ------------------------------------------------------------------------
@pytest.mark.parametrize("text,layout", [
    ("ABC1D23", "Mercosur"), ("ABC1234", "Brazilian"), ("ABC12D3", None),   # motorcycle-like
    ("AB1C234", None), ("ABC123", None), ("ABC12345", None), ("", None)])
def test_layout_of(text, layout):
    assert layout_of(text) == layout


def test_pattern_of():
    assert pattern_of("GGB8A08") == "LLLNLNN"
