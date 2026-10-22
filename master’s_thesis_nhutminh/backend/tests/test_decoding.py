"""The API's decoding helpers must agree with the training code in src/."""

import numpy as np
import torch

from configs.config import Config
from src.utils.postprocess import PlateLayout, constrained_beam_decode

from backend.app.services.decoding import char_confidences, match_layout, topk_constrained

CONFIG = Config()
LAYOUT = PlateLayout()


def _random_log_probs(seed: int, batch: int = 64, steps: int = 16) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    # Peaky logits, like a trained CTC model.
    logits = torch.randn(batch, steps, CONFIG.NUM_CLASSES, generator=gen) * 4
    return logits.log_softmax(-1)


def test_rank1_matches_constrained_beam_decode():
    for seed in range(3):
        lp = _random_log_probs(seed)
        reference = constrained_beam_decode(lp, CONFIG.IDX2CHAR, layout=LAYOUT, beam_width=16)
        arr = lp.numpy()
        for i, (text, conf) in enumerate(reference):
            top = topk_constrained(arr[i], CONFIG.IDX2CHAR, LAYOUT, beam_width=16, top_k=5)
            assert top, "constrained search found no full-length plate"
            assert top[0][0] == text
            assert abs(top[0][1] - conf) < 1e-6
            assert [c for _, c in top] == sorted((c for _, c in top), reverse=True)
            assert all(match_layout(t, LAYOUT) for t, _ in top)


def test_char_confidences_on_clean_alignment():
    text = "ABC1D23"
    steps = 16
    lp = np.full((steps, CONFIG.NUM_CLASSES), np.log(1e-4))
    lp[:, 0] = np.log(0.99)  # blank everywhere …
    for k, ch in enumerate(text):  # … except one confident frame per char
        lp[2 * k + 1, :] = np.log(1e-4)
        lp[2 * k + 1, CONFIG.CHAR2IDX[ch]] = np.log(0.9)
    conf = char_confidences(lp, text, CONFIG.CHAR2IDX)
    assert len(conf) == len(text)
    assert all(abs(c - 0.9) < 1e-6 for c in conf)


def test_match_layout():
    assert match_layout("ABC1D23", LAYOUT) == "LLLNLNN"
    assert match_layout("ABC1234", LAYOUT) in ("LLLNLNN", "LLLNNNN")
    assert match_layout("AB12345", LAYOUT) is None
