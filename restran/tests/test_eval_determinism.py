"""Checks requested in the review (ISSUE #9, A1/A2/A4): evaluation is deterministic and the
training-only regularisers (frame dropout, frame mask) never touch validation or test.

    python tests/test_eval_determinism.py        # no pytest needed
    python -m pytest tests/                       # also works
"""
import io
import contextlib
import json
import os
import sys
import types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs.config import Config  # noqa: E402
from src.models.components import AttentionFusion  # noqa: E402
from src.models.restran import ResTranOCR  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


def _batch(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(b, 5, 3, 32, 128, generator=g)
    targets = torch.randint(1, 37, (b * 7,), generator=g)
    lengths = torch.full((b,), 7, dtype=torch.long)
    labels = tuple("ABC1D23" for _ in range(b))
    tids = tuple(f"track_{i}" for i in range(b))
    return images, targets, lengths, labels, tids, None, None, None, None, None


class _Loader:
    """Minimal stand-in for a DataLoader: iterable, sized, with a .dataset.layouts."""

    def __init__(self, batches):
        self.batches = batches
        self.dataset = types.SimpleNamespace(layouts={})

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _trainer(**overrides):
    cfg = Config()
    cfg.DEVICE = torch.device("cpu")
    cfg.EPOCHS = 1
    cfg.OUTPUT_DIR = os.path.join(REPO, "results", "_test_tmp")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    model = ResTranOCR(num_classes=cfg.NUM_CLASSES, frame_dropout=cfg.FRAME_DROPOUT)
    loader = _Loader([_batch(seed=1)])
    with contextlib.redirect_stdout(io.StringIO()):
        return Trainer(model, loader, _Loader([_batch(seed=2)]), cfg, cfg.IDX2CHAR)


def test_eval_forward_is_deterministic():
    torch.manual_seed(0)
    model = ResTranOCR(num_classes=37, frame_dropout=0.5).eval()
    x = _batch()[0]
    with torch.no_grad():
        a, fa = model(x, return_frame_logits=True)
        b, fb = model(x, return_frame_logits=True)
    assert torch.equal(a, b) and torch.equal(fa, fb)


def test_frame_dropout_runs_only_in_training():
    model = ResTranOCR(num_classes=37, frame_dropout=0.5)

    def forbidden(*_, **__):
        raise AssertionError("_drop_frames called")

    model._drop_frames = forbidden
    x = _batch()[0]
    model.eval()
    with torch.no_grad():
        model(x)                      # must not call it
    model.train()
    try:
        model(x)
    except AssertionError:
        return                        # wired into the training forward, as intended
    raise AssertionError("frame dropout is never applied in training")


def test_frame_mask_never_reaches_validation():
    trainer = _trainer(FRAME_MASK=1.0, FRAME_DROPOUT=0.5)

    def forbidden(*_, **__):
        raise AssertionError("_mask_frames called")

    trainer._mask_frames = forbidden
    m1, _ = trainer.validate()
    m2, _ = trainer.validate()
    assert m1["loss"] == m2["loss"] and m1["acc"] == m2["acc"]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.train_one_epoch()
    except AssertionError:
        return
    raise AssertionError("frame mask is never applied in training")


def test_luma_commutes_with_normalisation():
    import cv2
    from src.data.dataset import MultiFrameDataset
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 128, 3), dtype=np.uint8)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    luma = MultiFrameDataset._to_luma((t - 0.5) / 0.5)
    ref = (torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0) - 0.5) / 0.5
    assert torch.allclose(luma[0], luma[1]) and torch.allclose(luma[1], luma[2])
    assert (luma[0] - ref).abs().max() <= 2.0 / 255 + 1e-6      # within uint8 rounding


def test_attention_fusion_frame_count():
    torch.manual_seed(0)
    fusion = AttentionFusion(64).eval()
    x5 = torch.randn(2 * 5, 64, 1, 16)
    assert fusion(x5).shape == (2, 64, 1, 16)                    # default stays 5 frames
    assert torch.equal(fusion(x5), fusion(x5, num_frames=5))
    assert fusion(torch.randn(2 * 3, 64, 1, 16), num_frames=3).shape == (2, 64, 1, 16)


def test_statistics_helpers_match_scipy():
    try:
        from scipy import stats
    except ImportError:
        print("   (scipy không có — bỏ qua so sánh)")
        return
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import eval_lockbox as ev
    x, y = [77.18, 77.78, 77.58, 77.78], [79.28, 78.48, 78.88, 79.18]
    t, df, p = ev.welch(x, y)
    ref = stats.ttest_ind(y, x, equal_var=False)
    assert abs(t - ref.statistic) < 1e-9 and abs(p - ref.pvalue) < 1e-9
    a = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1] * 20, bool)
    b = np.roll(a, 3)
    gain, loss, p = ev.mcnemar_exact(a, b)
    assert abs(p - stats.binomtest(gain, gain + loss, 0.5).pvalue) < 1e-9


def test_exclude_tracks_removed_from_training():
    from src.data.dataset import MultiFrameDataset
    cfg = Config()
    if not os.path.isdir(os.path.join(REPO, cfg.DATA_ROOT)):
        print("   (không có dữ liệu — bỏ qua)")
        return
    os.chdir(REPO)
    with contextlib.redirect_stdout(io.StringIO()):
        full = MultiFrameDataset(root_dir=cfg.DATA_ROOT, mode="train", split_ratio=0.9,
                                 char2idx=cfg.CHAR2IDX, val_split_file="splits/val_tracks.json",
                                 no_synthetic=True)
    some = sorted({s["track_id"] for s in full.samples})[:25]
    with contextlib.redirect_stdout(io.StringIO()):
        cut = MultiFrameDataset(root_dir=cfg.DATA_ROOT, mode="train", split_ratio=0.9,
                                char2idx=cfg.CHAR2IDX, val_split_file="splits/val_tracks.json",
                                no_synthetic=True, exclude_tracks=some)
    left = {s["track_id"] for s in cut.samples}
    assert not (left & set(some)) and len(cut.samples) == len(full.samples) - len(some)
    with open("splits/val_tracks.json") as fh:
        assert not (left & set(json.load(fh)))                   # val stays out of train



def _frozen_mask_frames(frame_mask, images, skip_rows=None):
    """_mask_frames exactly as the lock-box checkpoints were trained with (before N1)."""
    b, f, _, h, w = images.shape
    dev = images.device
    chosen = torch.rand(b, f, device=dev) < frame_mask
    if skip_rows is not None:
        chosen &= skip_rows.to(dev).view(b, 1) < 0.5
    vertical = torch.rand(b, f, device=dev) < 0.5
    vw = torch.randint(max(1, w // 16), max(2, w // 8) + 1, (b, f), device=dev)
    vx = (torch.rand(b, f, device=dev) * (w - vw + 1)).long()
    hh = torch.randint(max(1, h // 8), max(2, h // 4) + 1, (b, f), device=dev)
    hy = (torch.rand(b, f, device=dev) * (h - hh + 1)).long()
    xs = torch.arange(w, device=dev).view(1, 1, 1, w)
    ys = torch.arange(h, device=dev).view(1, 1, h, 1)
    vband = (xs >= vx[..., None, None]) & (xs < (vx + vw)[..., None, None])
    hband = (ys >= hy[..., None, None]) & (ys < (hy + hh)[..., None, None])
    band = torch.where(vertical[..., None, None], vband.expand(b, f, h, w),
                       hband.expand(b, f, h, w))
    band = band & chosen[..., None, None]
    return images.masked_fill(band.unsqueeze(2), 0.0)


def _mask(mode, orient, p, images, skip_rows=None):
    me = types.SimpleNamespace(frame_mask=p, frame_mask_mode=mode, frame_mask_orient=orient)
    return Trainer._mask_frames(me, images, skip_rows)


def _masked(out):
    """[B, F, C, H, W] -> [B, F, H, W] bool, True where the mask wrote its 0."""
    return (out == 0).all(dim=2)


def test_independent_mask_unchanged_from_frozen_code():
    images = torch.randn(64, 5, 3, 32, 128) + 3.0              # no natural zeros
    skip = (torch.arange(64) % 3 == 0).float()
    for p, rows in ((0.2, None), (1.0, skip)):
        torch.manual_seed(123)
        ref = _frozen_mask_frames(p, images, rows)
        torch.manual_seed(123)
        new = _mask("independent", "mixed", p, images, rows)
        assert torch.equal(ref, new)


def test_tube_mask_same_band_on_every_frame():
    images = torch.randn(256, 5, 3, 32, 128) + 3.0
    for orient in ("mixed", "vertical"):
        torch.manual_seed(0)
        m = _masked(_mask("tube", orient, 1.0, images))
        assert m.flatten(2).any(-1).all()                      # every frame of every track
        assert (m == m[:, :1]).all()                           # identical across frames
        if orient == "vertical":
            assert (m == m[:, :, :1]).all()                    # full-height columns only


def test_complementary_masks_are_disjoint_vertical_bands():
    images = torch.randn(256, 5, 3, 32, 128) + 3.0
    torch.manual_seed(0)
    m = _masked(_mask("complementary", "vertical", 1.0, images))
    assert m.flatten(2).any(-1).all()                          # every frame is masked
    assert (m == m[:, :, :1]).all()                            # full-height columns only
    cols = m[:, :, 0, :]                                       # [B, F, W]
    assert cols.sum(1).max() <= 1                              # a column is hidden in <= 1 frame
    width = cols.sum(-1)
    assert width.min() >= 128 // 16 and width.max() <= 128 // 8


def test_mask_modes_share_marginal_rate():
    images = torch.randn(8000, 5, 1, 8, 128) + 3.0
    for mode, orient in (("independent", "mixed"), ("independent", "vertical"),
                         ("tube", "vertical"), ("complementary", "vertical")):
        torch.manual_seed(1)
        rate = _masked(_mask(mode, orient, 0.2, images)).flatten(2).any(-1).float().mean().item()
        assert abs(rate - 0.2) < 0.01, (mode, orient, rate)


def test_mask_mode_validation():
    for mode, orient in (("complementary", "mixed"), ("bogus", "mixed"), ("tube", "diagonal")):
        try:
            _trainer(FRAME_MASK=0.2, FRAME_MASK_MODE=mode, FRAME_MASK_ORIENT=orient)
        except ValueError:
            continue
        raise AssertionError(f"accepted {mode}/{orient}")


def test_fixed_stn_applies_the_given_affine():
    import torch.nn.functional as F
    theta = [0.782, 0.025, 0.005, 0.062, 0.606, 0.169]
    torch.manual_seed(0)
    model = ResTranOCR(num_classes=37, fixed_stn=theta).eval()
    assert not model.use_stn and not hasattr(model, "stn")
    no_stn = ResTranOCR(num_classes=37, use_stn=False)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in no_stn.parameters())
    sd = model.state_dict()
    assert torch.allclose(sd["fixed_theta"].flatten(), torch.tensor(theta))
    seen = {}
    # A hook that returns something replaces the module output, so return None explicitly
    model.backbone.register_forward_hook(lambda m, inp, out: seen.update(x=inp[0]))
    x = _batch()[0]
    with torch.no_grad():
        model(x)
    flat = x.view(-1, 3, 32, 128)
    grid = F.affine_grid(torch.tensor(theta).view(1, 2, 3).expand(flat.size(0), 2, 3),
                         flat.size(), align_corners=False)
    assert torch.allclose(seen["x"], F.grid_sample(flat, grid, align_corners=False))
    # The checkpoint carries the transform: a fresh model rebuilt from it loads strictly
    rebuilt = ResTranOCR(num_classes=37, fixed_stn=sd["fixed_theta"].flatten().tolist())
    rebuilt.load_state_dict(sd, strict=True)



def test_eval_arms_rebuilds_each_architecture():
    import tempfile
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import eval_arms
    theta = [0.782, 0.025, 0.005, 0.062, 0.606, 0.169]
    x = _batch()[0]
    for kw, kind in (({}, "stn"), ({"use_stn": False}, "none"), ({"fixed_stn": theta}, "fixed")):
        torch.manual_seed(0)
        src = ResTranOCR(num_classes=37, **kw).eval()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pth")
            torch.save(src.state_dict(), path)
            model, got = eval_arms.load_auto(path, torch.device("cpu"))
        assert got == kind, (kw, got)
        with torch.no_grad():
            assert torch.equal(model(x), src(x))


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        try:
            fn()
            print(f"OK    {name}")
        except Exception as exc:  # report every test, not just the first failure
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
