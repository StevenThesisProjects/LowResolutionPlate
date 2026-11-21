"""The API's input transform must equal the training pipeline's, bit for bit."""
import glob

import cv2
import numpy as np
import pytest
import torch

from mfplate.preprocess import (ImageError, decode_image, frame_tensor, prepare_track, sharpness,
                                to_luma)

from conftest import DATA, needs_data


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


@needs_data
def test_tensor_bit_identical_to_dataset_pipeline():
    from src.data.dataset import MultiFrameDataset
    from src.data.transforms import get_val_transforms
    t = get_val_transforms(32, 128)
    paths = (sorted(glob.glob(str(DATA / "Scenario-B" / "*" / "track_1*" / "lr-*.jpg")))[:60]
             + sorted(glob.glob(str(DATA / "Scenario-A" / "*" / "track_0*" / "lr-*.png")))[:20])
    assert paths
    for p in paths:
        rgb_ref = MultiFrameDataset._read_rgb(p)
        rgb = decode_image(open(p, "rb").read(), p)
        assert np.array_equal(rgb, rgb_ref)
        ref = t(image=rgb_ref)["image"]
        assert torch.equal(frame_tensor(rgb), ref)
        assert torch.equal(to_luma(frame_tensor(rgb)), MultiFrameDataset._to_luma(ref))


def test_luma_is_replicated_and_weighted():
    x = torch.zeros(3, 2, 2)
    x[0], x[1], x[2] = 1.0, 0.0, 0.0
    y = to_luma(x)
    assert torch.allclose(y, torch.full_like(y, 0.299))


def test_prepare_track_shapes_any_frame_count():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (17, 47, 3), dtype=np.uint8) for _ in range(3)]
    tr = prepare_track(frames)
    assert tr.tensor.shape == (1, 3, 3, 32, 128)
    assert torch.equal(tr.tensor[0, :, 0], tr.tensor[0, :, 1])   # luma in every channel
    assert len(tr.sharpness) == 3 and tr.sizes[0] == (47, 17)


def test_rejects_non_image():
    with pytest.raises(ImageError) as e:
        decode_image(b"GIF89a....", "x.gif")
    assert e.value.code == "unsupported_type"


def test_rejects_truncated_jpeg():
    img = np.full((20, 60, 3), 128, np.uint8)
    data = cv2.imencode(".jpg", img)[1].tobytes()
    with pytest.raises(ImageError) as e:
        decode_image(data[:30], "cut.jpg")
    assert e.value.code in ("corrupt_image",)


def test_rejects_huge_dimensions_from_header():
    big = np.zeros((10, 3000, 3), np.uint8)
    with pytest.raises(ImageError) as e:
        decode_image(_png(big), "wide.png", max_side=2048)
    assert e.value.code == "image_too_large"


def test_rejects_tiny_image():
    with pytest.raises(ImageError) as e:
        decode_image(_png(np.zeros((2, 2, 3), np.uint8)), "dot.png")
    assert e.value.code == "image_too_small"


def test_grayscale_and_alpha_png_are_read_as_rgb():
    gray = np.tile(np.arange(64, dtype=np.uint8), (20, 1))
    assert decode_image(_png(gray), "g.png").shape == (20, 64, 3)
    rgba = np.dstack([np.zeros((20, 64, 3), np.uint8), np.full((20, 64), 255, np.uint8)])
    assert decode_image(_png(rgba), "a.png").shape == (20, 64, 3)


def test_sharpness_drops_with_blur():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (32, 128, 3), dtype=np.uint8)
    assert sharpness(cv2.GaussianBlur(img, (7, 7), 2.0)) < sharpness(img)
