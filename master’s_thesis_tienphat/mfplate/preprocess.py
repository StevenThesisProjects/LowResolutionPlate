"""Image decoding and the model's input transform, without albumentations.

The training code reads a frame with cv2.imread, converts BGR to RGB, resizes it to
32x128 (albumentations A.Resize = cv2.resize, INTER_LINEAR), normalizes with mean 0.5 and
std 0.5, and finally replaces the three channels by BT.601 luma (src/data/dataset.py).
This module repeats those steps so that the API needs only OpenCV and NumPy. The
normalization uses the same float32 arithmetic as albumentations 2.0.8, so the
tensors are bit-identical (tests/test_preprocess.py and scripts/parity_check.py check it).
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch

IMG_H, IMG_W = 32, 128
# albumentations Normalize(mean=0.5, std=0.5, max_pixel_value=255) on uint8:
# (x - 127.5) * float32(1 / 127.5), evaluated in float32.
_NORM_LUT = (np.arange(256, dtype=np.float32) - np.float32(127.5)) * np.float32(1 / 127.5)
_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(3, 1, 1)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


class ImageError(ValueError):
    """An uploaded file cannot be used as a frame. `code` is the API error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sniff_format(data: bytes) -> Optional[str]:
    """'png', 'jpeg' or None, from the file's magic bytes (the file name is not trusted)."""
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    return None


def decode_image(data: bytes, name: str = "frame", max_side: int = 2048,
                 min_side: int = 4) -> np.ndarray:
    """Bytes of a JPEG/PNG file -> RGB uint8 array [H, W, 3]."""
    fmt = sniff_format(data)
    if fmt is None:
        raise ImageError("unsupported_type", f"{name}: chỉ nhận ảnh JPEG hoặc PNG")
    # Read the header first so a huge image is refused before it is decompressed
    size = _header_size(data, fmt)
    if size is not None and max(size) > max_side:
        raise ImageError("image_too_large",
                         f"{name}: ảnh {size[0]}x{size[1]} px vượt giới hạn {max_side} px mỗi cạnh")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ImageError("corrupt_image", f"{name}: không giải mã được ảnh (file hỏng?)")
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        raise ImageError("image_too_large",
                         f"{name}: ảnh {w}x{h} px vượt giới hạn {max_side} px mỗi cạnh")
    if min(h, w) < min_side:
        raise ImageError("image_too_small", f"{name}: ảnh {w}x{h} px quá nhỏ")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _header_size(data: bytes, fmt: str):
    """(width, height) from the PNG IHDR chunk or the first JPEG SOF marker, else None."""
    try:
        if fmt == "png" and len(data) >= 24 and data[12:16] == b"IHDR":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if fmt == "jpeg":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg_len = int.from_bytes(data[i + 2:i + 4], "big")
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h = int.from_bytes(data[i + 5:i + 7], "big")
                    w = int.from_bytes(data[i + 7:i + 9], "big")
                    return w, h
                i += 2 + seg_len
    except Exception:  # malformed header: let cv2.imdecode decide
        return None
    return None


def frame_tensor(rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 [H, W, 3] -> normalized float tensor [3, 32, 128] (still RGB)."""
    resized = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(_NORM_LUT[resized].transpose(2, 0, 1)))


def to_luma(t: torch.Tensor) -> torch.Tensor:
    """[..., 3, H, W] -> BT.601 luma replicated to 3 channels (same op as the dataset)."""
    return (t * _LUMA.to(t.dtype)).sum(dim=-3, keepdim=True).expand_as(t).contiguous()


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the frame's luma, measured at the model's input size.

    Measuring after the resize to 32x128 keeps frames of different native sizes on one
    scale. Higher = sharper. Used only to explain a result, never by the model.
    """
    resized = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class PreparedTrack:
    tensor: torch.Tensor          # [1, F, 3, 32, 128]
    sharpness: List[float]        # per frame
    sizes: List[tuple]            # (width, height) per frame, native


def prepare_track(frames: List[np.ndarray], grayscale: bool = True) -> PreparedTrack:
    """Stack 1-5 RGB frames into the model input [1, F, 3, 32, 128]."""
    if not frames:
        raise ImageError("no_files", "cần ít nhất 1 ảnh")
    x = torch.stack([frame_tensor(f) for f in frames], dim=0)
    if grayscale:
        x = to_luma(x)
    return PreparedTrack(tensor=x.unsqueeze(0), sharpness=[sharpness(f) for f in frames],
                         sizes=[(f.shape[1], f.shape[0]) for f in frames])
