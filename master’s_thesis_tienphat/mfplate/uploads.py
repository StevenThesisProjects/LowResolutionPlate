"""Turn uploaded files (1-5 images, or one zip of a track) into named RGB frames."""
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import List, Optional, Tuple

import numpy as np

from .preprocess import ImageError, decode_image

IMAGE_EXT = (".jpg", ".jpeg", ".png")
ZIP_MAGIC = b"PK\x03\x04"


class UploadError(Exception):
    """A request-level problem. `status` is the HTTP status, `code` the API error code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass
class Frames:
    names: List[str]
    images: List[np.ndarray]
    raw: List[bytes]
    reference_plate: Optional[str] = None
    source: str = "images"          # images | zip
    notes: List[str] = field(default_factory=list)


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def is_zip(name: str, data: bytes) -> bool:
    return data.startswith(ZIP_MAGIC) or name.lower().endswith(".zip")


def _decode_all(named: List[Tuple[str, bytes]], max_side: int, min_side: int) -> List[np.ndarray]:
    images = []
    for name, data in named:
        try:
            images.append(decode_image(data, name, max_side=max_side, min_side=min_side))
        except ImageError as e:
            status = {"unsupported_type": 415, "image_too_large": 413}.get(e.code, 422)
            raise UploadError(status, e.code, str(e)) from None
    return images


def from_images(named: List[Tuple[str, bytes]], max_frames: int, max_side: int,
                min_side: int) -> Frames:
    if not named:
        raise UploadError(400, "no_files", "Cần gửi 1 đến 5 ảnh crop biển số (hoặc 1 file zip)")
    if len(named) > max_frames:
        raise UploadError(400, "too_many_files",
                          f"Nhận {len(named)} ảnh, tối đa {max_frames} ảnh của cùng một biển số")
    named = sorted(named, key=lambda kv: _natural_key(kv[0]))
    return Frames(names=[n for n, _ in named], images=_decode_all(named, max_side, min_side),
                  raw=[d for _, d in named])


def from_zip(name: str, data: bytes, max_frames: int, max_file_bytes: int,
             max_zip_bytes: int, max_side: int, min_side: int) -> Frames:
    """A zip of one track. When it holds lr-*/hr-* files (a dataset track folder), only the
    LR frames are used: HR frames are training data, never model input."""
    if len(data) > max_zip_bytes:
        raise UploadError(413, "zip_too_large",
                          f"{name}: {len(data) / 1e6:.1f} MB, tối đa {max_zip_bytes / 1e6:.0f} MB")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except (zipfile.BadZipFile, ValueError):
        raise UploadError(400, "zip_invalid", f"{name}: không đọc được file zip") from None
    files = [i for i in infos if not i.is_dir() and "__MACOSX" not in i.filename
             and not PurePosixPath(i.filename).name.startswith(".")]
    images = [i for i in files if i.filename.lower().endswith(IMAGE_EXT)]
    lr = [i for i in images if PurePosixPath(i.filename).name.lower().startswith("lr-")]
    chosen = sorted(lr or images, key=lambda i: _natural_key(i.filename))
    if not chosen:
        raise UploadError(400, "zip_no_images", f"{name}: zip không chứa ảnh .jpg/.png nào")
    if len(chosen) > max_frames:
        raise UploadError(400, "too_many_files",
                          f"{name}: zip có {len(chosen)} ảnh, tối đa {max_frames}")
    named = []
    for info in chosen:
        if info.file_size > max_file_bytes:
            raise UploadError(413, "file_too_large",
                              f"{info.filename}: {info.file_size / 1e6:.1f} MB, tối đa "
                              f"{max_file_bytes / 1e6:.0f} MB mỗi ảnh")
        try:
            with zf.open(info) as fh:     # read at most the limit: sizes in a zip can lie
                blob = fh.read(max_file_bytes + 1)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError):
            raise UploadError(400, "zip_invalid", f"{info.filename}: không giải nén được") from None
        if len(blob) > max_file_bytes:
            raise UploadError(413, "file_too_large", f"{info.filename}: vượt giới hạn")
        named.append((PurePosixPath(info.filename).name, blob))
    reference = None
    ann = [i for i in files if PurePosixPath(i.filename).name == "annotations.json"]
    if ann and ann[0].file_size < 64 * 1024:
        try:
            meta = json.loads(zf.read(ann[0]).decode("utf-8"))
            meta = meta[0] if isinstance(meta, list) else meta
            text = str(meta.get("plate_text", "")).strip().upper()
            reference = text or None
        except Exception:
            reference = None
    notes = []
    if lr and len(images) > len(lr):
        notes.append(f"Zip có {len(images) - len(lr)} ảnh HR; chỉ dùng {len(lr)} ảnh lr-*")
    return Frames(names=[n for n, _ in named], images=_decode_all(named, max_side, min_side),
                  raw=[d for _, d in named], reference_plate=reference, source="zip", notes=notes)
