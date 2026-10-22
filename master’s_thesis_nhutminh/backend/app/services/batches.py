"""Zip parsing and the background worker for `/batches`.

Accepted zip layouts (same as the dataset, so a val/test folder zips as-is)::

    tracks.zip
    ├── track_10007/
    │   ├── lr-001.jpg … lr-005.jpg     (hr-* files are ignored)
    │   └── annotations.json            (optional -> ground_truth + accuracy)
    └── track_10011/ …

Images placed directly at the zip root form one track named after the zip.
The zip is read in memory, never extracted, so path traversal is not possible.
"""

from __future__ import annotations

import io
import json
import logging
import posixpath
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from backend.app.core.config import Settings
from backend.app.core.errors import APIError
from backend.app.db import Database
from backend.app.services.inference import ModelRegistry, decode_image
from backend.app.services.predictions import build_row

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# One job at a time: the model is shared and CPU-bound, parallel jobs only
# slow each other down while also slowing interactive /predictions calls.
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch")


@dataclass
class TrackInput:
    track_id: str
    frames: list[tuple[str, bytes]] = field(default_factory=list)
    ground_truth: str | None = None


def parse_zip(data: bytes, filename: str, settings: Settings) -> list[TrackInput]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise APIError(422, "invalid_zip", f"'{filename}' không phải file zip hợp lệ.") from exc

    max_image = int(settings.max_image_mb * 1024 * 1024)
    root_name = posixpath.splitext(posixpath.basename(filename))[0] or "track"
    tracks: dict[str, TrackInput] = {}

    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = [p for p in info.filename.split("/") if p and p not in (".", "..")]
        if not parts or parts[0] == "__MACOSX" or parts[-1].startswith("."):
            continue
        name = parts[-1]
        track_id = parts[-2] if len(parts) >= 2 else root_name
        lower = name.lower()

        if lower == "annotations.json":
            try:
                label = json.loads(archive.read(info)).get("plate_text")
            except (ValueError, AttributeError):
                label = None
            if label:
                tracks.setdefault(track_id, TrackInput(track_id)).ground_truth = str(label).upper()
            continue
        if not lower.endswith(IMAGE_EXTS) or lower.startswith("hr-"):
            continue
        if info.file_size > max_image:
            raise APIError(
                413, "image_too_large",
                f"'{info.filename}' vượt {settings.max_image_mb:g} MB sau giải nén.",
            )
        tracks.setdefault(track_id, TrackInput(track_id)).frames.append((name, archive.read(info)))

    result = []
    for track in sorted(tracks.values(), key=lambda t: t.track_id):
        if not track.frames:
            continue
        track.frames.sort(key=lambda f: f[0])
        if len(track.frames) > settings.max_frames:
            raise APIError(
                422, "too_many_frames",
                f"Track '{track.track_id}' có {len(track.frames)} ảnh LR, tối đa {settings.max_frames}.",
            )
        result.append(track)

    if not result:
        raise APIError(422, "empty_zip", "Zip không chứa ảnh LR (.png/.jpg) nào.")
    if len(result) > settings.max_batch_tracks:
        raise APIError(
            413, "too_many_tracks",
            f"Zip có {len(result)} track, tối đa {settings.max_batch_tracks}.",
        )
    return result


def run_batch(
    batch_id: str,
    tracks: list[TrackInput],
    decode: str,
    db: Database,
    registry: ModelRegistry,
    settings: Settings,
) -> None:
    db.update_batch(batch_id, status="running")
    processed = 0
    try:
        recognizer = registry.get()
        size = settings.inference_batch_size
        for start in range(0, len(tracks), size):
            chunk = tracks[start:start + size]
            # A broken image fails only its own track's decode, not the job.
            images = [[decode_image(data, name) for name, data in t.frames] for t in chunk]
            results, latency = recognizer.predict(images, decode, top_k=settings.max_suggestions)
            per_track = latency / len(chunk)
            db.insert_predictions([
                build_row(
                    r, recognizer=recognizer, settings=settings, decode=decode,
                    num_frames=len(t.frames), latency_ms=per_track,
                    track_id=t.track_id, batch_id=batch_id, ground_truth=t.ground_truth,
                )
                for t, r in zip(chunk, results)
            ])
            processed += len(chunk)
            db.update_batch(batch_id, processed_tracks=processed)
        db.finish_batch(batch_id, "completed")
    except APIError as exc:
        db.finish_batch(batch_id, "failed", exc.detail)
    except Exception as exc:  # noqa: BLE001 — job must end in a terminal state
        logger.exception("Batch %s failed", batch_id)
        db.finish_batch(batch_id, "failed", f"{type(exc).__name__}: {exc}")
