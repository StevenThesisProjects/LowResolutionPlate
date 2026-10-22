"""`/batches` — nhận dạng nhiều track từ 1 file zip (roadmap: `/predict_batch`)."""

from __future__ import annotations

import csv
import io
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app.core.config import Settings
from backend.app.core.errors import APIError
from backend.app.db import Database
from backend.app.deps import get_db, get_recognizer, read_limited, settings_dep
from backend.app.schemas import Batch, DecodeMode, Page, Prediction
from backend.app.services import batches as batch_service
from backend.app.services.inference import PlateRecognizer, registry
from backend.app.services.predictions import batch_to_schema, to_schema

router = APIRouter(prefix="/batches", tags=["Batches"])

ZIP_TYPES = {"application/zip", "application/x-zip-compressed", "application/octet-stream", "multipart/x-zip"}

CSV_COLUMNS = [
    "track_id", "plate", "track_conf", "status", "final_plate",
    "ground_truth", "correct", "decode_mode", "model_version", "prediction_id",
]


def _get_batch(db: Database, batch_id: UUID) -> dict:
    row = db.get_batch(str(batch_id))
    if row is None:
        raise APIError(404, "batch_not_found", f"Không có batch '{batch_id}'.")
    return row


@router.post("", response_model=Batch, status_code=202, summary="Tạo batch job từ file zip")
async def create_batch(
    response: Response,
    file: Annotated[UploadFile, File(description="Zip: mỗi thư mục con = 1 track (lr-*.jpg|png)")],
    decode: DecodeMode | None = Query(None),
    db: Database = Depends(get_db),
    _: PlateRecognizer = Depends(get_recognizer),
    settings: Settings = Depends(settings_dep),
) -> Batch:
    filename = file.filename or "upload.zip"
    if file.content_type not in ZIP_TYPES and not filename.lower().endswith(".zip"):
        raise APIError(415, "unsupported_media_type", f"'{filename}' không phải file .zip.")
    data = await read_limited(file, settings.max_zip_mb)
    tracks = await run_in_threadpool(batch_service.parse_zip, data, filename, settings)

    mode = (decode or DecodeMode(settings.default_decode)).value
    batch_id = db.insert_batch(filename, mode, len(tracks))
    batch_service.executor.submit(
        batch_service.run_batch, batch_id, tracks, mode, db, registry, settings,
    )
    response.headers["Location"] = f"{settings.api_prefix}/batches/{batch_id}"
    return batch_to_schema(db.get_batch(batch_id))


@router.get("", response_model=Page[Batch], summary="Danh sách batch")
def list_batches(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    db: Database = Depends(get_db),
) -> Page[Batch]:
    rows, total = db.list_batches(page, page_size)
    return Page(items=[batch_to_schema(r) for r in rows], total=total, page=page, page_size=page_size)


@router.get("/{batch_id}", response_model=Batch, summary="Trạng thái / tiến độ 1 batch")
def get_batch(batch_id: UUID, db: Database = Depends(get_db)) -> Batch:
    return batch_to_schema(_get_batch(db, batch_id))


@router.get(
    "/{batch_id}/predictions", response_model=Page[Prediction], response_model_exclude_none=True,
    summary="Kết quả từng track của batch",
)
def list_batch_predictions(
    batch_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Database = Depends(get_db),
) -> Page[Prediction]:
    _get_batch(db, batch_id)
    rows, total = db.list_predictions(status=None, batch_id=str(batch_id), page=page, page_size=page_size)
    return Page(items=[to_schema(r) for r in rows], total=total, page=page, page_size=page_size)


@router.get(
    "/{batch_id}/export",
    summary="Xuất kết quả batch ra CSV",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
)
def export_batch(batch_id: UUID, db: Database = Depends(get_db)) -> Response:
    batch = _get_batch(db, batch_id)
    if batch["status"] != "completed":
        raise APIError(409, "batch_not_completed", f"Batch đang ở trạng thái '{batch['status']}'.")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for row in db.batch_predictions(str(batch_id)):
        p = to_schema(row)
        writer.writerow({
            "track_id": p.track_id, "plate": p.plate, "track_conf": f"{p.track_conf:.4f}",
            "status": p.status.value, "final_plate": p.final_plate,
            "ground_truth": p.ground_truth or "",
            "correct": "" if p.correct is None else int(p.correct),
            "decode_mode": p.decode_mode.value, "model_version": p.model_version,
            "prediction_id": p.id,
        })
    stem = batch["filename"].rsplit(".", 1)[0] or "batch"
    return Response(
        buffer.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}_results.csv"'},
    )


@router.delete("/{batch_id}", status_code=204, summary="Xoá batch và toàn bộ prediction của nó")
def delete_batch(batch_id: UUID, db: Database = Depends(get_db)) -> Response:
    batch = _get_batch(db, batch_id)
    if batch["status"] in ("queued", "running"):
        raise APIError(409, "batch_in_progress", "Không xoá được batch đang chạy.")
    db.delete_batch(str(batch_id))
    return Response(status_code=204)
