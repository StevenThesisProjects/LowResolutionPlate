"""`/predictions` — nhận dạng 1 track (roadmap: `/predict`), tra cứu, gợi ý, review."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app.core.config import Settings
from backend.app.core.errors import APIError
from backend.app.db import Database
from backend.app.deps import get_db, get_recognizer, read_limited, settings_dep
from backend.app.schemas import (
    DecodeMode,
    Page,
    Prediction,
    PredictionStatus,
    ReviewIn,
    SRImage,
    SuggestionList,
)
from backend.app.services.inference import PlateRecognizer, decode_image, png_base64
from backend.app.services.predictions import build_row, suggestions_for, to_schema

router = APIRouter(prefix="/predictions", tags=["Predictions"])

IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def _get_row(db: Database, prediction_id: UUID) -> dict:
    row = db.get_prediction(str(prediction_id))
    if row is None:
        raise APIError(404, "prediction_not_found", f"Không có prediction '{prediction_id}'.")
    return row


@router.post(
    "",
    response_model=Prediction,
    response_model_exclude_none=True,
    status_code=201,
    summary="Nhận dạng 1 track (1–5 frame)",
)
async def create_prediction(
    response: Response,
    files: Annotated[list[UploadFile], File(description="1–5 ảnh crop biển số, JPG/PNG, ≤ 5 MB/ảnh")],
    track_id: Annotated[str | None, Form(max_length=128)] = None,
    decode: DecodeMode | None = Query(None, description="Mặc định theo cấu hình server (`constrained`)"),
    include_sr: bool = Query(False, description="Trả kèm ảnh SR (PNG base64) từng frame"),
    db: Database = Depends(get_db),
    recognizer: PlateRecognizer = Depends(get_recognizer),
    settings: Settings = Depends(settings_dep),
) -> Prediction:
    if not 1 <= len(files) <= settings.max_frames:
        raise APIError(
            422, "invalid_frame_count",
            f"Cần 1–{settings.max_frames} ảnh, nhận được {len(files)}.",
        )
    frames = []
    for upload in files:
        if upload.content_type not in IMAGE_TYPES:
            raise APIError(
                415, "unsupported_media_type",
                f"'{upload.filename}' có kiểu '{upload.content_type}', chỉ nhận JPG/PNG.",
            )
        data = await read_limited(upload, settings.max_image_mb)
        frames.append(decode_image(data, upload.filename or "file"))

    mode = (decode or DecodeMode(settings.default_decode)).value
    results, latency = await run_in_threadpool(
        recognizer.predict, [frames], mode, include_sr, settings.max_suggestions,
    )
    result = results[0]
    row = build_row(
        result, recognizer=recognizer, settings=settings, decode=mode,
        num_frames=len(frames), latency_ms=latency, track_id=track_id,
    )
    stored = await run_in_threadpool(db.insert_predictions, [row])
    prediction = to_schema(stored[0])
    if result.sr_frames is not None:
        prediction.sr_images = [
            SRImage(frame=i + 1, width=img.shape[1], height=img.shape[0], data_base64=png_base64(img))
            for i, img in enumerate(result.sr_frames)
        ]
    response.headers["Location"] = f"{settings.api_prefix}/predictions/{prediction.id}"
    return prediction


@router.get(
    "", response_model=Page[Prediction], response_model_exclude_none=True,
    summary="Danh sách prediction (lọc theo status / batch)",
)
def list_predictions(
    status: PredictionStatus | None = Query(None, description="Ví dụ `needs_review` cho hàng chờ review"),
    batch_id: UUID | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    db: Database = Depends(get_db),
) -> Page[Prediction]:
    rows, total = db.list_predictions(
        status=status.value if status else None,
        batch_id=str(batch_id) if batch_id else None,
        page=page, page_size=page_size,
    )
    return Page(items=[to_schema(r) for r in rows], total=total, page=page, page_size=page_size)


@router.get(
    "/{prediction_id}", response_model=Prediction, response_model_exclude_none=True,
    summary="Chi tiết 1 prediction",
)
def get_prediction(prediction_id: UUID, db: Database = Depends(get_db)) -> Prediction:
    return to_schema(_get_row(db, prediction_id))


@router.delete("/{prediction_id}", status_code=204, summary="Xoá 1 prediction")
def delete_prediction(prediction_id: UUID, db: Database = Depends(get_db)) -> Response:
    if not db.delete_prediction(str(prediction_id)):
        raise APIError(404, "prediction_not_found", f"Không có prediction '{prediction_id}'.")
    return Response(status_code=204)


@router.get(
    "/{prediction_id}/suggestions", response_model=SuggestionList,
    summary="Gợi ý top-k biển số thay thế cho người review",
)
def get_suggestions(
    prediction_id: UUID,
    db: Database = Depends(get_db),
    recognizer: PlateRecognizer = Depends(get_recognizer),
) -> SuggestionList:
    row = _get_row(db, prediction_id)
    prediction = to_schema(row)
    return SuggestionList(
        prediction_id=prediction.id,
        plate=prediction.plate,
        final_plate=prediction.final_plate,
        status=prediction.status,
        suggestions=suggestions_for(row, recognizer),
    )


@router.put(
    "/{prediction_id}/review", response_model=Prediction, response_model_exclude_none=True,
    summary="Ghi kết quả review (sửa tay) cho 1 prediction",
)
def put_review(prediction_id: UUID, body: ReviewIn, db: Database = Depends(get_db)) -> Prediction:
    _get_row(db, prediction_id)
    plate = "".join(body.plate.upper().split())
    if not plate.isalnum():
        raise APIError(422, "invalid_plate", "Biển số chỉ gồm chữ A–Z và số 0–9.")
    db.add_review(str(prediction_id), plate, body.reviewer, body.note)
    return to_schema(_get_row(db, prediction_id))
