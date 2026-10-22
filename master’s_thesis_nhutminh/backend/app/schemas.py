"""Pydantic models — the public contract of the API (mirrored in api-docs/).

Field names follow the roadmap §2 response:
``{plate, char_conf[], track_conf, status, sr_images[], latency_ms, model_version}``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

T = TypeVar("T")


class DecodeMode(str, Enum):
    constrained = "constrained"
    greedy = "greedy"


class PredictionStatus(str, Enum):
    accepted = "accepted"
    needs_review = "needs_review"
    reviewed = "reviewed"


class BatchStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Problem(BaseModel):
    """RFC 9457 error body (`application/problem+json`) — mọi response lỗi."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str = Field(description="Mã lỗi ổn định để client rẽ nhánh, ví dụ `invalid_image`")
    errors: list[dict[str, object]] | None = Field(default=None, description="Chỉ có khi `code = validation_error`")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int


# ── Health / model ──────────────────────────────────────────────────────────


class Health(BaseModel):
    status: str = Field(description="`ok` khi model + database sẵn sàng, ngược lại `degraded`")
    model_loaded: bool
    database: str = Field(description="`ok` | `unavailable`")
    uptime_s: float
    version: str
    detail: str | None = None


class ModelInfo(BaseModel):
    name: str
    model_version: str = Field(description="`<checkpoint>@<sha256[:12]>`")
    checkpoint: str
    checkpoint_sha256: str
    parameters: int
    architecture: dict[str, object]
    device: str
    input_size: list[int] = Field(description="[height, width] sau resize")
    max_frames: int
    default_decode: DecodeMode
    beam_width: int
    plate_layouts: list[str]
    tau: float = Field(description="Ngưỡng tin cậy τ: track_conf >= τ -> accepted")


# ── Predictions ─────────────────────────────────────────────────────────────


class CharConf(BaseModel):
    char: str
    conf: float


class Suggestion(BaseModel):
    rank: int
    plate: str
    conf: float
    layout: str | None = Field(description="Layout khớp (`LLLNLNN` Mercosur / `LLLNNNN` cũ)")


class SRImage(BaseModel):
    frame: int
    mime_type: str = "image/png"
    width: int
    height: int
    data_base64: str


class Review(BaseModel):
    plate: str
    reviewer: str | None = None
    note: str | None = None
    reviewed_at: datetime


class Prediction(BaseModel):
    id: UUID
    created_at: datetime
    batch_id: UUID | None = None
    track_id: str | None = None
    plate: str = Field(description="Chuỗi biển số model dự đoán")
    final_plate: str = Field(description="`review.plate` nếu đã review, ngược lại `plate`")
    track_conf: float
    char_conf: list[CharConf]
    status: PredictionStatus
    decode_mode: DecodeMode
    num_frames: int = Field(description="Số frame upload (trước khi pad lên 5)")
    latency_ms: float
    model_version: str
    ground_truth: str | None = None
    correct: bool | None = None
    review: Review | None = None
    sr_images: list[SRImage] | None = Field(
        default=None, description="Chỉ có trong response của POST khi `include_sr=true`; không lưu DB"
    )


class SuggestionList(BaseModel):
    prediction_id: UUID
    plate: str
    final_plate: str
    status: PredictionStatus
    suggestions: list[Suggestion]


class ReviewIn(BaseModel):
    plate: str = Field(min_length=1, max_length=16, examples=["ABC1D23"])
    reviewer: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=500)


# ── Batches ─────────────────────────────────────────────────────────────────


class Batch(BaseModel):
    id: UUID
    created_at: datetime
    finished_at: datetime | None = None
    filename: str
    status: BatchStatus
    decode_mode: DecodeMode
    total_tracks: int
    processed_tracks: int
    accepted: int
    needs_review: int
    reviewed: int
    labeled_tracks: int = Field(description="Số track có annotations.json trong zip")
    accuracy: float | None = Field(description="Exact-match (%) trên các track có nhãn")
    error: str | None = None
