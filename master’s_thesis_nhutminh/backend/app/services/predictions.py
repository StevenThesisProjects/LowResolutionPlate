"""Glue between recognizer output, the database row and the API schema."""

from __future__ import annotations

from typing import Any

from backend.app.core.config import Settings
from backend.app.schemas import (
    Batch,
    CharConf,
    Prediction,
    PredictionStatus,
    Review,
    Suggestion,
)
from backend.app.services.decoding import match_layout
from backend.app.services.inference import PlateRecognizer, TrackResult


def status_for(track_conf: float, settings: Settings) -> PredictionStatus:
    if track_conf >= settings.confidence_threshold:
        return PredictionStatus.accepted
    return PredictionStatus.needs_review


def build_row(
    result: TrackResult,
    *,
    recognizer: PlateRecognizer,
    settings: Settings,
    decode: str,
    num_frames: int,
    latency_ms: float,
    track_id: str | None = None,
    batch_id: str | None = None,
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """Columns for one `predictions` INSERT (id / created_at come from Postgres)."""
    return {
        "batch_id": batch_id,
        "track_id": track_id,
        "plate": result.plate,
        "track_conf": round(result.track_conf, 6),
        "char_conf": [round(c, 6) for c in result.char_conf],
        "suggestions": [[text, round(conf, 6)] for text, conf in result.suggestions],
        "status": status_for(result.track_conf, settings).value,
        "decode_mode": decode,
        "num_frames": num_frames,
        "latency_ms": round(latency_ms, 2),
        "model_version": recognizer.version,
        "ground_truth": ground_truth,
    }


def to_schema(row: dict[str, Any]) -> Prediction:
    review = None
    if row.get("review_plate"):
        review = Review(
            plate=row["review_plate"], reviewer=row.get("reviewer"),
            note=row.get("review_note"), reviewed_at=row["reviewed_at"],
        )
    plate = row["plate"]
    char_conf = row["char_conf"]
    return Prediction(
        id=row["id"],
        created_at=row["created_at"],
        batch_id=row.get("batch_id"),
        track_id=row.get("track_id"),
        plate=plate,
        final_plate=review.plate if review else plate,
        track_conf=row["track_conf"],
        char_conf=[
            CharConf(char=c, conf=char_conf[i] if i < len(char_conf) else 0.0)
            for i, c in enumerate(plate)
        ],
        status=row["status"],
        decode_mode=row["decode_mode"],
        num_frames=row["num_frames"],
        latency_ms=row["latency_ms"],
        model_version=row["model_version"],
        ground_truth=row.get("ground_truth"),
        correct=(plate == row["ground_truth"]) if row.get("ground_truth") else None,
        review=review,
    )


def suggestions_for(row: dict[str, Any], recognizer: PlateRecognizer) -> list[Suggestion]:
    return [
        Suggestion(rank=i + 1, plate=text, conf=conf, layout=match_layout(text, recognizer.layout))
        for i, (text, conf) in enumerate(row["suggestions"])
    ]


def batch_to_schema(row: dict[str, Any]) -> Batch:
    labeled = row["labeled_tracks"]
    return Batch(
        **{k: row[k] for k in (
            "id", "created_at", "finished_at", "filename", "status", "decode_mode",
            "total_tracks", "processed_tracks", "accepted", "needs_review", "reviewed",
            "labeled_tracks", "error",
        )},
        accuracy=round(100 * row["correct"] / labeled, 2) if labeled else None,
    )
