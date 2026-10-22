"""`/health` and `/model` — liveness and model metadata."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request, Response

from backend.app.core.config import Settings
from backend.app.deps import get_recognizer, settings_dep
from backend.app.schemas import Health, ModelInfo
from backend.app.services.inference import PlateRecognizer, registry

router = APIRouter(tags=["System"])


@router.get("/health", response_model=Health, summary="Trạng thái service")
def health(request: Request, response: Response, settings: Settings = Depends(settings_dep)) -> Health:
    db = getattr(request.app.state, "db", None)
    db_ok = db is not None and db.ping()
    model_ok = registry.recognizer is not None
    detail = None
    if not model_ok:
        detail = registry.load_error
    elif not db_ok:
        detail = getattr(request.app.state, "db_error", None) or "Database không kết nối được."
    if not (db_ok and model_ok):
        response.status_code = 503
    return Health(
        status="ok" if db_ok and model_ok else "degraded",
        model_loaded=model_ok,
        database="ok" if db_ok else "unavailable",
        uptime_s=round(time.monotonic() - request.app.state.started_at, 1),
        version=settings.api_version,
        detail=detail,
    )


@router.get("/model", response_model=ModelInfo, summary="Thông tin model đang phục vụ")
def model_info(
    recognizer: PlateRecognizer = Depends(get_recognizer),
    settings: Settings = Depends(settings_dep),
) -> ModelInfo:
    config = recognizer.config
    return ModelInfo(
        name=settings.model_name,
        model_version=recognizer.version,
        checkpoint=recognizer.checkpoint_name,
        checkpoint_sha256=recognizer.checkpoint_sha256,
        parameters=recognizer.parameters,
        architecture={k: list(v) if isinstance(v, tuple) else v for k, v in recognizer.architecture.items()},
        device=str(recognizer.device),
        input_size=[config.IMG_HEIGHT, config.IMG_WIDTH],
        max_frames=settings.max_frames,
        default_decode=settings.default_decode,
        beam_width=settings.beam_width,
        plate_layouts=list(recognizer.layout.patterns),
        tau=settings.confidence_threshold,
    )
