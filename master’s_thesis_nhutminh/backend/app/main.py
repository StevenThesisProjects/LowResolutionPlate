"""PlateReader LR — FastAPI entrypoint.

Chạy từ thư mục `master’s_thesis_nhutminh/` (để import được `src/`, `configs/`, `tools/`):

    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.core.config import get_settings
from backend.app.core.errors import register_error_handlers
from backend.app.db import Database
from backend.app.routers import batches, predictions, system
from backend.app.schemas import Problem
from backend.app.services import batches as batch_service
from backend.app.services.inference import registry

logger = logging.getLogger("platereader")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.started_at = time.monotonic()
    app.state.db = None
    try:
        app.state.db = Database(
            settings.dsn,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            timeout_s=settings.db_pool_timeout_s,
            statement_timeout_ms=settings.db_statement_timeout_ms,
        )
        app.state.db.fail_interrupted_batches()
    except Exception as exc:  # noqa: BLE001 — app still starts; /health reports it
        app.state.db_error = f"{type(exc).__name__}: {exc}"
        logger.error("Database unavailable: %s", app.state.db_error)

    registry.load(settings)
    if registry.recognizer is None:
        logger.error("Model not loaded: %s", registry.load_error)
    else:
        logger.info("Model %s loaded", registry.recognizer.version)

    yield

    batch_service.executor.shutdown(wait=False, cancel_futures=True)
    if app.state.db is not None:
        app.state.db.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description=(
            "Nhận dạng biển số độ phân giải thấp từ chuỗi 1–5 frame (MF-SR-OCR / S1), "
            "kèm cờ `needs_review` theo ngưỡng tin cậy τ. Xem `api-docs/` để có tài liệu đầy đủ."
        ),
        lifespan=lifespan,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Location", "Content-Disposition"],
    )
    register_error_handlers(app)
    # Khai báo cho Swagger: mọi lỗi 4xx/503 đều là problem+json (thay schema 422 mặc định).
    error_responses = {
        "4XX": {"model": Problem, "description": "Lỗi phía client — `application/problem+json`"},
        503: {"model": Problem, "description": "Model / database chưa sẵn sàng hoặc quá tải"},
    }
    for router in (system.router, predictions.router, batches.router):
        app.include_router(router, prefix=settings.api_prefix, responses=error_responses)

    # Lối tắt: mở http://localhost:8000/ hoặc /docs đều tới Swagger.
    @app.get("/", include_in_schema=False)
    @app.get("/docs", include_in_schema=False)
    def _to_swagger() -> RedirectResponse:
        return RedirectResponse(f"{settings.api_prefix}/docs")

    @app.get("/redoc", include_in_schema=False)
    def _to_redoc() -> RedirectResponse:
        return RedirectResponse(f"{settings.api_prefix}/redoc")

    return app


app = create_app()
