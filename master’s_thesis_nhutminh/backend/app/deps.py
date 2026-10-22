"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

from fastapi import Request, UploadFile

from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import APIError
from backend.app.db import Database
from backend.app.services.inference import PlateRecognizer, registry


def get_db(request: Request) -> Database:
    db: Database | None = getattr(request.app.state, "db", None)
    if db is None:
        raise APIError(503, "database_unavailable", "Database chưa sẵn sàng.")
    return db


def get_recognizer() -> PlateRecognizer:
    return registry.get()


def settings_dep() -> Settings:
    return get_settings()


async def read_limited(upload: UploadFile, max_mb: float) -> bytes:
    """Read an upload, rejecting it with 413 as soon as it passes `max_mb`."""

    limit = int(max_mb * 1024 * 1024)
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise APIError(413, "file_too_large", f"'{upload.filename}' vượt giới hạn {max_mb:g} MB.")
    return data
