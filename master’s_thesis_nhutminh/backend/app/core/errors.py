"""Uniform error bodies following RFC 9457 (``application/problem+json``).

Every non-2xx response has the same shape, so the React client needs a
single error handler::

    {"type": "...", "title": "...", "status": 422, "detail": "...",
     "instance": "/api/v1/predictions", "code": "invalid_image", "errors": [...]}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg import OperationalError
from psycopg.errors import QueryCanceled
from psycopg_pool import PoolTimeout
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_JSON = "application/problem+json"


class APIError(Exception):
    """Raise anywhere in a request to return a problem+json response."""

    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        title: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.title = title or code.replace("_", " ").capitalize()
        self.errors = errors


def _problem(
    request: Request,
    status: int,
    code: str,
    title: str,
    detail: str,
    errors: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"https://platereader.local/problems/{code}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "code": code,
    }
    if errors:
        body["errors"] = errors
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return _problem(request, exc.status, exc.code, exc.title, exc.detail, exc.errors)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
            for err in exc.errors()
        ]
        return _problem(
            request, 422, "validation_error", "Validation error",
            "Request parameters or body are invalid.", errors,
        )

    @app.exception_handler(PoolTimeout)
    async def _pool_timeout(request: Request, exc: PoolTimeout) -> JSONResponse:
        # Mọi connection đang bận: báo client thử lại thay vì treo request.
        response = _problem(
            request, 503, "database_busy", "Database busy",
            "Database đang quá tải, vui lòng thử lại sau vài giây.",
        )
        response.headers["Retry-After"] = "5"
        return response

    @app.exception_handler(QueryCanceled)
    async def _query_timeout(request: Request, exc: QueryCanceled) -> JSONResponse:
        return _problem(request, 503, "database_timeout", "Database timeout", "Truy vấn vượt statement_timeout.")

    @app.exception_handler(OperationalError)
    async def _db_down(request: Request, exc: OperationalError) -> JSONResponse:
        return _problem(request, 503, "database_unavailable", "Database unavailable", "Mất kết nối tới database.")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _problem(request, exc.status_code, code, code.replace("_", " ").capitalize(), str(exc.detail))
