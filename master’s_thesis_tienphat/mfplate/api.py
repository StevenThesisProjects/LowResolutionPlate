"""MF-PlateReader HTTP API (FastAPI).

    python3 -m uvicorn mfplate.api:app --host 0.0.0.0 --port 8000      # from masters_thesis_tienphat/

Endpoints
    GET  /health     model loaded? (503 while it is not)
    GET  /model      frozen model, decoder, thresholds, limits
    POST /predict    1-5 plate crops (jpg/png) or one zip of a track -> plate + details
    POST /feedback   the user says whether a prediction was right
    GET  /history    recent predictions (of one session)
    GET  /stats      counts, review rate, feedback accuracy
    GET  /samples    demo tracks in samples/ (when the folder exists)
Interactive docs at /docs, the OpenAPI spec at /openapi.json. When the React UI has been
built (webui/dist), it is served at / ; otherwise / redirects to /docs.
"""
import hashlib
import logging
import os
import re
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, File, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .decoding import DECODERS
from .engine import PlateReader
from .schemas import (ErrorResponse, FeedbackRequest, FeedbackResponse, HealthResponse,
                      HistoryItem, ModelInfo, PredictResponse)
from .settings import THESIS_DIR, Settings
from .storage import Store
from .uploads import UploadError, from_images, from_zip, is_zip

log = logging.getLogger("mfplate")
logging.basicConfig(level=os.environ.get("MFPLATE_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

ERRORS = {
    400: {"model": ErrorResponse, "description": "Yêu cầu sai (thiếu ảnh, quá 5 ảnh, zip hỏng...)"},
    401: {"model": ErrorResponse, "description": "Thiếu hoặc sai X-API-Key (khi bật khoá)"},
    413: {"model": ErrorResponse, "description": "File hoặc ảnh quá lớn"},
    415: {"model": ErrorResponse, "description": "Không phải ảnh JPEG/PNG"},
    422: {"model": ErrorResponse, "description": "Ảnh hỏng, quá nhỏ, hoặc tham số sai"},
    503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng"},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.started = time.time()
    app.state.reader = None
    app.state.spec = None
    app.state.load_error = None
    try:
        spec = settings.load()
        app.state.spec = spec
        app.state.reader = PlateReader(
            spec.model_path,
            expected_sha256=spec.expected_sha256 if settings.verify_checksum else None,
            device=settings.device, threads=settings.torch_threads,
            grayscale=spec.base["preprocessing"]["grayscale"])
        log.info("model %s loaded from %s", spec.model_version, spec.model_path)
    except Exception as e:  # keep serving /health so the failure is visible
        app.state.load_error = f"{type(e).__name__}: {e}"
        log.error("model not loaded: %s", app.state.load_error)
    app.state.store = Store(settings.db_path)
    yield


app = FastAPI(
    title="MF-PlateReader API",
    version=__version__,
    description=("Đọc biển số Brazil/Mercosur 1 dòng từ 1–5 ảnh crop độ phân giải thấp của cùng "
                 "một xe, bằng model ResTran đã đóng băng (công thức huấn luyện #50, lock-box). "
                 "Kết quả chỉ để tham khảo; track có cờ `needs_review` cần người kiểm tra."),
    lifespan=lifespan,
)
_origins = [o.strip() for o in os.environ.get("MFPLATE_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"],
                       allow_headers=["*"])


# ---------------------------------------------------------------- errors: one JSON shape
def _error(status: int, code: str, message: str, request_id: Optional[str] = None):
    return JSONResponse(status_code=status, content={
        "error": {"code": code, "message": message}, "request_id": request_id})


@app.exception_handler(UploadError)
async def _upload_error(request: Request, exc: UploadError):
    return _error(exc.status, exc.code, exc.message, getattr(request.state, "request_id", None))


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    if first.get("type") == "missing" and "files" in where:
        return _error(400, "no_files", "Cần gửi 1 đến 5 ảnh crop biển số (trường `files`)")
    return _error(422, "invalid_parameter", f"{where}: {first.get('msg', 'giá trị không hợp lệ')}")


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed",
            503: "model_unavailable"}.get(exc.status_code, "http_error")
    return _error(exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception):
    log.exception("unhandled error")
    return _error(500, "internal_error", "Lỗi không mong đợi ở server",
                  getattr(request.state, "request_id", None))


def require_key(x_api_key: Optional[str] = Header(default=None)):
    """Optional shared key: set MFPLATE_API_KEY to require the X-API-Key header."""
    expected = os.environ.get("MFPLATE_API_KEY")
    if expected and x_api_key != expected:
        raise StarletteHTTPException(401, "Thiếu hoặc sai X-API-Key")


def _reader(request: Request) -> PlateReader:
    reader = request.app.state.reader
    if reader is None:
        raise StarletteHTTPException(503, f"Model chưa sẵn sàng: {request.app.state.load_error}")
    return reader


# ---------------------------------------------------------------- endpoints
# Web UI (React, built into webui/dist) and demo samples. Both are optional: the API
# works without them, and the Docker image adds them only when they exist.
WEBUI_DIST = Path(os.environ.get("MFPLATE_WEBUI_DIST", THESIS_DIR / "webui" / "dist"))
SAMPLES_DIR = Path(os.environ.get("MFPLATE_SAMPLES_DIR", THESIS_DIR / "samples"))
SAMPLE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.zip$")
if (WEBUI_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEBUI_DIST / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def root():
    index = WEBUI_DIST / "index.html"
    if index.exists():
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return RedirectResponse("/docs")


@app.get("/samples", dependencies=[Depends(require_key)])
def samples():
    """Demo tracks (zip) in samples/. LRLPR-26 data: private demonstrations only."""
    if not SAMPLES_DIR.is_dir():
        return []
    return [{"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(SAMPLES_DIR.glob("*.zip")) if SAMPLE_NAME.match(p.name)]


@app.get("/samples/{name}", dependencies=[Depends(require_key)],
         responses={404: {"model": ErrorResponse}})
def sample_file(name: str):
    path = SAMPLES_DIR / name
    if not SAMPLE_NAME.match(name) or not path.is_file() or path.parent.resolve() != SAMPLES_DIR.resolve():
        return _error(404, "not_found", f"Không có track mẫu {name}")
    return FileResponse(path, media_type="application/zip", filename=name)


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(request: Request):
    st = request.app.state
    body = HealthResponse(
        status="ok" if st.reader else "unavailable", model_loaded=st.reader is not None,
        model_version=st.spec.model_version if st.spec else None,
        device=str(st.reader.device) if st.reader else None,
        uptime_s=round(time.time() - st.started, 1), detail=st.load_error)
    return JSONResponse(status_code=200 if st.reader else 503, content=body.model_dump())


@app.get("/model", response_model=ModelInfo, responses={503: ERRORS[503]})
def model_info(request: Request):
    reader = _reader(request)
    spec, s = request.app.state.spec, request.app.state.settings
    return ModelInfo(
        model_version=spec.model_version, checkpoint=spec.checkpoint_label,
        sha256=spec.expected_sha256, sha256_verified=reader.sha256 == spec.expected_sha256,
        parameters=spec.base["architecture"]["parameters"],
        default_decoder=spec.default_decoder, decoders=list(DECODERS),
        triage_score=spec.score_name,
        thresholds_by_frames={str(n): spec.threshold(n) for n in range(1, 6)},
        expected_accuracy_by_frames=spec.base.get("expected_accuracy_by_frames", {}),
        limits={"max_frames": s.max_frames, "max_file_bytes": s.max_file_bytes,
                "max_zip_bytes": s.max_zip_bytes, "max_side_px": s.max_side_px},
        scope=spec.base["scope"], selection_rule=spec.base["selection_rule"])


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)],
          responses={k: v for k, v in ERRORS.items()})
def predict(
    request: Request,
    files: List[UploadFile] = File(..., description="1–5 ảnh crop (jpg/png) của cùng một biển "
                                                    "số, hoặc 1 file zip của một track"),
    decoder: Optional[Literal["greedy", "beam", "constrained"]] = Query(
        None, description="Mặc định: bộ giải mã đã chốt trong artifacts/decoder_decision.json"),
    top_k: int = Query(3, ge=1, le=5, description="Số chuỗi ứng viên trả về"),
    session_id: Optional[str] = Query(None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
                                      description="Gom lịch sử theo phiên (UI tự sinh)"),
):
    t_start = time.perf_counter()
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    reader = _reader(request)
    spec, s = request.app.state.spec, request.app.state.settings

    # 1. Read and validate the upload
    named = []
    for f in files:
        data = f.file.read(max(s.max_file_bytes, s.max_zip_bytes) + 1)
        named.append((f.filename or f"frame{len(named) + 1}", data))
    zips = [kv for kv in named if is_zip(*kv)]
    if zips and len(named) > 1:
        raise UploadError(400, "mixed_upload", "Gửi 1 file zip HOẶC 1–5 ảnh, không gửi lẫn")
    if zips:
        frames = from_zip(zips[0][0], zips[0][1], s.max_frames, s.max_file_bytes,
                          s.max_zip_bytes, s.max_side_px, s.min_side_px)
    else:
        for name, data in named:
            if len(data) == 0:
                raise UploadError(422, "empty_file", f"{name}: file rỗng")
            if len(data) > s.max_file_bytes:
                raise UploadError(413, "file_too_large",
                                  f"{name}: vượt {s.max_file_bytes / 1e6:.0f} MB mỗi ảnh")
        frames = from_images(named, s.max_frames, s.max_side_px, s.min_side_px)
    read_ms = (time.perf_counter() - t_start) * 1000.0

    # 2-4. Model, decoding
    dec_name = decoder or spec.default_decoder
    reading = reader.read(frames.images, decoder=dec_name, top_k=top_k)
    out, dec = reading.output, reading.decoded
    n = len(frames.images)

    # 5. Triage
    score = dec.score(spec.score_name)
    threshold = spec.threshold(n)
    reasons = []
    if threshold is None:
        reasons.append("not_calibrated")
    elif score < threshold:
        reasons.append("low_confidence")
    if reading.layout is None:
        reasons.append("invalid_layout")

    warnings = list(frames.notes)
    acc_n, acc_5 = spec.expected_accuracy(n), spec.expected_accuracy(5)
    if n < 5:
        msg = f"Chỉ có {n} frame; model được thiết kế cho 5 frame liên tiếp của cùng một xe."
        if acc_n is not None and acc_5 is not None:
            msg += (f" Trên lock-box, accuracy trung bình với {n} frame là {acc_n:.1f}% "
                    f"(5 frame: {acc_5:.1f}%).")
        warnings.append(msg)
    widths = [w for w, _ in out.sizes]
    if n > 1 and max(widths) > 1.6 * min(widths):
        warnings.append("Các frame có kích thước rất khác nhau; có thể không cùng một biển số.")
    for name, (w, h) in zip(frames.names, out.sizes):
        ratio = w / h
        if ratio < 1.3 or ratio > 6.0:
            warnings.append(f"{name}: tỉ lệ rộng/cao {ratio:.1f}, không giống ảnh crop biển số "
                            "1 dòng (thường khoảng 2–3).")
    digests = [hashlib.sha256(b).hexdigest() for b in frames.raw]
    if len(set(digests)) < len(digests):
        warnings.append("Có frame trùng nhau hoàn toàn; trọng số fusion sẽ bị lệch về frame đó.")
    if reading.layout is None:
        warnings.append("Chuỗi đọc được không theo mẫu biển Brazil (LLLNNNN) hay Mercosur "
                        "(LLLNLNN).")

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    result = PredictResponse(
        request_id=request_id, plate=dec.text, confidence=round(dec.mean_char, 6),
        sequence_prob=round(dec.seq_prob, 6), char_conf=[round(c, 6) for c in dec.char_conf],
        top3=[{"plate": t, "prob": round(p, 6)} for t, p in dec.candidates],
        layout=reading.layout, n_frames=n,
        frame_weights=[round(w, 6) for w in out.frame_weights],
        sharpness=[round(v, 2) for v in out.sharpness],
        frames=[{"index": i, "name": name, "width": w, "height": h,
                 "weight": round(out.frame_weights[i], 6), "sharpness": round(out.sharpness[i], 2)}
                for i, (name, (w, h)) in enumerate(zip(frames.names, out.sizes))],
        needs_review=bool(reasons), review_reasons=reasons,
        triage={"needs_review": bool(reasons), "reasons": reasons, "score_name": spec.score_name,
                "score": round(score, 6), "threshold": threshold},
        warnings=warnings, expected_accuracy=acc_n, reference_plate=frames.reference_plate,
        decoder=dec_name, latency_ms=round(latency_ms, 2),
        timing={"read_ms": round(read_ms, 2), "model_ms": round(out.model_ms, 2),
                "decode_ms": round(reading.decode_ms, 2)},
        model_version=spec.model_version,
    )
    payload = result.model_dump()
    input_sha = hashlib.sha256("".join(digests).encode()).hexdigest()
    try:
        request.app.state.store.log_prediction(payload, input_sha, session_id, score)
    except Exception:  # the log must never break a prediction
        log.exception("could not log prediction %s", request_id)
    log.info("predict %s n=%d plate=%s conf=%.4f review=%s %.1f ms", request_id, n, dec.text,
             dec.mean_char, bool(reasons), latency_ms)
    return result


@app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_key)],
          responses={404: {"model": ErrorResponse}, 422: ERRORS[422]})
def feedback(request: Request, body: FeedbackRequest):
    corrected = body.corrected_plate.upper() if body.corrected_plate else None
    fid = request.app.state.store.add_feedback(body.request_id, body.correct, corrected)
    if fid is None:
        return _error(404, "not_found", f"Không có dự đoán với request_id {body.request_id}")
    return FeedbackResponse(feedback_id=fid, request_id=body.request_id)


@app.get("/history", response_model=List[HistoryItem], dependencies=[Depends(require_key)])
def history(request: Request,
            session_id: Optional[str] = Query(None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
            limit: int = Query(20, ge=1, le=200)):
    return request.app.state.store.history(session_id, limit)


@app.get("/stats", dependencies=[Depends(require_key)])
def stats(request: Request):
    return request.app.state.store.stats()
