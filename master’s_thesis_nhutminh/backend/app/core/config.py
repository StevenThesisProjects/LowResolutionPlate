"""Runtime settings, read from environment variables prefixed ``PLATEREADER_``.

Every knob has a default that works from the thesis repo root, so
``uvicorn backend.app.main:app`` runs without any ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from psycopg.conninfo import make_conninfo
from pydantic_settings import BaseSettings, SettingsConfigDict

# master's_thesis_nhutminh/ — where `src/`, `configs/` and `results/` live.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLATEREADER_", env_file=".env", extra="ignore")

    app_name: str = "PlateReader LR API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Model
    checkpoint_path: Path = PROJECT_ROOT / "results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth"
    model_name: str = "MF-SR-OCR (S1)"
    # Phải khớp lúc train: S1/J1 = 8, S4 (sr_scale=1) = 4. Không nằm trong state_dict.
    width_downsample: int = 8
    device: str = "cpu"
    torch_threads: int = 4
    default_decode: str = "constrained"  # constrained | greedy
    beam_width: int = 16
    plate_layouts: str = "LLLNLNN,LLLNNNN"

    # Selective prediction: track_conf >= tau -> accepted, ngược lại needs_review.
    # Giá trị tạm — τ chính thức chọn trên val ở Tuần 5 (precision >= 95%).
    confidence_threshold: float = 0.90
    max_suggestions: int = 5

    # Upload limits
    max_frames: int = 5
    max_image_mb: float = 5.0
    max_zip_mb: float = 200.0
    max_batch_tracks: int = 5000
    # Batch nhỏ: trên CPU batch lớn không nhanh hơn (đo: 32 track ~ 32 × 1 track),
    # mà lại giữ khoá model lâu -> request lẻ phải chờ. 4 track ≈ 1–3 s/lượt.
    inference_batch_size: int = 4

    # Storage — PostgreSQL (docker-compose service `db`). Khai báo từng phần để
    # mật khẩu có ký tự đặc biệt (@, :, /) không làm hỏng URL; `database_url`
    # (nếu đặt) được ưu tiên.
    database_url: str | None = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "platereader"
    db_password: str = "platereader"
    db_name: str = "platereader"
    db_pool_min: int = 1
    db_pool_max: int = 10
    # Chờ connection rảnh tối đa N giây rồi trả 503, thay vì treo request.
    db_pool_timeout_s: float = 10.0
    # Câu SQL chạy quá N ms bị Postgres huỷ (chặn query treo giữ connection).
    db_statement_timeout_ms: int = 15000

    # CORS — React dev server (Vite) mặc định chạy ở 5173.
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return make_conninfo(
            host=self.db_host, port=self.db_port, user=self.db_user,
            password=self.db_password, dbname=self.db_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
