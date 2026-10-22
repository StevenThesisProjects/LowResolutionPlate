"""PostgreSQL persistence: predictions, reviews and batch jobs (schema.sql)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

JSON_COLUMNS = ("char_conf", "suggestions")

# Prediction + its latest review, flattened into one row.
_PREDICTION_SELECT = """
    SELECT p.*, r.plate AS review_plate, r.reviewer, r.note AS review_note,
           r.created_at AS reviewed_at
    FROM predictions p
    LEFT JOIN LATERAL (
        SELECT plate, reviewer, note, created_at FROM reviews
        WHERE prediction_id = p.id ORDER BY created_at DESC, id DESC LIMIT 1
    ) r ON TRUE
"""

_BATCH_SELECT = """
    SELECT b.*,
           COUNT(p.id) FILTER (WHERE p.status = 'accepted')         AS accepted,
           COUNT(p.id) FILTER (WHERE p.status = 'needs_review')     AS needs_review,
           COUNT(p.id) FILTER (WHERE p.status = 'reviewed')         AS reviewed,
           COUNT(p.id) FILTER (WHERE p.ground_truth IS NOT NULL)    AS labeled_tracks,
           COUNT(p.id) FILTER (WHERE p.ground_truth = p.plate)      AS correct
    FROM batches b LEFT JOIN predictions p ON p.batch_id = b.id
"""


class Database:
    def __init__(
        self,
        url: str,
        min_size: int = 1,
        max_size: int = 10,
        timeout_s: float = 10.0,
        statement_timeout_ms: int = 15000,
    ) -> None:
        self.pool = ConnectionPool(
            url, min_size=min_size, max_size=max_size, timeout=timeout_s,
            kwargs={
                "row_factory": dict_row,
                "autocommit": True,
                "options": f"-c statement_timeout={statement_timeout_ms}",
                "application_name": "platereader-api",
            },
            open=True,
        )
        self.pool.wait(timeout=30)
        with self.pool.connection() as conn:
            conn.execute(SCHEMA_FILE.read_text())

    def close(self) -> None:
        self.pool.close()

    def ping(self) -> bool:
        try:
            with self.pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 — reported via /health
            return False

    def _all(self, sql: str, params: Any = ()) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: Any = ()) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchone()

    # ── predictions ────────────────────────────────────────────────────────

    def insert_predictions(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert and return the stored rows (with id / created_at filled in)."""
        if not rows:
            return []
        cols = list(rows[0])
        sql = (
            f"INSERT INTO predictions ({', '.join(cols)}) "
            f"VALUES ({', '.join('%s' for _ in cols)}) RETURNING id"
        )
        params = [tuple(Jsonb(r[c]) if c in JSON_COLUMNS else r[c] for c in cols) for r in rows]
        ids = []
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany(sql, params, returning=True)
            while True:
                ids.append(cur.fetchone()["id"])
                if not cur.nextset():
                    break
        return self._all(f"{_PREDICTION_SELECT} WHERE p.id = ANY(%s) ORDER BY p.track_id", (ids,))

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        return self._one(f"{_PREDICTION_SELECT} WHERE p.id = %s", (prediction_id,))

    def list_predictions(
        self, *, status: str | None, batch_id: str | None, page: int, page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = [], []
        if status:
            where.append("p.status = %s")
            params.append(status)
        if batch_id:
            where.append("p.batch_id = %s")
            params.append(batch_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        total = self._one(f"SELECT COUNT(*) AS n FROM predictions p {clause}", params)["n"]
        rows = self._all(
            f"{_PREDICTION_SELECT} {clause} ORDER BY p.created_at DESC, p.track_id LIMIT %s OFFSET %s",
            [*params, page_size, (page - 1) * page_size],
        )
        return rows, total

    def batch_predictions(self, batch_id: str) -> list[dict[str, Any]]:
        return self._all(f"{_PREDICTION_SELECT} WHERE p.batch_id = %s ORDER BY p.track_id", (batch_id,))

    def add_review(self, prediction_id: str, plate: str, reviewer: str | None, note: str | None) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO reviews (prediction_id, plate, reviewer, note) VALUES (%s, %s, %s, %s)",
                (prediction_id, plate, reviewer, note),
            )
            conn.execute("UPDATE predictions SET status = 'reviewed' WHERE id = %s", (prediction_id,))

    def delete_prediction(self, prediction_id: str) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM predictions WHERE id = %s", (prediction_id,)).rowcount > 0

    # ── batches ────────────────────────────────────────────────────────────

    def insert_batch(self, filename: str, decode_mode: str, total_tracks: int) -> str:
        row = self._one(
            "INSERT INTO batches (filename, decode_mode, total_tracks) VALUES (%s, %s, %s) RETURNING id",
            (filename, decode_mode, total_tracks),
        )
        return str(row["id"])

    def update_batch(self, batch_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k} = %s" for k in fields)
        with self.pool.connection() as conn:
            conn.execute(f"UPDATE batches SET {sets} WHERE id = %s", (*fields.values(), batch_id))

    def finish_batch(self, batch_id: str, status: str, error: str | None = None) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE batches SET status = %s, error = %s, finished_at = now() WHERE id = %s",
                (status, error, batch_id),
            )

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        return self._one(f"{_BATCH_SELECT} WHERE b.id = %s GROUP BY b.id", (batch_id,))

    def list_batches(self, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        total = self._one("SELECT COUNT(*) AS n FROM batches")["n"]
        rows = self._all(
            f"{_BATCH_SELECT} GROUP BY b.id ORDER BY b.created_at DESC LIMIT %s OFFSET %s",
            (page_size, (page - 1) * page_size),
        )
        return rows, total

    def delete_batch(self, batch_id: str) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM batches WHERE id = %s", (batch_id,)).rowcount > 0

    def fail_interrupted_batches(self) -> None:
        """Jobs left `queued`/`running` by a previous process can never finish."""
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE batches SET status = 'failed', finished_at = now(),"
                " error = 'Server restarted before the job finished.'"
                " WHERE status IN ('queued', 'running')"
            )
