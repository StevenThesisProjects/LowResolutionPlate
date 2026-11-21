"""SQLite log of predictions and user feedback.

Images are never stored: a prediction is logged with the SHA-256 of its input files, the
result, the confidence and the latency, so the log can be analysed (review rate, feedback
accuracy, latency) without keeping licensed data or anything that identifies a vehicle
beyond the plate string the user already sees.
"""
import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id    TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    session_id    TEXT,
    input_sha256  TEXT NOT NULL,
    n_frames      INTEGER NOT NULL,
    plate         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    triage_score  REAL NOT NULL,
    needs_review  INTEGER NOT NULL,
    decoder       TEXT NOT NULL,
    latency_ms    REAL NOT NULL,
    model_version TEXT NOT NULL,
    result_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions (session_id, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL REFERENCES predictions (request_id),
    created_at      TEXT NOT NULL,
    correct         INTEGER NOT NULL,
    corrected_plate TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback (request_id);
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.commit()

    def log_prediction(self, result: Dict[str, Any], input_sha256: str,
                       session_id: Optional[str], triage_score: float) -> None:
        row = (result["request_id"], _now(), session_id, input_sha256, result["n_frames"],
               result["plate"], result["confidence"], triage_score, int(result["needs_review"]),
               result["decoder"], result["latency_ms"], result["model_version"],
               json.dumps(result, ensure_ascii=False))
        with self._lock:
            self._db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self._db.commit()

    def add_feedback(self, request_id: str, correct: bool,
                     corrected_plate: Optional[str]) -> Optional[int]:
        """Store one feedback row. Returns its id, or None if the request id is unknown."""
        with self._lock:
            if self._db.execute("SELECT 1 FROM predictions WHERE request_id = ?",
                                (request_id,)).fetchone() is None:
                return None
            cur = self._db.execute(
                "INSERT INTO feedback (request_id, created_at, correct, corrected_plate) "
                "VALUES (?,?,?,?)", (request_id, _now(), int(correct), corrected_plate))
            self._db.commit()
            return int(cur.lastrowid)

    def history(self, session_id: Optional[str], limit: int = 20) -> List[Dict[str, Any]]:
        query = ("SELECT p.request_id, p.created_at, p.n_frames, p.plate, p.confidence, "
                 "p.needs_review, p.decoder, p.latency_ms, "
                 "(SELECT f.correct FROM feedback f WHERE f.request_id = p.request_id "
                 " ORDER BY f.id DESC LIMIT 1) AS feedback_correct, "
                 "(SELECT f.corrected_plate FROM feedback f WHERE f.request_id = p.request_id "
                 " ORDER BY f.id DESC LIMIT 1) AS corrected_plate "
                 "FROM predictions p ")
        params: tuple = ()
        if session_id:
            query += "WHERE p.session_id = ? "
            params = (session_id,)
        query += "ORDER BY p.created_at DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(query, params + (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["needs_review"] = bool(d["needs_review"])
            if d["feedback_correct"] is not None:
                d["feedback_correct"] = bool(d["feedback_correct"])
            out.append(d)
        return out

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            p = self._db.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(needs_review), 0) AS review, "
                "AVG(latency_ms) AS mean_latency FROM predictions").fetchone()
            # Latest feedback per request only, so a corrected vote is not counted twice
            f = self._db.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS correct FROM feedback f "
                "WHERE f.id = (SELECT MAX(id) FROM feedback g WHERE g.request_id = f.request_id)"
            ).fetchone()
            by_review = self._db.execute(
                "SELECT p.needs_review AS review, COUNT(*) AS n, SUM(f.correct) AS correct "
                "FROM feedback f JOIN predictions p ON p.request_id = f.request_id "
                "WHERE f.id = (SELECT MAX(id) FROM feedback g WHERE g.request_id = f.request_id) "
                "GROUP BY p.needs_review").fetchall()
        groups = {("needs_review" if r["review"] else "auto"):
                  {"feedback": r["n"], "correct": r["correct"],
                   "accuracy": (r["correct"] / r["n"]) if r["n"] else None} for r in by_review}
        return {
            "predictions": p["n"],
            "needs_review": p["review"],
            "review_rate": (p["review"] / p["n"]) if p["n"] else None,
            "mean_latency_ms": p["mean_latency"],
            "feedback": f["n"],
            "feedback_correct": f["correct"],
            "feedback_accuracy": (f["correct"] / f["n"]) if f["n"] else None,
            "feedback_by_triage": groups,
        }
