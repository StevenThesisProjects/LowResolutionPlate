"""End-to-end API tests: real checkpoint + real PostgreSQL.

Cần Postgres đang chạy (`docker compose up -d db`); thiếu DB / checkpoint /
dataset thì test tự skip.

    PLATEREADER_DB_HOST=localhost PLATEREADER_DB_PASSWORD='<POSTGRES_PASSWORD>' \
        backend/.venv/bin/python -m pytest backend/tests -q
"""

import glob
import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import PROJECT_ROOT
from backend.app.main import app

API = "/api/v1"
SUBMISSION = PROJECT_ROOT / "results/multi-seed/s1_mf_sr_ocr/submission_s1_seed42.txt"
VAL_FILE = PROJECT_ROOT / "dataset/val_tracks.json"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        health = c.get(f"{API}/health").json()
        if health["status"] != "ok":
            pytest.skip(f"stack not ready: {health}")
        yield c


@pytest.fixture(scope="module")
def val_tracks():
    if not VAL_FILE.exists() or not SUBMISSION.exists():
        pytest.skip("dataset / submission file not available")
    val = json.loads(VAL_FILE.read_text())[:8]
    dirs = {
        Path(p).parent.name: Path(p).parent
        for p in glob.glob(str(PROJECT_ROOT / "dataset/data/train/Scenario-B/*/*/annotations.json"))
    }
    offline = {
        line.split(",")[0]: line.strip().split(",", 1)[1].split(";")[0]
        for line in SUBMISSION.read_text().splitlines()
    }
    return [(t, dirs[t], offline[t]) for t in val]


def _frames(track_dir: Path, n: int = 5):
    return [
        ("files", (f"lr-00{i}.jpg", (track_dir / f"lr-00{i}.jpg").read_bytes(), "image/jpeg"))
        for i in range(1, n + 1)
    ]


def test_health_and_model(client):
    model = client.get(f"{API}/model").json()
    assert model["architecture"]["use_sr"] is True
    assert model["max_frames"] == 5


def test_single_prediction_matches_offline(client, val_tracks):
    track_id, track_dir, offline = val_tracks[0]
    r = client.post(f"{API}/predictions?include_sr=true", files=_frames(track_dir), data={"track_id": track_id})
    assert r.status_code == 201
    body = r.json()
    assert r.headers["location"].endswith(body["id"])
    assert body["plate"] == offline
    assert len(body["char_conf"]) == len(body["plate"])
    assert body["status"] in ("accepted", "needs_review")
    assert len(body["sr_images"]) == 5 and body["sr_images"][0]["width"] == 256

    sug = client.get(f"{API}/predictions/{body['id']}/suggestions").json()
    assert sug["suggestions"][0]["plate"] == body["plate"]

    rev = client.put(f"{API}/predictions/{body['id']}/review", json={"plate": "abc 1d23", "reviewer": "test"})
    assert rev.json()["final_plate"] == "ABC1D23" and rev.json()["status"] == "reviewed"
    assert client.delete(f"{API}/predictions/{body['id']}").status_code == 204


def test_fewer_frames_are_padded(client, val_tracks):
    _, track_dir, _ = val_tracks[1]
    r = client.post(f"{API}/predictions", files=_frames(track_dir, 2))
    assert r.status_code == 201 and r.json()["num_frames"] == 2


@pytest.mark.parametrize(
    ("files", "status", "code"),
    [
        ([("files", ("x.png", b"not-an-image", "image/png"))], 422, "invalid_image"),
        ([("files", ("x.txt", b"hello", "text/plain"))], 415, "unsupported_media_type"),
        ([("files", (f"{i}.png", b"x", "image/png")) for i in range(6)], 422, "invalid_frame_count"),
    ],
)
def test_prediction_errors(client, files, status, code):
    r = client.post(f"{API}/predictions", files=files)
    assert r.status_code == status
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == code


def test_not_found_and_bad_id(client):
    assert client.get(f"{API}/predictions/00000000-0000-0000-0000-000000000000").json()["code"] == "prediction_not_found"
    assert client.get(f"{API}/predictions/not-a-uuid").status_code == 422


def test_batch_parity(client, val_tracks):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for track_id, track_dir, _ in val_tracks:
            for f in track_dir.iterdir():
                z.write(f, f"{track_id}/{f.name}")
    r = client.post(f"{API}/batches", files={"file": ("val8.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 202
    batch_id = r.json()["id"]
    for _ in range(120):
        batch = client.get(f"{API}/batches/{batch_id}").json()
        if batch["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert batch["status"] == "completed", batch
    assert batch["labeled_tracks"] == len(val_tracks)

    csv_text = client.get(f"{API}/batches/{batch_id}/export").text
    plates = {line.split(",")[0]: line.split(",")[1] for line in csv_text.strip().splitlines()[1:]}
    assert plates == {t: offline for t, _, offline in val_tracks}
    assert client.delete(f"{API}/batches/{batch_id}").status_code == 204


def test_bad_zip(client):
    r = client.post(f"{API}/batches", files={"file": ("x.zip", b"nope", "application/zip")})
    assert r.status_code == 422 and r.json()["code"] == "invalid_zip"
