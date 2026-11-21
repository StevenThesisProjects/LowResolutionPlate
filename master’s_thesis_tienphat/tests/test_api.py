"""HTTP API: normal requests and every error path (roadmap week 9: >= 8 error cases)."""
import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from conftest import lr_files, track_dir

TRACK = "track_14926"          # val 999 (never trained on), label GGB8A08


def as_files(named, mime="image/jpeg"):
    return [("files", (n, d, mime)) for n, d in named]


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------- normal use
def test_health_and_model(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"]
    m = client.get("/model").json()
    assert m["sha256_verified"] is True
    assert m["parameters"] == 31_077_676
    assert set(m["decoders"]) == {"greedy", "beam", "constrained"}


def test_predict_five_frames(client):
    r = client.post("/predict", files=as_files(lr_files(TRACK)))
    assert r.status_code == 200
    j = r.json()
    assert j["plate"] == "GGB8A08" and j["layout"] == "Mercosur"
    assert j["n_frames"] == 5 and len(j["frame_weights"]) == 5 and len(j["sharpness"]) == 5
    assert abs(sum(j["frame_weights"]) - 1.0) < 1e-4
    assert len(j["char_conf"]) == 7 and len(j["top3"]) == 3
    assert j["top3"][0]["plate"] == j["plate"]
    assert j["triage"]["threshold"] is not None
    assert j["model_version"].startswith("restran-recipe-s42-ep30")


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_predict_fewer_frames_warns(client, n):
    r = client.post("/predict", files=as_files(lr_files(TRACK)[:n]))
    assert r.status_code == 200
    j = r.json()
    assert j["n_frames"] == n and len(j["frame_weights"]) == n
    assert any(f"Chỉ có {n} frame" in w for w in j["warnings"])
    assert j["expected_accuracy"] is not None


def test_predict_track_zip_uses_lr_only_and_reads_label(client):
    d = track_dir(TRACK)
    entries = [(f"{TRACK}/{p.name}", p.read_bytes()) for p in sorted(d.iterdir())]
    r = client.post("/predict", files=[("files", ("track.zip", make_zip(entries), "application/zip"))])
    assert r.status_code == 200
    j = r.json()
    assert j["n_frames"] == 5 and all(f["name"].startswith("lr-") for f in j["frames"])
    assert j["reference_plate"] == "GGB8A08"


@pytest.mark.parametrize("decoder", ["greedy", "beam", "constrained"])
def test_every_decoder(client, decoder):
    j = client.post(f"/predict?decoder={decoder}&top_k=5", files=as_files(lr_files(TRACK))).json()
    assert j["decoder"] == decoder and j["plate"] == "GGB8A08" and len(j["top3"]) <= 5


def test_feedback_history_stats(client):
    j = client.post("/predict?session_id=s1", files=as_files(lr_files(TRACK))).json()
    fb = client.post("/feedback", json={"request_id": j["request_id"], "correct": False,
                                       "corrected_plate": "ggb8a09"})
    assert fb.status_code == 200
    hist = client.get("/history?session_id=s1").json()
    assert hist[0]["request_id"] == j["request_id"]
    assert hist[0]["feedback_correct"] is False and hist[0]["corrected_plate"] == "GGB8A09"
    assert client.get("/stats").json()["feedback"] >= 1


def test_same_input_same_output(client):
    a = client.post("/predict", files=as_files(lr_files(TRACK))).json()
    b = client.post("/predict", files=as_files(lr_files(TRACK))).json()
    assert a["plate"] == b["plate"] and a["confidence"] == b["confidence"]
    assert a["frame_weights"] == b["frame_weights"]


def test_odd_inputs_still_answer(client):
    """Grey, rotated and non-plate images are valid images: the API answers, the triage
    flags them (none of these should be read as a confident plate)."""
    rgb = [cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR) for _, d in lr_files(TRACK)]
    rotated = [png(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)) for im in rgb]
    j = client.post("/predict", files=as_files([(f"r{i}.png", d) for i, d in enumerate(rotated)],
                                               "image/png")).json()
    assert j["needs_review"] and any("tỉ lệ" in w for w in j["warnings"])
    noise = np.random.default_rng(0).integers(0, 255, (17, 47, 3), dtype=np.uint8)
    j = client.post("/predict", files=as_files([("noise.png", png(noise))], "image/png")).json()
    assert j["needs_review"]


# ---------------------------------------------------------------- errors
def err(r):
    return r.status_code, r.json()["error"]["code"]


def test_error_no_files(client):
    assert err(client.post("/predict")) == (400, "no_files")


def test_error_too_many_files(client):
    files = lr_files(TRACK)
    assert err(client.post("/predict", files=as_files(files + files[:1]))) == (400, "too_many_files")


def test_error_not_an_image(client):
    r = client.post("/predict", files=[("files", ("plate.txt", b"ABC1D23", "text/plain"))])
    assert err(r) == (415, "unsupported_type")


def test_error_gif_renamed_jpg(client):
    r = client.post("/predict", files=[("files", ("plate.jpg", b"GIF89a\x01\x00", "image/jpeg"))])
    assert err(r) == (415, "unsupported_type")


def test_error_corrupt_image(client):
    data = lr_files(TRACK)[0][1]
    assert err(client.post("/predict", files=as_files([("cut.jpg", data[:200])]))) == (
        422, "corrupt_image")


def test_error_empty_file(client):
    assert err(client.post("/predict", files=as_files([("empty.jpg", b"")]))) == (422, "empty_file")


def test_error_file_too_large(client):
    big = b"\xff\xd8\xff" + b"\0" * (2 * 1024 * 1024 + 10)
    assert err(client.post("/predict", files=as_files([("big.jpg", big)]))) == (413, "file_too_large")


def test_error_image_dimensions_too_large(client):
    wide = png(np.zeros((8, 4000, 3), np.uint8))
    r = client.post("/predict", files=as_files([("wide.png", wide)], "image/png"))
    assert err(r) == (413, "image_too_large")


def test_error_image_too_small(client):
    r = client.post("/predict", files=as_files([("dot.png", png(np.zeros((2, 3, 3), np.uint8)))],
                                               "image/png"))
    assert err(r) == (422, "image_too_small")


def test_error_zip_invalid(client):
    r = client.post("/predict", files=[("files", ("t.zip", b"PK\x03\x04garbage", "application/zip"))])
    assert err(r) == (400, "zip_invalid")


def test_error_zip_without_images(client):
    z = make_zip([("notes.txt", b"hello"), ("annotations.json", json.dumps({"plate_text": "X"}))])
    r = client.post("/predict", files=[("files", ("t.zip", z, "application/zip"))])
    assert err(r) == (400, "zip_no_images")


def test_error_zip_with_too_many_frames(client):
    files = lr_files(TRACK)
    z = make_zip([(f"lr-{i:03d}.jpg", d) for i, (_, d) in enumerate(files + files, 1)])
    r = client.post("/predict", files=[("files", ("t.zip", z, "application/zip"))])
    assert err(r) == (400, "too_many_files")


def test_error_zip_mixed_with_images(client):
    files = lr_files(TRACK)
    z = make_zip([(n, d) for n, d in files])
    r = client.post("/predict", files=[("files", ("t.zip", z, "application/zip"))] + as_files(files[:1]))
    assert err(r) == (400, "mixed_upload")


def test_error_bad_parameters(client):
    files = as_files(lr_files(TRACK))
    assert err(client.post("/predict?decoder=viterbi", files=files)) == (422, "invalid_parameter")
    assert err(client.post("/predict?top_k=9", files=files)) == (422, "invalid_parameter")
    assert err(client.post("/predict?session_id=a%20b", files=files)) == (422, "invalid_parameter")


def test_error_feedback_unknown_request(client):
    r = client.post("/feedback", json={"request_id": "0" * 32, "correct": True})
    assert err(r) == (404, "not_found")


def test_error_feedback_bad_plate(client):
    r = client.post("/feedback", json={"request_id": "0" * 32, "correct": False,
                                       "corrected_plate": "ABC-123"})
    assert err(r) == (422, "invalid_parameter")


def test_error_api_key(client, monkeypatch):
    monkeypatch.setenv("MFPLATE_API_KEY", "secret")
    files = as_files(lr_files(TRACK))
    assert err(client.post("/predict", files=files)) == (401, "unauthorized")
    assert client.post("/predict", files=files, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200        # health stays open


# ---------------------------------------------------------------- web UI and demo samples
def test_root_serves_web_ui_or_docs(client):
    from mfplate.api import WEBUI_DIST
    r = client.get("/", follow_redirects=False)
    if (WEBUI_DIST / "index.html").exists():
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert '<div id="root">' in r.text
    else:
        assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_samples_list_and_download(client):
    from mfplate.api import SAMPLES_DIR
    items = client.get("/samples").json()
    names = sorted(p.name for p in SAMPLES_DIR.glob("*.zip")) if SAMPLES_DIR.is_dir() else []
    assert [i["name"] for i in items] == names
    if names:
        r = client.get(f"/samples/{names[0]}")
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert r.content[:4] == b"PK\x03\x04"


@pytest.mark.parametrize("name", ["nope.zip", "..%2Fserving_config.json", "README.md", "%2E%2E%2Fx.zip"])
def test_samples_reject_unknown_and_traversal(client, name):
    r = client.get(f"/samples/{name}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
