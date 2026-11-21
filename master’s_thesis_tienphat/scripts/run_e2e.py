#!/usr/bin/env python3
"""End-to-end cases against the running API (roadmap weeks 9-10).

  * e2e    : 30 cases = 20 val tracks spread over layout x sharpness quartile, sent as
             images or as a zip, + 10 edge cases (1-4 frames, grey, rotated, upside down,
             noise, a blank card, the same frame five times). A case passes when the API
             answers 200 with a complete response (MVP criterion 2: 30/30 without a crash).
             Reading accuracy is recorded, not required.
  * errors : >= 8 malformed requests; each must get its documented status and error code.

Start the server first, from masters_thesis_tienphat/ (python3 -m uvicorn mfplate.api:app --port 8000):
    python3 scripts/run_e2e.py --url http://127.0.0.1:8000

Writes artifacts/e2e_results.json, artifacts/test_report.md and artifacts/error_cases.md.
Tracks come from val 999 (never trained on); none is from the lock-box.
"""
import argparse
import datetime as dt
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mfplate.settings import ARTIFACTS_DIR, THESIS_DIR  # noqa: E402

CACHE = THESIS_DIR / "var" / "cache"
REQUIRED = {"request_id", "plate", "confidence", "char_conf", "top3", "layout", "frame_weights",
            "sharpness", "needs_review", "latency_ms", "model_version"}


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def zip_of(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in entries:
            z.writestr(n, d)
    return buf.getvalue()


def pick_tracks(meta, seed=2026):
    """20 tracks: 3 per (Mercosur, sharpness quartile) and 2 per (Brazilian, quartile)."""
    sharp = meta["sharpness"].mean(axis=1)
    q = np.digitize(sharp, np.quantile(sharp, [0.25, 0.5, 0.75]))      # 0..3, 0 = blurriest
    rng = np.random.default_rng(seed)
    chosen = []
    for lay, k in (("Mercosur", 3), ("Brazilian", 2)):
        for quart in range(4):
            idx = np.nonzero((meta["layouts"] == lay) & (q == quart))[0]
            chosen += [(int(i), lay, quart + 1) for i in rng.choice(idx, size=k, replace=False)]
    return chosen


def e2e_cases(meta):
    cases = []
    for n, (i, lay, quart) in enumerate(pick_tracks(meta)):
        paths = [Path(p) for p in meta["paths"][i]]
        label = str(meta["labels"][i])
        if n % 2 == 0:
            files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths]
            how = "5 ảnh"
        else:
            entries = [(p.name, p.read_bytes()) for p in sorted(paths[0].parent.iterdir())]
            files = [("files", ("track.zip", zip_of(entries), "application/zip"))]
            how = "zip track (lr + hr + annotations)"
        cases.append({"id": f"T{n + 1:02d}", "kind": "track", "track": str(meta["track_ids"][i]),
                      "layout": lay, "sharpness_quartile": quart, "label": label, "input": how,
                      "files": files})
    # Edge cases built from one val track
    i0 = cases[0]
    base_paths = [Path(p) for p in meta["paths"][int(np.nonzero(meta["track_ids"] == i0["track"])[0][0])]]
    raw = [p.read_bytes() for p in base_paths]
    imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in raw]
    jpg = lambda name, b: ("files", (name, b, "image/jpeg"))  # noqa: E731
    pngf = lambda name, b: ("files", (name, b, "image/png"))  # noqa: E731
    edge = [
        ("E01", "1 frame", [jpg("lr-001.jpg", raw[0])]),
        ("E02", "2 frame", [jpg(f"lr-00{k + 1}.jpg", raw[k]) for k in range(2)]),
        ("E03", "3 frame", [jpg(f"lr-00{k + 1}.jpg", raw[k]) for k in range(3)]),
        ("E04", "4 frame", [jpg(f"lr-00{k + 1}.jpg", raw[k]) for k in range(4)]),
        ("E05", "ảnh xám 1 kênh (PNG)",
         [pngf(f"g{k}.png", png(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))) for k, im in enumerate(imgs)]),
        ("E06", "ảnh xoay 90°",
         [pngf(f"r{k}.png", png(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE))) for k, im in enumerate(imgs)]),
        ("E07", "ảnh lộn ngược 180°",
         [pngf(f"u{k}.png", png(cv2.rotate(im, cv2.ROTATE_180))) for k, im in enumerate(imgs)]),
        ("E08", "không phải biển số: nhiễu ngẫu nhiên",
         [pngf("noise.png", png(np.random.default_rng(0).integers(0, 255, (18, 48, 3), dtype=np.uint8)))]),
        ("E09", "không phải biển số: thẻ trắng có viền",
         [pngf("card.png", png(cv2.rectangle(np.full((18, 48, 3), 235, np.uint8), (1, 1), (46, 16), (40, 40, 40), 1)))]),
        ("E10", "cùng 1 frame lặp 5 lần", [jpg(f"same{k}.jpg", raw[0]) for k in range(5)]),
    ]
    for cid, desc, files in edge:
        cases.append({"id": cid, "kind": "edge", "track": i0["track"], "label": i0["label"],
                      "input": desc, "files": files})
    return cases


ERROR_CASES = [
    ("không gửi file nào", [], 400, "no_files"),
    ("6 ảnh (quá 5)", "six", 400, "too_many_files"),
    ("file .txt", [("files", ("plate.txt", b"ABC1D23", "text/plain"))], 415, "unsupported_type"),
    ("ảnh GIF đổi đuôi .jpg", [("files", ("p.jpg", b"GIF89a\x01\x00\x01\x00", "image/jpeg"))], 415, "unsupported_type"),
    ("JPEG bị cắt cụt (file hỏng)", "cut", 422, "corrupt_image"),
    ("file rỗng", [("files", ("e.jpg", b"", "image/jpeg"))], 422, "empty_file"),
    ("ảnh > 2 MB", [("files", ("big.jpg", b"\xff\xd8\xff" + b"\0" * (2 * 1024 * 1024 + 1), "image/jpeg"))], 413, "file_too_large"),
    ("ảnh 4000 px chiều rộng", "wide", 413, "image_too_large"),
    ("ảnh 2×3 px", "tiny", 422, "image_too_small"),
    ("zip hỏng", [("files", ("t.zip", b"PK\x03\x04broken", "application/zip"))], 400, "zip_invalid"),
    ("zip không có ảnh", "zip_noimg", 400, "zip_no_images"),
    ("gửi lẫn zip và ảnh", "mixed", 400, "mixed_upload"),
    ("decoder không tồn tại", "bad_decoder", 422, "invalid_parameter"),
]


def build_error_request(spec, raw):
    jpg = lambda name, b: ("files", (name, b, "image/jpeg"))  # noqa: E731
    params = {}
    if spec == "six":
        files = [jpg(f"lr-{k}.jpg", raw[k % len(raw)]) for k in range(6)]
    elif spec == "cut":
        files = [jpg("cut.jpg", raw[0][:200])]
    elif spec == "wide":
        files = [("files", ("wide.png", png(np.zeros((8, 4000, 3), np.uint8)), "image/png"))]
    elif spec == "tiny":
        files = [("files", ("dot.png", png(np.zeros((2, 3, 3), np.uint8)), "image/png"))]
    elif spec == "zip_noimg":
        files = [("files", ("t.zip", zip_of([("readme.txt", b"x")]), "application/zip"))]
    elif spec == "mixed":
        files = [("files", ("t.zip", zip_of([("lr-001.jpg", raw[0])]), "application/zip")), jpg("a.jpg", raw[1])]
    elif spec == "bad_decoder":
        files, params = [jpg("lr-001.jpg", raw[0])], {"decoder": "viterbi"}
    else:
        files = spec
    return files, params


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    meta = np.load(CACHE / "val__meta.npz")
    cases = e2e_cases(meta)
    results = []
    with httpx.Client(timeout=60) as client:
        model = client.get(f"{args.url}/model").json()
        for c in cases:
            try:
                r = client.post(f"{args.url}/predict", files=c["files"])
                j = r.json()
                ok = r.status_code == 200 and REQUIRED <= set(j)
                results.append({k: v for k, v in c.items() if k != "files"} | {
                    "status": r.status_code, "passed": ok, "plate": j.get("plate"),
                    "correct": j.get("plate") == c["label"], "confidence": j.get("confidence"),
                    "needs_review": j.get("needs_review"), "reasons": j.get("review_reasons"),
                    "n_frames": j.get("n_frames"), "latency_ms": j.get("latency_ms"),
                    "warnings": j.get("warnings", [])})
            except Exception as e:  # a crash or a non-JSON answer is a failure
                results.append({k: v for k, v in c.items() if k != "files"} | {
                    "status": None, "passed": False, "error": repr(e)})
        raw = [Path(p).read_bytes() for p in meta["paths"][0]]
        errors = []
        for desc, spec, status, code in ERROR_CASES:
            files, params = build_error_request(spec, raw)
            r = client.post(f"{args.url}/predict", files=files or None, params=params)
            body = r.json()
            got = body.get("error", {}).get("code")
            errors.append({"case": desc, "expected_status": status, "expected_code": code,
                           "status": r.status_code, "code": got,
                           "message": body.get("error", {}).get("message"),
                           "passed": r.status_code == status and got == code})
        # Feedback with an unknown id
        r = client.post(f"{args.url}/feedback", json={"request_id": "0" * 32, "correct": True})
        errors.append({"case": "feedback cho request_id không tồn tại", "expected_status": 404,
                       "expected_code": "not_found", "status": r.status_code,
                       "code": r.json()["error"]["code"], "message": r.json()["error"]["message"],
                       "passed": r.status_code == 404})

    now = dt.datetime.now().isoformat(timespec="seconds")
    (ARTIFACTS_DIR / "e2e_results.json").write_text(json.dumps(
        {"created": now, "url": args.url, "model_version": model["model_version"],
         "e2e": results, "errors": errors}, indent=2, ensure_ascii=False) + "\n")

    tracks = [r for r in results if r["kind"] == "track"]
    lines = [f"# Báo cáo test end-to-end (sinh tự động bởi `scripts/run_e2e.py`, {now})", "",
             f"Server: `{args.url}`, model `{model['model_version']}`, decoder mặc định "
             f"`{model['default_decoder']}`. Track lấy từ val 999 (model chưa từng train), "
             "không lấy từ lock-box.", "",
             f"**Kết quả: {sum(r['passed'] for r in results)}/{len(results)} case trả về 200 với "
             "đủ trường JSON.** Tiêu chí MVP số 2 yêu cầu 30/30 không crash.", "",
             f"Đọc đúng {sum(r['correct'] for r in tracks)}/{len(tracks)} track; "
             f"{sum(bool(r.get('needs_review')) for r in tracks)} track bị gắn cờ cần kiểm tra, "
             f"trong đó {sum(bool(r.get('needs_review')) and not r['correct'] for r in tracks)} "
             "track đọc sai.", "",
             "| Case | Input | Track | Layout | Độ nét (Q1 = mờ nhất) | Nhãn | Đọc được | Đúng | Cần kiểm tra | ms | HTTP |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['input']} | {r['track']} | {r.get('layout', '')} | "
                     f"{('Q' + str(r['sharpness_quartile'])) if r.get('sharpness_quartile') else ''} | "
                     f"{r['label']} | {r.get('plate', '')} | {'✓' if r.get('correct') else '✗'} | "
                     f"{'có' if r.get('needs_review') else 'không'} | {r.get('latency_ms', '')} | "
                     f"{r['status']} |")
    edges = [r for r in results if r["kind"] == "edge"]
    lines += ["", "Ghi chú các case biên:", ""]
    for r in edges:
        lines.append(f"- {r['id']} ({r['input']}): {r.get('plate')} — "
                     f"{', '.join(r.get('reasons') or []) or 'không gắn cờ'}"
                     + (f"; cảnh báo: {r['warnings'][0]}" if r.get("warnings") else ""))
    (ARTIFACTS_DIR / "test_report.md").write_text("\n".join(lines) + "\n")

    elines = [f"# Các trường hợp lỗi (sinh tự động bởi `scripts/run_e2e.py`, {now})", "",
              f"**{sum(e['passed'] for e in errors)}/{len(errors)} trường hợp trả đúng mã HTTP và "
              "mã lỗi đã thiết kế.** Server không crash ở trường hợp nào. Mọi lỗi có cùng dạng JSON "
              "`{\"error\": {\"code\", \"message\"}, \"request_id\"}`.", "",
              "| # | Trường hợp | HTTP mong đợi | Mã lỗi mong đợi | HTTP nhận | Mã lỗi nhận | Thông báo | Đạt |",
              "|---|---|---|---|---|---|---|---|"]
    for k, e in enumerate(errors, 1):
        elines.append(f"| {k} | {e['case']} | {e['expected_status']} | `{e['expected_code']}` | "
                      f"{e['status']} | `{e['code']}` | {e['message']} | {'✓' if e['passed'] else '✗'} |")
    (ARTIFACTS_DIR / "error_cases.md").write_text("\n".join(elines) + "\n")
    print(f"e2e: {sum(r['passed'] for r in results)}/{len(results)} đạt; "
          f"lỗi: {sum(e['passed'] for e in errors)}/{len(errors)} đạt")
    print(f"📝 {ARTIFACTS_DIR / 'test_report.md'}\n📝 {ARTIFACTS_DIR / 'error_cases.md'}")


if __name__ == "__main__":
    main()
