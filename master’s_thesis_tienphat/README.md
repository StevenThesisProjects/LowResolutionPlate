# MF-PlateReader: phần luận văn thạc sĩ (Nguyễn Tiến Phát)

Thư mục này hiện thực roadmap 12 tuần (`documents/Roadmap.md`) trên **code mới** của repo `restran`. Roadmap được viết dựa trên Pull Request cũ, nên các hạng mục được đối chiếu sang những gì đã làm sau review của thầy, chứ không train lại theo roadmap. Giải thích đầy đủ ở [`reports/00_mo_dau.md`](reports/00_mo_dau.md).

Model đã đóng băng: công thức huấn luyện #50, run lock-box seed 42, epoch 30, SHA-256 `9754ebb1…7bba`. Thư mục này **không train thêm gì**.

## 1. Nội dung

```
masters_thesis_tienphat/
├── reports/      00_mo_dau.md + tuan_01.md … tuan_12.md (đối chiếu roadmap → phần thay thế)
├── mfplate/      gói phục vụ: API FastAPI, engine, decoder, tiền xử lý, upload, SQLite
├── webui/       giao diện web chính (React + Vite), build ra webui/dist, API phục vụ ở /
├── ui/app.py     giao diện Gradio cũ (dự phòng, client của API)
├── scripts/      đóng băng, cache offline, decoder, audit lỗi, ngưỡng, lát cắt, parity, latency, E2E, mẫu demo
├── tests/        67 test pytest (decoder, tiền xử lý, API, web UI)
├── artifacts/    kết quả: cấu hình phục vụ, quyết định, bảng, OpenAPI, báo cáo test
├── model/        checkpoint app dùng (bản chép từ restran/results_new/, cùng SHA-256; không lên git)
├── docs/         run_guide.md (cách chạy), api.md, demo_script.md, user_test.md
├── docker/       Dockerfile CPU (build cả web UI), docker-compose (api + tunnel; gradio tuỳ chọn)
├── samples/      3 track val 999 cho demo (dữ liệu LRLPR-26: không công bố)
└── var/          (tự sinh) cache log-prob, SQLite; không cần giữ
```

Thư mục này nằm **cùng cấp** với repo train `restran/` (tách ra ngày 21/11/2026; trước đó nằm bên trong `restran/`):

```
NewClaude/
├── restran/                   repo train: src/, configs/, results_new/ (các checkpoint train), dataset/, documents/
└── masters_thesis_tienphat/   thư mục này; model/ chứa checkpoint app dùng
```

**Checkpoint app dùng nằm trong [`model/`](model/README.md)**: bản chép của `restran/results_new/restran_lockbox_combo3_s42_ep30.pth`, cùng SHA-256 `9754ebb1…7bba`. File không đưa lên git (135 MB); `python3 scripts/freeze_serving_model.py` kiểm SHA-256 rồi chép lại. Code mạng (`src/`, `configs/`) vẫn được import từ `../restran`, không chép sang đây. `mfplate/settings.py` tự tìm repo theo thứ tự: biến `MFPLATE_REPO_ROOT`, thư mục `../restran`, rồi thư mục cha (bố cục cũ và trong Docker). Repo đặt ở chỗ khác thì chạy với `MFPLATE_REPO_ROOT=<đường dẫn tới restran>`.

## 2. Chạy app (máy local)

Hướng dẫn chi tiết từng bước, kèm các lỗi thường gặp: [`docs/run_guide.md`](docs/run_guide.md).

```bash
cd masters_thesis_tienphat
python3 -m pip install --user -r requirements.txt
(cd webui && npm install && npm run build)          # build web UI một lần (cần Node 18+)
python3 -m uvicorn mfplate.api:app --port 8000      # web UI: http://127.0.0.1:8000 · API docs: /docs
python3 -m pytest tests                             # 67 test
```

- Mở thẳng một track mẫu: `http://127.0.0.1:8000/?sample=review_track_22259.zip` (tên file trong `samples/`).
- Sửa giao diện có tự tải lại: chạy API như trên, rồi `cd webui && npm run dev` và mở `http://localhost:5173` (dùng chữ `localhost`; Vite chuyển các lời gọi API sang cổng 8000).
- Giao diện Gradio cũ vẫn chạy được: `python3 ui/app.py` → `http://127.0.0.1:7860`.
- Thử trên điện thoại cùng Wi-Fi: chạy uvicorn với `--host 0.0.0.0`, mở `http://<IP máy>:8000`.

Biến môi trường (đều tuỳ chọn):

| Biến | Tác dụng |
|---|---|
| `MFPLATE_REPO_ROOT` | Đường dẫn repo `restran` (mặc định `../restran`) |
| `MFPLATE_API_KEY` | Bật khoá API (header `X-API-Key`) |
| `MFPLATE_DB` | Đường dẫn file SQLite |
| `MFPLATE_THREADS` | Số luồng CPU cho PyTorch |
| `MFPLATE_API_URL` | Địa chỉ API mà UI Gradio gọi tới |
| `MFPLATE_WEBUI_DIST`, `MFPLATE_SAMPLES_DIR` | Thư mục web UI đã build và thư mục track mẫu (mặc định `webui/dist`, `samples/`) |
| `MFPLATE_CORS_ORIGINS` | Danh sách origin được phép, nếu sau này dùng frontend trình duyệt |

## 3. Docker

```bash
# từ thư mục masters_thesis_tienphat/ (repo restran/ phải nằm cùng cấp)
docker compose -f docker/docker-compose.yml up --build
# URL riêng tư cho điện thoại (Cloudflare quick tunnel):
MFPLATE_API_KEY=<bí mật> docker compose -f docker/docker-compose.yml --profile tunnel up
```

Build context là thư mục cha chứa cả `restran/` và `masters_thesis_tienphat/`; `docker/Dockerfile.dockerignore` chỉ gửi đúng các file image cần. Dockerfile chưa được build thử trên máy này (Docker daemon không chạy lúc làm). Xem Tuần 9.

## 4. Chạy lại toàn bộ phép đo

Thứ tự này đúng với thứ tự đã chạy ngày 20/11: mọi quyết định chốt trên val 999 trước, lock-box chỉ xem sau.

```bash
cd masters_thesis_tienphat
python3 scripts/freeze_serving_model.py                    # → artifacts/serving_config.json
python3 scripts/offline_cache.py --split val --runs combo3_s42 combo3_s43 combo3_s7 combo3_s2026
python3 scripts/offline_cache.py --split val --runs combo3_s42 --subsets
python3 scripts/eval_decoders.py                           # → results_decode.csv, decoder_decision.json
python3 scripts/syntax_audit.py                            # → error_syntax_audit.md
python3 scripts/calibrate_threshold.py                     # → threshold_calibration.json
# --- chỉ sau khi các quyết định đã ghi ra file ---
python3 scripts/offline_cache.py --split lockbox --runs combo3_s42 combo3_s43 combo3_s7 combo3_s2026
python3 scripts/offline_cache.py --split lockbox --runs combo3_s42 --subsets
python3 scripts/eval_decoders.py --lockbox                 # mô tả
python3 scripts/calibrate_threshold.py --lockbox           # → triage_lockbox.json (mô tả)
python3 scripts/slice_analysis.py                          # → slice_sharpness.csv
python3 scripts/parity_check.py --lockbox                  # → parity_final.json
python3 scripts/make_samples.py && python3 scripts/export_openapi.py
python3 -m uvicorn mfplate.api:app --port 8000 &
python3 scripts/bench_latency.py && python3 scripts/run_e2e.py
```

Cache trên CPU (Apple M4) mất khoảng 35 giây mỗi lượt qua 1.000 track; phần tập con frame mất khoảng 15 phút mỗi tập.

## 5. Kết quả chính

| | Kết quả | File |
|---|---|---|
| Parity app so với offline | 0/999 track lệch; CPU = GPU (val 76,68%, lock-box 79,90%, 0/1.000 track khác) | `artifacts/parity_final.json` |
| Decoder | greedy 77,66 / beam 77,66 / SC-CTC 77,70 (val, 4 seed) → giữ greedy | `artifacts/results_decode.csv` |
| Trần lợi ích SC-CTC | 12/893 lỗi vi phạm cú pháp; 0/893 lỗi decode sửa được | `artifacts/error_syntax_audit.md` |
| Triage (lock-box, 5 frame) | tự động 74,0%, precision 95,4%, giữ lại 167/201 lỗi | `artifacts/triage_lockbox.json` |
| E2E / lỗi | 30/30; 14/14 | `artifacts/test_report.md`, `artifacts/error_cases.md` |
| Latency CPU (M4) | p50 21,9 ms, p95 24,1 ms | `artifacts/latency_summary.json` |
