# PlateReader LR — Tổng quan hệ thống

> Tài liệu kỹ thuật cho backend của prototype **MF-SR-OCR**, viết bám theo `roadmap-3month-nhutminh.md` (§1, §2, §4).
> Cập nhật: 22/10/2026 · Trạng thái: **backend v1.0.0 chạy được trên Docker**, frontend React sẽ làm sau (Tuần 8).

## 1. Mục tiêu và phạm vi

| Mục | Nội dung (theo roadmap §1, §4) |
| --- | --- |
| Bài toán | Đọc biển số độ phân giải thấp (~46×19 px) từ chuỗi 1–5 frame camera giám sát |
| Model | S1 — Joint MF-SR-OCR, checkpoint `s1_seed42_best.pth` (val 79.78%, 29.6M tham số) |
| Selective prediction | `track_conf ≥ τ` thì `accepted`, ngược lại `needs_review` để người kiểm tra |
| Người dùng | Nhân viên vận hành / điều tra xem lại footage |
| Miền dữ liệu | **Chỉ biển số Brazil** (ICPR 2026 LRLPR), layout `LLLNLNN` / `LLLNNNN`. Biển VN nằm ngoài scope |
| Ngoài scope backend v1 | Xác thực người dùng, video → detect → track (stretch goal), INT8 |

Phần backend đã hoàn thành so với roadmap:

| Roadmap | Yêu cầu | Trạng thái |
| --- | --- | --- |
| Tuần 2 | FastAPI skeleton load S1; `/health`, `/predict` trả đúng 1 track val | ✅ `GET /health`, `POST /predictions` |
| Tuần 7 | `/predict_batch` (zip → CSV), ảnh SR, confidence từng ký tự, status theo τ | ✅ `POST /batches`, `GET /batches/{id}/export` |
| Tuần 7 | Parity API vs offline trên 999 track val | ✅ constrained 999/999 · greedy 999/999 (0 lệch), acc 79.78% |
| Tuần 8 (phần backend) | Lưu sửa tay của người review vào DB | ✅ `PUT /predictions/{id}/review` + **gợi ý** `GET …/suggestions` |
| Tuần 9 (phần backend) | Dockerfile CPU, xử lý lỗi đầu vào | ✅ Docker Compose; 15+ mã lỗi chuẩn hoá |
| Tuần 9 | Tunnel public | ⏳ Service `tunnel` đã khai báo (profile `public`), chưa có token |

## 2. Kiến trúc

```
                ┌────────────────────────────── docker compose (project: platereader) ─────────────────────────────┐
                │                                                                                                  │
 React (Vite)   │   ┌───────────────────────────────┐        ┌──────────────────────┐        ┌──────────────────┐   │
 :5173  ────────┼──►│ api  (FastAPI + Uvicorn)  :8000│──SQL──►│ db  (PostgreSQL 16)  │◄─dump──│ backup (profile) │   │
 (Tuần 8)  HTTP │   │  ├ routers  /api/v1/...        │  pool  │  volume pgdata       │  6h    │  tắt khi dev     │   │
                │   │  ├ services                    │  1–10  │  127.0.0.1:5432      │        │  → ./backups/    │   │
 curl / tests ──┼──►│  │  ├ inference (S1, CPU, lock) │        └──────────────────────┘        └──────────────────┘   │
                │   │  │  ├ decoding (beam top-k)    │                                                               │
                │   │  │  └ batches (1 worker queue) │◄── /models (read-only) = results/multi-seed/s1_mf_sr_ocr/     │
                │   │  └ db (psycopg3 pool)          │                                                               │
                │   └───────────────────────────────┘        ┌──────────────────────┐                               │
                │                ▲                           │ tunnel (cloudflared) │  profile "public" — Tuần 9    │
                │                └───────────────────────────│  URL demo public     │                               │
                │                                            └──────────────────────┘                               │
                └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Nguyên tắc tách biệt:** backend chỉ **import** (đọc) code model ở `src/`, `configs/`, `tools/eval_decode.py`
và **không sửa** file training nào. Tiền xử lý dùng đúng `get_val_transforms`, kiến trúc model được suy ngược từ
`state_dict` bằng `infer_architecture`, nên kết quả API trùng với offline.

### Cấu trúc thư mục

```
master’s_thesis_nhutminh/
├── docker-compose.yml          # db · api  (+ profile: backup, public/tunnel)
├── .env.example                # mẫu biến môi trường (copy thành .env)
├── .dockerignore
├── api-docs/                   # tài liệu này + từng endpoint + index.html
└── backend/
    ├── Dockerfile              # python:3.11-slim + torch CPU
    ├── requirements.txt
    ├── scripts/
    │   ├── backup.sh           # vòng lặp pg_dump định kỳ (service backup)
    │   ├── restore.sh          # khôi phục từ 1 file .dump
    │   ├── parity_check.py     # parity API vs offline (999 track val)
    │   └── load_test.py        # đọc/ghi đồng thời
    ├── app/
    │   ├── main.py             # create_app, lifespan (load model + DB), CORS, error handlers
    │   ├── schema.sql          # DDL PostgreSQL (idempotent)
    │   ├── db.py               # Database: connection pool + toàn bộ SQL
    │   ├── schemas.py          # Pydantic = hợp đồng API
    │   ├── deps.py             # dependency: db, recognizer, giới hạn upload
    │   ├── core/
    │   │   ├── config.py       # Settings đọc biến PLATEREADER_*
    │   │   └── errors.py       # RFC 9457 problem+json
    │   ├── routers/            # system.py · predictions.py · batches.py
    │   └── services/
    │       ├── inference.py    # PlateRecognizer: load checkpoint, tiền xử lý, forward, SR
    │       ├── decoding.py     # top-k constrained beam (gợi ý), char_conf (CTC alignment)
    │       ├── predictions.py  # row ↔ schema, τ → status
    │       └── batches.py      # parse zip, worker chạy nền
    └── tests/                  # test_decoding.py · test_api.py (12 test)
```

## 3. Luồng xử lý

### 3.1. Một track — `POST /predictions` (roadmap `/predict`)

```
Client ──multipart(1–5 ảnh)──► Router
  1. Validate: số frame 1–5, content-type JPG/PNG, ≤ 5 MB, decode được   ──lỗi──► 413/415/422 problem+json
  2. cv2 decode → RGB → Resize 32×128 → Normalize(0.5, 0.5); thiếu frame thì lặp frame cuối
  3. [threadpool] lock model → S1 forward (return_sr nếu include_sr) → log_probs [1, 16, 37]
  4. decode constrained | greedy → plate, track_conf
     + char_conf (Viterbi CTC alignment) + top-5 suggestions (constrained beam)
  5. status = accepted nếu track_conf ≥ τ, ngược lại needs_review
  6. [threadpool] INSERT predictions … RETURNING                     (connection lấy từ pool)
◄── 201 Created, Location: /api/v1/predictions/{id}, body JSON (+ sr_images base64 nếu yêu cầu)
```

### 3.2. Batch — `POST /batches` (roadmap `/predict_batch`)

```
Client ──zip──► Router: đọc ≤ 200 MB → parse zip trong RAM (không giải nén ra đĩa)
              → INSERT batches(status=queued) → submit job → 202 Accepted + Location
Worker (1 thread, các job xếp hàng tuần tự):
  status=running
  lặp mỗi 4 track:  lock model → forward → nhả lock
                    → 1 transaction INSERT 4 predictions
                    → UPDATE processed_tracks
  status=completed | failed(error)
Client poll GET /batches/{id} → GET /batches/{id}/predictions | /export (CSV)
```

### 3.3. Review — tab Review của React (Tuần 8)

```
GET /predictions?status=needs_review     → hàng chờ
GET /predictions/{id}/suggestions        → 5 nút chọn nhanh
PUT /predictions/{id}/review {plate}     → INSERT reviews + status=reviewed
```

## 4. Triển khai bằng Docker

```bash
cd master’s_thesis_nhutminh
cp .env.example .env                 # đổi POSTGRES_PASSWORD
docker compose up -d --build         # db → (healthy) → api
curl http://localhost:8000/api/v1/health
open http://localhost:8000/api/v1/docs            # Swagger UI
docker compose run --rm -e BACKUP_ONCE=1 backup   # chụp DB 1 lần (khi cần)
docker compose --profile backup up -d             # + backup định kỳ (bật từ Tuần 10)
docker compose --profile public up -d             # + Cloudflare Tunnel (khi có token)
```

| Service | Image | Vai trò | Port | Volume |
| --- | --- | --- | --- | --- |
| `db` | `postgres:16-alpine` | Lưu predictions / reviews / batches | `127.0.0.1:5432` | `pgdata` |
| `api` | `platereader-api` (build từ `backend/Dockerfile`) | FastAPI + model S1 (CPU) | `8000` | checkpoint dir → `/models` (ro) |
| `backup` | `postgres:16-alpine` | `pg_dump` định kỳ. Profile `backup`, **tắt mặc định** | — | `./backups` |
| `tunnel` | `cloudflare/cloudflared` | URL public (profile `public`) | — | — |

Thứ tự khởi động: `db` phải `healthy` (`pg_isready`) thì `api` (và `backup` nếu bật) mới start.
`api` có `HEALTHCHECK` gọi `/api/v1/health`. Checkpoint **không nằm trong image** mà được mount read-only, nên đổi model chỉ cần sửa `.env` rồi restart.

### Biến môi trường chính (`.env`)

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `platereader` / — / `platereader` | Tài khoản DB |
| `MODEL_CHECKPOINT_DIR` / `MODEL_CHECKPOINT_FILE` | `./results/multi-seed/s1_mf_sr_ocr` / `s1_seed42_best.pth` | Checkpoint mount vào `/models` |
| `PLATEREADER_WIDTH_DOWNSAMPLE` | `8` | Bắt buộc `4` nếu dùng checkpoint S4 |
| `PLATEREADER_CONFIDENCE_THRESHOLD` | `0.90` | τ (tạm, chốt ở Tuần 5) |
| `PLATEREADER_DEFAULT_DECODE` | `constrained` | |
| `PLATEREADER_TORCH_THREADS` | `4` | Số thread CPU cho 1 lần forward |
| `PLATEREADER_INFERENCE_BATCH_SIZE` | `4` | Số track mỗi lượt khi chạy batch (xem §6) |
| `PLATEREADER_DB_HOST` / `_PORT` / `_USER` / `_PASSWORD` / `_NAME` | compose tự đặt (`db`, `5432`, từ `POSTGRES_*`) | Kết nối DB, khai báo từng phần nên mật khẩu chứa `@` vẫn đúng |
| `PLATEREADER_DB_POOL_MAX` / `_TIMEOUT_S` / `_STATEMENT_TIMEOUT_MS` | `10` / `10` / `15000` | Pool và timeout DB (xem §6) |
| `BACKUP_INTERVAL_HOURS` / `BACKUP_RETENTION_DAYS` | `6` / `7` | Lịch backup khi bật profile `backup` (xem §7) |
| `PLATEREADER_CORS_ORIGINS` | `["http://localhost:5173", …]` | Origin của React |

Danh sách đầy đủ có chú thích nằm trong [`.env.example`](../.env.example).

### Chạy không dùng Docker (dev)

```bash
cd master’s_thesis_nhutminh
uv venv backend/.venv --python 3.11
uv pip install --python backend/.venv/bin/python torch torchvision -r backend/requirements.txt
docker compose up -d db                                   # chỉ cần Postgres
export PLATEREADER_DB_HOST=localhost PLATEREADER_DB_PASSWORD='<mật khẩu trong .env>'
backend/.venv/bin/uvicorn backend.app.main:app --reload --port 8000
backend/.venv/bin/python -m pytest backend/tests -q       # 12 test
```

## 5. Hiệu năng đo được (22/10/2026, MacBook Apple Silicon)

| Phép đo | Kết quả |
| --- | --- |
| 1 track, chạy native (4 thread) | ~270 ms |
| 1 track, trong Docker Desktop | 810–910 ms (3 lần đo) |
| Thành phần | Forward model chiếm gần hết; decode + gợi ý + alignment ≈ 9 ms/track |
| Batch CPU: 32 track/lượt so với 1 track/lượt | Không nhanh hơn (≈ 0.66 s/track cả hai), nên chọn lượt 4 track |
| Đọc DB trong lúc batch đang ghi | p50 6–8 ms, p95 20–46 ms, 0 lỗi (xem §6) |

Roadmap §4 đặt tiêu chí p95 ≤ 1 s cho 1 track trên CPU. Tiêu chí này sẽ đo chính thức ở Tuần 6/10 trên máy demo, không đo trên Docker Desktop của Mac.

## 6. Bài toán đọc / ghi liên tục

Phân tích đầy đủ, số đo và đề xuất mở rộng nằm trong **report tuần 2** (`tuan2_bao_cao.md`, mục 5). Tóm tắt các biện pháp **đã triển khai**:

| # | Biện pháp | Ở đâu |
| --- | --- | --- |
| 1 | PostgreSQL thay SQLite: MVCC cho phép đọc không chặn ghi, ghi không chặn đọc | `docker-compose.yml`, `schema.sql` |
| 2 | Connection pool 1–10, chờ tối đa 10 s rồi trả `503 database_busy` kèm `Retry-After` | `db.py`, `errors.py` |
| 3 | `statement_timeout = 15 s` cho mỗi connection, để query treo không giữ pool | `db.py` |
| 4 | Không giữ connection DB trong lúc model chạy (inference xong mới mượn connection để ghi) | `routers/predictions.py`, `services/batches.py` |
| 5 | Batch ghi theo lượt 4 track trong **1 transaction**, cập nhật tiến độ sau mỗi lượt | `services/batches.py` |
| 6 | Hàng đợi batch 1 worker, lượt nhỏ, nên request lẻ chen vào được sau tối đa 1 lượt | `services/batches.py`, `config.py` |
| 7 | Review append-only (INSERT, không UPDATE đè), nên 2 người review cùng lúc không mất dữ liệu | `schema.sql` (bảng `reviews`) |
| 8 | Thống kê batch tính bằng `COUNT … FILTER` lúc đọc, không có bộ đếm để tranh chấp | `db.py` |
| 9 | Index cho 2 truy vấn nóng: hàng chờ review, kết quả theo batch | `schema.sql` |
| 10 | Job đang chạy khi server restart được đánh `failed`, không bị treo `running` mãi | `main.py` lifespan |

## 7. Backup & khôi phục

**Chính sách theo giai đoạn.** Backup tự động chỉ cần khi DB chứa dữ liệu không tạo lại được.

| Giai đoạn | Dữ liệu trong DB | Backup |
| --- | --- | --- |
| **Dev (Tuần 2–9)** — hiện tại | Prediction test: chạy lại model là có lại | **Tắt.** Chụp tay khi cần: `docker compose run --rm -e BACKUP_ONCE=1 backup` |
| **User test → demo (Tuần 10–12)** | Review do người thật nhập, kết quả SUS, thời gian review | **Bật:** `docker compose --profile backup up -d`, 6 giờ / lần, giữ 7 ngày |
| Mốc bắt buộc chụp tay | — | Trước Tuần 6 (freeze), trước và sau Tuần 10 (user test), trước Tuần 12 (demo) |

| Mục | Giá trị khi bật |
| --- | --- |
| Công cụ | `pg_dump -Fc -Z 6`: backup logic, nén, chạy nóng (không dừng API) |
| Tần suất | 6 giờ / lần (`BACKUP_INTERVAL_HOURS`), chụp 1 bản ngay khi service start |
| Lưu giữ | 7 ngày (= 28 bản), tự xoá bản cũ (`BACKUP_RETENTION_DAYS`) |
| RPO (mất tối đa) | ≤ 6 giờ dữ liệu |
| RTO (khôi phục) | < 5 phút với `backend/scripts/restore.sh <file>` (đã chạy thử 22/10/2026) |
| Nơi lưu | `./backups/` trên host, nằm ngoài volume DB. Trước demo copy thêm 1 bản sang ổ / cloud khác |

**Cron hay container?** Mặc định dùng container `backup` chạy vòng lặp `sleep`, vì không phải cấu hình gì trên máy demo và lịch chạy đi kèm `docker compose`.
Nếu deploy lên server cố định có cron thì thay bằng 1 dòng crontab gọi chế độ one-shot:

```cron
0 */6 * * *  cd /path/master’s_thesis_nhutminh && docker compose run --rm -e BACKUP_ONCE=1 backup
```

Lý do chọn 6 giờ và các phương án nâng cấp được trình bày trong report tuần 2 (`tuan2_bao_cao.md`, mục 6).

## 8. Bảo mật và license

- **Không lưu ảnh** upload. Ảnh SR chỉ trả về trong response, không ghi đĩa / DB (license ICPR26-LRLPR).
- Postgres chỉ bind `127.0.0.1`. Mật khẩu nằm trong `.env` (đã có trong `.gitignore`).
- Container API chạy bằng user không phải root. Checkpoint mount read-only.
- Zip đọc trong RAM, mỗi entry được giới hạn kích thước sau giải nén (chống zip-slip / zip bomb).
- **Chưa có xác thực.** Trước khi public qua tunnel (Tuần 9) cần bật Cloudflare Access hoặc thêm API key.

## 9. Việc tiếp theo (theo roadmap)

| Tuần | Việc liên quan backend |
| --- | --- |
| 3 | Thêm CER/NED/Confidence-gap; dùng `GET /predictions?batch_id=` để lấy dữ liệu McNemar |
| 5 | Chốt τ (precision ≥ 95%) và cập nhật `PLATEREADER_CONFIDENCE_THRESHOLD`; `track_conf` đã có sẵn trong DB |
| 6 | Freeze: ghi `model_version` + SHA-256 từ `GET /model` vào `freeze_manifest.json`; đo latency p50/p95 |
| 7 | `parity_report.json` từ `backend/scripts/parity_check.py` |
| 8 | React gọi API theo `api-docs/`, thêm service `web` vào compose |
| 9 | Tunnel + xác thực; kiểm thử ≥ 8 case lỗi (đã có mã lỗi tương ứng) |
