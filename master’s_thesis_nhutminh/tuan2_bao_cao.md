# MF-SR-OCR — Báo cáo Tuần 2

> Người thực hiện: Minh · Ngày: 22/10/2026
> Roadmap Tuần 2: **Multi-seed S1** · **FastAPI skeleton load S1** (`/health`, `/predict` trả đúng 1 track val)
> Mã nguồn: [`master’s_thesis_nhutminh/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh) · Backend: [`backend/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/backend) · API docs: [`api-docs/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/api-docs)

---

## 0. TÓM TẮT

| Hạng mục | Kết quả |
| --- | --- |
| Multi-seed S1 | **79.95 ± 0.15%** (3 seed, val 999 track) |
| Backend Tuần 2 | FastAPI load checkpoint S1; `GET /health` và `POST /predictions` (endpoint `/predict` của roadmap) trả đúng track val |
| Parity API vs offline | **999/999 track trùng**, accuracy qua API **79.78%** (bằng offline seed 42), chạy cả native lẫn Docker. Greedy: **999/999** (0 track lệch, tiêu chí MVP #1 đạt) |
| Kiểm thử | 12/12 test tự động pass · parity 999 track · load test đọc/ghi đồng thời 0 lỗi · 8 loại lỗi đầu vào |
| Làm sẵn cho các tuần sau | Batch zip → CSV (Tuần 7) · review + endpoint gợi ý (Tuần 8) · Docker + PostgreSQL + cấu hình backup (Tuần 9) |
| Swagger | **<http://localhost:8000/api/v1/docs>** |

---

## 1. ĐÃ LÀM ĐƯỢC GÌ

### 1.1. Deliverable Tuần 2 theo roadmap

| # | Deliverable | Kết quả | Bằng chứng |
| --- | --- | --- | --- |
| 1 | `multiseed_results.csv` (mean ± std) | ✅ S1 3 seed: 79.78 / 80.08 / 79.98 → **79.95 ± 0.15%**. File gồm thêm J1, S4 để so sánh | [`results/multi-seed/multiseed_results.csv`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/results/multi-seed/multiseed_results.csv) |
| 2 | FastAPI skeleton load S1 | ✅ Load `s1_seed42_best.pth` (29.6M tham số), kiến trúc suy ra tự động từ checkpoint | [`backend/app/services/inference.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/services/inference.py) |
| 3 | `/health` | ✅ `GET /api/v1/health`, báo trạng thái model + DB | [`backend/app/routers/system.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/routers/system.py) |
| 4 | `/predict` trả đúng 1 track val | ✅ `POST /api/v1/predictions`: `track_10007` → `AYG7579` (conf 0.9996), trùng file submission offline | [`backend/app/routers/predictions.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/routers/predictions.py) |

### 1.2. Làm sẵn cho các tuần sau

Các phần này nằm trong cùng backend nên làm luôn để các tuần sau chỉ cần dùng, không phải viết lại:

| Thuộc roadmap | Đã làm | Mã nguồn |
| --- | --- | --- |
| Tuần 7 | `POST /batches` (zip nhiều track), `GET /batches/{id}/export` (CSV), parity 999 track | [`backend/app/routers/batches.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/routers/batches.py) · [`services/batches.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/services/batches.py) |
| Tuần 8 (phần backend) | Hàng chờ review, `PUT /predictions/{id}/review`, **endpoint gợi ý** `GET /predictions/{id}/suggestions` | [`backend/app/routers/predictions.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/routers/predictions.py) · [`services/decoding.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/services/decoding.py) |
| Tuần 9 (phần backend) | Dockerfile CPU, docker-compose, PostgreSQL, chuẩn hoá lỗi, cấu hình backup | [`backend/Dockerfile`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/Dockerfile) · [`docker-compose.yml`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/docker-compose.yml) |
| Tài liệu | API docs từng endpoint (Markdown + HTML), tổng quan hệ thống, schema DB | [`api-docs/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/api-docs) |

### 1.3. Điều chỉnh so với roadmap §2

| Roadmap | Thực hiện | Lý do |
| --- | --- | --- |
| SQLite | **PostgreSQL 16** | Ghi và đọc đồng thời không khoá nhau (mục 5), backup nóng (mục 6), kiểu `JSONB` / `UUID` |
| `/predict`, `/predict_batch` | `POST /predictions`, `POST /batches` | Chuẩn RESTful. Response giữ đúng tên trường của roadmap: `plate`, `char_conf[]`, `track_conf`, `status`, `sr_images[]`, `latency_ms`, `model_version` |
| Gradio | React (Vite), làm ở Tuần 8 | Đã chốt ở Tuần 1 (câu 5). Backend độc lập với frontend qua REST + CORS |

---

## 2. MULTI-SEED S1

### 2.1. Kết quả (val 999 track Scenario-B, constrained decode)

| Seed | Best acc | Greedy | CER | Best epoch / số epoch đã chạy | Giờ GPU |
| --- | --- | --- | --- | --- | --- |
| 42 | 79.78% | 79.78% | 0.0541 | 28 / 46 | 7.4 |
| 100 | 80.08% | 79.88% | 0.0525 | 32 / 50 | 8.0 |
| 2026 | 79.98% | 79.98% | 0.0556 | 24 / 42 | 6.8 |
| **Mean ± std** | **79.95 ± 0.15%** | | | | 22.2 h |

Nguồn: [`results/multi-seed/s1_mf_sr_ocr/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/results/multi-seed/s1_mf_sr_ocr) (`history_s1_seed*.csv`, `submission_s1_seed*.txt`)

- Accuracy tính lại từ file submission và nhãn thật **trùng 100%** với history.
- CI 95% (binomial, n = 999): **±2.5 điểm**, nên so sánh giữa các cấu hình cần kiểm định ghép cặp (McNemar, Tuần 3).
- **741 track** cả 3 seed đều đúng, **143 track** cả 3 seed đều sai (~14%). Nhóm này là lỗi "cứng" không phụ thuộc seed, sẽ là đầu vào cho error analysis ở Tuần 5.
- CER = NED vì mọi nhãn dài đúng 7 ký tự, nên thesis chỉ cần báo cáo CER.

### 2.2. Chất lượng SR (`sr_quality_s1_seed42.csv`, 999 track val)

| | PSNR (dB) | SSIM |
| --- | --- | --- |
| Bicubic | 15.87 | 0.354 |
| **S1 FrameSR** | **16.59** (+0.72) | **0.397** (+0.043) |

SR có PSNR tốt hơn bicubic ở **999/999** track. Số đo trên **Scenario-B** (HR là JPG). Số đo trên Scenario-A (HR là PNG) sẽ bổ sung ở Tuần 3.

### 2.3. So sánh với các cấu hình đã có (phục vụ gate Tuần 3)

| Mô hình | Seed 42 | Seed 100 | Seed 2026 | Mean ± std |
| --- | --- | --- | --- | --- |
| **S1** (MF-SR-OCR) | 79.78 | 80.08 | 79.98 | **79.95 ± 0.15** |
| J1 (CRNN ResBlock + GN, không SR) | 79.98 | 80.88 | 80.48 | 80.45 ± 0.45 |
| S4 (SR scale 1) | 78.98 | 79.78 | 79.68 | 79.48 ± 0.44 |

McNemar ghép cặp S1 vs J1 theo từng seed: p = 0.91 / 0.43 / 0.64, tức không có khác biệt có ý nghĩa.

**Đề xuất:** trình bày S1 là đạt accuracy **tương đương** model không SR (độ lệch chuẩn giữa các seed thấp nhất: 0.15 so với 0.45), và **có thêm ảnh SR** giúp người review kiểm chứng.
Giá trị của ảnh SR sẽ được đo trực tiếp ở Tuần 10 (thời gian review có / không có app).

---

## 3. BACKEND API — Ý TƯỞNG VÀ THIẾT KẾ

### 3.1. Ý tưởng

1. **Dùng lại đúng model đã kiểm chứng, không sửa code training.** Backend chỉ *import* `src/`, `configs/`, `tools/eval_decode.py`. Tiền xử lý dùng đúng `get_val_transforms`, kiến trúc suy ngược từ `state_dict`, nên kết quả API phải trùng offline (đã chứng minh ở mục 4).
2. **Selective prediction.** Mỗi kết quả có `track_conf`. Nếu `track_conf ≥ τ` thì `accepted`, ngược lại `needs_review` để chuyển cho người kiểm tra. τ = 0.90 là giá trị tạm, sẽ chốt ở Tuần 5.
3. **Hỗ trợ người review thay vì chỉ trả 1 đáp án:**
   - `char_conf`: độ tin cậy **từng ký tự**, tính bằng CTC forced alignment, chỉ ra ký tự nào cần nhìn kỹ.
   - **Endpoint gợi ý**: top-5 biển số ứng viên lấy từ constrained beam search, tất cả đều đúng layout Brazil. Người review chỉ cần **chọn**, không phải gõ lại.
   - `sr_images`: ảnh siêu phân giải từng frame để đối chiếu.
4. **Không lưu ảnh** (license ICPR26-LRLPR cấm phân phối lại), chỉ lưu kết quả dạng text.
5. **RESTful + lỗi chuẩn RFC 9457.** Mọi lỗi có cùng dạng `application/problem+json` với `code` ổn định, nên frontend React chỉ cần một hàm xử lý lỗi.

Ví dụ thật của endpoint gợi ý, `track_10011`: model đọc `DQK3320`, ký tự `Q` chỉ có conf 0.34. Gợi ý hạng 2–3 là `DOK3320` (0.81) và `DDK3320` (0.81), đúng các cặp hay nhầm `Q/O/D`.

### 3.2. Kiến trúc

```
React (Tuần 8) ──HTTP──► api: FastAPI + Uvicorn :8000 ──pool──► db: PostgreSQL 16
                          ├ routers   /api/v1/{health, model, predictions, batches}
                          ├ inference S1 (CPU), khoá model dùng chung
                          ├ decoding  constrained beam top-k + CTC alignment
                          └ batches   hàng đợi 1 worker chạy nền
                         checkpoint mount read-only: results/multi-seed/s1_mf_sr_ocr/
backup (profile, chưa bật): pg_dump định kỳ → ./backups/
```

Chi tiết: [`api-docs/system-overview.md`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/system-overview.md) · Schema DB (3 bảng `batches`, `predictions`, `reviews`): [`api-docs/database-schema.md`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/database-schema.md)

### 3.3. Danh sách endpoint

Base URL: `http://localhost:8000/api/v1`

| # | Method | Path | Chức năng | Roadmap | Tài liệu |
| --- | --- | --- | --- | --- | --- |
| 1 | GET | `/health` | Trạng thái model + DB | **Tuần 2** | [01](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/01-get-health.md) |
| 2 | GET | `/model` | Checkpoint, SHA-256, τ | Tuần 6 | [02](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/02-get-model.md) |
| 3 | POST | `/predictions` | Nhận dạng 1 track (= `/predict`) | **Tuần 2** | [03](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/03-post-predictions.md) |
| 4 | GET | `/predictions` | Danh sách, lọc `status` / `batch_id` | Tuần 8 | [04](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/04-get-predictions.md) |
| 5 | GET | `/predictions/{id}` | Chi tiết | Tuần 7 | [05](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/05-get-prediction.md) |
| 6 | DELETE | `/predictions/{id}` | Xoá | — | [06](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/06-delete-prediction.md) |
| 7 | GET | `/predictions/{id}/suggestions` | **Gợi ý** top-5 biển số | Tuần 8 | [07](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/07-get-suggestions.md) |
| 8 | PUT | `/predictions/{id}/review` | Ghi kết quả sửa tay | Tuần 8 | [08](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/08-put-review.md) |
| 9 | POST | `/batches` | Batch từ zip (= `/predict_batch`) | Tuần 7 | [09](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/09-post-batches.md) |
| 10 | GET | `/batches` | Danh sách batch | Tuần 8 | [10](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/10-get-batches.md) |
| 11 | GET | `/batches/{id}` | Tiến độ, accuracy | Tuần 7 | [11](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/11-get-batch.md) |
| 12 | GET | `/batches/{id}/predictions` | Kết quả từng track | Tuần 8 | [12](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/12-get-batch-predictions.md) |
| 13 | GET | `/batches/{id}/export` | Xuất CSV | Tuần 7 | [13](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/13-get-batch-export.md) |
| 14 | DELETE | `/batches/{id}` | Xoá batch | — | [14](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/endpoints/14-delete-batch.md) |

### 3.4. Link truy cập (sau khi `docker compose up -d`)

| Link | Địa chỉ đầy đủ |
| --- | --- |
| **Swagger UI** (gọi thử API trực tiếp) | **<http://localhost:8000/api/v1/docs>** |
| Swagger (lối tắt, tự chuyển hướng) | <http://localhost:8000/docs> |
| ReDoc | <http://localhost:8000/api/v1/redoc> |
| OpenAPI JSON | <http://localhost:8000/api/v1/openapi.json> |
| Health check | <http://localhost:8000/api/v1/health> |
| Tài liệu API (HTML, mở bằng trình duyệt) | [`api-docs/index.html`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/api-docs/index.html) |

**Hình 1. Swagger UI: 14 endpoint theo nhóm System / Predictions / Batches**

![Swagger endpoints](https://raw.githubusercontent.com/StevenThesisProjects/LowResolutionPlate/main/master%E2%80%99s_thesis_nhutminh/report/tuan2/swagger_endpoints.png)

**Hình 2. Swagger UI: `POST /predictions` (endpoint `/predict` của Tuần 2), tham số, response 201 / 4XX / 503**

![Swagger POST /predictions](https://raw.githubusercontent.com/StevenThesisProjects/LowResolutionPlate/main/master%E2%80%99s_thesis_nhutminh/report/tuan2/swagger_post_predictions.png)

**Hình 3. Tài liệu API (`api-docs/index.html`): danh sách endpoint và ánh xạ với roadmap**

![API docs overview](https://raw.githubusercontent.com/StevenThesisProjects/LowResolutionPlate/main/master%E2%80%99s_thesis_nhutminh/report/tuan2/apidocs_html_overview.png)

**Hình 4. Tài liệu API: `POST /predictions` với response thật của `track_10011`**

![API docs POST /predictions](https://raw.githubusercontent.com/StevenThesisProjects/LowResolutionPlate/main/master%E2%80%99s_thesis_nhutminh/report/tuan2/apidocs_html_post_predictions.png)

**Hình 5. Tài liệu API: endpoint gợi ý `GET /predictions/{id}/suggestions`**

![API docs suggestions](https://raw.githubusercontent.com/StevenThesisProjects/LowResolutionPlate/main/master%E2%80%99s_thesis_nhutminh/report/tuan2/apidocs_html_suggestions.png)

---

## 4. KIỂM THỬ

### 4.1. Test tự động: 12/12 pass

Mã nguồn: [`backend/tests/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/backend/tests) · Chạy: `backend/.venv/bin/python -m pytest backend/tests -q`

| # | Test | Kiểm tra gì | Kết quả |
| --- | --- | --- | --- |
| 1 | `test_rank1_matches_constrained_beam_decode` | Gợi ý hạng 1 trùng `constrained_beam_decode` của `src/` (3 × 64 mẫu ngẫu nhiên); thứ tự giảm dần; đúng layout | ✅ |
| 2 | `test_char_confidences_on_clean_alignment` | `char_conf` từ CTC alignment đúng giá trị kỳ vọng | ✅ |
| 3 | `test_match_layout` | Nhận diện layout `LLLNLNN` / `LLLNNNN` | ✅ |
| 4 | `test_health_and_model` | `/health` = ok; `/model` báo đúng kiến trúc S1 (SR, 5 frame) | ✅ |
| 5 | `test_single_prediction_matches_offline` | `POST /predictions` trả `201` + `Location`; biển số **trùng offline**; 5 ảnh SR 256×64; gợi ý hạng 1 = `plate`; review chuẩn hoá `abc 1d23 → ABC1D23`; xoá trả `204` | ✅ |
| 6 | `test_fewer_frames_are_padded` | Upload 2 frame vẫn chạy (pad lên 5) | ✅ |
| 7–9 | `test_prediction_errors` ×3 | Ảnh hỏng → `422 invalid_image`; file text → `415`; 6 ảnh → `422 invalid_frame_count`; đúng `application/problem+json` | ✅ |
| 10 | `test_not_found_and_bad_id` | ID không tồn tại → `404 prediction_not_found`; ID sai định dạng → `422` | ✅ |
| 11 | `test_batch_parity` | Zip 8 track: `202` → `completed` → CSV trùng offline 8/8 → xoá `204` | ✅ |
| 12 | `test_bad_zip` | Zip hỏng → `422 invalid_zip` | ✅ |

### 4.2. Parity API vs offline (tiêu chí MVP §4, #1)

| Kiểm tra | Môi trường | Kết quả |
| --- | --- | --- |
| Constrained, 999 track val qua `POST /batches` | Native (macOS) | **999/999 trùng**, accuracy 79.78% |
| Constrained, 999 track val qua `POST /batches` | **Docker** | **999/999 trùng**, accuracy 79.78%, 887 `accepted` / 112 `needs_review` (τ = 0.90) |
| Greedy, 999 track, đường tiền xử lý API vs `MultiFrameDataset` | Native | **999/999 trùng, 0 track lệch**, accuracy 79.78% cả hai, cùng SHA-256 checkpoint `a1053d89…` → **đạt tiêu chí MVP #1** ("0 track lệch, greedy, cùng checkpoint") |

Script: [`backend/scripts/parity_check.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/scripts/parity_check.py), dùng lại để sinh `parity_report.json` ở Tuần 7.

### 4.3. Load test đọc / ghi đồng thời (Docker)

Kịch bản: 1 batch 40 track đang ghi liên tục, đồng thời 5 client `POST /predictions` và 5 client đọc liên tục
(`GET /predictions?status=needs_review` + `GET /batches/{id}`, mỗi 0.25 s).
Script: [`backend/scripts/load_test.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/scripts/load_test.py)

| Thao tác | Số request | p50 | p95 | max | Lỗi |
| --- | --- | --- | --- | --- | --- |
| Đọc hàng chờ review | 200 | 8 ms | 46 ms | 136 ms | 0 |
| Đọc tiến độ batch | 200 | 6 ms | 20 ms | 70 ms | 0 |
| `POST /predictions` (chạy model) | 5 | 5.1 s | 6.2 s | 7.1 s | 0 |
| Batch 40 track | 1 | 40/40 trùng offline | | | 0 |

### 4.4. Xử lý lỗi đầu vào (chuẩn bị cho `error_cases.md` Tuần 9)

| # | Trường hợp | Phản hồi |
| --- | --- | --- |
| 1 | File không phải ảnh / ảnh hỏng | `422 invalid_image` |
| 2 | Sai định dạng (txt, pdf…) | `415 unsupported_media_type` |
| 3 | 0 hoặc > 5 frame | `422 invalid_frame_count` |
| 4 | Ảnh > 5 MB | `413 file_too_large` |
| 5 | Zip hỏng / zip rỗng / track > 5 ảnh | `422 invalid_zip` · `empty_zip` · `too_many_frames` |
| 6 | Zip > 200 MB / > 5000 track / ảnh trong zip > 5 MB (zip bomb) | `413` |
| 7 | ID không tồn tại / sai định dạng | `404` / `422` |
| 8 | Model chưa load, DB mất kết nối / quá tải | `503` (+ `Retry-After`) |

### 4.5. Latency 1 track (CPU)

| Môi trường | Latency |
| --- | --- |
| Native macOS, 4 thread | ~270 ms |
| Docker Desktop trên macOS | 810–910 ms |
| CPU server (Tuần 1) | 192.75 ms |

Gần như toàn bộ thời gian nằm ở forward của model. Decode + gợi ý + alignment chỉ ≈ 9 ms/track. Tiêu chí p95 ≤ 1 s sẽ đo chính thức trên máy demo Linux ở Tuần 6.

---

## 5. BÀI TOÁN ĐỌC / GHI LIÊN TỤC

### 5.1. Bài toán

| Ghi liên tục | Đọc liên tục (cùng lúc) |
| --- | --- |
| Batch worker: 1 zip 1.000 track tạo 1.000 INSERT + ~250 UPDATE tiến độ, kéo dài 5–20 phút | UI poll tiến độ batch mỗi 2–5 giây |
| `POST /predictions` từ nhiều người dùng | Tab Review đọc hàng chờ `needs_review` |
| Người review `PUT /review`, có thể 2 người cùng sửa 1 track | Xem kết quả batch đang chạy, export CSV |

Sáu vấn đề có thể xảy ra:

1. **Khoá ghi toàn DB.** SQLite chỉ cho một writer tại một thời điểm, nên batch đang ghi thì review và predict phải chờ (`database is locked`).
2. **Cạn connection.** Giữ connection trong lúc model chạy (0.3–0.9 s) thì vài request đồng thời là hết pool.
3. **Đói tài nguyên model.** Model CPU dùng chung, batch giữ lâu thì request lẻ phải chờ.
4. **Mất cập nhật.** Hai người review cùng một track, người sau ghi đè người trước.
5. **Tranh chấp bộ đếm.** `UPDATE count = count + 1` khiến mọi writer cùng tranh một dòng.
6. **Treo trạng thái.** Server tắt giữa chừng thì batch kẹt ở `running` mãi.

### 5.2. Giải pháp đã triển khai

| # | Vấn đề | Giải pháp | Mã nguồn |
| --- | --- | --- | --- |
| 1 | Khoá ghi | **PostgreSQL + MVCC**: đọc không chặn ghi, ghi không chặn đọc | [`schema.sql`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/schema.sql) |
| 2 | Cạn connection | **Connection pool** 1–10. Chờ tối đa 10 s rồi trả `503 database_busy` + `Retry-After`. **Không giữ connection khi model chạy**. `statement_timeout = 15 s` | [`db.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/db.py) · [`core/errors.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/core/errors.py) |
| 3 | Đói model | Batch chạy ở **1 worker nền**, chia **lượt 4 track**, nhả khoá model sau mỗi lượt. Đo được: trên CPU, 32 track/lượt không nhanh hơn 1 track/lượt (≈ 0.66 s/track), nên lượt nhỏ không làm giảm throughput | [`services/batches.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/services/batches.py) |
| 3b | Ghi lẻ tẻ | Mỗi lượt ghi trong **1 transaction**, không để lại lượt ghi dở | [`db.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/db.py) |
| 4 | Mất cập nhật | Bảng `reviews` **append-only**: mỗi review là 1 INSERT, dòng mới nhất có hiệu lực, giữ đủ lịch sử | [`schema.sql`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/schema.sql) |
| 5 | Tranh chấp bộ đếm | Không lưu bộ đếm. Thống kê tính bằng `COUNT(*) FILTER` lúc đọc, có index | [`db.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/db.py) |
| 6 | Treo trạng thái | Khi khởi động, batch `queued/running` cũ chuyển sang `failed` kèm lý do | [`main.py`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/main.py) |
| — | Truy vấn nóng | Index `(status, created_at)` và `(batch_id, track_id)` | [`schema.sql`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/app/schema.sql) |

**Kết quả** (load test mục 4.3):

- Đọc vẫn ở mức mili-giây khi đang ghi liên tục, 0 lỗi.
- Nút thắt còn lại là **CPU của model**: request predict đồng thời phải xếp hàng chờ model, không phải chờ DB.

### 5.3. Đề xuất mở rộng khi tải vượt 1 máy

| Khi nào | Đề xuất | Hiệu quả |
| --- | --- | --- |
| Request lẻ bị batch làm chậm | Hàng đợi ưu tiên: request lẻ được ưu tiên hơn lượt batch | Request lẻ chờ tối đa 1 forward |
| Nhiều máy / nhiều job | Tách worker thành container riêng, hàng đợi Redis (RQ / Celery) | Scale worker độc lập; job sống sót khi API restart |
| Polling nhiều | Server-Sent Events cho tiến độ batch | Bỏ polling |
| 2 người sửa cùng lúc | `ETag` + `If-Match` trên `PUT /review` (trả `412` khi xung đột) | Báo xung đột thay vì "người sau thắng" |
| Hàng trăm connection | PgBouncer | Giảm chi phí connection |
| Báo cáo nặng | Read replica | Tách tải đọc khỏi primary |
| > 10 triệu prediction | Partition theo tháng | Index nhỏ, dọn dữ liệu nhanh |

---

## 6. BACKUP DỮ LIỆU

**Trạng thái: đã cấu hình sẵn, chưa bật** (giai đoạn phát triển). Service `backup` nằm trong profile `backup` của docker-compose, nên `docker compose up` **không** chạy nó.

Mã nguồn: [`backend/scripts/backup.sh`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/scripts/backup.sh) · [`backend/scripts/restore.sh`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/scripts/restore.sh) · [`docker-compose.yml`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/docker-compose.yml)

### 6.1. Dữ liệu nào cần backup

| Dữ liệu | Nếu mất | Cần backup |
| --- | --- | --- |
| `predictions`, `batches` | Chạy lại model là có lại, cùng checkpoint cho cùng kết quả (đã chứng minh bằng parity 999/999) | Không bắt buộc |
| **`reviews`** | **Không tạo lại được.** Gồm kết quả user test và thời gian review ở Tuần 10 | **Có** |
| Checkpoint, dataset | Nằm ngoài DB, quản lý riêng | — |

### 6.2. Lịch backup đề xuất

| Giai đoạn | Backup | Lý do |
| --- | --- | --- |
| **Tuần 2–9 (phát triển)** | **Không chạy.** Chỉ chụp tay trước mốc freeze Tuần 6 | DB chỉ chứa dữ liệu thử, tạo lại được |
| **Tuần 10–12 (user test → demo)** | **Bật: 6 giờ / lần, giữ 7 ngày (28 bản)** | Bắt đầu có review của người thật |
| Mốc chụp tay | Trước và sau buổi user test Tuần 10 · trước demo Tuần 12 | Luôn có điểm khôi phục sạch |

Lý do chọn **6 giờ, giữ 7 ngày**:

- **RPO ≤ 6 giờ.** Một buổi user test gói trong nửa ngày. Lịch 6 giờ / lần cộng một bản chụp tay ngay sau buổi test thì không mất buổi nào.
- **Chi phí gần bằng 0.** Với ~1.000 prediction, bản dump nén chỉ khoảng **120 KB**. `pg_dump` chạy nóng: không khoá bảng, không dừng API.
- **7 ngày** đủ để quay lại trước một lỗi phát hiện muộn (ví dụ đổi τ sai), tổng dung lượng vẫn chỉ vài MB.
- **RTO < 5 phút.** Đã kiểm thử khôi phục bằng `restore.sh` trên dữ liệu thử: xoá dữ liệu rồi khôi phục, đủ số dòng ban đầu.

### 6.3. Cách bật / chạy (khi đến Tuần 10)

```bash
docker compose --profile backup up -d                  # bật lịch 6 giờ / lần
docker compose run --rm -e BACKUP_ONCE=1 backup        # chụp tay 1 bản
sh backend/scripts/restore.sh backups/<file>.dump      # khôi phục
```

### 6.4. Container hay cron

| Cách | Ưu | Nhược | Dùng khi |
| --- | --- | --- | --- |
| **Container `backup`** — đề xuất | Không phải cấu hình máy host; đi cùng `docker compose`; chạy trên Mac/Windows/Linux | Lịch tính từ lúc container start | Máy demo, lab |
| Cron trên host gọi chế độ one-shot | Cố định giờ chạy | Phải cấu hình từng máy | Server cố định |

```cron
0 */6 * * *  cd /path/master’s_thesis_nhutminh && docker compose run --rm -e BACKUP_ONCE=1 backup
```

**An toàn:**

- Ghi ra file `.partial` rồi mới đổi tên, nên không có file hỏng mang tên hợp lệ.
- Lưu ở `./backups/`, ngoài volume DB, và đã có trong `.gitignore`.
- Trước demo, copy thêm 1 bản ra ổ / cloud khác.
- Khi triển khai lâu dài, đề xuất chuyển sang WAL archiving + PITR (pgBackRest / wal-g).

---

## 7. ĐỀ XUẤT VÀ RỦI RO

| # | Vấn đề | Ảnh hưởng | Đề xuất |
| --- | --- | --- | --- |
| 1 | Chưa có test-holdout 1.000 track. Checkpoint S1 hiện tại đã train trên toàn bộ phần Scenario-B ngoài val | Tách holdout từ phần train cũ sẽ bị lộ dữ liệu | **Train lại S1 1 seed (42) trên split mới có holdout** (~7 h GPU, ~65.000 ⚡), dùng holdout đúng 1 lần ở Tuần 6. Test-public ICPR (không nhãn) chỉ dùng cho leaderboard |
| 2 | Không train lại baseline CRNN+STN | Bảng so sánh thiếu 3 run baseline trên cùng split | Dùng J1 + S4 có sẵn (mục 2.3) và baseline đo lại 75.78% |
| 3 | S1 ≈ J1 (p > 0.4) | Gate Tuần 3 (p < 0.05) khó đạt khi so với J1 | Trình bày S1 là "accuracy tương đương, ổn định hơn giữa các seed, có thêm ảnh SR cho người review" |
| 4 | Latency trong Docker trên macOS ~0.85 s | Sát ngưỡng p95 ≤ 1 s | Đo trên máy demo Linux (native ~0.27 s) |
| 5 | Chưa có xác thực API | Chưa public URL được | Bật Cloudflare Access trước khi mở tunnel ở Tuần 9 |
| 6 | τ = 0.90 là giá trị tạm | 112/999 track bị gắn cờ | Chốt τ ở Tuần 5 theo precision ≥ 95% |

---

## 8. KẾ HOẠCH TUẦN 3

- [ ] CER, NED, Confidence Gap vào `eval_report.md`
- [ ] PSNR/SSIM trên Scenario-A
- [ ] McNemar + bootstrap CI: S1 vs J1, S1 vs CRNN+STN → `stats.json`, `decision_gate.json`
- [ ] Tách split có test-holdout và train lại S1 seed 42 (đề xuất #1)

---

## 9. MÃ NGUỒN VÀ CÁCH CHẠY

| Thư mục / file | Nội dung |
| --- | --- |
| [`backend/app/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/backend/app) | FastAPI: `main.py`, `routers/`, `services/`, `db.py`, `schema.sql` |
| [`backend/tests/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/backend/tests) | 12 test pytest |
| [`backend/scripts/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/backend/scripts) | `backup.sh`, `restore.sh`, `parity_check.py`, `load_test.py` |
| [`backend/Dockerfile`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/backend/Dockerfile) · [`docker-compose.yml`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/docker-compose.yml) · [`.env.example`](https://github.com/StevenThesisProjects/LowResolutionPlate/blob/main/master%E2%80%99s_thesis_nhutminh/.env.example) | Docker: `db` + `api`; profile `backup`, `public` |
| [`api-docs/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/api-docs) | README, system-overview, database-schema, 14 endpoint, `index.html` |
| [`results/multi-seed/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/results/multi-seed) | Kết quả 3 seed S1 / J1 / S4, `multiseed_results.csv` |
| [`report/tuan2/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/main/master%E2%80%99s_thesis_nhutminh/report/tuan2) | Ảnh chụp Swagger và API docs |

```bash
cd master’s_thesis_nhutminh
cp .env.example .env                       # đặt POSTGRES_PASSWORD
docker compose up -d --build               # PostgreSQL + API
curl http://localhost:8000/api/v1/health   # {"status":"ok",...}
# Swagger: http://localhost:8000/api/v1/docs

# Test tự động (cần db đang chạy, venv backend/.venv)
PLATEREADER_DB_HOST=localhost PLATEREADER_DB_PASSWORD='<mật khẩu>' \
  backend/.venv/bin/python -m pytest backend/tests -q
```
