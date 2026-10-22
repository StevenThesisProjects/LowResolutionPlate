# PlateReader LR — API Docs

> Backend FastAPI cho prototype **MF-SR-OCR** (roadmap 12 tuần, §2 và Tuần 2/7).
> Phiên bản API: `1.0.0` · Base URL: `http://localhost:8000/api/v1`
> Swagger UI tương tác (sinh tự động): `http://localhost:8000/api/v1/docs`

| Tài liệu | Nội dung |
| --- | --- |
| [system-overview.md](system-overview.md) | Tổng quan hệ thống: kiến trúc, luồng xử lý, Docker, đọc/ghi đồng thời, backup |
| [database-schema.md](database-schema.md) | Mô tả 3 bảng PostgreSQL `batches`, `predictions`, `reviews` |
| [index.html](index.html) | Toàn bộ tài liệu này trong 1 file HTML (mở bằng trình duyệt) |

---

## Danh sách endpoint

| # | Method | Path | Mô tả | Roadmap |
| --- | --- | --- | --- | --- |
| 1 | `GET` | [`/health`](endpoints/01-get-health.md) | Trạng thái service (model + DB) | `/health` · Tuần 2 |
| 2 | `GET` | [`/model`](endpoints/02-get-model.md) | Thông tin model đang phục vụ | Tuần 6 (freeze manifest) |
| 3 | `POST` | [`/predictions`](endpoints/03-post-predictions.md) | Nhận dạng 1 track (1–5 frame) | **`/predict`** · Tuần 2 |
| 4 | `GET` | [`/predictions`](endpoints/04-get-predictions.md) | Danh sách prediction, lọc theo `status`/`batch_id` | Tab Review · Tuần 8 |
| 5 | `GET` | [`/predictions/{id}`](endpoints/05-get-prediction.md) | Chi tiết 1 prediction | Tuần 7 |
| 6 | `DELETE` | [`/predictions/{id}`](endpoints/06-delete-prediction.md) | Xoá 1 prediction | — |
| 7 | `GET` | [`/predictions/{id}/suggestions`](endpoints/07-get-suggestions.md) | **Gợi ý** top-k biển số thay thế cho người review | Tab Review · Tuần 8 |
| 8 | `PUT` | [`/predictions/{id}/review`](endpoints/08-put-review.md) | Ghi kết quả sửa tay | "người review sửa → DB" · §2 |
| 9 | `POST` | [`/batches`](endpoints/09-post-batches.md) | Tạo batch job từ file zip | **`/predict_batch`** · Tuần 7 |
| 10 | `GET` | [`/batches`](endpoints/10-get-batches.md) | Danh sách batch | Tab Batch · Tuần 8 |
| 11 | `GET` | [`/batches/{id}`](endpoints/11-get-batch.md) | Trạng thái / tiến độ / accuracy của batch | Tuần 7 |
| 12 | `GET` | [`/batches/{id}/predictions`](endpoints/12-get-batch-predictions.md) | Kết quả từng track của batch | Tab Batch · Tuần 8 |
| 13 | `GET` | [`/batches/{id}/export`](endpoints/13-get-batch-export.md) | Xuất CSV | "zip → CSV" · Tuần 7 |
| 14 | `DELETE` | [`/batches/{id}`](endpoints/14-delete-batch.md) | Xoá batch + toàn bộ prediction | — |

**Ánh xạ với roadmap.** Roadmap §2 ghi `/health`, `/predict`, `/predict_batch`. API giữ đúng 3 chức năng đó nhưng
đặt tên theo REST: danh từ số nhiều cho resource, method HTTP thể hiện hành động.
`POST /predict` → `POST /predictions`, `POST /predict_batch` → `POST /batches` + `GET /batches/{id}/export`.
Response giữ nguyên tên trường của roadmap: `plate`, `char_conf[]`, `track_conf`, `status`, `sr_images[]`,
`latency_ms`, `model_version`.

---

## Quy ước chung

### Định dạng

- Request upload dùng `multipart/form-data`; các request khác dùng `application/json`.
- Response thành công là `application/json` (riêng `/export` là `text/csv`).
- Thời gian theo ISO 8601 kèm múi giờ, ví dụ `2026-10-22T22:55:00.756+07:00`.
- ID là UUID v4. Nếu path chứa ID sai định dạng, API trả `422 validation_error`.
- Biển số luôn ở dạng IN HOA, không có khoảng trắng.

### Mã trạng thái HTTP

| Code | Khi nào |
| --- | --- |
| `200 OK` | GET / PUT thành công |
| `201 Created` | `POST /predictions`, kèm header `Location: /api/v1/predictions/{id}` |
| `202 Accepted` | `POST /batches`: job đã vào hàng đợi, header `Location: /api/v1/batches/{id}` |
| `204 No Content` | DELETE thành công |
| `404 Not Found` | Không có resource |
| `409 Conflict` | Thao tác không hợp lệ với trạng thái hiện tại (export batch chưa xong, xoá batch đang chạy) |
| `413 Content Too Large` | File vượt giới hạn (ảnh > 5 MB, zip > 200 MB, > 5000 track) |
| `415 Unsupported Media Type` | Sai kiểu file (không phải JPG/PNG/ZIP) |
| `422 Unprocessable Entity` | Dữ liệu sai: ảnh hỏng, số frame ∉ [1, 5], query sai enum… |
| `503 Service Unavailable` | Model chưa load, DB mất kết nối / quá tải (có `Retry-After`) |

### Định dạng lỗi (RFC 9457 — `application/problem+json`)

Mọi lỗi đều có cùng một dạng, nên frontend chỉ cần một hàm xử lý lỗi duy nhất. Trường `code` ổn định, dùng được để rẽ nhánh xử lý.

```json
{
  "type": "https://platereader.local/problems/invalid_frame_count",
  "title": "Invalid frame count",
  "status": 422,
  "detail": "Cần 1–5 ảnh, nhận được 6.",
  "instance": "/api/v1/predictions",
  "code": "invalid_frame_count",
  "errors": [ { "loc": ["query", "status"], "msg": "...", "type": "enum" } ]
}
```

`errors` chỉ có khi `code = validation_error`.

| `code` | HTTP | Ý nghĩa |
| --- | --- | --- |
| `validation_error` | 422 | Tham số / body sai kiểu, sai enum, thiếu trường |
| `invalid_frame_count` | 422 | Số ảnh ∉ [1, 5] |
| `invalid_image` | 422 | File không decode được thành ảnh |
| `invalid_plate` | 422 | Biển số review có ký tự ngoài A–Z, 0–9 |
| `invalid_zip` · `empty_zip` · `too_many_frames` | 422 | Zip hỏng / không có ảnh LR / track > 5 ảnh |
| `unsupported_media_type` | 415 | Sai kiểu file |
| `file_too_large` · `image_too_large` · `too_many_tracks` | 413 | Vượt giới hạn |
| `prediction_not_found` · `batch_not_found` · `not_found` | 404 | Không có resource |
| `batch_not_completed` · `batch_in_progress` | 409 | Sai trạng thái batch |
| `model_unavailable` | 503 | Checkpoint chưa load được |
| `database_unavailable` · `database_busy` · `database_timeout` | 503 | DB mất kết nối / hết connection trong pool / query quá `statement_timeout` |

### Phân trang

Các endpoint trả về danh sách nhận `page` (bắt đầu từ 1) và `page_size`, rồi trả về:

```json
{ "items": [ ... ], "total": 999, "page": 1, "page_size": 20 }
```

### Trạng thái prediction (selective prediction)

| `status` | Điều kiện |
| --- | --- |
| `accepted` | `track_conf ≥ τ`: tự động chấp nhận |
| `needs_review` | `track_conf < τ`: đưa vào hàng chờ review |
| `reviewed` | Đã có người sửa / xác nhận qua `PUT /review` |

τ (`PLATEREADER_CONFIDENCE_THRESHOLD`) hiện là **0.90 (tạm)**. Giá trị chính thức sẽ chọn ở Tuần 5 sao cho precision ≥ 95% trên val.
Với τ = 0.90, batch 999 track val có 887 track `accepted` và 112 track `needs_review`.

### CORS, xác thực

- CORS mặc định mở cho `http://localhost:5173` (Vite). Đổi bằng `PLATEREADER_CORS_ORIGINS`.
- **Chưa có xác thực.** API chỉ dùng cho mạng lab / demo. Nếu public qua Cloudflare Tunnel thì phải bật Cloudflare Access (Tuần 9).

### License dữ liệu

API **không lưu ảnh upload**, chỉ lưu kết quả dạng text. Ảnh SR chỉ trả về một lần trong response, không ghi vào DB.
Lý do: license ICPR26-LRLPR cấm phân phối lại ảnh dataset (xem `tuan1_xac_nhan.md`, câu 3).
