# Database schema — PostgreSQL 16

> Nguồn: [`backend/app/schema.sql`](../backend/app/schema.sql). API chạy file này mỗi lần khởi động; file idempotent (`CREATE … IF NOT EXISTS`).
> Roadmap §2 ban đầu ghi SQLite. Đã chuyển sang **PostgreSQL** vì cần đọc/ghi đồng thời (batch ghi liên tục trong lúc UI đọc và review ghi),
> backup nóng bằng `pg_dump`, và kiểu `JSONB` / `UUID` / `TIMESTAMPTZ`. Xem [system-overview.md §6](system-overview.md#6-bài-toán-đọc--ghi-liên-tục).

## Sơ đồ quan hệ

```
┌──────────────────────┐ 1      N ┌─────────────────────────┐ 1      N ┌───────────────────────┐
│ batches              │──────────│ predictions             │──────────│ reviews               │
│──────────────────────│          │─────────────────────────│          │───────────────────────│
│ id            UUID PK│◄─────────│ batch_id     UUID FK ?  │◄─────────│ prediction_id UUID FK │
│ created_at           │ CASCADE  │ id           UUID PK    │ CASCADE  │ id         BIGSERIAL  │
│ finished_at          │          │ track_id, plate         │          │ plate, reviewer, note │
│ filename, status     │          │ track_conf, char_conf   │          │ created_at            │
│ decode_mode          │          │ suggestions, status     │          └───────────────────────┘
│ total_tracks         │          │ decode_mode, num_frames │
│ processed_tracks     │          │ latency_ms              │
│ error                │          │ model_version           │
└──────────────────────┘          │ ground_truth, created_at│
                                  └─────────────────────────┘
```

- `predictions.batch_id = NULL`: prediction đến từ `POST /predictions` (1 track lẻ).
- Xoá 1 batch thì prediction của batch đó bị xoá theo, và review của các prediction đó cũng bị xoá theo.
- **Không bảng nào lưu ảnh.** License ICPR26-LRLPR cấm phân phối lại, nên chỉ lưu kết quả dạng text.

---

## Bảng `batches`

Mỗi file zip upload qua `POST /batches` tạo 1 dòng, tương ứng 1 batch job.

| Cột | Kiểu | Ràng buộc | Mô tả |
| --- | --- | --- | --- |
| `id` | `UUID` | PK, `gen_random_uuid()` | |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, `now()` | Thời điểm nhận zip |
| `finished_at` | `TIMESTAMPTZ` | | Thời điểm chuyển sang `completed` / `failed` |
| `filename` | `TEXT` | NOT NULL | Tên file zip gốc |
| `status` | `TEXT` | CHECK ∈ `queued`, `running`, `completed`, `failed` | Vòng đời job |
| `decode_mode` | `TEXT` | CHECK ∈ `constrained`, `greedy` | |
| `total_tracks` | `INTEGER` | NOT NULL, > 0 | Số track parse được từ zip |
| `processed_tracks` | `INTEGER` | NOT NULL, ≥ 0 | Cập nhật sau mỗi lượt inference (4 track) |
| `error` | `TEXT` | | Lý do lỗi khi `failed` |

Các con số `accepted`, `needs_review`, `reviewed`, `labeled_tracks`, `accuracy` trong API **không lưu** trong bảng này.
Chúng được tính bằng `COUNT(*) FILTER (…)` trên `predictions` mỗi lần đọc, nên luôn khớp dữ liệu thật,
kể cả khi người review sửa một prediction sau khi batch đã xong.

## Bảng `predictions`

Mỗi track được nhận dạng tạo 1 dòng, dù đến từ `POST /predictions` hay từ một track trong batch.

| Cột | Kiểu | Ràng buộc | Mô tả |
| --- | --- | --- | --- |
| `id` | `UUID` | PK | |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, `now()` | |
| `batch_id` | `UUID` | FK → `batches.id`, `ON DELETE CASCADE`, nullable | |
| `track_id` | `TEXT` | | Mã track do client gửi hoặc tên thư mục trong zip |
| `plate` | `VARCHAR(16)` | NOT NULL | Dự đoán của model (không đổi sau review) |
| `track_conf` | `REAL` | NOT NULL, 0–1 | Độ tin cậy cả track |
| `char_conf` | `JSONB` | NOT NULL | Mảng số, mỗi phần tử ứng với 1 ký tự của `plate`. Ví dụ `[0.98, 0.34, 0.72, 0.99, …]` |
| `suggestions` | `JSONB` | NOT NULL | Top-k beam: `[["DQK3320", 0.818], ["DOK3320", 0.812], …]` |
| `status` | `TEXT` | CHECK ∈ `accepted`, `needs_review`, `reviewed` | Ban đầu theo τ, chuyển sang `reviewed` khi có review |
| `decode_mode` | `TEXT` | CHECK | |
| `num_frames` | `SMALLINT` | CHECK 1–5 | Số ảnh upload thật (trước khi pad) |
| `latency_ms` | `REAL` | NOT NULL | Thời gian tiền xử lý + forward + decode |
| `model_version` | `TEXT` | NOT NULL | `<checkpoint>@<sha256[:12]>`, để truy vết khi đổi model |
| `ground_truth` | `VARCHAR(16)` | | Nhãn từ `annotations.json` (chỉ có ở batch) |

Index:

| Index | Cột | Phục vụ |
| --- | --- | --- |
| `idx_predictions_batch` | `(batch_id, track_id)` | `GET /batches/{id}/predictions`, `/export` (sắp theo track) |
| `idx_predictions_status` | `(status, created_at DESC)` | Hàng chờ review `GET /predictions?status=needs_review` |

## Bảng `reviews`

Lịch sử sửa tay. **Chỉ thêm, không sửa / xoá** (append-only).

| Cột | Kiểu | Ràng buộc | Mô tả |
| --- | --- | --- | --- |
| `id` | `BIGSERIAL` | PK | Tăng dần, dùng để phân thứ tự khi 2 review cùng timestamp |
| `prediction_id` | `UUID` | FK → `predictions.id`, `ON DELETE CASCADE`, NOT NULL | |
| `plate` | `VARCHAR(16)` | NOT NULL | Biển số người review nhập |
| `reviewer` | `VARCHAR(64)` | | |
| `note` | `TEXT` | | |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, `now()` | |

Index `idx_reviews_prediction (prediction_id, created_at DESC)` dùng để lấy review mới nhất bằng `LEFT JOIN LATERAL … LIMIT 1`.

Tách review thành bảng riêng thay vì thêm cột vào `predictions` vì ba lý do:

1. **Giữ lịch sử**, cần cho việc đo "thời gian review / số lần sửa" ở Tuần 10.
2. **Tránh ghi đè lẫn nhau** khi 2 người cùng review 1 track: mỗi người INSERT một dòng riêng, không ai mất dữ liệu, kết quả có hiệu lực là dòng mới nhất.
3. Cột `plate` của model không bao giờ bị sửa, nên vẫn tính lại được accuracy của model sau khi đã có review.

---

## Truy vấn mẫu

```sql
-- Hàng chờ review, ký tự yếu nhất trước
SELECT id, track_id, plate, track_conf
FROM predictions
WHERE status = 'needs_review'
ORDER BY track_conf ASC
LIMIT 50;

-- Accuracy model vs accuracy sau review trên 1 batch có nhãn
SELECT
  AVG((p.plate = p.ground_truth)::int) * 100                         AS model_acc,
  AVG((COALESCE(r.plate, p.plate) = p.ground_truth)::int) * 100      AS after_review_acc
FROM predictions p
LEFT JOIN LATERAL (SELECT plate FROM reviews WHERE prediction_id = p.id
                   ORDER BY created_at DESC, id DESC LIMIT 1) r ON TRUE
WHERE p.batch_id = '3444f665-6ae0-4a31-8caf-c7a2526fd95b';

-- Precision của nhóm accepted theo τ (dữ liệu cho Tuần 5)
SELECT COUNT(*) FILTER (WHERE status = 'accepted')                                  AS n_accepted,
       AVG((plate = ground_truth)::int) FILTER (WHERE status = 'accepted') * 100    AS precision_accepted
FROM predictions WHERE ground_truth IS NOT NULL;
```

## Kết nối

| Cách | Lệnh |
| --- | --- |
| psql trong container | `docker compose exec db psql -U platereader` |
| Từ máy host (DBeaver / psql) | host `127.0.0.1`, port `5432`, user `platereader`, db `platereader`, mật khẩu = `POSTGRES_PASSWORD` trong `.env` (nếu dùng dạng URL thì ký tự `@` trong mật khẩu phải viết thành `%40`) |
