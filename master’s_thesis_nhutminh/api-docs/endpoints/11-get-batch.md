# GET /batches/{id}

Trạng thái, tiến độ và thống kê của 1 batch. Client poll endpoint này (gợi ý mỗi 2–5 giây) cho tới khi
`status` là `completed` hoặc `failed`.

```
GET /api/v1/batches/{id}
```

## Request — Path

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID batch (lấy từ header `Location` của `POST /batches`) |

## Response `200 OK` — `Batch`

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | |
| `created_at` / `finished_at` | datetime | `finished_at = null` khi chưa xong |
| `filename` | string | Tên file zip |
| `status` | `queued` → `running` → `completed` \| `failed` | |
| `decode_mode` | string | |
| `total_tracks` | int | Số track tìm thấy trong zip |
| `processed_tracks` | int | Số track đã xử lý. Tiến độ = `processed_tracks / total_tracks` |
| `accepted` / `needs_review` / `reviewed` | int | Số prediction theo trạng thái (cập nhật ngay khi có review) |
| `labeled_tracks` | int | Số track có `annotations.json` |
| `accuracy` | number? | Exact-match (%) trên các track có nhãn; `null` nếu không track nào có nhãn |
| `error` | string? | Lý do khi `failed` |

Ví dụ thật, 999 track val chạy trong Docker:

```json
{
  "id": "3444f665-6ae0-4a31-8caf-c7a2526fd95b",
  "created_at": "2026-10-22T22:58:08.846027+07:00",
  "finished_at": "2026-10-22T23:18:47.370879+07:00",
  "filename": "val999.zip",
  "status": "completed",
  "decode_mode": "constrained",
  "total_tracks": 999,
  "processed_tracks": 999,
  "accepted": 887,
  "needs_review": 112,
  "reviewed": 0,
  "labeled_tracks": 999,
  "accuracy": 79.78,
  "error": null
}
```

Nếu server khởi động lại khi job đang chạy, job đó chuyển sang `failed` với
`error = "Server restarted before the job finished."` Các prediction đã ghi trước thời điểm đó vẫn được giữ.

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `batch_not_found` | Không có ID này |
| 422 | `validation_error` | ID không phải UUID |

## Ví dụ

```bash
curl -s http://localhost:8000/api/v1/batches/3444f665-6ae0-4a31-8caf-c7a2526fd95b
```
