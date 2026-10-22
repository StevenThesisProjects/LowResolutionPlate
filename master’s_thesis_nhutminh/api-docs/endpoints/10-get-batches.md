# GET /batches

Danh sách batch, mới nhất trước. Màn hình **Batch** (Tuần 8) dùng endpoint này để hiển thị lịch sử upload.

```
GET /api/v1/batches?page=1&page_size=20
```

## Request — Query

| Tham số | Kiểu | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `page` | int ≥ 1 | 1 | |
| `page_size` | int 1–200 | 20 | |

## Response `200 OK`

`Page<Batch>`. Mỗi phần tử có cấu trúc như [`GET /batches/{id}`](11-get-batch.md).

```json
{
  "items": [
    {
      "id": "2e24556e-d45a-4a55-bb2d-a0044ab3abdd",
      "created_at": "2026-10-22T22:52:48.916276+07:00",
      "finished_at": "2026-10-22T22:53:25.104761+07:00",
      "filename": "val40.zip",
      "status": "completed",
      "decode_mode": "constrained",
      "total_tracks": 40,
      "processed_tracks": 40,
      "accepted": 33,
      "needs_review": 7,
      "reviewed": 0,
      "labeled_tracks": 40,
      "accuracy": 62.5,
      "error": null
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

## Ví dụ

```bash
curl -s "http://localhost:8000/api/v1/batches?page_size=10"
```
