# GET /batches/{id}/predictions

Kết quả từng track của 1 batch, có phân trang. Dùng cho bảng kết quả trong màn hình **Batch** (Tuần 8).
Có thể gọi ngay **khi batch đang chạy**: endpoint trả về những track đã xử lý xong.

```
GET /api/v1/batches/{id}/predictions?page=1&page_size=50
```

Tương đương `GET /predictions?batch_id={id}`, nhưng trả `404` nếu batch không tồn tại.

## Request

| Tham số | Vị trí | Kiểu | Mặc định | Mô tả |
| --- | --- | --- | --- | --- |
| `id` | path | uuid | | ID batch |
| `page` | query | int ≥ 1 | 1 | |
| `page_size` | query | int 1–500 | 50 | |

## Response `200 OK`

`Page<Prediction>`. Mỗi phần tử có `track_id`, `batch_id`, và có thêm `ground_truth`, `correct` nếu zip kèm nhãn.

```json
{
  "items": [
    {
      "id": "b1f0…",
      "batch_id": "3444f665-6ae0-4a31-8caf-c7a2526fd95b",
      "track_id": "track_10007",
      "plate": "AYG7579",
      "final_plate": "AYG7579",
      "track_conf": 0.99956,
      "char_conf": [{"char": "A", "conf": 1.0}, "..."],
      "status": "accepted",
      "decode_mode": "constrained",
      "num_frames": 5,
      "latency_ms": 742.1,
      "model_version": "s1_seed42_best@a1053d892457",
      "ground_truth": "AYG7579",
      "correct": true
    }
  ],
  "total": 999,
  "page": 1,
  "page_size": 50
}
```

Với prediction thuộc batch, `latency_ms` là thời gian của cả lượt inference chia đều cho số track trong lượt đó (mặc định 4 track).

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `batch_not_found` | Không có ID này |

## Ví dụ

```bash
curl -s "http://localhost:8000/api/v1/batches/3444f665-6ae0-4a31-8caf-c7a2526fd95b/predictions?page_size=100"
```
