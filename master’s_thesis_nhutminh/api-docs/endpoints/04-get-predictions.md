# GET /predictions

Danh sách prediction có phân trang, sắp xếp mới nhất trước.
Tab **Review** (Tuần 8) dùng endpoint này với `status=needs_review` để lấy hàng chờ kiểm tra.

```
GET /api/v1/predictions?status=needs_review&page=1&page_size=20
```

## Request — Query

| Tham số | Kiểu | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `status` | `accepted` \| `needs_review` \| `reviewed` | — | Lọc theo trạng thái |
| `batch_id` | uuid | — | Chỉ lấy prediction thuộc 1 batch |
| `page` | int ≥ 1 | 1 | |
| `page_size` | int 1–200 | 20 | |

## Response `200 OK`

`Page<Prediction>`. Cấu trúc mỗi phần tử giống [`POST /predictions`](03-post-predictions.md) nhưng **không có** `sr_images`.
Nếu prediction đã được review thì có thêm trường `review`. Nếu prediction thuộc một batch có nhãn thì có thêm `ground_truth` và `correct`.

```json
{
  "items": [
    {
      "id": "3eeca7a5-6d41-497f-97ab-b13607bdb144",
      "created_at": "2026-10-22T22:55:00.756732+07:00",
      "track_id": "track_10011",
      "plate": "DQK3320",
      "final_plate": "DQK3320",
      "track_conf": 0.818021,
      "char_conf": [{"char": "D", "conf": 0.981311}, {"char": "Q", "conf": 0.34467}, "..."],
      "status": "needs_review",
      "decode_mode": "constrained",
      "num_frames": 5,
      "latency_ms": 754.92,
      "model_version": "s1_seed42_best@a1053d892457"
    }
  ],
  "total": 112,
  "page": 1,
  "page_size": 20
}
```

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 422 | `validation_error` | `status` sai enum, `batch_id` không phải UUID, `page_size` > 200 |

## Ví dụ

```bash
curl -s "http://localhost:8000/api/v1/predictions?status=needs_review&page_size=50"
```
