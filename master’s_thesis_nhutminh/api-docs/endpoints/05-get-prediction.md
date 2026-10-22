# GET /predictions/{id}

Chi tiết 1 prediction, gồm review mới nhất nếu có.

```
GET /api/v1/predictions/{id}
```

## Request — Path

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID trả về từ `POST /predictions` hoặc từ danh sách |

## Response `200 OK`

Cùng cấu trúc với [`POST /predictions`](03-post-predictions.md), trừ `sr_images` (ảnh không được lưu). Sau khi review sẽ có thêm:

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `review.plate` | string | Biển số người review nhập |
| `review.reviewer` | string? | Tên người review |
| `review.note` | string? | Ghi chú |
| `review.reviewed_at` | datetime | |
| `ground_truth` | string? | Nhãn thật, chỉ có khi đến từ batch zip kèm `annotations.json` |
| `correct` | bool? | `plate == ground_truth` |

```json
{
  "id": "3eeca7a5-6d41-497f-97ab-b13607bdb144",
  "created_at": "2026-10-22T22:55:00.756732+07:00",
  "track_id": "track_10011",
  "plate": "DQK3320",
  "final_plate": "DQK3320",
  "track_conf": 0.818021,
  "char_conf": [{"char": "D", "conf": 0.981311}, {"char": "Q", "conf": 0.34467}, "..."],
  "status": "reviewed",
  "decode_mode": "constrained",
  "num_frames": 5,
  "latency_ms": 754.92,
  "model_version": "s1_seed42_best@a1053d892457",
  "review": {
    "plate": "DQK3320",
    "reviewer": "minh",
    "note": "đọc lại frame 3",
    "reviewed_at": "2026-10-22T22:55:00.865359+07:00"
  }
}
```

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `prediction_not_found` | Không có ID này |
| 422 | `validation_error` | ID không phải UUID |

## Ví dụ

```bash
curl -s http://localhost:8000/api/v1/predictions/3eeca7a5-6d41-497f-97ab-b13607bdb144
```
