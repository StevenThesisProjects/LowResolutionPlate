# GET /batches/{id}/export

Xuất toàn bộ kết quả của batch ra **CSV**, ứng với "zip → CSV" ở Tuần 7. Mỗi track 1 dòng, sắp theo `track_id`.
Nếu đã có review thì cột `final_plate` lấy giá trị sau review.

```
GET /api/v1/batches/{id}/export
```

## Request — Path

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID batch |

## Response `200 OK`

- `Content-Type: text/csv; charset=utf-8`
- `Content-Disposition: attachment; filename="<tên zip>_results.csv"`

| Cột | Mô tả |
| --- | --- |
| `track_id` | Tên thư mục track trong zip |
| `plate` | Dự đoán của model |
| `track_conf` | 4 chữ số thập phân |
| `status` | `accepted` / `needs_review` / `reviewed` |
| `final_plate` | Sau review (nếu có) |
| `ground_truth` | Nhãn nếu zip có `annotations.json`, rỗng nếu không |
| `correct` | `1` / `0` / rỗng |
| `decode_mode` | |
| `model_version` | |
| `prediction_id` | Dùng để gọi lại `/predictions/{id}` |

```csv
track_id,plate,track_conf,status,final_plate,ground_truth,correct,decode_mode,model_version,prediction_id
track_10007,AYG7579,0.9996,accepted,AYG7579,AYG7579,1,constrained,s1_seed42_best@a1053d892457,b1f0…
track_10011,DQK3320,0.8180,needs_review,DQK3320,DQK3320,1,constrained,s1_seed42_best@a1053d892457,9c2e…
```

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `batch_not_found` | Không có ID này |
| 409 | `batch_not_completed` | Batch đang `queued` / `running` / `failed` |

## Ví dụ

```bash
curl -s -OJ http://localhost:8000/api/v1/batches/3444f665-6ae0-4a31-8caf-c7a2526fd95b/export
# lưu thành val999_results.csv
```
