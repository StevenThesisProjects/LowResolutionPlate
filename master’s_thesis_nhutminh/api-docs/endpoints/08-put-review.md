# PUT /predictions/{id}/review

Ghi kết quả kiểm tra thủ công cho 1 prediction, ứng với bước "người review sửa → DB" trong roadmap §2.
Endpoint dùng được cho cả hai trường hợp: **sửa** (nhập biển khác) và **xác nhận** (gửi lại đúng `plate` của model).

`PUT` là idempotent: gửi lại cùng nội dung không làm thay đổi kết quả cuối.
Mỗi lần gọi thêm 1 dòng vào bảng `reviews` (lưu lịch sử, không ghi đè), và dòng mới nhất là kết quả có hiệu lực.

```
PUT /api/v1/predictions/{id}/review
Content-Type: application/json
```

## Request

### Path

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID prediction |

### Body

| Trường | Kiểu | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `plate` | string 1–16 | ✔ | Biển số đúng. Server tự viết hoa, bỏ khoảng trắng, và chỉ chấp nhận A–Z, 0–9 |
| `reviewer` | string ≤ 64 | | Tên / mã người review |
| `note` | string ≤ 500 | | Ghi chú, ví dụ "frame 3 bị chói" |

```json
{ "plate": "DQK3320", "reviewer": "minh", "note": "đọc lại frame 3" }
```

`plate` **không bị ép** theo layout Brazil, để người review vẫn ghi được những biển ngoài miền dữ liệu (ví dụ biển VN khi test định tính ở Tuần 10).

## Response `200 OK`

Trả về prediction đã cập nhật (cấu trúc như [`GET /predictions/{id}`](05-get-prediction.md)), với `status = "reviewed"`, `final_plate = review.plate`.

```json
{
  "id": "3eeca7a5-6d41-497f-97ab-b13607bdb144",
  "plate": "DQK3320",
  "final_plate": "DQK3320",
  "status": "reviewed",
  "review": {
    "plate": "DQK3320",
    "reviewer": "minh",
    "note": "đọc lại frame 3",
    "reviewed_at": "2026-10-22T22:55:00.865359+07:00"
  },
  "...": "các trường còn lại như GET /predictions/{id}"
}
```

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `prediction_not_found` | Không có ID này |
| 422 | `invalid_plate` | Có ký tự ngoài A–Z, 0–9 |
| 422 | `validation_error` | Thiếu `plate`, `plate` rỗng / > 16 ký tự |

## Ví dụ

```bash
curl -s -X PUT http://localhost:8000/api/v1/predictions/3eeca7a5-6d41-497f-97ab-b13607bdb144/review \
  -H "Content-Type: application/json" \
  -d '{"plate":"DQK3320","reviewer":"minh","note":"đọc lại frame 3"}'
```
