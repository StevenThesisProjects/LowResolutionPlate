# GET /predictions/{id}/suggestions

**Endpoint gợi ý.** Trả về top-k (mặc định 5) biển số ứng viên mà model cho là có khả năng cao nhất,
tất cả đều khớp layout Brazil. Khi model đọc sai, người review chỉ cần **chọn** một ứng viên thay vì gõ lại cả biển.
Cách này phục vụ trực tiếp việc đo "thời gian review có/không có app" ở Tuần 10.

```
GET /api/v1/predictions/{id}/suggestions
```

## Cách tính

Gợi ý lấy từ cùng một lần **constrained CTC prefix beam search** dùng để decode
(beam 16, mỗi vị trí ký tự bị khoá vào lớp chữ hoặc số theo layout `LLLNLNN` / `LLLNNNN`):

- Hạng 1 **luôn trùng** với kết quả `constrained_beam_decode` trong `src/utils/postprocess.py` (có test kiểm chứng).
- Các hạng sau là những chuỗi đầy đủ 7 ký tự còn lại trong beam, sắp xếp theo xác suất.
- `conf` là trung bình hình học xác suất từng ký tự, cùng thang 0–1 với `track_conf`.
- Gợi ý được tính lúc predict và lưu trong cột `predictions.suggestions`, nên endpoint này **không chạy lại model**.

Gợi ý luôn dựa trên constrained beam, kể cả khi prediction dùng `decode=greedy`.

## Request — Path

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID prediction |

## Response `200 OK`

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `prediction_id` | uuid | |
| `plate` | string | Dự đoán của model |
| `final_plate` | string | Sau review (nếu có) |
| `status` | string | Trạng thái hiện tại |
| `suggestions[].rank` | int | 1 = tốt nhất |
| `suggestions[].plate` | string | Biển số ứng viên |
| `suggestions[].conf` | number 0–1 | Độ tin cậy |
| `suggestions[].layout` | string? | `LLLNLNN` (Mercosur) hoặc `LLLNNNN` (mẫu cũ) |

Ví dụ thật, `track_10011` trong val. Ký tự thứ 2 có `conf` chỉ 0.34 (xem `char_conf`), nên các ứng viên hạng 2–3 thay đúng vị trí đó bằng `O` hoặc `D`, là những cặp hay nhầm với `Q`:

```json
{
  "prediction_id": "3eeca7a5-6d41-497f-97ab-b13607bdb144",
  "plate": "DQK3320",
  "final_plate": "DQK3320",
  "status": "needs_review",
  "suggestions": [
    {"rank": 1, "plate": "DQK3320", "conf": 0.818021, "layout": "LLLNNNN"},
    {"rank": 2, "plate": "DOK3320", "conf": 0.812367, "layout": "LLLNNNN"},
    {"rank": 3, "plate": "DDK3320", "conf": 0.811056, "layout": "LLLNNNN"},
    {"rank": 4, "plate": "DQE3320", "conf": 0.711898, "layout": "LLLNNNN"},
    {"rank": 5, "plate": "DOE3320", "conf": 0.706977, "layout": "LLLNNNN"}
  ]
}
```

Gợi ý cho UI: hiển thị dạng các nút chọn nhanh. Khi người dùng bấm một nút, gọi
[`PUT /predictions/{id}/review`](08-put-review.md) với `plate` tương ứng.

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `prediction_not_found` | Không có ID này |
| 503 | `model_unavailable` | Cần layout của model để gắn `layout` |

## Ví dụ

```bash
curl -s http://localhost:8000/api/v1/predictions/3eeca7a5-6d41-497f-97ab-b13607bdb144/suggestions
```
