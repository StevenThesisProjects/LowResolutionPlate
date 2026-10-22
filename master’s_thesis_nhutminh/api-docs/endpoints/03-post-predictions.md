# POST /predictions

Nhận dạng biển số của **một track** gồm 1–5 ảnh crop. Đây là endpoint `/predict` trong roadmap §2.
Kết quả được lưu vào bảng `predictions`. Nếu `track_conf < τ`, prediction được gắn cờ `needs_review`.

```
POST /api/v1/predictions
Content-Type: multipart/form-data
```

## Request

### Query

| Tham số | Kiểu | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `decode` | `constrained` \| `greedy` | `constrained` | `constrained` = beam search ràng buộc layout Brazil (dùng trong mọi số liệu thesis). `greedy` = argmax |
| `include_sr` | bool | `false` | Trả kèm ảnh siêu phân giải (PNG base64, 256×64) cho từng frame đã upload |

### Form-data

| Trường | Kiểu | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `files` | file[] | ✔ | 1–5 ảnh **JPG/PNG**, mỗi ảnh ≤ 5 MB. Gửi lặp key `files` cho nhiều ảnh, theo đúng thứ tự frame |
| `track_id` | string | | Mã track phía client (≤ 128 ký tự), dùng để tra cứu sau |

### Xử lý (roadmap §2, flow bước 1–5)

1. **Validate**: kiểu file, kích thước, số frame. Nếu ít hơn 5 frame thì **lặp lại frame cuối** cho đủ 5.
2. Resize về 32×128, normalize (mean = std = 0.5), giống hệt pipeline val khi train.
3. S1: STN → DCNv2 → FrameSR ×2 → ResBackbone (GN) → Attention Fusion → BiLSTM → CTC.
4. Decode `constrained` hoặc `greedy`. Tính `char_conf` bằng CTC forced alignment, lấy top-5 gợi ý.
5. So `track_conf` với τ để ra `accepted` hoặc `needs_review`, rồi ghi DB.

## Response `201 Created`

Header `Location: /api/v1/predictions/{id}`

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID prediction |
| `created_at` | datetime | |
| `track_id` | string? | Lấy từ request (bỏ qua nếu không gửi) |
| `plate` | string | Biển số model đọc được |
| `final_plate` | string | `review.plate` nếu đã review, ngược lại bằng `plate` |
| `track_conf` | number 0–1 | Độ tin cậy cả track: trung bình hình học xác suất từng ký tự (constrained) hoặc trung bình max-prob (greedy) |
| `char_conf` | `{char, conf}[]` | Độ tin cậy từng ký tự. Ký tự có `conf` thấp là chỗ người review nên nhìn kỹ |
| `status` | `accepted` \| `needs_review` | Theo τ |
| `decode_mode` | string | Chế độ decode đã dùng |
| `num_frames` | int | Số ảnh thực tế đã upload (trước khi pad) |
| `latency_ms` | number | Thời gian xử lý gồm tiền xử lý, forward và decode (không tính upload) |
| `model_version` | string | Xem [`GET /model`](02-get-model.md) |
| `sr_images` | `{frame, mime_type, width, height, data_base64}[]` | Chỉ có khi `include_sr=true`. **Không lưu DB** |

```json
{
  "id": "3eeca7a5-6d41-497f-97ab-b13607bdb144",
  "created_at": "2026-10-22T22:55:00.756732+07:00",
  "track_id": "track_10011",
  "plate": "DQK3320",
  "final_plate": "DQK3320",
  "track_conf": 0.818021,
  "char_conf": [
    {"char": "D", "conf": 0.981311},
    {"char": "Q", "conf": 0.34467},
    {"char": "K", "conf": 0.724929},
    {"char": "3", "conf": 0.999876},
    {"char": "3", "conf": 0.999957},
    {"char": "2", "conf": 0.999963},
    {"char": "0", "conf": 0.999904}
  ],
  "status": "needs_review",
  "decode_mode": "constrained",
  "num_frames": 5,
  "latency_ms": 754.92,
  "model_version": "s1_seed42_best@a1053d892457",
  "sr_images": [
    {"frame": 1, "mime_type": "image/png", "width": 256, "height": 64,
     "data_base64": "iVBORw0KGgoAAAANSUhEUgAAAQAAAABACAIAAAB6..."}
  ]
}
```

Hiển thị ảnh SR trong React:

```jsx
<img src={`data:${img.mime_type};base64,${img.data_base64}`} alt={`SR frame ${img.frame}`} />
```

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 413 | `file_too_large` | 1 ảnh > 5 MB |
| 415 | `unsupported_media_type` | Content-type không phải `image/jpeg` / `image/png` |
| 422 | `invalid_frame_count` | 0 hoặc > 5 ảnh |
| 422 | `invalid_image` | File không decode được thành ảnh |
| 422 | `validation_error` | `decode` sai enum, thiếu `files` |
| 503 | `model_unavailable` · `database_*` | Model / DB chưa sẵn sàng |

## Ví dụ

```bash
curl -s -X POST "http://localhost:8000/api/v1/predictions?include_sr=true" \
  -F track_id=track_10011 \
  -F files=@lr-001.jpg -F files=@lr-002.jpg -F files=@lr-003.jpg \
  -F files=@lr-004.jpg -F files=@lr-005.jpg
```

```js
// React / fetch
const form = new FormData();
frames.forEach((f) => form.append("files", f));      // File[] từ <input type="file" multiple>
form.append("track_id", "track_10011");
const res = await fetch(`${API}/predictions?include_sr=true`, { method: "POST", body: form });
if (!res.ok) throw await res.json();                  // problem+json
const prediction = await res.json();
```

## Hiệu năng (đo 22/10/2026)

| Môi trường | Latency 1 track (5 frame) |
| --- | --- |
| Apple Silicon, chạy native, 4 thread | ~270 ms |
| Docker Desktop (VM Linux) trên cùng máy | ~810–910 ms |
| Báo cáo cũ (`tuan1`): CPU server | 192.75 ms |

Gần như toàn bộ thời gian nằm ở forward của model. Decode cộng gợi ý cộng alignment chỉ tốn khoảng 9 ms/track.
Tiêu chí MVP là p95 ≤ 1 s (roadmap §4) và sẽ được đo chính thức ở Tuần 6 và 10.
