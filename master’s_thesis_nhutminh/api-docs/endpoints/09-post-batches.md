# POST /batches

Nhận dạng **nhiều track** từ 1 file zip. Đây là endpoint `/predict_batch` trong roadmap (Tuần 7).
Batch 1.000 track trên CPU mất vài phút, nên request **không chờ** kết quả: server kiểm tra zip, đưa job vào hàng đợi
rồi trả `202 Accepted` ngay. Client theo dõi tiến độ bằng [`GET /batches/{id}`](11-get-batch.md),
xong thì lấy kết quả qua [`/predictions`](12-get-batch-predictions.md) hoặc [`/export`](13-get-batch-export.md).

```
POST /api/v1/batches
Content-Type: multipart/form-data
```

## Request

### Query

| Tham số | Kiểu | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `decode` | `constrained` \| `greedy` | `constrained` | Như `POST /predictions` |

### Form-data

| Trường | Kiểu | Bắt buộc | Mô tả |
| --- | --- | --- | --- |
| `file` | file `.zip` | ✔ | ≤ 200 MB, ≤ 5000 track |

### Cấu trúc zip

Giống thư mục dataset, nên có thể zip thẳng một thư mục val hoặc test:

```
tracks.zip
├── track_10007/
│   ├── lr-001.jpg … lr-005.jpg      ← 1–5 ảnh LR (.jpg/.jpeg/.png), sắp theo tên
│   ├── hr-001.jpg …                  ← bỏ qua
│   └── annotations.json              ← tuỳ chọn: {"plate_text": "AYG7579"} → tính accuracy
└── track_10011/ …
```

- Thư mục cha gần nhất của ảnh được dùng làm `track_id`. Ảnh nằm ngay gốc zip được gộp thành 1 track, đặt tên theo tên file zip.
- File `hr-*`, `__MACOSX/` và file ẩn đều bị bỏ qua.
- Zip được đọc thẳng trong bộ nhớ, **không giải nén ra đĩa**, nên không bị tấn công path traversal (zip-slip).

## Response `202 Accepted`

Header `Location: /api/v1/batches/{id}`. Body là object `Batch` (xem [`GET /batches/{id}`](11-get-batch.md)) ở trạng thái `queued` hoặc `running`.

```json
{
  "id": "3444f665-6ae0-4a31-8caf-c7a2526fd95b",
  "created_at": "2026-10-22T22:58:08.846027+07:00",
  "finished_at": null,
  "filename": "val999.zip",
  "status": "queued",
  "decode_mode": "constrained",
  "total_tracks": 999,
  "processed_tracks": 0,
  "accepted": 0,
  "needs_review": 0,
  "reviewed": 0,
  "labeled_tracks": 0,
  "accuracy": null,
  "error": null
}
```

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 413 | `file_too_large` | Zip > 200 MB |
| 413 | `image_too_large` | 1 ảnh trong zip > 5 MB sau giải nén (chống zip bomb) |
| 413 | `too_many_tracks` | > 5000 track |
| 415 | `unsupported_media_type` | Không phải `.zip` |
| 422 | `invalid_zip` | File hỏng / không phải zip |
| 422 | `empty_zip` | Không có ảnh LR nào |
| 422 | `too_many_frames` | 1 track có > 5 ảnh LR |
| 503 | `model_unavailable` · `database_*` | |

Ảnh hỏng **bên trong** zip không làm request lỗi. Job sẽ chuyển sang `failed`, lý do ghi trong trường `error`.

## Ví dụ

```bash
curl -s -X POST "http://localhost:8000/api/v1/batches" -F file=@val999.zip -i | grep -i location
# location: /api/v1/batches/3444f665-6ae0-4a31-8caf-c7a2526fd95b
```

## Kiểm chứng (parity, roadmap Tuần 7)

Ngày 22/10/2026, gửi zip 999 track val qua endpoint này (cả native lẫn Docker) với checkpoint `s1_seed42_best.pth` cho kết quả:
**accuracy 79.78%**, và **999/999** biển số trùng file submission offline `submission_s1_seed42.txt`.
Parity ở chế độ greedy cũng đạt 999/999 (script `backend/scripts/parity_check.py`).
