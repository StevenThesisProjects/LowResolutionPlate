# Tuần 7 (16–22/11): Backend v2 (API)

Tuần hiện tại. Đây là phần trọng tâm của prototype.

## Roadmap yêu cầu

1. Tích hợp model đã freeze + SC-CTC + triage vào `/predict`.
2. Trả `top3`, `char_conf`, `frame_weights`, `sharpness`.
3. Xử lý trường hợp < 5 ảnh.
4. Parity test lần 2 trên val → `parity_final.json`.
5. OpenAPI spec.

## Dùng gì thay thế

| Hạng mục roadmap | Làm thế nào (code mới) | Khác roadmap ở đâu, vì sao |
|---|---|---|
| Model freeze | `mfplate/engine.py` nạp đúng `src/models/restran.py` của repo, không chép code model. SHA-256 checkpoint được kiểm mỗi lần khởi động; sai thì `/health` trả 503 | — |
| SC-CTC | `mfplate/decoding.py`: greedy, beam, **constrained** (= SC-CTC: chỉ phát triển tiền tố của `LLLNLNN`/`LLLNNNN`, giữ top-beam theo từng độ dài tiền tố). Xác suất của mỗi ứng viên tính chính xác bằng thuật toán forward | Mặc định là **greedy**, không phải SC-CTC: SC-CTC không qua tiêu chí đặt trước (Tuần 2). SC-CTC vẫn gọi được bằng `?decoder=constrained` |
| Triage | Ngưỡng `mean_char` riêng cho từng số frame, chốt trên val 999 (Tuần 5); thêm cờ khi chuỗi sai mẫu biển | Roadmap dùng một ngưỡng; app nhận 1–5 frame nên cần ngưỡng theo N |
| `top3`, `char_conf`, `frame_weights`, `sharpness` | Đủ cả bốn. `frame_weights` lấy trực tiếp từ softmax của `AttentionFusion` qua forward hook, không đoán lại | — |
| < 5 ảnh | **Đưa thẳng N frame** vào model và trả cảnh báo kèm accuracy lock-box với đúng N đó (E3) | Roadmap đề xuất lặp frame. AttentionFusion nhận số frame bất kỳ, và E3 đo đúng theo cách này; lặp frame sẽ làm softmax lệch về frame bị lặp |
| Parity | `scripts/parity_check.py` → `artifacts/parity_final.json` | Thêm phép so CPU với GPU trên cả val và lock-box |
| OpenAPI | Tự sinh từ FastAPI (`/docs`, `/openapi.json`) → `artifacts/openapi.json`; tài liệu tiếng Việt `docs/api.md` | — |

## API

| Endpoint | Chức năng |
|---|---|
| `GET /health` | 200 khi model đã nạp, 503 khi chưa (Docker healthcheck dùng endpoint này) |
| `GET /model` | Phiên bản model, SHA-256 đã kiểm, decoder mặc định, ngưỡng theo số frame, accuracy lock-box theo số frame, giới hạn upload |
| `POST /predict` | 1–5 ảnh JPEG/PNG **hoặc** 1 zip của track (chỉ lấy `lr-*`, bỏ `hr-*`; đọc nhãn từ `annotations.json` nếu có). Tham số: `decoder`, `top_k` (1–5), `session_id` |
| `POST /feedback` | Báo đúng/sai, kèm biển số đúng nếu sai |
| `GET /history`, `GET /stats` | Lịch sử theo phiên; thống kê tỉ lệ cần kiểm tra và accuracy theo phản hồi |

Response của `/predict` có đủ các trường trong flow của roadmap: `plate, confidence, char_conf, top3, layout, frame_weights, sharpness, needs_review, latency_ms, model_version`. Thêm: `sequence_prob`, `review_reasons`, `triage` (điểm, ngưỡng), `warnings`, `expected_accuracy`, `reference_plate`, `timing`. Ví dụ đầy đủ và bảng mã lỗi nằm trong `docs/api.md`.

**An toàn đầu vào:**
- định dạng nhận theo magic bytes;
- kích thước ảnh đọc từ header trước khi giải nén;
- zip giới hạn 12 MB, mỗi ảnh đọc tối đa 2 MB dù header zip khai báo gì;
- tùy chọn khóa `X-API-Key` (biến `MFPLATE_API_KEY`);
- mọi lỗi trả cùng một dạng JSON;
- log SQLite chỉ lưu SHA-256 của ảnh, không lưu ảnh.

## Kết quả parity (`artifacts/parity_final.json`)

| So sánh | Kết quả | Tiêu chí MVP số 1 (≤ 2 track lệch) |
|---|---|---|
| API so với offline trên val 999, cùng CPU | **0/999 track lệch biển số**. Confidence lệch tối đa 2,9e-6; confidence từng ký tự lệch tối đa 1,7e-5 (chạy batch 1 so với batch 64) | ✅ |
| API trên CPU so với server GPU, val 999 | 76,68% (766/999) = `val_acc` epoch 30 trong log train trên V100 (76,6767%) | ✅ |
| API trên CPU so với lần chấm lock-box trên GPU | 79,90% = 79,90%; **0/1.000 track** khác đúng/sai | ✅ |

Tiền xử lý của API khớp **từng bit** với pipeline train (resize, chuẩn hoá, luma) trên 80 frame JPEG/PNG (`tests/test_preprocess.py`).

## Test tự động

`python3 -m pytest tests` → **61 test, đạt cả 61** (22 decoder, 9 tiền xử lý, 30 API). Phần API gồm 17 test lỗi, chẳng hạn không có file, quá 5 ảnh, không phải ảnh, GIF đổi đuôi, JPEG hỏng, file rỗng, file > 2 MB, ảnh 4000 px, ảnh 2×3 px, zip hỏng, zip không có ảnh, zip > 5 ảnh, gửi lẫn zip và ảnh, tham số sai, feedback với id không tồn tại hoặc biển số sai định dạng, thiếu API key.

## Lỗi tìm ra khi làm (đã sửa)

- SC-CTC với beam 10 có 1/999 track (seed 7) không tìm được chuỗi 7 ký tự hợp lệ: model đọc chắc 6 ký tự, nên các biến thể 6 ký tự chiếm hết beam. Code cũ lùi về beam không ràng buộc và trả chuỗi 6 ký tự. Đã sửa bằng cách giữ top-beam theo từng độ dài tiền tố, và thêm test `test_constrained_completes_a_six_character_reading`. Sau khi sửa, SC-CTC ra 0 chuỗi sai mẫu ở cả 8 lượt chấm (4 seed × val/lock-box). Accuracy và quyết định decoder không đổi.

## Deliverable

| Roadmap | File | Trạng thái |
|---|---|---|
| API v2 | `mfplate/` (`api.py`, `engine.py`, `decoding.py`, `preprocess.py`, `uploads.py`, `storage.py`, `schemas.py`, `settings.py`) | ✅ |
| `parity_final.json` | `artifacts/parity_final.json` | ✅ |
| OpenAPI spec | `artifacts/openapi.json` + `docs/api.md` | ✅ |

## Lệnh

```bash
cd masters_thesis_tienphat
python3 -m uvicorn mfplate.api:app --port 8000          # http://127.0.0.1:8000/docs
python3 -m pytest tests                       # 61 test
python3 scripts/parity_check.py --lockbox     # cần cache: scripts/offline_cache.py
python3 scripts/export_openapi.py
```
