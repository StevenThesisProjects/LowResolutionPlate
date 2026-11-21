# MF-PlateReader API

API đọc biển số từ 1–5 ảnh crop độ phân giải thấp của cùng một xe. Model phía sau là ResTran đã đóng băng: cấu hình #50, run lock-box seed 42, epoch 30. Spec máy đọc được nằm ở [`artifacts/openapi.json`](../artifacts/openapi.json). Khi server chạy, trang thử trực tiếp nằm ở `/docs` (Swagger UI).

- **Phạm vi:** biển Brazil/Mercosur 1 dòng, 7 ký tự (LRLPR-26). Không dùng cho biển Việt Nam. Đầu vào phải là ảnh đã crop biển số, API không có bước phát hiện biển.
- **Chạy:** `cd masters_thesis_tienphat && python3 -m uvicorn mfplate.api:app --port 8000`, hoặc dùng Docker (README, mục 3).
- **Xác thực:** mặc định tắt. Đặt biến môi trường `MFPLATE_API_KEY` thì mọi endpoint trừ `/health` đòi header `X-API-Key`. Nên bật khi mở URL ra ngoài qua Cloudflare Tunnel.

## 1. Luồng xử lý một request

Luồng này giống flow ở mục 2 của roadmap. Chỗ nào khác thì có ghi lý do.

| Bước | Làm gì | Ghi chú |
|---|---|---|
| 1. Kiểm tra | 1–5 ảnh JPEG/PNG hoặc 1 file zip; mỗi ảnh ≤ 2 MB; mỗi cạnh từ 4 đến 2048 px; zip ≤ 12 MB | Định dạng nhận theo magic bytes, không theo đuôi file. Kích thước ảnh đọc từ header trước khi giải nén |
| 2. Tiền xử lý | Sắp xếp frame theo tên (`lr-001` < `lr-002` < …), resize 32×128 (INTER_LINEAR), chuẩn hoá mean 0,5 / std 0,5, đổi sang luma BT.601 lặp 3 kênh | Khớp từng bit với pipeline train (`tests/test_preprocess.py`) |
| 3. Model | ResTran → log-prob [16 × 37]; giữ lại trọng số attention của từng frame | Ít hơn 5 ảnh thì đưa thẳng N frame vào model, **không lặp frame** như roadmap gợi ý (lý do ở mục 4) |
| 4. Giải mã | `greedy` (mặc định), `beam` hoặc `constrained` (beam có ràng buộc mẫu `LLLNLNN`/`LLLNNNN`, tức SC-CTC); trả top-k với xác suất CTC tính chính xác | Greedy là mặc định vì `constrained` không qua tiêu chí đặt trước: +0,05 điểm, cần ≥ +0,30 (`artifacts/decoder_decision.json`) |
| 5. Phân loại | `needs_review = true` khi điểm `mean_char` < ngưỡng của đúng số frame, hoặc chuỗi sai mẫu biển | Ngưỡng chốt trên val 999 để nhóm "tự động" đạt precision ≥ 95% (`artifacts/threshold_calibration.json`) |
| 6. Ghi log | SQLite: hash SHA-256 của ảnh đầu vào, kết quả, độ tin cậy, latency, phản hồi | **Không lưu ảnh** |

## 2. Endpoint

| Method | Đường dẫn | Mục đích |
|---|---|---|
| GET | `/health` | Model đã nạp chưa. 200 = sẵn sàng, 503 = chưa (Docker healthcheck dùng endpoint này) |
| GET | `/model` | Phiên bản model, SHA-256 đã kiểm chưa, decoder mặc định, ngưỡng theo số frame, accuracy lock-box theo số frame, giới hạn upload |
| POST | `/predict` | Đọc biển số |
| POST | `/feedback` | Người dùng báo kết quả đúng/sai (kèm biển số đúng nếu sai) |
| GET | `/history?session_id=&limit=` | Các lần đọc gần nhất, mặc định 20, tối đa 200 |
| GET | `/stats` | Số lần đọc, tỉ lệ cần kiểm tra, latency trung bình, accuracy theo phản hồi (tách nhóm tự động / cần kiểm tra) |
| GET | `/samples`, `/samples/{name}` | Danh sách và file zip của các track mẫu trong `samples/` (chỉ khi thư mục tồn tại; tên file phải là `*.zip`, chặn đường dẫn ra ngoài thư mục) |
| GET | `/` | Giao diện web React (`webui/dist`); nếu chưa build thì chuyển sang `/docs` |

### POST /predict

`multipart/form-data`, trường `files` lặp lại cho từng ảnh. Tham số query:

| Tham số | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `decoder` | `greedy` \| `beam` \| `constrained` | theo `decoder_decision.json` (greedy) | Bộ giải mã |
| `top_k` | 1–5 | 3 | Số phương án trả về |
| `session_id` | chữ, số, `_`, `-`; tối đa 64 ký tự | — | Gom lịch sử theo phiên. UI tự sinh |

```bash
# 5 ảnh của một track
curl -s -X POST http://127.0.0.1:8000/predict \
  -F files=@lr-001.jpg -F files=@lr-002.jpg -F files=@lr-003.jpg \
  -F files=@lr-004.jpg -F files=@lr-005.jpg
# 1 file zip của cả thư mục track: chỉ lấy lr-*, bỏ hr-*; nếu có annotations.json thì trả kèm nhãn
curl -s -X POST "http://127.0.0.1:8000/predict?decoder=constrained&top_k=3" \
  -F files=@samples/review_track_22259.zip
```

Response thật của lệnh thứ hai (track `review_track_22259`, model đọc sai E→I nên bị gắn cờ):

```json
{
  "request_id": "c1f0…",
  "plate": "IAR5J37",
  "confidence": 0.905395,
  "sequence_prob": 0.449242,
  "char_conf": [0.856962, 0.895974, 0.999933, 0.999671, 0.999992, 0.585461, 0.999773],
  "top3": [{"plate": "IAR5J37", "prob": 0.449242},
           {"plate": "IAR5J27", "prob": 0.317079},
           {"plate": "IXR5J37", "prob": 0.051201}],
  "layout": "Mercosur",
  "n_frames": 5,
  "frame_weights": [0.234945, 0.218495, 0.10451, 0.216603, 0.225447],
  "sharpness": [436.29, 483.41, 351.68, 507.72, 395.54],
  "frames": [{"index": 0, "name": "lr-001.jpg", "width": 56, "height": 18,
              "weight": 0.234945, "sharpness": 436.29}, "… 4 frame nữa"],
  "needs_review": true,
  "review_reasons": ["low_confidence"],
  "triage": {"needs_review": true, "reasons": ["low_confidence"], "score_name": "mean_char",
             "score": 0.905395, "threshold": 0.9746480153143721},
  "warnings": [],
  "expected_accuracy": 79.12,
  "reference_plate": "EAR5J37",
  "decoder": "greedy",
  "latency_ms": 36.95,
  "timing": {"read_ms": 0.97, "model_ms": 32.0, "decode_ms": 2.38},
  "model_version": "restran-recipe-s42-ep30-9754ebb1915e"
}
```

Response trên được gọi với decoder mặc định. Với `decoder=constrained`, kết quả trùng ở track này.

| Trường | Ý nghĩa |
|---|---|
| `plate` | Biển số đọc được |
| `confidence` | Trung bình độ tin cậy từng ký tự, đúng định nghĩa của cuộc thi (dùng cho Confidence Gap). **Không phải xác suất đúng:** trên val 999, ECE của nó là 0,20, tức mô hình tự tin quá mức. Dùng để xếp hạng và so với ngưỡng |
| `sequence_prob` | p(chuỗi \| track), tính chính xác bằng thuật toán forward của CTC. Hiệu chỉnh tốt hơn `confidence` (ECE 0,08) |
| `char_conf` | Độ tin cậy từng ký tự. Ký tự thấp nhất là chỗ nên nhìn lại; ví dụ ở trên là vị trí 6, 0,59 |
| `top3` | Phần tử đầu luôn là `plate`, sau đó là các chuỗi khả dĩ nhất tiếp theo. Độ dài = `top_k` |
| `layout` | `Mercosur` / `Brazilian`, hoặc `null` nếu chuỗi sai mẫu |
| `frame_weights`, `sharpness`, `frames` | Trọng số attention của từng frame trong fusion (cộng lại = 1) và độ nét Laplacian (ảnh xám 32×128), để giải thích kết quả |
| `needs_review`, `review_reasons`, `triage` | Cờ "cần kiểm tra" và lý do: `low_confidence`, `invalid_layout`, hoặc `not_calibrated` (khi chưa có ngưỡng) |
| `warnings` | Ít hơn 5 frame, frame khác cỡ nhau, tỉ lệ ảnh không giống biển số, frame trùng nhau, chuỗi sai mẫu, zip có ảnh HR bị bỏ qua |
| `expected_accuracy` | Accuracy trung bình trên lock-box với đúng số frame này (E3: 55,6 / 69,0 / 74,1 / 77,0 / 79,1%) |
| `reference_plate` | Nhãn trong `annotations.json` nếu gửi zip của một track có nhãn; chỉ dùng cho demo và test |
| `model_version` | `restran-recipe-s42-ep30-` + 12 ký tự đầu SHA-256 của checkpoint |

### POST /feedback

```bash
curl -s -X POST http://127.0.0.1:8000/feedback -H 'Content-Type: application/json' \
  -d '{"request_id": "<request_id>", "correct": false, "corrected_plate": "EAR5J37"}'
# {"feedback_id": 1, "request_id": "<request_id>"}
```

## 3. Lỗi

Mọi lỗi có cùng một dạng JSON:

```json
{"error": {"code": "too_many_files", "message": "Nhận 6 ảnh, tối đa 5 ảnh của cùng một biển số"}, "request_id": "…"}
```

| HTTP | `code` | Khi nào |
|---|---|---|
| 400 | `no_files` | Không có trường `files` |
| 400 | `too_many_files` | Hơn 5 ảnh, hoặc zip có hơn 5 ảnh LR |
| 400 | `mixed_upload` | Gửi lẫn zip và ảnh |
| 400 | `zip_invalid` / `zip_no_images` | Zip hỏng / zip không có ảnh .jpg/.png |
| 401 | `unauthorized` | Đã bật `MFPLATE_API_KEY` mà thiếu hoặc sai `X-API-Key` |
| 404 | `not_found` | `/feedback` với `request_id` không tồn tại |
| 413 | `file_too_large` / `zip_too_large` / `image_too_large` | Ảnh > 2 MB / zip > 12 MB / cạnh ảnh > 2048 px |
| 415 | `unsupported_type` | Không phải JPEG/PNG (xét magic bytes, nên file GIF đổi đuôi `.jpg` vẫn bị từ chối) |
| 422 | `corrupt_image` / `empty_file` / `image_too_small` | Ảnh hỏng / file rỗng / cạnh < 4 px |
| 422 | `invalid_parameter` | `decoder`, `top_k`, `session_id` hoặc body `/feedback` sai |
| 503 | `model_unavailable` | Model chưa nạp được, chẳng hạn sai SHA-256 hoặc thiếu file |
| 500 | `internal_error` | Lỗi không lường trước (có ghi log) |

Bằng chứng: `tests/test_api.py` (23 test lỗi và test thường) và [`artifacts/error_cases.md`](../artifacts/error_cases.md) (gửi tới server thật).

## 4. Các quyết định thiết kế và bằng chứng

| Quyết định | Lý do |
|---|---|
| Nạp checkpoint theo một quy tắc cố định (cấu hình chốt, seed 42, epoch 30) và kiểm SHA-256 mỗi lần khởi động | Không chọn checkpoint theo điểm lock-box. SHA-256 phải trùng `results_new/freeze_manifest.json`, file ghi ngày 24/10 trước khi chấm lock-box |
| Ít hơn 5 ảnh thì đưa thẳng N frame, không lặp frame | AttentionFusion nhận số frame tuỳ ý, và E3 đo accuracy đúng theo cách này. Lặp frame sẽ làm trọng số softmax lệch về frame bị lặp |
| Ngưỡng riêng cho từng số frame | Cùng một ngưỡng thì precision của nhóm tự động tụt mạnh khi ít frame. Ngưỡng được chốt cho từng N trên val 999, mỗi N gộp mọi tập con frame |
| Greedy là decoder mặc định | Trên val 999, 4 seed: greedy 77,66, constrained 77,70. 0/893 lỗi có chuỗi đúng mang xác suất cao hơn chuỗi model chọn, nên decode không sửa được lỗi |
| Một lock quanh lần forward | Hook lấy trọng số attention phải thuộc đúng request. Trên CPU, PyTorch đã dùng mọi nhân cho một forward |
| Không lưu ảnh | Dữ liệu LRLPR-26 có giấy phép, và biển số là thông tin nhạy cảm. Log chỉ giữ hash |
