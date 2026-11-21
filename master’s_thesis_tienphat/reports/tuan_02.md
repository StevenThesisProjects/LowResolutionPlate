# Tuần 2 (12–18/10): Đánh giá SC-CTC và seed của mốc

## Roadmap yêu cầu

1. Train run #13 thêm seed 123 và 999.
2. So greedy / beam / SC-CTC trên cả 3 checkpoint, McNemar cặp đôi → `results_decode.csv` (3 seed × 3 decoder, theo layout).
3. Đọc paper tổng kết ICPR, lập bảng so sánh top-5 → `related_work_top5.md`.
4. Song song: FastAPI `/health`, `/predict`, API trả đúng 1 track val.

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng | Lý do |
|---|---|---|
| #13 với seed 123, 999 | Seed của **cấu hình chốt và baseline**: giao thức công bố (val 999 của BTC) run #52–#58, 4 seed mỗi bên, ngày 18/10; lock-box run #63–#70, 4 seed mỗi bên, ngày 24–25/10 | Mốc đã đổi (xem `00_mo_dau.md`). Dùng 4 seed thay 3 seed, vì giữa các seed khác biệt khoảng 0,5 điểm |
| So 3 decoder trên 3 checkpoint #13 | So 3 decoder trên **4 model chốt** (lock-box, epoch cuối), val 999, có McNemar và tách theo layout | Đo trên model thật sự phục vụ |
| `related_work_top5.md` | Bảng ở mục Kết quả, lấy từ RUNS.md (01/10) và roadmap; còn thiếu đội hạng 4 | Bài báo ICCIES dùng phần related work riêng |
| API `/health`, `/predict` | `mfplate/api.py` bản đầy đủ (Tuần 7) | Làm một lần, đủ chức năng |

## Kết quả

**Seed của cấu hình chốt** (số đã có, không train thêm):

| Giao thức | Baseline | Công thức #50 | Chênh | Kiểm định |
|---|---|---|---|---|
| Công bố, val 999, epoch cuối (4 seed) | 77,25 ± 0,46 | 78,73 ± 0,30 | +1,48 | Welch p = 0,003 |
| Lock-box 1.000 track, epoch cuối (4 seed) | 77,60 ± 0,24 | 79,12 ± 1,14 | +1,52 | CI 95% theo track [+0,53; +2,52]; Welch p = 0,072 |

**So sánh decoder** (`scripts/eval_decoders.py` → `artifacts/results_decode.csv`; val 999; beam 10):

| Seed | Greedy | Beam | SC-CTC (constrained) | Track SC-CTC đổi kết quả | Sửa đúng / làm sai | McNemar p |
|---|---|---|---|---|---|---|
| 42 | 76,68 | 76,68 | 76,68 | 1 | 0 / 0 | 1,0 |
| 43 | 79,08 | 79,08 | 79,08 | 0 | 0 / 0 | 1,0 |
| 7 | 77,98 | 77,98 | 77,98 | 1 | 0 / 0 | 1,0 |
| 2026 | 76,88 | 76,88 | 77,08 | 10 | 2 / 0 | 0,5 |
| **Trung bình** | **77,66** | **77,66** | **77,70** | | | |

- Theo layout (seed 42): Mercosur 82,89 / Brazilian 53,33, giống nhau ở cả 3 decoder.
- **Quyết định** (tiêu chí ghi trong `scripts/eval_decoders.py` trước khi chạy): một decoder beam chỉ thành mặc định nếu hơn greedy trung bình ≥ +0,30 điểm và không thua ở seed nào. SC-CTC chỉ hơn +0,05, nên **decoder mặc định của app là greedy** (`artifacts/decoder_decision.json`). SC-CTC vẫn chọn được qua tham số `decoder=constrained`, và luôn trả chuỗi đúng mẫu (0 chuỗi sai mẫu ở cả 4 seed).
- Lock-box, chỉ để mô tả, tính sau khi đã có quyết định: greedy 79,12 / SC-CTC 79,15 (`artifacts/results_decode_lockbox.csv`).
- Lý do SC-CTC không giúp được có ở Tuần 1: chỉ 1,3% lỗi vi phạm cú pháp, và 0% lỗi có chuỗi đúng xác suất cao hơn.

**Top-5 ICPR 2026 LRLPR** (blind test; nguồn: paper tổng kết arXiv 2604.22506, tóm tắt trong RUNS.md mục 01/10 §4 và roadmap mục Lưu ý 2):

| Hạng | Blind test | Hướng chính | Điểm chung với đề tài |
|---|---|---|---|
| 1 | 82,13% | SR + OCR teacher–student, dữ liệu ngoài, ensemble 3 model | Dùng ensemble và dữ liệu ngoài, hai thứ đề tài loại khỏi phạm vi |
| 2 | 81,73% | STN + SE-ResNet34 + Transformer + CTC + attention fusion; aux CTC, center loss, OHEM; ensemble 4 + ràng buộc layout | Cùng họ kiến trúc với ResTran; cũng có aux CTC và ràng buộc layout |
| 3 | 80,17% | Không dùng SR | — |
| 4 | **chưa điền** | — | Phát cần đọc lại paper tổng kết để điền |
| 5 | — | Decode theo cấu trúc biển (structure-aware decoding) + trọng số frame theo chất lượng | Trùng ý tưởng SC-CTC và LGTF |

Hệ quả cho luận văn: không tuyên bố "đầu tiên" cho decode ràng buộc hay fusion theo chất lượng. Đóng góp được định vị là công thức huấn luyện **một model, không ensemble, không dữ liệu ngoài**, kèm đánh giá đăng ký trước.

**API:** `GET /health` và `POST /predict` đã có (Tuần 7). Test `test_predict_five_frames` gửi 5 frame của `track_14926` (val 999, nhãn GGB8A08) và nhận đúng GGB8A08, layout Mercosur.

## Deliverable

| Roadmap | File thay thế | Trạng thái |
|---|---|---|
| `results_decode.csv` (3 seed × 3 decoder, theo layout) | `artifacts/results_decode.csv` (4 seed × 3 decoder, theo layout, McNemar) | ✅ |
| `related_work_top5.md` | Bảng ở trên | ⚠️ thiếu đội hạng 4 |
| API trả đúng 1 track val | `tests/test_api.py::test_predict_five_frames` | ✅ |

## Việc còn lại

- Điền đội hạng 4 và kiểm lại mô tả của các đội 1–5 theo paper tổng kết, trước khi đưa vào chương Related Work của luận văn.
