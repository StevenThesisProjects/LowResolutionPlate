# Tuần 1 (05–11/10): Khóa mốc và đo trần SC-CTC

## Roadmap yêu cầu

1. Chạy lại run #13 với tên riêng, lưu SHA-256 checkpoint → `freeze_baseline.json`.
2. Sửa import `restran_lgtf.py`.
3. Viết unit test cho decoder (≥ 10 case) → `tests/test_decoder.py`.
4. Trên val: đếm % lỗi vi phạm cú pháp (trần lợi ích của SC-CTC), confusion matrix ký tự theo layout → `error_syntax_audit.md`.
5. Trả lời 5 câu hỏi ở Mục 0.

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng (code mới) | Lý do |
|---|---|---|
| Khóa mốc run #13 | **Hai cấu hình đóng băng của lock-box:** baseline đã ổn định hoá và công thức #50, mỗi bên 4 seed, train ngày 24/10. `freeze_manifest.json` ghi SHA-256 của 56 checkpoint, 24 file code và 3 file split (digest `39c39dfc…7824`). **Model phục vụ app** lấy theo quy tắc cố định: công thức #50, seed 42, epoch 30, SHA-256 `9754ebb1…7bba`, trùng manifest | Sau review 23/10, #13 không còn là mốc. Bản thân #13 đã được chạy lại hai lần (#25: 77,04%; #29: 77,64%), cho thấy dao động khoảng 0,5 điểm giữa các lần chạy cùng cấu hình |
| Sửa `restran_lgtf.py` | **Không cần:** file không có trong code mới. Ý tưởng LGTF (`score = ScoreNet(F) + γ·log Var_Laplacian`) có trong code mới dưới dạng cờ `--sharpness-prior` của `AttentionFusion` (`src/models/components.py`), import đủ | Roadmap viết theo PR cũ |
| `tests/test_decoder.py` | `tests/test_decoding.py`: **22 test, đạt cả 22** | Decoder mới nằm trong `mfplate/decoding.py` |
| `error_syntax_audit.md` trên model #13 | `artifacts/error_syntax_audit.md` trên **4 model chốt** (lock-box, epoch cuối), val 999 | Đo trên model thật sự được dùng |
| 5 câu hỏi | Trả lời ở cuối file | — |

## Kết quả

**Model phục vụ** (`scripts/freeze_serving_model.py` → `artifacts/serving_config.json`):
- Quy tắc chọn, cố định trước mọi phép đo của app:
  - công thức #50, vì đây là cấu hình đã đóng băng và đã xác nhận trên lock-box;
  - epoch cuối, vì đây là thước đo chính của protocol nên không phải chọn epoch;
  - seed 42, vì là seed đầu tiên trong protocol. Không chọn seed 43 (80,30% trên lock-box), vì như vậy là chọn theo lock-box.
- Model train trên 18.001 track, chưa từng thấy val 999 và lock-box, nên dùng được cả hai tập để đo app.

**Unit test decoder** (`tests/test_decoding.py`, 22 test):
- greedy gộp ký tự lặp và bỏ blank; "AA" cần blank xen giữa; toàn blank cho chuỗi rỗng;
- greedy cho đúng chuỗi và confidence như `decode_with_confidence()` của code train (50 ma trận ngẫu nhiên);
- xác suất CTC bằng thuật toán forward khớp phép liệt kê mọi đường đi (bảng chữ 3 ký hiệu, T = 5);
- chuỗi 7 ký tự không thể khớp vào 4 bước thời gian thì có xác suất 0;
- beam search: phần tử đầu bằng greedy khi phân phối nhọn; kết quả sắp giảm dần, không trùng; tìm được chuỗi mà greedy bỏ sót khi xác suất dồn từ nhiều đường đi;
- beam có ràng buộc: chỉ trả chuỗi đúng mẫu; sửa được chữ số nằm ở vị trí phải là chữ cái; vẫn ra biển 7 ký tự khi model chỉ đọc chắc 6 ký tự;
- confidence theo căn chỉnh Viterbi trùng confidence greedy;
- phần tử đầu của top-k luôn là kết quả trả về;
- `layout_of` cho 7 trường hợp, gồm cả mẫu biển xe máy `LLLNNLN`, bị coi là sai mẫu.

**Audit lỗi** (`scripts/syntax_audit.py`; val 999; 4 seed; decoder greedy):

| Seed | Accuracy | Số lỗi | Vi phạm cú pháp | Đúng mẫu nhưng của layout kia | p(nhãn đúng) > p(dự đoán) |
|---|---|---|---|---|---|
| 42 | 76,68 | 233 | 1 | 0 | 0 |
| 43 | 79,08 | 209 | 0 | 3 | 0 |
| 7 | 77,98 | 220 | 1 | 1 | 0 |
| 2026 | 76,88 | 231 | 10 | 1 | 0 |

- **Trần lợi ích của SC-CTC là 12/893 lỗi (1,3%).** 12 lỗi này là những lỗi sai độ dài; không có lỗi nào đủ 7 ký tự mà sai mẫu.
- **0/893 lỗi** có chuỗi đúng mang xác suất CTC cao hơn chuỗi model chọn. Vì vậy không cách decode nào sửa được các lỗi này, kể cả SC-CTC. Kết quả giống phép đo trên model #9 ngày 01/10 (456/456 lỗi).
- 58,7% lỗi sai đúng 1 ký tự, khớp con số khoảng 58% trong bài báo. Vị trí 2–3 (chữ cái) sai nhiều nhất.
- Cặp nhầm nhiều nhất: 6↔8 (89 lần), H↔M (50), 4↔6 (47), O↔Q (46), 5↔6 (44). Bảng theo layout nằm trong `artifacts/error_syntax_audit.md`.

## Trả lời 5 câu hỏi ở Mục 0

1. **Test 3.000 track có nhãn:** không dùng. Bản đang có (`Pa7a3Hin-test-public`, `TKzFBtn7-test-blind`) không có `annotations.json`. Ngày 31/10 thầy quyết định không dùng tập test chính thức. Lock-box 1.000 track thay vai trò tập test (Tuần 6).
2. **Hạn nộp và ngày bảo vệ:** roadmap đặt demo vào 27/12/2026. **Ngày bảo vệ chính thức Phát cần xác nhận với khoa**; tài liệu trong repo chưa ghi.
3. **React/Next.js:** chọn phương án A của roadmap (Gradio). UI chỉ gọi API qua HTTP, nên sau này đổi sang Next.js không phải sửa backend.
4. **V100:** không có liên tục mà thuê theo đợt: lock-box trên `fierce-ibex` (24–25/10), E4 ngày 28/10, N1 ngày 30/10, mỗi đợt một server. Demo chạy trên CPU (latency ở Tuần 10), không cần GPU.
5. **Biển xe máy Mercosur (`LLLNNLN`):** **không có.** Đếm toàn bộ 20.000 nhãn train: Scenario-A có 5.000 Brazilian (`LLLNNNN`) và 5.000 Mercosur (`LLLNLNN`); Scenario-B có 2.000 Brazilian và 8.000 Mercosur. Có 1 nhãn thừa dấu cách (`track_06546`: `"ARO3383 "`). Dataset bỏ ký tự ngoài bảng chữ, nên nhãn train vẫn đúng 7 ký tự. Lỗi `detect_plate_layout()` chỉ xét ký tự thứ 5 không còn trong code mới: app xác định layout bằng `layout_of()`, xét đủ 7 ký tự.

## Deliverable

| Roadmap | File thay thế | Trạng thái |
|---|---|---|
| `freeze_baseline.json` | `results_new/freeze_manifest.json` + `artifacts/serving_config.json` | ✅ |
| `tests/test_decoder.py` (≥ 10 case) | `tests/test_decoding.py` (22 test) | ✅ |
| `error_syntax_audit.md` | `artifacts/error_syntax_audit.md` (+ `.json`) | ✅ |

## Lệnh tái lập

```bash
# trong thư mục masters_thesis_tienphat/
python3 scripts/freeze_serving_model.py
python3 scripts/offline_cache.py --split val --runs combo3_s42 combo3_s43 combo3_s7 combo3_s2026
python3 scripts/syntax_audit.py
python3 -m pytest tests/test_decoding.py
```
