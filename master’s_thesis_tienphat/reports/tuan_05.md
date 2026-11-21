# Tuần 5 (02–08/11): Quyết định cấu hình cuối và hiệu chỉnh ngưỡng

## Roadmap yêu cầu

1. Áp improvement gate cho pipeline và LGTF, chốt cấu hình → `stats_report.md`, `final_config_decision.json`.
2. Hiệu chỉnh ngưỡng confidence trên val, sao cho precision của nhóm "tự động" ≥ 95% → `threshold_calibration.json`.
3. Viết nháp Chương 3 (Method).

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng | Lý do |
|---|---|---|
| Gate + chốt cấu hình | Cấu hình **#50 (combo3)** được **đóng băng ngày 23/10** theo ISSUE #3 của review: giữ nguyên, mọi thí nghiệm sau đó chỉ để mô tả. Quy tắc quyết định (a)/(b) viết trong `documents/lockbox_protocol.md` **trước khi chạy** lock-box | Đổi cấu hình sau khi thấy kết quả là thêm một vòng chọn trên val |
| `stats_report.md` | `documents/REPORT_lockbox.md` (hai tầng thống kê: Welch, Cohen's d theo seed; bootstrap ghép cặp, McNemar theo track) và `documents/REPORT_E4_N1.md` | Đã có |
| `final_config_decision.json` | `results_new/freeze_manifest.json` (cấu hình + SHA-256) và `artifacts/serving_config.json` (cờ train, quy tắc chọn checkpoint) | Đã có |
| Ngưỡng confidence | **Làm mới:** `scripts/calibrate_threshold.py` → `artifacts/threshold_calibration.json`, chốt trên val 999 cho model phục vụ, **riêng cho từng số frame (1–5)** | App nhận 1–5 frame, và cùng một ngưỡng sẽ cho precision rất khác nhau theo số frame |
| Nháp Chương 3 | Mục 3 của bài báo ICCIES (`paper/main_v2.tex`, Phát tự viết) mô tả đủ phương pháp: 6 thành phần, công thức aux CTC (Eq. 1), frame dropout (Eq. 2), che từng frame | Phát viết lại thành chương luận văn |

## Kết quả: ngưỡng "cần kiểm tra"

**Quy tắc, viết trong docstring của script trước khi chạy:**
1. Trong ba điểm tin cậy (`mean_char` = trung bình độ tin cậy từng ký tự, đúng định nghĩa của cuộc thi; `min_char`; `seq_prob` = xác suất CTC của cả chuỗi), chọn điểm có AURC thấp nhất trên val 999 với 5 frame.
2. Ngưỡng cho N frame là giá trị t thấp nhất mà nhóm có điểm ≥ t đạt precision ≥ 95% trên val 999. Với N < 5, gộp mọi tập con N frame.
3. Chỉ xem lock-box sau khi ngưỡng đã ghi ra file.

**Chọn điểm:** AURC trên val: `mean_char` 0,04126, `seq_prob` 0,04136, `min_char` 0,04246 → chọn `mean_char`. Hai điểm đầu gần như bằng nhau.

**Ngưỡng và kết quả trên val 999** (model phục vụ, greedy):

| N frame | Ngưỡng | Tỉ lệ tự động | Precision nhóm tự động (cận dưới Wilson 95%) | Lỗi giữ lại ở nhóm cần kiểm tra |
|---|---|---|---|---|
| 1 | 0,99795 | 30,1% | 95,01% (93,79%) | 2.186 / 2.261 |
| 2 | 0,99217 | 54,7% | 95,00% (94,39%) | 2.926 / 3.199 |
| 3 | 0,98665 | 64,4% | 95,01% (94,45%) | 2.358 / 2.679 |
| 4 | 0,98198 | 69,4% | 95,01% (94,23%) | 1.035 / 1.208 |
| 5 | 0,97465 | 72,8% | 95,05% (93,22%) | 197 / 233 |

**Kiểm lại trên lock-box** (`--lockbox`; mô tả; tính sau khi ngưỡng đã ghi ra file):

| N frame | Tỉ lệ tự động | Precision nhóm tự động (cận dưới Wilson) | Lỗi giữ lại |
|---|---|---|---|
| 1 | 31,9% | 95,99% (94,91%) | 2.151 / 2.215 |
| 2 | 55,0% | 95,46% (94,87%) | 2.866 / 3.116 |
| 3 | 64,6% | 95,15% (94,60%) | 2.262 / 2.575 |
| 4 | 69,6% | 95,46% (94,71%) | 972 / 1.130 |
| 5 | **74,0%** | **95,41%** (93,65%) | **167 / 201** |

- Ngưỡng chốt trên val vẫn giữ precision ≥ 95% trên 1.000 track chưa dùng, ở mọi số frame.
- Với 5 frame: app tự xử lý 74% track với khoảng 4,6% sai; 26% còn lại chuyển cho người kiểm tra và chứa 83% số lỗi.
- **Độ tin cậy không phải xác suất đúng.** ECE trên val (15 bin): `mean_char` 0,20, `min_char` 0,12, `seq_prob` 0,08. Model tự tin quá mức: mỗi ký tự thường có độ tin cậy trên 0,98, trong khi accuracy chỉ 77%. Điểm số dùng tốt để **xếp hạng** (AURC thấp), nhưng không nên đọc như "99% là đúng". UI hiển thị cả hai con số và ghi rõ ngưỡng. Hiệu chỉnh nhiệt độ (temperature scaling) chưa làm.
- Confidence Gap (định nghĩa của cuộc thi), model phục vụ trên val 999: 0,1046. Trên lock-box, 4 seed: công thức 0,1135, baseline 0,1060. Công thức không làm giảm Confidence Gap (trả lời MINOR 12 của review).

Hình: `artifacts/risk_coverage_val.png`. Số liệu theo mức coverage: `artifacts/risk_coverage_val.csv`.

## Deliverable

| Roadmap | File thay thế | Trạng thái |
|---|---|---|
| `stats_report.md` | `documents/REPORT_lockbox.md`, `documents/REPORT_E4_N1.md` | ✅ |
| `final_config_decision.json` | `results_new/freeze_manifest.json`, `artifacts/serving_config.json`, `artifacts/decoder_decision.json` | ✅ |
| `threshold_calibration.json` | `artifacts/threshold_calibration.json` + `artifacts/triage_lockbox.json` | ✅ |
| Nháp Chương 3 | Mục 3 của `paper/main_v2.tex` | ⚠️ Phát chuyển thành chương luận văn |
