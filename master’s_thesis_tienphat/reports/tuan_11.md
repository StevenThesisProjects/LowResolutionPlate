# Tuần 11 (14–20/12): Viết kết quả và chuẩn bị demo

Chưa đến lịch. File này ghi những gì đã chuẩn bị sẵn.

## Roadmap yêu cầu

1. Viết Chương 4 (Experiments: 22 run + 3 seed + test) và Chương 5 (Prototype & Evaluation).
2. Kịch bản demo 5 phút với 3 tình huống (dễ / mờ / cần kiểm tra) → `demo_script.md`.
3. Ghi video dự phòng → `demo_backup.mp4`.
4. Chọn venue → `venue_decision.md`.

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng | Ghi chú |
|---|---|---|
| Chương 4: "22 run + 3 seed + test" | Nội dung thay đổi theo code mới: ablation cộng dồn (dev), 4 seed giao thức công bố, **lock-box chấm một lần**, E3/E4/N1, phân tích lỗi, các hướng không thành công (SR, EDVR, fusion temporal…) | Phát tự viết; nguồn số liệu ở bảng dưới |
| Chương 5 | API, UI, parity, triage, E2E, latency, user test | Phát tự viết; nguồn: báo cáo Tuần 5–10 |
| `demo_script.md` | `docs/demo_script.md` (đã viết) + 3 track mẫu trong `samples/` | ✅ |
| `demo_backup.mp4` | — | Phát ghi sau khi chạy Docker (Tuần 9) |
| `venue_decision.md` | Bài báo đã nhắm **ICCIES 2027** (LNCS, 12–15 trang, hạn 15/11/2026; `paper/main_v2.tex`). Hướng đi sau thesis theo review của thầy (mục 7.5): ICIP / IJCNN / ICDAR, hoặc tạp chí Q2–Q3 | Phát cập nhật trạng thái nộp và track đã chọn trên CMT |

## Nguồn số liệu cho từng chương

| Chương / mục | Số liệu | File |
|---|---|---|
| 4.1 Giao thức (dev, công bố, lock-box) | 18.001 / 19.001 track train; val 1.999 và 999; lock-box 1.000 track | `documents/lockbox_protocol.md`, bảng 2 trong `paper/main_v2.tex` |
| 4.2 Kết quả chính | 77,60 → 79,12 (+1,52, CI [+0,53; +2,52]) | `results_new/lockbox_eval.json`, `documents/REPORT_lockbox.md` |
| 4.3 Giao thức công bố | 77,25 → 78,73 (epoch cuối) | RUNS.md run #52–#58 |
| 4.4 E3 / E4 / N1 | Bảng ở Tuần 4 | `documents/REPORT_E4_N1.md` |
| 4.5 Ablation cộng dồn | #9 → #50 | Bảng 5 trong `paper/main_v2.tex`, RUNS.md |
| 4.6 Phân tích lỗi + decode | 58,7% lỗi sai 1 ký tự; 0/893 lỗi decode sửa được; SC-CTC +0,05 | `artifacts/error_syntax_audit.md`, `artifacts/results_decode.csv` |
| 4.7 Độ nét × layout (thăm dò) | Q1 +3,40, Q4 −0,10 | `artifacts/slice_sharpness.csv` |
| 5.1 Kiến trúc hệ thống, API | Flow, endpoint, mã lỗi | `docs/api.md`, `artifacts/openapi.json` |
| 5.2 Parity | 0/999; CPU = GPU | `artifacts/parity_final.json` |
| 5.3 Triage | Ngưỡng theo N; lock-box precision 95,4%, coverage 74,0% | `artifacts/threshold_calibration.json`, `artifacts/triage_lockbox.json`, `artifacts/risk_coverage_val.png` |
| 5.4 E2E, lỗi, latency | 30/30; 14/14; p95 24 ms | `artifacts/test_report.md`, `artifacts/error_cases.md`, `artifacts/latency_summary.json` |
| 5.5 User test | SUS | `artifacts/sus_results.csv` (chưa có) |
