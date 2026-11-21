# Tuần 6 (09–15/11): Đóng băng và chấm một lần

## Roadmap yêu cầu

1. Freeze weights, config, decoder và ngưỡng → `freeze_manifest.json`.
2. Chạy test 3.000 track **đúng 1 lần** → `test_results_oneshot.json` + log của lần chạy đó. Báo cáo tổng, theo layout, Confidence Gap, coverage/precision của triage.
3. Export TorchScript, đo latency CPU → `latency_cpu.csv`.

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng | Lý do |
|---|---|---|
| Test 3.000 track | **Lock-box 1.000 track Scenario-B**, chọn tất định (`random.Random(2027)`) từ 8.001 track trước đó chỉ dùng để train, không trùng hai tập val. Gồm 192 Brazilian và 808 Mercosur. Train lại 4 + 4 run không có các track này, **chấm đúng một lần ngày 25/10** | Ngày 31/10 thầy quyết định không dùng tập test chính thức. Phương án lock-box là phương án dự phòng trong ISSUE #1 của review |
| Freeze manifest | `results_new/freeze_manifest.json`, viết ngày 24/10 trước khi chấm: 24 file code, 3 split, 56 checkpoint, môi trường (V100, PyTorch 2.14.1+cu126); digest `39c39dfc…7824` | Đã có |
| Freeze decoder + ngưỡng | `artifacts/decoder_decision.json` (greedy) và `artifacts/threshold_calibration.json`, cả hai chốt trên val 999. App kiểm SHA-256 checkpoint mỗi lần khởi động | Làm mới trong thư mục này |
| Triage trên tập test | Triage trên lock-box, chỉ để mô tả (Tuần 5) | — |
| TorchScript | **Không cần.** Model eager trên CPU đã nhanh (bên dưới), dưới tiêu chí 1 s rất xa. Roadmap cũng chỉ ghi "TorchScript nếu cần giảm latency" | Tránh thêm một bản model khác cần kiểm parity |
| `latency_cpu.csv` | `artifacts/latency_cpu.csv` + `artifacts/latency_summary.json` (Tuần 10) | — |

## Kết quả lock-box (chấm một lần, 25/10)

| Thước đo | Baseline | Công thức #50 | Chênh | CI 95% bootstrap theo track | Welch p theo seed | Cohen's d |
|---|---|---|---|---|---|---|
| **Epoch cuối** (chính) | 77,60 ± 0,24 | **79,12** ± 1,14 | **+1,52** | [+0,53; +2,52] | 0,072 | 1,85 |
| Last-6 (quyết định) | 77,07 ± 0,40 | 78,71 ± 1,15 | +1,64 | [+0,78; +2,48] | 0,059 | 1,90 |
| Best theo val 999 | 77,62 ± 0,31 | 79,20 ± 1,30 | +1,57 | [+0,57; +2,55] | 0,090 | 1,67 |

- Quy tắc đăng ký trước (chênh Last-6 ≥ +1,0 và cận dưới CI > 0) chọn **hướng (b)**.
- Theo seed thì kết quả chưa đạt p < 0,05, vì công thức dao động giữa các seed nhiều hơn. Dù vậy, cả 4 seed công thức đều cao hơn cả 4 seed baseline. Kiểm định hoán vị cho p = 0,029, nhưng phép này không đăng ký trước.
- McNemar từng cặp seed (epoch cuối): seed 42: +60/−35 (p = 0,013); seed 43: +59/−30 (p = 0,003); seed 7: +49/−47 (p = 0,92); seed 2026: +51/−46 (p = 0,68).
- Theo layout: Mercosur 81,62 → 83,23; Brazilian 60,68 → 61,85 (n = 192, chưa đủ lực thống kê).
- Confidence Gap: baseline 0,1060, công thức 0,1135.
- Model phục vụ app (seed 42, epoch 30) đạt 79,90% trên lock-box. Con số này chỉ để tham khảo; con số của phương pháp là trung bình 4 seed, 79,12%.

**Chạy lại trên CPU** (`scripts/parity_check.py --lockbox`): API trên Apple M4 cho 79,90% trên lock-box, **0/1.000 track** khác đúng/sai so với lần chấm GPU. Cache offline của cả 4 seed cũng cho đúng từng con số GPU (79,90 / 80,30 / 78,10 / 78,20).

Chi tiết: `documents/REPORT_lockbox.md`; số liệu: `results_new/lockbox_eval.json`, `results_new/lockbox_eval.log`.

## Deliverable

| Roadmap | File thay thế | Trạng thái |
|---|---|---|
| `freeze_manifest.json` | `results_new/freeze_manifest.json` (+ `artifacts/serving_config.json`) | ✅ |
| `test_results_oneshot.json` + log | `results_new/lockbox_eval.json` + `results_new/lockbox_eval.log` | ✅ (lock-box thay test) |
| `latency_cpu.csv` | `artifacts/latency_cpu.csv` | ✅ |
| Export TorchScript | Không làm, vì không cần | — |
