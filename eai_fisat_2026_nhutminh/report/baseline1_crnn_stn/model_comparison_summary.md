# So sánh model — baseline CRNN+STN · J1 · S1 · S4

> **Phạm vi**: chỉ baseline CRNN+STN và 3 cấu hình có **error bar trên 3 seed**
> (J1, S1, S4). Các ablation 1-seed đã chạy trước đây (λ_SR, perceptual loss,
> single-frame SR, AdamW tuning, các hướng SR thất bại ban đầu) nằm ở `backup/report/`
> — không đưa vào bảng chính vì không có error bar.
>
> 🎯 **Phương pháp đề xuất: S1** — `79.95% ± 0.15`.

## 1. Bảng chính

Val = **999 track** Scenario-B. Biên nhiễu **±13 track (±1.3 điểm)**.

| # | Model | SR | `T` | Seed | **Val Acc** | CER ↓ | GFLOPs | Vai trò |
|---|---|---|---:|:---:|---:|---:|---:|---|
| 0 | **CRNN + STN** (baseline report ICPR) | ❌ | 16 | — | **77.00%** | — | — | mốc tham chiếu |
| 1 | CRNN + STN (đo lại trên dataset project) | ❌ | 16 | 1 | 75.78% | — | — | mốc tham chiếu |
| 2 | ResBlock backbone (`norm=none`) | ❌ | 16 | 1 | 76.68% | — | — | nâng cấp backbone |
| 3 | **J1** — ResBlock + GroupNorm, không SR | ❌ | 16 | **3** | **80.45% ± 0.45** | **0.0525** | **26.14** | ablation _"bỏ hẳn SR"_ |
| 4 | **S1** — Joint MF-SR-OCR ×2 | ✅ ×2 | 32 | **3** | **79.95% ± 0.15** | 0.0541 | 109.08 | 🎯 **đề xuất** |
| 5 | **S4** — Joint MF-SR-OCR ×1 | ✅ ×1 | 32 | **3** | **79.48% ± 0.44** | 0.0543 | chưa đo | ablation _"bỏ phóng to ảnh"_ |

> ⚠️ Dòng 0–2 là **1 seed, `cudnn.benchmark=True`** — đọc như mốc tham chiếu lịch sử,
> không so trực tiếp với dòng 3–5 (3 seed, deterministic).

## 2. Kiểm định từng cặp (chỉ 3 model có error bar)

Ngưỡng: chênh > **2× sai số hiệu** mới coi là thật.

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| J1 vs S1 | +0.50 | ±0.28 | ⚠️ trong nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

## 3. Vì sao chọn S1 dù J1 có điểm trung bình nhỉnh hơn

1. **J1 không hơn S1 có ý nghĩa** (+0.50, trong nhiễu) → thống kê là **hoà**; chọn
   giữa hai cấu hình hoà nhau thì quyết định bằng **độ ổn định**.
2. **S1 ổn định nhất** — std `0.15`, nhỏ hơn **3×** so với J1 (0.45) và S4 (0.44).
3. **S1 bất biến với `cudnn.benchmark`** — cùng seed 42, chạy `benchmark=True` và
   `deterministic=True` cho **đúng cùng con số** (797/999, lệch **0 track**), trong khi
   S4 lệch **16 track**. Con số của S1 **tái lập được nhất** trong cả project.
4. S1 là kiến trúc hoàn chỉnh mà bài xây câu chuyện quanh nó; J1/S4 đóng đúng vai
   **ablation** của chính S1.

## 4. Ma trận cấu hình — cái gì bật ở model nào

| | **J1** | **S1** | **S4** |
|---|:---:|:---:|:---:|
| `--use-sr` | ❌ | ✅ `--sr-scale 2` | ✅ `--sr-scale 1` |
| `--use-dcn` (DCNv2) | ❌ | ✅ | ✅ |
| Multi-frame SR | ❌ | ✅ | ✅ |
| `--width-downsample` | 8 | 8 | **4** |
| Chuỗi CTC `T` | **16** | 32 | 32 |
| `--backbone-norm group` | ✅ | ✅ | ✅ |
| STN pool | `(4,8)` | `(4,8)` | `(4,8)` |
| `--lr-domain-match` | ✅ | ✅ | ✅ |
| `--decode constrained` | ✅ | ✅ | ✅ |
| `--use-ema` | ✅ | ✅ | ✅ |
| `λ_SR` | — | 0.1 | 0.1 |
| `α` (perceptual) | — | **0** | **0** |
| Params | 29,442,700 | 29,577,214 | 29,549,470 |

**4 dòng cuối cùng của cụm cờ nền** (GroupNorm, STN pool, domain-match, constrained,
EMA) **giống hệt nhau ở cả 3** → J1 vs S1 là ablation **1 cụm biến sạch**: chỉ khác
SR + DCN + MFSR.

$$T = \frac{\text{IMG\_WIDTH} \times (\text{SR\_SCALE nếu USE\_SR})}{\text{WIDTH\_DOWNSAMPLE}}$$

- J1: $128 / 8 = 16$ · S1: $(128 \times 2)/8 = 32$ · S4: $(128 \times 1)/4 = 32$

## 5. Chi phí compute

| | GFLOPs/track | Latency (ms) | Phút/epoch | Tổng 3 seed | vs J1 |
|---|---:|---:|---:|---:|---:|
| **J1** | **26.14** | **51.30** | **2.60** | **5.55 h** | **1.00×** |
| S1 | 109.08 | 192.75 | 9.67 | 22.24 h | **3.76×** |
| S4 | chưa đo | chưa đo | 4.28 | 11.27 h | — |

Phút/epoch là **số đo thật** từ cột `epoch_time_s` của 424 epoch, không phải ước tính.

> ⚠️ `tools/benchmark.py` chưa benchmark được S4 — thiếu cờ `--sr-scale` và
> `--width-downsample` nên không dựng được kiến trúc `sr_scale=1 + width/4`.

## 6. 🚨 Kết quả âm tính bắt buộc công bố

- **Nhánh SR chưa chứng minh được đóng góp đo được.** J1 (bỏ hẳn SR/DCN/MFSR, dùng
  chung toàn bộ cụm cờ nền với S1) **hoà** S1 trong khi rẻ hơn **3.76× compute**.
  Kết luận đúng: _"SR không cho thấy lợi ích trên tập val này"_ — **không** phải
  _"bỏ SR thì tốt hơn"_ (chênh trong nhiễu, không kết luận được chiều nào).
- **Phần cải thiện thật đến từ cụm cờ nền**: STN pool `(4,8)` + `--lr-domain-match` +
  constrained decode + EMA. J1 hơn phiên bản J1 lịch sử **+3.57 điểm** (76.88% →
  80.45%) với **cùng** kiến trúc không SR.
- **Giả thuyết `T=32` bị bác bỏ** — J1 chạy `T=16` mà vẫn ngang S1 và hơn S4.
- **Chưa tách được từng thành phần trong cụm cờ nền** — câu hỏi mở lớn nhất còn lại.

## 7. Tài liệu nào trả lời câu hỏi gì

| Câu hỏi | Đọc file |
|---|---|
| Kiến trúc baseline CRNN+STN gốc | [baseline1_architecture.md](baseline1_architecture.md) |
| Backbone ResBlock mà cả 3 model dùng | [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) |
| J1 — ablation bỏ hẳn SR | [j1_groupnorm_nosr.md](j1_groupnorm_nosr.md) |
| S1 — phương pháp đề xuất | [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) |
| S4 — SR ×1, tách biến `T` | [s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md) |
| Số chính thức Mean ± Std của cả 3 | [multi_seed_results.md](multi_seed_results.md) |
| CER / NED / PSNR / SSIM | [../buoc2_metrics.md](../buoc2_metrics.md) |
| Công thức loss đã sửa cho paper | [../paper/loss_formula_corrected.md](../paper/loss_formula_corrected.md) |
| Lệnh chạy | [../training_runs/run_gpu.md](../training_runs/run_gpu.md) |
| Cấu trúc dataset | [../summary_project/dataset/dataset_overview.md](../summary_project/dataset/dataset_overview.md) |
