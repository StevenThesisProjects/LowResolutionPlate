# Bước 1 (Multi-Seed J1/S1/S4) + Bước 2 (Metrics) + Bước 3 (Qualitative Figures)

`feature/run-multi-seeda-and-evulate` → `main` · 9 commits · 91 files (+8,731/−280), code 7 files (+748/−22)

> 🚨 **Kết quả đảo ngược giả định ban đầu**: cấu hình **không có SR (J1)** đạt điểm
> **cao nhất** — hoà S1 (đề xuất) và **hơn S4 có ý nghĩa thống kê**, trong khi rẻ hơn
> **3.76× compute**. **Nhánh SR chưa chứng minh được đóng góp nào.** Kết luận paper
> cần viết lại theo hướng này.

---

## 1. Bước 1 — Multi-Seed ✅

3 seed `42/100/2026`, deterministic (`--no-cudnn-benchmark`).

| Model | SR | `T` | GFLOPs | **Mean ± Std** | CER ↓ | Hạng |
|---|---|---:|---:|---:|---:|:---:|
| **J1** — GroupNorm, **không SR** | ❌ | 16 | **26.14** | **80.45% ± 0.45** | **0.0525** | 🥇 |
| **S1** — Joint MF-SR-OCR ×2 (đề xuất) | ×2 MFSR+DCN | 32 | 109.08 | 79.95% ± 0.15 | 0.0541 | 🥈 |
| **S4** — SR ×1, `T=32` qua backbone | ×1 MFSR+DCN | 32 | chưa đo | 79.48% ± 0.44 | 0.0543 | 🥉 |

**Kiểm định** (ngưỡng: chênh > 2× sai số hiệu):

| Cặp | Chênh | Sai số | Kết luận |
|---|---:|---:|---|
| J1 vs S1 | +0.50 | ±0.28 | ⚠️ trong nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

### Hai giả thuyết bị bác bỏ

1. **"SR đóng góp vào độ chính xác"** — J1 và S1 dùng **chung cụm cờ nền** (GroupNorm,
   STN pool `(4,8)`, domain-match, constrained decode, EMA); S1 chỉ thêm SR+DCN+MFSR.
   Thêm cụm đó **không cải thiện** mà tốn 3.76× compute. Phần cải thiện thật (**+3.57
   điểm**, 76.88% → 80.45%) đến từ **cụm cờ nền**, với cùng kiến trúc không SR.
2. **"`T=32` là yếu tố chính"** — J1 chạy **`T=16`** mà vẫn ngang/hơn S1, S4 (đều `T=32`).

### 2 lưu ý

- **J1 chạy cờ nền S1/S4, không phải cờ lịch sử** (STN pool `(4,8)`, params
  29,442,700 vs 29,313,452). Đây là ablation 1-cụm-biến sạch so với S1 — tốt hơn về
  khoa học, nhưng **80.45% không so được với 76.88%** của J1 lịch sử.
- **`cudnn.benchmark` ảnh hưởng không đồng đều**: cùng seed 42, S1 lệch **0 track**
  nhưng S4 lệch **−16 track**. Không khái quát *"1-seed luôn thổi phồng"* thành quy luật.

📄 Số từng seed + phân tích đầy đủ 9 run: [`multi_seed_results.md`](baseline1_crnn_stn/multi_seed_results.md)

---

## 2. Bước 2 — CER, NED, PSNR/SSIM ✅

**CER + NED** — cài trong `Trainer.validate()`, in console mỗi epoch + 2 cột
`val_cer`/`val_ned` trong `history_*.csv` (nay đủ **14 cột**). Số ở bảng §1. J1 tốt
nhất cả exact match lẫn CER, và là model duy nhất đạt **0/999 sai độ dài ở cả 3 seed**.

**PSNR/SSIM** — tool mới `tools/eval_sr_quality.py` (`skimage.metrics`), chạy hậu kỳ
trên checkpoint. Con số đáng đọc là cột **Chênh** (so với `base` = ảnh chưa qua SR):

| Cấu hình | Chênh PSNR | Chênh SSIM |
|---|---:|---:|
| S1 (×2) | +1.0717 dB | +0.0698 |
| **S2** (×2, λ=0.5) | **+2.2879 dB** | **+0.1681** |
| S3 (×2, perceptual) | +1.0855 dB | +0.0664 |
| S4 (×1) | +1.0238 dB | +0.0618 |

> ⚠️ **Vì sao không có J1 ở bảng này**: PSNR/SSIM đo `I_SR` với `I_HR`, mà **J1
> không có nhánh SR nên `I_SR` không tồn tại** — giới hạn cấu trúc, không phải thiếu
> sót. Checkpoint J1 chỉ có `backbone/fusion/head/rnn/stn`;
> [`eval_sr_quality.py:131`](../tools/eval_sr_quality.py) chặn thẳng.

### 🔬 PSNR **nghịch** với khả năng đọc biển số — 3 mức bằng chứng

| Mức | Bằng chứng | Chiều |
|---|---|---|
| Track (n=999) | track đọc **sai** có PSNR **cao hơn ~2 dB**; r ≈ −0.35…−0.41, nhất quán 4 cấu hình | **nghịch** |
| Cấu hình | S2 có PSNR tốt nhất (+2.29 dB, gấp đôi) nhưng OCR **kém nhất** (794/999) | **nghịch** |
| **Kiến trúc** | **J1 bỏ hẳn SR → không có PSNR → OCR tốt nhất** | **nghịch** |

→ Khi reviewer hỏi *"sao không tối ưu theo PSNR"*: **"PSNR nghịch với OCR, có bằng
chứng ở cả 3 mức"** — chứ không chỉ "PSNR không phản ánh OCR".

📄 Chi tiết: [`buoc2_metrics.md`](crnn_and_stn/report/buoc2_metrics.md))

---

## 3. Bước 3 — Trực quan hoá định tính ✅

Script mới `tools/visualize_paper_figures.py` — grid **4 cột**
`I_LR → I_SR → Attention heatmap → Prediction`, **5 success + 5 failure** mỗi cấu hình.
Mặc định `--pick extreme`: lấy case đúng **tự tin nhất** và case sai **mơ hồ nhất**
(hai đầu phân phối), thay vì mấy track đầu danh sách.

**Đã sinh đủ 4 cấu hình** — 11 file/cấu hình (10 ảnh track + 1 grid tổng):

| Cấu hình | Thư mục |
|---|---|
| S1 | [`results/mf_sr_ocr/s1_mf_sr_ocr/paper_figures/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/feature/run-multi-seeda-and-evulate/crnn_and_stn/results/mf_sr_ocr/s1_mf_sr_ocr/paper_figures) |
| S2 | [`results/mf_sr_ocr/s2_mf_sr_ocr_lam05/paper_figures/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/feature/run-multi-seeda-and-evulate/crnn_and_stn/results/mf_sr_ocr/s2_mf_sr_ocr_lam05/paper_figures) |
| S3 | [`results/mf_sr_ocr/s3_l_perceptual/paper_figures/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/feature/run-multi-seeda-and-evulate/crnn_and_stn/results/mf_sr_ocr/s3_l_perceptual/paper_figures) |
| S4 | [`results/mf_sr_ocr/s4_sr_scale1/paper_figures/`](https://github.com/StevenThesisProjects/LowResolutionPlate/tree/feature/run-multi-seeda-and-evulate/crnn_and_stn/results/mf_sr_ocr/s4_sr_scale1/paper_figures) |

### Track trùng nhau giữa các cấu hình — tiện chọn Figure 4

| Track | Xuất hiện | Dùng để minh hoạ |
|---|---|---|
| `track_22161` | **SAI ở cả 4** | **giới hạn thật của dữ liệu**, không phải điểm yếu của một model |
| `track_19095` | SAI ở S1, S3, S4 | case khó nhất quán |
| `track_21455` | ĐÚNG ở S1, S2, S3 | case dễ, đọc chắc chắn |
| `track_12478` · `track_17959` | SAI ở S2 và S3 | |
| `track_16997` · `track_17125` | ĐÚNG ở S3 và S4 | |

⚠️ **2 caveat khi chọn hình cuối:**

1. **Hình sinh từ checkpoint 1-seed** (`results/mf_sr_ocr/*`), trong khi số trong
   paper nay là **multi-seed** → hình và số đến từ 2 model khác nhau. Nên sinh lại từ
   `results/multi-seed/*/*_seed42_best.pth` (~20 phút CPU, 0 GPU) hoặc ghi rõ trong
   caption.
2. **J1 chưa có hình** — cột `I_SR` không tồn tại khi không có nhánh SR (cùng lý do
   với PSNR ở §2). Nếu paper dùng J1 làm cấu hình chính thì Figure 4 phải đổi sang
   grid **3 cột** (`I_LR → Attention → Prediction`).

> Danh sách "10 track tiêu biểu" **không tái lập chính xác khi đổi phần cứng** — thứ
> tự theo confidence lệch ở các track có confidence gần bằng nhau (khác biệt số thực
> GPU vs CPU). Nên chốt một bộ hình và giữ nguyên.

---

## 4. Thay đổi code (7 file, +748/−22)

| File | |
|---|---|
| `tools/eval_sr_quality.py` | **mới (+282)** — PSNR/SSIM bằng `skimage.metrics`, kiến trúc suy ngược từ `state_dict` |
| `tools/visualize_paper_figures.py` | **mới (+291)** — Bước 3: grid 4 cột, `--pick extreme` |
| `src/training/trainer.py` | +98 — CER/NED trong `validate()`, đo thời gian mỗi epoch, 5 cột CSV mới, `build_optimizer_param_groups()` |
| `src/data/dataset.py` · `transforms.py` | +86 — nhánh `sr_eval_mode`: augment **tất định** để PSNR/SSIM tái lập được |
| `train.py` · `configs/config.py` | +13 — cờ `--weight-decay`, `--wd-skip-bias-norm`; banner in seed + trạng thái deterministic |

> `--wd-skip-bias-norm` **mặc định TẮT** có chủ đích — bật mặc định sẽ khiến
> multi-seed khác S1–S4 ở **hai** biến cùng lúc, không biết số đổi do biến nào.

---

## 5. Verify

```bash
# Kiểm định claim chính: J1 (không SR) vs S1 (đề xuất)
python tools/aggregate_seeds.py \
  --acc 79.9800 80.8809 80.4805 --label "J1 (khong SR)" \
  --baseline-acc 79.7798 80.0801 79.9800 --baseline-label "S1 (de xuat)"

# CSV phải đủ 14 cột
head -1 results/multi-seed/s1_mf_sr_ocr/history_s1_seed42.csv
```

Đã tự kiểm: 9/9 run đúng chế độ deterministic + đúng seed · `nan_batches = 0` mọi
epoch · CSV đủ 14 cột · CER in console · **Val Acc khớp chính xác khi chấm lại**
`submission_*.txt` với `plate_text` gốc (không chỉ tin log).

---

## 6. Ba chỗ cố ý lệch review

1. **Không dùng `editdistance`** — `postprocess.py` đã có sẵn `edit_distance`
   (Levenshtein có cache). Kết quả tương đương, không thêm dependency.
2. **Không đo PSNR/SSIM trên Scenario-A** — val **không có track Scenario-A nào**
   (0/999), và cả 10.000 track Scenario-A **nằm trong tập TRAIN** → đo ở đó là đo
   trên dữ liệu đã học, **không hợp lệ**. Đã đo trên 999 track Scenario-B.
3. **PSNR/SSIM là tool hậu kỳ**, không nằm trong log val mỗi epoch (nhét vào training
   loop sẽ buộc multi-seed chạy lại). ⚠️ **CER/NED thì không lệch** — có đủ trong
   console + CSV đúng như review yêu cầu.

---

## 7. Limitations

1. Chênh J1↔S1 nằm **trong** biên nhiễu → kết luận đúng là *"SR không giúp"*,
   **không phải** *"bỏ SR thì tốt hơn"*.
2. **Chưa tách được từng thành phần trong cụm cờ nền** (STN pool vs domain-match vs
   decode vs EMA) — câu hỏi mở quan trọng nhất còn lại.
3. Ablation là **tích luỹ**, không tách 1 biến; `T` cũng nhảy 16→32 giữa J1 và S1.
4. **3 ablation chỉ 1 seed** (phụ lục, nhãn *unverified*): multi-frame vs single-frame
   (J2 77.18%), perceptual (S3 80.58%), `λ_SR=0.5` (S2 79.48%).
5. **Không đo được PSNR/SSIM cho J1** — bảng PSNR không phủ được model có OCR cao nhất.
6. **Chưa chạy test lần nào** — mọi số là validation Scenario-B; PSNR/SSIM đo trên
   cặp synthetic.
7. Thiếu `log_j1p_seed42.txt` (CSV/submission/checkpoint vẫn đủ); phút/epoch của J1
   hiện là **ước tính** từ tỷ lệ GFLOPs.
8. **Hình Bước 3 sinh từ checkpoint 1-seed**, chưa khớp với số multi-seed; **J1 chưa
   có hình** (không có cột `I_SR`). Xem §3.

---

## 8. Việc còn lại (không thuộc PR này)

- [ ] 🚨 Viết lại kết luận chính của paper — SR không chứng minh được đóng góp
- [ ] Sửa **4 lỗi công thức loss** (bỏ warp kép · L1 không phải Smooth L1 · thêm
      stop-gradient `sg[θ]` · vị trí `λ_Perceptual`) — chi tiết ở
      [`checklist_review.md`](checklist_review.md) mục Nhóm 1
- [ ] Benchmark compute cho J1 và S4 — cần cho claim *"S1 tốn 3.76× để đổi lấy −0.50 điểm"*
- [ ] Chọn hình cuối cho Figure 4 (cân nhắc sinh lại từ checkpoint multi-seed)
- [ ] Nhóm 3 — chống overfitting · Bước 4 — so sánh SOTA (PARSeq/SVTR)
