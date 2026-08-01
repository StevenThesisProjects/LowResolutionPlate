# Checklist công việc theo review

> Bám đúng cấu trúc review: 3 nhóm giải pháp + 4 bước.
> Dữ liệu S1–S4: `results/mf_sr_ocr/<cấu hình>/` · Lệnh chạy:
> [training_runs/run_gpu.md §0](training_runs/run_gpu.md).
> Cập nhật: 2026-08-01.

## Tình trạng nhanh

| Mục review                       | Trạng thái                                         |
| -------------------------------- | -------------------------------------------------- |
| Nhóm 1 — công thức loss          | ⚠️ đã tìm ra 3 lỗi, **chưa sửa vào paper**         |
| Nhóm 2 — quy trình deterministic | ✅ **xong**                                        |
| Nhóm 3 — chống overfitting       | ❌ **chưa làm mục nào** (cố ý hoãn tới sau Bước 1) |
| **Bước 1** — multi-seed          | 🔄 đang chạy S4; S1 và S3 chưa (S2 đã bỏ)          |
| **Bước 2** — CER/NED/PSNR/SSIM   | ✅ **xong**, đủ 4 cấu hình                         |
| **Bước 3** — hình định tính      | ✅ **xong**, đủ 4 cấu hình (còn chọn hình cuối)    |
| Bước 4 — PARSeq/SVTR             | ❌ để cuối, làm riêng                              |

**Đường găng duy nhất là Bước 1** — đã chốt chạy **S1 + S3 + S4** (~85–94h), bỏ S2.
Lý do: [mục tối ưu chi phí](#-tối-ưu-chi-phí-multi-seed--cắt-được-cái-nào).

---

## 💰 Tối ưu chi phí multi-seed — cắt được cái nào?

Review yêu cầu multi-seed cả S1/S2/S3/S4 = **~118–127h GPU**. Không phải cấu hình nào
cũng đáng tiền như nhau.

**Dữ liệu 1-seed đã có đủ cho cả 4** (không thiếu gì để phân tích):

|     | history | submission | .pth | CER/NED | PSNR/SSIM | hình | log |
| --- | :-----: | :--------: | :--: | :-----: | :-------: | :--: | :-: |
| S1  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ✅  |
| S2  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ❌  |
| S3  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ❌  |
| S4  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ✅  |

### ✅ ĐÃ CHỐT: multi-seed **S1 + S3 + S4**, bỏ S2 (~85–94h, tiết kiệm ~33h)

|        | Multi-seed? | Chi phí | Lý do                                                                                                                                                                                                                               |
| ------ | :---------: | ------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **S1** |     ✅      |    ~33h | Phương pháp đề xuất, con số headline của paper                                                                                                                                                                                      |
| **S3** |     ✅      | ~36–45h | Giữ được `λ_Perceptual` trong công thức đúng như review viết; đang có **CER tốt nhất** (0.0522)                                                                                                                                     |
| **S4** |     ✅      |    ~16h | Claim _"bằng S1 nhưng rẻ hơn 2.34×"_ đang dựa trên 8 track — **trong biên nhiễu ±13**, mong manh nhất                                                                                                                               |
| **S2** |  ⏭️ **bỏ**  |    ~33h | Kết quả **âm tính** (794, kém nhất nhóm S). Kết luận "tăng λ_SR không giúp" đã có bằng chứng độc lập mạnh hơn: tương quan PSNR↔đọc-đúng âm (r ≈ −0.35 đến −0.41, **n=999, nhất quán 4 cấu hình**) — không phụ thuộc thứ hạng của S2 |

**Vì sao bộ này mạch lạc**: cả 3 đều là ablation **1 biến sạch** tách từ S1, mỗi cái
trả lời một câu hỏi _thiết kế_ riêng:

| So sánh      | Câu hỏi trả lời                                                             |
| ------------ | --------------------------------------------------------------------------- |
| S3 vs S1     | _Perceptual loss có giúp không?_ — câu hỏi về hàm mục tiêu                  |
| S4 vs S1     | _Có cần phóng to ảnh không, hay chỉ cần `T=32`?_ — câu hỏi về cơ chế        |
| ~~S2 vs S1~~ | _λ=0.5 có hơn 0.1 không?_ — chỉ dò siêu tham số, giá trị khoa học thấp nhất |

**Nguyên tắc cắt S2**: error bar chỉ cần cho claim **khẳng định điều gì đó tốt hơn**.
Kết quả âm tính không cần — báo cáo 1 seed kèm nhãn _"1 seed, chưa xác nhận"_ trong
bảng ablation là chuẩn mực bình thường.

### Hệ quả với paper

✅ **Công thức giữ nguyên** như review viết, không phải sửa:

```
L_Total = L_CTC + λ_SR·L_SR + λ_Perceptual·L_VGG      (λ_Perceptual = 0.01)
```

vì S3 — cấu hình bật perceptual — có trong bộ multi-seed.

⚠️ S2 phải được ghi rõ **"1 seed, chưa xác nhận"** ở mọi chỗ xuất hiện trong bảng.

### Thứ tự chạy

**S4 (~16h, đang chạy) → S1 (~33h) → S3 (~36–45h)** — rẻ trước, lỗi quy trình lộ sớm
với ít GPU nhất.

---

## Nhóm 1 — Hoàn thiện công thức loss (việc phía paper, 0 GPU)

- [x] Xác nhận `λ_Perceptual = λ_SR × α` → S3 (`--sr-perceptual-weight 0.1`) = 0.01
- [x] **Chốt phạm vi multi-seed = S1+S3+S4** → S3 có trong bộ nên **giữ nguyên
      `λ_Perceptual` trong công thức**, không phải sửa
- [ ] **Sửa `L_SR`: bỏ `Warp_θ` khỏi `I_SR`.** Code chỉ warp HR, vì `I_SR` đã nằm
      trong khung STN đã nắn — warp hai lần là sai
- [ ] **Đổi "Smooth L1" → "L1"** — code dùng `F.l1_loss`

## Nhóm 2 — Chuẩn hoá quy trình deterministic

- [x] `seed_everything` với `deterministic=True, benchmark=False` — **đã có sẵn** trong
      `src/utils/common.py`, bật bằng `--no-cudnn-benchmark`
- [x] `tools/aggregate_seeds.py` (Mean ± Std + CI 95% + kiểm định) — **đã có sẵn**
- [x] Smoke test 1 epoch xác nhận deterministic ăn + CSV đủ 14 cột

## Nhóm 3 — Chống overfitting — **làm SAU multi-seed**

- [x] Cờ `--weight-decay` và `--wd-skip-bias-norm` (param-grouping)
- [ ] `Dropout(0.3)` trước BiLSTM — **chưa code**
- [ ] Thêm `MotionBlur`, `GaussNoise` vào train transform — **chưa code**
      (hiện chỉ có trong degradation của ảnh synthetic; `RandomBrightnessContrast` và
      `ShiftScaleRotate`≈`Affine` thì đã có)
- [ ] `patience` 18 → 12
- [ ] Chạy thử + multi-seed riêng nếu đưa vào paper

> ⚠️ Đổi model rồi thì phải multi-seed lại từ đầu → **không trộn vào Bước 1**.
> Bật `--wd-skip-bias-norm` **cùng lúc** với `--weight-decay 1e-3`, không bật riêng.

---

## 🔴 Bước 1 — Multi-Seed Runs (đường găng)

Đã chốt phạm vi **S1 + S3 + S4**, bỏ S2 — xem [lý do](#-đã-chốt-multi-seed-s1--s3--s4-bỏ-s2-8594h-tiết-kiệm-33h).

- [x] `mkdir -p results` trên server (thiếu thì `tee` fail, mất log)
- [ ] **S4 × 3 seed** (42/100/2026, ~16h) — 🔄 **đang chạy**
- [ ] **S1 × 3 seed** (~33h)
- [ ] **S3 × 3 seed** (~36–45h) — cờ khác S1: thêm `--sr-perceptual-weight 0.1`
- [ ] ~~S2 × 3 seed~~ — **đã quyết định bỏ**, giữ kết quả 1 seed kèm nhãn "chưa xác nhận"
- [ ] `aggregate_seeds.py` → bảng Mean ± Std cho 3 cấu hình
- [ ] Cập nhật `model_comparison_summary.md` + kết luận paper

> ⚠️ Con số headline (79.78% / 80.58%) **có thể đổi** vì lần này chạy deterministic,
> còn S1–S4 cũ chạy `cudnn.benchmark=True`. Chuẩn bị chuyển sang báo cáo Mean ± Std
> thay vì best-of-run.

## 🔴 Bước 2 — CER, NED, PSNR/SSIM ✅ **XONG**

- [x] CER + NED trong `validate()` → in console mỗi epoch + 2 cột `val_cer`, `val_ned`
      trong `history_*.csv`
- [x] Số CER/NED cho cả S1–S4 (tính lại từ `submission_*.txt`, không cần train lại)
- [x] `tools/eval_sr_quality.py` — PSNR/SSIM bằng `skimage.metrics`
- [x] PSNR/SSIM cho cả 4 cấu hình (999 track mỗi cấu hình), lưu tại
      `results/mf_sr_ocr/<cấu hình>/sr_quality_*.csv`
- [x] Tương quan PSNR ↔ đọc đúng ở mức từng track (n=999)
- [x] Báo cáo: [buoc2_metrics.md](buoc2_metrics.md)

**Kết quả mạnh nhất để đưa vào paper — PSNR _nghịch_ với khả năng đọc:**

| Cấu hình | PSNR track đọc **đúng** | PSNR track đọc **sai** | r(PSNR, đúng) |
| -------- | ----------------------: | ---------------------: | ------------: |
| S1       |                  16.288 |                 18.240 |        −0.362 |
| S2       |                  18.102 |                 20.154 |        −0.346 |
| S3       |                  16.492 |                 18.894 |        −0.412 |
| S4       |                  17.080 |                 19.371 |        −0.400 |

Track đọc **sai** lại có PSNR **cao hơn ~2 dB**, nhất quán ở cả 4 cấu hình. Cùng chiều
với mức cấu hình: S2 tối ưu PSNR mạnh nhất (+2.29 dB) nhưng OCR kém nhất (794/999).

→ Khi reviewer hỏi _"sao không tối ưu theo PSNR"_, câu trả lời không còn là "PSNR không
phản ánh OCR" mà là **"PSNR nghịch với OCR, có bằng chứng trên 999 track ở 4 cấu hình
độc lập"**.

**3 chỗ lệch so với nguyên văn review** (phải nêu trong paper):

1. Không dùng thư viện `editdistance` — `postprocess.py` đã có sẵn `edit_distance`
2. **Không đo trên Scenario-A**: val không có track Scenario-A nào, và cả 10.000 track
   Scenario-A đều nằm trong tập TRAIN → đo ở đó là đo trên dữ liệu đã học, không hợp lệ
3. PSNR/SSIM là tool hậu kỳ, không nằm trong log validation mỗi epoch (nhét vào training
   loop sẽ phải khởi động lại multi-seed đang chạy)

## 🟠 Bước 3 — Hình định tính

- [x] `tools/visualize_paper_figures.py` — grid 4 cột, 5 đúng + 5 sai
- [x] Verify chạy được (bản 2 track, tự tái lập đúng 805/999)
- [x] Chạy full 10 track cho **cả 4 cấu hình** (S1/S2/S3/S4) → mỗi cấu hình có
      `paper_figures/figure4_qualitative_grid.png` + 10 ảnh track riêng
- [ ] Chọn hình cuối cho Figure 4

Chạy cả 4 (không chỉ 3 cấu hình vào paper) vì rẻ, và có đủ thì so được **cùng một
track qua các cấu hình** — hữu ích khi muốn minh hoạ vì sao S2 tái tạo ảnh đẹp hơn
nhưng lại đọc sai.

## 🟠 Bước 4 — PARSeq / SVTR — **LÀM RIÊNG, CUỐI CÙNG**

- [ ] Dựng repo + môi trường riêng (không đụng `crnn_and_stn`)
- [ ] Chốt cách so công bằng: PARSeq là single-image, ta là multi-frame → đề xuất cho
      PARSeq ăn **frame giữa**, ghi rõ là baseline single-frame
- [ ] Xuất danh sách 999 track val + nhãn để chấm trên **đúng cùng một tập**
- [ ] Chạy + đưa vào bảng so sánh

> Cắt được nếu thiếu thời gian — chỉ cần ghi vào phần Limitations.

---

## Sau Bước 1 — chạy test

- [ ] Chốt cấu hình cuối theo Mean ± Std, **trước khi** nhìn bất kỳ số test nào
- [ ] Inference test public — ⚠️ `--submission-mode` **train lại từ đầu và tắt early
      stopping**, không phải chỉ inference; xem
      [paper/paper_revision_plan.md §2b](paper/paper_revision_plan.md)
- [ ] Test blind cuối cùng

---

## Tình trạng

| Hạng mục                | Xong       | Đang chạy              | Còn lại              |
| ----------------------- | ---------- | ---------------------- | -------------------- |
| Nhóm 1 — công thức      | 1/4        |                        | 3 việc phía paper    |
| Nhóm 2 — deterministic  | 3/3 ✅     |                        |                      |
| Nhóm 3 — chống overfit  | 1/5        |                        | sau Bước 1           |
| **Bước 1 — multi-seed** | 1/7        | S4 × 3 seed (GPU thuê) | S1, S2 (+S3?)        |
| **Bước 2 — metrics**    | **7/7** ✅ |                        | —                    |
| Bước 3 — hình           | 2/6        |                        | chạy full 3 cấu hình |
| Bước 4 — SOTA           | 0/4        |                        | cuối cùng            |

### Việc kế tiếp theo thứ tự

1. **Chờ S4 × 3 seed xong** → gửi `tail -3 results/log_s4_seed42.txt`. Đây là lần đầu
   biết chế độ deterministic có làm đổi con số 80.58% hay không.
2. **Chốt 3 hay 4 cấu hình** (quyết định ở đầu file) → định đoạt công thức trong paper.
3. Chạy S1, S2 (+S3 nếu chọn 4).
4. Chọn hình cuối cho Figure 4 từ `results/mf_sr_ocr/*/paper_figures/` (đã sinh xong,
   không cần chạy lại).
