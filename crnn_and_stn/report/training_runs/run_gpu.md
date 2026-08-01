# Lệnh chạy GPU

Chỉ lệnh + kết quả. Phân tích tại [../baseline1_crnn_stn/](../baseline1_crnn_stn/).
Phạm vi hiện tại: **J1/J2/J3 (đã có) + S1-S4 (đã xong)**. Ablation/multi-seed/
benchmark không thuộc phạm vi này.

**S1 đã chạy — 79.78% (797/999 track), vượt J2 26 track.** Phân tích đầy đủ:
[s1_proposed_mf_sr_ocr.md](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md).

**S2 đã chạy — 79.48% (794/999 track), chênh S1 chỉ 3 track (trong biên nhiễu,
không cải thiện).** [s2_lam05_mf_sr_ocr.md](../baseline1_crnn_stn/s2_lam05_mf_sr_ocr.md).

**S3 đã chạy — 80.58% (805/999 track), điểm cao nhất đã đo, +8 track so với S1
(vẫn trong biên nhiễu, chưa kết luận chắc).** [s3_perceptual_mf_sr_ocr.md](../baseline1_crnn_stn/s3_perceptual_mf_sr_ocr.md).

**S4 đã chạy — 80.58% (805/999 track), hoà tuyệt đối với S3 (mô hình khác nhau,
185/999 dự đoán khác nhau).** SR scale=1 (không phóng to ảnh) + `T=32` qua
`width_downsample=4` vẫn đạt bằng S1 → nghiêng về giả thuyết `T=32` là yếu tố
chính, không phải bản thân việc SR phóng to ảnh.
[s4_sr_scale1_mf_sr_ocr.md](../baseline1_crnn_stn/s4_sr_scale1_mf_sr_ocr.md).

**Cả 4 cấu hình S1-S4 đã chạy xong.** Bảng so sánh đầy đủ + đánh giá:
[model_comparison_summary.md](../baseline1_crnn_stn/model_comparison_summary.md).

> Val 999 track → biên nhiễu **±1.3 điểm (±13 track)**. Chênh lệch nhỏ hơn = nhiễu.

---

## 0. CHẠY THEO THỨ TỰ NÀY (checklist cho đợt revision)

Kế hoạch đầy đủ + lý do: [../paper/paper_revision_plan.md](../paper/paper_revision_plan.md).

### ⬜ B0 — Cài thêm 1 gói (30 giây)

```bash
pip install scikit-image        # cho PSNR/SSIM; thiếu thì chỉ có PSNR
```

### ⬜ B1 — Lấy PSNR/SSIM + hình cho paper (vài phút, KHÔNG train)

Chạy trên checkpoint S1–S4 **đã có sẵn**. Làm trước vì rẻ và có kết quả ngay.
Lệnh đầy đủ cả 4 cấu hình: [mục 6b](#6b-psnrssim--figure-định-tính-review-bước-2b--3).

```bash
python tools/eval_sr_quality.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s4_sr_scale1/s4_sr_scale1_best.pth \
  --width-downsample 4 --lr-domain-match --output-csv results/sr_quality_s4.csv

python tools/visualize_paper_figures.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s4_sr_scale1/s4_sr_scale1_best.pth \
  --width-downsample 4 --output-dir results/paper_figures/s4
```

### ⬜ B2 — Smoke test 1 epoch (~4 phút) — ĐỪNG BỎ QUA

Code trainer vừa sửa (thêm CER/NED + thời gian vào CSV) **chưa chạy thật lần nào**.
Tốn 4 phút để chắc, thay vì phát hiện sai cột sau 16 giờ.

```bash
python train.py --preset stable --experiment-name smoke --epochs 1 \
  --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --width-downsample 4 \
  --decode constrained --use-ema --no-cudnn-benchmark \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_smoke.txt

head -2 results/history_smoke.csv
```

**Điều kiện đạt** — dòng header phải có đủ 14 cột, kết thúc bằng:
`...,nan_batches,val_cer,val_ned,train_time_s,val_time_s,epoch_time_s`

### ⬜ B3 — Multi-seed S4 (~16 h) — chạy cấu hình rẻ nhất trước

```bash
for SEED in 42 100 2026; do
  python train.py --preset stable --experiment-name s4_seed${SEED} --seed ${SEED} \
    --epochs 60 --batch-size 32 --grad-accum-steps 2 \
    --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
    --backbone-norm group --lr-domain-match --width-downsample 4 \
    --decode constrained --use-ema \
    --no-cudnn-benchmark --num-workers 8 --aug-level full \
    2>&1 | tee results/log_s4_seed${SEED}.txt
done
```

### ⬜ B4 — Multi-seed S1 (~33 h) và S3 (~36–45 h)

Lệnh đầy đủ: [mục 5](#5-s--proposed-method-joint-end-to-end-mf-sr-ocr). Nếu ngân sách
GPU eo hẹp thì bỏ S3, giữ S1 + S4.

### ⬜ B5 — Tổng hợp Mean ± Std

```bash
python tools/aggregate_seeds.py --from-logs results/log_s4_seed*.txt --label "S4"
python tools/aggregate_seeds.py --from-logs results/log_s1_seed*.txt --label "S1"
```

### ⬜ B6 — Chốt cấu hình → chạy test

**Chỉ sau khi B5 xong.** Chưa từng chạy test lần nào — xem
[paper_revision_plan.md §2b](../paper/paper_revision_plan.md) trước khi chạy, vì
`--submission-mode` **train lại từ đầu và tắt early stopping**, không phải chỉ inference.

> ⚠️ Mọi lệnh train đều phải có `2>&1 | tee results/log_*.txt`. S2/S3 mất vĩnh viễn
> số liệu thời gian chỉ vì quên `tee`.

---

## 1. Setup

```bash
unzip crnn_and_stn.zip -d crnn_and_stn && cd crnn_and_stn
nvidia-smi                                    # xem CUDA Version trước khi chọn index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install albumentations opencv-python tqdm numpy matplotlib scikit-image
apt update && apt install -y libgl1
```

Image `pytorch/pytorch`: bỏ dòng cài torch. Kiểm tra:

```bash
python -c "import torch,torchvision; from torchvision.ops import DeformConv2d; print(torch.__version__, torch.cuda.get_device_name(0))"
```

`CUDA out of memory` → thêm tiền tố `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 2. Kết quả đã có (mốc tham chiếu)

Val = 999 track Scenario-B. ±13 track = ±1.3 điểm là biên nhiễu.

| Run    | Cấu hình                                                     | Track đúng |    Val Acc |
| ------ | ------------------------------------------------------------ | ---------: | ---------: |
| —      | CRNN + STN (report ICPR gốc)                                 |          — |     77.00% |
| —      | CRNN + STN (đo lại)                                          |        757 |     75.78% |
| —      | aug light                                                    |        739 |     73.97% |
| SR-v1  | stacked-input SR (bản lỗi)                                   |        492 |     49.25% |
| SR-v2  | stacked-input SR, lr thấp + aug light                        |        550 |     55.06% |
| —      | ResBlock backbone `norm=none`                                |        766 |     76.68% |
| J1     | + GroupNorm, không SR                                        |    **768** |     76.88% |
| **J2** | **+ SR per-frame ×2 có giám sát**                            |    **771** | **77.18%** |
| J3     | + DCNv2 (kernel init ngẫu nhiên — đã sửa)                    |        762 |     76.28% |
| **S1** | **Joint MF-SR-OCR (đề xuất, mục 5), λ_SR=0.1**               |    **797** | **79.78%** |
| S2     | Joint MF-SR-OCR, λ_SR=0.5 (mục 5)                            |        794 |     79.48% |
| **S3** | **Joint MF-SR-OCR, + L_Perceptual α=0.1 (mục 5)**            |    **805** | **80.58%** |
| **S4** | **Joint MF-SR-OCR, SR scale=1 + width_downsample=4 (mục 5)** |    **805** | **80.58%** |

J2 hơn J1 3 track, J3 kém J2 9 track — cả hai trong biên nhiễu ±13. **S1 hơn J2
tới 26 track — gấp đôi biên nhiễu**, lần đầu tiên một cấu hình vượt qua ngưỡng đó
kể từ baseline gốc 77.00%. S2 (tăng `λ_SR` lên 0.5) kém S1 3 track — trong biên
nhiễu, không cải thiện. **S3 (+L_Perceptual) và S4 (SR scale=1, không phóng to
ảnh) đều hơn S1 8 track và hoà tuyệt đối với nhau (805/999)** — từng cặp vẫn
trong biên nhiễu ±13 nên chưa "chắc chắn" theo tiêu chuẩn thống kê, nhưng 2 hướng
độc lập cùng hội tụ về 1 điểm là tín hiệu đồng thuận đáng chú ý. S4 còn cho bằng
chứng gián tiếp rằng `T=32` (không phải bản thân việc SR phóng to ảnh) mới là yếu
tố chính đứng sau lợi ích đo được ở S1 — xem mục 5 và
[s4_sr_scale1_mf_sr_ocr.md §3](../baseline1_crnn_stn/s4_sr_scale1_mf_sr_ocr.md#3-trả-lời-câu-hỏi-t-confound--kết-quả-chính-của-s4).
Chi tiết + giới hạn cần nêu: [s1_proposed_mf_sr_ocr.md](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md),
[s2_lam05_mf_sr_ocr.md](../baseline1_crnn_stn/s2_lam05_mf_sr_ocr.md),
[s3_perceptual_mf_sr_ocr.md](../baseline1_crnn_stn/s3_perceptual_mf_sr_ocr.md),
[s4_sr_scale1_mf_sr_ocr.md](../baseline1_crnn_stn/s4_sr_scale1_mf_sr_ocr.md).

**Chi phí compute** (`tools/benchmark.py --all`, CPU). Hai bảng khác nhau vì
kiến trúc mặc định đã đổi (STN pool `(1,1)` → `(4,8)`) giữa lúc J1/J2/J3 và lúc
S1 được benchmark — mỗi bảng phản ánh đúng kiến trúc thật đã chạy tại thời điểm đó:

Lúc J1/J2/J3 (STN pool `(1,1)` cũ):

| Cấu hình            |     Params | GFLOPs/track | Latency (ms) | vs base |
| ------------------- | ---------: | -----------: | -----------: | ------: |
| ResBlock (không SR) | 29,298,220 |        25.90 |        53.35 |   1.00x |
| + GroupNorm         | 29,313,452 |        25.90 |        52.84 |   0.99x |
| + SR per-frame      | 29,426,895 |       107.99 |       195.04 |   3.66x |
| + SR + DCNv2        | 29,445,854 |       108.75 |       201.60 |   3.78x |

Đo lại với kiến trúc hiện tại (STN pool `(4,8)` mặc định — khớp `--all` mới):

| Cấu hình              |         Params | GFLOPs/track | Latency (ms) |   vs base |
| --------------------- | -------------: | -----------: | -----------: | --------: |
| ResBlock (không SR)   |     29,427,468 |        26.14 |        50.03 |     1.00x |
| + GroupNorm           |     29,442,700 |        26.14 |        51.30 |     1.03x |
| + SR per-frame        |     29,558,255 |       108.31 |       184.43 |     3.69x |
| **+ SR + DCNv2 = S1** | **29,577,214** |   **109.08** |   **192.75** | **3.85x** |

Dòng cuối khớp chính xác `Model params: 29,577,214` trong `log_s1.txt` — xác
nhận đây đúng là kiến trúc S1. Ở J2 (kiến trúc cũ), SR đổi 3.66x compute lấy 3
track (trong biên nhiễu). **Ở S1, 3.85x compute đổi lấy 26 track** — ngoài biên
nhiễu, tỷ lệ đổi chác tốt hơn hẳn J2.

---

## 3. Sanity check (3 epoch, trước khi chạy S1-S4 đầy đủ)

Không dùng `--preset debug` (batch 8 + LR 1e-3 → CTC kẹt blank plateau).

```bash
python train.py --preset stable --experiment-name sanity --epochs 3 \
  --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --decode constrained \
  --num-workers 8 --aug-level full
```

Mốc J2: ep1 `0.00%` · ep2 `31.83%` · ep3 `49.15%`. Đạt nếu ep2 > 25% và `sr_loss` giảm.

---

## 4. P0 — Chấm lại checkpoint cũ (không tốn GPU, chạy trước tiên)

20.000 nhãn dài 7 ký tự, chỉ 2 layout (`LLLNLNN` 13k, `LLLNNNN` 7k) → 6/7 vị trí
khoá cứng lớp chữ/số. Greedy hiện tại không dùng ràng buộc này.

```bash
python tools/eval_decode.py --checkpoint results/crnn_resblock_sr_supervised_best.pth
```

---

## 5. S — Proposed Method: Joint End-to-End MF-SR-OCR

```
5 LR frames → [1] STN per-frame → [2] DCNv2 align → [3] MFSR ×2 (+L_SR)
            → [4] backbone + Attention Fusion → [5] BiLSTM + CTC
L_Total = L_CTC + λ_SR·L_SR       L_SR = ‖I_SR − I_HR‖₁ + α·L_Perceptual
```

Khác J2: `--use-dcn` (Step 2, kernel identity-init) · MFSR thay single-frame SR
(Step 3) · `--lr-domain-match` (Nguyên nhân #4) · STN pool `4,8` mặc định thay
`1,1` (Mục A) · `--decode constrained` · `--use-ema`.

```bash
## S1 — λ_SR = 0.1 — ĐÃ CHẠY: 79.78% (797/999), early-stopped epoch 55, đỉnh
##      epoch 37. Chi tiết: ../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md
python train.py \
  --preset stable --experiment-name s1_proposed \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s1.txt

## S2 — λ_SR = 0.5 (đầu kia khoảng review đề xuất) — ĐÃ CHẠY: 79.48% (794/999),
##      early-stopped epoch 56, đỉnh epoch 38 — chênh S1 chỉ 3 track (biên nhiễu),
##      không cải thiện. Chi tiết: ../baseline1_crnn_stn/s2_lam05_mf_sr_ocr.md
##      (run này không dùng `tee`, không có log_s2.txt)
python train.py \
  --preset stable --experiment-name s2_lam05 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.5 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s2.txt

## S3 — + L_Perceptual (số hạng α=0.1) — ĐÃ CHẠY: 80.58% (805/999), điểm cao
##      nhất đã đo, early-stopped epoch 43, đỉnh epoch 25 (sớm hơn S1/S2 rõ rệt).
##      +8 track so với S1 — vẫn trong biên nhiễu ±13, chưa kết luận chắc chắn.
##      Chi tiết: ../baseline1_crnn_stn/s3_perceptual_mf_sr_ocr.md
##      (run này không dùng `tee`, không có log_s3.txt)
python train.py \
  --preset stable --experiment-name s3_perceptual \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 --sr-perceptual-weight 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full

## S4 — SR ×1: HR gốc ~115x42px, target ×2 (256px) có 55% là nội suy bicubic
##      thuần. ×1 giữ output 32x128 ≈ đúng độ phân giải HR thật — ĐÃ CHẠY: 80.58%
##      (805/999), hoà tuyệt đối với S3, +8 track so với S1, early-stopped epoch
##      58, đỉnh epoch 40. Vì T=32 vẫn giữ được (qua width_downsample=4) mà điểm
##      không giảm dù SR không phóng to ảnh, bằng chứng nghiêng về T=32 là yếu tố
##      chính, không phải SR có phóng to ảnh hay không.
##      Chi tiết: ../baseline1_crnn_stn/s4_sr_scale1_mf_sr_ocr.md
python train.py \
  --preset stable --experiment-name s4_sr_scale1 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --width-downsample 4 \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s4.txt
```

**Đọc `results/history_*.csv` khi đang chạy:**

| Cột                                            | Ý nghĩa                                                                            |
| ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| `val_acc` vs `val_acc_greedy`                  | Khoảng cách = giá trị thật của constrained decode                                  |
| `sr_loss` vs `sr_loss_bilinear`                | `sr_loss ≥ sr_loss_bilinear` kéo dài = SR học không hơn nội suy, tốn 3.66x compute |
| `nan_batches`                                  | Phải luôn = 0                                                                      |
| `val_cer`                                      | Character Error Rate mức corpus (tổng edit distance / tổng ký tự nhãn). Thấp = tốt |
| `val_ned`                                      | Normalized Edit Distance trung bình mỗi track. Thấp = tốt (paper hay ghi `1−NED`)  |
| `train_time_s` / `val_time_s` / `epoch_time_s` | Thời gian thực mỗi epoch (giây)                                                    |

> 5 cột cuối được **thêm từ 2026-08-01** (phục vụ Bước 2 của review + đo thời gian).
> CSV của S1–S4 chạy trước đó không có các cột này — `csv.DictReader` trả `None`,
> không crash, nhưng đừng vẽ chart CER cho S1–S4 cũ vì không có dữ liệu.

---

## 6. Chart cho báo cáo

Có sẵn 3 chart (`tools/plot_results.py`) + ảnh định tính (`tools/visualize.py`).
Chưa có heatmap. Cần `pip install matplotlib`.

```bash
## Val Exact Match + Loss theo epoch, so J1 vs S1
python tools/plot_results.py curves \
  --history results/history_crnn_resblock_groupnorm_nosr.csv \
            results/history_s1_proposed.csv \
  --labels "J1 GroupNorm (không SR)" "S1 Proposed MF-SR-OCR" \
  --baseline 77.00 --output-dir results/charts
# -> results/charts/training_curves.png

## So sánh J1/J2/J3/S1 (dot plot)
python tools/plot_results.py ablation \
  --names "ResBlock" "+GroupNorm (J1)" "+SR (J2)" "+SR+DCN (J3)" "S1 Proposed" \
  --acc 76.68 76.88 77.18 76.28 79.78 \
  --baseline 77.00 --output-dir results/charts
# -> results/charts/ablation_comparison.png

## Accuracy vs chi phí (số ở bảng "Chi phí compute" mục 2 — mỗi điểm dùng đúng
## cost đo tại kiến trúc đã chạy: J1/J2/J3 = bảng cũ, S1 = bảng mới đã xác nhận
## khớp log_s1.txt)
python tools/plot_results.py cost \
  --names "ResBlock" "+GroupNorm (J1)" "+SR (J2)" "+SR+DCN (J3)" "S1 Proposed" \
  --acc 76.68 76.88 77.18 76.28 79.78 \
  --latency 53.35 52.84 195.04 201.60 192.75 \
  --params 29298220 29313452 29426895 29445854 29577214 \
  --baseline 77.00 --output-dir results/charts
# -> results/charts/accuracy_vs_cost.png

## Ảnh định tính: LR frames -> SR output -> prediction.
## Attention = độ dày viền khung (dày hơn = frame được tin cậy hơn).
python tools/visualize.py --checkpoint results/s1_proposed_best.pth \
  --use-sr --use-dcn --backbone-norm group --decode constrained \
  --num-samples 12 --output-dir results/viz

python tools/visualize.py --checkpoint results/s1_proposed_best.pth \
  --use-sr --use-dcn --backbone-norm group --decode constrained \
  --only-errors --num-samples 20 --output-dir results/viz_errors
```

`curves` cũng đọc log stdout qua `--from-log results/log_s1.txt` khi chưa có CSV.

---

## 6b. PSNR/SSIM + figure định tính (review Bước 2b & 3)

> **Chạy được NGAY trên checkpoint S1–S4 đã có — không cần chờ multi-seed, không tốn
> giờ train.** Cả hai tool đều chạy hậu kỳ trên `.pth`, chỉ forward validation.
> Kiến trúc được suy ngược từ `state_dict` nên không phải nhớ lại flag lúc train —
> trừ `--width-downsample` và `--lr-domain-match` (không nằm trong state_dict).

### PSNR/SSIM của nhánh SR

```bash
# S1 (sr_scale=2, width/8)
python tools/eval_sr_quality.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s1_mf_sr_ocr/mf_sr_ocr.pth \
  --lr-domain-match --output-csv results/sr_quality_s1.csv

# S2 (sr_scale=2, width/8)
python tools/eval_sr_quality.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s2_mf_sr_ocr_lam05/s2_lam05_best.pth \
  --lr-domain-match --output-csv results/sr_quality_s2.csv

# S3 (sr_scale=2, width/8, có perceptual)
python tools/eval_sr_quality.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s3_l_perceptual/s3_perceptual_best.pth \
  --lr-domain-match --output-csv results/sr_quality_s3.csv

# S4 (sr_scale=1 => BẮT BUỘC --width-downsample 4)
python tools/eval_sr_quality.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s4_sr_scale1/s4_sr_scale1_best.pth \
  --width-downsample 4 --lr-domain-match --output-csv results/sr_quality_s4.csv
```

Xuất ra: PSNR/SSIM của SR **và** của mốc `base` (không học), kèm chênh lệch — con số
đáng báo cáo là **chênh lệch**, không phải PSNR tuyệt đối.

> ⚠️ **Không đặt PSNR của S4 chung cột với S1/S2/S3.** S1–S3 xuất ảnh 64×256, S4 xuất
> 32×128 — hai thang khác nhau, so trực tiếp là sai. Với S4, mốc `base` là ảnh giữ
> nguyên (không nội suy) nên đọc là "SR có hơn việc không làm gì".
>
> Val vốn không có cặp (ảnh vào, HR) — tool bật `sr_eval_mode` để sinh cặp synthetic
> khớp pixel, tắt augment ngẫu nhiên (tái lập được), giữ degradation (giống phân phối
> lúc train), và warp HR theo `theta` đúng như `Trainer._sr_loss`. Số đo là trên
> **cặp synthetic**, cần ghi rõ trong paper.

### Figure định tính 4 cột (I_LR → I_SR → Attention → Prediction)

```bash
python tools/visualize_paper_figures.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s4_sr_scale1/s4_sr_scale1_best.pth \
  --width-downsample 4 --output-dir results/paper_figures/s4

# Đổi checkpoint để so cùng một track qua các cấu hình:
python tools/visualize_paper_figures.py \
  --checkpoint report/csv-report-process/mf_sr_ocr/s1_mf_sr_ocr/mf_sr_ocr.pth \
  --output-dir results/paper_figures/s1
```

Xuất ra `figure4_qualitative_grid.png` (5 case đúng + 5 case sai) và ảnh từng track
riêng. Mặc định `--pick extreme`: lấy case đúng **tự tin nhất** và case sai **mơ hồ
nhất** — hai đầu phân phối, thay vì mấy track đầu danh sách. Dùng `--pick first` nếu
muốn tuần tự.

> Với S4 lưu ý: cấu hình này có **0/999 track confidence < 0.55**, nên "case sai mơ hồ
> nhất" của S4 vẫn có confidence khá cao — không giống S1/S3. Đây là đặc điểm của model,
> không phải lỗi tool.

---

## 7. Submission

Chỉ chạy sau khi đã chốt cấu hình ở mục 5.

```bash
python train.py \
  --preset stable --experiment-name submission_final \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --submission-mode --num-workers 8 --aug-level full
```

`--submission-mode` bỏ validation split → không còn accuracy trong lúc train.
