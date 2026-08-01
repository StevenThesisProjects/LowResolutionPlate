# Lệnh chạy GPU

Chỉ lệnh + kết quả. Phân tích tại [../baseline1_crnn_stn/](../baseline1_crnn_stn/) —
bảng so sánh tổng hợp tất cả cấu hình:
[model_comparison_summary.md](../baseline1_crnn_stn/model_comparison_summary.md).
Phạm vi hiện tại: **J1/J2/J3 (đã có) + S1-S4**. Ablation/multi-seed/benchmark
không thuộc phạm vi này.

**S1 đã chạy — 79.78% (797/999 track), vượt J2 26 track.** Phân tích đầy đủ:
[s1_proposed_mf_sr_ocr.md](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md).

**S2 đã chạy — 79.48% (794/999 track), chênh S1 chỉ 3 track (trong biên nhiễu,
không cải thiện).** [s2_lam05_mf_sr_ocr.md](../baseline1_crnn_stn/s2_lam05_mf_sr_ocr.md).

**S3 đã chạy — 80.58% (805/999 track), điểm cao nhất đã đo, +8 track so với S1
(vẫn trong biên nhiễu, chưa kết luận chắc).** [s3_perceptual_mf_sr_ocr.md](../baseline1_crnn_stn/s3_perceptual_mf_sr_ocr.md).

> Val 999 track → biên nhiễu **±1.3 điểm (±13 track)**. Chênh lệch nhỏ hơn = nhiễu.

---

## 1. Setup

```bash
unzip crnn_and_stn.zip -d crnn_and_stn && cd crnn_and_stn
nvidia-smi                                    # xem CUDA Version trước khi chọn index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install albumentations opencv-python tqdm numpy matplotlib
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

| Run    | Cấu hình                                  | Track đúng |    Val Acc |
| ------ | ----------------------------------------- | ---------: | ---------: |
| —      | CRNN + STN (report ICPR gốc)              |          — |     77.00% |
| —      | CRNN + STN (đo lại)                       |        757 |     75.78% |
| —      | aug light                                 |        739 |     73.97% |
| SR-v1  | stacked-input SR (bản lỗi)                |        492 |     49.25% |
| SR-v2  | stacked-input SR, lr thấp + aug light     |        550 |     55.06% |
| —      | ResBlock backbone `norm=none`             |        766 |     76.68% |
| J1     | + GroupNorm, không SR                     |    **768** |     76.88% |
| **J2** | **+ SR per-frame ×2 có giám sát**         |    **771** | **77.18%** |
| J3     | + DCNv2 (kernel init ngẫu nhiên — đã sửa) |        762 |     76.28% |
| **S1** | **Joint MF-SR-OCR (đề xuất, mục 5), λ_SR=0.1** |    **797** | **79.78%** |
| S2     | Joint MF-SR-OCR, λ_SR=0.5 (mục 5)         |        794 |     79.48% |
| **S3** | **Joint MF-SR-OCR, + L_Perceptual α=0.1 (mục 5)** | **805** | **80.58%** |

J2 hơn J1 3 track, J3 kém J2 9 track — cả hai trong biên nhiễu ±13. **S1 hơn J2
tới 26 track — gấp đôi biên nhiễu**, lần đầu tiên một cấu hình vượt qua ngưỡng đó
kể từ baseline gốc 77.00%. S2 (tăng `λ_SR` lên 0.5) kém S1 3 track — trong biên
nhiễu, không cải thiện. **S3 (+L_Perceptual) hơn S1 8 track — điểm cao nhất đã
đo, nhưng vẫn trong biên nhiễu so với S1** nên chưa kết luận chắc chắn tốt hơn.
Chi tiết + giới hạn cần nêu: [s1_proposed_mf_sr_ocr.md](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md),
[s2_lam05_mf_sr_ocr.md](../baseline1_crnn_stn/s2_lam05_mf_sr_ocr.md),
[s3_perceptual_mf_sr_ocr.md](../baseline1_crnn_stn/s3_perceptual_mf_sr_ocr.md).

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

| Cấu hình            |     Params | GFLOPs/track | Latency (ms) | vs base |
| ------------------- | ---------: | -----------: | -----------: | ------: |
| ResBlock (không SR) | 29,427,468 |        26.14 |        50.03 |   1.00x |
| + GroupNorm         | 29,442,700 |        26.14 |        51.30 |   1.03x |
| + SR per-frame      | 29,558,255 |       108.31 |       184.43 |   3.69x |
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

## S4 — chưa chạy. SR ×1: HR gốc ~115x42px, target ×2 (256px) có 55% là nội suy
##      bicubic thuần. ×1 giữ output 32x128 ≈ đúng độ phân giải HR thật. Dù S1
##      đã cho bằng chứng SR học vượt bilinear (mục 4 của báo cáo S1), S4 vẫn
##      cần chạy để biết SR có thể tốt hơn nữa ở đúng thang thông tin thật không.
python train.py \
  --preset stable --experiment-name s4_sr_scale1 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --width-downsample 4 \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s4.txt
```

**Đọc `results/history_*.csv` khi đang chạy:**

| Cột                             | Ý nghĩa                                                                            |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| `val_acc` vs `val_acc_greedy`   | Khoảng cách = giá trị thật của constrained decode                                  |
| `sr_loss` vs `sr_loss_bilinear` | `sr_loss ≥ sr_loss_bilinear` kéo dài = SR học không hơn nội suy, tốn 3.66x compute |
| `nan_batches`                   | Phải luôn = 0                                                                      |

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
