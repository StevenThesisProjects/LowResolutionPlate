# Lệnh chạy GPU — baseline CRNN+STN · J1 · S1 · S4

> Chỉ lệnh + kết quả. Phân tích tại [../baseline1_crnn_stn/](../baseline1_crnn_stn/).
>
> **Phạm vi**: baseline CRNN+STN và 3 cấu hình có error bar (**J1, S1, S4**).
> Lệnh của các ablation 1-seed đã chạy trước đây nằm ở `backup/report/`.
>
> 🎯 **Phương pháp đề xuất: S1** — `79.95% ± 0.15`.

## ✅ Kết quả multi-seed (3 seed deterministic, 42/100/2026)

| Model | SR | `T` | **Mean ± Std** | Std | Phút/epoch | Tổng 3 seed |
|---|---|---:|---:|---:|---:|---:|
| **S1** 🎯 | ×2 MFSR + DCN | 32 | **79.95% ± 0.15** | **0.15** | 9.67 | 22.24 h |
| **J1** | ❌ không SR | 16 | 80.45% ± 0.45 | 0.45 | **2.60** | **5.55 h** |
| **S4** | ×1 MFSR + DCN | 32 | 79.48% ± 0.44 | 0.44 | 4.28 | 11.27 h |

> 🚨 **Cấu hình không SR (J1) có điểm trung bình cao nhất** — hoà S1 (+0.50, trong
> nhiễu) và **hơn S4 có ý nghĩa** (+0.97 > 2×0.36), trong khi rẻ hơn **3.76× compute**.
> Nhánh SR chưa chứng minh được đóng góp. Phân tích:
> [multi_seed_results.md](../baseline1_crnn_stn/multi_seed_results.md).

> Val = **999 track** Scenario-B → biên nhiễu **±13 track (±1.3 điểm)**.
> Chênh lệch nhỏ hơn ngưỡng này = nhiễu.

---

## 0. Chuẩn bị (bắt buộc)

```bash
mkdir -p results          # `results/` không được commit -> server clone về KHÔNG có
                          # thư mục này; `tee` chạy TRƯỚC python nên sẽ fail và MẤT LOG
pip install scikit-image  # cho PSNR/SSIM
```

### Setup máy mới

```bash
unzip crnn_and_stn.zip -d crnn_and_stn && cd crnn_and_stn
nvidia-smi                                    # xem CUDA Version trước khi chọn index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install albumentations opencv-python tqdm numpy matplotlib scikit-image
apt update && apt install -y libgl1

python -c "import torch,torchvision; from torchvision.ops import DeformConv2d; print(torch.__version__, torch.cuda.get_device_name(0))"
```

`CUDA out of memory` → thêm tiền tố `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 1. Smoke test 1 epoch (xác nhận deterministic + CSV đủ cột)

```bash
python train.py --preset stable --experiment-name smoke --epochs 1 \
  --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --width-downsample 4 \
  --decode constrained --use-ema --no-cudnn-benchmark \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_smoke.txt

head -1 results/history_smoke.csv
```

**Điều kiện đạt** — header CSV phải đủ **14 cột**, kết thúc bằng:
`...,nan_batches,val_cer,val_ned,train_time_s,val_time_s,epoch_time_s`

---

## 2. Multi-seed — 3 model (ĐÃ CHẠY XONG)

### 2a. S1 — phương pháp đề xuất → **79.95% ± 0.15**

```bash
for SEED in 42 100 2026; do
  python train.py --preset stable --experiment-name s1_seed${SEED} --seed ${SEED} \
    --epochs 60 --batch-size 32 --grad-accum-steps 2 \
    --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
    --backbone-norm group --lr-domain-match \
    --decode constrained --use-ema \
    --no-cudnn-benchmark --num-workers 8 --aug-level full \
    2>&1 | tee results/log_s1_seed${SEED}.txt
done
```

> ⚠️ **S1 KHÔNG có `--width-downsample`** (dùng mặc định 8) và **có `--sr-scale 2`** —
> 2 chỗ khác S4. Gõ nhầm là dựng sai kiến trúc, chạy 22 giờ ra kết quả vô nghĩa.
>
> Params `29,577,214` · `T = (128×2)/8 = 32` · Dữ liệu: `results/multi-seed/s1_mf_sr_ocr/`

### 2b. J1 — ablation "bỏ hẳn SR" → **80.45% ± 0.45**

```bash
for SEED in 42 100 2026; do
  python train.py --preset stable --experiment-name j1p_seed${SEED} --seed ${SEED} \
    --epochs 60 --batch-size 32 --grad-accum-steps 2 \
    --backbone-norm group --lr-domain-match \
    --decode constrained --use-ema \
    --no-cudnn-benchmark --num-workers 8 --aug-level full \
    2>&1 | tee results/log_j1p_seed${SEED}.txt
done
```

> ⚠️ Dùng **đúng bộ cờ nền của S1/S4, chỉ bỏ `--use-sr` và `--use-dcn`** → ablation
> 1-cụm-biến sạch. **Không** phải cờ lịch sử `--stn-pool 1,1`, nên số **không** so
> được với J1 lịch sử 76.88%.
>
> Không SR → width không nhân đôi → **`T = 128/8 = 16`**.
> Params `29,442,700` · Dữ liệu: `results/multi-seed/crnn_resblock_groupnorm_nosr_j1/`
>
> ⚠️ **Thiếu `log_j1p_seed42.txt`** — CSV/submission/checkpoint của seed 42 vẫn đủ.

### 2c. S4 — ablation "bỏ phóng to ảnh" → **79.48% ± 0.44**

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

> ⚠️ `--sr-scale 1` **và** `--width-downsample 4` phải đi cùng nhau — thiếu cái thứ hai
> thì `T` tụt về 16, làm S4 khác S1 tới 2 biến chồng nhau.
>
> Params `29,549,470` · `T = (128×1)/4 = 32` · Dữ liệu: `results/multi-seed/s4_sr_scale1/`

> ⚠️ Mọi lệnh train **phải có** `2>&1 | tee results/log_*.txt` — đã từng mất vĩnh viễn
> số liệu thời gian của vài run chỉ vì quên `tee`.

---

## 3. Tổng hợp Mean ± Std + kiểm định

```bash
# Mean ± Std từng model, gom cả 3 vào 1 file CSV
rm -f results/multi_seed_summary.csv
for C in j1p s1 s4; do
  python tools/aggregate_seeds.py \
    --from-logs "results/multi-seed/*/log_${C}_seed*.txt" --label "${C}" \
    --output-csv results/multi_seed_summary.csv --append
done

# Kiểm định S1 (đề xuất) ↔ J1 (bỏ hẳn SR) — phép so quan trọng nhất
python tools/aggregate_seeds.py \
  --acc 79.7798 80.0801 79.9800 --label "S1 (de xuat)" \
  --baseline-acc 79.9800 80.8809 80.4805 --baseline-label "J1 (khong SR)"

# Kiểm định S1 ↔ S4
python tools/aggregate_seeds.py \
  --acc 79.7798 80.0801 79.9800 --label "S1 (SR x2)" \
  --baseline-acc 78.9790 79.7798 79.6797 --baseline-label "S4 (SR x1)"
```

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| J1 vs S1 | +0.50 | ±0.28 | ⚠️ trong nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

---

## 4. PSNR/SSIM + hình định tính (KHÔNG train)

> Chạy ở đâu cũng được — inference thuần trên `.pth`. GPU ~5–10 phút/cấu hình,
> CPU ~20 phút. Kiến trúc suy ngược từ `state_dict`, trừ `--width-downsample` và
> `--lr-domain-match` (không nằm trong state_dict).
>
> ⚠️ `*.pth` **không đi theo `git pull`** (112–113 MB/file). Gặp
> `❌ Không tìm thấy checkpoint` là do thiếu file, không phải lỗi lệnh.

```bash
# PSNR/SSIM — dùng --num-workers 0 để tái lập tuyệt đối
# (mặc định 4 gây dao động ±0.05 dB vì mỗi worker có trạng thái random riêng)
python tools/eval_sr_quality.py --lr-domain-match --num-workers 0 \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --output-csv results/multi-seed/s1_mf_sr_ocr/sr_quality_s1_seed42.csv

# S4: sr_scale=1 => BẮT BUỘC --width-downsample 4
python tools/eval_sr_quality.py --lr-domain-match --num-workers 0 --width-downsample 4 \
  --checkpoint results/multi-seed/s4_sr_scale1/s4_seed42_best.pth \
  --output-csv results/multi-seed/s4_sr_scale1/sr_quality_s4_seed42.csv
```

> 🚨 **Không chạy được cho J1** — J1 không có nhánh SR nên `I_SR` không tồn tại;
> `eval_sr_quality.py:131` chặn thẳng. Giới hạn cấu trúc, không phải thiếu sót.
>
> ⚠️ **Không đặt PSNR tuyệt đối của S4 chung cột với S1** — S1 xuất ảnh 64×256, S4 xuất
> 32×128, hai thang khác nhau. Chỉ so được cột "Chênh" (so với `base` của chính nó).

```bash
# Figure 4 — grid 4 cột (I_LR -> I_SR -> Attention -> Prediction), 5 đúng + 5 sai
python tools/visualize_paper_figures.py --decode constrained --pick extreme \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --output-dir results/multi-seed/s1_mf_sr_ocr/paper_figures
```

> ⚠️ **zsh không tách từ khi expand biến** (khác bash). Đừng gom cờ vào biến kiểu
> `$extra` rồi truyền vào — `--width-downsample 4` sẽ thành **một** tham số và argparse
> báo `unrecognized arguments`. Viết thẳng cờ.
>
> ⚠️ Danh sách "10 track tiêu biểu" **không tái lập chính xác khi đổi phần cứng** — thứ
> tự theo confidence lệch ở các track gần bằng nhau. **Chốt một bộ hình rồi giữ nguyên.**

---

## 5. Benchmark compute

```bash
python tools/benchmark.py --backbone-norm group --cpu
```

> ⚠️ **Chưa benchmark được S4** — `tools/benchmark.py` thiếu cờ `--sr-scale` và
> `--width-downsample` nên không dựng được kiến trúc `sr_scale=1 + width/4`.
>
> 📌 **Phút/epoch KHÔNG cần chạy lại** — đã có số đo thật từ cột `epoch_time_s` của
> toàn bộ 424 epoch (bảng đầu tài liệu).

Đo với kiến trúc hiện tại (STN pool `(4,8)`):

| Cấu hình | Params | GFLOPs/track | Latency (ms) | vs base |
|---|---:|---:|---:|---:|
| ResBlock (không SR) | 29,427,468 | 26.14 | 50.03 | 1.00× |
| **+ GroupNorm = J1** | **29,442,700** | **26.14** | **51.30** | 1.03× |
| + SR per-frame | 29,558,255 | 108.31 | 184.43 | 3.69× |
| **+ SR + DCNv2 = S1** | **29,577,214** | **109.08** | **192.75** | **3.85×** |

Dòng cuối khớp chính xác `Model params: 29,577,214` in trong log S1 — xác nhận đúng
kiến trúc.

---

## 6. Mốc tham chiếu baseline

| Run | Cấu hình | Track đúng | Val Acc | Seed |
|---|---|---:|---:|:---:|
| — | CRNN + STN (report ICPR gốc) | — | **77.00%** | — |
| — | CRNN + STN (đo lại trên dataset project) | 757 | 75.78% | 1 |
| — | ResBlock backbone `norm=none` | 766 | 76.68% | 1 |
| — | + GroupNorm (J1 **lịch sử**, STN pool `(1,1)`) | 768 | 76.88% | 1 |
| **J1** | multi-seed, cụm cờ nền đầy đủ | — | **80.45% ± 0.45** | **3** |
| **S1** 🎯 | Joint MF-SR-OCR ×2 | — | **79.95% ± 0.15** | **3** |
| **S4** | Joint MF-SR-OCR ×1 | — | **79.48% ± 0.44** | **3** |

> ⚠️ 4 dòng đầu là **1 seed, `cudnn.benchmark=True`** — đọc như mốc lịch sử, không so
> trực tiếp với 3 dòng cuối. Chênh **+3.57 điểm** giữa J1 lịch sử (76.88%) và J1
> multi-seed (80.45%) là đóng góp của **cụm cờ nền**, không phải SR.

---

## 7. Đọc `results/history_*.csv`

| Cột | Ý nghĩa |
|---|---|
| `val_acc` vs `val_acc_greedy` | Khoảng cách = giá trị thật của constrained decode |
| `sr_loss` vs `sr_loss_bilinear` | `sr_loss ≥ sr_loss_bilinear` kéo dài = SR học không hơn nội suy |
| `nan_batches` | Phải luôn = 0 |
| `val_cer` | Character Error Rate mức corpus. Thấp = tốt |
| `val_ned` | Normalized Edit Distance trung bình mỗi track. Thấp = tốt (paper hay ghi `1−NED`) |
| `train_time_s` / `val_time_s` / `epoch_time_s` | Thời gian thực mỗi epoch (giây) |

---

## 8. Chart

```bash
python tools/plot_results.py curves \
  --history results/multi-seed/s1_mf_sr_ocr/history_s1_seed42.csv \
            results/multi-seed/crnn_resblock_groupnorm_nosr_j1/history_j1p_seed42.csv \
            results/multi-seed/s4_sr_scale1/history_s4_seed42.csv \
  --labels "S1 (de xuat)" "J1 (khong SR)" "S4 (SR x1)" \
  --baseline 77.00 --output-dir results/charts

python tools/plot_results.py ablation \
  --names "Baseline CRNN+STN" "J1 (khong SR)" "S1 (de xuat)" "S4 (SR x1)" \
  --acc 77.00 80.45 79.95 79.48 \
  --baseline 77.00 --output-dir results/charts
```

> 📌 Nên vẽ **Mean ± Std** thay vì 1 điểm. Biểu đồ dạng text tương đương đã có ở
> [model_comparison_summary.md](../baseline1_crnn_stn/model_comparison_summary.md).

---

## 9. Submission / test — **CHƯA CHẠY LẦN NÀO**

Mọi số ở trên đều là **validation**. Chỉ chạy test sau khi đã chốt cấu hình (**S1**).

> 🚨 **`--submission-mode` KHÔNG phải chỉ là inference — nó train lại từ đầu** trên
> toàn bộ 20.000 track (`full_train=True`), **bỏ validation** → `Trainer.fit()` **tắt
> early stopping** và lưu checkpoint theo _train loss_ (thứ gần như luôn giảm).
> Chạy đủ 60 epoch ở chế độ này sẽ **lưu ra model của epoch cuối, đã overfit nặng**
> (S1 đỉnh val ở epoch 24–32).
>
> → Nếu dùng, **bắt buộc đặt `--epochs` bằng epoch tốt nhất học được từ validation**
> (S1: trung bình **28** qua 3 seed 28/32/24 — tính từ
> `results/multi-seed/s1_mf_sr_ocr/history_s1_seed*.csv`).

```bash
python train.py \
  --preset stable --experiment-name submission_s1_final \
  --epochs 28 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --submission-mode --num-workers 8 --aug-level full \
  2>&1 | tee results/log_submission_s1.txt
```

> ⚠️ Model ra từ lệnh này **khác** 3 checkpoint `s1_seed*_best.pth` đã có — train mới
> trên 20.000 track (thêm 5%, gồm cả val), không phải checkpoint cũ được inference lại.

**Phương án an toàn hơn (khuyến nghị)**: dùng lại checkpoint đã được validate, chỉ chạy
inference trên test — số val và số test đến từ **cùng một model**. Hiện chưa có tool
(`tools/eval_decode.py` chỉ chạy `mode="val"`), cần viết thêm ~1 giờ.
Phân tích đánh đổi: [../paper/paper_revision_plan.md §3](../paper/paper_revision_plan.md).

Test public 1.000 track, test blind 3.000 track — file ra là
`submission_<tên>_final.txt`, khác `submission_<tên>.txt` (dự đoán validation).
