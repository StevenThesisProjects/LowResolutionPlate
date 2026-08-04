# Lệnh chạy trên GPU thuê — phần việc còn lại

> **Chốt phạm vi (2026-08-04)**: Bước 1 và Bước 2 **đã xong**. Nhóm 3 (chống overfit)
> **đã chốt không áp dụng** → code liên quan đã revert, repo về đúng trạng thái lúc
> chạy 9 run multi-seed.
>
> Tài liệu này chỉ chứa **phần việc còn lại cần máy GPU**. Lệnh của 9 run multi-seed
> đã chạy nằm ở [run_gpu.md §0 B3/B4](run_gpu.md).

---

## 0. Kết quả kiểm tra Bước 1 & Bước 2 (verify trực tiếp trên đĩa)

### 🔴 Bước 1 — Multi-Seed cho S1, S4, J1

| Yêu cầu | Trạng thái |
|---|---|
| 3 seed × 3 model = 9 run | ✅ **9/9** |
| `history_*.csv` per-epoch | ✅ **9/9**, đủ 14 cột |
| `submission_*.txt` từng track | ✅ **9/9** |
| Checkpoint `*_best.pth` | ✅ **9/9** |
| Log stdout | ⚠️ **8/9** — thiếu `log_j1p_seed42.txt` |
| Mean ± Std + kiểm định cặp | ✅ có trong [multi_seed_results.md](../baseline1_crnn_stn/multi_seed_results.md) |
| Bảng tổng hợp xuất ra **file CSV** | ✅ **đã thêm `--output-csv`** (2026-08-04) — xem [§5b](#5b--output-csv-cho-aggregate_seedspy--đã-làm-2026-08-04) |

→ **Bước 1 XONG.** Chỗ hở duy nhất còn lại là `log_j1p_seed42.txt` — chỉ mất **banner
cấu hình** của riêng seed đó, còn CSV/submission/checkpoint vẫn đủ để chấm lại, nên
**không chặn paper**.

### 🔴 Bước 2 — CER, NED, PSNR/SSIM

| Yêu cầu | Trạng thái |
|---|---|
| CER + NED, in console mỗi epoch | ✅ **9/9 run** |
| CER + NED, ghi vào CSV | ✅ **9/9 CSV** có cột `val_cer`, `val_ned` |
| CER Mean ± Std cho cả 3 model | ✅ |
| PSNR/SSIM (`skimage.metrics`) | ✅ có tool + số cho **S1, S2, S3, S4** |
| PSNR/SSIM trên **checkpoint multi-seed** | ❌ **chỉ có trên checkpoint 1-seed** |
| PSNR/SSIM cho **J1** | ❌ **không thể** — J1 không có nhánh SR nên `I_SR` không tồn tại |
| Đo trên **Scenario-A** | ⚠️ **cố ý lệch** — val có 0/999 track Scenario-A, và cả 10.000 track Scenario-A nằm trong tập TRAIN → đo ở đó là đo trên dữ liệu đã học |

→ **CER/NED: XONG hoàn toàn.** **PSNR/SSIM: xong nhưng lệch nguồn** — đo trên
checkpoint 1-seed, trong khi bảng accuracy là multi-seed. Đây là việc **đáng chạy lại**
và là nội dung chính của tài liệu này.

---

## 1. Chuẩn bị máy (bắt buộc, ~5 phút)

```bash
# Trong thư mục crnn_and_stn
mkdir -p results                # `results/` bị gitignore -> clone về KHÔNG có
pip install scikit-image        # cho PSNR/SSIM
nvidia-smi                      # xác nhận GPU nhận được
python -c "import torch; from torchvision.ops import DeformConv2d; print(torch.__version__, torch.cuda.get_device_name(0))"
```

> 🚨 **Checkpoint KHÔNG đi theo `git pull`** — `*.pth` bị gitignore (112–113 MB/file).
> Phải **upload thủ công** trước khi chạy. Với toàn bộ việc dưới đây chỉ cần **2 file**:
>
> ```
> results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth     (113 MB)
> results/multi-seed/s4_sr_scale1/s4_seed42_best.pth     (113 MB)
> ```
>
> Gặp `❌ Không tìm thấy checkpoint` là do thiếu file, **không phải lỗi lệnh**.

---

## 2. 🔴 Việc chính — PSNR/SSIM trên checkpoint MULTI-SEED

**Vì sao chạy lại**: bảng PSNR hiện tại đo trên checkpoint 1-seed
(`results/mf_sr_ocr/`), còn bảng accuracy trong paper là multi-seed. Hai bảng đến từ
**hai model khác nhau** — reviewer sẽ hỏi. Chạy lại trên seed 42 của multi-seed là
khớp nguồn.

⏱️ **~5–10 phút/cấu hình trên GPU** (CPU thì ~20 phút). Đây là **inference thuần**.

```bash
mkdir -p results/multi-seed/s1_mf_sr_ocr results/multi-seed/s4_sr_scale1

# S1 (sr_scale=2, width/8 -> KHÔNG truyền --width-downsample)
python tools/eval_sr_quality.py \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --lr-domain-match --num-workers 0 \
  --output-csv results/multi-seed/s1_mf_sr_ocr/sr_quality_s1_seed42.csv \
  2>&1 | tee results/log_psnr_s1_seed42.txt

# S4 (sr_scale=1 -> BẮT BUỘC --width-downsample 4)
python tools/eval_sr_quality.py \
  --checkpoint results/multi-seed/s4_sr_scale1/s4_seed42_best.pth \
  --width-downsample 4 --lr-domain-match --num-workers 0 \
  --output-csv results/multi-seed/s4_sr_scale1/sr_quality_s4_seed42.csv \
  2>&1 | tee results/log_psnr_s4_seed42.txt
```

> ⚠️ **`--num-workers 0`** (khác lệnh cũ dùng mặc định 4). Lý do: pipeline degradation
> ngẫu nhiên, mỗi worker DataLoader có trạng thái random riêng mà albumentations không
> nhận seed → số dao động **±0.05 dB** giữa các lần chạy. `0` thì **tái lập tuyệt đối**,
> đổi lại chậm hơn. Đáng, vì đây là số đưa vào paper.
>
> ⚠️ **Không chạy cho J1** — J1 không có nhánh SR nên không có `I_SR` để so với `I_HR`;
> [`eval_sr_quality.py:131`](../../tools/eval_sr_quality.py) chặn thẳng. Đây là **giới
> hạn cấu trúc**, không phải thiếu sót.
>
> ⚠️ **Không đặt PSNR tuyệt đối của S4 chung cột với S1** — S1 xuất ảnh 64×256, S4 xuất
> 32×128, hai thang khác nhau. Chỉ so được cột **"Chênh"** (so với `base` của chính nó).

**Sau khi chạy xong**: chép số vào [../buoc2_metrics.md](../buoc2_metrics.md) §2
(thư mục `results/` bị gitignore nên chỉ `.md` mới được commit).

---

## 3. 🟠 Figure 4 — sinh lại từ checkpoint multi-seed

**Vì sao chạy lại**: cùng lý do §2 — hình hiện có sinh từ checkpoint 1-seed
(02/08 00:39), trước cả khi có checkpoint multi-seed. Đã chốt **S1 là cấu hình chính**
nên chỉ cần sinh lại cho S1.

⏱️ **~5 phút trên GPU.**

```bash
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --decode constrained --pick extreme \
  --output-dir results/multi-seed/s1_mf_sr_ocr/paper_figures \
  2>&1 | tee results/log_fig_s1_seed42.txt
```

Xuất ra `figure4_qualitative_grid.png` (grid **4 cột**
`I_LR → I_SR → Attention → Prediction`, 5 case đúng + 5 case sai) + 10 ảnh track riêng.

> 📌 Giữ grid **4 cột** được vì cấu hình chính là S1 (có `I_SR`). Nếu sau này muốn thêm
> hình J1 cho phụ lục thì phải đổi sang grid **3 cột**.
>
> ⚠️ Danh sách "10 track tiêu biểu" **không tái lập chính xác khi đổi phần cứng** — thứ
> tự theo confidence lệch ở các track có confidence gần bằng nhau (số thực GPU vs CPU).
> **Chốt một bộ hình rồi giữ nguyên**, đừng sinh đi sinh lại.

**Tuỳ chọn** — sinh thêm cho S4 để so cùng một track qua 2 kiến trúc:

```bash
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/s4_sr_scale1/s4_seed42_best.pth \
  --width-downsample 4 --decode constrained --pick extreme \
  --output-dir results/multi-seed/s4_sr_scale1/paper_figures
```

---

## 4. 🟡 Benchmark compute cho J1 (cho claim "S1 tốn 3.76×")

Chạy được **ngay, trên CPU** — không cần GPU, không cần checkpoint.

```bash
python tools/benchmark.py --backbone-norm group --cpu
```

> ⚠️ **`tools/benchmark.py` HIỆN KHÔNG benchmark được S4** — thiếu cờ `--sr-scale` và
> `--width-downsample`, nên không dựng được kiến trúc `sr_scale=1 + width/4`. Muốn có
> số cho S4 thì phải **thêm 2 cờ đó vào tool** (sửa ~10 dòng, 0 GPU, không đụng gì tới
> model hay kết quả). Chưa làm vì chưa được yêu cầu.
>
> 📌 **Phút/epoch thì KHÔNG cần chạy lại** — đã có số đo thật từ cột `epoch_time_s`
> của toàn bộ 424 epoch: **J1 2.60 · S1 9.67 · S4 4.28** phút/epoch.

---

## 5. ⬜ Tuỳ chọn — 2 việc có thể bỏ qua

### 5a. Khôi phục `log_j1p_seed42.txt` (~1.6 h GPU)

Chỉ mất **banner cấu hình** của riêng seed đó; CSV/submission/checkpoint vẫn đủ.
Chạy lại sẽ vừa khôi phục log, **vừa kiểm chứng tính tất định** — nếu deterministic
hoạt động đúng thì phải ra **đúng 799/999 (79.98%)**, khớp tuyệt đối.

```bash
python train.py --preset stable --experiment-name j1p_seed42_rerun --seed 42 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --no-cudnn-benchmark --num-workers 8 --aug-level full \
  2>&1 | tee results/log_j1p_seed42_rerun.txt
```

> 🎯 **Đây là phép kiểm chứng đáng giá hơn cả việc khôi phục log**: nếu ra đúng
> 799/999 thì có bằng chứng trực tiếp rằng quy trình deterministic của Nhóm 2 thật sự
> tái lập được — mạnh hơn nhiều so với chỉ trích dẫn banner. Nếu **lệch**, đó là phát
> hiện quan trọng phải điều tra trước khi nộp bài.

### 5b. ✅ `--output-csv` cho `aggregate_seeds.py` — **ĐÃ LÀM (2026-08-04)**

Đóng nốt mục *"Xác minh: ghi nhận bảng kết quả 3 seeds vào CSV log"* của Bước 1.
Chạy **0 GPU, ~10 giây** — làm ở đâu cũng được, kể cả máy local.

```bash
# Gom cả 3 model vào 1 file bằng --append
rm -f results/multi_seed_summary.csv
python tools/aggregate_seeds.py --acc 79.7798 80.0801 79.9800 --label "S1 (de xuat)" \
  --baseline-acc 79.9800 80.8809 80.4805 --baseline-label "J1 (khong SR)" \
  --output-csv results/multi_seed_summary.csv
python tools/aggregate_seeds.py --acc 78.9790 79.7798 79.6797 --label "S4 (SR x1)" \
  --output-csv results/multi_seed_summary.csv --append
```

Hoặc đọc thẳng từ log, không cần gõ tay số nào:

```bash
rm -f results/multi_seed_summary.csv
for C in j1p s1 s4; do
  python tools/aggregate_seeds.py \
    --from-logs "results/multi-seed/*/log_${C}_seed*.txt" --label "${C}" \
    --output-csv results/multi_seed_summary.csv --append
done
```

**13 cột**: `label, n_seeds, accs, mean, std, min, max, ci95, ci_low, ci_high,
delta_vs_baseline, delta_stderr, verdict`. Cột `verdict` nhận
`significant` / `within_noise` / `single_seed` — kết quả kiểm định gắn vào dòng của
cấu hình đang xét, dòng baseline để trống.

Kết quả kiểm chứng với số thật (khớp chính xác bảng trong
[multi_seed_results.md](../baseline1_crnn_stn/multi_seed_results.md)):

```
label,n_seeds,accs,mean,std,...,delta_vs_baseline,delta_stderr,verdict
S1 (de xuat),3,79.7798 80.0801 79.9800,79.9466,0.1529,...,-0.5005,0.2751,within_noise
J1 (khong SR),3,79.9800 80.8809 80.4805,80.4471,0.4514,...,,,
S4 (SR x1),3,78.9790 79.7798 79.6797,79.4795,0.4363,...,,,
```

> ⚠️ `results/` bị **gitignore** → CSV này **không được commit**. Vẫn phải chép số vào
> `.md` trong `report/` như trước; file CSV chỉ để thoả đúng chữ "ghi nhận vào CSV log"
> của review và để nộp kèm nếu ban tổ chức yêu cầu.
> Muốn commit được thì ghi thẳng vào `report/`:
> `--output-csv report/baseline1_crnn_stn/multi_seed_summary.csv`

---

## 6. ❌ KHÔNG chạy trong đợt này

| | Lý do |
|---|---|
| **Nhóm 3 — chống overfitting** | Đã chốt không áp dụng; code đã revert. Áp dụng = thí nghiệm mới ~34h + phải multi-seed lại cả 3 model. Xem [../checklist_review.md §Nhóm 3](../checklist_review.md) |
| **Bước 4 — PARSeq/SVTR** | Cần repo + môi trường riêng. Review cho phép cắt nếu thiếu thời gian |
| **Chạy test / submission** | ⚠️ `--submission-mode` **train lại từ đầu và TẮT early stopping**, không phải chỉ inference. Phải chốt `--epochs` bằng epoch tốt nhất (S1: 24–32) trước khi chạy, **đừng để 60**. Đọc [../paper/paper_revision_plan.md §2b](../paper/paper_revision_plan.md) trước |

---

## 7. Tóm tắt thứ tự chạy + thời gian

| # | Việc | Thời gian | Bắt buộc? |
|---|---|---:|:---:|
| 1 | Chuẩn bị máy + upload 2 checkpoint | ~10 ph | ✅ |
| 2 | PSNR/SSIM cho S1 + S4 (multi-seed) | ~20 ph | ✅ |
| 3 | Figure 4 từ S1 multi-seed | ~5 ph | ✅ |
| 4 | Benchmark J1 (CPU) | ~5 ph | 🟡 |
| 5a | Rerun J1 seed 42 (kiểm chứng tất định) | ~1.6 h | ⬜ |
| 5b | Xuất `multi_seed_summary.csv` (0 GPU) | ~10 giây | ✅ tool đã sẵn |

**Tổng phần bắt buộc: ~35 phút.** Nếu làm cả 5a thì ~2.2 giờ.

> 💾 **Nhớ tải kết quả về trước khi trả máy** — `results/` bị gitignore, không có gì
> tự động lưu:
> ```bash
> tar czf results_remaining.tar.gz \
>   results/multi-seed/*/sr_quality_*.csv \
>   results/multi-seed/*/paper_figures \
>   results/log_*.txt
> ```
