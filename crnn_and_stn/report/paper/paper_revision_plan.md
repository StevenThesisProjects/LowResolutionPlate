# Kế hoạch còn lại cho paper

> **Phạm vi**: baseline CRNN+STN · J1 · S1 · S4. Nhật ký lập kế hoạch cũ (chọn ablation
> nào để multi-seed, dự toán GPU cho các cấu hình đã loại) đã hoàn thành nhiệm vụ và
> được lưu ở `backup/report/`.
>
> 🎯 **Phương pháp đề xuất: S1** — `79.95% ± 0.15`.
> Bối cảnh: [model_comparison_summary.md](../baseline1_crnn_stn/model_comparison_summary.md).

## 1. Trạng thái các bước của review

| Bước | Trạng thái | GPU còn cần |
|---|---|---:|
| **Bước 1** — Multi-seed J1/S1/S4 | ✅ **XONG** — 9/9 run, có Mean ± Std | 0 |
| **Bước 2** — CER/NED/PSNR/SSIM | ✅ **XONG** (3 chỗ cố ý lệch, có lý do) | 0 |
| **Bước 3** — Hình định tính | 🟡 script xong, còn chốt hình cho Figure 4 | 0 |
| **Bước 4** — PARSeq/SVTR | ❌ chưa bắt đầu, cắt được nếu thiếu thời gian | riêng |
| Nhóm 1 — công thức loss | 🟡 đã soạn bản sửa, còn dán vào bản thảo | 0 |
| Nhóm 2 — deterministic | ✅ **XONG**, verify trên 9/9 run | 0 |
| Nhóm 3 — chống overfitting | ⏭️ **chốt không áp dụng** → Future work | 0 |

## 2. Việc phía paper — tất cả 0 GPU

1. ✍️ **Dán khối công thức đã sửa** vào bản thảo (4 lỗi + 3 câu chú thích bắt buộc) —
   [loss_formula_corrected.md](loss_formula_corrected.md).
2. ✍️ **Đổi con số headline sang Mean ± Std**: `79.95% ± 0.15` (S1, 3 seed). Bỏ hẳn
   lối báo cáo best-of-run.
3. 🚨 **Viết lại phần "đóng góp của nhánh SR"** — không đổi phương pháp chính, mà đổi
   **lời giải thích vì sao nó hoạt động**: phần cải thiện đo được đến từ **cụm cờ nền**
   (STN pool `(4,8)` + `--lr-domain-match` + constrained decode + EMA), còn SR chưa
   chứng minh được đóng góp.
4. ✍️ **Cập nhật Limitations** — danh sách đầy đủ ở mục 4.
5. 🖼️ **Chốt 10 hình cho Figure 4** — dùng S1, giữ grid 4 cột.
6. 📏 **Benchmark GFLOPs/latency cho S4** — `tools/benchmark.py` thiếu cờ `--sr-scale`
   và `--width-downsample`. Phút/epoch thì đã có số đo thật, không cần chạy lại.

## 3. Tập test — CHƯA từng chạy

**Toàn bộ số hiện có đều là validation.** Phải chốt cấu hình xong **rồi mới** chạy test.

### Bằng chứng đã kiểm tra

| Kiểm tra | Kết quả |
|---|---|
| Header log các run | `Submission : False` ở tất cả |
| Chuỗi `"Running inference on test"` trong log | 0 lần xuất hiện |
| Số dòng mọi file `submission_*.txt` | **999** — đúng bằng số track **validation** |
| Test public / blind có bao nhiêu track | 1.000 / 3.000 — không khớp 999 |

> ⚠️ **Bẫy đặt tên**: các file `submission_*.txt` hiện có **không phải bài nộp**. Đó là
> dự đoán trên **validation**, do `Trainer.save_submission()` ghi ra mỗi khi val acc
> cải thiện. Dự đoán test thật sẽ do `predict_test()` ghi ra với tên
> `submission_<tên>_final.txt`, có 1.000 dòng (public) hoặc 3.000 dòng (blind).

### Hai điều phải biết trước khi chạy test

**(1) `--submission-mode` KHÔNG phải chỉ là inference — nó train lại từ đầu.**

```python
train.py:309-317   → MultiFrameDataset(..., full_train=True, ...)
dataset.py:154-156 → "FULL TRAIN: dùng toàn bộ tracks, không chia val" → val_tracks = []
```

Train lại trên **toàn bộ 20.000 track** (gồm cả 999 track val), **không còn validation**.
Hệ quả trong `Trainer.fit()`: khi `val_loader is None` thì **early stopping không hoạt
động**, và checkpoint lưu theo **train loss** — thứ gần như luôn giảm. Chạy đủ 60 epoch
ở chế độ này sẽ **lưu ra model của epoch cuối, đã overfit nặng** (S1 đỉnh val ở epoch
24–32 rồi tụt).

→ Nếu dùng, **bắt buộc đặt `--epochs` bằng epoch tốt nhất học được từ validation**
(S1: trung bình 28 qua 3 seed), không để 60.

**(2) Chưa có công cụ chạy test trên checkpoint có sẵn.** `tools/eval_decode.py` chạy
`mode="val"`, không nhận test set.

### Hai phương án

| | **A — dùng lại checkpoint đã có** | **B — `--submission-mode`** |
|---|---|---|
| Cách làm | Viết script inference nhỏ (~1 h người), nạp `*_best.pth`, chạy trên test | Train lại toàn bộ 20k track rồi predict test |
| Chi phí GPU | ~0 (chỉ inference) | **thêm 6–7 h** |
| Dữ liệu train | 19.001 track | 20.000 track (**+5%**) |
| Rủi ro | Không có — model đúng bằng model đã báo cáo val | **Cao**: không có val để kiểm chứng, không early stopping, dễ ship model overfit |
| Nhất quán cho paper | ✅ số val và số test cùng **một** model | ❌ **hai** model khác nhau |

**Khuyến nghị: phương án A.** Với một bài báo (khác đua leaderboard thuần tuý), việc số
val và số test đến từ *cùng một model* quan trọng hơn 5% dữ liệu train thêm. Phương án B
tạo ra một model mà **không cách nào biết nó tốt hay xấu** trước khi nộp.

Nếu vẫn muốn B: chạy **cả hai**, nộp bản A trước để có mốc an toàn, rồi so bản B trên
public test.

**Lệnh phương án B, sẵn để chạy** (`--epochs 28` = trung bình best epoch của S1 qua
3 seed: 28/32/24 — tính từ `results/multi-seed/s1_mf_sr_ocr/history_s1_seed*.csv`):

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

> ⚠️ Model ra từ lệnh này **khác hẳn** 3 checkpoint `s1_seed*_best.pth` đã có — đây là
> 1 lần train mới trên 20.000 track (thêm 5% dữ liệu, gồm cả 999 track val), dùng seed
> mặc định của `--preset stable`. Số val đã có (79.78–80.08%) **không phải** số của
> chính model này.
>
> Phương án A (khuyến nghị, dùng `s1_seed42_best.pth` có sẵn + inference thuần trên
> test) **chưa có lệnh sẵn** — cần viết thêm tool trước (~1 giờ, xem bảng trên).

### Thứ tự đúng

```
multi-seed (val) ✅  →  chốt cấu hình = S1 ✅  →  inference test public
                                              →  (nếu có leaderboard) đối chiếu
                                              →  test blind cuối cùng
```

> 🚨 **Tuyệt đối không** dùng test public để chọn cấu hình — sẽ biến nó thành tập
> validation thứ hai và làm số blind test mất giá trị. Việc chọn cấu hình đã **xong
> trước** khi nhìn bất kỳ con số test nào.

## 4. Limitations phải ghi vào paper

1. **Nhánh SR chưa chứng minh được đóng góp** — J1 (bỏ hẳn SR/DCN/MFSR) hoà S1 với
   1/3.76 chi phí. Chênh trong nhiễu → kết luận đúng là _"SR không giúp"_, **không**
   phải _"bỏ SR thì tốt hơn"_.
2. **Chưa tách được từng thành phần trong cụm cờ nền** (STN pool vs domain-match vs
   decode vs EMA) — câu hỏi mở lớn nhất còn lại.
3. **Ablation là tích luỹ**, không tách 1 biến: J1→S1 đổi cùng lúc SR + DCN + MFSR, và
   `T` cũng nhảy 16→32.
4. **Các biến thể khác của nhánh SR chỉ có 1 seed** (trọng số λ_SR, perceptual loss,
   single-frame vs multi-frame) — dữ liệu ở `backup/`, dùng cho phụ lục.
5. **Không đo được PSNR/SSIM cho J1** — bảng PSNR không phủ được cấu hình có điểm trung
   bình cao nhất (J1 không có `I_SR` để đo).
6. **PSNR/SSIM và hình định tính đo trên checkpoint 1 seed**, khác nguồn với bảng
   accuracy Mean ± Std.
7. **Chưa chạy test lần nào** — mọi số là validation Scenario-B (999 track);
   PSNR/SSIM đo trên **cặp synthetic**.
8. **J1 dùng cờ nền của S1/S4**, nên 80.45% **không so được** với J1 lịch sử 76.88%.
9. Thiếu `log_j1p_seed42.txt` (CSV/submission/checkpoint vẫn đủ).
10. **Biên nhiễu ±13 track (±1.3 điểm)** trên val 999 track — mọi chênh lệch nhỏ hơn
    ngưỡng này không kết luận được.
11. **Chưa khảo sát các biện pháp chống overfitting** (weight decay 1e-3, Dropout 0.3,
    patience 12, MotionBlur/GaussNoise) — cả 9/9 run đều overfit rõ (val loss chạm đáy
    ep 11–22 rồi tăng 38–67%). Hướng cải thiện có cơ sở nhất còn lại.

## 5. Rủi ro chính

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Reviewer tự tính ra "J1 hoà S1 mà rẻ hơn 3.76×" | **cao** | **Chủ động nêu** ở Limitations #1 thay vì để họ phát hiện |
| Số test thấp hơn val đáng kể | trung bình | Dùng phương án A để val và test cùng một model |
| Bước 4 (PARSeq/SVTR) không kịp | trung bình | Review cho phép cắt — ghi vào Limitations |
| Hình Figure 4 không khớp số trong bảng | thấp | Với S1 thì chênh 1-seed ↔ multi-seed **bằng 0** |
