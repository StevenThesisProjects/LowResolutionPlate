# S1 — Joint End-to-End MF-SR-OCR (🎯 phương pháp đề xuất)

> **Con số chính thức: 79.95% ± 0.15** trên 3 seed deterministic (42/100/2026).
>
> **Đã chốt S1 là phương pháp đề xuất chính của paper** — không phải vì điểm cao nhất
> (J1 nhỉnh hơn 0.50, nhưng nằm trong biên nhiễu → **hoà**), mà vì **ổn định nhất**:
> std `0.15`, nhỏ hơn **3×** so với J1 và S4.
>
> 🚨 **Kết quả âm tính bắt buộc công bố kèm**: J1 — **cùng bộ cờ nền, chỉ bỏ
> SR/DCN/MFSR** — hoà điểm với S1 trong khi rẻ hơn **3.76× compute**. Nghĩa là
> **nhánh SR chưa chứng minh được đóng góp đo được**.
> Chi tiết: [j1_groupnorm_nosr.md](j1_groupnorm_nosr.md).

## 1. Kiến trúc — 5 bước của pipeline đề xuất

```
5 LR frames
  → [1] STN per-frame (affine 6 tham số, pool (4,8))
  → [2] DCNv2 align (kernel identity-init)
  → [3] Multi-frame SR ×2  (+ L_SR)
  → [4] backbone ResBlock + Attention Fusion
  → [5] BiLSTM + CTC  (constrained decode)
```

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}} \cdot \mathcal{L}_{\text{SR}}, \qquad \lambda_{\text{SR}} = 0.1$$

$$\mathcal{L}_{\text{SR}} = \frac{1}{N}\sum \left\| I_{\text{SR}} - \mathrm{Warp}_{\mathrm{sg}[\theta]}(I_{\text{HR}}) \right\|_1$$

Công thức đầy đủ + 4 chỗ paper đang ghi sai:
[../paper/loss_formula_corrected.md](../paper/loss_formula_corrected.md).

> 📌 `α` (perceptual) và `β` (Sobel edge) đều đặt **0** ở S1 — đã verify trên banner
> cả 3 log. Params: **29,577,214** · `T = (128 × 2) / 8 = 32`.

## 2. Cấu hình đã chạy

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
> đây là 2 chỗ khác S4. Gõ nhầm là dựng sai kiến trúc, chạy 22 giờ ra kết quả vô nghĩa.
>
> Dữ liệu: `results/multi-seed/s1_mf_sr_ocr/` · 19.001 track train, 999 track val
> (Scenario-B) · `nan_batches = 0` mọi epoch.

## 3. Kết quả 3 seed

| Seed | Track đúng | Val Acc | Best epoch | Số epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 797/999 | 79.78% | 28 | 46 | 0.2094 | 0.0541 | 0.9655 | 6 | 0 |
| 100 | 800/999 | 80.08% | 32 | 50 | 0.2141 | 0.0525 | 0.9692 | 4 | 0 |
| 2026 | 799/999 | 79.98% | 24 | 42 | 0.1988 | 0.0556 | 0.9585 | 12 | 1 |
| **Mean ± Std** | | **79.95% ± 0.15** | | | | 0.0541 ± 0.0016 | 0.9644 ± 0.0054 | | |

**Std 0.15 — nhỏ nhất trong 3 model** (J1: 0.45 · S4: 0.44). Đây là cấu hình duy nhất
mà cả 3 seed rơi trong khoảng **0.3 điểm**.

## 4. 🎯 Vì sao chọn S1 làm phương pháp đề xuất

1. **J1 không hơn S1 có ý nghĩa** (+0.50, sai số hiệu ±0.28) → thống kê là **hoà**.
   Chọn giữa hai cấu hình hoà nhau thì quyết định bằng **độ ổn định**.
2. **Ổn định nhất qua seed** — std 0.15, nhỏ hơn 3× hai cấu hình còn lại.
3. **Bất biến với `cudnn.benchmark`** — cùng seed 42, chạy `benchmark=True` và
   `deterministic=True` cho **đúng cùng con số** (797/999, lệch **0 track**), trong khi
   S4 lệch **16 track**. Con số của S1 **tái lập được nhất** trong cả project.
4. Là **kiến trúc hoàn chỉnh** mà toàn bộ câu chuyện của bài xây quanh nó; J1 và S4
   đóng đúng vai **ablation** của chính S1.

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| J1 vs S1 | +0.50 | ±0.28 | ⚠️ trong nhiễu → **hoà** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

## 5. 🚨 Giới hạn phải công bố — nhánh SR chưa chứng minh được đóng góp

J1 dùng **chung toàn bộ cụm cờ nền** với S1 (GroupNorm, STN pool `(4,8)`,
`--lr-domain-match`, constrained decode, EMA); S1 chỉ thêm **SR + DCN + MFSR**.
Thêm cụm đó vào **không cải thiện** mà tốn **3.76× compute**.

| | GFLOPs/track | Phút/epoch | Val Acc (3 seed) |
|---|---:|---:|---:|
| J1 (không SR) | **26.14** | **2.60** | 80.45% ± 0.45 |
| **S1** | 109.08 (**3.76×**) | 9.67 | **79.95% ± 0.15** |

**Cách viết trung thực**: _"SR không cho thấy lợi ích đo được trên tập val này"_ —
**không** phải _"SR có ích"_, cũng **không** phải _"bỏ SR thì tốt hơn"_ (chênh trong
nhiễu, không kết luận được chiều nào).

📌 **Một điểm ủng hộ việc giữ nhánh SR**: J1 overfit **sớm và sâu hơn** (val loss chạm
đáy ep 11–16 so với 15–22 của S1; train loss xuống 0.0074 so với ~0.03). Nhánh SR có
vẻ hoạt động như **regularizer đa nhiệm** — không cải thiện exact match nhưng có ghìm
được mức overfit. Đây là quan sát, chưa phải ablation riêng.

## 6. Constrained decode đóng góp bao nhiêu

20.000 nhãn đều dài 7 ký tự và chỉ có **2 layout** (`LLLNLNN` 13k · `LLLNNNN` 7k) →
khoá cứng 6/7 vị trí lớp chữ/số, beam 16.

| Seed | Constrained | Greedy | Chênh |
|---|---:|---:|---:|
| 42 | 79.78% | 79.78% | 0 |
| 100 | 80.08% | 79.88% | **+2 track** |
| 2026 | 79.98% | 79.98% | 0 |

Đóng góp nhỏ (**0–2 track**) nhưng miễn phí về compute — chạy ở khâu decode.

## 7. Chất lượng ảnh SR

| | PSNR (SR) | PSNR (base) | Chênh | SSIM (SR) | SSIM (base) | Chênh |
|---|---:|---:|---:|---:|---:|---:|
| S1 (×2) | 16.6827 | 15.6110 | +1.0717 dB | 0.4179 | 0.3481 | +0.0698 |

> ⚠️ Đo trên checkpoint **1 seed**, khác nguồn với bảng accuracy multi-seed.
> 🔬 Track model đọc **sai** lại có PSNR **cao hơn ~2 dB** (r = −0.362, n=999) — PSNR
> **nghịch** với khả năng đọc. Chi tiết: [../buoc2_metrics.md](../buoc2_metrics.md).

## 8. Overfit

| Seed | Val loss chạm đáy | Val acc đỉnh | Val loss cuối | Train loss cuối |
|---|---:|---:|---:|---:|
| 42 | ep 20 (0.1875) | ep 28 | 0.2849 | 0.0344 |
| 100 | ep 16 (0.1852) | ep 32 | 0.2701 | 0.0312 |
| 2026 | ep 19 (0.1888) | ep 24 | 0.2556 | 0.0411 |

Val loss chạm đáy sớm (ep 16–20) rồi tăng 36–52%, trong khi train loss tụt về ~0.03.
Val acc vẫn nhích lên tới ep 24–32 **dù val loss đã tăng** → early stopping theo
**Val Exact Match Accuracy** (không theo val loss) là lựa chọn đúng.

Các biện pháp chống overfit đã cài sẵn nhưng **chưa khảo sát** — xem Future work ở
[../checklist_review.md](../checklist_review.md).

## 9. Kết luận

1. **S1 = 79.95% ± 0.15** — con số headline của paper, ổn định và tái lập được nhất.
2. **Nhánh SR chưa chứng minh được đóng góp đo được** — Limitation bắt buộc, kèm con
   số chi phí **3.76×**.
3. **Phần cải thiện thật đến từ cụm cờ nền** (STN pool `(4,8)` + domain-match +
   constrained decode + EMA), không phải SR.
4. **Chưa tách được từng thành phần trong cụm cờ nền** — câu hỏi mở lớn nhất còn lại.
5. Chỉ đo trên **validation Scenario-B**; **chưa chạy test lần nào**.

📄 [multi_seed_results.md](multi_seed_results.md) ·
[j1_groupnorm_nosr.md](j1_groupnorm_nosr.md) ·
[s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md) ·
[model_comparison_summary.md](model_comparison_summary.md)
