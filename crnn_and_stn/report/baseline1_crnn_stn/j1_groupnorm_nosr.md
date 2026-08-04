# J1 — ResBlock + GroupNorm, KHÔNG SR (ablation "bỏ hẳn nhánh SR")

> **Vai trò trong paper**: ablation của S1 — trả lời câu hỏi _"pipeline SR đề xuất có
> thật sự hơn backbone không SR không?"_, có error bar trên 3 seed.
>
> **Kết quả: 80.45% ± 0.45** — hoà S1 (chênh trong biên nhiễu) và **hơn S4 có ý nghĩa
> thống kê**, trong khi rẻ hơn **3.76× compute**.

## 1. Vì sao có J1

GroupNorm được thêm vào để sửa lỗi NaN khi bật SR trên backbone `norm=none`. Nhưng
**GroupNorm tự nó** có thể đã cải thiện accuracy, không liên quan gì tới SR. Nếu không
đo riêng một cấu hình **không SR**, khi cấu hình có SR vượt baseline sẽ không biết công
lao thuộc về SR hay thuộc về phần nền.

J1 chính là cấu hình "phần nền, không SR" đó.

## 2. ⚠️ Hai phiên bản J1 — KHÔNG đặt chung cột

| | J1 **lịch sử** | J1 **multi-seed** (dùng cho paper) |
|---|---|---|
| STN pool | `(1,1)` | **`(4,8)`** |
| `--lr-domain-match` | ❌ | ✅ |
| Decode | greedy | **constrained** |
| EMA | ❌ | ✅ |
| Params | 29,313,452 | **29,442,700** |
| Số seed | 1 | **3** (42/100/2026) |
| Val Acc | 76.88% (768/999) | **80.45% ± 0.45** |

Hai con số này **lệch nhau 4-5 tham số cùng lúc**, không so trực tiếp được. J1
multi-seed dùng **đúng bộ cờ nền của S1/S4, chỉ bỏ `--use-sr` và `--use-dcn`** — đó
mới là ablation 1-cụm-biến sạch.

📌 Chênh **+3.57 điểm** giữa hai phiên bản (76.88% → 80.45%) chính là đóng góp của
**cụm cờ nền**: STN pool `(4,8)` + `--lr-domain-match` + constrained decode + EMA.
Kiểm chứng ở mức track: **+31 / +40 / +36 track** qua 3 seed.

## 3. Cấu hình multi-seed đã chạy

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

> ⚠️ Không `--use-sr` / `--use-dcn` → width **không** nhân đôi → **`T = 128 ÷ 8 = 16`**
> (S1 và S4 đều `T=32`).
>
> 📌 `j1p` chỉ là **tên experiment** lúc gõ lệnh, không phải tên model.
>
> Dữ liệu: `results/multi-seed/crnn_resblock_groupnorm_nosr_j1/`

## 4. Kết quả 3 seed

| Seed | Track đúng | Val Acc | Best epoch | Số epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 799/999 | 79.98% | 22 | 40 | 0.2030 | 0.0525 | 0.9642 | 6 | 0 |
| 100 | **808/999** | **80.88%** | 17 | 35 | 0.1855 | 0.0532 | 0.9589 | 11 | 0 |
| 2026 | 804/999 | 80.48% | 35 | 53 | 0.2571 | 0.0518 | 0.9765 | 2 | 0 |
| **Mean ± Std** | | **80.45% ± 0.45** | | | | **0.0525 ± 0.0007** | 0.9665 ± 0.0089 | | **0/999** |

Đã chấm lại trực tiếp `submission_*.txt` với `plate_text` gốc — khớp chính xác số in
trong log. **0/999 track sai độ dài ở cả 3 seed** — cấu hình duy nhất đạt được điều này.

## 5. 🚨 So với S1 và S4 — SR không mang lại lợi ích đo được

| So sánh | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| **J1 vs S1** (79.95% ± 0.15) | **+0.50** | ±0.28 | ⚠️ trong biên nhiễu → **hoà** |
| **J1 vs S4** (79.48% ± 0.44) | **+0.97** | ±0.36 | ✅ **> 2× sai số → J1 tốt hơn thật** |

Ở mức từng track (cùng seed, net gain của J1) — cùng chiều ở **cả 3 seed**:

| So với | seed 42 | seed 100 | seed 2026 | TB |
|---|---:|---:|---:|---:|
| S1 | +2 | +8 | +5 | **+5.0 track** |
| S4 | +10 | +11 | +8 | **+9.7 track** |

### Hai giả thuyết trung tâm bị bác bỏ

**1. "Nhánh SR đóng góp vào độ chính xác" — KHÔNG có bằng chứng.**
J1 bỏ hẳn SR/DCN/MFSR mà vẫn **hoà S1** và **hơn S4 rõ rệt**. Phần cải thiện thật đến
từ **cụm cờ nền** (mục 2), không phải từ SR.

**2. "`T=32` là yếu tố chính" — cũng KHÔNG đứng vững.**
J1 chạy **`T=16`** mà vẫn ngang S1 và hơn S4 (đều `T=32`). Khi cụm cờ nền đã bật,
`T=16` không hề kém `T=32`. Giả thuyết nêu ở
[s4_sr_scale1_mf_sr_ocr.md §3](s4_sr_scale1_mf_sr_ocr.md) mất cơ sở.

> ⚠️ **Cách đọc trung thực**: kết luận đúng là _"SR không cho thấy lợi ích đo được
> trên tập val này"_ — **không** phải _"bỏ SR thì tốt hơn"_. Chênh J1↔S1 nằm trong
> biên nhiễu nên không kết luận được chiều nào.

## 6. Chi phí — J1 rẻ nhất mà điểm cao nhất

| | GFLOPs/track | Latency (ms) | Phút/epoch | vs J1 | Val Acc (3 seed) |
|---|---:|---:|---:|---:|---:|
| **J1** (không SR) | **26.14** | **51.30** | **2.60** | **1.00×** | **80.45% ± 0.45** |
| S1 (MFSR+DCN) | 109.08 | 192.75 | 9.67 | **3.76×** | 79.95% ± 0.15 |
| S4 (SR ×1) | chưa đo | chưa đo | 4.28 | — | 79.48% ± 0.44 |

**S1 tốn 3.76× compute để đổi lấy −0.50 điểm** (trong biên nhiễu). Đây là đánh đổi
chi phí–lợi ích trung tâm phải nêu khi bảo vệ lựa chọn S1 làm phương pháp đề xuất.

Phút/epoch là **số đo thật** từ cột `epoch_time_s` của toàn bộ 424 epoch, không phải
ước tính. Tổng chi phí J1: **5.55 h** cho 3 seed.

## 7. Overfit — J1 overfit sớm nhất và sâu nhất

| Seed | Val loss chạm đáy | Val acc đỉnh | Val loss cuối | Train loss cuối |
|---|---:|---:|---:|---:|
| 42 | ep 16 (0.1836) | ep 22 | 0.2665 | **0.0160** |
| 100 | ep 12 (0.1820) | ep 17 | 0.2516 | **0.0209** |
| 2026 | ep 11 (0.1819) | ep 35 | 0.3044 | **0.0074** |

Val loss chạm đáy ngay **ep 11–16** (S1/S4: ep 15–22) và train loss tụt tới **0.0074**.

📌 Hợp lý vì J1 **không có nhánh SR đóng vai regularizer đa nhiệm**. Đây là lập luận
đáng nêu khi bảo vệ việc **giữ nhánh SR** trong phương pháp đề xuất: SR không cải thiện
exact match, nhưng **có** ghìm được mức overfit.

## 8. Caveat thống kê

- Validation chỉ **999 track** → biên nhiễu **±13 track (±1.3 điểm)**. Mọi chênh lệch
  nhỏ hơn ngưỡng này không kết luận được.
- Trước đợt multi-seed, cùng một cấu hình cùng seed từng chạy 2 lần lệch nhau **6.5
  điểm** ở epoch 3 do `cudnn.benchmark=True` khiến thuật toán convolution không xác
  định. Toàn bộ 3 seed ở đây đều chạy `--no-cudnn-benchmark`.
- ⚠️ **Thiếu `log_j1p_seed42.txt`** — CSV + submission + checkpoint của seed 42 vẫn đủ
  nên không mất số liệu, nhưng không tra lại được banner cấu hình của riêng seed đó.

## 9. Kết luận

1. J1 đạt **80.45% ± 0.45** — điểm trung bình cao nhất trong 3 model, nhưng **hoà S1**
   về mặt thống kê.
2. **Nhánh SR chưa chứng minh được đóng góp đo được** — đây là Limitation bắt buộc
   phải công bố kèm khi trình bày S1.
3. Phần cải thiện thật so với baseline đến từ **cụm cờ nền**, không phải SR.
4. Chưa tách được **từng thành phần** trong cụm cờ nền (STN pool vs domain-match vs
   decode vs EMA) — câu hỏi mở quan trọng nhất còn lại.

📄 Phân tích đầy đủ 9 run: [multi_seed_results.md](multi_seed_results.md) ·
So sánh 3 model: [model_comparison_summary.md](model_comparison_summary.md)
