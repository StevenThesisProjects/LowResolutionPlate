# S4 — Joint MF-SR-OCR với SR ×1 (ablation "bỏ phóng to ảnh")

> **Vai trò trong paper**: ablation của S1 — tách biến `T` khỏi biến "SR có phóng to
> ảnh hay không".
>
> **Kết quả: 79.48% ± 0.44** trên 3 seed — **kém J1 có ý nghĩa thống kê** (−0.97,
> vượt 2× sai số hiệu ±0.36), hoà S1 (−0.47, trong nhiễu). Đây là cặp **duy nhất**
> trong bộ 3 model cho khác biệt thật.

## 1. Câu hỏi S4 đặt ra — "T confound"

Bật `--use-sr --sr-scale 2` làm **hai việc cùng lúc**:

1. Phóng to ảnh ×2 (32×128 → 64×256) — mục đích thật của SR.
2. **Nhân đôi số bước thời gian CTC**: `T` từ 16 lên 32 — tác dụng phụ.

$$T = \frac{\text{IMG\_WIDTH} \times (\text{SR\_SCALE nếu USE\_SR})}{\text{WIDTH\_DOWNSAMPLE}}$$

Nên mọi lợi ích quan sát ở S1 **không biết** đến từ ảnh nét hơn hay từ `T` dài hơn.

**S4 tách hai thứ đó**: giữ `T = 32` **mà không** phóng to ảnh —
`--sr-scale 1` + `--width-downsample 4`.

| | S1 | S4 | J1 |
|---|---|---|---|
| SR phóng to ảnh | ✅ ×2 | ❌ (×1) | ❌ không có SR |
| Kích thước vào backbone | 64×256 | **32×128** | 32×128 |
| `T` | 32 | **32** | **16** |

## 2. Vì sao `--sr-scale 1` hợp lý về mặt dữ liệu

Ảnh HR gốc chỉ khoảng **115×42 px**. Target ×2 (rộng 256 px) có tới **~55% là nội suy
bicubic thuần** — tức đang bắt model học lại phép nội suy chứ không phải chi tiết thật.
`--sr-scale 1` giữ output 32×128, xấp xỉ **đúng độ phân giải HR thật**.

## 3. Cấu hình đã chạy

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

> ⚠️ **Hai tham số phải đi cùng nhau**: `--sr-scale 1` không phóng to ảnh (`FrameSR.forward`
> trả `base = x` thay vì `F.interpolate`), nên nếu giữ `width_downsample=8` mặc định thì
> `T` tụt về 16 — khi đó S4 khác S1 tới **2 biến chồng nhau** thay vì tách được đúng
> biến muốn đo.
>
> Params: **29,549,470** (S1: 29,577,214). Dữ liệu: `results/multi-seed/s4_sr_scale1/`.

## 4. Kết quả 3 seed

| Seed | Track đúng | Val Acc | Best epoch | Số epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 789/999 | 78.98% | 41 | 59 | 0.2887 | 0.0543 | 0.9748 | 0 | 0 |
| 100 | 797/999 | 79.78% | 23 | 41 | 0.1945 | 0.0543 | 0.9580 | 11 | 1 |
| 2026 | 796/999 | 79.68% | 40 | 58 | 0.2628 | 0.0542 | 0.9743 | 1 | 0 |
| **Mean ± Std** | | **79.48% ± 0.44** | | | | 0.0543 ± **0.0001** | 0.9691 ± 0.0096 | | |

📌 **CER ổn định nhất trong 3 model** (std `0.0001`, nhỏ hơn S1 **16×**) nhưng **exact
match kém ổn định** (0.44 vs S1 0.15). Hai đại lượng **không đi cùng chiều** — đáng nêu.

## 5. 🚨 Câu trả lời cho "T confound" — giả thuyết bị bác bỏ

Ban đầu S4 được kỳ vọng chứng minh *"`T=32` mới là yếu tố chính, không phải việc SR
phóng to ảnh"*. Multi-seed **bác bỏ chính giả thuyết đó**:

| | `T` | SR phóng to | Val Acc (3 seed) |
|---|---:|:---:|---:|
| **J1** | **16** | ❌ | **80.45% ± 0.45** |
| S1 | 32 | ✅ ×2 | 79.95% ± 0.15 |
| **S4** | 32 | ❌ | **79.48% ± 0.44** |

**J1 chạy `T=16` mà vẫn cao nhất.** Nếu `T=32` là yếu tố quyết định thì S1 và S4 phải
hơn J1 — thực tế ngược lại. Khi cụm cờ nền đã bật, **`T=16` không hề kém `T=32`**.

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **> 2× sai số → J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

## 6. ⚠️ Bài học phương pháp luận — số 1-seed lạc quan 16 track

Con số 1-seed từng báo cáo cho S4 là **805/999 (80.58%)** — cao hơn mức thật **1.10
điểm** và **cao hơn cả 3 seed** (max 79.78%).

Cùng seed 42, chỉ đổi chế độ cudnn:

| | Track đúng | Val Acc |
|---|---:|---:|
| `benchmark=True` (1-seed gốc) | 805/999 | 80.58% |
| `deterministic=True` (multi-seed) | 789/999 | 78.98% |
| **Chênh** | **16 track** | **1.60 điểm** |

**16 track vượt biên nhiễu ±13** — riêng tính không xác định của thuật toán convolution
đã đủ tạo chênh lệch lớn hơn ngưỡng dùng để phán xét "có cải thiện hay không".

> 🔬 **Nhưng KHÔNG khái quát hoá được**: cùng phép thử này, **S1 lệch 0 track**
> (797/999 ở cả hai chế độ). Mức nhạy cảm với `cudnn.benchmark` **phụ thuộc cấu hình
> cụ thể** — có thể liên quan tới việc S4 dùng `width_downsample=4`, một cấu hình
> backbone khác. Nêu như **hiện tượng đã đo được**, không suy diễn nguyên nhân.
>
> Đây cũng là lý do thứ hai chọn **S1** làm phương pháp đề xuất: con số của S1 **tái
> lập được** bất kể chế độ cudnn.

## 7. Chi phí — rẻ hơn S1 nhưng không đổi lại được điểm

| | Phút/epoch | Tổng 3 seed | GFLOPs/track |
|---|---:|---:|---:|
| J1 | **2.60** | **5.55 h** | **26.14** |
| S1 | 9.67 | 22.24 h | 109.08 |
| **S4** | 4.28 | 11.27 h | chưa đo |

S4 **nhanh hơn S1 2.26×** khi train (backbone xử lý ảnh 32×128 thay vì 64×256 đã phóng
to) — nhưng vẫn **chậm hơn J1 1.65×** trong khi điểm thấp hơn có ý nghĩa thống kê.

> ⚠️ `tools/benchmark.py` **chưa benchmark được S4** — thiếu cờ `--sr-scale` và
> `--width-downsample`. Phút/epoch ở trên là **số đo thật** từ cột `epoch_time_s`,
> không phải ước tính.

## 8. Chất lượng ảnh SR

| | PSNR (SR) | PSNR (base) | Chênh | SSIM (SR) | SSIM (base) | Chênh |
|---|---:|---:|---:|---:|---:|---:|
| S4 (×1) | 17.4613 | 16.4241 | +1.0372 dB | 0.4955 | 0.4199 | +0.0756 |

> ✅ Đo trên **checkpoint multi-seed seed 42** (2026-08-05), cùng nguồn với bảng
> accuracy Mean ± Std. Val acc in ra lúc chạy: **789/999**, khớp chính xác.
>
> ⚠️ **PSNR tuyệt đối của S4 không so được với S1** — S4 xuất ảnh 32×128, S1 xuất
> 64×256, hai thang khác nhau. Chỉ so được cột "Chênh".
>
> Với `sr_scale=1`, mốc `base` là **ảnh giữ nguyên** (không nội suy), nên chênh
> +1.04 dB đọc là *"SR có hơn việc không làm gì"*.
>
> 🔬 **Nghịch lý đáng nêu**: S4 tái tạo ảnh **tốt hơn S1** (+1.04 vs +0.72 dB) nhưng
> **đọc kém hơn** (79.48% vs 79.95%) — thêm một bằng chứng PSNR nghịch với OCR.
> Ở mức từng track: track đọc **sai** có PSNR cao hơn **2.21 dB** (r = −0.4044, n=999).

Chi tiết + tương quan PSNR ↔ đọc đúng: [../buoc2_metrics.md](../buoc2_metrics.md).

## 9. Kết luận

1. **Giả thuyết `T=32` bị bác bỏ** — J1 ở `T=16` vẫn cao nhất. Câu hỏi "T confound"
   coi như đã có câu trả lời: `T` **không** phải yếu tố quyết định.
2. **S4 là cấu hình yếu nhất trong bộ 3** — thua J1 có ý nghĩa thống kê.
3. **Số 1-seed của S4 lạc quan 16 track** — mọi kết quả 1-seed trong dự án phải đọc
   kèm cảnh báo này, nhưng **không** khái quát thành quy luật vì S1 lệch 0 track.
4. S4 vẫn hữu ích như **ablation cho thấy việc phóng to ảnh không cần thiết**, và như
   bằng chứng CER/exact-match ổn định theo hai chiều khác nhau.

📄 [multi_seed_results.md](multi_seed_results.md) ·
[j1_groupnorm_nosr.md](j1_groupnorm_nosr.md) ·
[s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md)
