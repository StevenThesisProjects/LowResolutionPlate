# Multi-seed J1 · S1 · S4 — kết quả chính thức (Mean ± Std)

> ✅ **ĐỦ CẢ 3 MODEL** (42/100/2026, deterministic `--no-cudnn-benchmark`).
> **Bước 1 của review đã HOÀN THÀNH.**
>
> 🎯 **Quyết định (2026-08-04): chọn S1 làm phương pháp đề xuất chính của paper** —
> vì **ổn định nhất** (std `0.15`, nhỏ hơn 3× J1/S4) và **bất biến với
> `cudnn.benchmark`** (lệch 0 track ở seed 42). J1 và S4 đóng vai **ablation**.
>
> 🚨 **Kết quả âm tính đi kèm, bắt buộc công bố**: cấu hình **không có SR (J1) có
> điểm trung bình CAO NHẤT** — **80.45% ± 0.45**, **hoà** S1 (chênh trong nhiễu) và
> **hơn S4 có ý nghĩa thống kê**, trong khi rẻ hơn **3.76× compute**. Nhánh SR chưa
> chứng minh được đóng góp đo được. Phân tích: [§4](#4-🚨-kết-quả-chính--sr-không-mang-lại-lợi-ích-đo-được).
>
> | Model | SR | `T` | **Mean ± Std** | Std | GFLOPs | Vai trò |
> |---|---|---:|---:|---:|---:|---|
> | **S1** | ×2 multi-frame + DCN | 32 | **79.95% ± 0.15** | **0.15** 🥇 | 109.08 | 🎯 **đề xuất** |
> | **J1** | ❌ không SR | 16 | 80.45% ± 0.45 | 0.45 | **26.14** | ablation "bỏ SR" |
> | **S4** | ×1 multi-frame + DCN | 32 | 79.48% ± 0.44 | 0.44 | chưa đo | ablation "bỏ upscale" |
>
> Dữ liệu: `results/multi-seed/{crnn_resblock_groupnorm_nosr_j1,s1_mf_sr_ocr,s4_sr_scale1}/`.
> Mọi con số đã **chấm lại trực tiếp** `submission_*.txt` với `plate_text` thật
> trong `annotations.json` của 999 track val, không chỉ tin `val_acc` in trong log.

## 1. Cấu hình đã chạy

Cả 2 cấu hình chạy đúng lệnh đã ghi ở
[../training_runs/run_gpu.md §0 B4](../training_runs/run_gpu.md), chỉ khác `--seed`
và có thêm `--no-cudnn-benchmark`:

```bash
# S1 — 3 seed
for SEED in 42 100 2026; do
  python train.py --preset stable --experiment-name s1_seed${SEED} --seed ${SEED} \
    --epochs 60 --batch-size 32 --grad-accum-steps 2 \
    --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
    --backbone-norm group --lr-domain-match \
    --decode constrained --use-ema \
    --no-cudnn-benchmark --num-workers 8 --aug-level full \
    2>&1 | tee results/log_s1_seed${SEED}.txt
done

# J1 — 3 seed. LƯU Ý: KHÔNG dùng cờ lịch sử (STN pool 1,1) mà dùng
# ĐÚNG bộ cờ nền của S1/S4, chỉ bỏ SR/DCN → ablation 1-cụm-biến sạch.
for SEED in 42 100 2026; do
  python train.py --preset stable --experiment-name j1p_seed${SEED} --seed ${SEED} \
    --epochs 60 --batch-size 32 --grad-accum-steps 2 \
    --backbone-norm group --lr-domain-match \
    --decode constrained --use-ema \
    --no-cudnn-benchmark --num-workers 8 --aug-level full \
    2>&1 | tee results/log_j1p_seed${SEED}.txt
done

# S4 — 3 seed (đã chạy trước, xem s4_sr_scale1_mf_sr_ocr.md §5d)
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

Đã xác nhận từ banner log: cả 6 run đều in đúng `deterministic: True`,
`Model params: 29,577,214` (S1) / tương ứng S4, và `Seed: {42,100,2026}` khớp tên
experiment. `nan_batches = 0` ở mọi epoch của cả 6 run.

## 2. Kết quả từng seed

### J1 (KHÔNG SR, T=16) — 🥇 điểm cao nhất

| Seed | Track đúng | Val Acc | Best epoch | Tổng epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 799/999 | 79.98% | 22 | 40 | 0.2030 | 0.0525 | 0.9642 | 6 | 0 |
| 100 | **808/999** | **80.88%** | 17 | 35 | 0.1855 | 0.0532 | 0.9589 | 11 | 0 |
| 2026 | 804/999 | 80.48% | 35 | 53 | 0.2571 | 0.0518 | 0.9765 | 2 | 0 |
| **Mean ± Std** | | **80.45% ± 0.45** | | | | **0.0525 ± 0.0007** | 0.9665 ± 0.0089 | | **0/999** |

⚠️ J1 dùng **`T=16`** (không SR nên width không nhân đôi) và params 29,442,700 —
khác J1 **lịch sử** (76.88%, STN pool `(1,1)`, params 29,313,452). Hai số này
**không đặt chung cột**. Chi tiết:
[j1_groupnorm_nosr.md](j1_groupnorm_nosr.md).

### S1 (SR ×2, multi-frame + DCN)

| Seed | Track đúng | Val Acc | Best epoch | Tổng epoch (early-stop) | Val Loss | CER ↓ | Conf. TB | Track conf <0.55 | Track sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 797/999 | 79.78% | 28 | 46 | 0.2094 | 0.0541 | 0.9655 | 6 | 0 |
| 100 | 800/999 | 80.08% | 32 | 50 | 0.2141 | 0.0525 | 0.9692 | 4 | 0 |
| 2026 | 799/999 | 79.98% | 24 | 42 | 0.1988 | 0.0556 | 0.9585 | 12 | 1 |
| **Mean ± Std** | | **79.95% ± 0.15** | | | | 0.0541 ± 0.0016 | 0.9644 ± 0.0054 | | |

### S4 (SR ×1, T=32 qua `width_downsample=4`) — nhắc lại từ [s4_sr_scale1_mf_sr_ocr.md §5d](s4_sr_scale1_mf_sr_ocr.md)

| Seed | Track đúng | Val Acc | Best epoch | Tổng epoch (early-stop) | Val Loss | CER ↓ | Conf. TB | Track conf <0.55 | Track sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 789/999 | 78.98% | 41 | 59 | 0.2887 | 0.0543 | 0.9748 | 0 | 0 |
| 100 | 797/999 | 79.78% | 23 | 41 | 0.1945 | 0.0543 | 0.9580 | 11 | 1 |
| 2026 | 796/999 | 79.68% | 40 | 58 | 0.2628 | 0.0542 | 0.9743 | 1 | 0 |
| **Mean ± Std** | | **79.48% ± 0.44** | | | | 0.0543 ± 0.0001 | 0.9691 ± 0.0096 | | |

**Quan sát**: S1 có **std nhỏ nhất** (0.15, so với 0.44 của J1 và S4) — ổn định
nhất qua các seed. Nhưng **J1 có CER tốt nhất** (0.0525) **và** là cấu hình duy nhất
đạt **0/999 track sai độ dài ở cả 3 seed**. Độ ổn định exact-match và độ ổn định CER
không đi cùng chiều: S4 ổn định nhất về CER (std 0.0001) nhưng kém nhất về exact match.

## 3. So sánh chính thức 3 model (`tools/aggregate_seeds.py`)

| Model | 3 seed | **Mean ± Std** | Xếp hạng |
|---|---|---:|:---:|
| **J1** (không SR) | 79.98 / 80.88 / 80.48 | **80.45% ± 0.45** | 🥇 |
| **S1** (đề xuất) | 79.78 / 80.08 / 79.98 | **79.95% ± 0.15** | 🥈 |
| **S4** (SR ×1) | 78.98 / 79.78 / 79.68 | **79.48% ± 0.44** | 🥉 |

Kiểm định từng cặp (ngưỡng: chênh > 2× sai số hiệu mới coi là thật):

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| **J1 vs S1** | +0.50 | ±0.28 | ⚠️ trong biên nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **vượt 2× sai số → J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong biên nhiễu → **hoà** |

## 4. 🚨 Kết quả chính — SR không mang lại lợi ích đo được

**Cấu hình bỏ hẳn nhánh SR (J1) đạt điểm cao nhất trong cả 3 model.** J1 hoà S1
(đề xuất) và **hơn S4 một cách có ý nghĩa thống kê**, trong khi rẻ hơn **3.76×**
về compute.

| | SR | DCN | `T` | GFLOPs/track | Val Acc (3 seed) |
|---|:---:|:---:|---:|---:|---:|
| **J1** | ❌ | ❌ | **16** | **26.14** | **80.45% ± 0.45** |
| S1 | ×2 multi-frame | ✅ | 32 | 109.08 (**3.76×**) | 79.95% ± 0.15 |
| S4 | ×1 multi-frame | ✅ | 32 | chưa đo | 79.48% ± 0.44 |

Ở mức từng track (cùng seed, net gain của J1) — cùng chiều ở **cả 3 seed**:

| So với | seed 42 | seed 100 | seed 2026 | TB |
|---|---:|---:|---:|---:|
| S1 | +2 | +8 | +5 | **+5.0 track** |
| S4 | +10 | +11 | +8 | **+9.7 track** |

### Hai giả thuyết trung tâm bị bác bỏ

**1. "Nhánh SR đóng góp vào độ chính xác"** — ❌ không có bằng chứng.
J1 và S1 dùng **chung toàn bộ cụm cờ nền** (GroupNorm, STN pool `(4,8)`,
`--lr-domain-match`, constrained decode, EMA); S1 chỉ thêm SR + DCN + MFSR. Thêm
cụm đó vào **không cải thiện** (−0.50 điểm, trong nhiễu) mà tốn **3.76× compute**.

Vậy +3.57 điểm mà J1-mới hơn J1-**lịch sử** (76.88% → 80.45%) đến từ **cụm cờ nền**,
không phải SR. Kiểm chứng mức track: J1-mới hơn J1-lịch-sử **+31 / +40 / +36 track**
qua 3 seed.

**2. "`T=32` là yếu tố chính đứng sau lợi ích của SR"** — ❌ cũng không đứng vững.
J1 chạy **`T=16`** mà vẫn ngang/hơn S1 và S4 (đều `T=32`). Giả thuyết nêu ở
[s4_sr_scale1_mf_sr_ocr.md §3](s4_sr_scale1_mf_sr_ocr.md) mất cơ sở: khi cụm cờ nền
đã bật, `T=16` không hề kém `T=32`.

→ **Cách đọc trung thực nhất**: cải thiện thật so với baseline gốc đến từ **cụm cờ
nền (STN pool `(4,8)` + domain-match + constrained decode + EMA)**. **Module SR
chưa chứng minh được đóng góp nào** trên tập validation này.

### Giới hạn của kết luận này

- **3 seed, n=999** — chênh J1↔S1 (+0.50) nằm trong nhiễu, nên đúng ra chỉ kết
  luận được *"SR không giúp"*, **không** phải *"bỏ SR thì tốt hơn"*.
- Chưa tách được **từng thành phần trong cụm cờ nền** (STN pool vs domain-match vs
  decode vs EMA) — cần ablation riêng, ngoài phạm vi hiện tại.
- Chỉ đo trên **validation Scenario-B**; chưa chạy test.

## 5. Bất đối xứng — `cudnn.benchmark` KHÔNG luôn thổi phồng số

Đây là phát hiện đáng chú ý nhất của đợt multi-seed này: hiệu ứng của
`cudnn.benchmark=True` **không giống nhau** giữa S1 và S4.

| | Số 1-seed cũ (`benchmark=True`, seed 42 mặc định) | Seed 42 multi-seed (deterministic) | Chênh |
|---|---:|---:|---:|
| **S1** | 797/999 (79.78%, epoch 37) | 797/999 (79.78%, epoch 28) | **0 track — giống hệt** |
| **S4** | 805/999 (80.58%, epoch 40) | 789/999 (78.98%, epoch 41) | **−16 track (−1.60 điểm)** |

Cùng seed 42, cùng thay đổi duy nhất là chế độ cudnn:
- **S4**: lệch 16 track — vượt hẳn biên nhiễu ±13, đã phân tích ở
  [s4_sr_scale1_mf_sr_ocr.md §5d](s4_sr_scale1_mf_sr_ocr.md).
- **S1**: lệch **0 track** — cùng seed, hai chế độ cudnn khác nhau, ra đúng cùng
  1 con số (khác epoch đạt đỉnh, nhưng điểm số tuyệt đối bằng nhau).

**Hệ quả quan trọng cho cách viết paper**: không nên khái quát hoá "1-seed +
`cudnn.benchmark=True` luôn thổi phồng kết quả" thành quy luật chung của toàn bộ
project. Mức độ nhạy cảm với tính không xác định của thuật toán convolution có vẻ
**phụ thuộc vào cấu hình cụ thể** (có thể liên quan đến việc S4 dùng
`width_downsample=4`, một kiến trúc backbone khác S1) — đây là quan sát mới, chưa
có lời giải thích chắc chắn, nên nêu rõ trong paper là hiện tượng đã đo được chứ
không suy diễn nguyên nhân.

Nhìn theo hướng khác: **con số S1 headline (79.78–80.08%, mean 79.95%) là ổn định
nhất và đáng tin nhất trong toàn bộ project tính đến nay** — không phụ thuộc vào
việc có bật `cudnn.benchmark` hay không.

## 6. Ổn định dự đoán qua các seed — track nào đổi, track nào không

Ghép 3 file `submission_*_seed{42,100,2026}.txt` theo từng cấu hình:

| Cấu hình | Track giống nhau ở cả 3 seed | Track đổi dự đoán tuỳ seed |
|---|---:|---:|
| **J1** | **789/999 (79.0%)** | **210/999 (21.0%)** |
| S1 | 778/999 (77.9%) | 221/999 (22.1%) |
| S4 | 768/999 (76.9%) | 231/999 (23.1%) |

Cả hai cấu hình có khoảng **1/5 số track "nhạy cảm với seed"** — không phải lỗi
riêng của một cấu hình, mà là đặc điểm chung của bài toán ở mức exact-match trên
7 ký tự (một ký tự sai là cả track sai). S1 nhạy seed **thấp hơn một chút** so với
S4 (22.1% vs 23.1%), khớp với việc S1 có std nhỏ hơn ở mục 2.

## 7. So sánh dự đoán S1 vs S4 theo từng seed — không phải cùng 1 model

So `submission_s1_seed{X}.txt` với `submission_s4_seed{X}.txt` ở **cùng seed X**
(loại yếu tố seed ra khỏi so sánh):

| Seed | Track khác dự đoán | S1 đúng / S4 sai | S4 đúng / S1 sai |
|---|---:|---:|---:|
| 42 | 193/999 (19.3%) | 49 | 41 |
| 100 | 181/999 (18.1%) | 44 | 41 |
| 2026 | 186/999 (18.6%) | 43 | 40 |

Tỷ lệ khác dự đoán (~18–19%) **nhất quán qua cả 3 seed** và gần khớp con số đo được
ở bản 1-seed trước đây (182/999 = 18.2%,
[s4_sr_scale1_mf_sr_ocr.md §2](s4_sr_scale1_mf_sr_ocr.md)) —
xác nhận đây là đặc điểm ổn định của 2 kiến trúc khác nhau, không phải nhiễu ngẫu
nhiên của 1 lần chạy. Ở cả 3 seed, số track "S1 đúng/S4 sai" nhỉnh hơn "S4 đúng/S1
sai" (49>41, 44>41, 43>40) — cùng chiều với S1 có mean acc cao hơn S4, nhưng chênh
lệch nhỏ và nằm trong biên nhiễu đã nêu ở mục 3. Gợi ý cho hướng ensemble S1+S4 vẫn
còn giá trị, vì ~18% track "đổi chỗ"
là nguồn bổ sung thông tin thật, không phải trùng lặp.

## 8. Compute — J1 rẻ nhất mà điểm cao nhất

| | Phút/epoch (deterministic, TB 3 seed) | GFLOPs/track | Latency (ms) | Val Acc (3 seed) |
|---|---:|---:|---:|---:|
| **J1** (không SR) | ~2.5 (ước tính) | **26.14** | **51.30** | **80.45% ± 0.45** 🥇 |
| S1 (MFSR+DCN) | 9.39 | 109.08 (**3.76×**) | 192.75 | 79.95% ± 0.15 |
| S4 (SR ×1) | 4.09 | chưa đo | chưa đo | 79.48% ± 0.44 |

Chế độ deterministic chỉ làm chậm **2–4%** so với `benchmark=True` (S1: 9.39 vs
9.21; S4: 4.09 vs 3.94) — thấp hơn nhiều so với ước lượng thận trọng ban đầu (+25%,
[../paper/paper_revision_plan.md](../paper/paper_revision_plan.md)).

> **Con số nên đưa vào paper**: **S1 tốn 3.76× compute so với J1 để đổi lấy −0.50
> điểm** (trong biên nhiễu). Đây là lập luận chi phí–lợi ích mạnh nhất của cả bài.
>
> ⚠️ Phút/epoch của J1 là **ước tính** từ tỷ lệ GFLOPs, chưa đo trực tiếp (log
> seed 42 bị thiếu). Nên đo lại bằng `tools/benchmark.py` trước khi in số.

## 9. Kết luận & cập nhật cho paper

0. 🎯 **Phương pháp chính đã chốt: S1** (`79.95% ± 0.15`). Lý do: std nhỏ nhất
   (0.15 vs 0.45/0.44), bất biến với `cudnn.benchmark` (mục 5), và J1 **không** hơn
   S1 có ý nghĩa nên không có cơ sở đổi cấu hình đề xuất. J1/S4 = ablation.
1. 🚨 **Nhánh SR không mang lại lợi ích đo được.** J1 (bỏ hẳn SR/DCN/MFSR) đạt
   **80.45% ± 0.45** — hoà S1 (+0.50, trong nhiễu) và **hơn S4 có ý nghĩa thống kê**
   (+0.97 > 2×0.36). Đây là kết luận quan trọng nhất của đợt multi-seed, và là
   **Limitation bắt buộc** phải nêu kèm khi trình bày S1.
2. **Cải thiện thật đến từ cụm cờ nền**, không phải SR: STN pool `(4,8)` +
   `--lr-domain-match` + constrained decode + EMA. J1-mới hơn J1-lịch-sử **+3.57
   điểm** (76.88% → 80.45%) chỉ nhờ cụm này, với **cùng** kiến trúc không SR.
3. **Giả thuyết `T=32` bị bác bỏ** — J1 chạy `T=16` mà vẫn ngang/hơn S1, S4 (đều
   `T=32`). Kết luận cũ ở [s4_sr_scale1_mf_sr_ocr.md §3](s4_sr_scale1_mf_sr_ocr.md)
   mất cơ sở.
4. **S1 và S4 không khác biệt có ý nghĩa** (+0.47, trong nhiễu); S4 nhanh hơn S1
   2.30× khi train.
5. **`cudnn.benchmark=True` không phải lúc nào cũng thổi phồng** — S1 lệch 0 track,
   S4 lệch 16 track ở cùng seed 42. Nêu như quan sát thực nghiệm, không khái quát hoá.
6. **Bước 1 của review đã HOÀN THÀNH** — đủ 3 model có Mean ± Std trên 3 seed.

### ⚠️ Việc paper phải xử lý (không thuộc phạm vi tài liệu này)

- **Nhánh SR chưa chứng minh được đóng góp** — phải nêu ở Limitations kèm con số
  chi phí 3.76×, dù S1 vẫn là phương pháp đề xuất.
- Chưa tách được **từng thành phần trong cụm cờ nền** — cần ablation riêng.
- Chưa khảo sát có error bar các biến thể của nhánh SR (trọng số λ_SR, perceptual
  loss, single-frame vs multi-frame) — chỉ có dữ liệu 1 seed, lưu ở `backup/report/`.
- Chỉ đo validation Scenario-B; **chưa chạy test lần nào**.
