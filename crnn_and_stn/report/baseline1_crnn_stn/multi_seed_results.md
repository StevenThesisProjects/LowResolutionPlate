# Multi-seed S1 & S4 — kết quả chính thức đầu tiên (Mean ± Std)

> ✅ **S1 và S4 đã có đủ 3 seed (42/100/2026), chế độ deterministic
> (`--no-cudnn-benchmark`).** Đây là 2 cấu hình đầu tiên trong project có số liệu
> **Mean ± Std** thay vì best-of-1-run — số trong file này là **số chính thức cho
> paper**, thay thế mọi số 1-seed của S1/S4 đã báo cáo trước đó.
> Còn thiếu **J1** (~10h) để hoàn thành phạm vi cuối cùng — **3 model J1 + S1 + S4**
> (bỏ J2, S2, S3) — xem [../checklist_review.md](../checklist_review.md).
>
> Dữ liệu nguồn: `results/multi-seed/s1_mf_sr_ocr/`, `results/multi-seed/s4_sr_scale1/`
> (`history_*_seed{42,100,2026}.csv`, `log_*_seed{42,100,2026}.txt`,
> `submission_*_seed{42,100,2026}.txt`). Đối chiếu bằng cách chấm trực tiếp
> `submission_*.txt` với `plate_text` thật trong `annotations.json` của 999 track
> val (không chỉ tin số `val_acc` in trong log/CSV).

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

**Quan sát đầu tiên**: S1 có **std nhỏ hơn S4 gần 3 lần** (0.15 vs 0.44 điểm) —
S1 ổn định hơn qua các seed dù kiến trúc phức tạp hơn (multi-frame SR + DCN so với
SR đơn giản của S4). CER của S4 lại **ổn định hơn** S1 (std 0.0001 vs 0.0016) —
hai đại lượng không nhất thiết đi cùng chiều.

## 3. So sánh chính thức S1 vs S4 (`tools/aggregate_seeds.py`)

```
S1 (SR x2, multi-frame+DCN)
  Số seed      : 3  (79.78%, 80.08%, 79.98%)
  Mean ± Std   : 79.95% ± 0.15
  Min / Max    : 79.78% / 80.08%
  CI 95% (n=999) : ±2.48 điểm  → [77.46%, 82.43%]

S4 (SR x1)
  Số seed      : 3  (78.98%, 79.78%, 79.68%)
  Mean ± Std   : 79.48% ± 0.44
  Min / Max    : 78.98% / 79.78%
  CI 95% (n=999) : ±2.50 điểm  → [76.98%, 81.98%]

============================================================
SO SÁNH
============================================================
  Chênh lệch   : +0.47 điểm (S1 so với S4)
  Sai số hiệu  : ±0.27
  ⚠️ Chênh lệch NẰM TRONG biên độ nhiễu → chưa đủ bằng chứng kết luận.
```

**Kết luận chính thức đầu tiên có error bar của project**: **S1 và S4 không khác
biệt có ý nghĩa thống kê** (+0.47 điểm < 2× sai số ±0.27, và cũng < biên nhiễu dự
án dùng xuyên suốt ±1.3 điểm). Claim *"S4 bằng S1 nhưng rẻ hơn 2.34×"* — đặt ra từ
[s4_sr_scale1_mf_sr_ocr.md §5c](s4_sr_scale1_mf_sr_ocr.md) —
**nay được xác nhận bằng multi-seed**, không còn là so sánh khập khiễng 1 bên
multi-seed 1 bên 1-seed như trước.

So với các mốc xa hơn, cả hai vẫn vượt rõ biên nhiễu ±1.3 điểm:

| So sánh | Chênh lệch | Trong biên nhiễu? |
|---|---:|:---:|
| S1 (79.95%) vs J2 1-seed (77.18%) | +2.77 | **không — vượt rõ** |
| S1 (79.95%) vs baseline gốc (77.00%) | +2.95 | **không — vượt rõ** |
| S4 (79.48%) vs J2 1-seed (77.18%) | +2.30 | **không — vượt rõ** |
| S4 (79.48%) vs baseline gốc (77.00%) | +2.48 | **không — vượt rõ** |
| **S1 (79.95%) vs S4 (79.48%)** | **+0.47** | **có — không phân biệt được** |

## 4. Bất đối xứng quan trọng — `cudnn.benchmark` KHÔNG luôn thổi phồng số

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

## 5. Ổn định dự đoán qua các seed — track nào đổi, track nào không

Ghép 3 file `submission_*_seed{42,100,2026}.txt` theo từng cấu hình:

| Cấu hình | Track giống nhau ở cả 3 seed | Track đổi dự đoán tuỳ seed |
|---|---:|---:|
| S1 | 778/999 (77.9%) | 221/999 (22.1%) |
| S4 | 768/999 (76.9%) | 231/999 (23.1%) |

Cả hai cấu hình có khoảng **1/5 số track "nhạy cảm với seed"** — không phải lỗi
riêng của một cấu hình, mà là đặc điểm chung của bài toán ở mức exact-match trên
7 ký tự (một ký tự sai là cả track sai). S1 nhạy seed **thấp hơn một chút** so với
S4 (22.1% vs 23.1%), khớp với việc S1 có std nhỏ hơn ở mục 2.

## 6. So sánh dự đoán S1 vs S4 theo từng seed — không phải cùng 1 model

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
còn giá trị (như đã nêu ở phân tích S1 vs S2 trước đây), vì ~18% track "đổi chỗ"
là nguồn bổ sung thông tin thật, không phải trùng lặp.

## 7. Compute — xác nhận lại tỷ lệ 2.3× dưới chế độ deterministic

| | Phút/epoch (deterministic, trung bình 3 seed) | So với 1-seed `benchmark=True` trước đó |
|---|---:|---|
| S1 | 9.39 | 9.21 (chỉ chậm hơn ~2.0%) |
| S4 | 4.09 | 3.94 (chỉ chậm hơn ~3.8%) |
| **Tỷ lệ S1/S4** | **2.30×** | 2.34× (số cũ) |

Chế độ deterministic chỉ làm chậm 2–4%, thấp hơn nhiều so với ước lượng thận trọng
ban đầu (+25%,
[../paper/paper_revision_plan.md](../paper/paper_revision_plan.md)) — cùng kết
luận với S4 đã ghi nhận trước đó. **Tỷ lệ tốc độ 2.3× của S4 so với S1 giữ nguyên**
dưới cả 2 chế độ cudnn, củng cố thêm cho claim "S4 rẻ hơn" — nay đã có cả 2 vế của
so sánh (accuracy tương đương + tốc độ nhanh hơn) đều đến từ multi-seed.

## 8. Kết luận & cập nhật cho paper

1. **S1 và S4 không khác biệt có ý nghĩa thống kê** (79.95% ± 0.15 so với
   79.48% ± 0.44, chênh +0.47 điểm < 2× sai số ±0.27) — đây là kết luận chính thức
   đầu tiên có error bar của toàn bộ project.
2. **S4 nhanh hơn S1 2.30× khi train, độ chính xác tương đương** — nếu ưu tiên chi
   phí compute cho submission/triển khai, **S4 là lựa chọn hợp lý hơn S1** dù S1 có
   mean acc điểm cao hơn một chút (không đáng kể về thống kê).
3. **`cudnn.benchmark=True` không phải lúc nào cũng thổi phồng kết quả** — với S1,
   seed 42 cho cùng 1 con số ở cả 2 chế độ; với S4 thì lệch 16 track. Đây là phát
   hiện mới, cần nêu trong paper như một quan sát thực nghiệm, không khái quát hoá
   thành quy luật.
4. **Bằng chứng bổ sung cho việc S1/S4 là 2 model thực sự khác nhau** (không chỉ
   khác điểm số): ~18-19% track đổi dự đoán ở mọi seed, ~22-23% track nhạy với seed
   ở mỗi cấu hình riêng — cả 2 tỷ lệ đều ổn định qua 3 seed, không phải nhiễu của
   1 lần chạy.
5. **Việc còn lại — 1 run**: multi-seed **J1** (không SR, T=16, ~10h) cho claim
   *"S1 vượt cấu hình không SR"* có error bar. Chạy **đúng cờ lịch sử**
   (`--stn-pool 1,1`) → tái lập được ~76.88% vì J1 không dùng SR. Lệnh chạy:
   [../training_runs/run_gpu.md §0 B4](../training_runs/run_gpu.md).
   ⏭️ **J2 đã ra ngoài phạm vi** — ablation multi-frame vs single-frame giữ ở mức
   1 seed (J2 77.18% vs S1 79.78%, chênh 26 track) + ghi Limitations.
6. **Vẫn để ngỏ (ghi vào Limitations)**: câu hỏi "T confound" S4 đặt ra **chưa được
   giải**. S1↔S4 chỉ cho biết SR×2 tương đương SR×1 khi cùng `T=32`; muốn tách đóng
   góp của module SR khỏi đóng góp của việc tăng `T` thì cần run
   `--width-downsample 4` **không** `--use-sr` — **J1 không thay thế được** (J1
   dùng T=16). **J2 cũng không tách được** — bật SR ×2 tự động nâng `T` 16→32, nên
   J1→J2 đổi 2 biến cùng lúc. J2 đáng chạy vì lý do khác: mốc single-frame để so
   multi-frame với S1.

Cập nhật liên quan: [model_comparison_summary.md §1c](model_comparison_summary.md#1c-kết-quả-multi-seed--số-chính-thức-cho-paper),
[../checklist_review.md](../checklist_review.md).
