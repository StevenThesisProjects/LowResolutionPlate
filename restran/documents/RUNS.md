# Nhật ký thí nghiệm ResTran

Ghi lại **lệnh chạy + kết quả** của từng lần train để đối chứng và so sánh. Mỗi run một mục, không sửa mục cũ (kể cả khi run đó hỏng — run hỏng là dữ liệu có giá trị).

Mốc tham chiếu từ slide ban tổ chức ([restran_baseline.md §6.2](baseline/restran_baseline.md)): ResTran **75,80%**, ResTran + STN **78,70%** trên val Scenario-B.

> ⚠️ **Luôn đặt `-n` khác nhau cho mỗi run.** Trùng tên sẽ ghi đè `{tên}_best.pth`, `{tên}_metrics.csv` và `submission_{tên}.txt` của run trước. Từ ngày 12/08/2026 `train.py` in cảnh báo khi phát hiện tên đã tồn tại, nhưng vẫn chạy tiếp nếu bạn bỏ qua.

---

## Bảng tổng hợp

| # | Ngày | Tên run | Epochs | LR đỉnh | clip | AMP | Best val acc | Kết cục |
|---|---|---|---|---|---|---|---|---|
| 1 | 12/08 12:18 | `restran_stn` | 30 (dừng ở 14) | 5e-4 | 5.0 | fp16 | 61,76% (ep10) | ❌ nổ ở ep12 |
| 2 | 12/08 ~13:00 | `restran_stn_v2` (bf16) | — | 3e-4 | 1.0 | **bf16** | — | ❌ huỷ: chậm 38× |
| 3 | 12/08 13:14 | `restran_stn_v2` | 30 | 3e-4 | 1.0 | fp16 | **76,68% (ep27)** | ✅ **tốt nhất** |
| 4 | 12/08 13:41 | `restran_stn_v2` | 60 | 3e-4 | 1.0 | fp16 | 67,97% (ep23) | ❌ nổ ở ep25 |
| 5 | 12/08 14:39 | `restran_stn_v2` | 60 | 3e-4 | 1.0 | fp16 | 67,47% (ep22) | ❌ nổ ở ep25 |
| 6 | 12/08 16:10 | `restran_30ep_lr3e4` | 30 | 3e-4 | 1.0 | fp16 | **77,18% (ep28)** | ✅ **tốt nhất** |
| 7 | 12/08 ~17:00 | `restran_60ep_lr3e4` | 60 | 3e-4 | 1.0 | fp16 | 65,87% (ep21) | ❌ nổ ep22, cầu dao dừng ở ep23 |
| 8 | 12/08 ~17:30 | `restran_60ep_lr3e4` | 60 | 3e-4 | 1.0 | fp16 | 67,17% (ep23) | ❌ nổ ep25, cầu dao dừng ở ep25 |

**Cấu hình tốt nhất hiện tại (cập nhật 01/10/2026): run #38 `restran_degcal` — 77,94% trên val 1999 (TB6 77,30).** Bảng run #23–#38 và vòng E1–E4 ở mục "Run #23 → #38" phía dưới. (Mốc cũ: run #6, 77,18% trên val 999.)

---

## Run #1 — baseline gốc, nổ giữa chừng

```bash
python -u train.py -n restran_stn --num-workers $(nproc) \
    2>&1 | tee results/restran_stn_train.log
```

Toàn bộ mặc định: `lr 5e-4`, `GRAD_CLIP 5.0`, fp16, không có chốt chặn gradient, 30 epoch.

| Epoch | train_loss | grad_norm | val_acc |
|---|---|---|---|
| 10 | 0,245 | 3,79 | **61,76% ← đỉnh** |
| 11 | 0,227 | 3,46 | 59,26% |
| 12 | 1,554 | **899,6** | 0% |
| 13 | 3,238 | **1325,2** | 0% |
| 14 | 3,389 | 6,10 | 0% |

**Kết luận:** grad norm nổ 260× ở epoch 12 rồi CTC rơi vào nghiệm suy biến "chỉ xuất blank" (loss đứng 3,39, độ lệch chuẩn tương đối sụp từ 29% xuống 4,3% → đầu ra gần như độc lập với ảnh vào). Đáng chú ý: LR lúc nổ là 4,76e-4, **thấp hơn** đỉnh 5e-4 đã đi qua an toàn ở epoch 9 — nên nguyên nhân không phải riêng giá trị LR.

---

## Run #2 — thử bf16, huỷ

```bash
python -u train.py -n restran_stn_v2 --num-workers $(nproc) \
    --lr 5e-4 --grad-clip 1.0 --amp-dtype bf16 --skip-grad-norm 100 \
    2>&1 | tee results/restran_stn_v2_train.log
```

**Huỷ sau ~20 batch: 3,19 s/it so với 0,084 s/it của fp16 — chậm 38×**, mỗi epoch 30 phút thay vì 50 giây.

**Nguyên nhân:** `seed_everything` đặt `cudnn.deterministic = True`, mà cuDNN có rất ít thuật toán conv tất định cho bf16 → rơi xuống đường chạy chậm.

**Kết luận: không dùng bf16 trên máy này.** Benchmark trước khi thử lại — xem [RUN_SERVER.md §4.1](run_cpu/RUN_SERVER.md).

---

## Run #3 — ✅ tốt nhất: 76,68%

```bash
python -u train.py -n restran_stn_v2 --num-workers $(nproc) \
    --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    2>&1 | tee results/restran_stn_v2_train.log
```

| | |
|---|---|
| Best val acc | **76,6767%** tại epoch 27 |
| Tổng thời gian | 25,4 phút (30 epoch × ~50,4 s) |
| Throughput | ~763 samples/s, VRAM đỉnh 1196 MB |
| grad_norm cuối | 1,64 (đỉnh trong epoch: 6,3) |
| Batch bị bỏ | 0–3 mỗi epoch, tổng ~9 |

Diễn biến: acc leo đều 54,9% (ep9) → 64,2% (ep13) → 71,9% (ep19) → 75,6% (ep24) → 76,68% (ep27), val_loss chạm đáy 0,2176 ở ep25 rồi đi ngang. Không có dấu hiệu bất ổn định ở bất kỳ epoch nào.

**So với baseline slide (78,70%): còn kém 2,02 điểm.** Nhưng val ~1.000 mẫu có sai số ±1,5 điểm nên khoảng cách này chưa kết luận chắc chắn được.

> ⚠️ Checkpoint của run này **đã bị run #4 và #5 ghi đè** vì dùng chung tên `restran_stn_v2`. Trọng số 76,68% coi như mất, cần chạy lại nếu muốn dùng.

---

## Run #4 và #5 — kéo dài 60 epoch, cả hai đều nổ ở epoch 25

```bash
python -u train.py -n restran_stn_v2 --num-workers $(nproc) --epochs 60 \
    --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    2>&1 | tee results/restran_stn_v2_train.log
```

| | Run #4 | Run #5 |
|---|---|---|
| Best val acc | 67,97% (ep23) | 67,47% (ep22) |
| Epoch nổ | 25 | 25 |
| Batch bị bỏ ở ep25 | 243/594 | 536/594 |
| Tổng batch bị bỏ | 21.019 (59%) | 21.310 (60%) |
| Thời gian | 49,0 phút | 48,8 phút |

### Vì sao 60 epoch hỏng còn 30 epoch thì không

Cùng `--lr 3e-4`, nhưng `OneCycleLR` với `pct_start=0.3` **giãn theo tổng số epoch**:

| Tại epoch 25 | LR | Trạng thái |
|---|---|---|
| Lịch 30 epoch | **4,0e-5** (đã hạ sâu) | loss 0,053 · acc 75,78% — an toàn |
| Lịch 60 epoch | **2,82e-4** (còn sát đỉnh) | nổ |

Với 30 epoch, LR đạt đỉnh ở ep9 rồi hạ liên tục. Với 60 epoch, đỉnh dời tới ep18 và LR còn trên 2,8e-4 tới tận ep25 — model ngồi ở vùng LR cao lâu gấp ~3 lần. **Thứ quyết định là thời gian ở LR cao, không phải giá trị LR đỉnh.**

### Chữ ký của cú nổ

| Run #4 | loss_mean | loss_max | grad_norm | grad_norm_max |
|---|---|---|---|---|
| ep24 | 0,147 | 0,367 | 2,73 | 7,78 |
| ep25 | 0,321 | **0,837** | **135,1** | **1399,1** |

**Loss vẫn nhỏ trong khi gradient lên 1399.** Nếu do một mẫu dữ liệu hỏng thì loss phải vọt lên hàng đơn vị — nên đây là bất ổn định số học/tối ưu, không phải dữ liệu bẩn. Xảy ra đúng lúc model đã rất khớp (loss 0,145) mà LR vẫn cao: gradient CTC trở nên nhọn khi model tự tin, LR cao đủ để hất nó ra khỏi vùng hội tụ.

### Hai run cùng seed nhưng khác kết quả

67,97%@ep23 vs 67,47%@ep22. `seed_everything` chỉ đặt `cudnn.deterministic = True`, **không** gọi `torch.use_deterministic_algorithms(True)`. `F.grid_sample` (backward của STN) và CTC backward dùng atomics nên không tất định trên CUDA. Cạnh một điểm bất ổn định, sai khác chữ số cuối khuếch đại thành khác biệt thấy được.

### Lỗi của chốt chặn (đã sửa 12/08)

Từ ep26 trở đi cả hai run bỏ **594/594 batch mỗi epoch** — 35 epoch, ~30 phút GPU không cập nhật gì. Hai dấu vết trong CSV:

- **LR đóng băng**: run #5 giữ nguyên `2.847329e-04` suốt ep26–29 vì `scheduler.step()` nằm trong nhánh không-bị-bỏ.
- **val_acc vẫn trôi** dù trọng số bất động (17,42 → 17,92 → 18,02): `model.train()` vẫn chạy forward nên BatchNorm của ResNet34 tiếp tục cập nhật running mean/var.

Đã sửa: xem [phần Thay đổi code](#thay-đổi-code-12082026).

---

## Run #6 — ✅ tốt nhất: 77,18%

```bash
python -u train.py -n restran_30ep_lr3e4 --num-workers $(nproc) \
    --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    2>&1 | tee results/restran_30ep_lr3e4_train.log
```

Cùng cấu hình run #3, chạy lại với tên riêng. **77,1772% tại epoch 28**, 24,4 phút (16:10 → 16:34). Batch bị bỏ: 9 trong toàn bộ 30 epoch, **cả 9 đều là inf/nan** — đó là hành vi bình thường của `GradScaler` fp16, không phải bất ổn định.

Chênh so với run #3 (76,68%) là **+0,50 điểm** dù cấu hình y hệt → đây là mức nhiễu run-to-run, không phải cải tiến.

---

## Run #7 và #8 — 60 epoch, cầu dao hoạt động đúng

```bash
python -u train.py -n restran_60ep_lr3e4 --num-workers $(nproc) --epochs 60 \
    --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    2>&1 | tee results/restran_60ep_lr3e4_train.log
```

| | Run #7 | Run #8 |
|---|---|---|
| Best val acc | 65,87% (ep21) | 67,17% (ep23) |
| Epoch nổ | 22 | 25 |
| Cầu dao dừng ở | **ep23** | **ep25** |
| Thời gian | 19,5 phút | 21,1 phút |

**Cầu dao làm đúng việc**: dừng ở epoch 23 và 25 thay vì chạy hết 60 epoch — tiết kiệm ~30 phút GPU mỗi run so với run #4/#5.

### Bằng chứng quyết định: KHÔNG phải tràn số fp16

Cột `skipped_nonfinite` mới tách bạch được hai nguyên nhân:

| | Tổng batch bị bỏ | inf/nan (tràn fp16) | Hữu hạn nhưng quá lớn |
|---|---|---|---|
| Run #7 | 651 | 16 (**2,5%**) | 635 (**97,5%**) |
| Run #8 | 357 | 13 (**3,6%**) | 344 (**96,4%**) |

Hơn 96% số batch bị bỏ có gradient **hữu hạn** nhưng khổng lồ. Đây là bất ổn định tối ưu thật sự, **không phải tràn số fp16** → chuyển sang bf16 sẽ không cứu được gì. Giả thuyết fp16 chính thức bị loại.

### Điều kiện gây nổ — đo được, có tính dự báo

Gộp 4 lần nổ (run #4, #5, #7, #8) và 2 lần thành công (run #3, #6):

| Run | Epoch khoẻ cuối | train_loss | LR | Nổ ở |
|---|---|---|---|---|
| #7 | 21 | 0,1637 | 2,963e-4 | ep22 |
| #4 | 24 | 0,1469 | 2,852e-4 | ep25 |
| #5 | 24 | 0,1443 | 2,852e-4 | ep25 |
| #8 | 24 | 0,1489 | 2,852e-4 | ep25 |

**Vùng nguy hiểm: `train_loss ≲ 0,165` trong khi `LR ≳ 2,85e-4`.**

Lịch 30 epoch không bao giờ vào vùng đó: khi loss chạm 0,167 (ep15) thì LR đã hạ còn 2,44e-4, và tiếp tục giảm nhanh hơn tốc độ loss giảm. Lịch 60 epoch giữ LR quanh 2,85–2,96e-4 suốt các epoch 20–25 — đúng lúc loss xuống dưới 0,165.

Diễn giải: gradient của CTC trở nên nhọn khi model đã rất tự tin (loss thấp); LR còn cao ở thời điểm đó đủ để hất model ra khỏi vùng hội tụ. Không liên quan đến dữ liệu bẩn — `train_loss_max` ở epoch nổ chỉ 0,71–0,96, tức không mẫu nào có loss bất thường.

**Suy ra cho lần sau:** muốn chạy 60 epoch thì LR tại epoch ~24 phải ≤ 2,4e-4. Với `OneCycleLR` (`pct_start=0.3`), LR ở ep24/60 ≈ 0,95 × max_lr, nên **max_lr phải ≤ 2e-4** để có biên an toàn.

---

## Run #10 — cân bằng layout: ❌ KHÔNG hiệu quả

```bash
python -u train.py -n restran_30ep_balance173 --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --layout-balance 1.73 \
    2>&1 | tee results/restran_30ep_balance173_train.log
```

| | Run #9 (mốc) | Run #10 (balance 1.73) | Chênh |
|---|---|---|---|
| **Tổng** | **77,19%** | 76,24% (ep27) | **−0,95** |
| Mercosur | 83,06% | 82,06% | −1,00 |
| Brazilian | 54,07% | 53,33% | −0,74 |

Chạy sạch (9 batch bỏ, đều inf/nan), 25,5 phút. Brazilian trong luồng mẫu tăng từ 36,1% lên 49,6%.

**Kết quả: Brazilian không nhúc nhích.** Tăng gấp rưỡi tỉ trọng Brazilian trong train cho ra đúng 0 điểm cải thiện; Mercosur mất 1 điểm. Chênh lệch tổng −0,95 ± 1,33 nằm trong nhiễu, nhưng **không có dấu hiệu nào của lợi ích**.

→ **Giả thuyết "Brazilian kém vì ít dữ liệu" bị bác bỏ bằng thực nghiệm trực tiếp.**

### Giả thuyết "ảnh Brazilian mờ hơn" cũng bị bác bỏ

Chia val theo tứ phân vị độ nét rồi so hai layout **trong cùng một mức độ nét**:

| Độ nét (Laplacian var) | Mercosur | Brazilian | Chênh |
|---|---|---|---|
| Q1 (65–2709) | 61,70% (n=376) | 31,45% (n=124) | 30,25 |
| Q2 (2709–4636) | 85,03% (n=394) | 59,43% (n=106) | 25,59 |
| Q3 (4636–6907) | 91,20% (n=409) | 59,34% (n=91) | 31,86 |
| Q4 (6907–15880) | 92,55% (n=416) | 75,00% (n=84) | 17,55 |

Khoảng cách tồn tại **trong mọi tứ phân vị** (18–32 điểm). Độ nét không giải thích được nó.

Cả hai nguyên nhân khả dĩ đều đã bị loại bằng đo đạc → khoảng cách này là **đặc tính nội tại của biển Brazilian** (kiểu chữ, độ tương phản thiết kế), không sửa được bằng cách lấy mẫu.

### Phát hiện có giá trị hơn: lỗi tập trung ở ảnh mờ

| Tứ phân vị độ nét | Accuracy | Số lỗi | % tổng lỗi |
|---|---|---|---|
| **Q1 (mờ nhất)** | **54,20%** | **229** | **50,2%** |
| Q2 | 79,60% | 102 | 22,4% |
| Q3 | 85,40% | 73 | 16,0% |
| Q4 (nét nhất) | 89,60% | 52 | 11,4% |

**25% ảnh mờ nhất gây ra một nửa tổng số lỗi.** Độ nét là biến dự báo mạnh hơn layout rất nhiều: chênh Q1→Q4 là 35 điểm ở Mercosur và 44 điểm ở Brazilian. Đây mới là trục đáng khai thác.

---

## Run #13 — augment nhất quán theo track + bỏ ChannelShuffle

```bash
python -u train.py -n restran_30ep_aug --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --consistent-aug --no-channel-shuffle \
    2>&1 | tee results/restran_30ep_aug_train.log
```

| | Run #9 (mốc) | Run #13 | Chênh |
|---|---|---|---|
| **Tổng** | 77,19% | **77,59%** (ep29) | +0,40 |
| Mercosur | 83,06% | 83,12% | +0,06 |
| Brazilian | 54,07% | **55,80%** | +1,73 |
| **val_loss thấp nhất** | 0,2342 | **0,2111** | **−0,023 (−9,9%)** |
| train_loss cuối | 0,0330 | 0,0556 | +0,023 |
| Thời gian | 25,6 phút | **37,5 phút** | +46% |

**Accuracy +0,40 điểm là trong nhiễu** (±0,94). Nhưng hai tín hiệu khác thì không:

1. **val_loss tốt hơn 9,9%** — 0,2111 so với 0,2342, thấp nhất trong mọi run từ trước tới nay. val_loss có phương sai nhỏ hơn exact-match rất nhiều nên đây là tín hiệu thật.
2. **train_loss lại CAO hơn** (0,0556 vs 0,0330) và accuracy **vẫn đang lên ở ep29–30**. Model **chưa hội tụ** — bài toán khó hơn (CoarseDropout giờ che cùng một chỗ trên cả 5 frame) nên 30 epoch không còn đủ.

→ Cấu hình này **bị thiếu epoch**, không phải vô dụng. `max LR/loss = 1,23e-3` (mốc là 1,6e-3) nên còn nhiều biên an toàn để kéo dài.

**Giá phải trả: chậm 46%** (70,9 s/epoch so với 47,0 s), do `ReplayCompose` tốn chi phí ghi/phát lại tham số.

---

## Run #15 — ĐO TRẦN bằng ảnh HR: 98,45%

```bash
python -u train.py -n restran_hr_ceiling --num-workers $(nproc) \
    --epochs 60 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --use-hr \
    2>&1 | tee results/restran_hr_ceiling_train.log
```

Train và val đều dùng `hr-*` không degrade. 60 epoch × 282 batch ≈ cùng số bước tối ưu với các run thường. 28,1 phút. Checkpoint đã xác minh lại bằng eval độc lập: **98,4492%**.

| | LR (run #13) | **HR (run #15)** | Chênh |
|---|---|---|---|
| Tổng | 77,59% | **98,45%** | **+20,86** |
| Mercosur | 83,12% | **99,44%** | +16,32 |
| Brazilian | 55,80% | **94,57%** | **+38,77** |
| val_loss | 0,2111 | **0,0216** | −90% |

### Kết luận

**Kiến trúc không phải nút thắt.** ResTran đọc gần như hoàn hảo khi ảnh đủ nét. Toàn bộ 21 điểm thiếu hụt là chất lượng ảnh → không cần đổi model/decoder, mà phải cải thiện đầu vào.

### ⚠️ Đính chính kết luận ở run #10

Sau run #10 tôi viết khoảng cách Brazilian là "đặc tính nội tại của biển Brazilian, không sửa được bằng cách lấy mẫu". **Sai.**

| | Mercosur | Brazilian | Khoảng cách |
|---|---|---|---|
| Ảnh LR | 83,12% | 55,80% | **27,3 điểm** |
| Ảnh HR | 99,44% | 94,57% | **4,9 điểm** |

Biển Brazilian không khó đọc về bản chất — nó **xuống cấp nặng hơn ở độ phân giải thấp**. Cải thiện chất lượng ảnh sẽ giúp Brazilian nhiều gấp đôi Mercosur. Sai lầm: kết luận "nội tại" chỉ vì đã loại được hai nguyên nhân, thay vì thừa nhận chưa biết.

### ⚠️ Đính chính tiêu chí LR/loss

Run này có `max LR/loss = 38,06e-3` — gấp 24 lần ngưỡng "nguy hiểm" 1,6e-3 — và **không nổ**, vì train_loss xuống 0,0005 làm tỉ số phình lên do mẫu số. Tiêu chí đó chỉ là **quy luật kinh nghiệm trong vùng loss 0,03–0,20**, không phải định luật như tôi đã trình bày.

### Hệ quả cho hướng đi

Teacher HR 98,45% chính là đầu vào mà [super-reslution.md §3 Mức 2](baseline/super-reslution.md) cần: distillation `CTC + β·‖feat_student − feat_teacher‖²`. So khớp ở **không gian đặc trưng** nên **né được** vấn đề cặp LR–HR không đăng ký (§2.4) vốn chặn mọi hướng SR pixel-wise.

Nhưng 98,45% đo trên ảnh HR thật (~130×46 px) còn test chỉ có LR (~48×18 px). SR không tạo ra thông tin không tồn tại — 21 điểm là **trần lý thuyết tuyệt đối**, không phải mức đạt được.

---

## Run #16 (32×192) và #17 (distillation) — cả hai không hiệu quả

**Run #16 — độ phân giải 32×192 (T=16 → 24):**

| | Run #9 (32×128) | Run #16 (32×192) |
|---|---|---|
| Best acc | 77,19% | 77,5888% |
| val_loss thấp nhất | 0,2342 | **0,2346** |
| VRAM | 1.555 MB | 4.182 MB |

val_loss giống hệt → **T=16 chưa bao giờ là nút cổ chai**. Gấp 2,7 lần VRAM đổi lấy đúng con số cũ.

**Run #17 — distillation từ teacher HR (`--distill-weight 0.1`):**

```bash
python -u train.py -n restran_30ep_distill01 --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --teacher results/restran_hr_ceiling_best.pth --distill-weight 0.1 \
    2>&1 | tee results/restran_30ep_distill01_train.log
```

| | Run #9 (mốc) | Run #17 (distill) |
|---|---|---|
| **Tổng** | **77,19%** | 76,29% (−0,90) |
| Mercosur / Brazilian | 83,06 / 54,07 | 81,93 / 54,07 |
| val_loss thấp nhất | **0,2342** | 0,2349 |
| train_loss (CTC) cuối | **0,0330** | 0,0434 |
| `distill_loss` | — | 1,4374 → **0,1753** |

Tiêu chuẩn đặt ra **trước** khi chạy: thắng nếu vượt 79% hoặc val_loss rõ dưới 0,21. **Không đạt cả hai.**

### Vì sao thất bại

`distill_loss` giảm **8 lần** (1,44 → 0,175) — student học được cách tạo đặc trưng gần teacher hơn hẳn. Nhưng CTC lại **xấu đi** (0,0434 so với 0,0330) và val không nhúc nhích.

Nghĩa là student khớp được phần **cấu trúc chung** của đặc trưng teacher, còn phần **tần số cao mang tính phân biệt ký tự** thì không — vì thông tin đó không tồn tại trong ảnh LR. Mục tiêu phụ cạnh tranh với CTC thay vì hỗ trợ nó.

Dự đoán của tôi trước khi chạy là 77–79%; kết quả 76,29% nằm **ngoài khoảng, phía dưới**. Tôi đã lạc quan quá.

---

## ⚠️ Cảnh báo phương pháp luận: overfit tập val do chọn lọc

17 thí nghiệm, tất cả đánh giá trên cùng 1.999 track val. Với sai số ±0,94 điểm, nếu các cấu hình thực chất ngang nhau thì riêng do may rủi, cái tốt nhất trong 17 lần rút sẽ cao hơn giá trị thật khoảng **+1,6 điểm**.

Bằng chứng: bốn cấu hình khác hẳn nhau (mốc, augment, 40 epoch, 32×192) mà **ba trong số đó cho đúng cùng một con số 1551/1999**.

→ **Mọi chênh lệch dưới ~1,5 điểm từ đây trở đi là không đáng tin.** Chỉ hiệu ứng cỡ run #15 (+20,9 điểm) mới là tín hiệu thật.

---

## Run #18 — ablation 1 frame: fusion đáng **+20,66 điểm**

```bash
python -u train.py -n restran_30ep_1frame ... --single-frame
```

| | Accuracy | Mercosur | Brazilian | val_loss |
|---|---|---|---|---|
| **1 frame** | **56,53%** | 61,73% | 36,05% | 0,3753 |
| 5 frame (run #9) | 77,19% | 83,06% | 54,07% | 0,2342 |
| Chênh | **+20,66** | +21,33 | +18,02 | −0,141 |

Dự đoán của tôi là 72–76% (fusion đáng 1–5 điểm) — **sai hoàn toàn**. Tôi suy từ "bề rộng chỉ dao động 6–9,5% giữa các frame", nhưng đó là nhầm *thay đổi tỉ lệ* với *đa dạng lấy mẫu pixel*.

Đối xứng đáng chú ý: **fusion đa khung lấy lại đúng một nửa khoảng cách tới trần HR** (+20,66 trên tổng 41,92 điểm từ 56,53% lên 98,45%). 5 frame chứa thông tin bổ sung thật sự.

---

## Run #19 — SR front-end: ❌ −5,10 điểm

```bash
python -u train.py -n restran_30ep_sr ... --sr
```

| | Run #9 (mốc) | Run #19 (SR) |
|---|---|---|
| **Tổng** | **77,19%** | **72,09%** (−5,10) |
| Mercosur / Brazilian | 83,06 / 54,07 | 78,67 / 46,17 |
| val_loss thấp nhất | **0,2342** | 0,2510 |
| train_loss (CTC) cuối | **0,0330** | 0,0941 |
| grad_norm cuối | 1,64 | 4,70 |
| VRAM | 1.555 MB | 1.835 MB |

### ⚠️ Lỗi thiết kế thí nghiệm của tôi

Run này đổi **hai** biến cùng lúc: thêm module SR **và** đổi kích thước đầu vào 32×128 → 16×64. Nên không tách được "SR có hại" khỏi "thu nhỏ đầu vào có hại". Đúng cái lỗi tôi đã phê phán ở các bước trước.

### Và control hiển nhiên lại không dùng được

Chạy 16×64 **không có SR** để so sánh thì không hợp lệ: bề rộng 64 cho **T=8 timestep** cho nhãn 7 ký tự — CTC gần như không hoạt động được. Nên bản thân module SR đang gánh luôn vai trò tạo đủ chiều dài chuỗi, không chỉ là "nâng cấp ảnh".

| Đầu vào | T |
|---|---|
| 16×64 không SR | **8** |
| 32×128 không SR | 16 |
| 16×64 + SR ×2 | 16 |

### Control đúng

Cho SR chạy ở **32×128 không đổi kích thước** (scale = 1). Khi đó biến duy nhất so với mốc là *có thêm một module xử lý chéo-frame ở mức pixel* — đúng câu hỏi cần trả lời, không lẫn chuyện độ phân giải hay chiều dài chuỗi.

Ghi nhận thêm: `train_loss` 0,0941 so với mốc 0,0330 và grad_norm 4,70 so với 1,64 → model **chưa hội tụ**, module SR 0,213M tham số phải học phép nâng cấp từ đầu chỉ bằng tín hiệu CTC gián tiếp.

---

## Run #20 — control SR scale=1: ❌ −2,85 điểm

```bash
python -u train.py -n restran_30ep_sr_scale1 ... --sr --sr-scale 1
```

Biến duy nhất so với mốc: thêm module trộn chéo-frame ở mức pixel. Độ phân giải 32×128, T=16, dataset y hệt.

| | Mốc (#9) | SR scale=1 (#20) | SR scale=2 (#19) |
|---|---|---|---|
| **Tổng** | **77,19%** | 74,34% (−2,85) | 72,09% (−5,10) |
| Mercosur / Brazilian | 83,06 / 54,07 | 80,11 / 51,60 | 78,67 / 46,17 |
| val_loss thấp nhất | **0,2342** | 0,2452 | 0,2510 |
| train_loss (CTC) cuối | **0,0330** | 0,0565 | 0,0941 |

**Phân tách được nguyên nhân của −5,10:**
- Bản thân module SR: **−2,85**
- Thu nhỏ đầu vào xuống 16×64: **−2,25**

Cả hai đều hại, mức tương đương nhau.

### Một khuyết điểm thiết kế trong module tôi viết

`StackedSRNet` **không có đường tắt từ ảnh vào tới ảnh ra**:

```python
feat = self.head(x)
feat = self.body(feat) + feat     # residual chỉ trong không gian đặc trưng
feat = self.upsample_conv(self.upsampler(feat))
out  = self.tail(feat)            # 32 kênh -> 15 kênh, khởi tạo ngẫu nhiên
```

Lúc khởi tạo, `tail` là ngẫu nhiên nên **đầu ra gần như nhiễu** — module phá ảnh trước khi học được cách không phá. Toàn bộ ResNet + Transformer phía sau phải học trên ảnh hỏng rồi mới dần hồi phục.

Đây là thiết kế EDSR gốc, nhưng EDSR được train bằng **loss pixel** để tái tạo trực tiếp. Ở đây ta chỉ có tín hiệu CTC gián tiếp qua cả mạng, yếu hơn nhiều — nên xuất phát từ ánh xạ đồng nhất là điều kiện gần như bắt buộc.

Khớp với dữ liệu: train_loss xếp đúng theo mức độ "module phải làm nhiều": mốc 0,0330 → scale=1 0,0565 → scale=2 0,0941.

**Cách sửa:** thêm `out = out + x` (với scale=2 thì cộng vào bản upsample bilinear của `x`) — đúng công thức chuẩn "SR = phần dư học được cộng lên phép nội suy". Module khi đó khởi động **bằng đúng mốc** và chỉ học phần cải thiện.

---

## Run #21 — SR + identity-skip: chẩn đoán ĐÚNG, nhưng vẫn không thắng mốc

```bash
python -u train.py -n restran_30ep_sr_skip ... --sr --sr-scale 1
```

| | Mốc (#9) | #19 SR sc=2 | #20 SR sc=1 | **#21 SR sc=1 + skip** |
|---|---|---|---|---|
| **Tổng** | **77,19%** | 72,09% | 74,34% | **76,89%** |
| Mercosur / Brazilian | 83,06 / 54,07 | 78,67 / 46,17 | 80,11 / 51,60 | 82,81 / 53,58 |
| val_loss thấp nhất | 0,2342 | 0,2510 | 0,2452 | **0,2234** |
| train_loss (CTC) cuối | **0,0330** | 0,0941 | 0,0565 | **0,0316** |

**Chẩn đoán identity-skip là đúng.** `train_loss` từ 0,0565 về **0,0316 — ngang/dưới mốc**, vấn đề hội tụ đã hết. Accuracy hồi từ 74,34% lên 76,89% (+2,55). Và `val_loss 0,2234` **tốt hơn mốc 4,6%**.

Nhưng accuracy vẫn **−0,30 so với mốc**, tức nằm gọn trong nhiễu. Rơi vào ô "75–79%" đã định trước: **module không thêm gì**.

Diễn tiến cả nhánh SR nói rõ bản chất: mỗi lần sửa một khuyết điểm thì lấy lại được đất (72,09 → 74,34 → 76,89) nhưng **chưa lần nào vượt mốc**. Module đang hội tụ về chỗ tái tạo lại đúng những gì baseline vốn làm.

→ **Đóng hướng Super Resolution**, theo đúng cam kết đặt ra trước khi chạy.

---

## Bảng tổng kết 21 thí nghiệm

| Hạng | Cấu hình | Accuracy | val_loss |
|---|---|---|---|
| 1 | **#13 augment nhất quán** | 77,59% | **0,2111** |
| 2 | #14 augment + 40 epoch | 77,59% | 0,2159 |
| 3 | #16 độ phân giải 32×192 | 77,59% | 0,2346 |
| 4 | **#9 mốc** | 77,19% | 0,2342 |
| 5 | #21 SR + identity-skip | 76,89% | 0,2234 |
| 6 | #17 distillation | 76,29% | 0,2349 |
| 7 | #12 pretrained | 74,39% | 0,2852 |
| 8 | #20 SR scale=1 | 74,34% | 0,2452 |
| 9 | #19 SR scale=2 | 72,09% | 0,2510 |
| — | #18 một frame *(ablation)* | 56,53% | 0,3753 |
| — | #15 trần HR *(đo lường)* | **98,45%** | 0,0216 |

**Không cải tiến kiến trúc nào thắng được mốc.** Bốn cấu hình đầu bảng chênh nhau 0,40 điểm — dưới sai số ±0,94 và dưới cả biên chọn lọc từ 21 lần thử.

Hai kết quả có giá trị thật đều là **đo lường**, không phải cải tiến:
- **Trần HR 98,45%** → kiến trúc không phải nút thắt, chất lượng ảnh mới là.
- **Fusion đáng +20,66 điểm** → 5 frame chứa thông tin bổ sung thật, và fusion mức đặc trưng đã khai thác gần hết.

### Chọn cấu hình để nộp bài

Accuracy giữa 4 cấu hình đầu là nhiễu, nên phải dùng tiêu chí phương sai thấp hơn: **val_loss**. Run #13 thắng rõ (0,2111 so với 0,2342 của mốc), nên đó là cấu hình nên chốt:

```
--consistent-aug --no-channel-shuffle --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100
```

---

## Run #22 — EDVR-lite (PCD alignment + TSA fusion)

```bash
python -u train.py -n restran_30ep_edvr ... --sr --sr-arch edvr
```

| | Accuracy | Mercosur | Brazilian | val_loss | train_loss |
|---|---|---|---|---|---|
| Không fusion (1 frame, #18) | 56,53% | 61,73 | 36,05 | 0,3753 | — |
| **EDVR, pixel + căn chỉnh (#22)** | **74,89%** | 80,55 | 52,59 | 0,2474 | 0,0501 |
| AttentionFusion, đặc trưng (#9) | **77,19%** | 83,06 | 54,07 | 0,2342 | 0,0330 |
| Trần HR (#15) | 98,45% | 99,44 | 94,57 | 0,0216 | — |

45,0 phút (84,5 s/epoch, chậm 1,8× do deformable conv).

### Kết quả rơi vào ô "tinh tế" đã dự kiến

Căn chỉnh tường minh **hoạt động rõ ràng**: +18,36 điểm so với không fusion (56,53 → 74,89), tức đạt **89%** hiệu quả của AttentionFusion (+20,66). Module học được thật, không phải hỏng.

Nhưng nó **không vượt** fusion ngầm mức đặc trưng. Với 5 frame và biên độ dịch chuyển nhỏ của dataset này, attention theo từng pixel đã đủ; alignment tường minh tốn 0,48M tham số và 1,8× thời gian mà không lấy thêm được gì.

### ⚠️ So sánh với run #19 bị nhiễu biến

Cám dỗ là kết luận "EDVR (74,89%) hơn stacked (72,09%) 2,80 điểm → alignment thắng early-fusion". **Không hợp lệ**: run #19 chạy *trước* khi có identity-skip, còn `EDVRLite` có sẵn từ đầu. Hai run khác nhau hai biến.

Phép so sánh sạch còn thiếu: **EDVR scale=1 + skip** đối chiếu với **stacked scale=1 + skip (76,89%, run #21)** — cùng 32×128, cùng identity-skip, chỉ khác cơ chế hợp nhất.

---

## Kết luận rút ra

1. **Cấu hình dùng được: 30 epoch, `--lr 3e-4`, `--grad-clip 1.0`, fp16.** Cho 76,68% và 77,18% qua hai lần chạy.
2. **Vùng nguy hiểm đo được: `loss ≲ 0,165` khi `LR ≳ 2,85e-4`.** 4/4 run vào vùng này đều nổ, 2/2 run tránh được đều chạy trọn. Muốn 60 epoch thì `--lr` phải ≤ **2e-4**.
3. **Nguyên nhân KHÔNG phải tràn số fp16** — 96–97% batch bị bỏ có gradient hữu hạn (run #7, #8). bf16 không cứu được. Cũng không phải dữ liệu bẩn: `train_loss_max` ở epoch nổ chỉ 0,71–0,96.
4. **Không dùng bf16** trên máy này (chậm 38× do cudnn deterministic) — và giờ đã biết nó cũng không giải quyết được vấn đề gì.
5. **`--grad-clip 5.0` mặc định là quá lỏng** cho CTC: giai đoạn khoẻ grad norm chỉ 1,6–14 nên ngưỡng 5.0 gần như không ràng buộc gì.
6. **Nhiễu run-to-run là 0,5–1,3 điểm** (đo trên 3 cặp run cùng cấu hình: 76,68 vs 77,18 · 67,97 vs 67,47 · 65,87 vs 67,17). `seed_everything` không gọi `torch.use_deterministic_algorithms(True)`, mà `F.grid_sample` (backward của STN) và CTC backward dùng atomics. **Chênh lệch dưới ~1,5 điểm giữa hai cấu hình là không kết luận được.**

---

## Đổi tập val: 999 → 1999 track (13/08/2026)

Sai số chuẩn của accuracy trên 999 mẫu ở mức ~77% là **±1,33 điểm (1σ)**, nên mọi cải tiến dưới ~1,5 điểm đều chìm trong nhiễu — đúng bằng biên độ dao động giữa các run cùng cấu hình. Tăng val lên 1999 mẫu giảm sai số còn **±0,94 điểm**.

```bash
python -u train.py -n restran_30ep_val2000 --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    2>&1 | tee results/restran_30ep_val2000_train.log
```

Đã kiểm chứng bằng dữ liệu thật:

| | Cũ (`SPLIT_RATIO=0.9`) | Mới (`0.8`) |
|---|---|---|
| Val | 999 track | **1999 track** |
| Train | 19.001 track (38.002 mẫu) | 18.001 track (36.002 mẫu) |
| Sai số accuracy (1σ, ở 77%) | ±1,33 điểm | **±0,94 điểm** |

`_load_or_create_split` xáo Scenario-B bằng `random.Random(42)` rồi lấy `[:val_size]`, nên **val 1999 bao trùm trọn vẹn val 999 cũ** (đã xác minh: 999/999 track cũ nằm trong tập mới, thêm đúng 1000 track).

### ⚠️ Kết quả trước và sau KHÔNG so sánh trực tiếp được

1000 track vừa chuyển sang val **đã nằm trong tập train của mọi run #1–#8**. Do đó:
- Không được đánh giá checkpoint cũ (77,18%) trên val 1999 — 1000 track trong đó model đã học thuộc, con số sẽ bị thổi phồng.
- Muốn đối chiếu với 77,18%, hãy tính accuracy của model mới **trên đúng tập con 999 track cũ**; lúc đó chỉ còn khác biệt do train ít hơn 1000 track (−5%).

### Hai cờ CLI mới

| Cờ | Mặc định | Ghi chú |
|---|---|---|
| `--split-ratio` | 0.9 | **Chỉ có tác dụng khi file split chưa tồn tại** |
| `--val-split-file` | `splits/val_tracks.json` | Trỏ sang đường dẫn mới để tạo split khác mà không đụng file cũ |

Cái bẫy đã được vá: trước đây nếu `splits/val_tracks.json` còn đó thì `_load_or_create_split` nạp thẳng file cũ và **bỏ qua `split_ratio` trong im lặng** ([dataset.py:116](src/data/dataset.py#L116)). Nay có cảnh báo:

```
⚠️ CẢNH BÁO: 'splits/val_tracks.json' cho val 999 track, nhưng split_ratio=0.8 ứng với 1999 track.
   File split có sẵn được ưu tiên -> split_ratio bị BỎ QUA.
```

---

## Run #9 — val 1999, tập val lớn (13/08/2026)

```bash
python -u train.py -n restran_30ep_val2000 --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    2>&1 | tee results/restran_30ep_val2000_train.log
```

| | |
|---|---|
| Best val acc | **77,1886%** tại epoch 28 (trên **1999** mẫu) |
| val_loss thấp nhất | 0,2342 tại ep26 |
| Batch bị bỏ | 8 trong toàn bộ run, **cả 8 đều inf/nan** (hành vi bình thường của GradScaler) |
| Thời gian | 25,6 phút · 563 batch/epoch |
| Sai số đo | **±0,94 điểm** (trước là ±1,33) |

Chạy sạch, không một dấu hiệu bất ổn định: grad_norm giảm đều 6,0 → 1,66, `grad_norm_max` không lần nào vượt 39.

**77,19% (val 1999, train 18.001) so với 77,18% (val 999, train 19.001)** — gần như trùng khít, dù cả tập train lẫn tập val đều đổi. Bớt 1.000 track train (−5%) không gây thiệt hại đo được.

> Máy này là **Tesla V100 (sm_70, Volta)**. CUDA 13 đã bỏ Volta nên phải dùng build ≤ cu126. Đây cũng là lời giải cho vụ bf16 chậm 38× ở run #2: Volta **không có phần cứng bf16**, nó bị mô phỏng bằng phần mềm.

---

## Phân tích chênh lệch layout (13/08/2026)

Đo trên 999 mẫu val bằng checkpoint 76,68%:

| | Accuracy | n |
|---|---|---|
| Mercosur | **81,50%** | 789 |
| Brazilian | **58,57%** | 210 |

Tỉ lệ lỗi theo từng vị trí ký tự:

| Vị trí | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Mercosur | 3,30% | 6,73% | 7,87% | 3,81% | 2,66% | 4,70% | 3,30% |
| Brazilian | 6,22% | 14,83% | 16,27% | 9,57% | 10,53% | 8,61% | 10,53% |
| Tỉ lệ | 1,9× | 2,2× | 2,1× | 2,5× | 4,0× | 1,8× | 3,2× |

**Không phải do pattern chuỗi**: vị trí 1–3 là chữ cái ở cả hai layout, cùng loại ký tự cùng vị trí, mà Brazilian vẫn sai gấp ~2×. Phần lớn ca hỏng chỉ sai **1 ký tự** (Mercosur 89/145, Brazilian 47/87).

Hai nguyên nhân cộng dồn, đo được:

| | Mercosur | Brazilian |
|---|---|---|
| Tỉ lệ trong train | **63,4%** (11.406) | **36,6%** (6.595) |
| Kích thước LR | 48,6 × 17,9 px | 49,6 × 18,1 px |
| Độ nét (Laplacian var) | 5002 | **4371 (−13%)** |
| Tương phản (std) | 58,7 | **51,3 (−13%)** |

Ảnh Brazilian **không nhỏ hơn** nhưng **mờ hơn và nhạt hơn ~13%**, đồng thời **ít hơn 1,7 lần** trong tập train. Ít mẫu hơn cho một lớp khó hơn.

> ⚠️ Lưu ý khi cân bằng: test set có cùng phân phối với Scenario-B, tức chỉ ~20% Brazilian. Cân bằng train về 50/50 là tối ưu cho một phân phối **không khớp** với test. Phải theo dõi accuracy **cả hai layout**, vì mất 1 điểm ở Mercosur đã tốn 0,8 điểm tổng.

### Đo lại trên checkpoint run #9 (1999 mẫu, thống kê tốt hơn)

| | Accuracy | n | Sai số |
|---|---|---|---|
| Mercosur | **83,06%** | 1594 | ±0,94 |
| Brazilian | **54,07%** | 405 | ±2,48 |
| Tổng | 77,19% | 1999 | ±0,94 |

**Khoảng cách 29 điểm** — rộng hơn con số 23 điểm đo trên 999 mẫu trước đó. Nếu kéo Brazilian lên ngang Mercosur thì tổng thể thành 83,06%, tức **+5,87 điểm**.

Kiểm tra tính liên tục giữa các run:

| Đo trên | Run #6 (train 19.001) | Run #9 (train 18.001) |
|---|---|---|
| Tập con 999 cũ | 77,18% | **77,08%** |
| 1000 track mới | — | 77,30% |

Chênh 0,10 điểm trên cùng một tập val → **bớt 1.000 track train không gây thiệt hại đo được**. Confidence Gap của run #9: 0,0951.

---

## Bước 1 + 2 đã implement (13/08/2026)

**Accuracy theo layout vào thẳng CSV và log.** Hai cột mới `val_acc_mercosur`, `val_acc_brazilian`; dòng log mỗi epoch nay là `Val Acc: 77.19% (M 83.06% / B 54.07%)`. Không có nó thì mọi thí nghiệm cân bằng đều mù — chỉ thấy con số tổng mà không biết đang đánh đổi cái gì.

**Cờ `--layout-balance`** dùng `WeightedRandomSampler`, giữ nguyên số batch mỗi epoch nên thời gian chạy không đổi. Tỉ lệ thực tế đo bằng mô phỏng trên đúng thành phần train (6.595 Brazilian / 11.406 Mercosur):

| `--layout-balance` | Brazilian trong luồng mẫu |
|---|---|
| 1.0 (mặc định, tắt) | 36,1% |
| 1.3 | 42,7% |
| **1.73** | **49,6%** (cân bằng hoàn toàn) |

```bash
python -u train.py -n restran_30ep_balance173 --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --layout-balance 1.73 \
    2>&1 | tee results/restran_30ep_balance173_train.log
```

---

## Run #23 → #38 (14/08 → 30/09/2026)

Cờ chung của mọi run trong bảng (trừ khi ghi khác): `--epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json` (val 1999 track, SE ±0,94). **TB6** = trung bình 6 epoch cuối. Con số "best epoch" thường cao hơn TB6 khoảng 0,5–0,9 điểm vì được chọn trên chính tập val.

| # | Tên run | Cờ thêm | Best (ep) | TB6 | M / B | val_loss | Kết cục |
|---|---|---|---|---|---|---|---|
| 23 | `restran_30ep_edvr_s1` | `--sr --sr-arch edvr --sr-scale 1` | 72,24 (26) | 71,80 | 78,42 / 47,90 | 0,2826 | ❌ |
| 24 | `restran_30ep_sr_pixel` | `--sr --sr-scale 1 --sr-pixel-weight 0.1 --sr-ncc-min 0.8` | 75,99 (30) | 75,57 | 81,87 / 52,84 | 0,2384 | ❌ |
| 25 | `restran_30ep_aug_recheck` | chạy lại #13 (`--consistent-aug --no-channel-shuffle`) | 77,04 (29) | 76,60 | 82,81 / 54,32 | 0,2116 | đo nhiễu: #13 77,59 → 77,04 |
| 26 | `restran_30ep_letterbox` | `--letterbox --img-height 48` | 77,14 (28) | 76,66 | 83,19 / 53,33 | 0,2442 | ≈ mốc |
| 27 | `restran_30ep_srsup` | `--sr --sr-scale 1 --sr-init results/sr_pretrained.pth --sr-pixel-weight 0.1 --sr-ncc-min 0.75` | 71,04 (20) | — | 77,04 / 47,41 | 0,2486 | ❌ nổ ep21 |
| 28 | `restran_30ep_srsup_v2` | như #27, bỏ pixel loss | 76,99 (28) | 76,36 | 82,25 / 56,30 | 0,2324 | ≈ mốc |
| 29 | `restran_aug_ema` | #13 + `--ema-decay 0.999` | 77,64 (29) | 77,31 | 82,87 / 57,04 | 0,2115 | EMA 77,59: vô dụng khi OneCycle đưa LR về 0 |
| 30 | `restran_stnshared` | `--stn-shared` | 76,34 (30) | 75,64 | 82,06 / 53,83 | 0,2332 | ❌ |
| 31 | `restran_fusion2d` | `--fusion-2d` | 76,69 (28) | 76,19 | 82,12 / 55,31 | 0,2329 | ❌ (lần chạy đầu nổ ep22, 70,79) |
| 32 | `restran_pseudo` | #13 + pseudo-label từ blind test | 78,39 (30) | 77,78 | 83,88 / 56,79 | 0,2049 | ⛔ loại: dùng dữ liệu test |
| 33 | `restran_cwta` | #13 + `--sharpness-prior` | 76,69 (30) | 76,16 | 82,43 / 54,07 | 0,2051 | ❌ |
| 34 | `restran_degstrong` | #13 + `--degrade-strength strong` | 72,94 (21) | — | 79,30 / 47,90 | 0,2231 | ❌ nổ ep22 |
| 35 | `restran_final` | #13 + pseudo + sharpness + strong | 77,04 (26) | 76,66 | 82,69 / 54,81 | 0,1994 | ⛔ loại (pseudo) |
| 36 | `restran_augdeg_light` | #13 + `--aug-degrade light` | 77,24 (29) | 76,71 | 82,62 / 56,05 | 0,2062 | ≈ mốc |
| 37 | `restran_augdeg_medium` | #13 + `--aug-degrade medium` | 76,94 (27) | 76,71 | 82,00 / 57,04 | 0,2021 | ≈ mốc |
| 38 | **`restran_degcal`** | #13 + `--degrade-strength calibrated` | **77,94 (29)** | **77,30** | 83,63 / 55,56 | 0,2106 | ✅ **model đơn tốt nhất (giám sát thuần)** |

Lệnh của #38 — **cấu hình gốc cho mọi run sau**:

```bash
python -u train.py -n restran_degcal --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    2>&1 | tee results/restran_degcal_train.log
```

Rút ra từ 16 run:
- Cải thiện thật duy nhất đến từ **dữ liệu**: sửa `GaussNoise` mặc định của nhánh synthetic (làm lệch pixel ~52/255) cho #38. Không thay đổi kiến trúc nào thắng được mốc.
- Toàn bộ nhánh SR/alignment (#19–#24, #27–#28) đều thua hoặc ngang mốc. Nguyên nhân đã xác định được: (1) SR chỉ học qua gradient CTC; (2) target HR đăng ký ECC là một ảnh chụp khác (NCC ~0,70), nên target bị nhiễu; (3) ở #27, input được augment còn target thì không, tức bài toán vô nghiệm, và run nổ.
- #32 và #35 dùng pseudo-label từ blind test nên bị loại khỏi so sánh công bằng với baseline.

---

## Phân tích 01/10/2026 — đo trực tiếp trên checkpoint (máy local)

### 1. Decode không phải nút thắt: constrained decoding +0,00

Beam search ràng buộc layout (`LLLNLNN` / `LLLNNNN`, beam 16), áp lên đúng log-prob của model:

| Checkpoint | Greedy | Constrained | Δ |
|---|---|---|---|
| #9 `restran_30ep_val2000` | 77,19 | 77,24 | +0,05 |
| #29 `restran_aug_ema` | 77,64 | 77,64 | +0,00 |

- Trong 456 lỗi của #9: **446 lỗi có đúng 7 ký tự, và 0 lỗi nào sai layout**. Model đã tự học layout.
- **Ở 456/456 lỗi, chuỗi GT có xác suất CTC *thấp hơn* chuỗi model chọn**, nên không cách decode nào sửa được.
- Lỗi là nhầm ký tự **cùng lớp**: 8↔6 (46), V↔Y, 3↔1, O↔Q, D↔B, M↔H, A↔B. 57% lỗi chỉ sai 1 ký tự. Vị trí 2–3 (chữ cái thứ 2–3) sai nhiều nhất.

→ Bỏ hướng decode. Repo của bạn cũng chỉ đo được +0,20 nhờ decode.

### 2. Split 999 nằm trong 1999 và trùng split của BTC/repo bạn

- Sinh lại split bằng code gốc (seed 42, ratio 0.9) ra **đúng 999 track** của `splits/val_tracks.json`, và cả 999 nằm trong tập 1999. Repo của bạn và slide 78,70% đều dùng logic này, nên số trên 999 so sánh được trực tiếp.
- Cùng một model, kết quả trên 999 so với 1999: 77,08 vs 77,19 (#9) và 77,78 vs 77,64 (#29). Chênh dưới 0,15.
- Cờ mới `--val-subset-file splits/val_tracks.json` ghi cột `val_acc_sub` (accuracy trên 999) mỗi epoch.

### 3. Domain gap của nhánh synthetic: model đọc ảnh synthetic dễ hơn ảnh thật 20 điểm

Model #29, 600 track val, lấy HR của chính các track đó rồi tạo ảnh synthetic theo từng công thức:

| Ảnh đầu vào | Accuracy | LapVar median |
|---|---|---|
| **LR thật** | **78,33%** | 340 |
| HR sạch | 98,67% | 3939 |
| synthetic `normal` | 97,83% | 4027 |
| synthetic `calibrated` (#38 đang dùng) | 98,33% | 2392 |
| HR → hạ về đúng cỡ LR thật (INTER_AREA) | 98,00% | 905 |
| HR → cỡ LR thật + blur 3px + noise + JPEG 30–65 (`native`) | 93,17% | 266 |

Một nửa mỗi epoch là ảnh model đọc đúng ~98%, trong khi ảnh được chấm điểm chỉ ~78%. Riêng việc hạ về đúng số pixel (98,0%) không làm ảnh khó lên; phải degrade **ở cỡ LR gốc** thì mới khó lên rõ (93,2%).

### 4. Repo của bạn (S1 79,78%) và cuộc thi ICPR 2026 LRLPR

- **S1** (CRNN + STN + DCN + MF-SR, 60 epoch warmup-cosine, EMA, constrained decode): best 79,78% ở ep37/55, greedy 79,58%, epoch cuối 78,88%, TB6 ≈ 79,06. Đo trên **cùng 999 track**. Có 6 thay đổi gộp trong 1 run, 1 seed, nên không tách được đóng góp riêng; repo chỉ chứng minh SR học hơn bilinear (`sr_loss` 0,217 so với 0,257).
- Đường cong S1: ep24 đã đạt 79,08, sau ep21 val_loss tăng dần. Tức 60 epoch chỉ thêm khoảng 0,3–0,7 so với 30, vẫn trong vùng nhiễu.
- Cuộc thi ([arXiv 2604.22506](https://arxiv.org/abs/2604.22506)), blind test: hạng 1 82,13% (SR + OCR teacher-student, data ngoài, ensemble 3); **hạng 2 81,73% dùng đúng họ kiến trúc của ta** (STN + SE-ResNet34 + Transformer + CTC + attention fusion, 30 epoch OneCycle, thêm auxiliary CTC/center loss/OHEM, ensemble 4 + ràng buộc layout); hạng 3 80,17% **không dùng SR**. Bài tổng kết kết luận không có thiết kế nào áp đảo; điểm chung của các đội mạnh là khai thác tốt 5 frame.

---

## Thay đổi code 01/10/2026 — mọi cờ mặc định TẮT

| Cờ | Mặc định | Tác dụng |
|---|---|---|
| `--syn-pipeline {legacy,native}` | legacy | `native`: augment hình học HR **một lần** trên canvas đích → hạ về đúng cỡ frame LR thật cùng chỉ số → blur/noise/JPEG ở cỡ đó → jitter màu dùng chung cho input và target → resize về input. Không áp CoarseDropout/ChannelShuffle cho mẫu synthetic. Khi bật, `--degrade-strength` không còn tác dụng lên mẫu synthetic |
| `--sr-syn-weight λ` | 0 | L1 giữa output SR và HR target **khớp pixel** của mẫu synthetic native. Cần `--sr --sr-arch stacked --syn-pipeline native` |
| `--aux-frame-ctc λ` | 0 | CTC phụ cho **từng frame đọc riêng** (dùng chung Transformer + head) |
| `--scheduler {onecycle,cosine}` | onecycle | `cosine`: warmup tuyến tính `--warmup-epochs` (3), rồi cosine giảm về sàn `--min-lr-ratio` (0,05) × lr |
| `--patience N` | 0 | Dừng khi N epoch liền không vượt best |
| `--val-subset-file F` | — | Ghi thêm `val_acc_sub` = accuracy trên các track trong F |
| `--relative-dropout` | tắt | Lỗ CoarseDropout = 1/8–1/4 chiều cao input thay vì cố định 4–8px. Ở 32×128 giữ nguyên như cũ; ở input 16×64 của các run SR, lỗ cố định che **gấp 4 lần** diện tích (~12% so với ~3%), nên hai run SR ×2 cũ (#19, #22, input 16×64) đã bị augment nặng hơn baseline; các run `--sr-scale 1` dùng 32×128 nên không bị |

Cột CSV mới: `sr_syn_loss`, `sr_syn_bilinear` (L1 của bilinear thuần trên cùng target; SR chỉ có ích khi `sr_syn_loss` < cột này), `aux_ctc_loss`, `val_acc_sub`.

Đã kiểm tra: (a) model sau refactor cho output **giống hệt** bản cũ trên checkpoint #29 (sai khác tối đa 0,0); (b) cặp native khớp pixel: NCC 0,861 khi đúng vị trí, 0,653 khi lệch 2px; (c) mỗi cấu hình train thử 4 batch và validate trên CPU đều chạy được, ở bước 0 SR có loss đúng bằng bilinear (0,219 = 0,219) như thiết kế; (d) `val_acc_sub` của #29 tính ra 77,78, khớp phép đo độc lập.

### Vòng thí nghiệm E1–E4 (mỗi run chỉ khác #38 đúng một yếu tố)

Nếu chạy cả 4 song song, thay `$(nproc)` bằng `$(( $(nproc) / 4 ))` để các DataLoader không tranh CPU.

```bash
# E1 — synthetic degrade ở cỡ LR gốc (dữ liệu)
python -u train.py -n restran_native --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --syn-pipeline native \
    2>&1 | tee results/restran_native_train.log

# E2 — E1 + SR đa frame có giám sát pixel trên cặp synthetic (so với E1)
python -u train.py -n restran_native_srsyn --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --syn-pipeline native --sr --sr-arch stacked --sr-scale 2 --sr-syn-weight 0.1 \
    --relative-dropout \
    2>&1 | tee results/restran_native_srsyn_train.log

# E3 — lịch dài: cosine 60 epoch, LR 1.5e-4, EMA, early stopping (so với #38)
python -u train.py -n restran_cos60 --num-workers $(nproc) \
    --epochs 60 --lr 1.5e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --scheduler cosine --warmup-epochs 3 --min-lr-ratio 0.05 \
    --ema-decay 0.999 --patience 15 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    2>&1 | tee results/restran_cos60_train.log

# E4 — auxiliary CTC cho từng frame (so với #38)
python -u train.py -n restran_auxframe --num-workers $(nproc) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --aux-frame-ctc 0.2 \
    2>&1 | tee results/restran_auxframe_train.log
```

Tiêu chí đặt trước khi chạy (1 seed, nhiễu giữa các lần chạy đo được 0,5–1,3):
- Hơn #38 ≥ 0,8 điểm ở **TB6** → giữ và chạy thêm seed. Chênh dưới mức đó → chưa kết luận được.
- E2: ngoài accuracy, SR chỉ được tính là "học được" khi `sr_syn_loss` thấp hơn hẳn `sr_syn_bilinear` và khoảng cách nới rộng theo epoch.
- E3: nếu TB6 không hơn #38 ≥ 0,5 → kết luận 30 epoch là đủ.

---

## Run #39 → #42 — vòng E1–E4 (06/10/2026, server `fabulous-mongoose`, 4 run song song)

Lệnh đúng như mục "Vòng thí nghiệm E1–E4" ở trên, `--num-workers $(( $(nproc) / 4 ))`. Do 4 run chia CPU nạp dữ liệu nên mỗi epoch mất 244–279 s (#38 chạy một mình chỉ 83 s).

| # | Tên run | Best (ep) | TB6 | Δ TB6 so với #38 | 999: @best / TB6 | M / B @best | val_loss @best | train_loss cuối | Kết cục |
|---|---|---|---|---|---|---|---|---|---|
| 38 | `restran_degcal` (mốc) | 77,94 (29) | 77,30 | — | — | 83,63 / 55,56 | 0,2106 | 0,0483 | mốc |
| 39 | `restran_native` | 78,34 (28) | 77,36 | +0,06 | 78,18 / 77,16 | 84,07 / 55,80 | 0,2041 | 0,0574 | ≈ mốc |
| 40 | `restran_native_srsyn` | 76,44 (29) | 76,00 | −1,30 (−1,36 so với #39) | 76,08 / 75,41 | 81,87 / 55,06 | 0,2251 | 0,0463 | ❌ |
| 41 | `restran_cos60` | 77,24 (35) | 76,45 | −0,85 | 76,38 / 75,89 | 83,06 / 54,32 | 0,2205 | 0,0309 (ep46) | ❌ (CSV dừng ở ep46, chưa tới điều kiện patience 35+15=50) |
| 42 | `restran_auxframe` | 77,74 (25) | **77,53** | +0,23 | **78,28** / **77,79** | 83,19 / 56,30 | 0,2152 | 0,0323 | ≈ mốc, hội tụ nhanh hơn |

**Độ nhiễu giữa các lần chạy** (3 run cùng cấu hình #13/#25/#29): best 77,59 / 77,04 / 77,64 (std 0,33); TB6 76,75 / 76,60 / 77,31 (std 0,37). Hiệu của 2 run đơn có độ lệch ~0,52, nên mọi chênh lệch dưới ~1 điểm giữa hai run đơn đều nằm trong nhiễu.

Rút ra:
- **Không run nào đạt tiêu chí +0,8 TB6.** 78,34% của #39 là epoch tốt nhất trong 30 epoch; trung bình 6 epoch cuối chỉ hơn #38 0,06.
- **30 epoch là đủ (câu hỏi đã đóng).** Cosine 60 epoch thua 0,85. val_loss thấp nhất ở ep19 (0,196) rồi tăng đều lên 0,2465, train_loss xuống 0,031, tức là overfit. Accuracy đi ngang 76,3–77,2 từ ep25. Kết hợp với #14 (40 epoch = 30 epoch), kéo dài lịch train không giúp.
- **SR đa frame có giám sát đúng: dựng ảnh tốt hơn nhưng đọc kém hơn.** `sr_syn_loss` thua bilinear trong ep3–15 (ep8 tệ hơn 21%, do CTC kéo SR lệch khỏi HR), rồi thắng dần, tới ep30 thấp hơn bilinear 9,3%. Dù vậy accuracy thấp hơn đúng pipeline đó không có SR 1,36 điểm. Đây là kết quả ablation sạch cho bài báo: độ trung thực pixel không chuyển thành khả năng đọc.
- **Synthetic khó hơn (native) không làm tăng accuracy**, dù đo được nó thu hẹp domain gap (98,3 → 93,2%). val_loss thấp nhất trong mọi run (0,2041) và train_loss cao hơn, tức là có tác dụng điều chuẩn. Độ khó của nửa synthetic không phải nút thắt.
- **Aux CTC từng frame giúp học nhanh hơn rõ rệt** ở giữa quá trình train (ep15: 69,73 so với 62,38; ep23: 76,79 so với 73,69), nhưng cuối cùng chỉ hơn +0,23 TB6 (trong nhiễu) và overfit hơn: train_loss 0,032 so với 0,048, val_loss tăng từ 0,2152 lên 0,2219.

### Vòng tiếp theo: seed + giao thức chính thức (999)

78,70% của BTC và 79,78% của bạn đều đo theo giao thức split 0.9: train 19.001 track, val 999. Ta đang train trên 18.001 track (giữ 1999 cho val), nghĩa là thiếu 1.000 track Scenario-B, đúng phân bố của tập test. Con số cuối cho bài báo phải đo theo đúng giao thức của mốc so sánh, dạng mean ± std của 3 seed cho cả baseline lẫn phương pháp.

**Cập nhật 06/10: thứ tự ưu tiên.** Sửa phương pháp trước; seed của phương pháp để cuối, vì seed một cấu hình còn sắp đổi là phí. Seed **baseline** thì chạy ngay: nó độc lập với mọi thay đổi sau này và bài báo kiểu gì cũng cần. Chỉ thử những thay đổi có thể mang lại ≥ 1 điểm, vì nhiễu một run là ~0,37 (TB6) nên hiệu ứng nhỏ không đo được. Vì vậy bỏ `native_aux`: cả hai thành phần đều nằm trong nhiễu.

**Nhãn test chính thức đã có** trong bản phát hành LRLPR-26 (3.000 track: 600 Brazilian, 2.400 Mercosur; xin qua license, email trường đại học, khoảng 5 ngày làm việc). Bản đang dùng là bản lúc thi, chưa có nhãn. Giao thức cuối cho bài báo: train split 0.9 (val 999 để theo dõi và để so với 78,70), **báo cáo trên test 3.000 track**. Cách này có SE ±0,75 thay vì ±1,33, không bị thiên lệch do chọn trên val, và so được với bảng xếp hạng cuộc thi.

Vòng này, chạy 3 run song song:

```bash
# (a) backbone thấy ảnh 64x256: stem ResNet34 hạ 4x ngay đầu, 32x128 -> 8x32, nhỏ hơn cả LR gốc (~17x47).
#     Với 64x256 thì sau stem là 16x64, T=32. Không cần sửa code. Base = degcal + aux (#42).
python -u train.py -n restran_hires_aux --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --aux-frame-ctc 0.2 --img-height 64 --img-width 256 --relative-dropout \
    2>&1 | tee results/restran_hires_aux_train.log

# (b),(c) baseline ổn định (cấu hình run #6) theo giao thức cuối, seed 43 và 7. Run #6 = seed 42.
#     GIỮ LẠI *_best.pth: khi có nhãn test sẽ chấm lại trên 3.000 track.
python -u train.py -n restran_base999_s43 --seed 43 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_base999_s43_train.log

python -u train.py -n restran_base999_s7 --seed 7 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_base999_s7_train.log
```

### Thay đổi code 06/10/2026 — fusion có tương tác giữa các frame (mặc định TẮT)

| Cờ | Mặc định | Tác dụng |
|---|---|---|
| `--fusion {attention,temporal}` | attention | `temporal`: 5×W' cột đặc trưng của một track thành một chuỗi token (frame embedding + positional encoding), chạy qua Transformer pre-LN, rồi gộp các frame bằng softmax theo từng vị trí trên token đã có ngữ cảnh. `AttentionFusion` chấm điểm từng frame chỉ từ đặc trưng của chính frame đó, nên các frame không bao giờ được so với nhau |
| `--fusion-layers N` | 2 | Số lớp Transformer của fusion `temporal` (8 head, FFN 1024, +4,24M tham số với 2 lớp) |

Khởi tạo **đúng bằng phép trung bình 5 frame**: lớp ra của các nhánh residual và bộ chấm điểm đều khởi tạo 0, positional encoding được trừ lại sau Transformer. Đã kiểm tra:
- Lúc khởi tạo, output khớp phép trung bình (sai khác tối đa 2,4e-7).
- Checkpoint cũ nạp `strict` bình thường với fusion mặc định.
- Sau 3 bước, gradient đã chảy vào attention, FFN và bộ chấm điểm.
- Trainer chạy được với aux CTC ở cả 32×128 và 64×256.
- `--sharpness-prior` + `--fusion temporal` bị chặn.

```bash
# Base = degcal + aux (#42), chỉ đổi fusion. So với #42: TB6 77,53; 999 TB6 77,79
python -u train.py -n restran_tfusion_aux --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --aux-frame-ctc 0.2 --fusion temporal \
    2>&1 | tee results/restran_tfusion_aux_train.log
```

---

## Run #43, #44 — độ phân giải 64×256 và fusion temporal (06/10/2026)

Cả hai lấy base là #42 (degcal + aux), mỗi run chỉ đổi một yếu tố. Lệnh ở hai mục ngay trên.

| # | Tên run | Thay đổi | Best (ep) | TB6 | So với #42 (TB6) | 999: @best / TB6 | val_loss @best (min) | train_loss cuối | Kết cục |
|---|---|---|---|---|---|---|---|---|---|
| 42 | `restran_auxframe` (base) | — | 77,74 (25) | 77,53 | — | 78,28 / 77,79 | 0,2152 (0,2132) | 0,0323 | |
| 43 | `restran_hires_aux` | input 64×256 | 77,84 (28) | 77,33 | −0,20 | 77,48 / 76,76 | 0,2449 (0,2325) | 0,0197 | ❌ ≈ base |
| 44 | `restran_tfusion_aux` | `--fusion temporal` | 77,24 (29) | 76,25 | −1,28 | 77,28 / 76,33 | 0,2131 (0,2091) | 0,0328 (ep29) | ❌ |

#44 chỉ có 29 epoch (dòng cuối lúc 16:52, thiếu epoch 30), nên so trên **cùng epoch 24–29**: #42 77,36, #43 77,19, #44 76,25 (−1,11, khoảng 2σ của chênh lệch giữa hai run đơn).

Rút ra:
- **Độ phân giải không phải nút thắt.** 64×256 học nhanh hơn hẳn lúc đầu (ep4: 40,77 so với 17,61; vượt 60% ở ep7 thay vì ep11), nhưng về đích ngang base. Model fit tập train mạnh hơn (train_loss 0,020 so với 0,032) và val_loss cao hơn rõ (0,245 so với 0,215), tức là quá tự tin. Kết quả này khớp với #16 (32×192) và #21 (SR skip): tăng độ phân giải đầu vào không giúp ResTran đọc tốt hơn.
- **Fusion có tương tác giữa các frame thua attention fusion** khoảng 1,1 điểm. Model học chậm hơn ở cuối (vượt 75% ở ep24 so với ep22) và vẫn đang lên khi LR về 0. val_loss lại thấp hơn base, tức là model ít tự tin hơn chứ không đọc tốt hơn. Theo tiêu chí đã đặt thì bỏ hướng này, không tinh chỉnh thêm.
- **Tổng kết vòng sửa (#39–#44):** không thay đổi kiến trúc hay lịch train nào vượt nhiễu. Cấu hình đang ở vùng bão hoà TB6 77,3–77,5 (1999) và 77,2–77,8 (999). So với baseline #9 (TB6 76,25), mức tăng ~+1,1–1,3 là thật, nhưng **đến từ dữ liệu** (sửa nhánh synthetic, augment nhất quán), không đến từ kiến trúc.

### Kế hoạch sau #44: chốt cấu hình, chạy seed theo giao thức cuối

Ngừng thử kiến trúc. Bảng ablation cho bài báo, mỗi dòng 3 seed (42, 43, 7) theo giao thức cuối (train split 0.9, val 999), chấm trên test 3.000 track khi có nhãn:

| Dòng | Cấu hình | Run cần chạy |
|---|---|---|
| Baseline | cấu hình run #6 | #6 (seed 42) + `restran_base999_s43`, `restran_base999_s7` (đã giao lệnh) |
| + dữ liệu | degcal | 3 run mới |
| + aux CTC | degcal + `--aux-frame-ctc 0.2` | 3 run mới |

```bash
# Mẫu lệnh: đổi -n và --seed cho 42 / 43 / 7. Bỏ --aux-frame-ctc 0.2 cho dòng "+ dữ liệu".
python -u train.py -n restran_final999_aux_s42 --seed 42 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --aux-frame-ctc 0.2 \
    2>&1 | tee results/restran_final999_aux_s42_train.log
```

---

## Phân tích 07/10/2026 — lỗi tập trung ở loại ảnh nào (model #29, val 1999, accuracy 77,64%)

Thống kê tính trên 5 frame LR gốc của mỗi track, chia theo tứ phân vị (Q1 = thấp nhất):

| Đặc trưng ảnh | Acc Q1 / Q2 / Q3 / Q4 | % tổng số lỗi nằm ở Q1 |
|---|---|---|
| Tương phản (std mức xám) | **57,2** / 78,2 / 86,0 / 89,2 | **47,9%** |
| Độ nét (LapVar) | **54,8** / 80,4 / 83,6 / 91,8 | **50,6%** |
| Độ sáng trung bình | 72,4 / 80,6 / 77,6 / 80,0 | 30,9% |
| Độ bão hoà màu | 66,4 / 82,0 / 85,0 / 77,2 | 37,6% (lẫn với biển Brazilian nền xám) |
| Bề rộng LR (px) | 73,4 / 77,5 / 78,1 / 81,4 | 29,3% |

- Một nửa số lỗi nằm ở 1/4 số track mờ nhất / tương phản thấp nhất. Ở nhóm nét nhất, model đọc đúng 91,8%.
- Ảnh gần như xám (bão hoà < 25, nhiều khả năng ảnh đêm/IR) chỉ chiếm 3,6% số track, accuracy 55,6%.
- Train và val chia ngẫu nhiên từ cùng phân bố, nên ảnh mờ và tương phản thấp vốn đã có trong train với đúng tỉ lệ này. Augment làm ảnh mờ/nhiễu thêm (#36, #37) đã không giúp. Đây là mất thông tin, augmentation không khôi phục được.

---

## Rà soát pipeline dưới góc độ xử lý ảnh (07/10/2026)

Đo trên model #29 (val 1999) và trên ảnh train thật. Mỗi giả thuyết được kiểm tra bằng số liệu trước khi đưa thành thí nghiệm.

**1. Tương phản thấp gây lỗi độc lập với độ mờ, nhưng là do mất thông tin, không chữa bằng xử lý ảnh được.**
- Tương phản và độ nét đã chuẩn hoá tương phản (LapVar / phương sai) gần như không tương quan (Spearman 0,10).
- Bảng 2×2 accuracy: tương phản thấp + mờ 60,7%; thấp + nét 75,3%; cao + mờ 82,8%; cao + nét 92,0%.
- Hiệu ứng này có trong từng layout: Mercosur 75,2% so với 90,6%; Brazilian 47,5% so với 66,5%.
- Kéo giãn tương phản lúc test cho ảnh tương phản thấp: **76,64% (−1,00)**. Nhóm tương phản thấp nhất còn tụt từ 57,2 xuống 54,4, vì kéo tương phản cũng khuếch đại nhiễu và vệt JPEG. Tức là giới hạn nằm ở tỉ lệ tín hiệu/nhiễu, không ở biên độ.

**2. Các frame lệch nhau và STN không căn được, nhưng căn bằng phase correlation thì hại.**
- Lệch ngang so với frame giữa: trung vị 1,63px, 19,4% số cặp lệch quá 4px (nửa cột đặc trưng). Sau STN: 1,60px và 17,9%, tức STN không căn frame. Hệ số θ cho thấy STN chủ yếu **zoom vào khoảng 1,3×**.
- Căn frame bằng phase correlation lúc test: **76,24% (−1,40)**, sửa đúng thêm 26 track nhưng làm sai 54. Phép nội suy khi dịch làm ảnh mờ thêm, và ước lượng độ lệch trên ảnh LR không đáng tin. Kết hợp với #44 (fusion temporal), model có vẻ đã chịu được độ lệch tự nhiên giữa các frame.

**3. Augment hình học gây lệch phân bố giữa train và test.** Đây là thứ sửa được.
- Mỗi phép warp bilinear trên ảnh đã phóng to làm giảm độ nét vùng giữa ảnh: Affine còn 0,77, Perspective 0,61, Rotate 0,86. Khoảng 3/4 ảnh train dính ít nhất một phép như vậy, ảnh val thì không.
- Màu tô viền không nhất quán: Affine tô xám (128), còn Rotate và Perspective tô **đen** (0). 7–8% pixel thành vùng tô phẳng với cạnh sắc nhân tạo.
- Với `--clean-geo-aug` (bicubic + nhân bản viền): độ nét lên **0,92 / 0,74 / 1,02**, vùng tô màu còn **0%**.

**4. Toàn bộ LR Scenario-B là JPEG lấy mẫu màu 4:2:0** (200/200 file kiểm tra). Kênh màu chỉ có ¼ độ phân giải, khoảng 9px chiều cao trên biển 17px, nên chỉ chứa nhiễu và vệt lem màu. Scenario-A là PNG không nén. Val/test giống B, gợi ý thử chỉ dùng kênh sáng (luma).

**5. Với `--consistent-aug`, CoarseDropout được lặp lại y hệt trên cả 5 frame**: lỗ che nằm cùng chỗ, nên phần bị che mất khỏi cả track. Che độc lập từng frame buộc fusion phải lấy phần bị che từ frame khác, đúng với mục đích của multi-frame.

### Thay đổi code 07/10/2026 — mọi cờ mặc định TẮT

| Cờ | Tác dụng |
|---|---|
| `--frame-dropout p` | Lúc train, mỗi frame với xác suất p bị thay đặc trưng bằng trung bình các frame còn lại của track, ngay trước fusion; luôn giữ ít nhất 1 frame. Nhánh aux CTC vẫn đọc đặc trưng gốc |
| `--frame-mask p` | Lúc train, mỗi frame với xác suất p bị xoá **độc lập** một dải dọc (1/16–1/8 chiều rộng, nửa tới 1 ký tự) hoặc dải ngang (1/8–1/4 chiều cao). Không áp lên mẫu có HR target của SR |
| `--grayscale` | Chuyển input sang luma (BT.601, nhân ra 3 kênh) ở **bước cuối**, trên tensor đã chuẩn hoá. Áp cho train/val/test và mọi target. **Phải truyền lại khi predict** |
| `--clean-geo-aug` | Affine/Perspective/Rotate dùng nội suy bicubic + nhân bản viền thay vì bilinear + tô đen/xám |

Đã kiểm tra:
- Frame dropout: chế độ eval cho kết quả y hệt model không dropout; frame bị thay đúng bằng trung bình các frame giữ lại; không track nào bị bỏ hết.
- Frame mask: tỉ lệ frame bị che 0,202 (mục tiêu 0,2); diện tích che 6,2–25%; 0 cặp frame trùng mask; mẫu SR không bị đụng.
- Grayscale: 3 kênh giống hệt nhau trên ảnh thật, synthetic legacy, synthetic native, HR target, val và test; khớp `cv2` dưới 1 mức xám.
- Mỗi cờ train thử 4 batch qua Trainer và chạy qua CLI đều được.

### Vòng thí nghiệm: 4 run, base = #42 (degcal + aux), so với TB6 77,53

Chạy song song thì dùng `nproc/4`:

```bash
python -u train.py -n restran_fdrop_aux --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-dropout 0.1 \
    2>&1 | tee results/restran_fdrop_aux_train.log
```

Ba run còn lại chỉ đổi `-n`, tên file log và dòng cuối: `restran_fmask_aux` → `--frame-mask 0.2`; `restran_gray_aux` → `--grayscale`; `restran_cleangeo_aux` → `--clean-geo-aug`.

---

## Run #45 → #48 — frame dropout, frame mask, grayscale, clean-geo (12/10/2026)

Base = #42 (degcal + aux, TB6 77,53). Mỗi run chỉ thêm một cờ. Lệnh ở mục "Rà soát pipeline" ngay trên. Cả 4 checkpoint đã chấm lại trên máy local, khớp CSV đến từng chữ số.

| # | Tên run | Cờ thêm | Best (ep) | TB6 | So với #42 (TB6) | 999: @best / TB6 | val_loss cuối | train_loss cuối | Kết cục |
|---|---|---|---|---|---|---|---|---|---|
| 45 | `restran_fdrop_aux` | `--frame-dropout 0.1` | 78,69 (26) | 78,28 | +0,75 | **78,48** / 78,19 | **0,2077** | 0,0394 | ✅ |
| 46 | `restran_fmask_aux` | `--frame-mask 0.2` | **78,89** (30) | **78,47** | **+0,94** | 78,18 / 78,09 | 0,2102 | 0,0358 | ✅ |
| 47 | `restran_gray_aux` | `--grayscale` | 78,64 (29) | 78,02 | +0,49 | 78,28 / 77,69 | 0,2091 | 0,0337 | ✅ (yếu hơn) |
| 48 | `restran_cleangeo_aux` | `--clean-geo-aug` | 77,44 (29) | 77,16 | −0,37 | 77,78 / 77,61 | 0,2153 | 0,0326 | ❌ |

**So với mốc 78,70% của tác giả (tập 999):** epoch tốt nhất (chọn theo val 1999) cho 78,48 (#45), 78,28 (#47), 78,18 (#46). Nếu chọn epoch theo chính tập 999, như cách mốc được báo cáo, thì #45 đạt 78,68 ở ep28. Tức là **ngang mốc, chưa vượt**. 78,89% là trên tập 1999, không so trực tiếp được.

Kiểm định McNemar trên cùng 1999 track (số track B đúng mà A sai / A đúng mà B sai):
- So với #29 (model đơn cũ tốt nhất có checkpoint): fmask +94/−69 (p=0,06), fdrop +94/−73 (p=0,12), gray +91/−71 (p=0,14), clean-geo +76/−80 (p=0,81).
- fmask so với baseline #9: +105/−71, **p=0,013**.
- fdrop so với fmask: +76/−72 (p=0,81), tức không phân biệt được giữa hai cái.

Rút ra:
- **Hai kỹ thuật điều chuẩn dành riêng cho multi-frame thắng, đúng cơ chế đã dự kiến**: train loss cao hơn, val loss cuối thấp hơn (0,208–0,210 so với 0,222). Ở base, val loss tăng dần từ ep24; ở hai run này thì không. Aux CTC của fmask cao hơn (0,30 so với 0,24): từng frame khó đọc riêng hơn, nhưng kết quả sau fusion tốt nhất.
- Cải thiện trải đều ở mọi nhóm ảnh, nhỉnh hơn ở nhóm khó: độ nét Q1 tăng từ 60,6 lên 63,2–63,4; tương phản Q1 từ 57,2 lên 58,4–58,6.
- Tập track sai của hai model bất kỳ chỉ trùng khoảng 70% (Jaccard 0,68–0,73), nên mỗi run đơn vẫn mang nhiễu đáng kể.
- **Clean-geo không giúp**, dù đã sửa đúng lệch pha đo được. Độ mờ và viền nhân tạo do augment có thể đang đóng vai trò điều chuẩn.
- Confidence Gap (tiêu chí phụ của cuộc thi) giảm nhẹ ở các model được điều chuẩn: 0,098–0,107 so với 0,115 của #29.

### Vòng tiếp theo: ghép các cờ thắng + đo trực tiếp trên giao thức của tác giả

Chạy 4 run song song:

```bash
# combo3: cấu hình ứng viên (giao thức 1999)
python -u train.py -n restran_combo3 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_combo3_train.log

# combo2: như combo3 nhưng bỏ grayscale
python -u train.py -n restran_combo2 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 \
    2>&1 | tee results/restran_combo2_train.log

# combo3, 40 epoch (LR 2.5e-4 như #14 để không nổ)
python -u train.py -n restran_combo3_40ep --num-workers $(( $(nproc) / 4 )) \
    --epochs 40 --lr 2.5e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.8 --val-split-file splits/val_tracks_2000.json \
    --val-subset-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_combo3_40ep_train.log

# combo3 theo GIAO THỨC CỦA TÁC GIẢ: train 19.001 track, val 999 -> so trực tiếp với 78,70
python -u train.py -n restran_combo3_999 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_combo3_999_train.log
```

Quy tắc chọn:
- combo3 và combo2 chênh dưới 0,3 TB6 → chọn combo2, đơn giản hơn và không cần `--grayscale` khi predict.
- 40 epoch hơn 30 epoch từ 0,5 TB6 trở lên → dùng 40 epoch cho các seed cuối.
- Sau vòng này: 3 seed của cấu hình chốt theo giao thức 999, cùng với baseline seed 43 và 7 (vẫn chưa có kết quả).

---

## Run #49 → #51 — ghép cờ và giao thức của tác giả (12/10/2026)

Lệnh ở mục "Vòng tiếp theo: ghép các cờ thắng" ngay trên. Run `restran_combo3_40ep` chưa có kết quả.

| # | Tên run | Giao thức | Best (ep) | TB6 | M / B @best | val_loss cuối | train_loss cuối |
|---|---|---|---|---|---|---|---|
| 49 | `restran_combo2` (fmask + fdrop) | train 18.001 / val 1999 | 78,74 (29) | 78,02 | 84,57 / 55,80 | 0,1962 | 0,0435 |
| 50 | `restran_combo3` (fmask + fdrop + gray) | train 18.001 / val 1999 | 78,89 (27) | 78,22 | 84,07 / 58,52 | 0,2015 | 0,0457 |
| 51 | **`restran_combo3_999`** | **train 19.001 / val 999 (như tác giả)** | **79,28 (29)** | **78,33** | 84,16 / 60,95 | 0,2044 | 0,0443 |

Trên tập 999, #50 đạt 78,68 ở epoch tốt nhất (TB6 77,98). Thêm 1.000 track train (#51) cộng thêm khoảng +0,6 ở best và +0,35 ở TB6, vẫn trong vùng nhiễu.

**So với các mốc trên cùng giao thức 999:**

| | Best | TB6 |
|---|---|---|
| Tác giả (slide BTC, 1 run) | 78,70 | — |
| Baseline mình chạy lại, #6 | 77,18 | 75,98 |
| Baseline mình chạy lại, #3 | 76,68 | 76,29 |
| **#51 combo3_999** | **79,28** | **78,33** |

- Hơn baseline chạy lại khoảng **+2,1 ở best và +2,0–2,35 ở TB6**, và cao hơn ở **mọi epoch từ 20 trở đi**. Đây là cải thiện chắc chắn nhất cho tới nay.
- So với 78,70 của tác giả: **+0,58 ở best, nhưng TB6 thấp hơn (78,33)**. Một run đơn nên mới là "ngang, nhỉnh hơn", chưa phải "vượt" có ý nghĩa thống kê.

Rút ra:
- **Ghép cờ không cộng dồn.** TB6 trên 1999: combo3 78,22 và combo2 78,02, so với fmask đứng riêng 78,47 và fdrop 78,28. Điều chuẩn đã bão hoà: val loss thấp nhất từ trước tới nay (0,196–0,202), train loss cao nhất (0,044–0,046), nhưng accuracy không tăng thêm. Điều chuẩn mạnh hơn nữa khó giúp.
- Riêng biển Brazilian tăng rõ khi có grayscale: B = 58,52 (#50) và 60,95 (#51), so với 55,80 (#49, không gray). Biển Brazilian nền xám nên màu sắc ít ích lợi. Tuy vậy, Brazilian chỉ khoảng 200–405 track nên sai số ±2,5–3,4.

### Thử cách đọc output: gộp ở mức đặc trưng so với trung bình dự đoán từng frame

Cùng 1 model, 1 lần forward: model có aux CTC sinh cả dự đoán sau fusion lẫn dự đoán từng frame. Trọng số ghép cố định 0,5, không tinh chỉnh trên val.

| Model | Sau fusion (mặc định) | Trung bình dự đoán 5 frame | Ghép 0,5 / 0,5 |
|---|---|---|---|
| #46 fmask | **78,89** | 76,19 | 78,84 |
| #45 fdrop | **78,69** | 75,24 | 78,69 |
| #47 gray | **78,64** | 75,29 | 78,59 |

- Ghép không giúp, nên giữ cách đọc mặc định.
- **Gộp ở mức đặc trưng hơn trung bình dự đoán từng frame khoảng 2,7–3,5 điểm.** Cách thứ hai về bản chất là kiểu "bỏ phiếu/cộng logit theo frame" mà đội hạng 1 và hạng 3 cuộc thi dùng. Đây là một luận điểm cho bài báo.

### Vòng cuối: seed theo giao thức của tác giả (4 run song song)

Cấu hình chốt = combo3 (#51 là seed 42). Baseline = cấu hình run #6 (seed 42). Mỗi bên thêm seed 43 và 7. Giữ lại **mọi** `*_best.pth` để chấm trên test 3.000 track khi có nhãn.

```bash
python -u train.py -n restran_combo3_999_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_combo3_999_s43_train.log

python -u train.py -n restran_combo3_999_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_combo3_999_s7_train.log

python -u train.py -n restran_base999_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_base999_s43_train.log

python -u train.py -n restran_base999_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_base999_s7_train.log
```

Bảng cuối cho bài báo: mean ± std (3 seed) của best và TB6 trên val 999, cho baseline và cho phương pháp. Thêm cột test 3.000 track khi có nhãn.

---

## Run #52 → #54 — seed trên giao thức của tác giả (18/10/2026)

Lệnh ở mục "Vòng cuối: seed theo giao thức của tác giả". Run `restran_base999_s43` **chưa có kết quả**.

| # | Run | Seed | Best (ep) | TB6 | Epoch cuối | M / B @best |
|---|---|---|---|---|---|---|
| 51 | `restran_combo3_999` | 42 | 79,28 (29) | 78,33 | 78,48 | 84,16 / 60,95 |
| 52 | `restran_combo3_999_s43` | 43 | 78,48 (30) | 77,83 | 78,48 | 83,78 / 58,57 |
| 53 | `restran_combo3_999_s7` | 7 | 78,88 (30) | 78,41 | 78,88 | 84,03 / 59,52 |
| 6 | `restran_30ep_lr3e4` (baseline) | 42 | 77,18 (28) | 75,98 | 76,58 | — |
| 3 | `restran_stn_v2` (baseline, chạy lại seed 42) | 42 | 76,68 (27) | 76,29 | 76,68 | — |
| 54 | `restran_base999_s7` (baseline) | 7 | 77,58 (30) | 76,99 | 77,58 | 82,13 / 60,48 |

> ⚠️ **SUPERSEDED** (MINOR 15 của review): bảng dưới chỉ có 3 run baseline, trong đó 2 run trùng seed 42. Không dùng cho luận văn/bài báo. Bảng thay thế: "Run #58" (val 999, 4 seed khác nhau) và kết quả lock-box "Run #63 → #70".

**Tổng hợp (val 999, cùng giao thức với tác giả):**

| | Phương pháp (3 seed) | Baseline chạy lại (3 run) | Chênh | Welch t-test |
|---|---|---|---|---|
| Best epoch | **78,88 ± 0,40** | 77,14 ± 0,45 | **+1,74** | p = 0,008 |
| TB6 | **78,19 ± 0,32** | 76,42 ± 0,52 | **+1,77** | p = 0,012 |
| Epoch cuối | **78,61 ± 0,23** | 76,94 ± 0,55 | **+1,67** | p = 0,022 |

- **Hơn baseline chạy lại khoảng +1,7 điểm, có ý nghĩa thống kê**, và ổn định qua các seed: seed tệ nhất của phương pháp vẫn hơn run tốt nhất của baseline (best 78,48 so với 77,58; TB6 77,83 so với 76,99).
- **So với 78,70 tác giả công bố: ngang mốc.** Best trung bình 78,88 ± 0,40 không khác 78,70 có ý nghĩa thống kê (p = 0,52); 2/3 seed vượt. TB6 trung bình 78,19 thấp hơn nhưng cũng không có ý nghĩa (p = 0,11). Trong bài nên viết là "đạt ngang mức công bố", không viết "vượt".
- Baseline chạy lại (77,14 ± 0,45) thấp hơn 78,70 khoảng 1,6 điểm. Cấu hình gốc của tác giả (LR 5e-4, clip 5) bị nổ khi mình chạy (run #1), còn bản đã ổn định cho 77,14. Con số 78,70 có thể là một run thuận lợi, hoặc do khác môi trường chạy. Trong bài cần báo cáo cả hai con số và giải thích.
- Lưu ý: hai trong ba run baseline cùng seed 42 (#3 và #6, khác nhau do GPU không tất định). Cần `base999_s43` để đủ 3 seed khác nhau.

---

## Run #55 → #57 — cấu hình gốc của tác giả, seed 2026 (18/10/2026)

```bash
# #55 — cấu hình GỐC của tác giả (LR 5e-4, clip 5, không skip guard), seed 43
python -u train.py -n restran_orig999_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 5e-4 --grad-clip 5.0 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_orig999_s43_train.log

# #56 — phương pháp (combo3), seed 2026
python -u train.py -n restran_combo3_999_s2026 --seed 2026 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_combo3_999_s2026_train.log

# #57 — baseline đã ổn định (cấu hình run #6), seed 2026
python -u train.py -n restran_base999_s2026 --seed 2026 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_base999_s2026_train.log
```

| # | Run | Best (ep) | TB6 | Epoch cuối | Kết cục |
|---|---|---|---|---|---|
| 55 | `restran_orig999_s43` (cấu hình gốc) | 54,25 (9) | 0,00 | 0,00 | ❌ **nổ ở ep10**: grad norm 49.348, 21 batch inf/nan; accuracy về 0 và train loss kẹt ở 3,65 tới hết |
| 56 | `restran_combo3_999_s2026` | 79,18 (26) | **78,91** | 79,08 | ✅ |
| 57 | `restran_base999_s2026` | 77,78 (28) | 77,14 | 77,48 | baseline |

**Cấu hình gốc của tác giả nổ 2/2 lần** trên giao thức 999: run #1 (seed 42) nổ ở ep12, run #55 (seed 43) nổ ở ep10. Cả hai đều nổ đúng lúc OneCycle đưa LR lên đỉnh 5e-4, trong khi train loss đã xuống khoảng 0,3. Đây là bằng chứng cho bài báo rằng con số 78,70 không tái lập được ổn định bằng cấu hình đã công bố, và là lý do phải dùng baseline đã ổn định hoá.

> **Sửa theo ISSUE #4 (25/10):** claim trên phải có phạm vi: "Trong môi trường của chúng tôi (V100, PyTorch 2.14.1+cu126, AMP fp16), cấu hình công bố diverge ở 2/2 seed khi LR đạt đỉnh 5e-4." Run #55 có 21 batch inf/nan, nên chưa loại trừ được tràn số fp16. Lệnh chạy cấu hình gốc ở fp32 (E5, `restran_orig999_fp32_s42`) đã có ở mục "Lock-box: lệnh chạy".

> ⚠️ **SUPERSEDED** (MINOR 15 của review): "Baseline ổn định (4 run)" dưới đây có 2 run trùng seed 42 (#3, #6). Không dùng cho luận văn/bài báo. Bảng thay thế: "Run #58" (4 seed khác nhau) và kết quả lock-box "Run #63 → #70".

**Bảng tổng hợp cập nhật (val 999, giao thức của tác giả):**

| | Phương pháp (4 seed: 42, 43, 7, 2026) | Baseline ổn định (4 run) | Chênh | Welch t-test |
|---|---|---|---|---|
| Best epoch | **78,95 ± 0,36** | 77,30 ± 0,49 | **+1,65** | p = 0,002 |
| TB6 | **78,37 ± 0,44** | 76,60 ± 0,56 | **+1,77** | p = 0,003 |
| Epoch cuối | **78,73 ± 0,30** | 77,08 ± 0,52 | **+1,65** | p = 0,003 |

- Baseline "4 run" gồm #6 (seed 42), #3 (chạy lại seed 42), seed 7, seed 2026. Nếu chỉ tính 3 seed khác nhau (42, 7, 2026): best 77,51 ± 0,31, TB6 76,70 ± 0,64. Chênh vẫn +1,44 / +1,67 (p = 0,003 / 0,024).
- Không chồng lấn: seed tệ nhất của phương pháp vẫn hơn run baseline tốt nhất (best 78,48 so với 77,78; TB6 77,83 so với 77,14).
- So với 78,70 tác giả công bố: best trung bình 78,95 ± 0,36, có 3/4 seed vượt, nhưng p = 0,25 nên **"ngang hoặc nhỉnh hơn, chưa có ý nghĩa thống kê"**.

---

## Run #58 — baseline seed 43; bảng chính 4 seed vs 4 seed (18/10/2026)

`restran_base999_s43` (lệnh ở mục "Vòng cuối: seed theo giao thức của tác giả"): best 77,78 (ep29), TB6 77,21, epoch cuối 77,38, M/B 83,02 / 58,10.

> Cập nhật 25/10: kết quả chính cho bài báo nay là **lock-box** ("Run #63 → #70"). Bảng dưới vẫn đúng, nhưng là kết quả trên val 999, tập đã dùng để chọn cấu hình.

**Bảng chính cho bài báo (val 999, giao thức của tác giả, 4 seed khác nhau mỗi bên: 42, 43, 7, 2026):**

| | Phương pháp (combo3) | Baseline đã ổn định | Chênh | Welch t-test |
|---|---|---|---|---|
| Best epoch | **78,95 ± 0,36** | 77,58 ± 0,28 | **+1,38** | p = 0,001 |
| TB6 | **78,37 ± 0,44** | 76,83 ± 0,58 | **+1,54** | p = 0,006 |
| Epoch cuối | **78,73 ± 0,30** | 77,25 ± 0,46 | **+1,48** | p = 0,003 |
| Tác giả công bố (1 run, cấu hình gốc) | — | 78,70 | | cấu hình gốc nổ 2/2 lần khi chạy lại (#1, #55) |

Phương pháp so với 78,70: p = 0,25 (best) và 0,23 (TB6), tức ngang mức công bố. Run #3 (chạy lại seed 42) không còn dùng trong bảng.

Đường cong trung bình 4 seed của phương pháp: accuracy tăng tới ep29 (78,90). **val loss đứng yên khoảng 0,200 từ ep24 tới ep30**, khác hẳn các cấu hình chưa điều chuẩn, nơi val loss tăng ở cuối. Đây là lý do thử lại train dài hơn.

### Thí nghiệm 40 epoch với cấu hình tốt nhất (giao thức 999, cùng 4 seed)

LR đỉnh giảm từ 3e-4 xuống 2,5e-4 theo tiêu chí LR/loss (cùng mức với #14, 40 epoch, không nổ). Mỗi seed so cặp với run 30 epoch cùng seed.

```bash
for SEED in 42 43 7 2026; do
python -u train.py -n restran_combo3_999_40ep_s$SEED --seed $SEED --num-workers $(( $(nproc) / 4 )) \
    --epochs 40 --lr 2.5e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    > results/restran_combo3_999_40ep_s${SEED}_train.log 2>&1 &
done
wait
```

Tiêu chí đặt trước: chỉ dùng 40 epoch nếu TB6 trung bình 4 seed hơn bản 30 epoch (78,37) ít nhất 0,5. Nếu đạt, **phải chạy thêm baseline 40 epoch** (cùng LR 2,5e-4) để bảng so sánh cùng ngân sách train. Bảng chính so với tác giả vẫn giữ 30 epoch.

---

## Run #59 → #62 — cấu hình tốt nhất với 40 epoch (19/10/2026)

Lệnh ở mục "Thí nghiệm 40 epoch với cấu hình tốt nhất" (`--epochs 40 --lr 2.5e-4`, giao thức 999). Các file `*_best.pth` đã có ở `results/`.

| # | Seed | 30 epoch: best / TB6 / cuối | 40 epoch: best (ep) / TB6 / cuối | Chênh TB6 | val_loss cuối (40ep) | train_loss cuối (40ep) |
|---|---|---|---|---|---|---|
| 59 | 42 | 79,28 / 78,33 / 78,48 | **79,68** (39) / 79,08 / 79,48 | +0,75 | 0,2161 | 0,0308 |
| 60 | 43 | 78,48 / 77,83 / 78,48 | 78,88 (38) / 78,55 / 78,58 | +0,72 | 0,2264 | 0,0286 |
| 61 | 7 | 78,88 / 78,41 / 78,88 | 78,68 (38) / 78,08 / 78,68 | −0,33 | 0,2184 | 0,0284 |
| 62 | 2026 | 79,18 / 78,91 / 79,08 | 79,18 (36) / 78,75 / 78,68 | −0,17 | 0,2109 | 0,0276 |

| | 30 epoch (4 seed) | 40 epoch (4 seed) | Chênh theo cặp seed |
|---|---|---|---|
| Best | 78,95 ± 0,36 | 79,10 ± 0,44 | +0,15 (paired t-test p = 0,39) |
| TB6 | 78,37 ± 0,44 | 78,61 ± 0,42 | +0,24 (p = 0,46) |
| Epoch cuối | 78,73 ± 0,30 | 78,85 ± 0,42 | +0,13 (p = 0,71) |

- **Không đạt tiêu chí đặt trước** (TB6 ≥ 78,87): 2 seed tăng, 2 seed giảm, chênh trung bình không có ý nghĩa thống kê. **Giữ 30 epoch** cho cấu hình chốt. Như vậy không cần chạy baseline 40 epoch.
- Train dài hơn làm overfit quay lại: val loss cuối 0,211–0,226 so với khoảng 0,200 ở bản 30 epoch, train loss 0,028 so với 0,042.
- Run đơn cao nhất từ trước tới nay: 79,68% (#59, seed 42, 40 epoch). Không dùng làm con số báo cáo, vì đó là chọn run tốt nhất.

Report tổng hợp gửi thầy: [`documents/REPORT.md`](REPORT.md).

---

## Kế hoạch theo review của thầy (`documents/Teacher_Review.md`, 23/10/2026)

**Đóng băng cấu hình:** #50 (combo3) giữ nguyên làm cấu hình chốt, theo ISSUE #3. Mọi thí nghiệm sau đây chỉ để mô tả, không dùng để đổi cấu hình.

**Đã kiểm tra ngay (mục 3 của review):**

| Mục | Kết quả |
|---|---|
| A1 frame dropout | Chỉ chạy khi `self.training` (`restran.py:173–174`); đã có test eval cho kết quả y hệt model không dropout |
| A2 frame mask | Chỉ gọi trong vòng train (`trainer.py:278`); `validate()` không gọi |
| A3 skip guard | backward → `unscale_` → norm trên gradient đã unscale → bỏ batch → `zero_grad` + `scaler.update()` (`trainer.py:359–396`) |
| A4 grayscale | Luma BT.601 lặp 3 kênh, chuyển ở bước cuối. Normalize(0,5; 0,5) như nhau cho mọi kênh và trọng số luma cộng bằng 1, nên không cần tính lại mean/std |
| A6 split | SHA-256 `val_tracks.json` = `8a560294…927b`; `val_tracks_2000.json` = `8de23c86…53de` |
| A7 aux CTC | Đo trên CPU (batch 16): train +8% mỗi batch (1.509 → 1.626 ms); suy luận không đổi |
| ISSUE #4 cấu hình | Transformer 3 layer, 8 head, FFN 2048, dropout 0,1; batch 64, wd 1e-4: khớp cấu hình tác giả |
| A5 regression test | Chưa làm. Thư mục local không phải git repo; cần chạy trong bản clone có nhánh `main` |

**Thứ tự việc:**
1. (Hôm nay, người dùng) Gửi đơn xin nhãn test LRLPR-26. Bản đang có (`Pa7a3Hin-test-public`, `TKzFBtn7-test-blind`) không có `annotations.json`. Xuất diff 5 file cho thầy. Báo trạng thái prototype API (ISSUE #11).
2. (Trước 31/10) Lock-box: `lockbox_protocol.md`, tạo split 1.000 track Scenario-B chưa từng dùng để chọn, 8 run (4 baseline + 4 #50) train không có các track này, chấm một lần. Freeze manifest SHA-256.
3. Test xác định ở chế độ eval + regression test không cờ so với `main` (ISSUE #9).
4. E3 (accuracy theo số frame N = 1..5, chỉ suy luận) và N1 (kiểu che: không che / tube / độc lập / bổ sung, 3 seed): quyết định độ mạnh của hướng (b).
5. E2 leave-one-out (7 thành phần × 3 seed), E4 (`--no-stn` và crop cố định ~1,3×), E5 (cấu hình gốc fp32), E6 (#39/#40 thêm 2 seed), N3 (hiệu chỉnh độ tin cậy, risk–coverage).
6. Sửa REPORT theo ISSUE #2, #4–#7 và MINOR 12–19.

---

## Code cho lock-box và kiểm chứng (23/10/2026)

**Cờ mới, đều mặc định tắt:**

| Cờ | Tác dụng |
|---|---|
| `--exclude-tracks FILE` | Loại các track trong FILE khỏi cả train lẫn val. Phép chia train/val được tính trước khi loại, nên tập val không bị đổi. Log in SHA-256 của FILE |
| `--save-last-k K` | Lưu thêm trọng số của K epoch cuối: `{tên}_epNN.pth` |
| `--amp-dtype fp32` | Tắt hẳn mixed precision (không autocast, không scale loss), dùng cho E5 |

**Script và tài liệu mới:**
- `tools/make_lockbox.py`: tạo và kiểm chứng (`--verify`) lock-box.
- `tools/eval_lockbox.py`: chấm một lần. Tính 3 thước đo, Welch / Cohen's d, paired bootstrap, McNemar, E3 theo số frame, rồi áp quy tắc (a)/(b).
- `tools/freeze_manifest.py`: ghi SHA-256 của code, split, checkpoint và môi trường.
- `tools/regression_check.py`: so cách train khi không bật cờ với code gốc.
- `tests/test_eval_determinism.py`: 7 test.
- `documents/lockbox_protocol.md`: protocol đăng ký trước.

**Sửa trong code:**
- `AttentionFusion` nhận số frame tuỳ ý (mặc định 5), để chạy E3.
- Backbone gộp chiều cao bằng `adaptive_avg_pool2d` ở cuối, giống code gốc; `_sequence_head` chỉ gộp khi chiều cao khác 1. Về giá trị thì không đổi gì, nhưng thứ tự phép toán giờ trùng code gốc nên train giống từng bit. Trước đó lệch ở mức làm tròn số thực 32-bit (3,7e-9 sau 1 batch), không ảnh hưởng kết quả các run cũ.

**Phát hiện:** albumentations 2.0.8 không tuân theo seed toàn cục; augment ngẫu nhiên dùng bộ sinh riêng. Đây là nguồn dao động giữa các lần chạy cùng seed, bên cạnh GPU không tất định.

## Lock-box: lệnh chạy (theo `documents/lockbox_protocol.md`)

**Trước khi chạy:** đồng bộ code và file `splits/lockbox_tracks.json` lên server, rồi kiểm tra:

```bash
python tools/make_lockbox.py --verify            # phải in: ✅ Khớp ... sha256 09f90d2b...
python tests/test_eval_determinism.py            # phải in 7 dòng OK
```

Đầu log mỗi run phải có `🔒 Loại 1000 track khỏi train, 0 khỏi val` và `[TRAIN] Loaded 18001 tracks.`

**Baseline, 4 seed:**

```bash
python -u train.py -n restran_lockbox_base_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    2>&1 | tee results/restran_lockbox_base_s42_train.log

python -u train.py -n restran_lockbox_base_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    2>&1 | tee results/restran_lockbox_base_s43_train.log

python -u train.py -n restran_lockbox_base_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    2>&1 | tee results/restran_lockbox_base_s7_train.log

python -u train.py -n restran_lockbox_base_s2026 --seed 2026 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    2>&1 | tee results/restran_lockbox_base_s2026_train.log
```

**Phương pháp (#50), 4 seed:**

```bash
python -u train.py -n restran_lockbox_combo3_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_combo3_s42_train.log

python -u train.py -n restran_lockbox_combo3_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_combo3_s43_train.log

python -u train.py -n restran_lockbox_combo3_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_combo3_s7_train.log

python -u train.py -n restran_lockbox_combo3_s2026 --seed 2026 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_combo3_s2026_train.log
```

**Sau khi cả 8 run xong** (không chấm lock-box trước bước này):

```bash
python tools/make_lockbox.py --verify
python tools/freeze_manifest.py --out results/freeze_manifest.json \
    --checkpoints "results/restran_lockbox_*_s*_ep*.pth" "results/restran_lockbox_*_best.pth"
python tools/eval_lockbox.py --frames-analysis --out results/lockbox_eval.json
```

**E5: cấu hình gốc của tác giả ở fp32** (ISSUE #4). Chạy kèm được; không dùng lock-box. Fp32 trên V100 chậm hơn fp16 khoảng 2–3 lần.

```bash
python -u train.py -n restran_orig999_fp32_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 5e-4 --grad-clip 5.0 --amp-dtype fp32 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_orig999_fp32_s42_train.log
```

---

## Run #63 → #70 — LOCK-BOX: kết quả chấm một lần (25/10/2026, server `fierce-ibex`)

Theo `documents/lockbox_protocol.md`. Kiểm tra trước khi chấm:
- `make_lockbox.py --verify`: khớp `09f90d2b…`.
- 8/8 run có đủ 6 checkpoint epoch và 1 checkpoint best.
- 7/7 test đạt.
- Không run nào bị dừng giữa chừng.
- Freeze manifest: 24 file code, 3 split, 56 checkpoint, digest `39c39dfceb86a2d5b9d338ade096c5a00a85bebe88ad0127e049d318b7117824`.
- Không có file `restran_lockbox_base_s42_train.log`, nhưng `summary.txt` của cả 8 run đều ghi `Train samples : 36002` (đã loại lock-box). Bốn run phương pháp có cấu hình giống hệt nhau trong log.

**Theo từng seed** (accuracy trên 1.000 track lock-box):

| Seed | Baseline: cuối / Last-6 / best | Phương pháp: cuối / Last-6 / best |
|---|---|---|
| 42 | 77,40 / 76,50 / 77,60 | **79,90** / 79,28 / 80,00 |
| 43 | 77,40 / 77,40 / 77,80 | **80,30** / 80,05 / 80,60 |
| 7 | 77,90 / 77,17 / 77,90 | 78,10 / 77,88 / 78,00 |
| 2026 | 77,70 / 77,23 / 77,20 | 78,20 / 77,63 / 78,20 |

**So sánh:**

| Thước đo | Baseline | Phương pháp | Chênh | CI 95% bootstrap (theo track) | Welch (theo seed) | Cohen's d |
|---|---|---|---|---|---|---|
| **Epoch cuối** (chính) | 77,60 ± 0,24 | 79,12 ± 1,14 | **+1,52** | [+0,53; +2,52] | p = 0,072 | 1,85 |
| **Last-6** (quyết định) | 77,07 ± 0,40 | 78,71 ± 1,15 | **+1,64** | [+0,78; +2,48] | p = 0,059 | 1,90 |
| Best theo val 999 | 77,62 ± 0,31 | 79,20 ± 1,30 | +1,57 | [+0,57; +2,55] | p = 0,090 | 1,67 |

McNemar ở epoch cuối, từng cặp seed: 42: +60/−35 (p = 0,013); 43: +59/−30 (p = 0,003); 7: +49/−47 (p = 0,92); 2026: +51/−46 (p = 0,68).

**E3: accuracy theo số frame** (epoch cuối, trung bình mọi tập con N frame):

| N | Baseline | Phương pháp | Chênh [CI 95%] |
|---|---|---|---|
| 1 | 50,59 | 55,59 | **+5,00** [+4,32; +5,69] |
| 2 | 66,14 | 69,02 | +2,88 [+2,29; +3,48] |
| 3 | 72,18 | 74,13 | +1,95 [+1,27; +2,63] |
| 4 | 75,31 | 77,00 | +1,69 [+0,94; +2,43] |
| 5 | 77,60 | 79,12 | +1,52 [+0,53; +2,52] |

**Quy tắc đăng ký trước** (chênh Last-6 ≥ +1,0 và cận dưới CI > 0): **→ hướng (b).**

Rút ra:
- **Mức lợi không bị thổi phồng bởi việc chọn cấu hình trên val** (H2 của review không được ủng hộ). Trên val 999: +1,38 / +1,54 / +1,48 (best / TB6 / cuối). Trên lock-box chưa từng dùng: +1,57 / +1,64 / +1,52. Baseline trên lock-box (77,6) cũng ngang trên val 999 (77,58), tức hai tập khó tương đương.
- **Hai tầng thống kê cho kết luận khác nhau, phải báo cáo cả hai.** Theo track, CI không chứa 0 ở cả 3 thước đo. Theo seed, Welch p = 0,06–0,09: chưa đạt 0,05, vì độ lệch giữa các seed của phương pháp lớn (std 1,1–1,3; seed 42, 43 tăng +2,5 đến +2,9, seed 7, 2026 chỉ tăng +0,2 đến +0,5). Dù vậy, cả 4 seed phương pháp đều cao hơn cả 4 seed baseline ở mọi thước đo. *Phân tích bổ sung, không đăng ký trước:* kiểm định hoán vị chính xác với phân tách hoàn toàn 4 so với 4 cho p hai phía = 2/70 = 0,029.
- **E3 rơi vào dạng B đã đăng ký trước**: mức lợi lớn nhất khi chỉ có 1 frame (+5,0) và giảm dần tới 5 frame (+1,5). Mọi CI đều không chứa 0. Diễn giải theo protocol: phương pháp làm model **bền hơn khi thiếu frame hoặc frame kém**, đúng với thiết kế frame dropout và che frame. Đây không phải dạng A (khai thác thông tin bổ sung giữa các frame). N1 (các kiểu che) sẽ kiểm tra riêng cơ chế đó.
- Phương pháp trên lock-box: epoch cuối 79,12, best 79,20. Đây là tập khác với tập val của số công bố 78,70, nên chỉ đặt cạnh nhau để tham khảo.

**Phân tích thêm từ `results_new/`** (25/10/2026):
- **Kiểm chứng:** 56/56 checkpoint, 24/24 file code, 3/3 split khớp SHA-256 với manifest. Môi trường: V100-SXM2-32GB, PyTorch 2.14.1+cu126, CUDA 12.6, cuDNN 9.10.2 (`cudnn.version()` = 91002; bản trước ghi nhầm 9.1.2). Tính lại CI với seed bootstrap khác ra gần như y hệt. Chấm lại `base_s42_ep30` và `combo3_s43_ep30` trên Apple GPU: 77,40% và 80,30%, trùng server.
- **Theo loại biển** (lock-box, epoch cuối, trung bình 4 seed): Brazilian (n = 192) 60,68 → 61,85 (+1,17); Mercosur (n = 808) 81,62 → 83,23 (+1,61). Brazilian chưa đủ lực thống kê.
- **Confidence Gap** (lock-box, epoch cuối): baseline 0,1060 (0,103–0,108), phương pháp **0,1135** (0,112–0,116), cao hơn ở cả 4 seed. Với cấu hình chốt, đây không phải đánh đổi (trả lời MINOR 12).
- **Batch bị bỏ qua:** baseline 8–11, phương pháp 8–9 trên khoảng 16.900 batch mỗi run, tương đương nhau (trả lời MINOR 13).
- **Val 999 của chính 8 run này** (train 18.001 track): baseline best 76,78–77,08, TB6 76,03–76,46; phương pháp best 76,88–79,48, TB6 76,23–78,33. Seed yếu trên val (42, 2026) khác seed yếu trên lock-box (7, 2026): đây là nhiễu ngẫu nhiên quanh mức lợi trung bình, không phải seed kém cố định. So với các run train 19.001 track, val 999 thấp hơn khoảng 0,7 (baseline) và 0,9 (phương pháp): thêm 1.000 track Scenario-B giúp đáng kể.
- **Thời gian mỗi epoch:** baseline 105–205 s, phương pháp 326–401 s, nhưng đo khi 4 run chạy song song nên không so được. Nếu cần số cho luận văn, đo bằng một run chạy riêng.

## Thay đổi code 25/10/2026: N1 (kiểu che frame) và E4 (STN so với crop cố định), mặc định TẮT

Theo review của thầy: N1 (xếp hạng cao nhất trong các đề xuất) và E4/N4 (ISSUE #10). Không bật cờ mới thì code chạy y như cũ.

**Cờ mới:**

| Cờ | Giá trị | Mặc định |
|---|---|---|
| `--frame-mask-mode` | `independent` / `tube` / `complementary` | `independent` (cấu hình chốt) |
| `--frame-mask-orient` | `mixed` (dải dọc hoặc ngang) / `vertical` (chỉ dải dọc) | `mixed` (cấu hình chốt) |
| `--fixed-stn` | 6 số `a,b,tx,c,d,ty` | tắt |

**Thiết kế N1.** Ở cả ba kiểu, mỗi frame bị che với xác suất biên 0,2, và bề rộng dải che có cùng phân bố (1/16 đến 1/8 bề ngang). Các kiểu chỉ khác nhau ở quan hệ giữa các frame của cùng một track:
- **Độc lập** (`independent`): mỗi frame tự bốc có bị che hay không và che ở đâu. Đây là cấu hình chốt.
- **Tube** (`tube`): bốc track với xác suất 0,2, rồi che **cùng một dải** trên cả 5 frame. Vùng bị che mất hẳn khỏi track, model phải đoán từ ngữ cảnh. VideoMAE dùng kiểu che này để ngăn rò rỉ thông tin giữa các frame.
- **Bổ sung** (`complementary`): bốc track với xác suất 0,2. Bề ngang ảnh chia thành 5 ô, mỗi frame che một dải dọc trong một ô riêng. Các dải không chồng nhau, nên cột nào cũng còn thấy ở ít nhất 4 frame.
- Trong N1 cả ba kiểu đều chỉ dùng dải dọc (`--frame-mask-orient vertical`), vì kiểu bổ sung chỉ định nghĩa được với dải dọc. Cấu hình chốt (độc lập, dải dọc hoặc ngang) đã có sẵn 4 run lock-box combo3 để đối chiếu.
- Cặp so sánh sạch nhất là **tube với bổ sung**: cùng tỷ lệ track bị che, cùng số frame, cùng kích thước dải, chỉ khác vị trí tương đối giữa các frame. Kiểu độc lập còn khác ở chỗ số frame bị che trong một track dao động từ 0 đến 5 (trung bình 1).

**Thiết kế E4.** Crop cố định là **trung bình phép biến đổi mà STN đã học**, đo trên val 999 từ 4 model phương pháp (lock-box, epoch 30): θ = [[0,782; 0,025; 0,005], [0,062; 0,606; 0,169]]. Tức là phóng ngang 1,28×, phóng dọc 1,65×, dịch xuống một chút. Mọi điểm lấy mẫu đều nằm trong ảnh nên không sinh viền đen. Theo từng mẫu, STN học được chỉ dao động quanh giá trị này với độ lệch chuẩn 0,04 (a, d, tx) và 0,11 (ty). So với "crop 1,3× ở tâm" mà review gợi ý, đây là phép thử mạnh hơn cho giả thuyết "STN chỉ là một crop cố định". Phép biến đổi được lưu trong checkpoint (buffer `fixed_theta`). Model có 30.793.702 tham số, ít hơn bản có STN 283.974 tham số (bằng đúng số tham số của bản `--no-stn`).

**Kiểm tra (25/10/2026):**
- `tests/test_eval_determinism.py`: **14/14** test đạt. 7 test mới kiểm tra:
  - kiểu độc lập cho kết quả **giống từng bit** code cũ, tức code đã train các checkpoint lock-box;
  - tube che giống nhau trên mọi frame;
  - các dải của kiểu bổ sung không chồng nhau;
  - cả ba kiểu có cùng tỷ lệ che biên 0,2;
  - giá trị cờ không hợp lệ bị từ chối;
  - crop cố định áp đúng phép biến đổi đã cho;
  - `eval_arms.py` dựng đúng kiến trúc từ checkpoint.
- `tools/regression_check.py`: vẫn **PASS** so với code gốc (train loss 4,1085 → 2,8737; val loss 3,3356 → 2,7804).
- Chạy thử trên dữ liệu nhỏ: 6 cấu hình đều train được. Cờ sai (`complementary` mà không có `vertical`; đặt kiểu che nhưng không có `--frame-mask`; `--fixed-stn` không đủ 6 số; `--fixed-stn` đi cùng `--no-stn`) thì dừng trong 2–4 giây, trước khi quét dữ liệu.
- **Chấm lại checkpoint lock-box bằng code mới** (seed 42, 43 của cả hai nhánh, 14 checkpoint, trên máy local): combo3 trùng số cũ ở cả 3 thước đo. Có 3/28.000 dự đoán khác server, đều là các trường hợp hai ký tự gần hoà (chênh log-prob giữa lớp nhất và nhì ≤ 0,008, ví dụ T/Y của `track_13651`). CPU và Apple GPU cho cùng kết quả với nhau. Lệch lớn nhất là 0,1 điểm trên một checkpoint (`base_s43_ep30`: 77,3 so với 77,4 trên server). Đây là sai số số học giữa các phần cứng, không phải do code. **Vì vậy phải chấm N1/E4 trên chính server đã chấm lock-box.**

## N1 + E4: lệnh chạy (giao thức lock-box, 21 run trên server mới)

> **Cập nhật 25/10: server `fierce-ibex` đã trả trước khi chạy N1/E4.**
> - **Kết quả lock-box không bị ảnh hưởng.** Mọi file (56 checkpoint, log, CSV, `lockbox_eval.*`, manifest) đã có trong `results_new/`.
> - **Code đóng băng được dựng lại** ở `results_new/code_lockbox_frozen.tgz`, từ bản sao lưu tự động của Claude Code. Giải nén ra và kiểm tra: 27/27 file (24 code + 3 split) khớp SHA-256 trong `freeze_manifest.json`. Gói này thay cho bước "lưu code trên server" đã không làm được.
> - **N1/E4 chạy trên server khác**, nên thêm nhánh `restran_lockbox_combo3_rerun` (cấu hình chốt, 3 seed) chạy trên **cùng máy** làm nhánh tham chiếu. Nhánh "STN học được" của E4 và nhánh "che độc lập dọc + ngang" của N1 đều chính là cấu hình chốt. Nếu so với 4 run cũ thì mọi so sánh đó lẫn cả khác biệt về máy (GPU, phiên bản PyTorch và albumentations; server cũ không ghi phiên bản albumentations). Nhánh này còn cho biết mức lợi của phương pháp có lặp lại trên máy khác không. Nếu cần tiết kiệm GPU thì bỏ 3 lệnh `combo3_rerun` (xem cách chấm ở cuối mục).
> - Chạy cả 21 run trên **cùng một server**, rồi chấm mọi nhánh trên server đó.

Giao thức giống 8 run lock-box: train 18.001 track, val 999, loại lock-box, lưu 6 epoch cuối. Cấu hình nền là combo3; mỗi nhánh chỉ khác combo3 ở đúng một điểm. Seed 42, 43, 7. Nhánh tham chiếu cho mọi so sánh là `combo3_rerun`; 8 run lock-box cũ (baseline, combo3, 4 seed) vẫn được chấm cùng để đối chiếu.

| Tên run | Khác combo3 ở | Để trả lời |
|---|---|---|
| `restran_lockbox_combo3_rerun` | không khác | Nhánh tham chiếu trên cùng máy; lặp lại combo3 trên máy khác |
| `restran_lockbox_n1_nomask` | bỏ `--frame-mask` | Che frame có tác dụng không (cũng là 1 dòng của E2 leave-one-out) |
| `restran_lockbox_n1_tube` | che tube, dải dọc | |
| `restran_lockbox_n1_indepv` | che độc lập, chỉ dải dọc | Độc lập so với tube/bổ sung; so với combo3 thì thấy vai trò của dải ngang |
| `restran_lockbox_n1_comp` | che bổ sung, dải dọc | |
| `restran_lockbox_e4_nostn` | `--no-stn` | STN có cần không |
| `restran_lockbox_e4_fixstn` | `--fixed-stn` (θ trung bình) | STN có hơn một crop cố định không |

**Đăng ký trước cách đọc kết quả** (viết trước khi chạy, để tránh kể chuyện sau khi thấy số liệu). Thước đo chính là Last-6 trên lock-box, CI 95% bootstrap theo track. Kết quả N1/E4 **không** làm thay đổi cấu hình chốt, chỉ dùng để mô tả cơ chế:
- **N1, so sánh chính: bổ sung − tube.**
  - Dương, CI không chứa 0, và bổ sung/độc lập hơn nomask nhiều hơn tube hơn nomask: che frame có tác dụng vì **buộc fusion lấy phần bị che từ frame khác**. Đây là cơ chế của hướng (b).
  - Tube ≈ độc lập ≈ bổ sung > nomask: chỉ là điều chuẩn kiểu Random Erasing; cơ chế "giữa các frame" không được ủng hộ.
  - Tube > bổ sung: che có tác dụng vì buộc model **suy luận từ ngữ cảnh** (như MAE), không phải nhờ thông tin bổ sung giữa các frame.
- **E4** (so với `combo3_rerun`, tức STN học được):
  - Crop cố định ≈ STN (CI chứa 0): chưa thấy tính thích nghi theo từng ảnh của STN đóng góp gì. STN thực chất hoạt động như một phép zoom cố định.
  - Crop cố định < STN: tính thích nghi theo từng ảnh có ích.
  - Không STN < crop cố định ≈ STN: phần có ích là **phóng to vùng ký tự**, không phải căn chỉnh.
  - Không STN ≈ STN: STN không cần thiết.
- **Giới hạn:** 3 seed mỗi nhánh chỉ đủ phát hiện chênh lệch lớn ở tầng seed (độ lệch giữa các seed của combo3 khoảng 1,1). Chênh dưới khoảng 1 điểm chỉ nên đọc như xu hướng. Nếu kết quả nằm sát ranh giới thì thêm seed 2026.

### Chuẩn bị server mới

Máy mới chưa có gì, nên phải đưa lên toàn bộ code, các file split và dữ liệu (không chỉ 6 file đã sửa). Nên thuê đúng loại GPU cũ (Tesla V100 32 GB) nếu có.

**1. Trên máy local** (nén code + split, và dữ liệu nếu server chưa có; dữ liệu khoảng 1,1 GB):

```bash
cd /Users/nguyentienphat/Learning/Paper/NewClaude/restran
COPYFILE_DISABLE=1 tar czf restran_code.tgz --exclude __pycache__ train.py configs src tools tests splits
COPYFILE_DISABLE=1 tar czf restran_data.tgz dataset/data/train
scp -P <PORT> restran_code.tgz restran_data.tgz root@<IP>:/workspace/
```

**2. Trên server:**

```bash
mkdir -p /workspace/restran && cd /workspace/restran
tar xzf /workspace/restran_code.tgz
tar xzf /workspace/restran_data.tgz
mkdir -p results
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install opencv-python-headless albumentations==2.0.8 scipy
python -c "import sys, torch, albumentations, cv2, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version(), '| gpu', torch.cuda.get_device_name(0), '| albumentations', albumentations.__version__, '| cv2', cv2.__version__, '| numpy', numpy.__version__)" | tee results/env_n1e4.txt
```

`albumentations==2.0.8` là bản trên máy local, nơi đã chạy mọi test. Dòng cuối ghi lại môi trường vào `results/env_n1e4.txt`; nhớ gửi file này cùng kết quả.

**3. Kiểm tra trên server:**

```bash
md5sum train.py configs/config.py src/data/dataset.py src/data/transforms.py src/training/trainer.py src/models/restran.py src/models/components.py tools/eval_arms.py tests/test_eval_determinism.py
python tools/make_lockbox.py --verify            # phải in: ✅ Khớp ... sha256 09f90d2b...
python tests/test_eval_determinism.py            # phải in 14 dòng OK
```

md5 phải trùng bảng ở mục "Các file code đã sửa" bên dưới. Đầu log mỗi run phải có `🔒 Loại 1000 track khỏi train, 0 khỏi val`, và dòng `FRAME_MASK_MODE: ... | FRAME_MASK_ORIENT: ... | FIXED_STN: ...` đúng như tên run.

**4. Trên máy local, đưa checkpoint tham chiếu lên** (56 file, khoảng 7,4 GB; chỉ cần cho bước chấm, nên chạy song song với đợt 1):

```bash
cd /Users/nguyentienphat/Learning/Paper/NewClaude/restran
scp -P <PORT> results_new/restran_lockbox_base_s*.pth results_new/restran_lockbox_combo3_s*.pth root@<IP>:/workspace/restran/results/
```

### Phần A: E4 (9 run, chạy trước)

3 nhánh × 3 seed. Mỗi đợt là một seed, gồm 3 run song song: STN học được (`combo3_rerun`, tức cấu hình chốt chạy lại trên máy này), không STN, crop cố định. Xong đợt nào là có ngay một bộ so sánh trọn vẹn cho seed đó. Mỗi run chỉ dùng khoảng 1,8 GB bộ nhớ GPU.

Kiểm tra đầu log:
- Mọi run: `🔒 Loại 1000 track khỏi train, 0 khỏi val`.
- `combo3_rerun`: `USE_STN: True`, `31,077,676 total params`.
- `e4_nostn`: `USE_STN: False`, `30,793,702 total params`.
- `e4_fixstn`: `USE_STN: fixed affine (--fixed-stn)`, `FIXED_STN: 0.782,0.025,0.005,0.062,0.606,0.169`, `30,793,702 total params`.

**Đợt E4-1** (seed 42, 3 terminal)

```bash
python -u train.py -n restran_lockbox_combo3_rerun_s42 --seed 42 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_combo3_rerun_s42_train.log

python -u train.py -n restran_lockbox_e4_nostn_s42 --seed 42 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale --no-stn \
    2>&1 | tee results/restran_lockbox_e4_nostn_s42_train.log

python -u train.py -n restran_lockbox_e4_fixstn_s42 --seed 42 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    --fixed-stn 0.782,0.025,0.005,0.062,0.606,0.169 \
    2>&1 | tee results/restran_lockbox_e4_fixstn_s42_train.log
```

**Đợt E4-2** (seed 43, 3 terminal)

```bash
python -u train.py -n restran_lockbox_combo3_rerun_s43 --seed 43 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_combo3_rerun_s43_train.log

python -u train.py -n restran_lockbox_e4_nostn_s43 --seed 43 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale --no-stn \
    2>&1 | tee results/restran_lockbox_e4_nostn_s43_train.log

python -u train.py -n restran_lockbox_e4_fixstn_s43 --seed 43 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    --fixed-stn 0.782,0.025,0.005,0.062,0.606,0.169 \
    2>&1 | tee results/restran_lockbox_e4_fixstn_s43_train.log
```

**Đợt E4-3** (seed 7, 3 terminal)

```bash
python -u train.py -n restran_lockbox_combo3_rerun_s7 --seed 7 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_combo3_rerun_s7_train.log

python -u train.py -n restran_lockbox_e4_nostn_s7 --seed 7 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale --no-stn \
    2>&1 | tee results/restran_lockbox_e4_nostn_s7_train.log

python -u train.py -n restran_lockbox_e4_fixstn_s7 --seed 7 --num-workers $(( $(nproc) / 3 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-dropout 0.1 --grayscale \
    --fixed-stn 0.782,0.025,0.005,0.062,0.606,0.169 \
    2>&1 | tee results/restran_lockbox_e4_fixstn_s7_train.log
```

### Phần B: N1 (12 run)

4 nhánh × 3 seed. Mỗi đợt là một seed, gồm 4 run song song.

**Đợt N1-1** (seed 42, 4 terminal)

```bash
python -u train.py -n restran_lockbox_n1_nomask_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_nomask_s42_train.log

python -u train.py -n restran_lockbox_n1_tube_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-mode tube --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_tube_s42_train.log

python -u train.py -n restran_lockbox_n1_indepv_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_indepv_s42_train.log

python -u train.py -n restran_lockbox_n1_comp_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-mode complementary --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_comp_s42_train.log
```

**Đợt N1-2** (seed 43, 4 terminal)

```bash
python -u train.py -n restran_lockbox_n1_nomask_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_nomask_s43_train.log

python -u train.py -n restran_lockbox_n1_tube_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-mode tube --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_tube_s43_train.log

python -u train.py -n restran_lockbox_n1_indepv_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_indepv_s43_train.log

python -u train.py -n restran_lockbox_n1_comp_s43 --seed 43 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-mode complementary --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_comp_s43_train.log
```

**Đợt N1-3** (seed 7, 4 terminal)

```bash
python -u train.py -n restran_lockbox_n1_nomask_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_nomask_s7_train.log

python -u train.py -n restran_lockbox_n1_tube_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-mode tube --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_tube_s7_train.log

python -u train.py -n restran_lockbox_n1_indepv_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_indepv_s7_train.log

python -u train.py -n restran_lockbox_n1_comp_s7 --seed 7 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --exclude-tracks splits/lockbox_tracks.json --save-last-k 6 \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 \
    --frame-mask 0.2 --frame-mask-mode complementary --frame-mask-orient vertical \
    --frame-dropout 0.1 --grayscale \
    2>&1 | tee results/restran_lockbox_n1_comp_s7_train.log
```

### Phần C: E5 (tuỳ chọn, 1 run)

Cấu hình gốc của tác giả ở fp32 (ISSUE #4), giao thức val 999, không dùng lock-box. Chậm hơn fp16 khoảng 2–3 lần. Chạy kèm vào đợt nào còn chỗ trống.

```bash
python -u train.py -n restran_orig999_fp32_s42 --seed 42 --num-workers $(( $(nproc) / 4 )) \
    --epochs 30 --lr 5e-4 --grad-clip 5.0 --amp-dtype fp32 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    2>&1 | tee results/restran_orig999_fp32_s42_train.log
```

### Chấm E4 (sau Phần A)

Chạy trên chính server đã train. Hai nhánh `base`, `combo3` (8 run lock-box cũ) cần 56 checkpoint từ bước 4 ở trên. Nếu chưa đưa lên thì xoá hai dòng `--arm base:...` và `--arm combo3:...`.

```bash
python tools/eval_arms.py --seeds 42 43 7 --frames-analysis \
    --splits splits/lockbox_tracks.json splits/val_tracks.json \
    --arm base:restran_lockbox_base:42,43,7,2026 \
    --arm combo3:restran_lockbox_combo3:gray:42,43,7,2026 \
    --arm combo3new:restran_lockbox_combo3_rerun:gray \
    --arm nostn:restran_lockbox_e4_nostn:gray \
    --arm fixstn:restran_lockbox_e4_fixstn:gray \
    --ref combo3new --pairs nostn:fixstn --out results/arms_eval_e4.json \
    2>&1 | tee results/arms_eval_e4.log
```

Kết quả lưu ở `results/arms_eval_e4.json` và `results/arms_eval_e4_per_track.npz`.

- **Kiểm tra nhanh:** hai dòng `base` và `combo3` trên lock-box phải gần đúng các số trong `results_new/lockbox_eval.json` (epoch cuối 77,60 và 79,12; Last-6 77,07 và 78,71). Do khác phần cứng, mỗi checkpoint có thể lệch tối đa khoảng 0,1 (vài dự đoán gần hoà bị lật). Lệch nhiều hơn thì dừng lại kiểm tra.
- Dòng `combo3 − combo3new` cho biết cấu hình chốt trên máy mới khác máy cũ bao nhiêu.
- **Muốn xem sớm sau đợt E4-1:** chạy cùng lệnh với `--seeds 42`, đổi `:42,43,7,2026` của hai nhánh `base`, `combo3` thành `:42`, và đổi `--out` thành `results/arms_eval_e4_s42.json`.

### Chấm toàn bộ (sau Phần A và Phần B)

Chạy trên chính server mới, sau khi đã đưa 56 checkpoint tham chiếu lên (bước 4). Lệnh in bảng cho lock-box và val 999, rồi lưu `results/arms_eval.json` và `results/arms_eval_per_track.npz`:

```bash
python tools/eval_arms.py --seeds 42 43 7 --frames-analysis \
    --splits splits/lockbox_tracks.json splits/val_tracks.json \
    --arm base:restran_lockbox_base:42,43,7,2026 \
    --arm combo3:restran_lockbox_combo3:gray:42,43,7,2026 \
    --arm combo3new:restran_lockbox_combo3_rerun:gray \
    --arm nomask:restran_lockbox_n1_nomask:gray \
    --arm tube:restran_lockbox_n1_tube:gray \
    --arm indepv:restran_lockbox_n1_indepv:gray \
    --arm comp:restran_lockbox_n1_comp:gray \
    --arm nostn:restran_lockbox_e4_nostn:gray \
    --arm fixstn:restran_lockbox_e4_fixstn:gray \
    --ref combo3new --pairs tube:comp indepv:comp tube:indepv nomask:tube nomask:indepv nomask:comp nostn:fixstn \
    2>&1 | tee results/arms_eval.log
```

Kiểm tra nhanh như ở mục "Chấm E4": hai dòng `base` và `combo3` gần đúng `results_new/lockbox_eval.json`, lệch tối đa khoảng 0,1 mỗi checkpoint.

> **Cập nhật 31/10: N1 chạy trên server khác server E4** (đã upload lại source). Server mới không có 21 checkpoint `combo3_rerun` (máy local cũng không có, vì E4 chỉ tải log về), nên lệnh "Chấm toàn bộ" ở trên báo `Nhánh combo3new: thiếu 21 checkpoint`. Chấm N1 bằng lệnh dưới, chỉ dùng 4 nhánh N1. Mọi so sánh đăng ký trước của N1 (bổ sung − tube; độc lập so với tube và bổ sung; che so với không che) đều nằm trong 4 nhánh này và cùng một server. Chỉ còn mất một so sánh phụ: `indepv` so với cấu hình chốt (vai trò của dải ngang). So sánh này sẽ đặt cạnh số của `combo3new` trong `arms_eval_e4.json` (Last-6 78,29) để tham khảo, không có CI.
>
> ```bash
> cd /workspace/restran
> python tools/eval_arms.py --seeds 42 43 7 --frames-analysis \
>     --splits splits/lockbox_tracks.json splits/val_tracks.json \
>     --arm nomask:restran_lockbox_n1_nomask:gray \
>     --arm tube:restran_lockbox_n1_tube:gray \
>     --arm indepv:restran_lockbox_n1_indepv:gray \
>     --arm comp:restran_lockbox_n1_comp:gray \
>     --ref nomask --pairs tube:comp indepv:comp tube:indepv \
>     --out results/arms_eval_n1.json \
>     2>&1 | tee results/arms_eval_n1.log
> ```

---

## Run #71 → #79 — E4: STN học được, crop cố định, không STN (28/10/2026, server mới)

Lệnh train: mục "Phần A: E4". Lệnh chấm: mục "Chấm E4", **bỏ hai nhánh cũ** `base`, `combo3` (không đưa 56 checkpoint cũ lên):

```bash
python tools/eval_arms.py --seeds 42 43 7 --frames-analysis \
    --splits splits/lockbox_tracks.json splits/val_tracks.json \
    --arm combo3new:restran_lockbox_combo3_rerun:gray \
    --arm nostn:restran_lockbox_e4_nostn:gray \
    --arm fixstn:restran_lockbox_e4_fixstn:gray \
    --ref combo3new --pairs nostn:fixstn --out results/arms_eval_e4.json \
    2>&1 | tee results/arms_eval_e4.log
```

| Run | Tên | Seed |
|---|---|---|
| #71–#73 | `restran_lockbox_combo3_rerun` (STN học được = cấu hình chốt) | 42, 43, 7 |
| #74–#76 | `restran_lockbox_e4_nostn` | 42, 43, 7 |
| #77–#79 | `restran_lockbox_e4_fixstn` | 42, 43, 7 |

Kết quả: `results_new/arms_eval_e4.json` (md5 `621b292f4bc77b53cb00f89e1d3325b9`).

**Kiểm tra:**
- sha256 của split lock-box trong kết quả là `09f90d2b…`, khớp `make_lockbox.py --verify`.
- `eval_arms.py` nhận kiến trúc từ checkpoint: `stn`, `none`, `fixed` đúng cho 3 nhánh. Vậy checkpoint `fixstn` có buffer `fixed_theta`, checkpoint `nostn` không có STN.
- `arms_eval_e4.log` khớp JSON. Đã có `arms_eval_e4_per_track.npz`.
- **`combo3_rerun_s42` không có log train** (server không có file; chỉ có `summary.txt` và `metrics.csv`, đã tải về). Từ hai file này: `Train samples : 36002` (đã loại lock-box), val 999, 30/30 epoch, Use STN True. `aux_ctc_loss` khác 0 ở mọi epoch (CTC phụ đang bật). Val Acc epoch 30 là 78,08, khớp `eval_arms.py`. Batch bị bỏ 9 (9 do inf/nan), GradNorm lớn nhất 58,2, bộ nhớ GPU đỉnh 1.835 MB. Khối cấu hình in ra lúc bắt đầu không xem được, nhưng mọi số trên đều giống `s43`, `s7`.
- **8 log train còn lại:**
  - Cả 8 run đều có `🔒 Loại 1000 track khỏi train, 0 khỏi val`, `Loaded 18001 tracks`, `Total: 36002 samples`, val 999 track.
  - So toàn bộ khối cấu hình với `combo3_rerun_s43`: `nostn` chỉ khác ở `USE_STN: False`; `fixstn` chỉ khác ở `USE_STN: fixed affine` và `FIXED_STN: 0.782,0.025,0.005,0.062,0.606,0.169`. Số tham số: 31.077.676 (STN) và 30.793.702 (`nostn`, `fixstn`). Chênh 283.974, đúng bằng số tham số của STN.
  - Cả 8 run chạy đủ 30 epoch. Val Acc ở epoch 30 trong log khớp với số `eval_arms.py` chấm lại trên val 999 ở cả 8 run.
- **Batch bị bỏ qua:** STN 9, 9, 9; crop cố định 7, 6, 6; không STN 6, 7, 6 (trên 16.890 batch). Mọi batch bị bỏ đều do gradient inf/nan ở fp16; ngưỡng 100 không lần nào kích hoạt. GradNorm lớn nhất: khoảng 57 ở nhánh STN, 29–32 ở hai nhánh không có STN học được.
- **Thời gian:** khoảng 240 s mỗi epoch, 2,0 giờ mỗi run (3 run song song).
- **Môi trường** (server `frosty-tanager`, ghi 28/10 sau khi chạy E4; lệnh `tee` báo lỗi vì chạy từ trong `results/`, nên chép từ terminal vào `results_new/env_n1e4.txt`): Python 3.10.13, PyTorch 2.14.1+cu126, CUDA 12.6, cuDNN 9.10.2 (91002), Tesla V100-SXM2-32GB, albumentations 2.0.8, cv2 5.0.0, numpy 2.2.6. **Giống hệt `fierce-ibex`** ở GPU, Python, PyTorch, CUDA, cuDNN; albumentations của `fierce-ibex` không được ghi lại.

**Theo từng seed** (lock-box 1.000 track; cuối / Last-6 / best):

| Seed | STN học được | Crop cố định | Không STN |
|---|---|---|---|
| 42 | 79,10 / 78,25 / 78,80 | 77,40 / 76,50 / 77,30 | 74,90 / 74,22 / 74,10 |
| 43 | 79,10 / 78,57 / 79,10 | 77,30 / 76,78 / 77,10 | 75,30 / 74,90 / 75,30 |
| 7 | 78,50 / 78,05 / 78,70 | 76,90 / 76,53 / 77,20 | 76,80 / 75,70 / 76,90 |
| **TB ± std** | 78,90 ± 0,35 / **78,29 ± 0,26** / 78,87 ± 0,21 | 77,20 ± 0,26 / **76,60 ± 0,15** / 77,20 ± 0,10 | 75,67 ± 1,00 / **74,94 ± 0,74** / 75,43 ± 1,40 |

**So sánh trên lock-box** (thước đo chính đăng ký trước: Last-6, CI 95% bootstrap theo track):

| So sánh | Thước đo | Chênh | CI 95% (theo track) | Welch (theo seed) |
|---|---|---|---|---|
| Crop cố định − STN | **Last-6** | **−1,68** | [−2,78; −0,59] | p = 0,002 |
| | Cuối | −1,70 | [−2,97; −0,43] | p = 0,003 |
| | Best | −1,67 | [−2,93; −0,43] | p = 0,001 |
| Không STN − STN | **Last-6** | **−3,35** | [−4,69; −2,02] | p = 0,010 |
| | Cuối | −3,23 | [−4,67; −1,80] | p = 0,021 |
| | Best | −3,43 | [−4,93; −1,97] | p = 0,049 |
| Crop cố định − Không STN | **Last-6** | **+1,67** | [+0,39; +2,96] | p = 0,055 |
| | Cuối | +1,53 | [+0,10; +2,93] | p = 0,109 |
| | Best | +1,77 | [+0,37; +3,17] | p = 0,161 |

Cohen's d trong JSON rất lớn (−3 đến −10) vì std giữa 3 seed nhỏ; không nên dùng d để nói về độ lớn hiệu ứng.

**Val 999** (Last-6, cùng chiều với lock-box): STN 77,67 ± 0,36; crop cố định 74,64 ± 0,94; không STN 73,01 ± 0,18. Crop − STN −3,03 [−4,21; −1,92]; không STN − STN −4,66 [−6,04; −3,33]; crop − không STN +1,63 [+0,31; +2,94].

**Đọc theo đăng ký trước: "Crop cố định < STN" → tính thích nghi theo từng ảnh của STN có ích.**
- CI không chứa 0 ở cả 3 thước đo, và Welch p ≤ 0,003. Chênh 1,7 điểm, không nằm sát ranh giới, nên **không cần thêm seed 2026** cho so sánh này.
- Thêm vào đó, crop cố định > không STN (+1,67, CI không chứa 0; theo seed p = 0,055, mới là xu hướng). Vậy **lợi ích của STN (+3,35 Last-6) chia gần đều làm hai phần**: phóng to vùng ký tự (+1,67) và điều chỉnh theo từng ảnh (+1,68).
- Mức lợi của STN so với không STN (+3,35 Last-6, +3,23 epoch cuối) cùng cỡ với +2,90 mà slide của tác giả báo cáo.

**Theo số frame** (lock-box, epoch cuối; *phân tích thêm, không đăng ký trước*):

| N | STN | Crop cố định | Không STN | STN − crop [CI] | Crop − không STN [CI] |
|---|---|---|---|---|---|
| 1 | 55,48 | 51,28 | 50,23 | +4,20 [+3,39; +5,01] | +1,05 [+0,23; +1,87] |
| 2 | 68,86 | 65,52 | 64,40 | +3,34 [+2,55; +4,12] | +1,12 [+0,25; +2,01] |
| 3 | 74,33 | 71,46 | 70,30 | +2,87 [+2,00; +3,76] | +1,17 [+0,17; +2,14] |
| 4 | 77,17 | 74,84 | 73,57 | +2,33 [+1,34; +3,32] | +1,27 [+0,14; +2,43] |
| 5 | 78,90 | 77,20 | 75,67 | +1,70 [+0,43; +2,97] | +1,53 [+0,10; +2,93] |

- Phần "điều chỉnh theo từng ảnh" lớn nhất khi chỉ có 1 frame (+4,2) và giảm dần tới 5 frame (+1,7). Phần "phóng to" gần như không đổi (+1,1 đến +1,5).
- Cách hiểu hợp lý: khi có nhiều frame, fusion bù được một phần cho việc từng frame không được căn chỉnh riêng. Đây là quan sát sau khi thấy số liệu, chỉ nên viết như giả thuyết.
- Val 999 cho cùng dạng: STN − crop từ +4,40 (N = 1) giảm còn +2,77 (N = 5).

**Lặp lại cấu hình chốt trên máy mới** (so với `results_new/lockbox_eval.json`; khác máy vật lý nhưng cùng loại GPU và cùng phiên bản PyTorch/CUDA/cuDNN; chỉ để tham khảo, không có CI):

| | Máy cũ (4 seed) | Máy cũ (seed 42, 43, 7) | Máy mới (seed 42, 43, 7) |
|---|---|---|---|
| Cuối | 79,12 ± 1,14 | 79,43 | 78,90 ± 0,35 |
| Last-6 | 78,71 ± 1,15 | 79,07 | 78,29 ± 0,26 |
| 1 frame (E3) | 55,59 | — | 55,48 |

- Thấp hơn trung bình 4 seed cũ 0,2 (cuối) và 0,4 (Last-6). Mức này nhỏ hơn độ lệch giữa các seed trên máy cũ (1,1). Train không tất định nên cùng seed vẫn cho model khác nhau; so theo từng seed không có ý nghĩa.
- So với baseline của máy cũ (77,60 / 77,07): +1,30 (cuối), +1,22 (Last-6). Cả 3 seed mới (Last-6 thấp nhất 78,05) đều cao hơn cả 4 seed baseline cũ (Last-6 cao nhất 77,40). **Chiều của mức lợi lặp lại trên máy khác**, độ lớn nhỏ hơn một chút. Muốn nói chính thức "lặp lại trên máy thứ hai" thì phải chạy lại baseline trên máy mới (3 run nữa).
- Độ lệch giữa các seed trên máy mới nhỏ hơn hẳn (0,26 so với 1,15 trên máy cũ). Với 3–4 seed, std còn dao động nhiều, nên chưa kết luận được gì từ điểm này.

**Giới hạn:**
1. **"Thích nghi" gồm hai thứ chưa tách được:** (i) θ khác nhau cho từng ảnh lúc suy luận; (ii) STN được khởi tạo bằng phép đồng nhất và đổi dần trong lúc train, còn crop cố định áp mức zoom cuối ngay từ epoch 1. Phép thử rẻ để tách hai thứ này, không cần train lại: lấy checkpoint `combo3_rerun`, lúc chấm thay θ của từng ảnh bằng θ trung bình. Nếu accuracy giảm khoảng 1,7 điểm thì lợi ích đến từ (i); nếu gần như không đổi thì đến từ (ii). Cần thêm một cờ cho lúc chấm, chưa code.
2. Crop cố định là trung bình θ của chính STN (đo trên val 999 bằng 4 model lock-box cũ, không dùng lock-box). Chưa thử các crop cố định khác, nên kết luận đúng là "STN hơn crop cố định bằng trung bình của chính nó".
3. Không có log train của `combo3_rerun_s42`; đã kiểm bằng `summary.txt` và `metrics.csv` (xem phần Kiểm tra).

## Run #80 → #91 — N1: các kiểu che frame (30/10/2026, server thứ ba, khác server E4)

Lệnh train: mục "Phần B: N1". Mỗi đợt 4 run song song (`nproc/4`). Lệnh chấm: bản chỉ có 4 nhánh N1 (xem "Cập nhật 31/10" ở mục "Chấm toàn bộ"), `--ref nomask`. Kết quả ở `results_n1/`: `arms_eval_n1.json` (md5 `ce6d12c55bd1e41e7525ad07a9a2d52f`), `arms_eval_n1.log`, `arms_eval_n1_per_track.npz`, 84 checkpoint (12 GB). `arms_eval.log` trong thư mục đó là lần chấm đầu bị lỗi `Nhánh combo3new: thiếu 21 checkpoint`.

| Run | Tên | Seed | Khác cấu hình chốt ở |
|---|---|---|---|
| #80–#82 | `restran_lockbox_n1_nomask` | 42, 43, 7 | bỏ `--frame-mask` |
| #83–#85 | `restran_lockbox_n1_tube` | 42, 43, 7 | che tube, dải dọc |
| #86–#88 | `restran_lockbox_n1_indepv` | 42, 43, 7 | che độc lập, chỉ dải dọc |
| #89–#91 | `restran_lockbox_n1_comp` | 42, 43, 7 | che bổ sung, dải dọc |

**Kiểm tra:**
- 12/12 run đủ 30 epoch, 7 checkpoint mỗi run. `eval_arms.py` nhận cả 4 nhánh là `stn`.
- **11/12 log train** (thiếu `restran_lockbox_n1_nomask_s42_train.log`; run này có `summary.txt` ghi `Train samples : 36002`, 30/30 epoch, và `metrics.csv`). Cả 11 log có `🔒 Loại 1000 track khỏi train, 0 khỏi val`, 18.001 track train, 999 val, 31.077.676 tham số. So toàn bộ khối cấu hình với `indepv_s43`: `tube`, `comp` chỉ khác ở `FRAME_MASK_MODE`; `nomask` chỉ khác ở `FRAME_MASK: off` (orient về mặc định `mixed`, không dùng tới).
- Val Acc epoch 30 trong `metrics.csv` khớp số `eval_arms.py` chấm lại ở cả 12 run.
- **Batch bị bỏ:** 8–10 mỗi run, đều do inf/nan, trừ `tube_s7`: 14 batch, trong đó **4 batch vượt ngưỡng 100** (GradNorm lớn nhất 145,5). Đây là lần đầu ngưỡng 100 kích hoạt trong các run lock-box. GradNorm lớn nhất các run khác: 46–86.
- Thời gian: 304–359 s mỗi epoch (4 run song song), bộ nhớ GPU đỉnh 1.832–1.835 MB.
- **Thiếu `env_n1.txt`** và server đã trả, nên không biết GPU và phiên bản thư viện của server này. Không ảnh hưởng các so sánh giữa 4 nhánh N1 (cùng server), nhưng so với `combo3new` của E4 là so khác máy chưa rõ cấu hình.
- Các file `submission_restran_lockbox_n1_*.txt` là dự đoán trên val 999 do `train.py` tự ghi khi kết thúc (`SUBMISSION_MODE: False`), không phải tập test.

**Theo nhánh** (lock-box 1.000 track, 3 seed; TB ± std):

| Nhánh | Cuối | **Last-6** | Best | Cuối theo seed 42 / 43 / 7 |
|---|---|---|---|---|
| Không che | 78,60 ± 0,89 | **78,38 ± 0,42** | 78,23 ± 0,57 | 78,9 / 77,6 / 79,3 |
| Tube | 79,10 ± 0,26 | **78,71 ± 0,44** | 78,90 ± 1,04 | 79,0 / 79,4 / 78,9 |
| Độc lập (dọc) | 78,70 ± 0,95 | **78,49 ± 0,97** | 78,67 ± 1,10 | 78,8 / 79,6 / 77,7 |
| Bổ sung | 78,40 ± 0,46 | **78,19 ± 0,05** | 77,93 ± 0,38 | 78,5 / 77,9 / 78,8 |

**So sánh trên lock-box** (Last-6, thước đo chính; CI 95% bootstrap theo track; Welch theo seed):

| So sánh | Last-6 | Cuối | Best |
|---|---|---|---|
| **Bổ sung − tube** (so sánh chính) | **−0,52 [−1,39; +0,36]**, p = 0,18 | −0,70 [−1,73; +0,33], p = 0,10 | −0,97 [−2,10; +0,13], p = 0,25 |
| Tube − không che | +0,33 [−0,51; +1,16], p = 0,40 | +0,50 [−0,43; +1,47] | +0,67 [−0,33; +1,67] |
| Độc lập − không che | +0,11 [−0,75; +0,93], p = 0,87 | +0,10 [−0,90; +1,10] | +0,43 [−0,63; +1,47] |
| Bổ sung − không che | −0,19 [−0,96; +0,57], p = 0,52 | −0,20 [−1,20; +0,77] | −0,30 [−1,30; +0,67] |
| Bổ sung − độc lập | −0,30 [−1,17; +0,54], p = 0,65 | −0,30 [−1,37; +0,77] | −0,73 [−1,80; +0,33] |
| Độc lập − tube | −0,22 [−1,06; +0,63], p = 0,75 | −0,40 [−1,43; +0,63] | −0,23 [−1,33; +0,87] |

**Val 999** cho cùng hình ảnh (Last-6): không che 78,02; tube 78,26; độc lập 77,97; bổ sung 77,70. Mọi chênh lệch nằm trong ±0,6 và mọi CI đều chứa 0.

**Theo số frame** (lock-box, epoch cuối): N = 1: 55,75 / 55,09 / 55,34 / 55,35 (không che / tube / độc lập / bổ sung); N = 5: 78,60 / 79,10 / 78,70 / 78,40. Không nhánh nào khác nhánh nào ở bất kỳ N nào quá 0,9 điểm.

**Đọc theo đăng ký trước: không rơi vào trường hợp nào trong ba trường hợp đã viết.**
- *Bổ sung > tube:* không. Ước lượng còn ngược chiều (−0,52) và CI chứa 0. **Cơ chế "che buộc fusion lấy phần bị che từ frame khác" không được ủng hộ.**
- *Tube ≈ độc lập ≈ bổ sung > không che:* vế đầu đúng, nhưng không kiểu che nào hơn không che (cả ba CI chứa 0, ước lượng từ −0,19 đến +0,33). Vậy cũng không thấy tác dụng điều chuẩn kiểu Random Erasing.
- *Tube > bổ sung:* ước lượng cùng chiều, nhưng CI chứa 0; chỉ là xu hướng nhỏ hơn ngưỡng 1 điểm đã ghi trong phần giới hạn.
- **Kết luận: trong cấu hình chốt, che frame dải dọc không có tác dụng đo được trên lock-box, dù che kiểu nào.** CI theo track loại được tác dụng lớn hơn khoảng +1,2 điểm (cận trên cao nhất +1,16 cho tube − không che). Kết quả không nằm sát ranh giới, nên không cần thêm seed 2026.

**Đối chiếu với cấu hình chốt** (`combo3new` từ E4; **khác server**, không có CI, chỉ để tham khảo):

| | Cuối | Last-6 | Best | N = 1 |
|---|---|---|---|---|
| Cấu hình chốt (`combo3new`, server E4) | 78,90 | 78,29 | 78,87 | 55,48 |
| Không che (N1) | 78,60 | 78,38 | 78,23 | 55,75 |
| Độc lập, chỉ dải dọc (N1) | 78,70 | 78,49 | 78,67 | 55,34 |

- **Bỏ che frame khỏi cấu hình chốt không làm giảm accuracy** (78,38 so với 78,29 Last-6). Đây chính là dòng "bỏ che frame" của E2 (leave-one-out): đóng góp ước lượng gần 0.
- Bỏ dải ngang (độc lập, chỉ dải dọc) cũng không đổi gì (78,49 so với 78,29).
- **Độ bền khi chỉ có 1 frame (E3, +5,0 so với baseline) không đến từ che frame:** không che vẫn đạt 55,75 ở N = 1, ngang cấu hình chốt (55,48). Nếu là do huấn luyện, ứng viên còn lại là frame dropout và CTC phụ từng frame; E2 cần đo hai thành phần này.
- Kết quả này khớp với ablation phát triển (REPORT mục 5): che frame một mình giúp #42 (#46, +0,9 Last-6, 1 seed), nhưng ghép với frame dropout thì không cộng thêm (#49, #50 ngang #45). Tức là khi đã có frame dropout, che frame thành thừa.

**Hệ quả cho bài báo:**
- Cấu hình chốt giữ nguyên (theo đăng ký trước, N1 không đổi cấu hình). Nhưng **không được viết** rằng che độc lập từng frame là lý do phương pháp tốt hơn, hay rằng nó buộc fusion khai thác thông tin bổ sung giữa các frame.
- ISSUE #8 (novelty, đặc thù đa khung hình): N1 không cung cấp được bằng chứng cho cơ chế đặc thù đa khung hình. Claim chính vẫn là E3 dạng B ("bền hơn khi thiếu frame"), cộng với phân tích STN của E4.
- E2 trở nên quan trọng hơn: cần biết +1,6 điểm và độ bền ở N = 1 đến từ thành phần nào (frame dropout, CTC phụ, augment nhất quán, ảnh xám, sửa degradation).

---

## Việc chưa làm

- [x] ~~Constrained decoding theo pattern biển số~~ → đo được +0,00–0,05 (mục Phân tích 01/10, §1)
- [x] ~~Bỏ `ChannelShuffle` + augment nhất quán~~ → run #13
- [x] ~~Vòng E1–E4~~ → run #39–#42
- [ ] Xin bản LRLPR-26 chính thức có nhãn test: ký `license-agreement.pdf` (repo `github.com/raysonlaroca/lrlpr-26-dataset`), gửi tới `rayson@ppgia.pucpr.br` với tiêu đề `[LRLPR-26] Signed License Agreement`, **từ email trường** (.edu hoặc tương tự, không dùng Gmail). Phản hồi trong khoảng 5 ngày làm việc. Bản chính thức có 20.000 track train + 3.000 track test có nhãn
- [x] ~~Vòng hires, fusion temporal~~ → run #43, #44: không cải thiện
- [x] ~~3 seed `degcal` + 3 seed `degcal + aux` theo giao thức 999~~ → thay bằng seed của combo3 (cấu hình chốt sau run #51)
- [ ] Chấm mọi checkpoint giao thức 999 trên test 3.000 track khi có nhãn
- [x] ~~Vòng frame-dropout / frame-mask / grayscale / clean-geo~~ → run #45–#48: fmask, fdrop, gray thắng
- [x] ~~Vòng combo~~ → run #49–#51: combo3_999 đạt 79,28 (TB6 78,33) trên giao thức của tác giả
- [ ] Kết quả `restran_combo3_40ep` (nếu đã chạy)
- [x] ~~Vòng seed cuối~~ → run #52–#54: phương pháp 78,88 ± 0,40 so với baseline 77,14 ± 0,45 (p = 0,008)
- [x] ~~Kiểm tra cấu hình gốc của tác giả~~ → run #55 nổ ở ep10 (cùng với run #1: 2/2 lần nổ)
- [x] ~~Seed 2026~~ → run #56, #57
- [x] ~~Kết quả `restran_base999_s43`~~ → run #58; bảng chính 4 vs 4 seed
- [x] ~~Thí nghiệm 40 epoch, 4 seed~~ → run #59–#62: +0,24 TB6, p = 0,46 → giữ 30 epoch
- [x] ~~Report tổng hợp gửi thầy~~ → `documents/REPORT.md`
- [x] ~~Ablation `--no-stn` để đối chiếu +2,90 điểm mà slide báo~~ → run #74–#76: STN hơn không STN +3,35 Last-6 trên lock-box, cùng cỡ với slide
- [x] ~~Code lock-box + test + regression check~~ (23/10): 7/7 test OK, regression PASS so với code gốc
- [x] ~~8 run lock-box, đóng băng, chấm~~ → run #63–#70: hướng (b), E3 dạng B
- [x] ~~Xác nhận base_s42 loại lock-box; cấu hình 4 run combo3~~ → cả 8 run: Train samples 36002; 4 run combo3 cùng cấu hình
- [ ] E5 (cấu hình gốc fp32): không có kết quả trong `results_new/`; lệnh đã đưa vào đợt 6 của N1/E4
- [x] ~~Code N1 (kiểu che) + E4 (crop cố định)~~ (25/10): 14/14 test, regression PASS
- [x] ~~Trả server `fierce-ibex`~~ (25/10): kết quả lock-box đã ở `results_new/`; code đóng băng dựng lại ở `results_new/code_lockbox_frozen.tgz` (27/27 khớp manifest)
- [x] ~~Thuê server mới, chạy E4, chấm E4~~ → run #71–#79 (28/10): crop cố định < STN → thích nghi theo từng ảnh có ích; phóng to và thích nghi mỗi phần khoảng +1,7
- [x] ~~Gửi file E4 còn thiếu~~ (28/10): log chấm, npz, 8/9 log train, summary + metrics của `combo3_rerun_s42` (không có log), môi trường (chép từ terminal)
- [x] ~~Chạy N1~~ → run #80–#91 (30/10, server khác E4): che frame dải dọc không có tác dụng đo được ở cả 3 kiểu; bổ sung − tube −0,52 [−1,39; +0,36]. Không còn `env_n1.txt` và log `nomask_s42` (server đã trả)
- [ ] **E2 (leave-one-out, 3 seed)**, ưu tiên bỏ frame dropout và bỏ CTC phụ: N1 cho thấy độ bền ở N = 1 không đến từ che frame
- [ ] (Tuỳ chọn) E4b: chấm model STN với θ trung bình thay cho θ từng ảnh, để tách "thích nghi lúc suy luận" với "STN đổi dần lúc train" (cần thêm cờ lúc chấm). Checkpoint `combo3_rerun` chỉ nằm trên server E4 (mất nếu đã trả server), nhưng làm được trên máy local với 4 model combo3 lock-box trong `results_new/`
- [ ] (Tuỳ chọn) Chạy lại baseline 3 seed trên máy mới nếu bài báo cần claim "lặp lại trên máy thứ hai"
- [x] ~~Sửa REPORT theo review~~ (25/10): lock-box là kết quả chính; ISSUE #2, #4–#10 và MINOR 12–19 đã xử lý (bảng đối chiếu ở mục 10 của REPORT)
- [x] ~~Báo cáo riêng phần lock-box gửi thầy~~ (25/10): `documents/REPORT_lockbox.md`
- [ ] Điền 1 chỗ còn trống trong REPORT mục 10: ISSUE #9 (đã gửi diff 5 file chưa). ISSUE #11 đã điền: API chưa xong, đang ưu tiên train model
- [ ] Prototype API `/predict` (ISSUE #11): chưa xong, báo cáo thầy sau
- [ ] Các việc còn lại theo review của thầy: E2 (leave-one-out; dòng bỏ che frame đã nằm trong N1), E6, N3

---

## Các file code đã sửa — nhớ đồng bộ lên server

Sau mỗi lần sửa phải upload lại, nếu không sẽ gặp `train.py: error: unrecognized arguments`:

```bash
scp -P <PORT> train.py root@<IP>:/workspace/restran/
scp -P <PORT> configs/config.py root@<IP>:/workspace/restran/configs/
scp -P <PORT> src/data/dataset.py src/data/transforms.py root@<IP>:/workspace/restran/src/data/
scp -P <PORT> src/training/trainer.py root@<IP>:/workspace/restran/src/training/
scp -P <PORT> src/models/restran.py src/models/components.py src/models/sr.py src/models/edvr.py root@<IP>:/workspace/restran/src/models/
scp -P <PORT> tools/make_lockbox.py tools/eval_lockbox.py tools/freeze_manifest.py tools/regression_check.py tools/eval_arms.py root@<IP>:/workspace/restran/tools/
ssh -p <PORT> root@<IP> "mkdir -p /workspace/restran/tests"
scp -P <PORT> tests/test_eval_determinism.py root@<IP>:/workspace/restran/tests/
scp -P <PORT> splits/lockbox_tracks.json root@<IP>:/workspace/restran/splits/
```

Đối chiếu bằng md5 (`md5sum` trên Linux, `md5 -q` trên macOS) — hai bên phải trùng:

| File | md5 (25/10/2026) |
|---|---|
| `train.py` | `d9aec99ca097d2411189e2a58e727dd9` |
| `configs/config.py` | `652c96987d42128437b6467b824454f7` |
| `src/data/dataset.py` | `ffe6bac426de63cd6138bb3e5ae02bf6` |
| `src/data/transforms.py` | `7efed8c09cd5e413c3451e4e26d7a4db` |
| `src/training/trainer.py` | `28631499578a575db1d7d34555d871f7` |
| `src/models/restran.py` | `ec0e0d21ce95ccd25235b64d5b0ed846` |
| `src/models/components.py` | `c43c5cd401db63959871d478d673d5af` |
| `src/models/sr.py` | `53836e20f56b1c423e2da07a3cd890b5` |
| `src/models/edvr.py` | `4cdb1b931be313f331b0d9e29c07c22e` |
| `tools/make_lockbox.py` | `cdbb070eccbac8da359e9b28a6e1c7c8` |
| `tools/eval_lockbox.py` | `373702560b55b1a422993137bedf06e2` |
| `tools/freeze_manifest.py` | `4f79914a31731d06d9f92cc10f7c5a3b` |
| `tools/regression_check.py` | `bbb9f022524c804e200efeb7cae4c4b1` |
| `tools/eval_arms.py` (mới) | `fb1685502c29efdaabb6cfbe3a1d3300` |
| `tests/test_eval_determinism.py` | `d5d8ca4f64067d09880058571d76ad0c` |
| `splits/lockbox_tracks.json` | `3e6014aa7fc254409eeaff57e779c70a` |

---

## Thay đổi code (12/08/2026)

Các cờ thêm vào `train.py`, mặc định **giữ nguyên hành vi baseline**:

| Cờ | Mặc định | Tác dụng |
|---|---|---|
| `--grad-clip FLOAT` | 5.0 | Ngưỡng clip gradient |
| `--amp-dtype {fp16,bf16}` | fp16 | Kiểu mixed precision |
| `--skip-grad-norm FLOAT` | 0 (tắt) | Bỏ batch có grad norm trước clip vượt ngưỡng |
| `--skip-abort-ratio FLOAT` | 0.2 | Dừng hẳn khi tỉ lệ batch bị bỏ trong một epoch vượt ngưỡng |
| `--log-batch-loss` | tắt | Ghi loss từng batch ra `{tên}_batches.csv` |
| `--split-ratio` / `--val-split-file` | 0.9 / `splits/val_tracks.json` | Chỉ có tác dụng khi file split chưa tồn tại |
| `--layout-balance` | 1.0 (tắt) | Trọng số lấy mẫu cho biển Brazilian — **đã thử 1.73, không hiệu quả (run #10)** |
| `--pretrained` | tắt | Khởi tạo ResNet34 bằng trọng số ImageNet thay vì train from scratch |
| `--resume PATH` | — | Nạp trọng số từ checkpoint trước khi train |

Sửa lỗi và thêm chẩn đoán:

- **Cầu dao dừng sớm** thay vì chạy tiếp hàng chục epoch trống. Không rollback — nguyên nhân cần sửa ở config, không phải chữa cháy lúc chạy.
- **LR không còn đóng băng**: scheduler bước theo số batch đã xử lý, chỉ giữ lại điều kiện cũ khi AMP scaler giảm scale.
- **Chẩn đoán spike**: lần đầu mỗi epoch, in grad norm **trước clip tách theo khối** (`stn / backbone / fusion / transformer / head`), loss, và track_id của batch → biết ngay khối nào gây nổ.
- **Tách hai loại batch bị bỏ**: cột `skipped_nonfinite` (tràn số fp16) tách khỏi `skipped_batches` (vượt ngưỡng) — hai nguyên nhân này cần cách sửa khác nhau.
- **Cảnh báo trùng tên experiment** khi khởi động.

Cột trong `{tên}_metrics.csv`: `epoch, train_loss, train_loss_min, train_loss_max, train_loss_std, train_loss_median, grad_norm, grad_norm_max, skipped_batches, skipped_nonfinite, val_loss, val_acc, best_acc, lr, train_time_s, val_time_s, epoch_time_s, elapsed_s, samples_per_s, gpu_mem_peak_mb, timestamp`
