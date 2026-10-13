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

## Việc chưa làm

- [x] ~~Constrained decoding theo pattern biển số~~ → đo được +0,00–0,05 (mục Phân tích 01/10, §1)
- [x] ~~Bỏ `ChannelShuffle` + augment nhất quán~~ → run #13
- [x] ~~Vòng E1–E4~~ → run #39–#42
- [ ] Xin bản LRLPR-26 chính thức có nhãn test (license, email trường)
- [x] ~~Vòng hires, fusion temporal~~ → run #43, #44: không cải thiện
- [x] ~~3 seed `degcal` + 3 seed `degcal + aux` theo giao thức 999~~ → thay bằng seed của combo3 (cấu hình chốt sau run #51)
- [ ] Chấm mọi checkpoint giao thức 999 trên test 3.000 track khi có nhãn
- [x] ~~Vòng frame-dropout / frame-mask / grayscale / clean-geo~~ → run #45–#48: fmask, fdrop, gray thắng
- [x] ~~Vòng combo~~ → run #49–#51: combo3_999 đạt 79,28 (TB6 78,33) trên giao thức của tác giả
- [ ] Kết quả `restran_combo3_40ep` (nếu đã chạy)
- [ ] Vòng seed cuối: combo3_999 seed 43, 7 + baseline 999 seed 43, 7
- [ ] Ablation `--no-stn` để đối chiếu +2,90 điểm mà slide báo

---

## Các file code đã sửa — nhớ đồng bộ lên server

Sau mỗi lần sửa phải upload lại, nếu không sẽ gặp `train.py: error: unrecognized arguments`:

```bash
scp -P <PORT> train.py root@<IP>:~/restran/
scp -P <PORT> configs/config.py root@<IP>:~/restran/configs/
scp -P <PORT> src/data/dataset.py src/data/transforms.py root@<IP>:~/restran/src/data/
scp -P <PORT> src/training/trainer.py root@<IP>:~/restran/src/training/
scp -P <PORT> src/models/restran.py src/models/components.py src/models/sr.py src/models/edvr.py root@<IP>:~/restran/src/models/
```

Đối chiếu bằng md5 (`md5sum` trên Linux, `md5 -q` trên macOS) — hai bên phải trùng:

| File | md5 (07/10/2026) |
|---|---|
| `train.py` | `a39433f1e699f1b4033e35ece41fddf8` |
| `configs/config.py` | `0dc3e1983695f77d3a648da07310bb5b` |
| `src/data/dataset.py` | `63f833a32723146d4c4471b0bb92a0f0` |
| `src/data/transforms.py` | `7efed8c09cd5e413c3451e4e26d7a4db` |
| `src/training/trainer.py` | `dd57f2820c70f89b57fe406f615e6859` |
| `src/models/restran.py` | `98bf8a84796701da80cd42eb22f68e2a` |
| `src/models/components.py` | `09df74648ed078aec7c0d0503d71a569` |
| `src/models/sr.py` | `53836e20f56b1c423e2da07a3cd890b5` (không đổi) |
| `src/models/edvr.py` | `4cdb1b931be313f331b0d9e29c07c22e` (không đổi) |

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
