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

**Cấu hình tốt nhất hiện tại: run #6 — 77,18%**, checkpoint `results/restran_30ep_lr3e4_best.pth`.

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

## Việc chưa làm

- [ ] Chạy lại run #3 với tên riêng để **khôi phục checkpoint 76,68%** đã mất
- [ ] Thử 60 epoch với `--lr 1.5e-4` xem có vượt 76,68% không
- [ ] Ablation `--no-stn` để đối chiếu +2,90 điểm mà slide báo
- [ ] Bỏ `ChannelShuffle` và augment nhất quán theo track ([restran_baseline.md §7.2](baseline/restran_baseline.md))
- [ ] Constrained decoding theo pattern biển số — cải thiện không cần train lại

---

## Các file code đã sửa — nhớ đồng bộ lên server

Chỉ **5 file** này chứa toàn bộ thay đổi. Sau mỗi lần sửa phải upload lại, nếu không sẽ gặp `train.py: error: unrecognized arguments`.

```bash
scp -P <PORT> train.py configs/config.py root@<IP>:~/restran/
scp -P <PORT> src/training/trainer.py src/data/dataset.py root@<IP>:~/restran/src/training/ # xem lưu ý dưới
```

| File | Chứa gì |
|---|---|
| `train.py` | toàn bộ cờ CLI, sampler, cảnh báo trùng tên experiment |
| `configs/config.py` | các giá trị mặc định mới |
| `src/training/trainer.py` | thống kê CSV, đo thời gian, cầu dao, chẩn đoán spike, accuracy theo layout |
| `src/data/dataset.py` | `layouts`, `layout_weights()`, cảnh báo lệch split |
| `src/models/restran.py` | tham số `pretrained` |

Đối chiếu bằng md5 (`md5sum` trên Linux, `md5 -q` trên macOS) — hai bên phải trùng:

| File | md5 (13/08/2026, sau bước distillation) |
|---|---|
| `train.py` | `c77cf7ae9a30e344350df27b1eb1156c` |
| `configs/config.py` | `2c915d69d168233ec842db47d17e258a` |
| `src/data/dataset.py` | `57d6113d168c70a7f944fc49d83e2c8d` |
| `src/training/trainer.py` | `10925e2009c7ffa28513c69669a39db5` |
| `src/models/restran.py` | `8fb3da4a95f623d2d522c2f858ca132c` |
| `src/models/sr.py` | `d356f1494371bc35aa1aed058ee806ec` **(file mới)** |
| `src/models/edvr.py` | `04b56980918796d2234db976793d7117` **(file mới)** |
| `src/data/transforms.py` | `9abcac89ba5f28fa56ad94f6cec88a30` |

> `src/data/transforms.py` là file thứ 6, mới được thêm vào danh sách từ bước augmentation.

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
