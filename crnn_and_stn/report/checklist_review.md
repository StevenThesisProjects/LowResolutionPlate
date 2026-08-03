# Checklist công việc theo review

> Bám đúng cấu trúc review: 3 nhóm giải pháp + 4 bước.
> Dữ liệu S1–S4: `results/mf_sr_ocr/<cấu hình>/` · Multi-seed:
> `results/multi-seed/<cấu hình>/` · Lệnh chạy:
> [training_runs/run_gpu.md §0](training_runs/run_gpu.md).
> Cập nhật: 2026-08-03 — **phạm vi cuối cùng: 3 model J1 + S1 + S4** (bỏ J2, S2, S3).
> S1 và S4 **đã có multi-seed**; chỉ còn **J1 (~10h)**. Toàn bộ trạng thái dưới đây
> đã được đối chiếu với code thật (`grep`) và banner 6 log multi-seed.

## 🎯 Ba model của paper

| Model | Vai trò | Multi-seed | Kết quả |
|---|---|:---:|---|
| **J1** | Mốc nền: GroupNorm, **không SR**, T=16 | 🔴 ~10h | *chưa chạy* (kỳ vọng ~76.9%) |
| **S1** | **Phương pháp đề xuất** — Joint MF-SR-OCR ×2 | ✅ | **79.95% ± 0.15** |
| **S4** | Biến thể rẻ: SR ×1, T=32 qua backbone | ✅ | **79.48% ± 0.44** |

Mọi cấu hình khác (J2, J3, S2, S3, nhánh AdamW, SR-v1/v2) là **exploratory** — giữ
số 1-seed làm tham khảo/phụ lục, **không** đưa vào bảng chính có error bar.

## Tình trạng nhanh

| Mục review                       | Trạng thái                                                          |
| -------------------------------- | ---------------------------------------------------------------------- |
| Nhóm 1 — công thức loss          | ⚠️ đã tìm ra **4 lỗi**, **chưa sửa vào paper**                        |
| Nhóm 2 — quy trình deterministic | ✅ **XONG, đã verify trên 6/6 log**                                   |
| Nhóm 3 — chống overfitting       | 🟡 **1/4 đã có sẵn trong code**, 3 mục chưa (hoãn tới sau Bước 1)      |
| **Bước 1** — multi-seed          | 🟡 **2/3 xong**: S1 ✅ · S4 ✅ · **J1 🔴 (~10h)**                       |
| **Bước 2** — CER/NED/PSNR/SSIM   | ✅ **XONG** (2 chỗ cố ý lệch review, có lý do)                        |
| **Bước 3** — hình định tính      | 🟡 script + 4 bộ hình xong, **còn chọn hình cuối cho Figure 4**       |
| Bước 4 — PARSeq/SVTR             | ❌ chưa bắt đầu, để cuối cùng                                         |

**Đường găng duy nhất**: **J1 (~10h)** — 1 run GPU cuối cùng.

> ✅ **Số chính thức đã có**: S1 = **79.95% ± 0.15**, S4 = **79.48% ± 0.44** → chênh
> +0.47 điểm, sai số hiệu ±0.27 → **không khác biệt có ý nghĩa thống kê**. S4 rẻ hơn
> 2.30× khi train. Phân tích đầy đủ 6 run:
> [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).

### ✅ Đã verify gì trong đợt review này (2026-08-03)

Không chỉ đọc lại tài liệu — đã kiểm tra trực tiếp:

| Kiểm tra | Cách verify | Kết quả |
|---|---|---|
| 6/6 run đúng chế độ deterministic | `grep "Seed.*cudnn" log_*.txt` | ✅ cả 6 in `cudnn.benchmark: False (deterministic: True)` |
| 6/6 run đúng seed 42/100/2026 | banner log | ✅ khớp tên experiment |
| Kiến trúc S1 vs S4 đúng như thiết kế | banner `Model params` + `STN pool` | ✅ S1: 29,577,214 · `Width /8 → T=32`<br>S4: 29,549,470 · `Width /4 → T=32` |
| Không cấu hình nào lỡ bật perceptual | banner `SR: ... perceptual=0.0` | ✅ cả 6 đều `perceptual=0.0` |
| `nan_batches = 0` mọi epoch | cột `nan_batches` trong 6 CSV | ✅ toàn bộ = 0 |
| CSV đủ 14 cột (có CER/NED/thời gian) | `head -1 history_*.csv` | ✅ đủ 14 cột |
| CER/NED in ra console mỗi epoch | `grep "CER:" log_*.txt` | ✅ có (`CER: 0.2688 → 0.0964 → ...`) |
| Val Acc khớp khi chấm lại từ nhãn gốc | chấm `submission_*.txt` với `plate_text` trong `annotations.json` | ✅ khớp chính xác cả 6 run |
| Code dùng `F.l1_loss` hay Smooth L1 | `src/training/losses.py:84` | ⚠️ **L1 thuần** — review ghi "Smooth L1" là sai |
| Warp áp lên I_SR hay chỉ I_HR | `src/training/trainer.py:263-272` | ⚠️ **chỉ warp I_HR** — review ghi warp cả hai là sai |
| `theta` có detach khi warp không | `trainer.py:270` `theta_sel.detach()` | ⚠️ **có detach** — chi tiết này chưa có trong paper |
| Dropout trước BiLSTM/FC | `src/models/crnn.py:130,134` | 🟡 **trước FC đã có** (0.25), **trước BiLSTM chưa có** |
| MotionBlur/GaussNoise trong train transform | `src/data/transforms.py` | ❌ chỉ có trong degradation ảnh **synthetic**, không có trong train transform |
| `weight_decay` hiện tại | banner 6 log | ❌ vẫn `0.0001` (review yêu cầu `1e-3`) |
| `patience` hiện tại | `configs/config.py:122` | ❌ vẫn `18` (review yêu cầu `12`) |
| **J1/J2 lịch sử — chấm lại từ artefact còn lưu** | `submission_*.txt` vs nhãn gốc | ✅ **768/999 (76.88%)** và **771/999 (77.18%)** — khớp chính xác số đã báo cáo |
| **J1/J2 — kiến trúc thật từ `state_dict`** | đọc `.pth` | ✅ J1: 29,313,452 params, **không SR/DCN**, STN fc in=64 → pool **`(1,1)`** · J2: 29,426,895 |
| **J2 có bật edge loss không** | banner `log_j2.txt` | 🆕 ⚠️ **`edge=0.5` (BẬT)** — thiếu trong mọi tài liệu trước, S1/S4 đều `edge=0.0` |
| **CER của J1/J2** | tính lại từ submission | 🆕 ⚠️ **J2 (0.0645) TỆ hơn J1 (0.0608)** + 18/999 track sai độ dài (J1: 0) |

---

## 💰 Tối ưu chi phí multi-seed — cắt được cái nào?

Review yêu cầu multi-seed toàn bộ ablation từng chạy (J1/J2/S1/S2/S3/S4) —
tốn hơn ~140h GPU nếu làm hết. Không phải cấu hình nào cũng đáng tiền như nhau.

**Dữ liệu 1-seed đã có đủ cho cả 4 cấu hình S1-S4** (không thiếu gì để phân tích):

|     | history | submission | .pth | CER/NED | PSNR/SSIM | hình | log |
| --- | :-----: | :--------: | :--: | :-----: | :-------: | :--: | :-: |
| S1  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ✅  |
| S2  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ❌  |
| S3  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ❌  |
| S4  |   ✅    |     ✅     |  ✅  |   ✅    |    ✅     |  ✅  | ✅  |

### ✅ ĐÃ CHỐT (2026-08-03): multi-seed **J1 + S1 + S4** — bỏ J2, S2, S3

|        | Multi-seed? | Chi phí | Trạng thái / Lý do                                                                                                                                                                                                                  |
| ------ | :---------: | ------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **S1** |     ✅      |   ~22h  | **XONG — 79.95% ± 0.15.** Phương pháp đề xuất, con số headline của paper                                                                                                                                                             |
| **S4** |     ✅      |   ~12h  | **XONG — 79.48% ± 0.44.** Claim _"bằng S1 nhưng rẻ hơn 2.3×"_ nay đã xác nhận                                                                                                                                                        |
| **J1** |     ✅      |   ~10h  | 🔴 **CÒN LẠI.** Mốc nền: GroupNorm, **không SR**, T=16. Cho claim *"S1 vượt cấu hình không SR"* có error bar. Chạy **cờ lịch sử** (`--stn-pool 1,1`) → tái lập được ~76.88%                                                          |
| **J2** |  ⏭️ **bỏ**  |   ~27h  | SR single-frame ×2. Đắt nhất trong nhóm còn lại. Đánh đổi: mất ablation "multi-frame vs single-frame" → xử lý bằng số 1-seed + Limitations (xem dưới)                                                                                 |
| **S2** |  ⏭️ **bỏ**  |    ~33h | Kết quả **âm tính** (794, kém nhất nhóm S). Kết luận "tăng λ_SR không giúp" đã có bằng chứng độc lập mạnh hơn: tương quan PSNR↔đọc-đúng âm (r ≈ −0.35 đến −0.41, **n=999, nhất quán 4 cấu hình**) — không phụ thuộc thứ hạng của S2 |
| **S3** |  ⏭️ **bỏ**  |~36–45h  | Đồng hạng tuyệt đối với S4 (805/999) ở 1 seed — tốn gần gấp đôi S4 chỉ để xác nhận một ablation loss                                                                                                                                 |

#### 💰 Đánh đổi khi bỏ J2 — phải ghi vào Limitations

**Mất**: ablation _"multi-frame SR có hơn single-frame SR không"_ — đáng lưu ý vì
**"Multi-Frame" nằm ngay trong tên phương pháp (MF-SR-OCR)**.

**Cách trả lời khi reviewer hỏi** (0 GPU):

1. Số 1-seed có sẵn: **J2 (single-frame) 77.18% vs S1 (multi-frame) 79.78%** — chênh
   **26 track = gấp đôi biên nhiễu ±13**. Đủ mạnh để nêu, kèm nhãn *"1 seed; J2 lịch
   sử khác S1 ở 7 tham số"*.
2. Ghi thẳng **Limitations**: *"ablation multi-frame vs single-frame chỉ có 1 seed;
   phần có error bar giới hạn ở J1/S1/S4."*

#### ⚠️ Lưu ý: J1 KHÔNG giải được "T confound"

`T = IMG_WIDTH × (SR_SCALE nếu USE_SR) ÷ WIDTH_DOWNSAMPLE`
([`train.py:285`](../train.py)) → J1 có **T=16**, S1/S4 có **T=32**. Đi từ J1 sang S1
đổi **cả `T` lẫn SR cùng lúc**. Cấu hình duy nhất tách được là `--width-downsample 4`
**không** `--use-sr` (T=32, không SR) — ngoài phạm vi 3 model này → ghi Limitations.

#### Bộ 3 model trả lời được gì

| So sánh | Câu hỏi trả lời | Trạng thái |
|---|---|---|
| **S1 vs J1** | _Pipeline SR đề xuất có hơn backbone không SR không?_ — **claim headline** | ⏳ chờ J1 |
| **S4 vs S1** | _Có cần phóng to ảnh không, hay chỉ cần `T=32`?_ | ✅ **đã trả lời**: không cần |
| ~~S1 vs J2~~ | _Multi-frame có hơn single-frame?_ | 1 seed + Limitations |
| ~~S3/S2 vs S1~~ | _Perceptual / λ=0.5 có giúp không?_ | giữ 1 seed, phụ lục |

⚠️ **Ladder J1→S1→S4 là ablation TÍCH LUỸ**, không phải tách 1 biến — J1 khác S1 ở
nhiều tham số cùng lúc (SR, DCN, MFSR, STN pool, domain-match, decode, EMA). Cách đọc
đúng: *"gộp tất cả thay đổi được +X track"*, **không** phải *"SR đóng góp +X track"*.

### Hệ quả với paper

✅ **Công thức có perceptual vẫn giữ nguyên** như review viết:

```
L_Total = L_CTC + λ_SR·L_SR + λ_Perceptual·L_VGG      (λ_Perceptual = 0.01)
```

⚠️ Nhưng **S3 (cấu hình bật perceptual) không còn trong bộ multi-seed** — công thức
trong paper phải ghi rõ số hạng `λ_Perceptual` là **ablation 1-seed, chưa xác nhận**,
không phải một phần của con số headline (S1) hay của bộ so sánh có error bar
(J1/J2/S1/S4).

⚠️ **S2 và S3** phải được ghi rõ **"1 seed, chưa xác nhận"** ở mọi bảng.

⚠️ **J2 (77.18%) giữ nhãn "1 seed, exploratory"** — dùng trong phụ lục/Limitations,
không vào bảng chính.

### Thứ tự chạy

**S1 ✅ → S4 ✅ → J1 🔴 (~10h)** — chỉ còn 1 run. Lệnh đầy đủ:
[training_runs/run_gpu.md §0 B4](training_runs/run_gpu.md).

---

## Nhóm 1 — Hoàn thiện công thức loss (việc phía paper, 0 GPU)

**Công thức review viết:**

```
L_Total = L_CTC + λ_SR·L_SR + λ_Perceptual·L_VGG
L_SR    = (1/N) Σ ‖Warp_θ(I_SR) − Warp_θ(I_HR)‖₁     ← "Smooth L1"
λ_SR = 0.1 · λ_Perceptual = 0.01
```

**Công thức code thật sự chạy** (đã grep xác nhận 2026-08-03):

```
L_Total = L_CTC + λ_SR·(L1 + α·L_VGG)          # perceptual nằm TRONG L_SR
L_SR    = (1/N) Σ ‖I_SR − Warp_sg[θ](I_HR)‖₁   # chỉ warp HR; θ bị detach
```

- [x] Xác nhận `λ_Perceptual = λ_SR × α` → S3 (`--sr-perceptual-weight 0.1`) = 0.01
- [x] **Chốt phạm vi multi-seed = J1+S1+S4 (2026-08-03)** → S3 **không** còn
      trong bộ multi-seed, nên số hạng `λ_Perceptual` trong công thức phải ghi rõ
      là **ablation 1-seed, chưa xác nhận**, không phải một phần của con số
      headline đã có error bar. ⚠️ Cả S1 lẫn S4 (2 cấu hình có error bar) đều chạy
      `perceptual=0.0` — đã verify trong banner 6/6 log
- [ ] **Lỗi 1 — bỏ `Warp_θ` khỏi `I_SR`.** Code chỉ warp HR
      (`trainer.py:263-272`), vì `I_SR` đã nằm trong khung STN đã nắn — warp hai
      lần là sai
- [ ] **Lỗi 2 — đổi "Smooth L1" → "L1"** — code dùng `F.l1_loss` (`losses.py:84`),
      không phải `F.smooth_l1_loss`
- [ ] 🆕 **Lỗi 3 — `θ` bị DETACH khi warp target** (`trainer.py:270`:
      `theta_sel.detach()`). Nghĩa là **gradient của `L_SR` KHÔNG chảy ngược vào
      STN** — nhánh SR chỉ được phép làm ảnh nét hơn, không được kéo STN về phía
      hình học "dễ tái tạo nhất". Công thức trong paper hiện không thể hiện điều
      này; nên viết `Warp_sg[θ](I_HR)` (stop-gradient) và **giải thích 1 câu vì
      sao** — đây là lựa chọn thiết kế có chủ đích, reviewer sẽ hỏi
- [ ] 🆕 **Lỗi 4 — vị trí `λ_Perceptual` trong công thức.** Code nhân `λ_SR` cho
      **cả cụm** `(L1 + α·L_VGG)`, không cộng `λ_Perceptual·L_VGG` song song với
      `λ_SR·L_SR` như review viết. Hai cách viết chỉ **tương đương khi**
      `λ_Perceptual = λ_SR × α`; nếu paper sau này đổi `λ_SR` mà quên đổi
      `λ_Perceptual` thì hai công thức lệch nhau. Ghi rõ quy đổi trong paper

> 📌 Cả 4 lỗi đều là **việc sửa chữ trong paper, 0 GPU** — không lỗi nào bắt buộc
> phải train lại. Đã chốt ở P0.4: giữ L1, không đổi sang SmoothL1 (đổi code sẽ
> khiến toàn bộ S1–S4 + multi-seed phải chạy lại).

## Nhóm 2 — Chuẩn hoá quy trình deterministic ✅ **XONG**

- [x] `seed_everything` với `deterministic=True, benchmark=False` — **đã có sẵn**
      trong `src/utils/common.py:12-35`, bật bằng `--no-cudnn-benchmark`.
      **Đầy đủ hơn review**: code seed thêm `PYTHONHASHSEED` và `cuda.manual_seed`
      (bản review chỉ có `manual_seed_all`)
- [x] `tools/aggregate_seeds.py` (Mean ± Std + CI 95% + kiểm định) — **đã có sẵn**
- [x] Smoke test 1 epoch xác nhận deterministic ăn + CSV đủ 14 cột
- [x] 🆕 **Verify trên run thật, không chỉ smoke test**: 6/6 log multi-seed in đúng
      `cudnn.benchmark: False (deterministic: True)`, 6/6 CSV đủ 14 cột,
      `nan_batches = 0` mọi epoch

## Nhóm 3 — Chống overfitting — **làm SAU multi-seed**

- [x] Cờ `--weight-decay` và `--wd-skip-bias-norm` (param-grouping)
- [x] 🆕 **`Dropout` trước FC layer — ĐÃ CÓ SẴN** (`crnn.py:132-136`:
      `LayerNorm → Dropout(0.25) → Linear`) và giữa 2 lớp BiLSTM
      (`crnn.py:130`: `nn.LSTM(..., dropout=0.25)`). Review yêu cầu 0.3, hiện là
      **0.25** — chênh nhỏ, đổi bằng `RNN_DROPOUT` trong config, không cần code mới
- [ ] `Dropout(0.3)` **trước** BiLSTM (giữa CNN và RNN) — **chưa có**, đây là vị
      trí duy nhất còn thiếu
- [ ] Thêm `MotionBlur`, `GaussNoise` vào **train transform** — **chưa code**.
      ⚠️ Đã verify: 2 phép này *có* trong `transforms.py:77,80,104,107` nhưng chỉ
      thuộc **degradation pipeline của ảnh synthetic**, không nằm trong augment
      train. `RandomBrightnessContrast` (dòng 23) và `ShiftScaleRotate`≈`Affine`
      (dòng 15) thì đã có sẵn trong train transform
- [ ] `patience` 18 → 12 (`configs/config.py:122`, hiện vẫn 18)
- [ ] `weight_decay` 1e-4 → 1e-3 (banner 6/6 log xác nhận vẫn đang chạy 1e-4)
- [ ] Chạy thử + multi-seed riêng nếu đưa vào paper

> ⚠️ Đổi model rồi thì phải multi-seed lại từ đầu → **không trộn vào Bước 1**.
> Bật `--wd-skip-bias-norm` **cùng lúc** với `--weight-decay 1e-3`, không bật riêng.
>
> 📌 **Bằng chứng overfit nay là 6/6 run, không còn là quan sát 1-seed.** Pattern
> lặp lại y hệt ở cả 6:
>
> | | val_loss chạm đáy | val_acc đạt đỉnh | val_loss cuối | train_loss cuối |
> |---|---:|---:|---:|---:|
> | S1 seed 42 | ep 20 (0.1875) | ep 28 | 0.2849 | 0.0344 |
> | S1 seed 100 | ep 16 (0.1852) | ep 32 | 0.2701 | 0.0312 |
> | S1 seed 2026 | ep 19 (0.1888) | ep 24 | 0.2556 | 0.0411 |
> | S4 seed 42 | ep 15 (0.2065) | ep 41 | 0.3415 | 0.0273 |
> | S4 seed 100 | ep 22 (0.1924) | ep 23 | 0.2710 | 0.0408 |
> | S4 seed 2026 | ep 19 (0.1889) | ep 40 | 0.2985 | 0.0283 |
>
> Val loss chạm đáy rất sớm (**ep 15–22**) rồi **tăng 38–65%** tới lúc dừng, trong
> khi train loss tụt về ~0.03 (model thuộc lòng tập train). Val acc vẫn nhích lên
> tới tận ep 23–41 dù val loss đã tăng. → **Nhóm 3 là hướng cải thiện có cơ sở
> nhất còn lại**, nhưng vẫn làm **sau** khi Bước 1 chốt xong con số.

---

## 🔴 Bước 1 — Multi-Seed Runs (đường găng)

Đã chốt phạm vi **J1 + S1 + S4**, bỏ S2, S3 và J2 — xem
[lý do](#-tối-ưu-chi-phí-multi-seed--cắt-được-cái-nào).

- [x] `mkdir -p results` trên server (thiếu thì `tee` fail, mất log)
- [x] **S4 × 3 seed** ✅ **XONG** → `79.48% ± 0.44` (42: 78.98 · 100: 79.78 · 2026: 79.68)
- [x] **S1 × 3 seed** ✅ **XONG** → `79.95% ± 0.15` (42: 79.78 · 100: 80.08 · 2026: 79.98)
- [x] `aggregate_seeds.py` so S1 ↔ S4 → **không khác biệt có ý nghĩa** (+0.47 ± 0.27)
- [x] Báo cáo multi-seed: [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md)
- [ ] **J1 × 3 seed** (~10h) — 🔴 **run GPU cuối cùng** (mốc nền, không SR,
      cờ lịch sử `--stn-pool 1,1`, tái lập được ~76.88%)
- [ ] ~~J2 × 3 seed~~ — **đã bỏ**, tiết kiệm 27h; giữ số 1-seed 77.18% cho phụ lục
- [ ] ~~S2 × 3 seed~~ — **đã quyết định bỏ**, giữ kết quả 1 seed kèm nhãn "chưa xác nhận"
- [ ] ~~S3 × 3 seed~~ — **đã quyết định bỏ**, giữ kết quả 1 seed kèm nhãn "chưa xác nhận"
- [ ] `aggregate_seeds.py` → bảng Mean ± Std đủ **3** model (thêm J1)
- [ ] Cập nhật `model_comparison_summary.md` + kết luận paper (đã cập nhật phần S1/S4)

### ✅ Kết quả S1 multi-seed — con số headline TRỤ VỮNG

| Seed | Track đúng | Val Acc | Best epoch |
|---|---:|---:|---:|
| 42 | 797/999 | 79.78% | 28 |
| 100 | 800/999 | 80.08% | 32 |
| 2026 | 799/999 | 79.98% | 24 |
| **Mean ± Std** | | **79.95% ± 0.15** | |

**Ngược hẳn với S4**: cùng seed 42, đổi từ `benchmark=True` sang `deterministic`
cho **đúng cùng 1 con số** (797/999 cả hai) — 0 track lệch. Std giữa các seed cũng
nhỏ nhất từng đo (0.15, so với 0.44 của S4).

→ **Không được khái quát hoá "1-seed + `benchmark=True` luôn thổi phồng"**. Mức nhạy
cảm phụ thuộc cấu hình cụ thể; S4 lệch 16 track còn S1 lệch 0. Đây là quan sát thực
nghiệm cần nêu nguyên trạng trong paper, không suy diễn nguyên nhân.

**So sánh chính thức S1 ↔ S4**: +0.47 điểm, sai số hiệu ±0.27 → **không khác biệt có
ý nghĩa thống kê**. S4 rẻ hơn **2.30×** khi train (4.09 vs 9.39 phút/epoch). Claim
*"S4 bằng S1 nhưng rẻ hơn"* nay đã được xác nhận đúng cách.
Chi tiết: [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).

### 🚨 Kết quả S4 multi-seed — cảnh báo đã thành sự thật

Con số **80.58% không trụ được**. Multi-seed deterministic cho **79.48% ± 0.44**, thấp
hơn **1.10 điểm**, và 80.58% nằm **cao hơn cả 3 seed** (max 79.78%).

**Bằng chứng trực tiếp** — cùng seed 42, chỉ khác chế độ cudnn:

| | Track đúng | Val Acc |
|---|---:|---:|
| `benchmark=True` (lần chạy gốc) | 805/999 | 80.58% |
| `deterministic=True` (multi-seed) | 789/999 | 78.98% |
| **Chênh** | **16 track** | **1.60 điểm** |

16 track **vượt biên nhiễu ±13** — riêng tính không xác định của cudnn đã đủ tạo chênh
lệch lớn hơn ngưỡng dùng để phán xét "có cải thiện hay không". Mối lo của Reviewer #2
là **có cơ sở thật**.

Chi tiết: [s4_sr_scale1_mf_sr_ocr.md §5d](baseline1_crnn_stn/s4_sr_scale1_mf_sr_ocr.md).

**Hệ quả — đã cập nhật sau khi có S1 multi-seed (2026-08-03):**
1. ~~Số 1-seed của J1/J2/S1/S2/S3 nhiều khả năng cũng bị thổi phồng~~ →
   ⚠️ **ĐÃ BỊ BÁC BỎ với S1**: seed 42 của S1 cho **đúng cùng con số** ở cả 2 chế
   độ cudnn (797/999). Hiệu ứng `benchmark=True` **phụ thuộc cấu hình**, không phải
   quy luật chung. J1/J2/S2/S3 vẫn chưa biết
2. Thế hoà **S3 ↔ S4 ở 805/999 có thể không tồn tại** — vì S3 đã bị loại khỏi phạm
   vi multi-seed, việc này sẽ **không** được xác nhận trực tiếp; ghi rõ trong paper
   là giới hạn đã biết, không phải việc bỏ sót
3. ~~Chưa so được S4 với S1~~ → ✅ **ĐÃ SO ĐƯỢC**: S1 = 79.95% ± 0.15 vs
   S4 = 79.48% ± 0.44 → không khác biệt có ý nghĩa. Còn J1 chưa chạy
4. Paper phải chuyển hẳn sang báo cáo **Mean ± Std**, bỏ lối best-of-run

### 📋 "Xác minh: ghi nhận bảng kết quả 3 seeds vào CSV log" — trạng thái

Yêu cầu cuối của Bước 1 trong review. Đã có gì:

| Cần | Có chưa | Ở đâu |
|---|:---:|---|
| CSV per-epoch từng seed (14 cột, có CER/NED/thời gian) | ✅ | `results/multi-seed/<cfg>/history_*_seed{42,100,2026}.csv` |
| Log stdout đầy đủ từng seed | ✅ | `results/multi-seed/<cfg>/log_*_seed*.txt` |
| Dự đoán từng track để chấm lại | ✅ | `results/multi-seed/<cfg>/submission_*_seed*.txt` |
| Checkpoint từng seed | ✅ | `results/multi-seed/<cfg>/*_best.pth` |
| Bảng tổng hợp Mean ± Std | ✅ (dạng `.md`) | [multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md) |
| **Bảng tổng hợp dạng CSV** | ❌ | `aggregate_seeds.py` chỉ in ra console, chưa có `--output-csv` |

- [ ] 🆕 (tuỳ chọn, ~15 phút) Thêm `--output-csv` cho `tools/aggregate_seeds.py` để
      xuất bảng Mean ± Std thành file, đúng chữ "ghi nhận vào CSV log" của review.
      ⚠️ `results/` bị gitignore → dù có CSV vẫn phải chép số vào `.md` trong
      `report/` mới được commit, nên đây là việc **hình thức**, không chặn gì

## 🔴 Bước 2 — CER, NED, PSNR/SSIM ✅ **XONG**

- [x] CER + NED trong `validate()` (`trainer.py:387-460`) → in console mỗi epoch
      (verify: `grep "CER:" log_*.txt` có kết quả) + 2 cột `val_cer`, `val_ned`
      trong `history_*.csv` (verify: 14/14 cột đủ ở cả 6 CSV multi-seed)
- [x] Số CER/NED cho cả S1–S4 (tính lại từ `submission_*.txt`, không cần train lại)
- [x] 🆕 **CER multi-seed cho S1 và S4** — S1: `0.0541 ± 0.0016` ·
      S4: `0.0543 ± 0.0001`. Phát hiện chỉ multi-seed mới thấy: **S4 ổn định hơn
      16× về CER nhưng kém ổn định hơn 3× về exact match** — 2 đại lượng không đi
      cùng chiều
- [x] `tools/eval_sr_quality.py` — PSNR/SSIM bằng `skimage.metrics`
- [x] PSNR/SSIM cho cả 4 cấu hình (999 track mỗi cấu hình), lưu tại
      `results/mf_sr_ocr/<cấu hình>/sr_quality_*.csv`
- [x] Tương quan PSNR ↔ đọc đúng ở mức từng track (n=999)
- [x] Báo cáo: [buoc2_metrics.md](buoc2_metrics.md)

**Kết quả mạnh nhất để đưa vào paper — PSNR _nghịch_ với khả năng đọc:**

| Cấu hình | PSNR track đọc **đúng** | PSNR track đọc **sai** | r(PSNR, đúng) |
| -------- | ----------------------: | ---------------------: | ------------: |
| S1       |                  16.288 |                 18.240 |        −0.362 |
| S2       |                  18.102 |                 20.154 |        −0.346 |
| S3       |                  16.492 |                 18.894 |        −0.412 |
| S4       |                  17.080 |                 19.371 |        −0.400 |

Track đọc **sai** lại có PSNR **cao hơn ~2 dB**, nhất quán ở cả 4 cấu hình. Cùng chiều
với mức cấu hình: S2 tối ưu PSNR mạnh nhất (+2.29 dB) nhưng OCR kém nhất (794/999).

→ Khi reviewer hỏi _"sao không tối ưu theo PSNR"_, câu trả lời không còn là "PSNR không
phản ánh OCR" mà là **"PSNR nghịch với OCR, có bằng chứng trên 999 track ở 4 cấu hình
độc lập"**.

**3 chỗ lệch so với nguyên văn review** (phải nêu trong paper) — đã verify lại:

1. Không dùng thư viện `editdistance` — `postprocess.py` đã có sẵn `edit_distance`
   (Levenshtein có cache). Verify: `grep -rn editdistance src/ tools/` → **0 kết quả**,
   xác nhận không có dependency ngoài. Kết quả tương đương
2. **Không đo trên Scenario-A**: val không có track Scenario-A nào, và cả 10.000 track
   Scenario-A đều nằm trong tập TRAIN → đo ở đó là đo trên dữ liệu đã học, không hợp lệ
3. PSNR/SSIM là tool hậu kỳ, không nằm trong log validation mỗi epoch (nhét vào training
   loop sẽ phải khởi động lại multi-seed đang chạy). ⚠️ **CER/NED thì KHÔNG lệch** —
   2 chỉ số này có đủ trong console + CSV đúng như review yêu cầu

## 🟠 Bước 3 — Hình định tính

- [x] `tools/visualize_paper_figures.py` — grid 4 cột, 5 đúng + 5 sai
- [x] Verify chạy được (bản 2 track, tự tái lập đúng 805/999)
- [x] Chạy full 10 track cho **cả 4 cấu hình** (S1/S2/S3/S4) → mỗi cấu hình có
      `paper_figures/figure4_qualitative_grid.png` + 10 ảnh track riêng
- [ ] 🟠 **Chọn hình cuối cho Figure 4** — việc duy nhất còn lại của Bước 3, không
      tốn GPU, làm được ngay

⚠️ **Lưu ý mới sau multi-seed**: hình hiện có sinh từ checkpoint **1-seed** của
S1–S4 (`results/mf_sr_ocr/`), trong khi con số trong paper nay là multi-seed. Với
S4 thì checkpoint 1-seed đạt 805/999 còn mức thật là ~794 — **hình và số sẽ đến từ
2 model khác nhau**. Hai cách xử lý:

| Cách | Việc phải làm | Đánh giá |
|---|---|---|
| **A. Sinh lại hình từ checkpoint seed 42** của `results/multi-seed/` | chạy lại `visualize_paper_figures.py`, ~20 phút CPU/cấu hình, 0 GPU | ✅ **khuyến nghị** — hình và số cùng 1 model, tránh reviewer hỏi |
| B. Giữ hình cũ | ghi rõ caption "hình minh hoạ từ 1 run, số trong Bảng X là Mean ± Std của 3 seed" | chấp nhận được nhưng phải ghi chú |

Chạy cả 4 (không chỉ 3 cấu hình vào paper) vì rẻ, và có đủ thì so được **cùng một
track qua các cấu hình** — hữu ích khi muốn minh hoạ vì sao S2 tái tạo ảnh đẹp hơn
nhưng lại đọc sai.

## 🟠 Bước 4 — PARSeq / SVTR — **LÀM RIÊNG, CUỐI CÙNG**

- [ ] Dựng repo + môi trường riêng (không đụng `crnn_and_stn`)
- [ ] Chốt cách so công bằng: PARSeq là single-image, ta là multi-frame → đề xuất cho
      PARSeq ăn **frame giữa**, ghi rõ là baseline single-frame
- [ ] Xuất danh sách 999 track val + nhãn để chấm trên **đúng cùng một tập**
- [ ] Chạy + đưa vào bảng so sánh

> Cắt được nếu thiếu thời gian — chỉ cần ghi vào phần Limitations.

---

## Sau Bước 1 — chạy test

- [ ] Chốt cấu hình cuối theo Mean ± Std, **trước khi** nhìn bất kỳ số test nào
- [ ] Inference test public — ⚠️ `--submission-mode` **train lại từ đầu và tắt early
      stopping**, không phải chỉ inference; xem
      [paper/paper_revision_plan.md §2b](paper/paper_revision_plan.md)
- [ ] Test blind cuối cùng

---

## Tình trạng

| Hạng mục                | Xong        | Còn lại                                   | GPU cần |
| ----------------------- | ----------- | ----------------------------------------- | ------: |
| Nhóm 1 — công thức      | 2/6         | 4 lỗi công thức, việc phía paper          |       0 |
| Nhóm 2 — deterministic  | **4/4** ✅  | —                                         |       0 |
| Nhóm 3 — chống overfit  | 2/7         | sau Bước 1 (Dropout, aug, wd, patience)   |   nhiều |
| **Bước 1 — multi-seed** | 5/9         | **chỉ còn J1**                           | **~10h** |
| **Bước 2 — metrics**    | **8/8** ✅  | (tuỳ chọn: `--output-csv`)                |       0 |
| Bước 3 — hình           | 3/4         | chọn hình cuối (+ cân nhắc sinh lại)      |       0 |
| Bước 4 — SOTA           | 0/4         | cuối cùng, cắt được                       |   riêng |

**Tổng kết**: 2/7 hạng mục xong hẳn (Nhóm 2, Bước 2). Đường găng còn lại chỉ là
**~10h GPU cho J1** — 1 run duy nhất. Mọi việc khác đều **0 GPU**, làm song song được.

### Việc kế tiếp theo thứ tự

**Cần GPU (1 việc, ~10h):**

1. 🔴 **Chạy J1 × 3 seed** (~10h) — mốc nền (GroupNorm, không SR, T=16), cờ lịch sử
   `--stn-pool 1,1`. Kỳ vọng ra lại ~76.88%. Sau run này bảng ablation của paper đủ
   **3 dòng có error bar**. Lệnh: [training_runs/run_gpu.md §0 B4](training_runs/run_gpu.md).

**Làm ngay được, không cần GPU (song song lúc chờ):**

2. ✍️ **Sửa 4 lỗi công thức loss trong paper** (Nhóm 1) — đã xác định chính xác
   chỗ sai, chỉ là việc viết lại. Đây là phần review nhấn mạnh nhất mà hiện **chưa
   động vào chút nào**.
3. ✍️ **Viết mục Limitations** — 3 điều phải ghi rõ:
   (a) ablation **multi-frame vs single-frame chỉ có 1 seed** (J2 77.18% vs S1
   79.78%, chênh 26 track) — không có error bar;
   (b) chưa tách được `T=32` khỏi bản thân module SR (thiếu run `width-downsample 4`
   không `use-sr`); ladder J1→S1 là ablation **tích luỹ**, không tách 1 biến;
   (c) S2/S3 là 1 seed, chưa xác nhận.
4. 🖼️ **Chọn hình cuối cho Figure 4** — cân nhắc sinh lại từ checkpoint
   `results/multi-seed/*/[s1|s4]_seed42_best.pth` để hình khớp với số multi-seed.
5. 📊 Sau khi có J1: `aggregate_seeds.py` → bảng Mean ± Std đủ 3 model,
   cập nhật [model_comparison_summary.md §1c](baseline1_crnn_stn/model_comparison_summary.md#1c-kết-quả-multi-seed--số-chính-thức-cho-paper)
   và [multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).
6. ✅ **Chốt cấu hình cuối** theo Mean ± Std → S1 và S4 hiện **ngang nhau về
   accuracy** (chênh nằm trong nhiễu), nên quyết định dựa trên tiêu chí khác:
   S4 rẻ hơn 2.30× khi train · S1 std nhỏ hơn 3× và không nhạy với chế độ cudnn.
