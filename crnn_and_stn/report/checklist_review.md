# Checklist công việc theo review

> 📘 **Bản tường trình quy trình đầy đủ** (3 lần chạy, lệnh + lý do từng cờ, công thức
> toán, đi từ trên xuống đúng thứ tự review): [tong_hop_3_lan_chay.md](tong_hop_3_lan_chay.md).
> File hiện tại là bản **theo dõi tiến độ**.
>
> Bám đúng cấu trúc review: 3 nhóm giải pháp + 4 bước.
> Lệnh chạy: [training_runs/run_gpu.md §0](training_runs/run_gpu.md).
>
> 🗂️ **Sắp xếp thư mục (2026-08-04)** — `report/` và `results/` **chỉ còn** baseline
> CRNN+STN và J1/S1/S4; mọi thứ khác chuyển sang `backup/`:
>
> | Nội dung | Ở đâu |
> |---|---|
> | Multi-seed J1/S1/S4 (9 run) | `results/multi-seed/<cấu hình>/` |
> | Tài liệu baseline + J1/S1/S4 | `report/` |
> | Dữ liệu 1-seed S1–S4 | `backup/mf_sr_ocr/<cấu hình>/` |
> | Artefact các run 1-seed cũ | `backup/crnn_resblock_*` · `backup/DCNv2/` |
> | Tài liệu các ablation 1-seed | `backup/report/` |
>
> Toàn bộ link nội bộ đã cập nhật và verify: **202/202 trong `report/`** và
> **25/25 trong `backup/report/`** đều trỏ đúng file có thật.
> Cập nhật: 2026-08-04 — ✅ **BƯỚC 1 HOÀN THÀNH**: đủ 3 model J1/S1/S4 có
> Mean ± Std trên 3 seed deterministic. **Không còn run GPU nào** trong phạm vi.
> 🎯 **ĐÃ CHỐT: S1 là phương pháp đề xuất chính của paper** — xem mục ngay dưới.

## 🎯 Chốt phương pháp chính — **S1 (Joint MF-SR-OCR)**

| Model | SR | `T` | GFLOPs | **Mean ± Std** | Std | Vai trò trong paper |
|---|---|---:|---:|---:|---:|---|
| **S1** — Joint MF-SR-OCR | ×2 MFSR+DCN | 32 | 109.08 | **79.95% ± 0.15** | **0.15** 🥇 | 🎯 **phương pháp đề xuất** |
| J1 — GroupNorm, **không SR** | ❌ | 16 | **26.14** | 80.45% ± 0.45 | 0.45 | ablation _"bỏ hẳn nhánh SR"_ |
| S4 — SR ×1 | ×1 MFSR+DCN | 32 | chưa đo | 79.48% ± 0.44 | 0.44 | ablation _"bỏ phóng to ảnh"_ |

**Vì sao chọn S1 làm cấu hình chính** (dù J1 có điểm trung bình nhỉnh hơn):

1. **Ổn định nhất qua seed** — std `0.15`, nhỏ hơn **3×** so với J1 (0.45) và S4 (0.44).
   Đây là cấu hình duy nhất mà cả 3 seed đều rơi trong khoảng 0.3 điểm.
2. **Bất biến với `cudnn.benchmark`** — cùng seed 42, chạy `benchmark=True` và
   `deterministic=True` cho **đúng cùng một con số** (797/999, lệch **0 track**), trong
   khi S4 lệch tới **16 track**. Con số của S1 là con số **tái lập được nhất** của cả
   project tính đến nay.
3. **J1 không hơn S1 một cách có ý nghĩa** — chênh +0.50 nằm **trong** biên nhiễu →
   về mặt thống kê là **hoà**. Không có cơ sở để nói J1 "tốt hơn", nên việc chọn giữa
   hai cấu hình hoà nhau được quyết định bằng **độ ổn định** và tính hoàn chỉnh của
   kiến trúc đề xuất.
4. S1 là kiến trúc hoàn chỉnh (multi-frame SR + DCN) mà toàn bộ câu chuyện của bài
   được xây quanh nó; J1/S4 đóng đúng vai **ablation** của chính S1.

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| J1 vs S1 | +0.50 | ±0.28 | ⚠️ trong nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

### ⚠️ Điều BẮT BUỘC phải công bố kèm theo (không được lược bỏ)

Chọn S1 làm phương pháp chính **không** làm mất đi hai kết quả âm tính dưới đây.
Reviewer sẽ tự tính ra từ bảng Mean ± Std, nên phải chủ động nêu:

- **Nhánh SR chưa chứng minh được đóng góp đo được.** J1 (bỏ hẳn SR/DCN/MFSR, dùng
  **chung toàn bộ cụm cờ nền** với S1) hoà điểm với S1 trong khi **rẻ hơn 3.76×
  compute**. Kết luận trung thực là _"SR không cho thấy lợi ích trên tập val này"_,
  **không** phải _"SR có ích"_ — cũng **không** phải _"bỏ SR thì tốt hơn"_ (chênh
  nằm trong nhiễu, không kết luận được chiều nào).
- **Phần cải thiện thật đến từ cụm cờ nền**, không phải SR: STN pool `(4,8)` +
  `--lr-domain-match` + constrained decode + EMA. J1-mới hơn J1-**lịch sử** **+3.57
  điểm** (76.88% → 80.45%) với **cùng** kiến trúc không SR.
- **Giả thuyết `T=32` bị bác bỏ** — J1 chạy `T=16` mà vẫn ngang S1 và hơn S4 (đều `T=32`).

> ⚠️ **J1 multi-seed dùng cờ nền của S1/S4** (STN pool `(4,8)`, domain-match,
> constrained, EMA — chỉ bỏ SR/DCN), **không** phải cờ lịch sử `(1,1)`. Đây là
> ablation 1-cụm-biến sạch so với S1 nên tốt hơn về khoa học, nhưng **không đặt
> chung cột với 76.88%**. Chi tiết:
> [j1_groupnorm_nosr.md](baseline1_crnn_stn/j1_groupnorm_nosr.md).
>
> 📄 Phân tích đầy đủ 9 run: [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).

## Tình trạng nhanh

| Mục review                       | Trạng thái                                                          |
| -------------------------------- | ---------------------------------------------------------------------- |
| Nhóm 1 — công thức loss          | 🟡 **KHÔNG cần train lại.** Công thức sửa đã soạn xong ([paper/loss_formula_corrected.md](paper/loss_formula_corrected.md)), còn dán vào bản thảo |
| Nhóm 2 — quy trình deterministic | ✅ **XONG, đã verify trên 9/9 log**                                   |
| Nhóm 3 — chống overfitting       | ⏭️ **ĐÃ CHỐT KHÔNG ÁP DỤNG** — code sẵn (opt-in, mặc định tắt), đưa vào Future work |
| **Bước 1** — multi-seed          | ✅ **XONG** — J1 ✅ · S1 ✅ · S4 ✅ (3 seed mỗi model)                  |
| **Bước 2** — CER/NED/PSNR/SSIM   | ✅ **XONG** (2 chỗ cố ý lệch review, có lý do)                        |
| **Bước 3** — hình định tính      | 🟡 script xong; hình cũ sinh từ checkpoint 1-seed — **còn chọn/sinh lại hình cho Figure 4** |
| Bước 4 — PARSeq/SVTR             | ❌ chưa bắt đầu, để cuối cùng                                         |

**Không còn việc cần GPU** trong phạm vi đã chốt.

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
| **J1 lịch sử — kiến trúc thật từ `state_dict`** | đọc `.pth` | ✅ 29,313,452 params, **không SR/DCN**, STN fc in=64 → pool **`(1,1)`** — khác J1 multi-seed |

---

## 📦 Phạm vi đã chốt — chỉ 3 model có error bar

Review yêu cầu multi-seed toàn bộ ablation từng chạy (>140h GPU nếu làm hết).
Phạm vi cuối cùng chốt ở **3 model: J1 + S1 + S4** — đủ trả lời 2 câu hỏi cốt lõi:

| So sánh | Câu hỏi trả lời | Kết quả |
|---|---|---|
| **S1 vs J1** | _Pipeline SR đề xuất có hơn backbone không SR không?_ | ✅ **không** — hoà (+0.50, trong nhiễu), J1 rẻ hơn 3.76× → ghi Limitations |
| **S1 vs S4** | _Có cần phóng to ảnh không, hay chỉ cần `T=32`?_ | ✅ **không cần** — hoà (+0.47, trong nhiễu) |

Các ablation 1-seed khác (trọng số λ_SR, perceptual loss, single-frame SR, AdamW
tuning, các hướng SR thất bại ban đầu) **ra ngoài phạm vi** — dữ liệu và tài liệu
lưu ở `backup/`, dùng cho phụ lục nếu cần, **không** vào bảng chính vì không có
error bar.

⚠️ **Ladder J1→S1 là ablation 1-CỤM-biến**, không phải tách 1 biến: S1 thêm cùng lúc
SR + DCN + MFSR (và `T` nhảy 16→32). Cách đọc đúng: *"gộp cả cụm đó lại không cải
thiện"*, **không** phải *"riêng SR không đóng góp"*.

### Hệ quả với công thức trong paper

⚠️ Số hạng `λ_Perceptual · L_VGG` phải ghi rõ là **ablation 1 seed, chưa xác nhận** —
cả 3 cấu hình có error bar đều chạy `α = 0` (đã verify banner 8/8 log).

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

- [x] Xác nhận quy đổi `λ_Perceptual = λ_SR × α` (ablation từng bật: 0.1 × 0.1 = 0.01)
- [x] **Chốt phạm vi multi-seed = J1 + S1 + S4** → số hạng `λ_Perceptual` trong công
      thức phải ghi rõ là **ablation 1-seed, chưa xác nhận**, không phải một phần của
      con số headline. ⚠️ Cả 3 cấu hình có error bar đều chạy `perceptual = 0.0` —
      đã verify trong banner 8/8 log
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

> 📌 **KHÔNG cần train lại model.** Cả 4 lỗi là **sai lệch giữa mô tả trong paper và
> code đã chạy** — code vẫn đúng, mọi số liệu J1/S1/S4 giữ nguyên hiệu lực. **0 GPU.**
> Đã chốt ở P0.4: giữ L1, không đổi sang SmoothL1 (đổi code sẽ khiến toàn bộ
> S1–S4 + multi-seed phải chạy lại 39h).
>
> ✅ **Bản công thức đã sửa, sẵn để dán vào paper** (LaTeX + Unicode + 3 câu chú thích
> bắt buộc): [paper/loss_formula_corrected.md](paper/loss_formula_corrected.md).
> Repo **không chứa bản thảo paper** nên file đó là bản nguồn để copy sang Word/Overleaf.

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

## Nhóm 3 — Chống overfitting — ⏭️ **ĐÃ CHỐT KHÔNG ÁP DỤNG (2026-08-04)**

> ### ⏭️ Quyết định: **hoãn Nhóm 3, đưa vào Future work**
>
> **Lý do**: Nhóm 3 **không nằm trong 4 bước bắt buộc** của review. Áp dụng nó là
> **thí nghiệm mới** tốn ~34h GPU và **bắt buộc multi-seed lại cả 3 model** để bảng
> ablation nhất quán (không thể so S1-có-regularization với J1-không-regularization).
> Trong khi đó **toàn bộ việc bắt buộc còn lại đều 0 GPU**.
>
> **Trạng thái code**: 2 thay đổi từng cài thử (`--pre-rnn-dropout`, `--aug-level
> strong`) **đã được REVERT** (2026-08-04) theo yêu cầu — repo về đúng trạng thái lúc
> chạy 9 run multi-seed. ✅ Verify sau revert: params S1 vẫn **29,577,214**,
> `aug full` vẫn **10 phép**, `WD/patience/dropout` = **1e-4 / 18 / 0.25**.
> Muốn làm lại thì cài lại theo mô tả trong phần chi tiết bên dưới.
>
> **Phải ghi vào paper**: bằng chứng overfit 9/9 run (bảng dưới) là **quan sát có giá
> trị**, nêu ở Limitations/Future work kèm câu *"các biện pháp chống overfitting
> (weight decay 1e-3, Dropout 0.3, patience 12, MotionBlur/GaussNoise) đã được cài
> đặt nhưng chưa khảo sát trong phạm vi bài này."*
>
> 📌 Nếu sau này đổi ý: kế hoạch 3 giai đoạn có cổng quyết định (GĐ1 chỉ 6.4h để loại
> sớm, chỉ đi tiếp nếu có tín hiệu) — xem cuối mục này.

<details>
<summary>Chi tiết kỹ thuật (giữ lại cho tương lai)</summary>

> 🚨 **Trả lời "có cần train lại model không": CÓ — bắt buộc.**
> Khác hẳn Nhóm 1. Dropout / weight decay / augmentation / patience đều là **tham số
> lúc HUẤN LUYỆN**; đổi chúng **không tác động gì** lên 9 checkpoint đã có. Muốn có số
> mới thì phải train lại, và muốn đưa vào paper thì phải **multi-seed lại từ đầu**
> (≈39h GPU cho 3 model) — không trộn vào Bước 1.
>
> ✅ **Nhưng phần CODE thì đã chuẩn bị xong (0 GPU)** — 2 chỗ còn thiếu đã được cài,
> theo nguyên tắc **opt-in, mặc định giữ nguyên hành vi cũ** để 9 run J1/S1/S4 vẫn
> tái lập được bit-for-bit. Chỉ cần thêm cờ vào lệnh là chạy được ngay.

### ✅ Đã có sẵn — KHÔNG cần code thêm, chỉ cần thêm cờ khi chạy

- [x] Cờ `--weight-decay` và `--wd-skip-bias-norm` (param-grouping)
- [x] Cờ `--patience` — ⚠️ **checklist cũ ghi sai** là phải sửa `configs/config.py:122`;
      thực tế `train.py:71` đã có cờ CLI, không cần đụng vào code
- [x] Cờ `--rnn-dropout` — đổi 0.25 → 0.3 (review yêu cầu) bằng CLI, không cần code
- [x] **`Dropout` trước FC layer — ĐÃ CÓ SẴN** (`crnn.py:132-136`:
      `LayerNorm → Dropout(0.25) → Linear`) và giữa 2 lớp BiLSTM
      (`crnn.py:130`: `nn.LSTM(..., dropout=0.25)`)

### ⏮️ Đã cài thử rồi REVERT (2026-08-04) — cách làm lại nếu đổi ý

- [ ] **`Dropout` TRƯỚC BiLSTM** — thêm tham số `pre_rnn_dropout` vào
      `MultiFrameCRNN.__init__` (`crnn.py`), một `nn.Dropout` áp lên `seq_input` ngay
      trước `self.rnn`, thêm `PRE_RNN_DROPOUT` vào config + cờ `--pre-rnn-dropout`.
      📌 Đặt mặc định **0.0** thì checkpoint cũ vẫn `load_state_dict(strict=True)` được
      (Dropout không có tham số nên `state_dict` không đổi) và params giữ đúng
      **29,577,214**
- [ ] **`MotionBlur` + `GaussNoise` vào train transform** — thêm mức
      **`--aug-level strong`** (= `full` + 2 phép này, `p=0.3` mỗi phép) qua tham số
      `noise_aug` trong `transforms.py::get_train_transforms` + nhánh chọn ở
      `dataset.py`.
      ⚠️ 2 phép này *có sẵn* trong `transforms.py:77,80,104,107` nhưng chỉ thuộc
      **degradation pipeline của ảnh synthetic**, không nằm trong augment train.
      `RandomBrightnessContrast` (dòng 23) và `ShiftScaleRotate`≈`Affine` (dòng 15)
      thì đã có sẵn trong train transform

### ⬜ Còn lại — đều CẦN GPU

- [ ] ⏭️ **HOÃN** — chạy thử 1 cấu hình với bộ cờ Nhóm 3 đầy đủ (lệnh ở dưới)
- [ ] ⏭️ **HOÃN** — nếu có cải thiện → **multi-seed lại** 3 seed mới đưa vào paper

#### Kế hoạch 3 giai đoạn (chỉ dùng khi đổi ý)

| GĐ | Chạy gì | Chi phí | Cổng quyết định |
|---|---|---:|---|
| **1** | S1 + Nhóm 3, **seed 42** | **~6.4h** | So với S1 seed 42 = **797/999**. Tụt > 13 track → **dừng hẳn**. Ngang/hơn → GĐ2 |
| **2** | S1 + Nhóm 3, seed 100 + 2026 | ~13h | So Mean±Std với **79.95% ± 0.15**. Cần hơn **~0.6 điểm** mới là thật |
| **3** | J1 + S4 + Nhóm 3, 3 seed | ~15h | Chỉ chạy nếu GĐ2 thắng — để bảng ablation nhất quán |

> ⚠️ **Cổng GĐ1 chỉ bắt được thảm hoạ, không xác nhận được cải thiện** — 1 seed vẫn
> nằm trong biên nhiễu ±13 track. Dùng để *loại sớm*, không để kết luận.
>
> ⚠️ **Rủi ro cho câu chuyện paper**: J1 overfit **nặng nhất** (val loss chạm đáy ep
> 11–16, train loss xuống 0.0074 — sớm và sâu hơn S1/S4), nên có khả năng **hưởng lợi
> từ regularization nhiều hơn S1**. Nếu vậy khoảng cách J1 > S1 **nới rộng ra**, làm
> lựa chọn S1 khó bảo vệ hơn hiện tại.

```bash
# S1 + toàn bộ Nhóm 3. Bật MỘT LẦN cả cụm, không bật lẻ từng cái —
# bật lẻ thì mỗi lần chạy lại tốn 6.4h mà vẫn không tách được biến.
python train.py --preset stable --experiment-name s1_reg_seed42 --seed 42 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --pre-rnn-dropout 0.3 --rnn-dropout 0.3 \
  --weight-decay 1e-3 --wd-skip-bias-norm \
  --patience 12 --aug-level strong \
  --no-cudnn-benchmark --num-workers 8 \
  2>&1 | tee results/log_s1_reg_seed42.txt
```

> ⚠️ Bật `--wd-skip-bias-norm` **cùng lúc** với `--weight-decay 1e-3`, không bật riêng —
> phạt weight decay lên bias/norm là phạt sai chỗ.
>
> ⚠️ `--patience 12` sẽ khiến run dừng sớm hơn: S1 đỉnh ở epoch 24–32, cộng patience 12
> → dừng khoảng epoch 36–44 (thay vì 42–50). Tiết kiệm ~1.5h/seed.
>
> 📌 **Bằng chứng overfit nay là 9/9 run, không còn là quan sát 1-seed.** Pattern
> lặp lại y hệt ở cả 9:
>
> | | val_loss chạm đáy | val_acc đạt đỉnh | val_loss cuối | train_loss cuối |
> |---|---:|---:|---:|---:|
> | J1 seed 42 | ep 16 (0.1836) | ep 22 | 0.2665 | **0.0160** |
> | J1 seed 100 | ep 12 (0.1820) | ep 17 | 0.2516 | **0.0209** |
> | J1 seed 2026 | ep 11 (0.1819) | ep 35 | 0.3044 | **0.0074** |
> | S1 seed 42 | ep 20 (0.1875) | ep 28 | 0.2849 | 0.0344 |
> | S1 seed 100 | ep 16 (0.1852) | ep 32 | 0.2701 | 0.0312 |
> | S1 seed 2026 | ep 19 (0.1888) | ep 24 | 0.2556 | 0.0411 |
> | S4 seed 42 | ep 15 (0.2065) | ep 41 | 0.3415 | 0.0273 |
> | S4 seed 100 | ep 22 (0.1924) | ep 23 | 0.2710 | 0.0408 |
> | S4 seed 2026 | ep 19 (0.1889) | ep 40 | 0.2985 | 0.0283 |
>
> Val loss chạm đáy rất sớm (**ep 11–22**) rồi **tăng 38–67%** tới lúc dừng, trong
> khi train loss tụt về ~0.01–0.04 (model thuộc lòng tập train). Val acc vẫn nhích lên
> tới tận ep 17–41 dù val loss đã tăng.
>
> 🆕 **J1 overfit sớm nhất và sâu nhất** (đáy ep 11–16, train loss xuống 0.0074) — hợp
> lý vì J1 không có nhánh SR đóng vai **regularizer đa nhiệm**. Đây là lập luận đáng
> nêu khi bảo vệ việc **giữ nhánh SR** trong S1: SR không cải thiện exact match, nhưng
> **có** ghìm được mức overfit.

</details>

---

## 🔴 Bước 1 — Multi-Seed Runs (đường găng)

Phạm vi: **J1 + S1 + S4** — xem [lý do](#-phạm-vi-đã-chốt--chỉ-3-model-có-error-bar).

- [x] `mkdir -p results` trên server (thiếu thì `tee` fail, mất log)
- [x] **S4 × 3 seed** ✅ **XONG** → `79.48% ± 0.44` (42: 78.98 · 100: 79.78 · 2026: 79.68)
- [x] **S1 × 3 seed** ✅ **XONG** → `79.95% ± 0.15` (42: 79.78 · 100: 80.08 · 2026: 79.98)
- [x] `aggregate_seeds.py` so S1 ↔ S4 → **không khác biệt có ý nghĩa** (+0.47 ± 0.27)
- [x] Báo cáo multi-seed: [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md)
- [x] **J1 × 3 seed** ✅ **XONG** → `80.45% ± 0.45` (42: 79.98 · 100: 80.88 · 2026: 80.48)
      ⚠️ Chạy bằng **cờ nền S1/S4** (STN pool `(4,8)` + domain-match + constrained +
      EMA, chỉ bỏ SR/DCN), **không** phải cờ lịch sử `(1,1)` — nên là ablation
      1-cụm-biến sạch so với S1, và số **không** so được với 76.88% lịch sử
- [x] `aggregate_seeds.py` → bảng Mean ± Std đủ **3** model ✅
- [x] 🆕 `aggregate_seeds.py --output-csv` — xuất bảng ra file CSV ✅
- [x] Cập nhật `model_comparison_summary.md` + `multi_seed_results.md` + `j1_groupnorm_nosr.md` ✅

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
1. ~~Mọi số 1-seed nhiều khả năng cũng bị thổi phồng~~ →
   ⚠️ **ĐÃ BỊ BÁC BỎ với S1**: seed 42 của S1 cho **đúng cùng con số** ở cả 2 chế
   độ cudnn (797/999). Hiệu ứng `benchmark=True` **phụ thuộc cấu hình**, không phải
   quy luật chung — nên không khái quát hoá được theo chiều nào
2. ~~Chưa so được S4 với S1~~ → ✅ **ĐÃ SO ĐƯỢC**: S1 = 79.95% ± 0.15 vs
   S4 = 79.48% ± 0.44 → không khác biệt có ý nghĩa. J1 = 80.45% ± 0.45 (cao nhất)
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
| **Bảng tổng hợp dạng CSV** | ✅ | `aggregate_seeds.py --output-csv` (thêm 2026-08-04) |

- [x] 🆕 **Đã thêm `--output-csv` + `--append` cho `tools/aggregate_seeds.py`** — xuất
      bảng Mean ± Std thành file **13 cột** (`label, n_seeds, accs, mean, std, min, max,
      ci95, ci_low, ci_high, delta_vs_baseline, delta_stderr, verdict`), đúng chữ
      "ghi nhận vào CSV log" của review. `--append` để gom cả 3 model vào 1 file.
      Cột `verdict` = `significant` / `within_noise` / `single_seed`.
      ✅ Verify với số thật: S1 `79.9466 ± 0.1529`, J1 `80.4471 ± 0.4514`,
      S4 `79.4795 ± 0.4363`, delta `-0.5005 ± 0.2751` → `within_noise` — khớp chính xác
      bảng trong [multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).
      ⚠️ `results/` bị gitignore → muốn commit thì ghi thẳng vào `report/`:
      `--output-csv report/baseline1_crnn_stn/multi_seed_summary.csv`

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
      `backup/mf_sr_ocr/<cấu hình>/sr_quality_*.csv`
- [x] Tương quan PSNR ↔ đọc đúng ở mức từng track (n=999)
- [x] Báo cáo: [buoc2_metrics.md](buoc2_metrics.md)

**Kết quả mạnh nhất để đưa vào paper — PSNR _nghịch_ với khả năng đọc:**

| Cấu hình | PSNR track đọc **đúng** | PSNR track đọc **sai** | r(PSNR, đúng) |
| -------- | ----------------------: | ---------------------: | ------------: |
| **S1**   |                  16.288 |                 18.240 |        −0.362 |
| **S4**   |                  17.080 |                 19.371 |        −0.400 |
| **J1**   |         không có nhánh SR |     không có nhánh SR |             — |

Track đọc **sai** lại có PSNR **cao hơn ~2 dB**, nhất quán ở cả 2 cấu hình có SR
(mỗi cấu hình n=999). Cùng chiều với mức kiến trúc: **J1 bỏ hẳn việc tái tạo ảnh lại
đọc tốt nhất** (80.45%).

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
- [x] Chạy full 10 track trên checkpoint 1-seed → `figure4_qualitative_grid.png`
      + 10 ảnh track riêng cho mỗi cấu hình

> 📌 **Đính chính**: từng ghi sai "J1 không có hình và không thể có, phải đổi sang grid
> 3 cột". Đọc lại code `tools/visualize_paper_figures.py::frame_column` mới phát hiện
> tool đã tự chặn `frames is None` và vẽ placeholder `"(khong co SR)"` ở cột `I_SR` thay
> vì crash — **J1 chạy được y nguyên với `--checkpoint` của nó**, không cần sửa code.
> Cái thật sự chặn J1 là PSNR/SSIM (`tools/eval_sr_quality.py`, `raise SystemExit` ở
> dòng 131) — mục đó ở Bước 2 vẫn đúng như đã ghi.
>
> ⏱️ **Thứ tự thời gian — hình chạy TRƯỚC multi-seed, không phải sau** (mtime thật):
>
> ```
> 2026-08-01 17:27   mf_sr_ocr/*/*.pth            ← checkpoint 1-seed (nguồn của hình)
> 2026-08-02 00:39   mf_sr_ocr/*/paper_figures/   ← 4 bộ hình sinh ở đây
> 2026-08-02 23:57   multi-seed/{s1,s4}/*.pth     ← multi-seed S1+S4 về sau
> 2026-08-04 00:34   multi-seed/j1/*.pth          ← multi-seed J1 cuối cùng
> ```
- [ ] 🟠 **Chọn hình cuối cho Figure 4** — việc duy nhất còn lại của Bước 3, không
      tốn GPU, làm được ngay. ✅ **Đã chốt cấu hình: S1** → có cột `I_SR` thật (J1
      chạy được nhưng cột đó chỉ là placeholder, chỉ hữu ích cho phụ lục)

⚠️ **Lưu ý mới sau multi-seed**: hình hiện có sinh từ checkpoint **1-seed** của
S1–S4 (`backup/mf_sr_ocr/`), trong khi con số trong paper nay là multi-seed. Với
S4 thì checkpoint 1-seed đạt 805/999 còn mức thật là ~794 — **hình và số sẽ đến từ
2 model khác nhau**. Hai cách xử lý:

| Cách | Việc phải làm | Đánh giá |
|---|---|---|
| **A. Sinh lại hình từ checkpoint seed 42** của `results/multi-seed/` | chạy lại `visualize_paper_figures.py`, ~20 phút CPU/cấu hình, 0 GPU | ✅ **khuyến nghị** — hình và số cùng 1 model, tránh reviewer hỏi |
| B. Giữ hình cũ | ghi rõ caption "hình minh hoạ từ 1 run, số trong Bảng X là Mean ± Std của 3 seed" | chấp nhận được nhưng phải ghi chú |

📌 Với S1 thì mức chênh 1-seed ↔ multi-seed là **nhỏ nhất trong 3 model** (seed 42
cho đúng 797/999 ở cả hai chế độ cudnn), nên cách B ít rủi ro hơn hẳn so với S4 —
nhưng cách A vẫn sạch hơn nếu còn thời gian.

## 🟠 Bước 4 — PARSeq / SVTR — **LÀM RIÊNG, CUỐI CÙNG**

- [ ] Dựng repo + môi trường riêng (không đụng `crnn_and_stn`)
- [ ] Chốt cách so công bằng: PARSeq là single-image, ta là multi-frame → đề xuất cho
      PARSeq ăn **frame giữa**, ghi rõ là baseline single-frame
- [ ] Xuất danh sách 999 track val + nhãn để chấm trên **đúng cùng một tập**
- [ ] Chạy + đưa vào bảng so sánh

> Cắt được nếu thiếu thời gian — chỉ cần ghi vào phần Limitations.

---

## Sau Bước 1 — chạy test

- [x] Chốt cấu hình cuối theo Mean ± Std, **trước khi** nhìn bất kỳ số test nào →
      ✅ **S1** (2026-08-04), chọn vì std nhỏ nhất (0.15) và bất biến với
      `cudnn.benchmark`. Quyết định này được chốt khi **chưa** có bất kỳ số test nào
- [ ] Inference test public — ⚠️ `--submission-mode` **train lại từ đầu và tắt early
      stopping**, không phải chỉ inference; lệnh sẵn dùng (`--epochs 28`, trung bình
      best epoch 3 seed S1) + phân tích 2 phương án A/B ở
      [paper/paper_revision_plan.md §3](paper/paper_revision_plan.md)
- [ ] Test blind cuối cùng

---

## Tình trạng

| Hạng mục                | Xong        | Còn lại                                   | GPU cần |
| ----------------------- | ----------- | ----------------------------------------- | ------: |
| Nhóm 1 — công thức      | 5/8         | dán công thức đã soạn vào bản thảo        |       0 |
| Nhóm 2 — deterministic  | **4/4** ✅  | —                                         |       0 |
| Nhóm 3 — chống overfit  | ⏭️ ngoài phạm vi | code sẵn; **đã chốt không áp dụng** → Future work |       0 |
| **Bước 1 — multi-seed** | **9/9** ✅  | —                                         |       0 |
| **Bước 2 — metrics**    | **8/8** ✅  | (tuỳ chọn: `--output-csv`)                |       0 |
| Bước 3 — hình           | 3/4         | chọn hình cuối (+ cân nhắc sinh lại)      |       0 |
| Bước 4 — SOTA           | 0/4         | cuối cùng, cắt được                       |   riêng |

**Tổng kết**: 3/7 hạng mục xong hẳn (Nhóm 2, **Bước 1**, Bước 2).
**Không còn việc nào cần GPU** trong phạm vi đã chốt.

### Việc kế tiếp theo thứ tự

**Không còn việc cần GPU** trong phạm vi đã chốt. Các việc còn lại đều **0 GPU**:

1. ✍️ **Sửa 4 lỗi công thức loss trong paper** (Nhóm 1) — chi tiết 4 lỗi + công
   thức đúng nằm ở [mục Nhóm 1](#nhóm-1--hoàn-thiện-công-thức-loss-việc-phía-paper-0-gpu)
   của chính file này.
2. ✍️ **Đổi cách trình bày con số headline sang Mean ± Std** — **giữ S1 làm phương
   pháp đề xuất** (đã chốt), nhưng con số headline phải là **79.95% ± 0.15 trên 3
   seed**, bỏ hẳn lối báo cáo best-of-run 79.78%/80.58%. Bảng chính = S1 (đề xuất)
   + J1, S4 (ablation), cả 3 đều có error bar.
3. 🚨 **Viết lại phần "đóng góp của nhánh SR"** — việc lớn nhất còn lại. Bản thảo
   hiện ngầm coi SR là nguồn cải thiện; số liệu **không ủng hộ điều đó** (J1 bỏ hẳn
   SR vẫn hoà S1, rẻ hơn 3.76×). Không phải đổi phương pháp chính, mà là **đổi lời
   giải thích vì sao nó hoạt động**: phần cải thiện đo được đến từ **cụm cờ nền**
   (STN pool `(4,8)` + domain-match + constrained decode + EMA). Viết theo hướng
   trung thực: _"SR không cho thấy lợi ích đo được trên tập val này"_.
4. ✍️ **Cập nhật Limitations** — đã có nháp trong file trên (§3), nhưng **cần bổ
   sung** 3 mục mới sau kết quả J1: (a) nhánh SR chưa chứng minh được đóng góp, J1
   hoà S1 với 1/3.76 chi phí; (b) chưa tách được từng thành phần trong cụm cờ nền;
   (c) J1 dùng cờ nền S1/S4 nên không so được với J1 lịch sử 76.88%.
5. 🖼️ **Chọn hình cuối cho Figure 4** — dùng **S1** (cấu hình chính) nên giữ được
   grid **4 cột** có cột `I_SR`. Cân nhắc sinh lại từ
   `results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth` để hình khớp số multi-seed.
6. 📏 **Benchmark compute cho J1 và S4** — `tools/benchmark.py` chưa có dòng cho
   `sr-scale=1`; phút/epoch của J1 mới là ước tính (thiếu `log_j1p_seed42.txt`).
   Cần số thật vì con số **"S1 tốn 3.76× compute so với J1 mà không hơn điểm"** là
   đánh đổi trung tâm phải nêu rõ khi bảo vệ lựa chọn S1.
