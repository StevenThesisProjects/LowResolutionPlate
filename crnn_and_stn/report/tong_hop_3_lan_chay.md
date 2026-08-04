# Tổng hợp 3 lần chạy — bám theo đúng thứ tự review

> **Mục đích**: gom toàn bộ những gì đã làm theo review vào **một** tài liệu, đi
> **từ trên xuống đúng thứ tự review viết**: 3 nhóm giải pháp → Bước 1 → Bước 2 →
> Bước 3 → Bước 4. Mỗi phần ghi rõ **lệnh đã chạy**, **dùng gì**, **vì sao dùng**,
> và **checklist việc đã làm**.
>
> Tài liệu này **không thay thế** [checklist_review.md](checklist_review.md) (bản
> theo dõi tiến độ) hay [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md)
> (bản phân tích số liệu) — nó là bản **tường trình quy trình** để đưa vào báo cáo.

## Quy ước tên gọi trong tài liệu này

| Tên trong tài liệu | Tên kỹ thuật | `--experiment-name` | Thư mục dữ liệu |
|---|---|---|---|
| **Lần chạy 1** | **J1** — ResBlock+GroupNorm, **không SR** | `j1p_seed{42,100,2026}` | `results/multi-seed/crnn_resblock_groupnorm_nosr_j1/` |
| **Lần chạy 2** | **S1** — Joint MF-SR-OCR, SR ×2 | `s1_seed{42,100,2026}` | `results/multi-seed/s1_mf_sr_ocr/` |
| **Lần chạy 3** | **S4** — Joint MF-SR-OCR, SR ×1 | `s4_seed{42,100,2026}` | `results/multi-seed/s4_sr_scale1/` |

> ⚠️ Chuỗi `j1p` chỉ là **tên experiment** lúc gõ lệnh (`p` = "proper flags"), không
> phải tên model. Model tên là **J1**. Giữ nguyên `j1p` ở mọi đường dẫn/lệnh vì đó
> là tên file có thật trên đĩa.
>
> ⚠️ **J1 ở đây ≠ J1 lịch sử (76.88%)**. J1 lịch sử dùng STN pool `(1,1)`, không
> domain-match/EMA/constrained decode, params 29,313,452. Hai số **không đặt chung cột**.

---

## 0. Bảng tổng quan 3 lần chạy

| | **Lần 1 — J1** | **Lần 2 — S1** 🎯 | **Lần 3 — S4** |
|---|---|---|---|
| Nhánh SR | ❌ không có | ✅ ×2 multi-frame | ✅ ×1 multi-frame |
| DCNv2 | ❌ | ✅ | ✅ |
| `width_downsample` | 8 (mặc định) | 8 (mặc định) | **4** |
| Chuỗi CTC `T` | **16** | 32 | 32 |
| Params | 29,442,700 | 29,577,214 | 29,549,470 |
| GFLOPs/track | **26.14** | 109.08 | chưa đo |
| Phút/epoch (đo thật) | **2.60** | 9.67 | 4.28 |
| Tổng giờ GPU (3 seed) | **5.55 h** | 22.24 h | 11.27 h |
| **Val Acc (Mean ± Std)** | 80.45% ± 0.45 | **79.95% ± 0.15** | 79.48% ± 0.44 |
| **Std** | 0.45 | **0.15** 🥇 | 0.44 |
| CER (Mean ± Std) | **0.0525 ± 0.0007** | 0.0541 ± 0.0016 | 0.0543 ± 0.0001 |
| Track sai độ dài | **0/999 cả 3 seed** | 1 (seed 2026) | 1 (seed 100) |
| Vai trò trong paper | ablation _"bỏ hẳn SR"_ | 🎯 **phương pháp đề xuất** | ablation _"bỏ phóng to ảnh"_ |

**Phút/epoch tính từ cột `epoch_time_s`** của toàn bộ 424 epoch trong 9 file
`history_*.csv` — **số đo thật, không phải ước tính**.

---

# PHẦN A — 3 NHÓM GIẢI PHÁP (theo đúng thứ tự review)

## Nhóm 1 — Hoàn thiện luồng Joint End-to-End Multi-Task Training

### 1.1. Công thức review đề xuất

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}} \cdot \mathcal{L}_{\text{SR}} + \lambda_{\text{Perceptual}} \cdot \mathcal{L}_{\text{VGG}}$$

$$\mathcal{L}_{\text{SR}} = \frac{1}{N} \sum \left\| \text{Warp}_{\theta}(I_{\text{SR}}) - \text{Warp}_{\theta}(I_{\text{HR}}) \right\|_1 \quad \text{(review ghi: Smooth L1)}$$

$$\lambda_{\text{SR}} = 0.1, \qquad \lambda_{\text{Perceptual}} = 0.01$$

### 1.2. Công thức CODE THẬT SỰ CHẠY (đã grep xác nhận)

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}} \cdot \Big( \underbrace{\big\| I_{\text{SR}} - \text{Warp}_{\text{sg}[\theta]}(I_{\text{HR}}) \big\|_1}_{\text{L1 thuần}} + \beta \cdot \mathcal{L}_{\text{Edge}} + \alpha \cdot \mathcal{L}_{\text{VGG}} \Big)$$

$$\mathcal{L}_{\text{Edge}} = \big\| \text{Sobel}(I_{\text{SR}}) - \text{Sobel}(\text{Warp}_{\text{sg}[\theta]}(I_{\text{HR}})) \big\|_1$$

Nguồn: [`src/training/trainer.py:322-336`](../src/training/trainer.py) (hợp tổng),
[`src/training/trainer.py:263-272`](../src/training/trainer.py) (warp + detach),
[`src/training/losses.py:72-89`](../src/training/losses.py) (`SRPixelLoss`).

**4 chỗ công thức review lệch so với code** — đều là **việc sửa chữ trong paper, 0 GPU**:

| # | Review viết | Code thật | Hệ quả |
|---|---|---|---|
| 1 | $\text{Warp}_\theta(I_{\text{SR}})$ | **Không warp** `I_SR` | `I_SR` đã nằm trong khung STN đã nắn → warp 2 lần là sai |
| 2 | "Smooth L1" | `F.l1_loss` — **L1 thuần** | Chỉ sai tên; đã chốt **không đổi code** (đổi thì phải chạy lại toàn bộ) |
| 3 | $\theta$ bình thường | $\text{sg}[\theta]$ — **`theta_sel.detach()`** | Gradient của $\mathcal{L}_{SR}$ **không** chảy về STN. Là chủ đích: SR chỉ được làm ảnh nét hơn, không được kéo STN về hình học "dễ tái tạo nhất" |
| 4 | $\lambda_{\text{Perc}} \cdot \mathcal{L}_{\text{VGG}}$ **song song** | $\lambda_{\text{SR}} \cdot (\ldots + \alpha \mathcal{L}_{\text{VGG}})$ — **lồng trong** | Chỉ tương đương khi $\lambda_{\text{Perc}} = \lambda_{\text{SR}} \times \alpha$. Ở ablation từng bật perceptual: $0.1 \times 0.1 = 0.01$ ✅ khớp review |

### 1.3. Giá trị siêu tham số THẬT của 3 lần chạy

| | $\lambda_{\text{SR}}$ | $\alpha$ (perceptual) | $\beta$ (edge) | Ghi chú |
|---|---:|---:|---:|---|
| **Lần 1 — J1** | — | — | — | Không có nhánh SR → $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}}$ |
| **Lần 2 — S1** | 0.1 | **0.0** | **0.0** | Đúng $\lambda_{SR}$ review yêu cầu |
| **Lần 3 — S4** | 0.1 | **0.0** | **0.0** | Giống S1, chỉ khác `sr_scale` |

> 🚨 **Cả 3 lần chạy có error bar đều chạy `perceptual = 0.0`** — đã verify banner
> 8/8 log. Số hạng $\lambda_{\text{Perceptual}} \mathcal{L}_{\text{VGG}}$ trong công
> thức chỉ được bật ở **một ablation 1 seed** (nay lưu ở `backup/report/`). Paper phải ghi
> rõ số hạng này là **ablation 1 seed, chưa xác nhận**.

### ✅ Checklist Nhóm 1

> 🚨 **CÓ CẦN TRAIN LẠI KHÔNG? — ❌ KHÔNG.** Cả 4 lỗi là **sai lệch giữa mô tả trong
> paper và code đã chạy**; code vẫn đúng, mọi số liệu J1/S1/S4 giữ nguyên hiệu lực.
> **0 giờ GPU.**
>
> ✅ **Công thức đã sửa, sẵn để dán** (LaTeX + Unicode + 3 câu chú thích bắt buộc):
> [paper/loss_formula_corrected.md](paper/loss_formula_corrected.md)

- [x] Xác nhận quy đổi $\lambda_{\text{Perceptual}} = \lambda_{\text{SR}} \times \alpha$
- [x] Verify `perceptual=0.0` trên banner của cả 3 lần chạy
- [x] Chốt **giữ L1**, không đổi sang Smooth L1 (đổi code = chạy lại 39h GPU)
- [x] **Soạn xong bản công thức đã sửa cả 4 lỗi** + 3 câu chú thích bắt buộc
- [x] Xác nhận **không cần train lại**
- [ ] ✍️ Dán khối công thức vào bản thảo, thay khối cũ
- [ ] ✍️ Thêm 3 câu chú thích ($\alpha=0$ · stop-gradient · $\mathcal{L}_{\text{Edge}}$)
- [ ] ✍️ Rà lại phần Method xem còn chỗ nào ghi "Smooth L1" hoặc warp kép không

---

## Nhóm 2 — Chuẩn hoá quy trình Multi-Seed Verification ✅ XONG

### 2.1. Review yêu cầu

```python
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2.2. Code đã có — [`src/utils/common.py:12-35`](../src/utils/common.py)

Bật bằng cờ **`--no-cudnn-benchmark`**. **Đầy đủ hơn review**: thêm `PYTHONHASHSEED`
và `torch.cuda.manual_seed` (bản review chỉ có `manual_seed_all`).

### 2.3. Cách verify — chạy trên cả 3 lần chạy

```bash
# 1) Banner phải in đúng chế độ deterministic
grep -h "Seed.*cudnn" results/multi-seed/*/log_*.txt

# 2) CSV phải đủ 14 cột (có val_cer, val_ned, *_time_s)
head -1 results/multi-seed/s1_mf_sr_ocr/history_s1_seed42.csv

# 3) Không có batch NaN nào
grep -c ",0," results/multi-seed/*/history_*.csv
```

| Kiểm tra | Kết quả |
|---|---|
| `cudnn.benchmark: False (deterministic: True)` | ✅ **8/8 log** (thiếu `log_j1p_seed42.txt`, CSV vẫn đủ) |
| Seed in ra khớp `--experiment-name` | ✅ 8/8 |
| CSV đủ 14 cột | ✅ **9/9 file** |
| `nan_batches = 0` mọi epoch | ✅ **424/424 epoch** |
| Val Acc khớp khi chấm lại từ nhãn gốc | ✅ **9/9 run** (chấm `submission_*.txt` với `plate_text` trong `annotations.json`) |

### ✅ Checklist Nhóm 2

- [x] `seed_everything` có `deterministic=True`, `benchmark=False`
- [x] `tools/aggregate_seeds.py` (Mean ± Std + CI 95% + kiểm định)
- [x] Smoke test 1 epoch xác nhận deterministic ăn + CSV đủ 14 cột
- [x] Verify trên **run thật** (9 run), không chỉ smoke test

---

## Nhóm 3 — Tối ưu chống Overfitting ⏭️ **ĐÃ CHỐT KHÔNG ÁP DỤNG (2026-08-04)**

> ⏭️ **Quyết định**: hoãn Nhóm 3, đưa vào **Future work**. Đây **không** nằm trong 4
> bước bắt buộc của review; áp dụng là thí nghiệm mới tốn ~34h GPU và **bắt buộc
> multi-seed lại cả 3 model** để bảng ablation nhất quán — không thể so
> S1-có-regularization với J1-không-regularization. Trong khi đó **toàn bộ việc bắt
> buộc còn lại đều 0 GPU**.
>
> ✅ **Code vẫn giữ trong repo**, mặc định TẮT nên không ảnh hưởng gì — 9 run
> J1/S1/S4 tái lập bit-for-bit. Thêm cờ là chạy được nếu sau này đổi ý.
>
> ✍️ **Phải ghi vào paper**: bằng chứng overfit 9/9 run (bảng dưới) là quan sát có giá
> trị → nêu ở **Limitations/Future work** kèm câu *"các biện pháp chống overfitting đã
> được cài đặt nhưng chưa khảo sát trong phạm vi bài này."*

| Review yêu cầu | Hiện trạng | Trạng thái |
|---|---|---|
| `RandomBrightnessContrast` | Đã có trong train transform (`transforms.py:23`) | ✅ |
| `ShiftScaleRotate` (≈`Affine`) | Đã có (`transforms.py:15`) | ✅ |
| `MotionBlur`, `GaussNoise` | Chỉ có trong **degradation ảnh synthetic**, **không** trong train transform | ❌ |
| `Dropout(0.3)` trước **FC** | Đã có nhưng là **0.25** (`crnn.py:132-136`) | 🟡 |
| `Dropout(0.3)` giữa 2 lớp **BiLSTM** | Đã có nhưng là **0.25** (`crnn.py:130`) | 🟡 |
| `Dropout(0.3)` **trước** BiLSTM | Chưa có — vị trí duy nhất còn thiếu | ❌ |
| `weight_decay` 1e-4 → 1e-3 | Cả 9 run vẫn chạy **1e-4** (cờ `--weight-decay` đã có sẵn) | ❌ |
| `patience` 18 → 12 | `configs/config.py:122` vẫn **18** | ❌ |

### 📌 Bằng chứng overfit — **9/9 run, không còn là quan sát 1 seed**

| Lần chạy | Seed | `val_loss` chạm đáy | `val_acc` đỉnh | `val_loss` cuối | `train_loss` cuối |
|---|---|---:|---:|---:|---:|
| 1 — J1 | 42 | ep 16 (0.1836) | ep 22 | 0.2665 | **0.0160** |
| 1 — J1 | 100 | ep 12 (0.1820) | ep 17 | 0.2516 | **0.0209** |
| 1 — J1 | 2026 | ep 11 (0.1819) | ep 35 | 0.3044 | **0.0074** |
| 2 — S1 | 42 | ep 20 (0.1875) | ep 28 | 0.2849 | 0.0344 |
| 2 — S1 | 100 | ep 16 (0.1852) | ep 32 | 0.2701 | 0.0312 |
| 2 — S1 | 2026 | ep 19 (0.1888) | ep 24 | 0.2556 | 0.0411 |
| 3 — S4 | 42 | ep 15 (0.2065) | ep 41 | 0.3415 | 0.0273 |
| 3 — S4 | 100 | ep 22 (0.1924) | ep 23 | 0.2710 | 0.0408 |
| 3 — S4 | 2026 | ep 19 (0.1889) | ep 40 | 0.2985 | 0.0283 |

Val loss chạm đáy rất sớm (**ep 11–22**) rồi **tăng 38–67%** tới lúc dừng, trong khi
train loss tụt về ~0.01–0.04 (model thuộc lòng tập train). Val acc vẫn nhích lên tới
tận ep 17–41 **dù val loss đã tăng**.

📌 **Lần chạy 1 (J1) overfit sớm nhất và nặng nhất** — val loss chạm đáy ngay ep 11–16
(so với 15–22 của S1/S4) và train loss xuống tận **0.0074**. Hợp lý: J1 không có nhánh
SR đóng vai **regularizer đa nhiệm**. Đây là lập luận đáng nêu khi bảo vệ việc giữ
nhánh SR trong phương pháp đề xuất — dù SR không cải thiện exact match, nó **có** ghìm
được mức overfit.

→ **Nhóm 3 là hướng cải thiện có cơ sở nhất còn lại.**

### ✅ Checklist Nhóm 3

> 🚨 **CÓ CẦN TRAIN LẠI KHÔNG? — ✅ CÓ, bắt buộc.** Ngược hẳn Nhóm 1. Dropout /
> weight decay / augmentation / patience đều là **tham số lúc HUẤN LUYỆN** — đổi
> chúng **không tác động gì** lên 9 checkpoint đã có. Muốn có số mới phải train lại;
> muốn đưa vào paper phải **multi-seed lại** (~39h GPU).
>
> ✅ **Phần code thì đã xong hết (0 GPU)** — cài theo nguyên tắc **opt-in, mặc định
> giữ nguyên hành vi cũ**, nên 9 run J1/S1/S4 vẫn tái lập được bit-for-bit.

**Đã có sẵn, chỉ cần thêm cờ khi chạy — không cần code:**

- [x] `--weight-decay` + `--wd-skip-bias-norm` (param-grouping)
- [x] `--patience` (⚠️ tài liệu cũ ghi sai là phải sửa `config.py:122`)
- [x] `--rnn-dropout` (0.25 → 0.3 bằng CLI)
- [x] Dropout trước FC + giữa 2 lớp BiLSTM đã có sẵn (0.25)

**Đã cài thử rồi REVERT (2026-08-04) — repo về đúng trạng thái lúc chạy 9 run:**

- [ ] `--pre-rnn-dropout` — Dropout **trước** BiLSTM. Đã cài, đã revert.
- [ ] `--aug-level strong` — `full` + `MotionBlur` + `GaussNoise`. Đã cài, đã revert.

✅ **Verify sau revert**: params S1 vẫn **29,577,214** · `aug full` vẫn **10 phép** ·
`WD / patience / rnn_dropout` = **1e-4 / 18 / 0.25** — không còn dấu vết nào.
Cách cài lại nếu đổi ý: [checklist_review.md §Nhóm 3](checklist_review.md).

**⏭️ Đã chốt KHÔNG làm — đưa vào Future work:**

- [ ] ⏭️ Chạy thử 1 cấu hình với cụm cờ Nhóm 3 đầy đủ
- [ ] ⏭️ Nếu có cải thiện → multi-seed lại 3 seed mới đưa vào paper

Kế hoạch 3 giai đoạn có cổng quyết định (dùng khi đổi ý) + lệnh đầy đủ:
[checklist_review.md §Nhóm 3](checklist_review.md).

---

# PHẦN B — 4 BƯỚC PHẢI LÀM NGAY

## 🔴 Bước 1 — Multi-Seed Runs cho J1, S1 và S4 ✅ XONG

> **Review yêu cầu**: chạy `train.py` với 3 seed `42/100/2026` cho cả J1, S1, S4 →
> tính Mean ± Std → ghi vào CSV log.

### B0 — Chuẩn bị (bắt buộc, 30 giây)

```bash
mkdir -p results          # `results/` bị gitignore → server clone về KHÔNG có thư mục
                          # này; `tee` chạy TRƯỚC python nên sẽ fail và MẤT LOG
pip install scikit-image  # cho PSNR/SSIM ở Bước 2
```

### B1 — Smoke test 1 epoch (xác nhận deterministic + CSV đủ cột)

```bash
python train.py --preset stable --experiment-name smoke --epochs 1 \
  --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --width-downsample 4 \
  --decode constrained --use-ema --no-cudnn-benchmark \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_smoke.txt
```

**Điều kiện đạt**: header CSV kết thúc bằng
`...,nan_batches,val_cer,val_ned,train_time_s,val_time_s,epoch_time_s` → ✅ đã đạt
(`results/multi-seed/s4_sr_scale1/history_smoke.csv`).

---

### 🔹 LẦN CHẠY 1 — J1 (không SR)

**Mục đích**: trả lời câu hỏi headline _"pipeline SR đề xuất có thật sự hơn backbone
không SR không?"_ — cho claim đó một **error bar**.

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

**Dùng gì và vì sao:**

| Cờ | Vì sao dùng |
|---|---|
| _(không có `--use-sr`)_ | Đây **là** biến đang ablate — bỏ hẳn nhánh SR để đo đóng góp thật của nó |
| _(không có `--use-dcn`)_ | DCNv2 chỉ tồn tại để align frame **trước khi** SR hợp nhất; bỏ SR thì bỏ luôn cho sạch cụm biến |
| `--backbone-norm group` | GroupNorm thay BatchNorm — batch nhỏ (32÷2 accum) làm thống kê BN nhiễu |
| `--lr-domain-match` | Ép ảnh vào đúng miền LR thật (46×19px) thay vì ảnh nét giả tạo |
| `--decode constrained` | 20.000 nhãn chỉ có 2 layout (`LLLNLNN`, `LLLNNNN`) → khoá cứng 6/7 vị trí lớp chữ/số |
| `--use-ema` | EMA decay 0.999 — làm phẳng dao động cuối training |
| `--no-cudnn-benchmark` | **Cốt lõi của Nhóm 2** — tắt chọn thuật toán conv ngẫu nhiên |
| `--aug-level full` | Giữ đúng augment như S1/S4 để so công bằng |
| `2>&1 \| tee` | Vài run cũ **mất vĩnh viễn** số liệu thời gian chỉ vì quên `tee` |

> ⚠️ **Khác kế hoạch ban đầu**: kế hoạch cũ định chạy cờ **lịch sử** (`--stn-pool 1,1`)
> để tái lập 76.88%. Run thật dùng **đúng bộ cờ nền của S1/S4, chỉ bỏ SR + DCN**.
> Đây là lựa chọn **tốt hơn về khoa học** (ablation 1-cụm-biến sạch so với S1) nên
> giữ nguyên — nhưng hệ quả là **80.45% không so được với 76.88%**.
>
> ⚠️ Không `--use-sr` → width **không** nhân đôi → **`T = 128 ÷ 8 = 16`**.

**Kết quả:**

| Seed | Track đúng | Val Acc | Best epoch | Tổng epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 799/999 | 79.98% | 22 | 40 | 0.2030 | 0.0525 | 0.9642 | 6 | 0 |
| 100 | **808/999** | **80.88%** | 17 | 35 | 0.1855 | 0.0532 | 0.9589 | 11 | 0 |
| 2026 | 804/999 | 80.48% | 35 | 53 | 0.2571 | 0.0518 | 0.9765 | 2 | 0 |
| **Mean ± Std** | | **80.45% ± 0.45** | | | | **0.0525 ± 0.0007** | | | **0/999** |

---

### 🔹 LẦN CHẠY 2 — S1 (SR ×2 multi-frame + DCN) 🎯 **phương pháp đề xuất**

**Mục đích**: con số **headline** của paper. Kiểm tra 79.78% (1 seed) có trụ được không.

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

**Dùng gì và vì sao** (chỉ liệt kê phần **khác Lần 1**):

| Cờ | Vì sao dùng |
|---|---|
| `--use-sr --sr-scale 2` | Nhánh MFSR phóng to ×2 → ảnh 64×256, đồng thời **`T` nhảy 16 → 32** |
| `--use-dcn` | DCNv2 align 5 frame trước khi hợp nhất; kernel **identity-init** (một run cũ init ngẫu nhiên → hỏng) |
| `--lambda-sr 0.1` | Đúng $\lambda_{SR}$ review chỉ định |
| _(KHÔNG có `--width-downsample`)_ | Dùng mặc định **8** — đây là 1 trong 2 chỗ khác S4 |

> ⚠️ **Hai chỗ dễ gõ nhầm thành S4**: S1 có `--sr-scale 2` (không phải 1) và **không**
> có `--width-downsample 4`. Nhầm là dựng sai kiến trúc, chạy 22 giờ ra kết quả vô nghĩa.

**Kết quả:**

| Seed | Track đúng | Val Acc | Best epoch | Tổng epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 797/999 | 79.78% | 28 | 46 | 0.2094 | 0.0541 | 0.9655 | 6 | 0 |
| 100 | 800/999 | 80.08% | 32 | 50 | 0.2141 | 0.0525 | 0.9692 | 4 | 0 |
| 2026 | 799/999 | 79.98% | 24 | 42 | 0.1988 | 0.0556 | 0.9585 | 12 | 1 |
| **Mean ± Std** | | **79.95% ± 0.15** | | | | 0.0541 ± 0.0016 | | | |

🎯 **Std 0.15 — nhỏ nhất trong 3 lần chạy** (J1: 0.45, S4: 0.44). Đây là lý do chính
chọn S1 làm phương pháp đề xuất.

---

### 🔹 LẦN CHẠY 3 — S4 (SR ×1, giữ `T=32` bằng backbone)

**Mục đích**: tách **"T confound"** — SR ×2 vừa phóng to ảnh **vừa** làm `T` nhảy
16→32, nên không biết lợi ích đến từ đâu. S4 giữ `T=32` **mà không** phóng to ảnh.

$$T = \frac{\text{IMG\_WIDTH} \times (\text{SR\_SCALE nếu USE\_SR})}{\text{WIDTH\_DOWNSAMPLE}}$$

- S1: $T = (128 \times 2) / 8 = 32$
- S4: $T = (128 \times 1) / 4 = 32$ ← **cùng `T`, khác cách đạt được**
- J1: $T = 128 / 8 = 16$

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

**Dùng gì và vì sao** (chỉ phần **khác Lần 2**):

| Cờ | Vì sao dùng |
|---|---|
| `--sr-scale 1` | HR gốc chỉ ~115×42px; target ×2 (256px) có **~55% là nội suy bicubic thuần** — tức đang bắt model học lại phép nội suy. ×1 giữ output 32×128 ≈ đúng độ phân giải HR thật |
| `--width-downsample 4` | Bù lại việc bỏ phóng to ảnh, để **`T` vẫn = 32** → tách được biến `T` khỏi biến "SR có phóng to hay không" |

**Kết quả:**

| Seed | Track đúng | Val Acc | Best epoch | Tổng epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 789/999 | 78.98% | 41 | 59 | 0.2887 | 0.0543 | 0.9748 | 0 | 0 |
| 100 | 797/999 | 79.78% | 23 | 41 | 0.1945 | 0.0543 | 0.9580 | 11 | 1 |
| 2026 | 796/999 | 79.68% | 40 | 58 | 0.2628 | 0.0542 | 0.9743 | 1 | 0 |
| **Mean ± Std** | | **79.48% ± 0.44** | | | | 0.0543 ± 0.0001 | | | |

---

### B5 — Tổng hợp Mean ± Std + kiểm định

```bash
# Mean ± Std từng lần chạy
for C in j1p s1 s4; do
  python tools/aggregate_seeds.py --from-logs results/multi-seed/*/log_${C}_seed*.txt \
    --label "${C:u}"
done

# Kiểm định S1 (đề xuất) ↔ J1 (bỏ hẳn SR) — phép so quan trọng nhất
python tools/aggregate_seeds.py \
  --acc 79.7798 80.0801 79.9800 --label "S1 (de xuat)" \
  --baseline-acc 79.9800 80.8809 80.4805 --baseline-label "J1 (khong SR)"

# Kiểm định S1 ↔ S4
python tools/aggregate_seeds.py \
  --acc 79.7798 80.0801 79.9800 --label "S1 (SR x2)" \
  --baseline-acc 78.9790 79.7798 79.6797 --baseline-label "S4 (SR x1)"
```

**Kết quả kiểm định** (ngưỡng: chênh > 2× sai số hiệu mới coi là thật):

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| J1 vs S1 | +0.50 | ±0.28 | ⚠️ trong nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

### 🚨 Hai kết quả âm tính bắt buộc công bố

1. **Nhánh SR chưa chứng minh được đóng góp đo được.** Lần 1 (bỏ hẳn SR/DCN) **hoà**
   Lần 2 trong khi rẻ hơn **3.76× compute**. Kết luận trung thực: _"SR không cho thấy
   lợi ích trên tập val này"_ — **không** phải _"bỏ SR thì tốt hơn"_ (chênh trong
   nhiễu, không kết luận được chiều nào).
2. **Giả thuyết `T=32` bị bác bỏ.** Lần 1 chạy `T=16` mà vẫn ngang Lần 2 và **hơn**
   Lần 3 (cả hai đều `T=32`). Phần cải thiện thật đến từ **cụm cờ nền**
   (STN pool `(4,8)` + `--lr-domain-match` + constrained decode + EMA).

### 🔬 Phát hiện phụ — `cudnn.benchmark` KHÔNG luôn thổi phồng số

Cùng seed 42, chỉ đổi chế độ cudnn:

| | 1-seed cũ (`benchmark=True`) | Multi-seed (`deterministic`) | Chênh |
|---|---:|---:|---:|
| **Lần 2 — S1** | 797/999 (79.78%) | 797/999 (79.78%) | **0 track** |
| **Lần 3 — S4** | 805/999 (80.58%) | 789/999 (78.98%) | **−16 track** |

→ Không được khái quát _"1 seed + `benchmark=True` luôn thổi phồng"_. Mức nhạy cảm
**phụ thuộc cấu hình**. Đây cũng là lý do thứ hai chọn S1: con số của nó **tái lập
được nhất** trong cả project.

### ✅ Checklist Bước 1

- [x] `mkdir -p results` trên server
- [x] Smoke test 1 epoch — CSV đủ 14 cột, deterministic ăn
- [x] **Lần chạy 1 (J1) × 3 seed** → 80.45% ± 0.45
- [x] **Lần chạy 2 (S1) × 3 seed** → 79.95% ± 0.15
- [x] **Lần chạy 3 (S4) × 3 seed** → 79.48% ± 0.44
- [x] `aggregate_seeds.py` — bảng Mean ± Std đủ 3 model + 3 phép kiểm định cặp
- [x] Chấm lại **toàn bộ 9 submission** với nhãn gốc, không chỉ tin `val_acc` trong log
- [x] Báo cáo: [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md)
- [x] Chốt cấu hình cuối = **S1** (trước khi nhìn bất kỳ số test nào)
- [x] 🆕 **`aggregate_seeds.py --output-csv`** (thêm 2026-08-04) — xuất bảng Mean ± Std
      ra CSV 13 cột, `--append` để gom cả 3 model vào 1 file

#### "Ghi nhận bảng kết quả 3 seeds vào CSV log" — trạng thái từng mục

| Review cần | Có chưa | Ở đâu |
|---|:---:|---|
| CSV per-epoch từng seed (14 cột) | ✅ | `results/multi-seed/<cfg>/history_*_seed*.csv` |
| Log stdout từng seed | ✅ 8/9 | `results/multi-seed/<cfg>/log_*_seed*.txt` (thiếu `log_j1p_seed42.txt`) |
| Dự đoán từng track để chấm lại | ✅ | `results/multi-seed/<cfg>/submission_*_seed*.txt` |
| Checkpoint từng seed | ✅ | `results/multi-seed/<cfg>/*_best.pth` |
| Bảng tổng hợp Mean ± Std | ✅ (`.md`) | [multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md) |
| Bảng tổng hợp dạng **CSV** | ✅ | `aggregate_seeds.py --output-csv` (thêm 2026-08-04) |

> ⚠️ `results/` bị **gitignore** → dù có CSV vẫn phải chép số vào `.md` trong `report/`
> mới commit được. Nên mục CSV còn thiếu là **hình thức**, không chặn gì.

---

## 🔴 Bước 2 — Bổ sung CER, NED và PSNR/SSIM ✅ XONG

> **Review yêu cầu**: (1) CER + NED bằng `editdistance` trong `postprocess.py`;
> (2) PSNR + SSIM giữa $I_{SR}$ và $I_{HR}$ bằng `torcheval`/`skimage.metrics` trên
> **Scenario-A val tracks**; xuất ra console và log validation.

### 📍 CHẠY Ở ĐÂU — điều review KHÔNG nói rõ

Review không chỉ định các chỉ số này đo trên checkpoint multi-seed hay 1-seed. Thực tế
đã làm **hai kiểu khác nhau cho hai nhóm chỉ số** — phải ghi rõ trong paper:

| Chỉ số | Chạy ở đâu | Cách chạy | Phủ được lần chạy nào |
|---|---|---|---|
| **CER / NED** | ✅ **Trong training loop, cả 9 run multi-seed** | `Trainer.validate()` → console mỗi epoch + 2 cột CSV | **Cả 3 lần chạy, có Mean ± Std** |
| **CER / NED** (bổ sung) | Tính lại hậu kỳ từ `submission_*.txt` | không cần train lại | các run 1-seed cũ |
| **PSNR / SSIM** | ⚠️ **Hậu kỳ, trên checkpoint 1-SEED** ở `backup/mf_sr_ocr/` | `tools/eval_sr_quality.py` | **Chỉ S1–S4, KHÔNG có multi-seed, KHÔNG có J1** |

> 🚨 **Hai hệ quả phải nêu trong paper:**
> 1. **PSNR/SSIM chưa có error bar** — đo trên checkpoint 1 seed, khác nguồn với bảng
>    accuracy Mean ± Std.
> 2. **Lần chạy 1 (J1) không có PSNR/SSIM** — không phải thiếu sót mà là **giới hạn
>    cấu trúc**: J1 không có nhánh SR nên $I_{SR}$ **không tồn tại**.
>    [`eval_sr_quality.py:131`](../tools/eval_sr_quality.py) chặn thẳng.

### 2.1. CER và NED

$$\text{CER} = \frac{\sum_i \text{EditDistance}(\hat{y}_i, y_i)}{\sum_i |y_i|} \qquad \text{(mức corpus)}$$

$$\text{NED} = \frac{1}{N}\sum_i \frac{\text{EditDistance}(\hat{y}_i, y_i)}{\max(|y_i|, |\hat{y}_i|)} \qquad \text{(trung bình mỗi track)}$$

Cả hai **thấp = tốt**; paper thường ghi $1 - \text{NED}$ (cao = tốt).
Cài tại [`src/training/trainer.py:387-460`](../src/training/trainer.py).

```bash
# Verify CER in ra console mỗi epoch
grep -m3 "CER:" results/multi-seed/s1_mf_sr_ocr/log_s1_seed42.txt

# Verify 2 cột val_cer, val_ned có trong CSV
head -1 results/multi-seed/s1_mf_sr_ocr/history_s1_seed42.csv
```

**Kết quả — có Mean ± Std cho cả 3 lần chạy:**

| Lần chạy | Exact Match | CER ↓ |
|---|---:|---:|
| **1 — J1** | 80.45% ± 0.45 | **0.0525 ± 0.0007** 🥇 |
| **2 — S1** | **79.95% ± 0.15** | 0.0541 ± 0.0016 |
| **3 — S4** | 79.48% ± 0.44 | 0.0543 ± **0.0001** |

📌 **Phát hiện chỉ multi-seed mới thấy**: Lần 3 **ổn định hơn 16× về CER**
(std 0.0001 vs 0.0016) nhưng **kém ổn định hơn 3× về exact match** (0.44 vs 0.15) —
hai đại lượng **không đi cùng chiều**. Đáng nêu trong paper.

### 2.2. PSNR và SSIM

> ✅ **Chạy trên checkpoint multi-seed** (seed 42) — không dùng checkpoint 1-seed cũ
> trong `backup/mf_sr_ocr/` nữa, để số PSNR khớp nguồn với bảng accuracy Mean ± Std.
> Chỉ **S1** (bắt buộc) và **S4** (nên) — **không chạy S2/S3** (đã ra ngoài phạm vi,
> dữ liệu ở `backup/`) và **không chạy được J1** (không có nhánh SR, xem cảnh báo dưới).

```bash
# S1 — BẮT BUỘC. --num-workers 0 để tái lập tuyệt đối (mặc định 4 gây dao động ±0.05 dB)
python tools/eval_sr_quality.py --lr-domain-match --num-workers 0 \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --output-csv results/multi-seed/s1_mf_sr_ocr/sr_quality_s1_seed42.csv

# S4 — NÊN chạy. sr_scale=1 => BẮT BUỘC --width-downsample 4
python tools/eval_sr_quality.py --lr-domain-match --num-workers 0 --width-downsample 4 \
  --checkpoint results/multi-seed/s4_sr_scale1/s4_seed42_best.pth \
  --output-csv results/multi-seed/s4_sr_scale1/sr_quality_s4_seed42.csv
```

**Con số đáng đọc là cột "Chênh"** (so với `base` = ảnh chưa qua SR), không phải PSNR
tuyệt đối. Bảng dưới đây vẫn là số đo trên checkpoint **1-seed cũ** (chưa chạy lại lệnh
trên) — cập nhật sau khi có kết quả từ checkpoint multi-seed:

| Cấu hình | PSNR (SR) | PSNR (base) | **Chênh** | SSIM (SR) | SSIM (base) | **Chênh** |
|---|---:|---:|---:|---:|---:|---:|
| **Lần 2 — S1** (×2) | 16.6827 | 15.6110 | +1.0717 dB | 0.4179 | 0.3481 | +0.0698 |
| **Lần 3 — S4** (×1) | 17.5249 | 16.5011 | +1.0238 dB | 0.5027 | 0.4408 | +0.0618 |
| **Lần 1 — J1** | — | — | — | — | — | — |

> ⚠️ **Không đặt PSNR tuyệt đối của S4 chung cột với S1**: S1 xuất ảnh 64×256,
> S4 xuất 32×128 — **hai thang khác nhau**. Chỉ so được cột "Chênh".
>
> ⚠️ Số dao động **~±0.05 dB** giữa các lần chạy (pipeline degradation ngẫu nhiên,
> `--num-workers > 0` khiến mỗi worker có trạng thái random riêng). Nhỏ hơn nhiều
> khoảng cách giữa các cấu hình nên không đổi kết luận nào.

### 2.3. 🔬 Kết quả mạnh nhất — PSNR **NGHỊCH** với khả năng đọc, 3 mức bằng chứng

| Mức | Bằng chứng | Chiều |
|---|---|---|
| **Track** (n=999) | Track đọc **sai** có PSNR **cao hơn ~2 dB**; $r \approx -0.35 \ldots -0.41$, nhất quán cả 4 cấu hình | **nghịch** |
| **Cấu hình** | Ablation có PSNR cao nhất toàn dự án (+2.29 dB) lại có OCR **kém nhất** | **nghịch** |
| **Kiến trúc** | **Lần 1 (J1) bỏ hẳn SR → không có PSNR nào → OCR không hề thua** (điểm TB cao nhất) | **nghịch** |

| Cấu hình | PSNR track **đọc đúng** | PSNR track **đọc sai** | Chênh | $r$(PSNR, đúng) |
|---|---:|---:|---:|---:|
| S1 | 16.288 | 18.240 | **−1.952 dB** | **−0.362** |
| S4 | 17.080 | 19.371 | −2.291 dB | −0.400 |

→ Khi reviewer hỏi _"sao không tối ưu theo PSNR"_, câu trả lời không còn là "PSNR không
phản ánh OCR" mà là **"PSNR NGHỊCH với OCR, có bằng chứng ở cả 3 mức"**.

**Giả thuyết giải thích** (chưa kiểm chứng riêng): PSNR bị chi phối bởi *nội dung ảnh*
hơn chất lượng SR. Ảnh mờ nhoè/tương phản thấp → sai khác pixel nhỏ → **PSNR cao**,
nhưng đúng là loại khó đọc nhất. PSNR đang đo _"ảnh này dễ tái tạo tới đâu"_, còn OCR
cần _"ảnh này chứa bao nhiêu chi tiết đọc được"_.

### ⚠️ 3 chỗ cố ý lệch so với nguyên văn review

| # | Review yêu cầu | Đã làm | Lý do |
|---|---|---|---|
| 1 | Dùng thư viện `editdistance` | Dùng `edit_distance` có sẵn trong `postprocess.py` (Levenshtein có cache) | Kết quả tương đương, **không thêm dependency**. Verify: `grep -rn editdistance src/ tools/` → 0 kết quả |
| 2 | Đo PSNR/SSIM trên **Scenario-A** val tracks | Đo trên **999 track Scenario-B** | Val **không có track Scenario-A nào** (0/999), và cả 10.000 track Scenario-A **nằm trong tập TRAIN** → đo ở đó là đo trên dữ liệu đã học, **không hợp lệ** |
| 3 | PSNR/SSIM trong "log validation" | Là **tool hậu kỳ**, không nằm trong log mỗi epoch | Nhét vào training loop sẽ buộc multi-seed chạy lại từ đầu. ⚠️ **CER/NED thì KHÔNG lệch** — có đủ trong console + CSV đúng như review yêu cầu |

### ✅ Checklist Bước 2

- [x] CER + NED trong `validate()` → console mỗi epoch + 2 cột CSV (9/9 run)
- [x] CER/NED cho các run 1-seed cũ (tính lại từ submission, không train lại)
- [x] **CER Mean ± Std cho cả 3 lần chạy**
- [x] `tools/eval_sr_quality.py` — PSNR/SSIM bằng `skimage.metrics`
- [x] PSNR/SSIM cho 4 cấu hình S1–S4 (999 track mỗi cấu hình)
- [x] Tương quan PSNR ↔ đọc đúng ở mức từng track (n=999)
- [x] Báo cáo: [buoc2_metrics.md](buoc2_metrics.md)
- [ ] _(nếu muốn)_ Chạy lại PSNR/SSIM trên checkpoint **multi-seed seed 42** để cùng
      nguồn với bảng accuracy

---

## 🟠 Bước 3 — Trực quan hoá định tính ✅ gần xong

> **Review yêu cầu**: script `tools/visualize_paper_figures.py` trích 10 track tiêu
> biểu (5 success + 5 failure), xuất grid **4 cột**
> $I_{LR} \rightarrow I_{SR} \rightarrow \text{Attention Heatmap} \rightarrow \text{Prediction}$.

### 📍 CHẠY Ở ĐÂU, VÀ VÌ SAO LÀ **4** BỘ HÌNH MÀ CHỈ **3** LẦN CHẠY

Đây là chỗ dễ nhầm nhất: **"4 bộ hình" và "3 model multi-seed" là hai tập hợp khác nhau.**

| | Có hình (Bước 3) | Có multi-seed (Bước 1) |
|---|:---:|:---:|
| **Lần 2 — S1** | ✅ | ✅ |
| **Lần 3 — S4** | ✅ | ✅ |
| **Lần 1 — J1** | 🟡 chạy được, chưa sinh | ✅ |

- **Vài ablation 1-seed cũng có hình**: lúc sinh hình thì cả 4 checkpoint S-series
  đã nằm sẵn trên đĩa, sinh thêm 2 bộ rất rẻ (~20 phút CPU, 0 GPU) và có ích để so
  **cùng một track qua nhiều cấu hình**. Quyết định cắt chúng khỏi multi-seed đến **sau đó**.
- **Lần 1 (J1) chưa sinh hình, nhưng chạy được** — không giống PSNR ở Bước 2. Tool
  `tools/visualize_paper_figures.py::frame_column` **tự chặn `frames is None`** và vẽ
  placeholder `"(khong co SR)"` ở cột $I_{SR}$ thay vì crash, nên chạy y nguyên với
  checkpoint J1, không cần sửa code.

### ⏱️ Thứ tự thời gian — hình chạy **TRƯỚC** multi-seed, không phải sau

mtime thật trên đĩa:

```
2026-08-01 17:27   mf_sr_ocr/*/*.pth            ← checkpoint 1-seed (nguồn của hình)
2026-08-02 00:39   mf_sr_ocr/*/paper_figures/   ← 4 bộ hình sinh ở đây
2026-08-02 23:57   multi-seed/{s1,s4}/*.pth     ← multi-seed Lần 2 + Lần 3 về sau
2026-08-04 00:34   multi-seed/j1/*.pth          ← multi-seed Lần 1 cuối cùng
```

→ Hình đi trước checkpoint multi-seed sớm nhất gần **1 ngày**, và trước Lần 1 tận
**2 ngày**. Toàn bộ hình hiện có đều sinh từ checkpoint **1 seed**.

| Đã sinh hình cho | Nguồn checkpoint | Có multi-seed không |
|---|---|---|
| S1, S4 (+ 2 ablation 1-seed) | `backup/mf_sr_ocr/<cfg>/*.pth` — **1 seed** (lịch sử, đã gỡ khỏi repo) | ❌ |
| **Lần 1 — J1** | **chưa sinh, nhưng chạy được** | ❌ |

> ✅ **Lệnh chạy lại — trên checkpoint multi-seed**, không dùng `backup/mf_sr_ocr/` nữa.
> **S1** (bắt buộc, dùng cho Figure 4) và **S4** (nên, để so cùng track qua 2 kiến trúc)
> — **không chạy S2/S3** (ngoài phạm vi). **J1** chạy được (tuỳ chọn, cho phụ lục) —
> tool tự vẽ placeholder ở cột `I_SR` thay vì crash, xem đính chính ở mục trên.

```bash
# S1 — BẮT BUỘC, cấu hình chính cho Figure 4
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --decode constrained --pick extreme \
  --output-dir results/multi-seed/s1_mf_sr_ocr/paper_figures

# S4 — NÊN chạy, để so cùng track qua 2 kiến trúc (BẮT BUỘC --width-downsample 4)
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/s4_sr_scale1/s4_seed42_best.pth \
  --width-downsample 4 --decode constrained --pick extreme \
  --output-dir results/multi-seed/s4_sr_scale1/paper_figures

# J1 — TUỲ CHỌN, cho phụ lục. Cột I_SR sẽ là placeholder "(khong co SR)", không phải lỗi
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/crnn_resblock_groupnorm_nosr_j1/j1p_seed42_best.pth \
  --decode constrained --pick extreme \
  --output-dir results/multi-seed/crnn_resblock_groupnorm_nosr_j1/paper_figures
```

Mỗi lệnh xuất `figure4_qualitative_grid.png` + 10 ảnh track riêng (5 đúng `NN_ok_*`,
5 sai `NN_err_*`). Mặc định `--pick extreme`: lấy case đúng **tự tin nhất** và case sai
**mơ hồ nhất** (hai đầu phân phối), thay vì mấy track đầu danh sách.

> 🚨 `*.pth` **không đi theo `git pull`** (113 MB/file) — phải upload checkpoint trước.
> Gặp `❌ Không tìm thấy checkpoint` là do thiếu file, không phải lỗi lệnh.

### Track trùng nhau giữa các cấu hình — tiện chọn Figure 4

| Track | Xuất hiện | Dùng để minh hoạ |
|---|---|---|
| `track_22161` | **SAI ở cả 4** | **giới hạn thật của dữ liệu**, không phải điểm yếu của một model |
| `track_19095` | SAI ở S1 và S4 | case khó nhất quán |
| `track_21455` | ĐÚNG ở S1 | case dễ, đọc chắc chắn |

### ⚠️ 2 caveat khi chốt hình

1. **Hình sinh từ checkpoint 1-seed**, trong khi số trong paper nay là multi-seed →
   hình và số đến từ **2 model khác nhau**. Với **Lần 3 (S4)** chênh này lớn nhất
   (checkpoint 1-seed đạt 805/999 còn mức thật ~794). Với **Lần 2 (S1)** chênh **bằng 0**
   (seed 42 ra đúng 797/999 ở cả 2 chế độ cudnn) → **dùng S1 làm Figure 4 là an toàn nhất**.
2. ✅ **Đã chốt cấu hình chính là S1** nên Figure 4 có cột $I_{SR}$ **thật**. Nếu muốn
   thêm hình cho Lần 1 (J1) ở phụ lục thì chạy được **ngay với grid 4 cột có sẵn** —
   cột $I_{SR}$ sẽ chỉ là placeholder vì J1 không có nhánh SR, không cần sửa code
   sang grid 3 cột như tài liệu này từng ghi nhầm.

> Danh sách "10 track tiêu biểu" **không tái lập chính xác khi đổi phần cứng** — thứ tự
> theo confidence lệch ở các track có confidence gần bằng nhau (khác biệt số thực
> GPU vs CPU). Nên chốt một bộ hình và giữ nguyên.

### ✅ Checklist Bước 3

- [x] `tools/visualize_paper_figures.py` — grid 4 cột, 5 đúng + 5 sai, `--pick extreme`
- [x] Verify chạy được (bản 2 track, tự tái lập đúng 805/999)
- [x] Chốt cấu hình cho Figure 4 = **S1** → giữ grid 4 cột
- [ ] 🖼️ **Sinh hình từ checkpoint multi-seed** — lệnh ở trên, chạy **S1** (bắt buộc) +
      **S4** (nên) trên `results/multi-seed/*/*_seed42_best.pth` (~5 phút GPU / ~20
      phút CPU mỗi cấu hình). Bộ hình 1-seed cũ đã gỡ khỏi repo, không dùng nữa.
- [ ] 🟠 **Chọn 10 hình cuối cho Figure 4** từ bộ hình mới sinh

---

## 🟠 Bước 4 — So sánh với SOTA ngoài repo (PARSeq / SVTR) ❌ CHƯA BẮT ĐẦU

> **Review yêu cầu**: chạy 1 mô hình SOTA OCR hiện đại trên cùng tập ICPR 2026 LRLPR
> để làm baseline so sánh mạnh.

### ✅ Checklist Bước 4

- [ ] Dựng repo + môi trường riêng (không đụng `crnn_and_stn`)
- [ ] Chốt cách so công bằng: PARSeq là **single-image**, ta là **multi-frame** →
      đề xuất cho PARSeq ăn **frame giữa**, ghi rõ là baseline single-frame
- [ ] Xuất danh sách 999 track val + nhãn để chấm trên **đúng cùng một tập**
- [ ] Chạy + đưa vào bảng so sánh

> Cắt được nếu thiếu thời gian — chỉ cần ghi vào phần Limitations.

---

# PHẦN C — TỔNG KẾT

## Bảng trạng thái theo đúng mục review

| Mục review | Trạng thái | GPU còn cần |
|---|---|---:|
| Nhóm 1 — công thức loss | 🟡 3/7 — đã xác định **4 lỗi**, chưa sửa vào paper | 0 |
| Nhóm 2 — quy trình deterministic | ✅ **4/4 XONG**, verify trên 9 run thật | 0 |
| Nhóm 3 — chống overfitting | ⏭️ **đã chốt không áp dụng** — code sẵn, đưa vào Future work | 0 |
| **Bước 1 — Multi-seed** | ✅ **9/9 XONG** — đủ 3 lần chạy có Mean ± Std | 0 |
| **Bước 2 — CER/NED/PSNR/SSIM** | ✅ **7/8 XONG** (3 chỗ cố ý lệch, có lý do) | 0 |
| **Bước 3 — hình định tính** | 🟡 4/6 — còn chọn 10 hình cuối | 0 |
| Bước 4 — PARSeq/SVTR | ❌ 0/4 — để cuối cùng, cắt được | riêng |

**Không còn việc nào cần GPU** trong phạm vi đã chốt.

## Việc còn lại theo thứ tự

1. ✍️ Sửa **4 lỗi công thức loss** trong paper (§Nhóm 1) — 0 GPU
2. ✍️ Đổi con số headline sang **Mean ± Std của Lần chạy 2 (S1): `79.95% ± 0.15`**,
   bỏ lối báo cáo best-of-run
3. 🚨 Viết lại phần _"đóng góp của nhánh SR"_ — **không** đổi phương pháp chính, mà
   đổi **lời giải thích vì sao nó hoạt động**
4. ✍️ Cập nhật **Limitations** (danh sách đầy đủ ngay dưới)
5. 🖼️ Chọn 10 hình cuối cho Figure 4
6. 📏 Benchmark GFLOPs/latency cho Lần 1 và Lần 3 (`tools/benchmark.py` chưa có dòng
   cho `sr-scale=1`) — phút/epoch thì **đã có số đo thật** từ cột `epoch_time_s`

## Limitations phải ghi vào paper

1. **Nhánh SR chưa chứng minh được đóng góp** — Lần 1 (bỏ hẳn SR) hoà Lần 2 với 1/3.76
   chi phí. Chênh nằm trong nhiễu → kết luận đúng là _"SR không giúp"_, **không phải**
   _"bỏ SR thì tốt hơn"_.
2. **Chưa tách được từng thành phần trong cụm cờ nền** (STN pool vs domain-match vs
   decode vs EMA) — câu hỏi mở quan trọng nhất còn lại.
3. **Ablation là tích luỹ**, không tách 1 biến; `T` cũng nhảy 16→32 giữa Lần 1 và Lần 2.
4. **3 ablation chỉ 1 seed** (phụ lục, nhãn _chưa xác nhận_): multi-frame vs
   single-frame, perceptual loss, $\lambda_{SR}=0.5$ — dữ liệu ở `backup/`.
5. **Không đo được PSNR/SSIM cho Lần 1** — bảng PSNR không phủ được cấu hình có điểm
   TB cao nhất (không có $I_{SR}$ để đo).
6. **PSNR/SSIM và hình định tính đo trên checkpoint 1 seed**, khác nguồn với bảng
   accuracy Mean ± Std.
7. **Chưa chạy test lần nào** — mọi số là validation Scenario-B (999 track);
   PSNR/SSIM đo trên **cặp synthetic**.
8. **Lần 1 dùng cờ nền của S1/S4**, nên 80.45% **không so được** với J1 lịch sử 76.88%.
9. Thiếu `log_j1p_seed42.txt` (CSV/submission/checkpoint vẫn đủ).
10. Biên nhiễu của tập val 999 track là **±13 track (±1.3 điểm)** — mọi chênh lệch nhỏ
    hơn ngưỡng này không kết luận được.
11. **Chưa khảo sát các biện pháp chống overfitting** (weight decay 1e-3, Dropout 0.3
    trước BiLSTM/FC, patience 12, MotionBlur/GaussNoise) — đã cài đặt sẵn trong code
    nhưng nằm ngoài phạm vi bài. Cả **9/9 run đều cho thấy overfit rõ** (val loss chạm
    đáy ep 11–22 rồi tăng 38–67%), nên đây là hướng cải thiện có cơ sở nhất còn lại.

## Sau cùng — chạy test (chưa làm)

```bash
## Phương án B — train lại toàn bộ data rồi predict test.
## ⚠️ --submission-mode TRAIN LẠI TỪ ĐẦU và TẮT early stopping, không phải chỉ inference.
## --epochs 28 = trung bình best epoch S1 qua 3 seed (28/32/24), tính từ
## results/multi-seed/s1_mf_sr_ocr/history_s1_seed*.csv
python train.py \
  --preset stable --experiment-name submission_s1_final \
  --epochs 28 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --submission-mode --num-workers 8 --aug-level full \
  2>&1 | tee results/log_submission_s1.txt
```

> ⚠️ Cờ trên đã đổi sang cấu hình **S1** (Lần chạy 2) theo quyết định chốt phương pháp.
> Chạy đủ 60 epoch ở chế độ này sẽ **lưu ra model của epoch cuối, đã overfit nặng**
> (S1 đỉnh val ở epoch 24–32 rồi tụt).
>
> ⚠️ Model ra từ lệnh này **khác** 3 checkpoint `s1_seed*_best.pth` đã có — train mới
> trên 20.000 track (thêm 5%, gồm cả val), không phải checkpoint cũ được inference lại.
>
> **Phương án an toàn hơn**: dùng lại checkpoint đã được validate, chỉ chạy inference
> trên test — số val và số test đến từ **cùng một model**. Hiện chưa có tool
> (`tools/eval_decode.py` chỉ chạy `mode="val"`), cần viết thêm ~1 giờ.
> Phân tích đánh đổi: [paper/paper_revision_plan.md §3](paper/paper_revision_plan.md).
