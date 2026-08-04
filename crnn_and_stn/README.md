# Bước 1 (Multi-Seed J1/S1/S4) + Bước 2 (Metrics) + Bước 3 (Qualitative Figures)

`feature/run-multi-seeda-and-evulate` → `main` · 55 commits · 101 files (+13,117/−156),
code 15 files (+2,689/−71)

> 🎯 **Đã chốt: S1 là phương pháp đề xuất chính** — `79.95% ± 0.15`. Ổn định nhất qua
> seed (std **0.15**, nhỏ hơn 3× so với J1/S4) và **bất biến với `cudnn.benchmark`**.
>
> 🚨 **Phải công bố kèm một kết quả âm tính**: cấu hình **không có SR (J1)** hoà điểm
> với S1 trong khi rẻ hơn **3.76× compute** → **nhánh SR chưa chứng minh được đóng góp
> đo được**. Đây là Limitation bắt buộc, **không** phải lý do đổi phương pháp chính
> (chênh nằm trong biên nhiễu, không kết luận được chiều nào).
>
> **Phạm vi**: chỉ baseline CRNN+STN và 3 cấu hình có error bar (J1/S1/S4). Các
> ablation 1-seed cũ đã chuyển sang `backup/` để tài liệu không bị nhiễu.

---

# PHẦN A — 3 nhóm giải pháp

## Nhóm 1 — Công thức loss đa nhiệm 🟡 KHÔNG cần train lại

Đã đối chiếu công thức trong review với code thật → tìm ra **4 chỗ lệch**. Cả 4 đều là
**sai lệch mô tả trong paper**, code vẫn đúng ⇒ **0 giờ GPU**, mọi số liệu giữ nguyên.

**Công thức code thật sự chạy:**

```
L_Total = L_CTC + λ_SR · ( ‖I_SR − Warp_sg[θ](I_HR)‖₁ + β·L_Edge + α·L_VGG )
λ_SR = 0.1        α = 0 (mọi cấu hình báo cáo)      β = 0
```

| #   | Paper viết                   | Code thật                                                                   | Nguồn                |
| --- | ---------------------------- | --------------------------------------------------------------------------- | -------------------- |
| 1   | `Warp_θ(I_SR)`               | **không warp** `I_SR` (đã nằm trong khung STN → warp 2 lần là sai)          | `trainer.py:263-272` |
| 2   | "Smooth L1"                  | **L1 thuần** (`F.l1_loss`)                                                  | `losses.py:84`       |
| 3   | `θ` bình thường              | **`sg[θ]` stop-gradient** (`theta_sel.detach()`) — chủ đích, không phải bug | `trainer.py:270`     |
| 4   | `λ_Perc·L_VGG` **song song** | perceptual **lồng trong** `λ_SR` — chỉ tương đương khi `λ_Perc = λ_SR × α`  | `trainer.py:336`     |

📌 **Đã soạn sẵn bản sửa** (LaTeX + Unicode + 3 câu chú thích bắt buộc) để dán thẳng
vào bản thảo: [`report/paper/loss_formula_corrected.md`](report/paper/loss_formula_corrected.md).
Quyết định: **giữ L1**, không đổi sang SmoothL1 (đổi code = chạy lại 39h GPU).

## Nhóm 2 — Quy trình deterministic ✅ XONG

`seed_everything` có `deterministic=True, benchmark=False`, bật bằng `--no-cudnn-benchmark`.
**Đầy đủ hơn review**: thêm `PYTHONHASHSEED` và `cuda.manual_seed`.

Verify **trên run thật, không chỉ smoke test**:

| Kiểm tra                                                 | Kết quả                                 |
| -------------------------------------------------------- | --------------------------------------- |
| Banner in `cudnn.benchmark: False (deterministic: True)` | ✅ 8/8 log (thiếu `log_j1p_seed42.txt`) |
| Seed khớp `--experiment-name`                            | ✅ 8/8                                  |
| CSV đủ **14 cột**                                        | ✅ **9/9**                              |
| `nan_batches = 0` mọi epoch                              | ✅ **424/424 epoch**                    |
| Val Acc khớp khi **chấm lại từ nhãn gốc**                | ✅ **9/9 run**                          |

## Nhóm 3 — Chống overfitting ⏭️ CHỐT KHÔNG ÁP DỤNG

**Cần train lại** (khác Nhóm 1) — dropout/weight-decay/augmentation/patience là tham số
lúc huấn luyện, không tác động gì lên 9 checkpoint đã có. Áp dụng = thí nghiệm mới
**~34h GPU** và **bắt buộc multi-seed lại cả 3 model** để bảng ablation nhất quán.

→ Chốt **hoãn, đưa vào Future work**. Code từng cài thử (`--pre-rnn-dropout`,
`--aug-level strong`) **đã revert** — repo về đúng trạng thái lúc chạy 9 run
(verify: params S1 vẫn `29,577,214`, `aug full` vẫn 10 phép, `WD/patience/dropout` =
`1e-4 / 18 / 0.25`).

📌 **Bằng chứng overfit nay là 9/9 run**: val loss chạm đáy rất sớm (**ep 11–22**) rồi
tăng **38–67%**, trong khi train loss tụt về ~0.01–0.04. **J1 overfit sớm và sâu nhất**
(đáy ep 11–16, train loss 0.0074) — hợp lý vì J1 không có nhánh SR đóng vai
**regularizer đa nhiệm**. Đây là lập luận ủng hộ việc **giữ** SR trong S1.

---

# PHẦN B — 4 bước phải làm ngay

## 🔴 Bước 1 — Multi-Seed cho J1, S1, S4 ✅ XONG

3 seed `42/100/2026`, deterministic. **9/9 run**, mọi số đã chấm lại từ nhãn gốc.

| Model                               | SR          | `T` |    GFLOPs |    **Mean ± Std** |         Std |      CER ↓ | Phút/epoch | Vai trò                      |
| ----------------------------------- | ----------- | --: | --------: | ----------------: | ----------: | ---------: | ---------: | ---------------------------- |
| **S1** — Joint MF-SR-OCR ×2         | ×2 MFSR+DCN |  32 |    109.08 | **79.95% ± 0.15** | **0.15** 🥇 |     0.0541 |       9.67 | 🎯 **đề xuất**               |
| **J1** — GroupNorm, **không SR**    | ❌          |  16 | **26.14** |     80.45% ± 0.45 |        0.45 | **0.0525** |   **2.60** | ablation _"bỏ hẳn SR"_       |
| **S4** — SR ×1, `T=32` qua backbone | ×1 MFSR+DCN |  32 |   chưa đo |     79.48% ± 0.44 |        0.44 |     0.0543 |       4.28 | ablation _"bỏ phóng to ảnh"_ |

**Kiểm định** (ngưỡng: chênh > 2× sai số hiệu):

| Cặp          |     Chênh | Sai số | Kết luận                 |
| ------------ | --------: | -----: | ------------------------ |
| J1 vs S1     |     +0.50 |  ±0.28 | ⚠️ trong nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** |  ±0.36 | ✅ **J1 tốt hơn thật**   |
| S1 vs S4     |     +0.47 |  ±0.27 | ⚠️ trong nhiễu → **hoà** |

### Vì sao chọn S1 dù J1 có điểm trung bình nhỉnh hơn

- **J1 không hơn S1 có ý nghĩa** (+0.50, trong nhiễu) → thống kê là **hoà**; chọn giữa
  hai cấu hình hoà nhau thì quyết định bằng **độ ổn định**.
- **S1 ổn định nhất**: std `0.15` (J1 `0.45`, S4 `0.44`); cùng seed 42 thì
  `benchmark=True` và `deterministic=True` cho **đúng cùng con số** (lệch **0 track**,
  trong khi S4 lệch **16**) → con số **tái lập được nhất** của cả project.
- S1 là kiến trúc hoàn chỉnh mà bài xây câu chuyện quanh nó; J1/S4 đóng đúng vai
  **ablation** của chính S1.

### Hai giả thuyết bị bác bỏ

1. **"SR đóng góp vào độ chính xác"** — J1 và S1 dùng **chung cụm cờ nền** (GroupNorm,
   STN pool `(4,8)`, domain-match, constrained decode, EMA); S1 chỉ thêm SR+DCN+MFSR.
   Thêm cụm đó **không cải thiện** mà tốn 3.76× compute. Phần cải thiện thật
   (**+3.57 điểm**, 76.88% → 80.45%) đến từ **cụm cờ nền**, với cùng kiến trúc không SR.
2. **"`T=32` là yếu tố chính"** — J1 chạy **`T=16`** mà vẫn ngang/hơn S1 và S4 (đều `T=32`).

### 2 lưu ý

- **J1 chạy cờ nền S1/S4, không phải cờ lịch sử** (STN pool `(4,8)`, params 29,442,700
  vs 29,313,452). Ablation 1-cụm-biến sạch so với S1 — tốt hơn về khoa học, nhưng
  **80.45% không so được với 76.88%** của J1 lịch sử.
- **`cudnn.benchmark` ảnh hưởng không đồng đều**: cùng seed 42, S1 lệch **0 track**
  nhưng S4 lệch **−16 track**. Không khái quát _"1-seed luôn thổi phồng"_ thành quy luật.

🆕 **`tools/aggregate_seeds.py --output-csv`** — đóng nốt yêu cầu _"ghi nhận bảng kết
quả 3 seeds vào CSV log"_. Xuất **13 cột** (`label, n_seeds, accs, mean, std, min, max,
ci95, ci_low, ci_high, delta_vs_baseline, delta_stderr, verdict`), `--append` để gom cả
3 model vào 1 file.

📄 [`report/baseline1_crnn_stn/multi_seed_results.md`](report/baseline1_crnn_stn/multi_seed_results.md)

## 🔴 Bước 2 — CER, NED, PSNR/SSIM ✅ XONG

**CER + NED** — cài trong `Trainer.validate()`, in console mỗi epoch + 2 cột
`val_cer`/`val_ned` trong `history_*.csv`. **Có Mean ± Std cho cả 3 model** (bảng §Bước 1).

📌 J1 tốt nhất cả exact match lẫn CER, và là model duy nhất đạt **0/999 sai độ dài ở cả
3 seed**. Phát hiện chỉ multi-seed mới thấy: **S4 ổn định hơn 16× về CER** (std 0.0001)
nhưng **kém ổn định hơn 3× về exact match** — hai đại lượng **không đi cùng chiều**.

**PSNR/SSIM** — tool mới `tools/eval_sr_quality.py` (`skimage.metrics`), chạy hậu kỳ.
Con số đáng đọc là cột **Chênh** (so với `base` = ảnh chưa qua SR):

| Cấu hình    | Chênh PSNR | Chênh SSIM |
| ----------- | ---------: | ---------: |
| **S1** (×2) | +0.7166 dB |    +0.0434 |
| **S4** (×1) | +1.0372 dB |    +0.0756 |
| **J1**      |          — |          — |

✅ **Đo trên checkpoint multi-seed seed 42** (`results/multi-seed/`, chạy 2026-08-05)
— cùng nguồn với bảng accuracy Mean ± Std, không còn dùng số của checkpoint 1-seed cũ.
Val acc in ra lúc chạy (S1 797/999, S4 789/999) khớp chính xác `multi_seed_results.md`,
xác nhận đúng checkpoint.

> 🚨 **J1 vẫn không chạy được ở tool này** (khác với Bước 3) — PSNR/SSIM đo `I_SR` với
> `I_HR`, mà **J1 không có nhánh SR nên `I_SR` không tồn tại** — giới hạn cấu trúc,
> không phải thiếu sót. `eval_sr_quality.py:131` chặn thẳng bằng `raise SystemExit`.
> Hệ quả: bảng PSNR **không phủ được cấu hình có điểm trung bình cao nhất**.
>
> ⏳ Bảng tương quan "PSNR nghịch với OCR" ở mức từng track (bên dưới) vẫn tính từ CSV
> cũ — cần CSV mới (`sr_quality_s1_seed42.csv`, `sr_quality_s4_seed42.csv`) để cập nhật.
>
> ⚠️ **Không đặt PSNR tuyệt đối của S4 chung cột với S1** — S1 xuất 64×256, S4 xuất
> 32×128, hai thang khác nhau.

### 🔬 PSNR **nghịch** với khả năng đọc biển số

| Mức                        | Bằng chứng                                                                | Chiều      |
| -------------------------- | ------------------------------------------------------------------------- | ---------- |
| **Track** (n=999/cấu hình) | track đọc **sai** có PSNR **cao hơn ~2 dB**; r = −0.362 (S1), −0.400 (S4) | **nghịch** |
| **Kiến trúc**              | **J1 bỏ hẳn SR → không tái tạo ảnh gì cả → OCR cao nhất**                 | **nghịch** |

→ Khi reviewer hỏi _"sao không tối ưu theo PSNR"_: **"PSNR nghịch với OCR"** — chứ
không chỉ "PSNR không phản ánh OCR".

📄 [`report/buoc2_metrics.md`](report/buoc2_metrics.md)

## 🟠 Bước 3 — Trực quan hoá định tính 🟡 SCRIPT XONG, CẦN CHẠY LẠI

Script mới `tools/visualize_paper_figures.py` — grid **4 cột**
`I_LR → I_SR → Attention heatmap → Prediction`, **5 success + 5 failure**.
Mặc định `--pick extreme`: lấy case đúng **tự tin nhất** và case sai **mơ hồ nhất**
(hai đầu phân phối) thay vì mấy track đầu danh sách.

> ⚠️ **Hình chưa có trên đĩa.** Bộ hình cũ sinh từ checkpoint **1-seed** (02/08), tức
> **trước** khi có checkpoint multi-seed (03–04/08) → hình và số đến từ **2 model khác
> nhau**. Đã gỡ khỏi repo, cần sinh lại từ checkpoint multi-seed.

### 📌 Chạy trên cấu hình nào?

| Cấu hình |     Có chạy không      | Lý do                                                                                                                             |
| -------- | :--------------------: | --------------------------------------------------------------------------------------------------------------------------------- |
| **S1**   |    ✅ **bắt buộc**     | Cấu hình chính → Figure 4 của paper. Có `I_SR` nên giữ được grid **4 cột**                                                        |
| **S4**   |       🟡 **nên**       | Để so **cùng một track qua 2 kiến trúc** — hữu ích cho hình ablation ở phụ lục                                                    |
| **J1**   |    🟡 **chạy được**    | Không có nhánh SR, nhưng tool **tự xử lý gracefully** — cột `I_SR` hiện placeholder `"(khong co SR)"` thay vì crash, không cần sửa code. Chỉ hữu ích cho phụ lục |

> 📌 **Đính chính**: trước đây ghi sai là "J1 không chạy được, phải sửa tool sang grid
> 3 cột". Đọc lại code `tools/visualize_paper_figures.py::frame_column` mới phát hiện
> tool đã tự chặn `frames is None` và vẽ placeholder — **không** crash, **không** cần
> sửa gì. Cái thật sự chặn J1 là `tools/eval_sr_quality.py` (PSNR/SSIM,
> `raise SystemExit` ở dòng 131) — mục đó vẫn đúng như đã ghi.

### Lệnh chạy

Inference thuần, **0 GPU-hour đáng kể** — GPU ~5 phút, CPU ~20 phút mỗi cấu hình.

```bash
# BẮT BUỘC — S1, cấu hình chính cho Figure 4
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --decode constrained --pick extreme \
  --output-dir results/multi-seed/s1_mf_sr_ocr/paper_figures

# NÊN — S4, để so cùng track qua 2 kiến trúc (BẮT BUỘC --width-downsample 4)
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/s4_sr_scale1/s4_seed42_best.pth \
  --width-downsample 4 --decode constrained --pick extreme \
  --output-dir results/multi-seed/s4_sr_scale1/paper_figures

# TUỲ CHỌN — J1, cho phụ lục. Cột I_SR sẽ là placeholder, không phải lỗi
python tools/visualize_paper_figures.py \
  --checkpoint results/multi-seed/crnn_resblock_groupnorm_nosr_j1/j1p_seed42_best.pth \
  --decode constrained --pick extreme \
  --output-dir results/multi-seed/crnn_resblock_groupnorm_nosr_j1/paper_figures
```

Mỗi lệnh xuất `figure4_qualitative_grid.png` + 10 ảnh track riêng
(5 đúng `NN_ok_*`, 5 sai `NN_err_*`).

> 🚨 `*.pth` **không đi theo `git pull`** (113 MB/file) — phải upload checkpoint trước.
> Gặp `❌ Không tìm thấy checkpoint` là do thiếu file, không phải lỗi lệnh.
>
> ⚠️ Danh sách "10 track tiêu biểu" **không tái lập chính xác khi đổi phần cứng** — thứ
> tự theo confidence lệch ở các track gần bằng nhau (số thực GPU vs CPU).
> **Chốt một bộ hình rồi giữ nguyên**, đừng sinh đi sinh lại.
>
> 📌 Với **S1** thì chênh 1-seed ↔ multi-seed **bằng 0** (seed 42 ra đúng 797/999 ở cả
> hai chế độ cudnn) — nên hình S1 khớp số trong bảng, không cần ghi chú caption.

## 🟠 Bước 4 — PARSeq / SVTR ❌ CHƯA BẮT ĐẦU

Cần repo + môi trường riêng. Review cho phép cắt nếu thiếu thời gian — chỉ cần ghi vào
Limitations. Khi làm: PARSeq là **single-image**, ta là **multi-frame** → cho PARSeq ăn
**frame giữa**, ghi rõ là baseline single-frame, chấm trên **đúng 999 track val**.

---

# PHẦN C — Code & tổ chức

## Thay đổi code (15 file, +2,689/−71)

| File                                                                         |                                                                                                                            |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `tools/eval_sr_quality.py`                                                   | **mới (+282)** — PSNR/SSIM bằng `skimage.metrics`, kiến trúc suy ngược từ `state_dict`                                     |
| `tools/visualize_paper_figures.py`                                           | **mới (+291)** — Bước 3: grid 4 cột, `--pick extreme`                                                                      |
| `tools/aggregate_seeds.py`                                                   | **+184** — Mean ± Std + CI 95% + kiểm định cặp; 🆕 `--output-csv` / `--append`                                             |
| `tools/plot_results.py` · `benchmark.py` · `eval_decode.py` · `visualize.py` | **+1,021** — chart, benchmark compute, chấm lại checkpoint, ảnh debug                                                      |
| `src/training/trainer.py`                                                    | **+98** — CER/NED trong `validate()`, đo thời gian mỗi epoch, 5 cột CSV mới, `build_optimizer_param_groups()`              |
| `src/utils/postprocess.py`                                                   | **+407** — constrained decode (layout `LLLNLNN,LLLNNNN`, beam 16), `edit_distance`                                         |
| `src/data/dataset.py` · `transforms.py`                                      | **+86** — nhánh `sr_eval_mode`: augment **tất định** để PSNR/SSIM tái lập được                                             |
| `src/utils/common.py` · `train.py` · `configs/config.py`                     | **+28** — `seed_everything` đầy đủ, cờ `--weight-decay` / `--wd-skip-bias-norm`, banner in seed + trạng thái deterministic |

> `--wd-skip-bias-norm` **mặc định TẮT** có chủ đích — bật mặc định sẽ khiến multi-seed
> khác các run trước ở **hai** biến cùng lúc, không biết số đổi do biến nào.

## Tổ chức lại thư mục

`report/` và `results/` **chỉ còn** baseline CRNN+STN và J1/S1/S4; mọi thứ khác sang `backup/`:

| Nội dung                                | Ở đâu                            |
| --------------------------------------- | -------------------------------- |
| Multi-seed J1/S1/S4 (9 run)             | `results/multi-seed/<cấu hình>/` |
| Tài liệu baseline + J1/S1/S4            | `report/`                        |
| Artefact + tài liệu các ablation 1-seed | `backup/` · `backup/report/`     |

Đã dọn **462 tham chiếu** tới các cấu hình ngoài phạm vi → **0**, và verify **toàn bộ
link nội bộ** (88/88 trong `report/`, 34/34 trong `backup/report/`) đều trỏ đúng file
có thật. Tài liệu J1 tách riêng thành
[`j1_groupnorm_nosr.md`](report/baseline1_crnn_stn/j1_groupnorm_nosr.md).

## Verify

```bash
# Kiểm định: S1 (đề xuất) vs J1 (ablation bỏ hẳn SR) — kỳ vọng "hoà"
python tools/aggregate_seeds.py \
  --acc 79.7798 80.0801 79.9800 --label "S1 (de xuat)" \
  --baseline-acc 79.9800 80.8809 80.4805 --baseline-label "J1 (khong SR)"

# CSV phải đủ 14 cột
head -1 results/multi-seed/s1_mf_sr_ocr/history_s1_seed42.csv
```

Đã tự kiểm: 9/9 run đúng chế độ deterministic + đúng seed · `nan_batches = 0` mọi epoch ·
CSV đủ 14 cột · CER in console · **Val Acc khớp chính xác khi chấm lại**
`submission_*.txt` với `plate_text` gốc (không chỉ tin log).

## Ba chỗ cố ý lệch review

1. **Không dùng `editdistance`** — `postprocess.py` đã có sẵn `edit_distance`
   (Levenshtein có cache). Kết quả tương đương, không thêm dependency.
2. **Không đo PSNR/SSIM trên Scenario-A** — val **không có track Scenario-A nào**
   (0/999), và cả 10.000 track Scenario-A **nằm trong tập TRAIN** → đo ở đó là đo trên
   dữ liệu đã học, **không hợp lệ**. Đã đo trên 999 track Scenario-B.
3. **PSNR/SSIM là tool hậu kỳ**, không nằm trong log val mỗi epoch (nhét vào training
   loop sẽ buộc multi-seed chạy lại). ⚠️ **CER/NED thì không lệch** — có đủ trong
   console + CSV đúng như review yêu cầu.

---

## Limitations

1. **Nhánh SR chưa chứng minh được đóng góp** — J1 hoà điểm với S1 trong khi rẻ hơn
   **3.76× compute**. Chênh nằm **trong** biên nhiễu → kết luận đúng là _"SR không cho
   thấy lợi ích đo được"_, **không phải** _"bỏ SR thì tốt hơn"_. S1 vẫn được chọn vì
   **ổn định nhất**, nhưng đánh đổi này phải nêu rõ.
2. **Chưa tách được từng thành phần trong cụm cờ nền** (STN pool vs domain-match vs
   decode vs EMA) — câu hỏi mở quan trọng nhất còn lại.
3. Ablation là **tích luỹ**, không tách 1 biến; `T` cũng nhảy 16→32 giữa J1 và S1.
4. **Các biến thể khác của nhánh SR chỉ có 1 seed** (trọng số λ_SR, perceptual loss,
   single-frame vs multi-frame) — dữ liệu ở `backup/`, dùng cho phụ lục.
5. **Không đo được PSNR/SSIM cho J1** — bảng PSNR không phủ được cấu hình có điểm TB
   cao nhất (không có `I_SR` để đo).
6. **PSNR/SSIM đo trên checkpoint 1 seed**, khác nguồn với bảng accuracy Mean ± Std.
7. **Chưa chạy test lần nào** — mọi số là validation Scenario-B (999 track);
   PSNR/SSIM đo trên **cặp synthetic**.
8. **Hình Bước 3 chưa sinh lại** từ checkpoint multi-seed cho S1/S4 (bắt buộc/nên).
   J1 **chạy được** nhưng cột `I_SR` chỉ là placeholder — không có nhánh SR để vẽ.
9. Thiếu `log_j1p_seed42.txt` (CSV/submission/checkpoint vẫn đủ).
10. **Biên nhiễu ±13 track (±1.3 điểm)** trên val 999 track — chênh nhỏ hơn ngưỡng này
    không kết luận được.
11. **Chưa khảo sát chống overfitting** — 9/9 run đều overfit rõ; hướng cải thiện có cơ
    sở nhất còn lại.

---

## Việc còn lại (không thuộc PR này)

> ⚠️ **3 mục ✍️ dưới đây phải sửa trong bản thảo bài báo (Overleaf/Word) — không nằm
> trong repo này.** `feature/run-multi-seeda-and-evulate` không chứa file `.tex`/`.docx`
> nào, nên không thể "dán" hộ được. Nội dung cần dán **đã soạn sẵn**, chỉ việc copy tay:
>
> - Công thức đã sửa (LaTeX + Unicode): [`loss_formula_corrected.md`](report/paper/loss_formula_corrected.md)
> - Headline `79.95% ± 0.15` — **đã áp dụng khắp `report/*.md`** trong repo; chỉ còn
>   thiếu ở bản thảo ngoài repo
> - Lập luận "đóng góp của SR" — đã viết đủ ở §Bước 1 "Hai giả thuyết bị bác bỏ" và
>   `j1_groupnorm_nosr.md`, chỉ cần diễn giải lại theo văn phong bài báo

- [ ] 🖼️ **Sinh lại hình Bước 3** cho **S1** (bắt buộc) + **S4** (nên) — lệnh ở §Bước 3
- [ ] ✍️ Dán khối công thức đã sửa vào bản thảo + 3 câu chú thích bắt buộc
- [ ] ✍️ Đổi con số headline sang **Mean ± Std của S1** (`79.95% ± 0.15`) trong bản thảo,
      bỏ lối báo cáo best-of-run
- [ ] 🚨 Viết lại phần _"đóng góp của nhánh SR"_ trong bản thảo — **không** đổi phương
      pháp chính (vẫn là S1), mà đổi **lời giải thích vì sao nó hoạt động**
- [ ] 📏 **Benchmark GFLOPs/latency cho S4** — `tools/benchmark.py` thiếu cờ `--sr-scale`
      và `--width-downsample`. _(Phút/epoch thì đã có **số đo thật** từ cột
      `epoch_time_s` của 424 epoch, không cần chạy lại.)_
- [ ] Chạy test — lệnh sẵn dùng (`--epochs 28`, trung bình best epoch 3 seed S1) ở
      [`paper_revision_plan.md §3`](report/paper/paper_revision_plan.md);
      đọc trước vì `--submission-mode` **train lại từ đầu và tắt early stopping**
- [ ] Nhóm 3 — chống overfitting (Future work) · Bước 4 — PARSeq/SVTR
