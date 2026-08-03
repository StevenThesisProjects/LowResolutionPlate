# Plan triển khai review (paper revision)

> Lập sau khi đối chiếu từng yêu cầu của review với code thật trong `src/`,
> `configs/`, `tools/` và log của S1/S4. Mục 0 là phần **phải đọc trước** — có 6
> chỗ review giả định khác hiện trạng, làm theo nguyên văn sẽ tốn GPU vô ích
> hoặc tạo lỗi mới.
> Bối cảnh kết quả hiện có: [model_comparison_summary.md](../baseline1_crnn_stn/model_comparison_summary.md).

---

## 0. Sáu điểm review giả định khác hiện trạng code

| # | Review nói | Thực tế trong code | Hành động |
|---|---|---|---|
| 1 | Đặt `λ_Perceptual = 0.01` (cấu hình mới) | **S3 đã chạy đúng cấu hình này rồi** — xem chứng minh bên dưới | Không cần chạy lại. Chỉ cần sửa cách trình bày công thức trong paper |
| 2 | Viết hàm `seed_everything` với `deterministic=True` | `src/utils/common.py` **đã có sẵn** đúng hàm này, có sẵn cả nhánh deterministic | Chỉ cần thêm flag `--no-cudnn-benchmark` khi chạy |
| 3 | Dùng thư viện `editdistance` để tính CER/NED | `src/utils/postprocess.py` **đã implement sẵn** `edit_distance`, `character_error_rate`, `batch_*` (Levenshtein có cache) | Không cài thư viện mới. Chỉ cần gọi chúng trong `validate()` (hiện gọi 0 lần) |
| 4 | Đo PSNR/SSIM **trên Scenario-A val tracks** | **Val set 100% là Scenario-B — 0 track Scenario-A** (đã đếm: 0 vs 999) | Phải chọn lại tập đo, xem P2 |
| 5 | Tăng `weight_decay` 1e-4 → 1e-3 | **Param-grouping không có trong code hiện tại** → wd đang áp lên cả bias và GroupNorm | **Sửa param-grouping TRƯỚC**, nếu không là khuếch đại anti-pattern lên 10 lần |
| 6 | `L_SR = ‖Warp_θ(I_SR) − Warp_θ(I_HR)‖₁`, "Smooth L1" | Code chỉ warp **HR target** (I_SR đã nằm trong khung đã nắn vì SR chạy *sau* STN); và dùng `F.l1_loss` thuần, không phải Smooth L1 | Sửa công thức trong paper (warp kép là sai), và chốt L1 hay SmoothL1 |

### Chứng minh điểm 1 — S3 chính là cấu hình review đề xuất

Review viết: `L_Total = L_CTC + λ_SR·L_SR + λ_Perceptual·L_VGG`, với `λ_SR=0.1`, `λ_Perceptual=0.01`.

Code gộp perceptual **vào bên trong** `L_SR` (`SRPixelLoss.forward`), rồi mới nhân `λ_SR` (`Trainer.train_one_epoch`):

```
L_total_code = L_CTC + λ_SR · (L1 + α·L_VGG)
             = L_CTC + 0.1·L1 + (0.1 × 0.1)·L_VGG      # S3: λ_SR=0.1, α=0.1
             = L_CTC + 0.1·L1 + 0.01·L_VGG             # ≡ đúng công thức review
```

→ **S3 (`--sr-perceptual-weight 0.1`) đã cho `λ_Perceptual` hiệu dụng = 0.01**, khớp
chính xác đề xuất của review. Việc cần làm chỉ là **trình bày lại** trong paper cho
khớp, hoặc refactor code tách `λ_Perceptual` thành tham số CLI độc lập để người đọc
không phải tự nhân 2 số. Khuyến nghị: refactor (rẻ, ~30 phút) để công thức trong
paper và flag CLI trùng khớp 1-1, tránh reviewer sau lại thắc mắc đúng chỗ này.

---

## 1. Thời gian chạy mỗi epoch (trả lời câu hỏi trực tiếp)

Đo từ tqdm trong log thật, 1188 step/epoch (19.001 track × 2 sample / batch 32 / accum 2):

| Cấu hình | Phút/epoch (train) | Epoch đã chạy | Tổng train | Ghi chú |
|---|---:|---:|---:|---|
| **S1** (sr-scale 2, T=32) | **9.21** | 55 | **8.45 h** | epoch đầu 9m28s (warm-up), sau đó rất ổn định 9m10s–9m19s |
| **S4** (sr-scale 1 + width/4, T=32) | **3.94** | 58 | **3.81 h** | epoch đầu 4m03s, sau đó 3m55s–3m59s |
| S2 | — | 56 | — | không có `log_s2.txt` (không chạy `tee`) |
| S3 | — | 43 | — | không có `log_s3.txt` (không chạy `tee`) |

> **Đây là một kết quả đáng đưa vào paper, không chỉ là thông tin vận hành.**
> S4 nhanh hơn S1 **2.34 lần mỗi epoch** (3.94 vs 9.21 phút) trong khi đạt **đúng
> cùng độ chính xác** (805/999 cả hai). Giả thuyết "S4 rẻ hơn" nêu trong
> [s4_sr_scale1_mf_sr_ocr.md §3](../baseline1_crnn_stn/s4_sr_scale1_mf_sr_ocr.md) nay
> đã có số đo thực nghiệm xác nhận, chứ không còn là suy luận định tính.
> Vẫn nên chạy `tools/benchmark.py` để có GFLOPs/latency inference chuẩn cho bảng.

**Bài học vận hành**: từ nay mọi run **bắt buộc** `2>&1 | tee results/log_<tên>.txt`.
S2 và S3 mất vĩnh viễn dữ liệu timing chỉ vì thiếu `tee`.

### Dự toán GPU cho multi-seed

Chế độ deterministic (`cudnn.deterministic=True, benchmark=False`) thường **chậm hơn
10–40%** với mạng conv. Lấy +25% làm ước lượng thận trọng.

> ⚠️ Bảng gốc dưới đây ước lượng cho S1+S3+S4 (kế hoạch ban đầu). **Phạm vi cuối
> cùng (2026-08-03) là J1+S1+S4** — xem bảng cập nhật ngay sau.

| Cấu hình | 1 run (đã +25% det.) | × 3 seed | Ghi chú |
|---|---:|---:|---|
| S1 | ~11 h | **~33 h** | phương pháp đề xuất của paper — bắt buộc |
| S4 | ~5.3 h | **~16 h** | rẻ nhất, đang đồng hạng nhất — bắt buộc, **đã xong** |
| ~~S3~~ | ~12–15 h (ước lượng) | ~36–45 h | ⏭️ loại — có thêm VGG16 forward mỗi batch, chi phí không đáng so với giá trị thông tin thêm |

**Bảng cập nhật — phạm vi đã chốt**:

| Cấu hình | × 3 seed | Trạng thái |
|---|---:|---|
| S4 | ~12 h thật | ✅ xong — **79.48% ± 0.44** |
| S1 | ~22 h thật | ✅ xong — **79.95% ± 0.15** |
| J1 | ~10 h | 🔴 **run cuối cùng** (cờ lịch sử `--stn-pool 1,1`) |
| ~~J2~~ | ~27 h | ⏭️ **đã bỏ** — giữ 1 seed 77.18% cho phụ lục |

**Tổng còn lại: ~10 h GPU — đúng 1 run (J1).** Đã trừ S1 + S4 xong.

> ✅ Ước lượng "+25% cho deterministic" là **quá thận trọng** — số thật chỉ chậm
> hơn **2–4%** (S1: 9.39 vs 9.21 phút/epoch; S4: 4.09 vs 3.94). Dùng con số này
> để ước tính J1 thay vì +25%.

---

## 2. Kế hoạch theo phase

### P0 — Chặn đường: phải xong trước khi tốn 1 giờ GPU nào — ✅ **PHẦN CODE ĐÃ XONG (2026-08-01)**

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| **P0.1** Param-grouping cho AdamW | ✅ xong | Thêm `build_optimizer_param_groups()` + flag `--wd-skip-bias-norm`, **mặc định TẮT** (xem cảnh báo bên dưới) |
| **P0.2** CER + NED vào `validate()` và CSV | ✅ xong | 2 cột `val_cer`, `val_ned`; in ra console mỗi epoch |
| **P0.2b** Thời gian mỗi epoch vào CSV | ✅ xong | 3 cột `train_time_s`, `val_time_s`, `epoch_time_s` |
| **P0.3** Tách `--lambda-perceptual` | ⏭️ bỏ qua | Chỉ là đổi cách viết, không đổi kết quả. Giữ `--sr-perceptual-weight`, ghi rõ quy đổi `λ_Perceptual = λ_SR × α` trong paper |
| **P0.4** Chốt L1 hay Smooth L1 | 📝 quyết định: **giữ L1** | Đổi sang SmoothL1 sẽ khiến toàn bộ S1–S4 phải chạy lại. Sửa chữ trong paper thay vì sửa code |
| **P0.5** Sửa công thức `L_SR` trong paper | 📝 việc phía paper | Bỏ `Warp_θ` khỏi `I_SR`, và ghi "L1" thay vì "Smooth L1" |

**Thay đổi code cụ thể:**
- `src/training/trainer.py`: thêm `build_optimizer_param_groups()`; đo thời gian train/val
  mỗi epoch; tính CER + NED trong `validate()`; ghi 5 cột mới vào `history_*.csv`;
  in CER/NED/Time ra console.
- `configs/config.py`: thêm `WEIGHT_DECAY_SKIP_BIAS_NORM = False`.
- `train.py`: thêm `--weight-decay`, `--wd-skip-bias-norm`; in weight-decay + seed +
  trạng thái deterministic vào banner (để log luôn ghi lại đủ cấu hình).

> ⚠️ **Vì sao `--wd-skip-bias-norm` mặc định TẮT.** Param-grouping làm **đổi hành vi
> huấn luyện**. Nếu bật mặc định, multi-seed sẽ khác S1–S4 ở **hai** biến cùng lúc
> (deterministic **và** cách áp weight decay) — khi con số ra khác 79.78% sẽ không biết
> do biến nào. Để mặc định tắt thì P1 chỉ đổi đúng một biến (deterministic), giữ đúng
> kỷ luật "mỗi lần một biến" của dự án. **Chỉ bật flag này ở P4**, cùng lúc tăng
> `--weight-decay 1e-3`.

**Bốn loại file report mỗi run sinh ra** (đã kiểm tra lại trong `trainer.py`):

| File | Sinh ra khi nào | Ghi chú |
|---|---|---|
| `results/history_<exp>.csv` | mỗi epoch | nay có đủ CER/NED/thời gian |
| `results/<exp>_best.pth` | mỗi khi val acc cải thiện | định dạng **không đổi** — `eval_decode.py`/`visualize.py` vẫn nạp được |
| `results/submission_<exp>.txt` | cùng lúc với `.pth` | dự đoán trên **validation** (999 dòng), không phải bài nộp |
| `results/log_<exp>.txt` | **chỉ khi có `2>&1 \| tee`** | ⚠️ khâu duy nhất còn phụ thuộc thao tác tay — S2/S3 mất log vì quên |

### P1 — Multi-seed verification (nội dung chính của review) — 🟡 còn ~10 h GPU

Đây là việc quan trọng nhất: nó xác nhận (hoặc bác bỏ) chính con số headline của paper.

#### Chọn cấu hình nào để multi-seed?

> **🔄 Cập nhật 2026-08-02 — quyết định cuối cùng đã đổi.** Phần dưới đây (khuyến
> nghị ban đầu: S1+S3+S4) được giữ lại làm lịch sử quyết định, nhưng **không còn là
> kế hoạch đang thực hiện**. Phạm vi chốt cuối là **J1 + S1 + S4** — bỏ S3 và J2,
> thêm J1 (chạy lại J1 với đúng bộ cờ nền của S1/S4). Lý do đổi: J1/J2 lịch
> sử không so 1-biến được với S1/S4 (lệch 5-6 tham số), nên không thể trả lời câu
> hỏi "T confound" mà S4 đặt ra ([s4_sr_scale1_mf_sr_ocr.md §3](../baseline1_crnn_stn/s4_sr_scale1_mf_sr_ocr.md#3-trả-lời-câu-hỏi-t-confound--kết-quả-chính-của-s4))
> bằng số có error bar; multi-seed thêm S3 (vốn đã đồng hạng 1-seed với S4) được
> đánh giá là giá trị thông tin thấp hơn so với việc đóng lập luận T-confound. Lệnh
> chạy + checklist mới nhất: [../training_runs/run_gpu.md §0 B4](../training_runs/run_gpu.md#-b4--multi-seed-j1-8-h-s1-26-h-j2-26-h--ladder-sạch),
> [../checklist_review.md](../checklist_review.md).

**Khuyến nghị ban đầu (lịch sử): S1 + S3 + S4** (không phải S1 + S2 + S4). Lý do
quan trọng nhất nằm ngay trong công thức của review:

```
L_Total = L_CTC + λ_SR·L_SR + λ_Perceptual·L_VGG      (λ_Perceptual = 0.01)
```

Công thức này **có số hạng perceptual** — tức là theo cách hiểu của reviewer,
**phương pháp đề xuất của paper đã bao gồm perceptual loss**, và đó chính là S3
(xem chứng minh ở mục 0). Multi-seed mà bỏ S3 nghĩa là không có error bar cho
đúng cấu hình mà reviewer đang mô tả là phương pháp chính — **đây là đánh đổi đã
chấp nhận** ở quyết định cuối cùng: công thức trong paper vẫn giữ nguyên (không
sửa), nhưng số hạng `λ_Perceptual` phải ghi rõ là ablation 1-seed chưa xác nhận.

| Cấu hình | Nên multi-seed? | Lý do |
|---|:---:|---|
| **S1** (λ_SR=0.1) | ✅ bắt buộc | Phương pháp đề xuất hiện tại của paper, là con số headline |
| **S3** (+perceptual) | ⏭️ **đã loại (2026-08-02)** | Chính là công thức review mô tả, nhưng đồng hạng 1-seed với S4 — giữ 1 seed, chuyển ngân sách sang J1 |
| **S4** (SR ×1) | ✅ bắt buộc | Đồng hạng cao nhất + rẻ hơn 2.34× — đóng góp riêng về hiệu quả — **đã xong** |
| **J1** (không SR) | ✅ **bổ sung** | Mốc nền T=16 của ladder — chạy **cờ lịch sử** (`--stn-pool 1,1`), tái lập được ~76.9% |
| ~~J2~~ (SR single-frame) | ⏭️ **đã loại (2026-08-03)** | Tốn ~27h. Ablation multi-frame vs single-frame giữ ở mức 1 seed (J2 77.18% vs S1 79.78%) + ghi Limitations |
| S2 (λ_SR=0.5) | ❌ loại | Kết quả âm tính đã rõ ở 1 seed (794, thấp nhất nhóm S). Không ai sẽ dùng cấu hình này |

**Khi nào S2/S3 mới đáng multi-seed sau này**: chỉ khi paper cần bảng ablation
riêng cho `λ_SR` (S2) hoặc muốn tách rạch ròi lợi ích của riêng perceptual loss
(S3) khỏi cụm S1/S4 đã có error bar. Ở phạm vi hiện tại, 1 seed kèm nhãn "chưa xác
nhận" là đủ cho cả hai.

**Lệnh** — lưu ý: `train.py` trơn **không** tái lập S1/S4/J1 (mặc định là
nhánh ResBlock cũ), phải ghi đủ flag. Thêm `--no-cudnn-benchmark` để bật
deterministic. S1 và S4 đã xong; chỉ còn J1 (~10h). Bản đầy đủ:
[../training_runs/run_gpu.md §0 B4](../training_runs/run_gpu.md#-b4--multi-seed-j1-8-h-s1-26-h-j2-26-h--ladder-sạch).

```bash
for SEED in 42 100 2026; do
  # J1 — mốc nền: GroupNorm, KHÔNG SR. Cờ lịch sử: ép --stn-pool 1,1
  # (mặc định nay là 4,8); không domain-match/EMA/constrained (mặc định đã đúng).
  python train.py --preset stable --experiment-name j1_seed${SEED} --seed ${SEED} \
    --backbone-norm group --stn-pool 1,1 \
    --no-cudnn-benchmark --num-workers 8 --aug-level full \
    2>&1 | tee results/log_j1_seed${SEED}.txt

done
```

> ✅ **J1 tái lập được ~76.88%** vì không dùng SR — mọi thay đổi code từ đó tới nay
> đều nằm ở nhánh SR. Nếu ra lệch hơn ±1.3 điểm thì kiểm tra lại cờ.
>
> 🚨 Viết `--stn-pool 1,1` **có dấu phẩy**; `1 1` sẽ parse thành `(11,)` và crash.

> **Trạng thái**: S1 (79.95% ± 0.15) và S4 (79.48% ± 0.44) **đã xong**, không cần
> chạy lại. Chỉ còn **J1** — lệnh ở
> [../training_runs/run_gpu.md §0 B4](../training_runs/run_gpu.md).

Tổng hợp — **`tools/aggregate_seeds.py` đã có sẵn**, tính Mean ± Std + CI 95% + kiểm
định so sánh 2 cấu hình, không cần viết mới:

```bash
python tools/aggregate_seeds.py --from-logs results/log_s1_seed*.txt --label "S1"
python tools/aggregate_seeds.py \
  --acc <3 số S4> --label "S4 (SR ×1)" \
  --baseline-acc <3 số S1> --baseline-label "S1 (SR ×2)"
```

**Quyết định cần chốt trước khi chạy** — có nên tái dùng run seed 42 đã có không?

- Các run S1–S4 hiện tại chạy với `cudnn.benchmark=True` (**không** deterministic).
- Trộn 1 run non-deterministic với 2 run deterministic vào cùng một bảng Mean ± Std
  là trộn 2 chế độ phương sai khác nhau — không nên.
- **Khuyến nghị: chạy lại cả 3 seed ở chế độ deterministic**, coi kết quả cũ là
  exploratory. Đây cũng đúng tinh thần review yêu cầu.

> ⚠️ **Rủi ro phải chấp nhận trước**: chạy lại seed 42 ở chế độ deterministic **có
> thể ra số khác 79.78% / 80.58%**. Dự án đã tự đo được `cudnn.benchmark=True` gây
> lệch tới 6.5 điểm giữa 2 lần chạy cùng seed. Nghĩa là **con số headline của paper
> có thể đổi**. Cần biết điều này trước, và chuẩn bị tinh thần báo cáo
> Mean ± Std thay vì một con số best-of-run.

### P2 — Metrics bổ sung — ✅ **ĐÃ XONG (2026-08-01)**

> Đã cài đặt: `tools/eval_sr_quality.py` + nhánh `sr_eval_mode` trong `dataset.py`.
> **Chạy hậu kỳ trên checkpoint, không đụng training loop** → chạy được ngay trên
> S1–S4 hiện có, song song lúc multi-seed đang chạy. Lệnh: [run_gpu.md §6b](../training_runs/run_gpu.md).
> Quyết định đã chốt: đo trên **999 track val Scenario-B** bằng **cặp synthetic**
> (khớp pixel, đúng thứ `L_SR` được huấn luyện), augment ngẫu nhiên tắt, degradation
> giữ, HR warp theo `theta`. Phần phân tích bên dưới giữ lại làm căn cứ.

**CER + NED**: chỉ là wiring, hàm đã có (điểm 3 mục 0). Ghi thêm 2 cột vào CSV.

**PSNR/SSIM** — đây là hạng mục **tốn công nhất trong Bước 2**, không phải "gọi thêm
một hàm skimage" như review mô tả. Có **4 vấn đề** phải giải quyết trước:

**(1) Val không có track Scenario-A** (0 vs 999) — review nói đo "trên Scenario-A val
tracks" nhưng tập đó không tồn tại. Chọn:
- **(a) Khuyến nghị**: đo trên chính 999 track val Scenario-B → nhất quán với mọi số
  liệu accuracy khác, không phải giải thích thêm tập thứ ba.
- (b) Cắt riêng tập Scenario-A từ train làm "SR eval set" → sát ý review hơn nhưng
  tạo tập thứ 3 và không so trực tiếp được với bảng accuracy.

**(2) Val hiện KHÔNG THỂ sinh ảnh HR để so — đây là blocker thật sự.**
Không chỉ là quên truyền `provide_sr_target`. Cấu trúc code chặn 2 lớp:
```python
# dataset.py:234 — sample synthetic (thứ duy nhất có HR target) chỉ tạo khi train
if self.mode == "train" and hr_files: ...
# dataset.py:295 — HR target chỉ tồn tại với sample synthetic
has_hr_target = bool(self.provide_sr_target and item["is_synthetic"])
```
→ Ở `mode="val"` **mọi sample đều `is_synthetic=False`**, nên dù có bật
`provide_sr_target=True` thì `has_hr_target` vẫn luôn `False`. Phải sửa dataset để
có nhánh sinh cặp (LR, HR) cho val. Hai cách, khác nhau về **ý nghĩa con số**:
- **Cặp synthetic** (degrade HR → LR, dùng `_build_sr_pair`): khớp pixel tuyệt đối,
  đúng thứ mà `L_SR` được huấn luyện. Nhưng đo trên ảnh synthetic, không phải LR thật.
- **Cặp thật** (LR thật + HR thật có sẵn trên đĩa): đúng thứ ta quan tâm, nhưng LR
  thật **không** là bản downscale chính xác của HR thật → lệch hình học, PSNR sẽ bị
  phạt oan vì lệch vị trí chứ không phải vì SR kém.

**(3) Phải warp HR theo `theta` giống hệt lúc tính loss.** `I_SR` nằm trong khung đã
được STN nắn; `I_HR` thì không. `Trainer._sr_loss` xử lý bằng
`grid_sample(hr_flat, affine_grid(theta))`. Nếu tính PSNR mà bỏ bước này, con số sẽ
**vô nghĩa** (so hai ảnh khác hệ toạ độ). Code tính PSNR phải tái sử dụng đúng logic đó.

**(4) ⚠️ PSNR của S1 và S4 KHÔNG so trực tiếp được với nhau.** Độ phân giải output
phụ thuộc `sr_scale` (`dataset.py:109-110`: `sr_target = img_size × sr_scale`):
- S1/S3 (`sr_scale=2`): so ảnh **64×256** với **64×256**
- S4 (`sr_scale=1`): so ảnh **32×128** với **32×128**

PSNR/SSIM ở hai độ phân giải khác nhau là hai đại lượng khác nhau — đặt chung một cột
trong bảng là sai. Nếu paper cần so PSNR giữa các cấu hình, phải upsample output của
S4 lên 64×256 (và nói rõ đã làm vậy), hoặc **báo cáo PSNR/SSIM chỉ cho nhóm
`sr_scale=2`** và ghi chú S4 đo ở thang khác. Khuyến nghị cách sau — trung thực và
đơn giản hơn.

Thư viện: `skimage.metrics` (`peak_signal_noise_ratio`, `structural_similarity`) là đủ,
không cần `torcheval`.

### P3 — Hình định tính cho paper — ✅ **ĐÃ XONG (2026-08-01)**

> Đã cài đặt: `tools/visualize_paper_figures.py` — grid 4 cột
> `I_LR → I_SR → Attention → Prediction`, 5 case đúng + 5 case sai, xuất
> `figure4_qualitative_grid.png` + ảnh từng track. Cũng chạy hậu kỳ trên checkpoint.
> Lệnh: [run_gpu.md §6b](../training_runs/run_gpu.md). Phần phân tích bên dưới giữ lại.

`tools/visualize.py` **đã có sẵn** cả `--only-errors` lẫn trọng số attention
(`AttentionFusion.last_weights`) — đây là mở rộng, không phải viết mới.

Cần thêm: layout grid 4 cột `I_LR → I_SR → Attention heatmap → Prediction`, xuất
10 track (5 đúng / 5 sai). **Track sai đã có sẵn danh sách ứng viên**: các track
confidence thấp đã liệt kê trong doc S1 (`track_19095` 0.494, `track_15594` 0.505,
`track_22161`, `track_19725`, `track_12513`, `track_12247`, `track_12478`) và S3.
Lưu ý S4 có **0 track** confidence < 0.55 → nếu minh hoạ bằng S4 phải chọn failure
case theo tiêu chí sai-so-với-ground-truth, không theo confidence.

### P4 — Chống overfitting (~2–3 ngày GPU, **làm SAU P1**)

Cả 3 đề xuất của review đều hợp lý và khớp với pattern overfit đã ghi nhận (val loss
chạm đáy epoch 15–21 rồi tăng, train loss về gần 0):

| Đề xuất | Hiện trạng | Ghi chú |
|---|---|---|
| `weight_decay` 1e-3 | đang 1e-4 | **Chỉ chạy sau P0.1**. Ablation cũ cho thấy wd cao có ích khi đứng riêng (config C: +0.40) |
| `Dropout(0.3)` trước BiLSTM + FC | FC đã có dropout; **trước BiLSTM thì chưa** | Code change thật trong `crnn.py` |
| Thêm `MotionBlur`, `GaussNoise` vào train augment | Đã có `RandomBrightnessContrast`, `Affine`(≈ShiftScaleRotate), `Perspective`, `Rotate`, `CoarseDropout`; MotionBlur/GaussNoise **chỉ có trong pipeline degradation** (ảnh synthetic), chưa có trong train transform | Bổ sung được, nhưng cẩn thận trùng lặp với degradation |
| `patience` 18 → 12 | đang 18 | An toàn: S1 đỉnh ở epoch 37 (patience 12 → dừng 49), S3 đỉnh 25 (→ 37), S4 đỉnh 40 (→ 52). Không cắt mất đỉnh nào, tiết kiệm ~20% GPU |

> ⚠️ **Không trộn P4 vào P1.** P4 làm thay đổi model → nếu đổi model rồi mới
> multi-seed, hoặc multi-seed xong rồi đổi model, đều phải chạy lại. Thứ tự đúng:
> **P1 chốt con số cho paper hiện tại → P4 là cải tiến riêng, có multi-seed riêng
> nếu muốn đưa vào paper.**

### P5 — Baseline SOTA ngoài repo: PARSeq / SVTR (tuỳ chọn, rủi ro cao nhất)

Đây là hạng mục **đắt và rủi ro nhất**, đề nghị để cuối và coi là optional:

- Cần quyết định **so sánh công bằng thế nào**: PARSeq/SVTR là mô hình *single-image*,
  còn đề xuất của mình là *multi-frame*. Cho PARSeq ăn frame nào? (frame giữa? frame
  rõ nhất? chạy cả 5 frame rồi vote?) Lựa chọn này ảnh hưởng trực tiếp tới tính công
  bằng của bảng so sánh và **chắc chắn sẽ bị reviewer hỏi**.
- Đề xuất: chạy PARSeq trên **frame giữa** (single-frame, đúng bản chất mô hình) và
  báo cáo rõ đây là baseline single-frame — điều đó tự nó làm nổi bật giá trị của
  hướng multi-frame, thay vì cố ép PARSeq thành multi-frame một cách khiên cưỡng.
- Nếu thiếu thời gian, **bỏ P5 và nói rõ trong Limitations** vẫn tốt hơn là một so
  sánh vội vàng, cấu hình sai, rồi bị bắt lỗi.

---

## 2b. Tập test: đã chạy chưa, và chạy lúc nào?

**Trả lời ngắn: CHƯA từng chạy test. Toàn bộ S1–S4 đều là số trên validation.**
Và đúng — phải multi-seed + chốt cấu hình xong **rồi mới** chạy test.

### Bằng chứng đã kiểm tra

| Kiểm tra | Kết quả |
|---|---|
| Header `log_s1.txt`, `log_s4.txt` | `Submission : False` ở cả hai |
| Chuỗi `"Running inference on test"` trong log | 0 lần xuất hiện |
| Số dòng mọi file `submission_*.txt` | **999** — đúng bằng số track **validation** |
| Test public / blind có bao nhiêu track | 1.000 / 3.000 — không khớp 999 |

> ⚠️ **Bẫy đặt tên**: các file `submission_s1_proposed.txt`, `submission_s4_*.txt`…
> **không phải bài nộp**. Đó là dự đoán trên **validation**, do
> `Trainer.save_submission()` ghi ra mỗi khi val acc cải thiện. Dự đoán test thật sẽ
> do `predict_test()` ghi ra với tên `submission_<tên>_final.txt` và có 1.000 dòng
> (public) hoặc 3.000 dòng (blind). Hiện chưa có file nào như vậy.

### Hai điều phải biết trước khi chạy test

**(1) `--submission-mode` KHÔNG phải chỉ là inference — nó train lại từ đầu.**
```python
train.py:309-317  → MultiFrameDataset(..., full_train=True, ...)
dataset.py:154-156 → "FULL TRAIN: dùng toàn bộ tracks, không chia val" → val_tracks = []
```
Nghĩa là: train lại trên **toàn bộ 20.000 track** (gồm cả 999 track val), **không còn
validation**. Hệ quả nghiêm trọng trong `Trainer.fit()`: khi `val_loader is None` thì
**early stopping không hoạt động**, và checkpoint được lưu theo **train loss** —
thứ gần như luôn giảm. Chạy đủ 60 epoch ở chế độ này sẽ **lưu ra model của epoch cuối,
tức model đã overfit nặng** (S1 đạt đỉnh val ở epoch 37 rồi tụt dần; S3 đỉnh ở 25).

→ Nếu dùng `--submission-mode`, **bắt buộc đặt `--epochs` bằng đúng epoch tốt nhất đã
học được từ validation** (S1: 37, S3: 25, S4: 40 — hoặc trung bình qua 3 seed sau P1),
chứ không để 60.

**(2) Chưa có công cụ chạy test trên checkpoint có sẵn.** `tools/eval_decode.py` chạy
`mode="val"` (chấm lại checkpoint trên validation), không nhận test set.

### Hai phương án tạo dự đoán test

| | **Phương án A — dùng lại checkpoint đã có** | **Phương án B — `--submission-mode`** |
|---|---|---|
| Cách làm | Viết script inference nhỏ (~1 h người), nạp `*_best.pth`, chạy trên test | Train lại toàn bộ 20k track, rồi predict test |
| Chi phí GPU | ~0 (chỉ inference) | **thêm 4–11 h/cấu hình** |
| Dữ liệu train | 19.001 track | 20.000 track (**+5%**) |
| Rủi ro | Không có — model đúng bằng model đã báo cáo val | **Cao**: không có val để kiểm chứng, không early stopping, dễ ship model overfit |
| Tính nhất quán cho paper | ✅ số val và số test cùng **một** model | ❌ số val và số test là **hai** model khác nhau |

**Khuyến nghị: Phương án A.** Với một bài báo (khác với đua leaderboard thuần tuý),
việc số val và số test đến từ *cùng một model* quan trọng hơn 5% dữ liệu train thêm.
Phương án B tạo ra một model mà **không cách nào biết nó tốt hay xấu** trước khi nộp.

Nếu vẫn muốn B để tận dụng thêm dữ liệu: chạy **cả hai**, nộp bản A trước để có mốc
an toàn, rồi so bản B trên public test.

### Thứ tự đúng

```
P1 multi-seed (val)  →  chốt cấu hình + epoch tốt nhất  →  inference test public
                                                        →  (nếu có leaderboard) đối chiếu
                                                        →  test blind cuối cùng
```

**Tuyệt đối không** dùng test public để chọn cấu hình — sẽ biến nó thành tập validation
thứ hai và làm số blind test mất giá trị. Việc chọn giữa S1/S3/S4 phải xong **trước
khi** nhìn bất kỳ con số test nào.

---

## 3. Thứ tự thực hiện đề xuất

```
Tuần 1  ├─ P0 (4–6h người, 0 GPU)  ← chặn đường, làm ngay
        └─ khởi động P1: S4 trước (rẻ nhất, ~16h GPU) để phát hiện sớm lỗi quy trình
Tuần 1–2├─ P1 tiếp: S1 × 3 seed (~33h), rồi S3 × 3 seed (~36–45h)
        └─ song song (CPU/người): P2 metrics + P3 hình vẽ
Tuần 2  ├─ Tổng hợp aggregate_seeds → chốt bảng Mean ± Std cho paper
        ├─ benchmark.py cho S4 → hoàn thiện bảng compute
        └─ CHỐT cấu hình cuối (trước khi nhìn bất kỳ số test nào)
Tuần 2–3├─ Inference test public (mục 2b, phương án A — ~0 GPU)
        └─ Test blind cuối cùng
Tuần 3  ├─ P4 chống overfit (nếu còn thời gian, cần multi-seed riêng)
        └─ P5 PARSeq (optional, cắt được nếu thiếu thời gian)
```

**Mẹo giảm rủi ro**: chạy S4 trước S1 trong P1. S4 chỉ mất ~5.3 h/run nên nếu quy
trình có lỗi (thiếu cột CSV, sai flag, quên `tee`) thì phát hiện sau 5 h thay vì
sau 11 h, và mất ít GPU hơn khi phải chạy lại.

## 4. Rủi ro chính

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Deterministic mode làm đổi con số headline (79.78% / 80.58%) | **Cao** | Chấp nhận trước; chuyển sang báo cáo Mean ± Std thay vì best-of-run. Đây vốn là điều đúng đắn hơn về mặt khoa học |
| Chạy multi-seed xong mới nhớ thiếu metric → mất 49 h GPU | **Cao** | P0.2 bắt buộc xong trước P1 |
| Tăng wd 1e-3 khi chưa có param-grouping → kết quả tệ đi mà không hiểu vì sao | Trung bình | P0.1 bắt buộc trước P4 |
| Quên `tee` → mất timing như S2/S3 | Trung bình | Đã đưa `tee` vào mọi lệnh ở P1 |
| P5 (PARSeq) ngốn hết thời gian còn lại | Trung bình | Coi là optional, cắt sớm nếu tuần 2 chưa xong P1–P3 |
| 3 seed vẫn không đủ tách S1 vs S3 vs S4 (chênh 8 track, biên nhiễu ±13) | Trung bình | Nếu Std lớn, kết luận trung thực là "ba cấu hình tương đương trong sai số, S4 rẻ hơn 2.34×" — bản thân điều đó **đã là một kết luận có giá trị cho paper** |
