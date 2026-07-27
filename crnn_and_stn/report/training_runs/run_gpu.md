# Lệnh chạy training trên GPU + log kết quả

Xem phân tích kiến trúc/thiết kế đi kèm tại [../baseline1_crnn_stn/](../baseline1_crnn_stn/).

**Dọn lại 2026-07-26**: file này giờ chỉ để **active** những lệnh thuộc hướng đang làm — fix issue #9 (SR per-frame có giám sát + GroupNorm + DCNv2). Mọi lệnh của các hướng đã kết thúc được gói trong comment HTML `<!-- -->` để giữ lịch sử tham khảo, **số liệu của chúng được tóm tắt lại ở bảng ngay dưới đây** nên không cần mở comment ra mới biết mốc so sánh.

## Mốc kết quả đã đo (dùng để đối chiếu)

**Phân biệt quan trọng**: baseline chuẩn theo report ICPR của tác giả gốc là **CRNN + STN (77.00%)**. `ResBlock backbone (76.68%)` **không phải baseline** — đó là cải tiến backbone làm sau ở PR #8, dùng làm điểm xuất phát cho nhánh J1/J2/M/N/O bên dưới. Hai cột chênh lệch tách riêng để không nhầm "vượt ResBlock" thành "vượt baseline gốc".

| Cấu hình                                        | Val Exact Match | Chênh vs baseline chuẩn (77.00%) | Ghi chú                                                   |
| ------------------------------------------------ | --------------: | --------------------------------: | ---------------------------------------------------------- |
| **CRNN + STN (baseline chuẩn, report ICPR)**     |      **77.00%** |                                — | Số liệu trong report, không phải đo trên dataset project    |
| CRNN + STN (đo thực tế trên dataset project)     |          75.78% |                            −1.22 | batch 64, epochs 30, lr 5e-4                                |
| CRNN + STN + AdamW tuning tốt nhất               |          76.28% |                            −0.72 | Backbone CNN+BatchNorm cũ, đã kết thúc hướng này            |
| ResNet + Transformer + STN (report, tốt nhất)    |          78.70% |                            +1.70 | Kiến trúc khác hẳn (ResNet+Transformer), không so trực tiếp |
| SR stacked-input v1 / v2                         | 49.25% / 55.06% |                     −27.75/−21.94 | Bản lỗi — bằng chứng cho Root cause #1 của issue #9         |
| ResBlock backbone (cải tiến PR #8, **không phải baseline**) | 76.68% |                    −0.32 | **Mốc xuất phát của nhánh J1/J2/M/N/O bên dưới**            |
| J1 — ResBlock + GroupNorm (không SR)             |          76.88% |                            −0.12 | +0.20 so với ResBlock — đối chứng cho J2                    |
| **J2 — ResBlock + GroupNorm + SR per-frame**     |      **77.18%** |                        **+0.18** | +0.50 so với ResBlock, +0.30 so với J1 — 1 seed, chưa multi-seed |

## Setup môi trường GPU

```bash
## giải nén tại chỗ
unzip crnn_and_stn.zip

## giải nén thư mục cùng tên
unzip crnn_and_stn.zip -d crnn_and_stn

## install pip
python -m pip install --upgrade pip

## kiểm tra CUDA driver trên máy GPU thuê trước khi chọn index (góc trên phải "CUDA Version")
nvidia-smi

## install torch — dùng cu124 để khớp pyproject.toml (index "pytorch-cu124", template EZYCLOUDX nvidia/cuda:12.4.1)
## Nếu nvidia-smi báo CUDA driver 12.1-12.3 (thấp hơn 12.4), đổi lại "cu121" trong URL bên dưới.
python -m pip install torch torchvision torchaudio \
 --index-url https://download.pytorch.org/whl/cu124

cd /home/crnn_and_stn

pip install albumentations opencv-python tqdm numpy

apt update
apt install -y libgl1
```

---

# HƯỚNG ĐANG LÀM: fix issue #9

Branch `feature/nhutminh-sr-frame-aware-supervised` (base `develop`/PR #8 — ResBlock backbone). Giữ nguyên AdamW + `WarmupCosineScheduler`, không đổi optimizer.

**4 Root cause của issue #9 đều đã sửa trong code:**

| Root cause                      | Cách sửa                                                    | Flag                                    |
| ------------------------------- | ----------------------------------------------------------- | --------------------------------------- |
| #1 Stacked Input phá STN        | SR per-frame trên `[B*5,3,H,W]`, chạy **sau** STN           | `--use-sr`                              |
| #2 Thiếu pixel supervision      | `L_CTC + λ·L_SR`, `L_SR` = L1 + Sobel edge (+ tuỳ chọn VGG) | `--lambda-sr`, `--sr-perceptual-weight` |
| #3 Bỏ BatchNorm không thay thế  | GroupNorm trong ResBlock/backbone/SR head                   | `--backbone-norm group`                 |
| #4 Domain gap synthetic/real LR | Degrade HR ở đúng cỡ LR gốc (19×46) trước khi resize        | `--lr-domain-match`                     |
| Step 2 pipeline (DCNv2)         | `DCNAlignment` căn chỉnh cục bộ giữa các frame              | `--use-dcn`                             |

## Đã chạy: chẩn đoán NaN (I-a, I-b) — 2026-07-26

Lần chạy đầu tiên (AMP + `norm=none`) **thất bại**: `Train Loss: nan`, val loss đứng im 3.3362 qua 2 epoch → GradScaler skip mọi optimizer step, weights không cập nhật lần nào.

| Run              | AMP |   norm    | Train Loss E1 | Val Loss E1 | Thời gian/epoch |
| ---------------- | :-: | :-------: | ------------: | ----------: | --------------: |
| I gốc            | ✅  |   none    |       **nan** |      3.3362 |            5:50 |
| I-a (chẩn đoán)  | ❌  |   none    |        3.3354 |      3.0368 |           13:59 |
| **I-b (đã fix)** | ✅  | **group** |    **3.0310** |  **2.7588** |        **8:55** |

→ Xác nhận **Root cause #3**: thiếu normalization là gốc rễ, AMP fp16 chỉ phơi bày nó ra khi SR nhân đôi kích thước không gian (T: 16 → 32). GroupNorm vừa sửa được NaN, vừa hội tụ tốt hơn cả bản tắt AMP, lại nhanh hơn 1.6x.

Ngoài GroupNorm, đã thêm 2 lớp phòng vệ: CTC loss ép chạy **fp32** tường minh, và **NaN guard** skip batch lỗi thay vì làm hỏng trung bình cả epoch.

<!-- Lệnh I-a/I-b đã chạy xong, giữ lại để tái lập nếu cần:
python train.py --preset stable --experiment-name crnn_resblock_sr_diag_noamp \
 --epochs 3 --use-sr --sr-scale 2 --lambda-sr 0.1 --no-amp --num-workers 8 --aug-level full

python train.py --preset stable --experiment-name crnn_resblock_sr_supervised_sanity \
 --epochs 5 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group \
 --num-workers 8 --aug-level full
-->

---

# CHECKLIST CẦN CHẠY

Chạy theo thứ tự **J → M → N → O**. Các nhóm ablation trong N độc lập nhau, chạy song song được nếu có nhiều GPU.

Quy ước ghi kết quả: thêm `=> <accuracy>%` vào cuối dòng comment của mỗi lệnh.

## J. Thí nghiệm chính + đối chứng bắt buộc

Mỗi lần chỉ đổi **một** biến so với baseline — bài học từ ablation B/F của hướng AdamW trước đây (đổi 2 biến cùng lúc → không kết luận được biến nào có tác dụng).

```bash
## J1. Đối chứng GroupNorm — KHÔNG SR. CHẠY TRƯỚC J2.
## Bắt buộc: I-b cho thấy GroupNorm TỰ NÓ cải thiện hội tụ. Bỏ qua bước này thì nếu J2
## vượt 76.68% sẽ KHÔNG THỂ biết công lao thuộc về SR hay GroupNorm.
## => 76.88% (best epoch 60/80, early stop epoch 78). +0.20 so với ResBlock 76.68%
## (KHÔNG phải baseline gốc — baseline chuẩn CRNN+STN report = 77.00%, J1 vẫn thấp hơn).
## GPU: RTX 4090, 1:31/epoch (nhanh ~5.9x so với V100 8:55/epoch).
python train.py \
 --preset stable \
 --experiment-name crnn_resblock_groupnorm_nosr \
 --backbone-norm group \
 --num-workers 8 --aug-level full

## J2. Đề xuất chính: SR per-frame + giám sát pixel-level.
## Chạy 60 epoch (thay vì 80 như J1) — SR tốn ~3.66x compute nên cắt bớt cho nhanh.
##
## ✅ MỐC SO SÁNH: J1 đạt best đúng ở epoch 60, nên với ngân sách 60 epoch thì so
## thẳng với 76.88% LÀ CÔNG BẰNG.
##
## ⚠️ YÊU CẦU GPU >= 32GB (V100). Batch 64 + SR cần ~23GB nên OOM trên RTX 4090 24GB
## — SR phóng ảnh 32x128 -> 64x256 (gấp 4 lần pixel), backbone giữ activation cho
## 320 ảnh. Nếu chỉ có card 24GB, dùng lệnh J2-alt bên dưới.
python train.py \
 --preset stable \
 --experiment-name crnn_resblock_sr_supervised \
 --epochs 60 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 \
 --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee results/log_j2.txt

## J2-alt. Chỉ dùng khi GPU 24GB (RTX 4090) — kết quả TƯƠNG ĐƯƠNG J2.
## batch 32 + grad-accum 2 => effective batch vẫn 64, vẫn 594 optimizer step/epoch,
## lịch LR giống hệt. Hợp lệ vì pipeline dùng GroupNorm (chuẩn hoá theo từng sample)
## — nếu là BatchNorm thì gradient accumulation sẽ làm sai lệch thống kê batch.
## => 77.18% (best epoch 57/60). +0.50 so với ResBlock, +0.30 so với J1,
## +0.18 so với baseline chuẩn (77.00%) — 1 seed, xem caveat thống kê bên dưới.
## Phân tích đầy đủ: ../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
 --preset stable \
 --experiment-name crnn_resblock_sr_supervised \
 --epochs 60 \
 --batch-size 32 --grad-accum-steps 2 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 \
 --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee results/log_j2.txt

## J3. + DCNv2 alignment (Step 2 pipeline issue #9)
## ⚠️ CHỈ CHẠY SAU KHI J2 XONG và J2 >= J1. Nếu J2 thua J1 thì J3 (thêm DCNv2 lên
## trên SR) gần như chắc chắn cũng thua — chạy 6 tiếng để rồi bỏ đi.
## ⚠️ KHÔNG chạy song song với J2: cùng 1 GPU sẽ OOM (mỗi process giữ ~15GB).
## ⚠️ Cần batch 32 + accum 2 giống J2 — batch 64 + SR sẽ OOM trên card 24GB.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
 --preset stable \
 --experiment-name crnn_resblock_sr_dcn \
 --epochs 60 \
 --batch-size 32 --grad-accum-steps 2 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 \
 --use-dcn \
 --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee results/log_j3.txt
```

### Kết quả J1 — GroupNorm một mình: 76.88% (2026-07-26)

| Mốc                                            |    Val Acc | Chênh vs ResBlock | Chênh vs baseline chuẩn (77.00%) |
| ----------------------------------------------- | ---------: | -----------------: | ---------------------------------: |
| ResBlock backbone (norm=none, **không phải baseline**) |     76.68% |                — |                              −0.32 |
| **J1 — + GroupNorm**                           | **76.88%** |          **+0.20** |                          **−0.12** |

**Diễn biến**: best ở epoch 60/80, early stopping kích hoạt ở epoch 78 (18 epoch không cải thiện). Thời gian 1:31/epoch trên RTX 4090.

**Hai điều phải nhớ khi đọc J2:**

1. **+0.20 điểm NẰM TRONG biên độ nhiễu.** Validation chỉ 999 sample → CI 95% ≈ ±2.7 điểm. Một mình con số này **không đủ** để khẳng định GroupNorm cải thiện accuracy. Giá trị thật của J1 không nằm ở +0.20 mà ở chỗ nó là **đối chứng**: từ giờ J2 phải so với 76.88%, không phải 76.68%.

2. **Model overfit rõ rệt** — đây là phát hiện quan trọng hơn cả con số accuracy:

| Epoch | Train Loss |         Val Loss |               Val Acc |
| ----: | ---------: | ---------------: | --------------------: |
|    19 |     0.0788 | **0.2591** ← đáy |                71.37% |
|    36 |     0.0281 |           0.2933 |                74.97% |
|    60 |     0.0074 |           0.3308 | **76.88%** ← best acc |
|    78 |     0.0042 |           0.3535 |                75.88% |

Train loss xuống 0.0042 = model gần như thuộc lòng tập train. Val loss chạm đáy từ epoch 19 rồi **tăng liên tục 36%**, trong khi val acc vẫn nhích lên — model đoán đúng nhiều hơn nhưng khi sai thì sai với độ tự tin cao hơn.

→ **Hệ quả cho các run sau**: 80 epoch là quá dài cho dataset ~19,000 track này. Dư địa cải thiện nhiều khả năng nằm ở **chống overfit** (augmentation mạnh hơn, dropout cao hơn, weight decay cao hơn) chứ không phải train lâu hơn hay thêm tham số. Đây cũng đúng với bài học từ run E của hướng AdamW cũ (60 epoch tệ hơn 30 epoch).

**Mốc so sánh theo ngân sách epoch** (dùng khi run sau chạy ít epoch hơn J1):

| Ngân sách         | J1 best trong khoảng đó | Đạt ở epoch |
| ----------------- | ----------------------: | ----------: |
| 30 epoch          |                  73.07% |          29 |
| 50 epoch          |                  74.97% |          36 |
| **60 epoch**      |              **76.88%** |      **60** |
| 80 epoch (đầy đủ) |                  76.88% |          60 |

J1 đạt đỉnh đúng ở epoch 60 rồi đi ngang tới khi early stop (epoch 78) — nên **60 epoch là ngân sách vừa đủ**, chạy thêm 20 epoch nữa không thu được gì. J2 chạy 60 epoch nên so thẳng với **76.88%** là hợp lệ.

### Kết quả J2 — SR per-frame + giám sát: 77.18% (2026-07-26)

Phân tích đầy đủ tại [../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md](../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md). Tóm tắt:

| Mốc | Val Acc | Chênh vs J1 | Chênh vs ResBlock | Chênh vs baseline chuẩn (77.00%) |
|---|---:|---:|---:|---:|
| J1 — ResBlock + GroupNorm | 76.88% | — | +0.20 | −0.12 |
| **J2 — + SR per-frame** | **77.18%** | **+0.30** | **+0.50** | **+0.18** |

**Caveat bắt buộc phải đọc trước khi kết luận**: cùng cấu hình J2 chạy 2 lần (lần đầu bị dừng giữa chừng do lỗi hạ tầng) cho kết quả lệch nhau **6.5 điểm** ở epoch 3, dù cùng seed — do `cudnn.benchmark=True` không xác định. Validation chỉ 999 sample → CI 95% ≈ ±2.7 điểm. **Chênh lệch +0.18 so với baseline chuẩn nằm gọn trong 2 nguồn nhiễu này — chưa đủ bằng chứng để khẳng định pipeline SR+GroupNorm đã vượt baseline gốc của tác giả.** Cần O1 (multi-seed) mới kết luận được.

## M. Sửa Root cause #4 — domain gap synthetic/real LR

`--lr-domain-match` hạ ảnh HR về đúng cỡ LR gốc (19×46) **trước khi** degrade, để nhiễu/blur sinh ra ở cùng thang không gian với ảnh LR thật — thay vì degrade ở cỡ HR (96×240) rồi mới thu nhỏ như hiện tại.

```bash
## M1. Chỉ bật domain-match, KHÔNG SR — cô lập tác động của riêng thay đổi này
python train.py \
 --preset stable \
 --experiment-name crnn_resblock_domainmatch \
 --backbone-norm group --lr-domain-match \
 --num-workers 8 --aug-level full

## M2. Domain-match + SR (cấu hình đầy đủ nhất) — chỉ chạy nếu M1 >= J1
python train.py \
 --preset stable \
 --experiment-name crnn_resblock_sr_domainmatch \
 --use-sr --sr-scale 2 --lambda-sr 0.1 \
 --backbone-norm group --lr-domain-match \
 --num-workers 8 --aug-level full
```

## N. Bốn nhóm Ablation bắt buộc theo issue #9

```bash
## N1 — Ablation 1: trọng số SR loss, λ ∈ {0.01, 0.1, 0.5, 1.0}
## (λ=0.1 chính là J2, nên chỉ cần chạy 3 giá trị còn lại)
for lam in 0.01 0.5 1.0; do
python train.py \
 --preset stable --experiment-name "abl1_lambda_${lam}" \
 --use-sr --sr-scale 2 --lambda-sr $lam \
 --backbone-norm group --num-workers 8 --aug-level full
done

## N1b — Ablation 1 mở rộng: perceptual loss VGG16 (L_Perceptual trong công thức issue)
## Lưu ý: tải ~528MB weights lần đầu, tăng VRAM + thời gian/epoch.
python train.py \
 --preset stable --experiment-name abl1_perceptual \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --sr-perceptual-weight 0.1 \
 --backbone-norm group --num-workers 8 --aug-level full

## N2 — Ablation 2: cơ chế fusion (attention là mặc định = J2)
for mode in avg max; do
python train.py \
 --preset stable --experiment-name "abl2_fusion_${mode}" \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --fusion-mode $mode \
 --backbone-norm group --num-workers 8 --aug-level full
done

## N3 — Ablation 3: số frame N ∈ {1,2,3,4,5} (5 = J2). Frame được chọn TRẢI ĐỀU trong track,
## không phải N frame đầu — với N=2 lấy frame đầu+cuối (đa dạng nhất về góc/blur).
for nf in 1 2 3 4; do
python train.py \
 --preset stable --experiment-name "abl3_frames_${nf}" \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --num-frames $nf \
 --backbone-norm group --num-workers 8 --aug-level full
done

## N4 — Ablation 4: độ phân giải đầu vào (32x128 = J2)
python train.py \
 --preset stable --experiment-name abl4_res_16x48 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --img-height 16 --img-width 48 \
 --backbone-norm group --num-workers 8 --aug-level full

python train.py \
 --preset stable --experiment-name abl4_res_24x96 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --img-height 24 --img-width 96 \
 --backbone-norm group --num-workers 8 --aug-level full
```

## O. Thống kê nhiều seed + chi phí tính toán + minh hoạ định tính

**Bắt buộc với cấu hình thắng cuộc.** Validation chỉ 999 sample → CI 95% khoảng ±2.7 điểm, nên chênh lệch 0.2-0.5 điểm giữa các cấu hình **không thể** kết luận từ 1 seed.

```bash
## O1. Chạy lại cấu hình tốt nhất với 3 seed (42 / 100 / 2026)
## Thay flag SR bên dưới bằng đúng flag của cấu hình thắng ở J/M/N.
for seed in 42 100 2026; do
python train.py \
 --preset stable --experiment-name "best_seed${seed}" --seed $seed \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee "results/log_seed${seed}.txt"
done

## O2. Tổng hợp Mean ± Std + kiểm định chênh lệch có vượt nhiễu không
python tools/aggregate_seeds.py --from-logs "results/log_seed*.txt" --label "SR + GroupNorm"

## Hoặc so trực tiếp 2 cấu hình (nhập accuracy đã đo):
python tools/aggregate_seeds.py \
 --acc 76.28 76.05 76.51 --label "SR + GroupNorm" \
 --baseline-acc 76.68 76.40 76.22 --baseline-label "ResBlock baseline"

## O3. Bảng Params / FLOPs / Latency — so sánh mọi cấu hình chính trong 1 lệnh
python tools/benchmark.py --all

## O4. Ảnh minh hoạ định tính: LR input -> SR output -> attention weight -> prediction
python tools/visualize.py \
 --checkpoint results/crnn_resblock_sr_supervised_best.pth \
 --use-sr --backbone-norm group \
 --num-samples 12 --output-dir results/viz

## Chỉ xuất các case dự đoán SAI — tìm pattern lỗi (0/O, 1/I, 5/S, 8/B...)
python tools/visualize.py \
 --checkpoint results/crnn_resblock_sr_supervised_best.pth \
 --use-sr --backbone-norm group \
 --only-errors --num-samples 20 --output-dir results/viz_errors
```

### Chi phí tính toán — đã đo sẵn (CPU, preset stable, batch=1)

| Cấu hình                     |     Params | GFLOPs/track | Latency (ms) | vs baseline |
| ---------------------------- | ---------: | -----------: | -----------: | ----------: |
| Baseline ResBlock (không SR) | 29,298,220 |        25.90 |        53.35 |       1.00x |
| + GroupNorm                  | 29,313,452 |        25.90 |        52.84 |       0.99x |
| + SR (per-frame)             | 29,426,895 |       107.99 |       195.04 |   **3.66x** |
| + SR + DCNv2                 | 29,445,854 |       108.75 |       201.60 |       3.78x |

**Đọc bảng này**: GroupNorm gần như **miễn phí** (+15K params, FLOPs không đổi) — nên giữ bất kể SR có thắng hay không. Ngược lại SR đắt thật: **+4.2x FLOPs, +3.66x latency** vì backbone phải xử lý ảnh 64×256 thay vì 32×128. Nếu J2 chỉ hơn J1 dưới 1 điểm, cái giá này khó biện minh cho triển khai thực tế. DCNv2 thêm vào gần như không tốn gì (+0.76 GFLOPs).

## Hạng mục CHƯA làm và lý do

| Hạng mục (issue #9)                                 | Trạng thái  | Lý do                                                                                                                                                                            |
| --------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline 3: EDSR / Real-ESRGAN pretrained           | ❌          | Cần tải weights + pipeline tiền xử lý riêng; là hướng "SR rời" khác hẳn SR end-to-end đang làm                                                                                   |
| SOTA: PARSeq / SVTR / ConvNeXt-OCR                  | ❌          | Kiến trúc OCR hoàn toàn riêng (không phải CRNN), cần repo ngoài + pretrained. Thuộc phạm vi một PR/issue độc lập                                                                 |
| Baseline 1: single-frame vs multi-frame avg pooling | ⚠️ Một phần | `--num-frames 1` (N3) cho single-frame; `--fusion-mode avg` (N2) cho average pooling. Chạy `--num-frames 1 --fusion-mode avg` nếu cần đúng chính xác mục này                     |
| Multi-Frame SR thật (gộp 5 frame → 1 ảnh SR)        | ❌          | SR hiện tại **cố ý** để per-frame nhằm giữ đa dạng frame cho Attention Fusion (sửa Root cause #1). MFSR thật mâu thuẫn với Attention Fusion ở Step 4 của chính sơ đồ trong issue |

---

<!-- ==================== LỊCH SỬ — CÁC HƯỚNG ĐÃ KẾT THÚC ====================
Số liệu của toàn bộ phần dưới đã được tóm tắt ở bảng "Mốc kết quả đã đo" đầu file.
Giữ lệnh lại để tái lập khi cần, KHÔNG phải việc cần làm tiếp theo.

## Baseline 1 gốc (CRNN + STN, chưa nâng cấp backbone/SR)

## batch = 64, epochs = 30, lr = 0.0005 => 75.78%
python train.py --experiment-name crnn_stn_pytorch \
 --batch-size 64 --epochs 30 --lr 0.0005 --num-workers 8 --aug-level full

## batch = 96, epochs = 50, lr = 0.001 => 75.08%
python train.py --experiment-name crnn_stn_pytorch \
 --batch-size 96 --epochs 50 --lr 0.001 --num-workers 8 --aug-level full

## batch = 64, epochs = 30, lr = 0.0005, aug light => 73.97%
python train.py --experiment-name exp_light_aug \
 --batch-size 64 --epochs 30 --lr 0.0005 --num-workers 8 --aug-level light

## Submission mode (không có accuracy vì submission-mode không giữ val split)
python train.py --submission-mode --epochs 30 --lr 0.0005

## ---------- Baseline 1 + AdamW tuning — ARCHIVED ----------
Chạy trên backbone CNN+BatchNorm cũ (trước ResBlock ở PR #8). Backbone đó không còn tồn tại
trên code hiện tại, và các flag `--weight-decay`/`--onecycle-pct-start` KHÔNG có trong
`train.py` hiện tại (nhánh đó dùng OneCycleLR, nhánh hiện tại dùng WarmupCosineScheduler
qua `--warmup-ratio`/`--min-lr-ratio`). Không copy-paste chạy trực tiếp được.
Chi tiết: ../baseline1_crnn_stn/optimizer_adamw_verification.md

A. param-grouping                    => 75.98%   --batch-size 64 --epochs 30 --lr 0.0005
B. wd 5e-4 + pct_start 0.15          => 75.58%   (2 biến cùng lúc → không kết luận được)
C. chỉ wd 5e-4                       => 76.18%
D. chỉ pct_start 0.15                => 76.28%   (tốt nhất của hướng này)
E. D + 60 epoch                      => 75.08%   (overfit)
F. wd 2e-4 + pct_start 0.2           => 76.08%
KẾT LUẬN: D tốt nhất (76.28%) nhưng bị ResBlock backbone (76.68%) vượt qua mà không cần
tuning này → dừng hướng optimizer-tuning-trên-backbone-cũ.

## ---------- Super Resolution stacked-input (bản LỖI) ----------
Bằng chứng gốc cho Root cause #1 của issue #9: gộp 5 frame → 1 → nhân bản, làm mất khả năng
căn chỉnh độc lập của STN. Phân tích: ../baseline1_crnn_stn/super_resolution_experiments.md

## v1 => 49.25%
python train.py --experiment-name crnn_stn_sr \
 --batch-size 64 --epochs 30 --lr 0.0005 --num-workers 8 --aug-level full \
 --use-sr --sr-scale 2

## v2 => 55.06%
python train.py --experiment-name crnn_stn_sr \
 --batch-size 64 --epochs 30 --lr 0.0003 --num-workers 8 --aug-level light \
 --use-sr --sr-scale 2

## ---------- ResBlock backbone (baseline hiện tại, 76.68%) ----------
Đây là MỐC SO SÁNH CHÍNH cho checklist phía trên — đã chạy xong, không cần chạy lại.

### Preset stable => ~76.68%
python train.py --preset stable --experiment-name crnn_resblock_stable \
 --batch-size 64 --epochs 80 --lr 0.0008 --grad-accum-steps 1 \
 --warmup-ratio 0.05 --min-lr-ratio 0.05 --patience 18 --num-workers 8 --aug-level full

### Các biến thể chưa chạy, chưa có số liệu:
python train.py --preset strong --experiment-name crnn_resblock_strong \
 --batch-size 96 --epochs 120 --lr 0.0006 --grad-accum-steps 2 \
 --warmup-ratio 0.08 --min-lr-ratio 0.03 --patience 24 --num-workers 8 --aug-level full

python train.py --experiment-name crnn_resblock_custom \
 --batch-size 64 --epochs 100 --lr 0.0006 --num-workers 8 --aug-level full \
 --backbone-base-channels 64 --backbone-blocks 2,2,3,3,4 \
 --backbone-stage-channels 64,128,256,256,512 --backbone-res-scale 0.08 \
 --fusion-dropout 0.05 --frame-dropout 0.08

python train.py --preset stable --no-se --experiment-name crnn_resblock_no_se \
 --batch-size 64 --epochs 80 --lr 0.0008 --num-workers 8 --aug-level full

### Submission cuối — chỉ chạy sau khi chốt cấu hình tốt nhất
python train.py --submission-mode --preset strong \
 --experiment-name crnn_resblock_submission \
 --batch-size 96 --epochs 120 --lr 0.0006 --grad-accum-steps 2 \
 --warmup-ratio 0.08 --min-lr-ratio 0.03 --num-workers 8 --aug-level full
==================== HẾT PHẦN LỊCH SỬ ==================== -->

## Ghi chú

- File này **chỉ chứa lệnh chạy trên GPU** (cloud/server). Lệnh chạy local trên máy dev (dùng `uv`, path cá nhân) không gộp vào đây — môi trường/dataset path khác nhau nên không dùng để so sánh kết quả chính thức.
- `--submission-mode` / `--full-train` train trên 100% dữ liệu, không giữ validation split → không tính được accuracy trong lúc train, chỉ dùng sau khi đã chốt cấu hình tốt nhất từ các run có split.
- Phần trong `<!-- -->` ở cuối file là lịch sử các hướng đã kết thúc — số liệu đã tóm tắt ở bảng đầu file, không cần mở ra để biết mốc so sánh.
