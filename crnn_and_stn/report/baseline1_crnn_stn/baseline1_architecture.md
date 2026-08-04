# Baseline 1: Multi-Frame CRNN + STN — Kiến trúc & Nguồn gốc

Tài liệu này gộp lại nội dung trích xuất Baseline 1 từ project gốc `MultiFrame-LPR-main` (trước đây là `extraction_guide.md`) và phần phân tích vai trò CRNN+STN đối chiếu với report ICPR (trước đây là phần 1 của `file.md`).

## 1. Baseline 1 là gì (theo report ICPR)

Report `Documents_2026-0_ICPR Challenge - Training_AIO2025_ICPR_TRAINING.pdf` định nghĩa 2 baseline:
- **Baseline 1: Multi-Frame CRNN (+ STN)** — dựa trên paper *"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"* (arxiv 1507.05717).
- **Baseline 2: ResNet34 + Transformer (+ STN)** — không dùng trong project này.

Kiến trúc Baseline 1: `CNN + BiLSTM + CTC`, dùng CTC (Connectionist Temporal Classification) để học alignment-free giữa ảnh và chuỗi ký tự, nhận dạng chuỗi độ dài bất kỳ mà không cần segment ký tự thủ công.

### Pipeline Multi-Frame CRNN (+ STN)

```
5 LR frames → [STN Block] → CNN Layers (weight sharing) → Attention Fusion → BiLSTM Layers → FC → CTC decode
```

- **STN Block**: dự đoán 6 tham số affine, warp từng frame để căn chỉnh hình học (xoay/lệch/méo), khởi tạo ở identity transform nên ban đầu không đổi ảnh, học dần qua training. (Spatial Transformer Networks, arxiv 1506.02025)
- **CNN Layers (weight sharing)**: trích feature map cho từng frame bằng cùng một bộ trọng số.
- **Attention Fusion**: Score Net chấm điểm chất lượng từng frame → Softmax → weighted sum → gộp 5 frame thành 1 fused feature map.
- **BiLSTM Layers**: đọc chuỗi feature theo trục width.
- **FC + CTC**: xuất log-probabilities cho 37 class (36 ký tự `0-9A-Z` + 1 blank), decode bằng merge-repeats rồi remove-blanks.

### Kết quả thực nghiệm gốc trong report (trang 50)

| Model | Accuracy (%) |
|---|---:|
| CRNN | 74.45 |
| **CRNN + STN** | **77.00** |
| ResNet + Transformer | 75.80 |
| ResNet + Transformer + STN | 78.70 |

→ **CRNN + STN (77.00%)** là mốc Baseline 1 chuẩn để so sánh mọi cải tiến (ResBlock backbone, Super Resolution...) áp dụng sau này trong project.

### Vai trò hệ thống của từng thành phần

- `STN` hỗ trợ hiệu chỉnh lệch góc, méo hình, vị trí ký tự chưa ổn định.
- `CRNN` đảm nhiệm nhận dạng chuỗi ký tự từ ảnh đã căn chỉnh.
- Với biển số xe, `CRNN + STN` là baseline hợp lý vì dữ liệu chịu ảnh hưởng của độ phân giải thấp, blur do chuyển động, lệch phối cảnh, vùng ký tự nhỏ khó đọc. Giữ một baseline không SR là cần thiết để so sánh trực tiếp với các cấu hình có SR — nếu không, rất khó xác định SR có thực sự cải thiện khả năng đọc ký tự hay chỉ làm ảnh "nhìn đẹp hơn".

## 2. Trích xuất Baseline 1 thành project độc lập

`MultiFrame-LPR-main` chứa 2 baseline; project `crnn_and_stn` **chỉ trích xuất Baseline 1** (Multi-Frame CRNN + STN), bỏ toàn bộ Baseline 2 (ResNet34 + Transformer).

### 2.1 Model (`src/models/`)

| Thành phần | File gốc | Hành động |
|---|---|---|
| `MultiFrameCRNN` | `crnn.py` | ✅ Copy + comment pipeline từng bước |
| `STNBlock` | `components.py` L10-50 | ✅ Copy — warp affine 6 tham số |
| `AttentionFusion` | `components.py` L53-87 | ✅ Copy — gộp 5 frame |
| `CNNBackbone` | `components.py` L90-111 | ✅ Copy — CNN 5 block |
| `ResNetFeatureExtractor` | `components.py` L114-166 | ❌ Bỏ — thuộc Baseline 2 |
| `PositionalEncoding` | `components.py` L169-194 | ❌ Bỏ — thuộc Baseline 2 |
| `ResTranOCR` | `restran.py` | ❌ Bỏ toàn file |

### 2.2 Data (`src/data/`)

| Thành phần | File gốc | Hành động |
|---|---|---|
| `MultiFrameDataset` | `dataset.py` | ✅ Copy + `label.strip()` + comment |
| `get_train_transforms` / `get_val_transforms` / `get_light_transforms` | `transforms.py` | ✅ Copy |
| `get_degradation_transforms` | `transforms.py` | ✅ Copy — sinh synthetic LR từ HR |

Logic dataset quan trọng:
```python
# Mỗi track train → 2 samples:
samples.append({"paths": lr_files, "is_synthetic": False})   # LR thật
samples.append({"paths": hr_files, "is_synthetic": True})    # HR → degrade → synthetic LR

# Validation chỉ từ Scenario-B:
scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
val_size = int(len(scenario_b_tracks) * (1 - split_ratio))  # 10% với ratio=0.9
```

### 2.3 Training (`src/training/`)

| Thành phần | File gốc | Hành động |
|---|---|---|
| `Trainer` | `trainer.py` | ✅ Copy — CTC loss, OneCycleLR, AMP |
| CTC Loss | `trainer.py` | `nn.CTCLoss(blank=0)` |
| Metric | `trainer.py` | Exact Match accuracy |
| Optimizer | `trainer.py` | AdamW, lr=5e-4 |

### 2.4 Utils (`src/utils/`)

| Thành phần | File gốc | Hành động |
|---|---|---|
| `decode_with_confidence` | `postprocess.py` | ✅ Copy — CTC greedy decode |
| `seed_everything` | `common.py` | ✅ Copy |

### 2.5 Config & Entry

| Thành phần | File gốc | Hành động |
|---|---|---|
| `Config` | `config.py` | ✅ Chỉ giữ CRNN params, sửa `DATA_ROOT` |
| `train.py` | `train.py` | ✅ Bỏ ResTran branch, chỉ `MultiFrameCRNN` |
| `run_ablation.py` | — | ❌ Bỏ — dùng `python train.py --no-stn` |

### 2.6 Thay đổi so với bản gốc

| Mục | Gốc | Trích xuất |
|---|---|---|
| `DATA_ROOT` | `data/train` | `dataset/data/train` |
| `TEST_DATA_ROOT` | `data/public_test` | `dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public` |
| `VAL_SPLIT_FILE` | `data/val_tracks.json` | `dataset/val_tracks.json` |
| `MODEL_TYPE` | `restran` default | Bỏ — chỉ CRNN |
| `NUM_WORKERS` | 10 | 4 (mặc định an toàn hơn) |
| Label loading | raw `plate_text` | `.strip()` loại space thừa |
| Comments | English | Tiếng Việt |

### 2.7 Hyperparameters gốc (report trang 48)

```python
BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
EPOCHS        = 30
SEED          = 42
HIDDEN_SIZE   = 256      # BiLSTM
RNN_DROPOUT   = 0.25
USE_STN       = True     # Baseline 1
IMG_SIZE      = 32 × 128
CHARS         = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 37 classes với blank
```

### 2.8 Luồng chạy end-to-end

```
1. train.py
   └─ Config() → hyperparameters

2. MultiFrameDataset(root="dataset/data/train")
   ├─ Scan track_* folders
   ├─ Split: val từ Scenario-B (10%)
   ├─ Index: real LR + synthetic LR samples
   └─ __getitem__: load 5 frames → [5,3,32,128]

3. MultiFrameCRNN(num_classes=37, use_stn=True)
   ├─ STNBlock: warp từng frame
   ├─ CNNBackbone: extract features (weight sharing)
   ├─ AttentionFusion: 5 frames → 1
   ├─ BiLSTM: sequence modeling
   └─ FC: [B, T, 37] log_softmax

4. Trainer.fit()
   ├─ CTCLoss(preds, targets)
   ├─ validate: Exact Match accuracy
   └─ save best checkpoint → results/

5. decode_with_confidence()
   └─ merge repeats → remove blank → text + confidence
```

### 2.9 Lệnh chạy nhanh (Baseline 1 gốc, chưa nâng cấp ResBlock/SR)

```bash
cd crnn_and_stn

# Train CRNN+STN
python train.py

# Ablation CRNN không STN
python train.py --no-stn --experiment-name crnn_no_stn

# Debug nhanh (ít augmentation)
python train.py --epochs 2 --aug-level light --batch-size 8

# Submission
python train.py --submission-mode --experiment-name crnn_stn_final
```

### 2.10 Dependencies

```
torch, torchvision    # model + training
albumentations        # augmentation + degradation
opencv-python         # đọc ảnh trong dataset
tqdm                  # progress bar
numpy                 # decode confidence
```

Không cần: `pandas`, `matplotlib`, `seaborn` (chỉ dùng cho analysis trong repo gốc).

## 3. Các bước cải tiến sau Baseline 1

> **Xem [model_comparison_summary.md](model_comparison_summary.md) để có bảng so sánh TẤT CẢ cấu hình đã chạy (20 dòng, từ baseline gốc tới S4 + multi-seed) trong 1 file duy nhất**, thay vì đọc rải rác qua từng tài liệu bên dưới.

### 🎯 Ba model cuối cùng của paper — ✅ đã có đủ multi-seed (2026-08-04)

| Model | Mô tả | `T` | GFLOPs | Kết quả (3 seed deterministic) |
|---|---|---:|---:|---|
| **J1** | ResBlock + GroupNorm, **không SR** | 16 | **26.14** | **80.45% ± 0.45** 🥇 |
| **S1** | Joint MF-SR-OCR ×2 (STN→DCN→MFSR→BiLSTM+CTC) | 32 | 109.08 | **79.95% ± 0.15** 🥈 |
| **S4** | SR ×1, T=32 qua `width_downsample=4` | 32 | chưa đo | **79.48% ± 0.44** 🥉 |

> 🚨 **Cấu hình KHÔNG SR đạt điểm cao nhất.** J1 hoà S1 (+0.50, trong nhiễu) và
> **hơn S4 có ý nghĩa thống kê** (+0.97 > 2×0.36), trong khi rẻ hơn **3.76×**
> compute. **Nhánh SR chưa chứng minh được đóng góp nào.**
>
> Phần cải thiện thật so với baseline gốc đến từ **cụm cờ nền** — STN pool `(4,8)`
> + `--lr-domain-match` + constrained decode + EMA: J1-mới hơn J1-**lịch sử**
> **+3.57 điểm** (76.88% → 80.45%) với **cùng** kiến trúc không SR.
>
> Giả thuyết *"`T=32` là yếu tố chính"* cũng **bị bác bỏ** — J1 chạy `T=16`.
>
> Chi tiết: [multi_seed_results.md](multi_seed_results.md) ·
> [groupnorm_sr_ablation_j1_j2.md §3c](groupnorm_sr_ablation_j1_j2.md).

⚠️ J1 multi-seed dùng **cờ nền của S1/S4** (STN pool `(4,8)`, domain-match,
constrained, EMA — chỉ bỏ SR/DCN), **không** phải cờ lịch sử `(1,1)`. Vì thế số
**không** đặt chung cột với 76.88%.

Mọi cấu hình khác (J2, J3, S2, S3, nhánh AdamW, SR-v1/v2) là **exploratory 1 seed** —
giữ làm phụ lục, không vào bảng chính có error bar.

### Lịch sử các hướng đã thử

Baseline 1 (CRNN+STN, mốc 77.00% theo report / ~75.78% đo trên dataset thực tế của project — xem [training_runs/run_gpu.md](../training_runs/run_gpu.md)) là điểm khởi đầu:

1. **Nâng cấp backbone CNN → ResBlock** (đã áp dụng, ~76.68%) — xem [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md).
2. **AdamW tuning trên backbone gốc** (đã kết thúc, mức trần 76.28%, bị ResBlock vượt qua) — xem [optimizer_adamw_verification.md](../../backup/report/baseline1_crnn_stn/optimizer_adamw_verification.md).
3. **Super Resolution per-frame + GroupNorm** (J1 = 76.88%, J2 = 77.18% — cả hai trong biên nhiễu ±13 so với baseline) — xem [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md). Lịch sử các hướng SR đã thử trước đó (kể cả hướng thất bại) xem [super_resolution_experiments.md](../../backup/report/summary_project/document/super_resolution_experiments.md).
4. **Cụm cờ nền** (STN pool `(4,8)` + `--lr-domain-match` + constrained decode + EMA) — ✅ **đây mới là hướng thắng thật**: áp lên đúng kiến trúc J1 (không SR) cho **80.45% ± 0.45**, hơn J1 lịch sử **+3.57 điểm** và vượt baseline gốc 77.00% hơn **3 điểm**. Xem [groupnorm_sr_ablation_j1_j2.md §3c](groupnorm_sr_ablation_j1_j2.md).
5. **Joint End-to-End MF-SR-OCR** (S1 = 79.95% ± 0.15) — thêm multi-frame SR + DCNv2 lên trên cụm cờ nền. ⚠️ **Không cải thiện** so với J1 (−0.50, trong nhiễu) mà tốn 3.76× compute. Xem [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md).
6. **Biến thể rẻ SR ×1** (S4 = 79.48% ± 0.44) — kém J1 có ý nghĩa thống kê. Giả thuyết `T=32` bị bác bỏ. Xem [s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md).
