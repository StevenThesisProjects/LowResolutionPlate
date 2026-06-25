# Baseline 1: Multi-Frame CRNN + STN

Trích xuất từ [MultiFrame-LPR-main](../MultiFrame-LPR-main/) — chỉ giữ phần **Baseline 1** (CRNN + STN), bỏ ResTran (Baseline 2).

Chạy trên dataset ICPR 2026 LRLPR trong thư mục `dataset/`.

## Cấu trúc project

```
crnn_and_stn/
├── train.py                    # Entry point — chạy train/inference
├── configs/config.py           # Hyperparameters (report Trang 48)
├── src/
│   ├── models/
│   │   ├── crnn.py             # MultiFrameCRNN — pipeline chính
│   │   └── components.py       # STNBlock, CNNBackbone, AttentionFusion
│   ├── data/
│   │   ├── dataset.py          # Load 5 frame/track, synthetic LR
│   │   └── transforms.py       # Augmentation + degradation
│   ├── training/
│   │   └── trainer.py          # CTC loss, train/val loop
│   └── utils/
│       ├── postprocess.py      # CTC decode + confidence
│       └── common.py           # seed_everything
├── dataset/                    # Dữ liệu ICPR LRLPR
├── results/                    # Checkpoint + submission (tự tạo khi train)
└── summary/                    # Tài liệu phân tích
```

## Pipeline model

```
5 LR frames [B, 5, 3, 32, 128]
    → STN Block          (căn chỉnh affine từng frame)
    → CNN Backbone       (weight sharing — cùng CNN cho 5 frame)
    → Attention Fusion   (gộp 5 feature map → 1)
    → BiLSTM (2 layer)   (mô hình chuỗi)
    → FC + CTC           (decode → "BAI8068")
```

## Cài đặt

### Cách A — EZYCLOUDX template PyTorch (khuyến nghị, dễ nhất)

Xem hướng dẫn đầy đủ: **[`summary/pytorch_template_run.md`](summary/pytorch_template_run.md)**

```bash
# Trên container (torch CUDA có sẵn):
pip install albumentations opencv-python tqdm numpy
python train.py --experiment-name crnn_stn_pytorch --num-workers 8
```

### Cách B — nvidia/cuda + uv (reproduce chuẩn hơn)

Xem: [`summary/uv_setup.md`](summary/uv_setup.md)

```bash
uv python pin 3.11 && uv sync && uv run python train.py
```

## Chạy training

```bash
# CRNN + STN (mặc định — Baseline 1)
uv run python train.py

# Ablation tự động: CRNN vs CRNN+STN
python run_ablation.py

# Hoặc chạy thủ công
python train.py --no-stn

# Tùy chỉnh
python train.py \
    --experiment-name my_run \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.0005 \
    --aug-level full

# Train toàn bộ data + tạo submission test public
python train.py --submission-mode
```

## Kết quả mong đợi (report Trang 50)

| Model          | Accuracy   |
| -------------- | ---------- |
| CRNN           | 74.45%     |
| **CRNN + STN** | **77.00%** |

## Output

Sau khi train, trong `results/`:

| File                               | Mô tả                                          |
| ---------------------------------- | ---------------------------------------------- |
| `crnn_stn_baseline_best.pth`       | Checkpoint tốt nhất (theo val acc)             |
| `submission_crnn_stn_baseline.txt` | Dự đoán validation: `track_id,text;confidence` |

## Mapping từ MultiFrame-LPR-main

| File gốc                   | File trích xuất            | Ghi chú                                          |
| -------------------------- | -------------------------- | ------------------------------------------------ |
| `src/models/crnn.py`       | `src/models/crnn.py`       | Giữ nguyên, thêm comment                         |
| `src/models/components.py` | `src/models/components.py` | Chỉ STN, CNN, Attention (bỏ ResNet, Transformer) |
| `src/data/dataset.py`      | `src/data/dataset.py`      | + strip label, comment tiếng Việt                |
| `src/data/transforms.py`   | `src/data/transforms.py`   | Giữ nguyên                                       |
| `src/training/trainer.py`  | `src/training/trainer.py`  | Giữ nguyên                                       |
| `src/utils/postprocess.py` | `src/utils/postprocess.py` | Giữ nguyên                                       |
| `src/utils/common.py`      | `src/utils/common.py`      | Giữ nguyên                                       |
| `configs/config.py`        | `configs/config.py`        | Chỉ CRNN, path → `dataset/`                      |
| `train.py`                 | `train.py`                 | Bỏ ResTran, đơn giản hóa                         |
| `src/models/restran.py`    | ❌ Không trích             | Baseline 2                                       |
| `run_ablation.py`          | `run_ablation.py`          | Chỉ 2 exp CRNN (bỏ ResTran)                      |

Chi tiết: xem `summary/extraction_guide.md`
