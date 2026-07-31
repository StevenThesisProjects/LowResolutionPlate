# MultiFrame-LPR

Multi-frame OCR solution for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition**.

This implementation uses **ResTranOCR** (ResNet34 + Transformer) with optional **STN** alignment, combining temporal information from 5 video frames via attention fusion.

🔗 **Challenge:** [ICPR 2026 LRLPR](https://icpr26lrlpr.github.io/)

---

## Quick Start

```bash
# Install dependencies
uv sync

# Train with default settings (ResTranOCR + STN)
python train.py

# Train without STN (ablation)
python train.py --no-stn --experiment-name restran34_no_stn

# Generate submission file
python train.py --submission-mode
```

---

## Key Features

- **ResTranOCR**: ResNet34 backbone + Transformer encoder + CTC decoding
- **Spatial Transformer Network**: Optional STN for automatic image alignment
- **Multi-Frame Fusion**: Attention-based fusion across 5-frame sequences
- **Smart Data Augmentation**: Scenario-B aware validation split
- **Production Ready**: Mixed precision training, gradient clipping, OneCycleLR scheduler

---

## Model Architecture

**Pipeline:** Multi-frame Input → STN Alignment → ResNet34 → Attention Fusion → Transformer → CTC

**Input shape:** `(Batch, 5, 3, 32, 128)` — 5 frames, 3 channels, 32×128 pixels.

---

## Installation

**Requirements:**
- Python 3.11+
- CUDA-enabled GPU (recommended)

**Using uv (recommended):**
```bash
git clone https://github.com/duongtruongbinh/MultiFrame-LPR.git
cd MultiFrame-LPR
uv sync
```

**Using pip:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Usage

### Data Preparation

```
dataset/data/train/
├── track_001/
│   ├── lr-001.png ... lr-005.png
│   ├── hr-001.png ... (optional, for synthetic LR)
│   └── annotations.json
└── ...
```

**annotations.json format:**
```json
{"plate_text": "ABC1234"}
```

### Training

```bash
python train.py \
    --experiment-name restran34_with_stn \
    --data-root dataset/data/train \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.0005 \
    --aug-level full
```

**Key arguments:**
- `-n, --experiment-name`: Experiment identifier
- `--data-root`: Path to training data
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Training epochs
- `--lr`: Learning rate (default: 5e-4)
- `--aug-level`: `full` or `light`
- `--no-stn`: Disable Spatial Transformer Network
- `--submission-mode`: Train on full dataset and generate test predictions
- `--output-dir`: Output directory (default: `results/`)

### STN Ablation

```bash
python run_ablation.py
```

Compares ResTranOCR with and without STN. Results saved in `experiments/ablation_summary.txt`.

### Inference (from checkpoint)

Put your trained model in `results/` (or any path), then run:

```bash
# Test on public test set (no labels) → submission file
python predict.py --checkpoint results/restran34_with_stn_best.pth --mode test

# Validate on val split (with labels) → accuracy + submission
python predict.py --checkpoint results/restran34_with_stn_best.pth --mode val

# If model was trained WITHOUT STN
python predict.py --checkpoint results/restran34_no_stn_best.pth --mode test --no-stn
```

### Outputs

- `{experiment_name}_best.pth` — Best model checkpoint
- `submission_{experiment_name}.txt` — Predictions: `track_id,predicted_text;confidence`

---

## Configuration

Key hyperparameters in `configs/config.py`:

```python
EXPERIMENT_NAME = "restran34_with_stn"
USE_STN = True
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 30
AUGMENTATION_LEVEL = "full"

TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 3
TRANSFORMER_FF_DIM = 2048
TRANSFORMER_DROPOUT = 0.1
```

---

## Project Structure

```
.
├── configs/config.py
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── restran.py         # ResTranOCR model
│   │   └── components.py      # STN, AttentionFusion, ResNet34, PositionalEncoding
│   ├── training/trainer.py
│   └── utils/
├── train.py
├── run_ablation.py
└── pyproject.toml
```
