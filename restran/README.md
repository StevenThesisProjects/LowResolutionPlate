# ResTran Baseline — ICPR 2026 LRLPR

Baseline **ResTranOCR** (ResNet34 + Transformer + CTC) cho **ICPR 2026 Challenge on Low-Resolution License Plate Recognition**.

Folder này chỉ chứa các thành phần phục vụ ResTran, tách ra từ repo `MultiFrame-LPR` (vốn có 2 nhánh model). Mọi thứ không thuộc đường đi của ResTran đã được loại bỏ.

🔗 **Challenge:** https://icpr26lrlpr.github.io/

---

## Kiến trúc

```
Input [B, 5, 3, 32, 128]
  → STN Block (affine, khởi tạo identity)          [B*5, 2, 3]
  → ResNet34 (weight sharing, layer3/4 stride 2,1) [B*5, 512, 1, 16]
  → Attention Fusion (softmax trên trục 5 frame)   [B, 512, 1, 16]
  → Positional Encoding + Transformer Encoder ×3   [B, 16, 512]
  → Linear(512 → 37) + log_softmax                 [B, 16, 37]
  → CTC (blank = 0)                                → "BAI8068", 0.95
```

| | Tham số |
|---|---|
| STN Block | 0,28 M |
| ResNet34 backbone | 21,28 M |
| Attention Fusion | 0,03 M |
| Transformer (3 lớp) | 9,46 M |
| FC head | 0,02 M |
| **Tổng (có STN)** | **31,08 M** |
| Tổng (`--no-stn`) | 30,79 M |

Chuỗi CTC có **T = 16 timestep** cho nhãn 7 ký tự; `NUM_CLASSES = 37` (36 ký tự + blank).

Phân tích chi tiết: [documents/baseline/restran_baseline.md](documents/baseline/restran_baseline.md) và [documents/baseline/dataset.md](documents/baseline/dataset.md).

---

## Cài đặt

```bash
uv sync
# hoặc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install albumentations opencv-python numpy tqdm
```

Yêu cầu: Python 3.11, GPU CUDA (khuyến nghị).

---

## Dữ liệu

`dataset/` là **bản copy độc lập** (1,2 GB, 240.013 file) — folder này chạy được mà không phụ thuộc repo `MultiFrame-LPR`:

```
dataset/
├── data/train/
│   ├── Scenario-A/{Mercosur,Brazilian}/track_XXXXX/   (10.000 track, .png, có corners)
│   └── Scenario-B/{Mercosur,Brazilian}/track_XXXXX/   (10.000 track, .jpg)
├── Pa7a3Hin-test-public/Pa7a3Hin-test-public/         (1.000 track, chỉ lr-*)
└── TKzFBtn7-test-blind/TKzFBtn7-test-blind/           (3.000 track, chỉ lr-*)
```

Mỗi track train chứa `annotations.json` + `lr-001..005` + `hr-001..005`.

Đường dẫn mặc định trong [configs/config.py](configs/config.py) **đã trỏ đúng** vào thư mục này nên chạy được ngay, không cần truyền `--data-root`.

`dataset/` được `.gitignore` bỏ qua để không đẩy 1,2 GB lên git.

---

## Sử dụng

```bash
# Train mặc định (ResTran + STN)
python train.py

# Ablation: tắt STN
python train.py --no-stn -n restran_nostn

# Train toàn bộ dữ liệu + sinh file dự đoán cho public test
python train.py --submission-mode -n restran_sub
```

**Các tham số chính:**

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `-n, --experiment-name` | `restran` | Tên experiment cho checkpoint/submission |
| `--epochs` | 30 | Số epoch |
| `--batch-size` | 64 | Batch size |
| `--lr` | 5e-4 | Learning rate (`OneCycleLR` max_lr) |
| `--data-root` | `dataset/data/train` | Thư mục dữ liệu train |
| `--seed` | 42 | Random seed |
| `--num-workers` | 10 | Số worker của DataLoader |
| `--transformer-heads` | 8 | Số attention head |
| `--transformer-layers` | 3 | Số lớp Transformer encoder |
| `--aug-level` | `full` | `full` hoặc `light` |
| `--no-stn` | tắt | Bỏ Spatial Transformer Network |
| `--submission-mode` | tắt | Train toàn bộ dữ liệu, không val, rồi infer test |
| `--output-dir` | `results` | Nơi lưu checkpoint và submission |

---

## Chia dữ liệu

Validation **chỉ lấy từ Scenario-B** (cùng phân phối với test set), qua `SPLIT_RATIO` trong config:

| `SPLIT_RATIO` | Train | Val | Mẫu train |
|---|---|---|---|
| `0.9` (mặc định) | 19.001 track | 999 track | 38.002 |
| `0.8` (theo slide baseline gốc) | 18.001 track | 1.999 track | 36.002 |

> Lệch 1 track so với con số tròn là do sai số dấu phẩy động: `1 − 0.9 = 0.09999999999999998`, nên `int(10000 × 0.0999…) = 999` chứ không phải 1000 ([dataset.py:148](src/data/dataset.py#L148)). Ảnh hưởng không đáng kể, giữ nguyên để trùng hành vi baseline gốc.

Mỗi track train sinh **2 mẫu**: LR thật + HR đã degrade thành LR tổng hợp. Split được lưu ở `splits/val_tracks.json` để tái sử dụng; xoá file này để tạo split mới.

---

## Đầu ra

Trong `--output-dir` (mặc định `results/`):

- `{experiment_name}_best.pth` — checkpoint tốt nhất theo val accuracy
- `submission_{experiment_name}.txt` — dự đoán trên val, định dạng `track_id,plate_text;confidence`
- `submission_{experiment_name}_final.txt` — dự đoán trên test (chỉ ở `--submission-mode`)

> ⚠️ **Trước khi nộp Codabench:** đổi tên file thành **`predictions.txt`** rồi zip **trực tiếp file** (không zip thư mục):
> ```bash
> cp results/submission_restran_sub_final.txt predictions.txt
> zip submission.zip predictions.txt
> ```
> Giới hạn: 5 lượt nộp/ngày, tối đa 25 lượt toàn giải.

---

## Metric

- **Recognition Rate** (xếp hạng chính): tỉ lệ track dự đoán **khớp tuyệt đối** toàn bộ ký tự.
- **Confidence Gap** (tie-break): `mean confidence dự đoán đúng − mean confidence dự đoán sai`, càng lớn xếp hạng càng cao.

Trainer hiện chỉ log exact-match accuracy trên val.

**Kết quả baseline tham chiếu** (val Scenario-B, theo slide gốc):

| Cấu hình | Accuracy |
|---|---|
| ResTran | 75,80% |
| ResTran + STN | **78,70%** |

---

## Cấu trúc project

```
.
├── configs/config.py          # Config dataclass (chỉ ResTran)
├── src/
│   ├── data/
│   │   ├── dataset.py         # MultiFrameDataset, split ưu tiên Scenario-B
│   │   └── transforms.py      # Augmentation + degradation HR→LR
│   ├── models/
│   │   ├── restran.py         # ResTranOCR
│   │   └── components.py      # STNBlock, AttentionFusion, ResNetFeatureExtractor, PositionalEncoding
│   ├── training/trainer.py    # Training loop, validation, inference
│   └── utils/
│       ├── common.py          # seed_everything
│       └── postprocess.py     # Greedy CTC decode + confidence
├── documents/baseline/        # dataset.md, restran_baseline.md + slide PDF gốc
├── splits/val_tracks.json     # Split val tất định (xoá để tạo lại)
├── dataset/                   # Dữ liệu train + test (bản copy độc lập, 1,2 GB)
└── train.py
```
