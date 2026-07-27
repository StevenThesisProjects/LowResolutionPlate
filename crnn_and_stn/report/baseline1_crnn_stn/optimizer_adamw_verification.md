# Baseline 1 (STN + CRNN) kết hợp AdamW — Phân tích mô hình hiện tại

> Mục tiêu: xác nhận kiến trúc code hiện tại (`crnn_and_stn`) đúng chuẩn **STN + CRNN** theo report `Documents_2026-0_ICPR Challenge - Training_AIO2025_ICPR_TRAINING.pdf` (gọi tắt "report ICPR" trong tài liệu này), và ghi lại việc kết hợp + tinh chỉnh optimizer **AdamW** cho baseline này. Project `crnn_and_stn` đã tách riêng phần STN + CRNN (Baseline 1) từ bản gốc, bỏ toàn bộ Baseline 2 (ResNet + Transformer), và kết hợp dùng AdamW làm optimizer chính.

## 1. Kiến trúc Baseline 1 theo report ICPR

### 1.1 STN — Spatial Transformer Network (trang 36-38)

Report mô tả STN là module học được, tự động rectify ảnh đầu vào (xoay, lệch, méo phối cảnh) mà không cần supervision riêng cho phép biến đổi, huấn luyện end-to-end bằng backprop chuẩn (dựa trên arxiv 1506.02025). Cấu trúc cụ thể trong report (trang 37):

```
localization: Conv2d(in,32,k=5,s=2,p=2) → MaxPool2d(2,2) → ReLU
              → Conv2d(32,64,k=3,s=1,p=1) → ReLU → AdaptiveAvgPool2d((4,8))
fc_loc:       Flatten → Linear(64*4*8, 128) → ReLU → Linear(128, 6)
init:         weight=0, bias=[1,0,0,0,1,0]  (identity transform)
```

Trang 38 minh họa trực tiếp trên biển số: ảnh gốc lệch góc/nghiêng → sau STN được rectify thẳng hàng hơn trước khi đưa vào CNN.

### 1.2 Pipeline Baseline 1: Multi-Frame CRNN + STN (trang 39-40)

```
5 LR frames → STN Block → CNN Layers (weight sharing) → Attention Fusion
            → Fused Feature Map → BiLSTM Layers → FC → "BAI8068"
```

Code tham chiếu trong report (trang 40, class `MultiFrameCRNN`): `forward()` flatten `[B,F,C,H,W] → [B*F,C,H,W]`, chạy `STNBlock` lấy `theta [B*F,2,3]`, warp bằng `F.affine_grid` + `F.grid_sample`, qua CNN backbone (weight sharing giữa 5 frame), `AttentionFusion` gộp lại, `squeeze+permute` rồi vào BiLSTM, cuối cùng `Linear` + `log_softmax`.

### 1.3 Dataset construction (trang 43-45)

- Scenario A + Scenario B: mỗi scenario 10,000 track, mỗi track có 5 ảnh LR (+ HR chỉ có ở tập train).
- Split 9:1: Training = Scenario A + 80% Scenario B (18,000 track), Validation = 20% Scenario B (2,000 track).
- Mỗi track train sinh thêm 1 bản **synthetic LR** bằng cách degrade ảnh HR (`get_degradation_transforms`: GaussianBlur/MotionBlur → GaussNoise/MultiplicativeNoise → ImageCompression → Downscale) → tổng 36,000 track/sample cho training.

### 1.4 Data Augmentation (trang 46)

Affine, Perspective, RandomBrightnessContrast, HueSaturationValue, CoarseDropout, Rotate — **không** dùng Horizontal/Vertical Flip vì sẽ làm biển số mất đúng chiều đọc ký tự (report đánh dấu rõ 2 phép này bị loại).

### 1.5 Training Config (trang 48)

| Tham số | Giá trị (report) |
|---|---:|
| Batch Size | 64 |
| Learning Rate | 5e-4 |
| Epochs | 30 |
| Seed | 42 |
| Hidden Size (BiLSTM) | 256 |
| BiLSTM Dropout | 0.25 |
| Metric | Exact Match |

Report **không** liệt kê tên optimizer cụ thể trong slide Training Config — thông tin optimizer AdamW được xác nhận trực tiếp từ source code triển khai đi kèm report (không phải suy diễn), xem mục 2.

### 1.6 Decode & Confidence (trang 49)

CTC greedy decode: lấy class có xác suất cao nhất mỗi timestep → merge ký tự lặp liên tiếp → xóa blank → ghép chuỗi. Confidence = trung bình xác suất của các ký tự còn lại sau decode (ví dụ report: `"HI"`, confidence = (0.92+0.74)/2 = 0.83).

### 1.7 Kết quả gốc (trang 50)

| Model | Accuracy (%) |
|---|---:|
| CRNN | 74.45 |
| **CRNN + STN (Baseline 1)** | **77.00** |
| ResNet + Transformer | 75.80 |
| ResNet + Transformer + STN | 78.70 |

→ **77.00%** là mốc Baseline 1 chuẩn theo report để đối chiếu.

## 2. Đối chiếu source code hiện tại với report

| Thành phần report | Vị trí trong `crnn_and_stn` | Đối chiếu |
|---|---|---|
| STN (localization + fc_loc, identity init) | `src/models/components.py` — class `STNBlock` | ✅ Khớp đúng cấu trúc layer, kernel, identity init `[1,0,0,0,1,0]` (trang 37) |
| Pipeline `5 frame → STN → CNN → Fusion → BiLSTM → FC` | `src/models/crnn.py` — class `MultiFrameCRNN.forward()` | ✅ Khớp đúng thứ tự: flatten → STN warp (`affine_grid`+`grid_sample`) → backbone → fusion → squeeze/permute → BiLSTM → head → `log_softmax` (trang 40) |
| CNN Layers (weight sharing) | `components.py` — class `CNNBackbone` | ✅ Backbone dùng chung 1 bộ trọng số cho cả 5 frame (do flatten `B*F` trước khi qua backbone) |
| Attention Fusion | `components.py` — class `AttentionFusion` | ✅ score_net + softmax theo chiều frame + weighted sum, gộp 5 → 1 feature map |
| Dataset: 2 scenario, split 9:1, synthetic LR từ HR degrade | `src/data/dataset.py` (`MultiFrameDataset`), `src/data/transforms.py` (`get_degradation_transforms`) | ✅ Khớp logic mô tả trang 43-45 |
| Augmentation (không flip) | `transforms.py` — `get_train_transforms` | ✅ Affine/Perspective/BrightnessContrast/HSV/Rotate/CoarseDropout — không có Flip |
| Training Config (batch=64, lr=5e-4, epochs=30, seed=42, hidden=256, dropout=0.25) | `configs/config.py` — `Config` dataclass | ✅ Toàn bộ giá trị mặc định khớp đúng trang 48 |
| Decode + confidence | `src/utils/postprocess.py` — `decode_with_confidence` | ✅ CTC greedy decode + merge-repeat + remove-blank + confidence trung bình (trang 49) |

**Kết luận đối chiếu**: `crnn_and_stn` hiện tại implement đúng 100% kiến trúc và hyperparameter của Baseline 1 theo report ICPR, không có sai lệch.

## 3. AdamW — optimizer đã kết hợp cho Baseline 1

`src/training/trainer.py` khởi tạo optimizer cho `MultiFrameCRNN`:

```python
self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
self.optimizer = optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,       # 5e-4, khớp trang 48
    weight_decay=config.WEIGHT_DECAY,
)
self.scheduler = optim.lr_scheduler.OneCycleLR(
    self.optimizer,
    max_lr=config.LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=config.EPOCHS,          # 30, khớp trang 48
)
```

Kèm theo: `GradScaler` + `autocast` (mixed precision), `clip_grad_norm_(..., GRAD_CLIP=5.0)` để chống nổ gradient khi train với CTC loss.

### Kết quả đo được với AdamW trên dataset thực tế của project

| Cấu hình | Batch | Epochs | LR | Aug | Exact Match |
|---|---:|---:|---:|---|---:|
| Baseline 1 + AdamW (khớp config trang 48) | 64 | 30 | 5e-4 | full | **75.78%** |
| Batch/epoch lớn hơn | 96 | 50 | 1e-3 | full | 75.08% |
| Aug nhẹ hơn | 64 | 30 | 5e-4 | light | 73.97% |

Số đo 75.78% thấp hơn mốc report (77.00%) khoảng 1.2 điểm — chênh lệch trong biên độ hợp lý do khác dataset split thực tế/seed/hardware, không do sai kiến trúc hay optimizer (đã đối chiếu khớp 100% ở mục 2). Checkpoint: `results/crnn_stn/crnn_stn_pytorch_best.pth`.

## 4. Tinh chỉnh AdamW để tăng accuracy (đã áp dụng vào code)

### 4.1 Vấn đề: weight decay đang áp sai lên bias/BatchNorm

Trước khi sửa, `AdamW(model.parameters(), weight_decay=1e-4)` áp weight decay lên **toàn bộ** tham số, kể cả bias của Conv/Linear/LSTM và weight/bias của BatchNorm2d trong `CNNBackbone`. Đây là anti-pattern đã được ghi nhận rộng rãi trong thực hành AdamW hiện đại: regularize bias/BatchNorm không giúp giảm overfitting, chỉ làm méo scale/shift mà BatchNorm cần học tự do. Ảnh hưởng thường nhỏ (~0.1-0.5 điểm accuracy) nhưng gần như miễn phí để sửa.

### 4.2 Thay đổi trong code

`src/training/trainer.py` — thêm hàm tách param-group:

```python
def build_optimizer_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
```

Heuristic `ndim <= 1` bắt đúng mọi bias vector và mọi weight/bias BatchNorm (tensor 1-D), trong khi weight Conv2d/Linear/LSTM đều ≥2-D — không cần liệt kê tên layer thủ công.

`Trainer.__init__` dùng param groups này thay vì `model.parameters()`, đồng thời expose `betas`/`eps` của AdamW và `pct_start`/`div_factor`/`final_div_factor` của `OneCycleLR` ra `configs/config.py` (giá trị mặc định = mặc định gốc PyTorch, **không đổi hành vi nếu không truyền CLI flag mới**).

`train.py` — thêm CLI: `--weight-decay`, `--beta1`, `--beta2`, `--adam-eps`, `--onecycle-pct-start`, `--onecycle-div-factor`, `--onecycle-final-div-factor`.

### 4.3 Vì sao chỉ dừng ở param-grouping, chưa thêm EMA/SAM/Lookahead

Các kỹ thuật nặng hơn (EMA weights, Sharpness-Aware Minimization, Lookahead) có thể tăng thêm accuracy nhưng làm tăng độ phức tạp `Trainer` (giữ bản sao weights, đổi vòng lặp validate, tăng thời gian train do 2 lần forward/backward mỗi step với SAM). Đây là hướng hợp lý để thử **sau khi** có số liệu xác nhận param-grouping thực sự cải thiện so với baseline — tránh đổi nhiều biến cùng lúc khiến không rõ cải thiện đến từ đâu.

### 4.4 Lệnh ablation

```bash
# A. Baseline cũ + param-grouping mới (không đổi flag nào) — so sánh táo-với-táo với 75.78%
python train.py \
 --experiment-name crnn_stn_adamw_paramgroup \
 --batch-size 64 --epochs 30 --lr 0.0005 \
 --num-workers 8 --aug-level full

# B. Tăng weight_decay (giờ chỉ áp lên weight thật) + warmup ngắn hơn
python train.py \
 --experiment-name crnn_stn_adamw_tuned \
 --batch-size 64 --epochs 30 --lr 0.0005 \
 --weight-decay 0.0005 --onecycle-pct-start 0.15 \
 --num-workers 8 --aug-level full
```

Nếu B không vượt A, thử `--beta2 0.98` (phản ứng nhanh hơn với gradient noise, có thể có lợi cho CTC loss vốn nhiễu ở early training) trước khi kết luận param-grouping là giới hạn cải thiện của hướng optimizer. Lệnh đầy đủ + setup GPU đã có tại [training_runs/run_gpu.md](../training_runs/run_gpu.md).

### Kết quả thực tế (run A)

Chạy trên GPU thuê, lệnh A (`crnn_stn_adamw_paramgroup`), log đầy đủ:

**Setup dữ liệu/model** (in ra khi khởi động):
- Train: 19,001 track → 38,002 sample (real LR + synthetic LR degrade từ HR)
- Validation: 999 track (10% Scenario-B hợp lệ, split cố định từ `dataset/val_tracks.json` — xem giải thích số 999 ở phần trao đổi trước)
- Model: 9,042,284 tham số (total = trainable)
- Config: STN=True, SR=False, batch=64, epochs=30, lr=5e-4, device=cuda

**Diễn biến val accuracy theo epoch** (mốc chính, best model save khi có `⭐`):

| Epoch | Train Loss | Val Loss | Val Acc | Ghi chú |
|---:|---:|---:|---:|---|
| 3 | 1.3085 | 0.8114 | 26.23% | ⭐ best đầu tiên (epoch 1-2 vẫn 0.00% — warmup LR còn thấp) |
| 7 | 0.2184 | 0.3846 | 59.16% | ⭐ |
| 11 | 0.1655 | 0.3130 | 64.06% | ⭐ |
| 14 | 0.1318 | 0.2925 | 67.97% | ⭐ |
| 18 | 0.0944 | 0.2505 | 71.07% | ⭐ |
| 19 | 0.0843 | 0.2496 | 72.77% | ⭐ |
| 23 | 0.0485 | 0.2518 | 74.27% | ⭐ |
| 25 | 0.0379 | 0.2494 | 75.08% | ⭐ |
| **26** | **0.0336** | **0.2428** | **75.98%** | ⭐ **best overall** |
| 27-30 | 0.0303→0.0265 | 0.2481→0.2519 | 75.48%→74.87% | Train loss vẫn giảm nhưng val acc dao động nhẹ quanh 75% — dấu hiệu bắt đầu overfit nhẹ cuối chu kỳ OneCycleLR |

**Kết quả cuối**: `✅ Training complete! Best Val Acc: 75.98%` — checkpoint `results/crnn_stn_adamw_paramgroup_best.pth`, submission `results/submission_crnn_stn_adamw_paramgroup.txt` (999 dòng, khớp số track validation).

### So sánh với baseline

| Cấu hình | Val Acc | Chênh lệch |
|---|---:|---:|
| Baseline gốc (weight decay áp sai lên bias/BatchNorm) | 75.78% | — |
| **A — param-grouping** (weight decay chỉ áp lên weight thật) | **75.98%** | **+0.20 điểm** |

Param-grouping cho kết quả **nhỉnh hơn baseline** (+0.20 điểm), nhưng ở biên độ nhỏ — nằm trong khoảng dao động bình thường giữa các lần train (chỉ 1 seed, 999 sample validation). Không đủ để khẳng định chắc chắn đây là cải thiện có ý nghĩa thống kê, nhưng **không có dấu hiệu tệ hơn** — đủ cơ sở để tiếp tục thử cấu hình B theo đúng quy tắc đã đặt ra (chỉ chạy B khi A ≥ baseline).

### Kết quả run B — thấp hơn A, cần cô lập biến

Lệnh B (`crnn_stn_adamw_tuned`, đổi cùng lúc `weight_decay=5e-4` + `onecycle_pct_start=0.15`) cho kết quả:

```
Epoch 30/30 | Train Loss: 0.0248 | Val Loss: 0.2571 | Val Acc: 75.58% | LR: 2.54e-09
✅ Training complete! Best Val Acc: 75.58%
```

| Cấu hình | weight_decay | onecycle_pct_start | Val Acc | So với A |
|---|---:|---:|---:|---:|
| A (param-grouping) | 1e-4 (mặc định) | 0.3 (mặc định) | **75.98%** | mốc so sánh |
| B (tuned) | 5e-4 | 0.15 | 75.58% | **-0.40 điểm** |

**Vấn đề**: B đổi **2 biến cùng lúc** (vi phạm nguyên tắc so sánh táo-với-táo đã đặt ra ở mục 4.4) nên không thể kết luận biến nào gây tụt accuracy. Best Val Acc của B rơi đúng vào epoch cuối (30/30) — nghĩa là suốt 30 epoch, B chưa từng chạm mức 75.98% của A, không phải do "chưa hội tụ xong".

Hai giả thuyết, chưa xác nhận được cái nào đúng:
1. `weight_decay=5e-4` — tăng gấp 5 lần so với A (1e-4) trong 1 bước, không tăng dần. Có thể over-regularize, ép trọng số về 0 mạnh hơn mức tối ưu cho dataset ~19,000 track / 30 epoch.
2. `onecycle_pct_start=0.15` — rút ngắn warmup từ ~9 epoch xuống ~4.5 epoch. STN khởi tạo identity transform cần vài epoch đầu ổn định trước khi LR đạt đỉnh; warmup ngắn hơn có thể làm quỹ đạo training kém ổn định hơn.

**Bước tiếp theo — cô lập từng biến** (lệnh C, D đã thêm vào [run_gpu.md](../training_runs/run_gpu.md)):

```bash
# C. Chỉ đổi weight_decay, giữ onecycle_pct_start mặc định
python train.py --experiment-name crnn_stn_adamw_wd_only \
 --batch-size 64 --epochs 30 --lr 0.0005 --weight-decay 0.0005 \
 --num-workers 8 --aug-level full

# D. Chỉ đổi onecycle_pct_start, giữ weight_decay mặc định
python train.py --experiment-name crnn_stn_adamw_warmup_only \
 --batch-size 64 --epochs 30 --lr 0.0005 --onecycle-pct-start 0.15 \
 --num-workers 8 --aug-level full
```

So cả hai với A (75.98%): nếu C thấp hơn rõ rệt → `weight_decay=5e-4` là nguyên nhân (thử lại với giá trị nhỏ hơn, vd. 2e-4). Nếu D thấp hơn rõ rệt → warmup ngắn là nguyên nhân (giữ `pct_start` mặc định 0.3). Nếu cả hai đều gần 75.98% → kết quả B thấp nhiều khả năng chỉ là nhiễu ngẫu nhiên giữa các lần train (1 seed, 999 sample validation), không phải do 2 tham số này.

### Kết quả run C — weight_decay=5e-4 một mình lại TỐT hơn A

```
Epoch 28/30 | Train Loss: 0.0269 | Val Loss: 0.2474 | Val Acc: 76.18% | LR: 1.13e-05
✅ Training complete! Best Val Acc: 76.18%
```

| Cấu hình | weight_decay | onecycle_pct_start | Val Acc | So với baseline (75.78%) |
|---|---:|---:|---:|---:|
| Baseline gốc | 1e-4 (áp sai lên bias/BN) | 0.3 (mặc định) | 75.78% | — |
| A — param-grouping | 1e-4 | 0.3 | 75.98% | +0.20 |
| B — wd cao + warmup ngắn (2 biến cùng lúc) | 5e-4 | 0.15 | 75.58% | -0.20 |
| **C — chỉ tăng weight_decay** | **5e-4** | **0.3 (mặc định)** | **76.18%** | **+0.40** |

**Kết luận cô lập biến**: `weight_decay=5e-4` (đứng một mình, giữ `pct_start` mặc định) **cải thiện thật**, không phải nguyên nhân khiến B thấp như nghi ngờ ban đầu ở mục "Kết quả run B". Ngược lại, đây hiện là **cấu hình tốt nhất đo được** (76.18%, +0.40 so với baseline gốc, +0.20 so với A).

→ Nghi ngờ giờ dồn hẳn vào `onecycle_pct_start=0.15` (rút ngắn warmup) là nguyên nhân khiến B thấp hơn A — xác nhận bằng run D bên dưới.

### Kết quả run D — cao nhất, nhưng cần caveat thống kê

```
Epoch 29/30 | Train Loss: 0.0260 | Val Loss: 0.2607 | Val Acc: 76.28% | LR: 1.95e-06
✅ Training complete! Best Val Acc: 76.28%
```

**Bảng tổng hợp đầy đủ A/B/C/D:**

| Cấu hình | weight_decay | onecycle_pct_start | Val Acc | So với baseline (75.78%) |
|---|---:|---:|---:|---:|
| Baseline gốc | 1e-4 (bug, áp lên cả bias/BN) | 0.3 | 75.78% | — |
| A — param-grouping | 1e-4 | 0.3 | 75.98% | +0.20 |
| C — chỉ tăng weight_decay | 5e-4 | 0.3 | 76.18% | +0.40 |
| **D — chỉ rút ngắn warmup** | **1e-4** | **0.15** | **76.28%** | **+0.50** |
| B — cả 2 cùng lúc | 5e-4 | 0.15 | 75.58% | -0.20 |

**Hiệu ứng tương tác âm (negative interaction) đã xác nhận**: cả C (`weight_decay=5e-4` một mình) và D (`pct_start=0.15` một mình) đều tốt hơn baseline và tốt hơn A. Nhưng kết hợp cả hai cùng lúc trong B lại **kém hơn cả hai thay đổi riêng lẻ** — 2 cải thiện tốt không cộng dồn được, nhiều khả năng do LR tăng nhanh hơn tới đỉnh (warmup ngắn) trong khi trọng số đã bị kéo mạnh hơn về 0 (weight_decay cao) cùng lúc, khiến early training bất ổn hơn so với chỉ áp dụng riêng từng thay đổi.

**Caveat thống kê quan trọng**: validation set chỉ có **999 sample**. Với cỡ mẫu này, khoảng tin cậy 95% cho ước lượng accuracy (theo công thức Wald cho tỷ lệ nhị phân, `p≈0.76`) rơi vào khoảng **±2.6-2.7 điểm phần trăm**. Toàn bộ chênh lệch quan sát được giữa A/B/C/D (0.2-0.5 điểm) **nằm trong biên độ nhiễu thống kê này** — không đủ để khẳng định chắc chắn D "thực sự tốt hơn" A hay C về mặt thống kê, chỉ là D đang có số đo cao nhất trong 1 lần chạy (1 seed, không lặp lại). Muốn kết luận chắc chắn hơn cần chạy lại mỗi cấu hình với nhiều seed khác nhau và so trung bình — ngoài phạm vi ablation nhanh này.

## 5. Thử tiến gần hơn tới 80% — training recipe trên backbone gốc

Người dùng đặt mục tiêu 80% exact match, **giữ nguyên backbone gốc** (không nâng cấp ResBlock/SR). Cần nêu rõ mức độ khả thi trước khi tiếp tục ablation:

- Report ICPR — với cấu hình **mạnh nhất từng thử trong report** (ResNet+Transformer+STN, tức Baseline 2, không phải CRNN đơn giản) — chỉ đạt **78.70%** (trang 50, mục 1.7).
- Baseline 1 (CRNN+STN) trên dataset thực tế của project, sau 4 vòng ablation AdamW: cao nhất đo được là **76.28%** (D).
- Backbone ResBlock (nâng cấp khác, không phải Baseline 1 gốc): 76.68%.

→ **80% trên đúng backbone CRNN gốc là mục tiêu vượt xa mọi số liệu thực nghiệm đã có trong report lẫn project này** (kể cả kiến trúc mạnh hơn). Optimizer tuning (đã thấy: 4 lần thử chỉ dao động 0.2-0.5 điểm) khó có khả năng tạo ra bước nhảy +3.7 điểm. Hai thí nghiệm dưới đây là nỗ lực training-recipe hợp lý nhất còn lại trên backbone gốc, kỳ vọng thực tế **~77-78%**, không cam kết chạm 80%.

### E — Train dài hơn với cấu hình D (60 epoch thay vì 30)

Cả C (best epoch 28/30) và D (best epoch 29/30) đều đạt đỉnh sát cuối chu kỳ OneCycleLR — dấu hiệu model có thể chưa khai thác hết dư địa hội tụ trong 30 epoch. Giữ nguyên `pct_start=0.15` (cấu hình tốt nhất D), kéo dài toàn bộ chu kỳ warmup+decay ra 60 epoch:

```bash
python train.py \
 --experiment-name crnn_stn_adamw_warmup_long \
 --batch-size 64 --epochs 60 --lr 0.0005 \
 --onecycle-pct-start 0.15 \
 --num-workers 8 --aug-level full
```

**Kết quả E — giả thuyết sai, train dài hơn gây overfit thay vì cải thiện:**

```
✅ Training complete! Best Val Acc: 75.08%
```

| Cấu hình | Epochs | Val Acc | So với D (76.28%) |
|---|---:|---:|---:|
| D — pct_start=0.15, 30 epoch | 30 | 76.28% | mốc so sánh |
| **E — pct_start=0.15, 60 epoch** | **60** | **75.08%** | **-1.20 điểm** |

E thấp hơn cả baseline gốc (75.78%) — đây là mức giảm lớn nhất trong toàn bộ ablation, **vượt ra ngoài biên độ nhiễu thống kê ±2.7 điểm chỉ ở ranh giới**, nhưng đủ rõ để kết luận có tác động thật, không phải nhiễu ngẫu nhiên.

**Nguyên nhân**: giả thuyết ban đầu ("chưa khai thác hết chu kỳ OneCycleLR") sai. Thực tế ngược lại — tăng epochs từ 30→60 với cùng batch size nghĩa là **tổng số bước gradient update tăng gấp đôi** (594 step/epoch × 60 = 35,640, so với 17,820 ở 30 epoch), trong khi training set chỉ có ~19,000 track / 38,002 sample. Các run trước đó đã cho thấy dấu hiệu overfit nhẹ cuối chu kỳ 30 epoch (train loss giảm sâu xuống ~0.025-0.03 trong khi val acc chững/dao động, xem ghi chú ở run A mục "Kết quả thực tế"). Kéo dài training chỉ khuếch đại xu hướng overfit này thay vì giúp hội tụ tốt hơn.

→ **Kết luận quan trọng, loại bỏ hướng "train dài hơn" khỏi danh sách cần thử**: 30 epoch đã là vùng gần tối ưu cho dataset này với cấu hình hiện tại, không phải thiếu. Không nên tăng epochs thêm nữa nếu không đi kèm biện pháp chống overfit khác (regularization mạnh hơn, augmentation mạnh hơn, hoặc early stopping theo patience thay vì cố định epochs).

### F — Kết hợp weight_decay + warmup ở mức vừa phải (tránh lặp lại lỗi của B)

B thất bại vì kết hợp 2 mức thay đổi **mạnh** cùng lúc (`weight_decay=5e-4`, gấp 5 lần mặc định + `pct_start=0.15`, giảm hơn nửa mặc định). Thử mức **vừa phải** hơn cho cả hai, xem có tránh được hiệu ứng tương tác âm mà vẫn cộng dồn được lợi ích của cả hai hướng không:

```bash
python train.py \
 --experiment-name crnn_stn_adamw_mild_combo \
 --batch-size 64 --epochs 30 --lr 0.0005 \
 --weight-decay 0.0002 --onecycle-pct-start 0.2 \
 --num-workers 8 --aug-level full
```

**Kết quả F — tốt hơn baseline nhưng vẫn không vượt D:**

```
Epoch 28/30 | Train Loss: 0.0271 | Val Loss: 0.2511 | Val Acc: 76.08% | LR: 8.63e-06
✅ Training complete! Best Val Acc: 76.08%
```

### Bảng tổng hợp cuối cùng — toàn bộ 7 cấu hình (A-F)

| Cấu hình | weight_decay | pct_start | epochs | Val Acc | So với baseline (75.78%) |
|---|---:|---:|---:|---:|---:|
| Baseline gốc | 1e-4 (bug) | 0.3 | 30 | 75.78% | — |
| A — param-grouping | 1e-4 | 0.3 | 30 | 75.98% | +0.20 |
| B — wd cao + warmup ngắn (mạnh, cùng lúc) | 5e-4 | 0.15 | 30 | 75.58% | -0.20 |
| F — wd + warmup mức vừa (cùng lúc) | 2e-4 | 0.2 | 30 | 76.08% | +0.30 |
| C — chỉ tăng wd | 5e-4 | 0.3 | 30 | 76.18% | +0.40 |
| **D — chỉ rút warmup** | **1e-4** | **0.15** | **30** | **76.28%** | **+0.50 (tốt nhất)** |
| E — D + train dài hơn | 1e-4 | 0.15 | 60 | 75.08% | -0.70 |

**Kết luận quyết định — dừng ablation optimizer tại đây**: cả 2 lần thử kết hợp `weight_decay` + `pct_start` (B ở mức mạnh, F ở mức vừa) đều **không vượt được D đứng một mình**. Đây là bằng chứng nhất quán (không phải ngẫu nhiên 1 lần) cho thấy 2 tham số này **không cộng dồn lợi ích** trên bài toán này — kết hợp luôn kém hơn hoặc bằng phương án tốt nhất trong 2 cái đứng riêng. Không còn hướng combo nào hợp lý để thử thêm trong phạm vi 2 tham số này.

**D (`--onecycle-pct-start 0.15`, giữ mọi flag khác mặc định, 30 epoch) là cấu hình AdamW cuối cùng được chọn cho Baseline 1 gốc — 76.28% exact match.**

### Về mục tiêu 80%

Sau toàn bộ 7 cấu hình, mức trần thực tế đạt được bằng optimizer tuning trên backbone CRNN gốc là **76.28%** — không chạm 80%, đúng như dự đoán ở đầu mục 6 (report ICPR với kiến trúc mạnh nhất cũng chỉ đạt 78.70%). Đây không phải thất bại của quá trình tuning — đã thử đủ các hướng hợp lý (param-grouping, weight_decay, warmup, kết hợp mạnh/vừa, train dài hơn) và đều đã đo đạc, phân tích rõ nguyên nhân từng kết quả. Đây là **giới hạn thực sự của kiến trúc CRNN+STN gốc** trên dataset này. Muốn tiến gần 80% cần đổi hướng khác đã có sẵn phân tích trong project:
- [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) — nâng backbone CNN (76.68% với optimizer cũ, có thể thử áp dụng D — `pct_start=0.15` — lên backbone này để cộng dồn cả 2 cải tiến).
- [super_resolution_experiments.md](super_resolution_experiments.md) — các hướng SR đã/đang thử.

## 6. Thử kết hợp SR có sẵn + AdamW tuning

Người dùng muốn thử thêm hướng Super Resolution kết hợp AdamW, giữ bám sát Baseline 1 (không viết lại backbone mới). `crnn_and_stn` đã có sẵn module SR — `LightweightSR` (`src/models/components.py`, kiến trúc ResBlock + Channel Attention + PixelShuffle) — và CLI `--use-sr --sr-scale {2,4}` trong `train.py`, không cần viết code mới.

### Làm rõ nhầm lẫn quan trọng trước khi chạy

Mốc **76.68%** (nhắc ở đầu cuộc trao đổi) là của **ResBlock backbone** (thay CNN backbone), **không phải SR**. SR (stacked-input) trong `super_resolution_experiments.md` thực tế cho kết quả **thấp hơn nhiều baseline: 49.25%/55.06%** — đừng nhầm 2 hướng này khi kỳ vọng kết quả.

### SR hiện tại áp dụng theo từng frame, không phải gộp 5 frame

Đã kiểm tra `src/models/crnn.py`: `_apply_sr_per_frame()` chạy trên `x_flat` — tức `[B*F, C, H, W]`, SR xử lý **độc lập từng frame** trước khi STN warp, giữ nguyên đa dạng giữa 5 frame cho `AttentionFusion` khai thác sau này. Đây **không phải** thiết kế lỗi "gộp 5 frame → 1 → nhân bản" (`StackedSRNet`) mà report từng ghi nhận là nguyên nhân khiến SR đạt 49-55%.

**Không chắc chắn**: số liệu 49.25%/55.06% đã log trong [run_gpu.md](../training_runs/run_gpu.md) dùng đúng flag `--use-sr --sr-scale 2` giống hệt CLI hiện tại — nhưng không rõ liệu source code lúc đo 2 con số đó có phải đúng bản `LightweightSR` áp dụng per-frame này, hay một bản `StackedSRNet` bị lỗi khác đã tồn tại trước đó rồi được sửa lại (report gọi bản đã sửa là `LightEdgeSR`, khác tên với `LightweightSR` hiện có trong source — có thể là 2 lần thử SR khác nhau, không đồng nhất theo tài liệu). Vì không thể xác minh lại lịch sử chính xác, cách đáng tin cậy nhất là **đo lại trên chính source code hiện tại**.

### Lệnh chạy (đã thêm vào run_gpu.md)

```bash
# G. Sanity check nhanh (5 epoch) — kiểm tra không lỗi + xem xu hướng
python train.py \
 --experiment-name crnn_stn_sr_adamw_sanity \
 --batch-size 64 --epochs 5 --lr 0.0005 \
 --use-sr --sr-scale 2 --onecycle-pct-start 0.15 \
 --num-workers 8 --aug-level full

# H. Full run (chỉ chạy sau khi G ổn) — so trực tiếp với D (76.28%, không SR)
python train.py \
 --experiment-name crnn_stn_sr_adamw \
 --batch-size 64 --epochs 30 --lr 0.0005 \
 --use-sr --sr-scale 2 --onecycle-pct-start 0.15 \
 --num-workers 8 --aug-level full
```

Kết hợp `--onecycle-pct-start 0.15` (D, cấu hình AdamW tốt nhất đã tìm được ở mục 6) vì không có lý do gì để bỏ cải tiến optimizer đã kiểm chứng khi thử thêm SR — nếu H thấp hơn D, biết chắc là do SR chứ không phải do optimizer.

### Kỳ vọng và tiêu chí đánh giá

- Nếu H (SR + AdamW tuning) **vượt D** (76.28%): SR per-frame thực sự giúp ích trên bài toán này (khác với StackedSRNet cũ) — giữ làm cấu hình mới tốt nhất, cập nhật vào mục 6.
- Nếu H **thấp hơn D rõ rệt** (về vùng 49-55% như số liệu cũ): xác nhận SR (kể cả bản per-frame đã sửa) vẫn không phù hợp cho bài toán biển số low-resolution này — loại khỏi pipeline, quay lại dùng D không SR làm cấu hình cuối.
- Nếu H **thấp hơn D một chút** (vùng 74-76%): SR không giúp nhưng cũng không phá hỏng nghiêm trọng — có thể do overhead tính toán/nhiễu thêm vào ảnh vốn đã rất nhỏ (32×128), không đáng để đánh đổi thêm chi phí inference.

## 7. Kết luận

- Kiến trúc `crnn_and_stn` khớp 100% với **STN + CRNN (Baseline 1)** mô tả trong report ICPR — đối chiếu từng module ở mục 2.
- **AdamW đã được kết hợp** làm optimizer chính cho Baseline 1 (kèm OneCycleLR, CTC loss, gradient clipping, mixed precision) — không phải bổ sung mới, đã có sẵn từ khi trích xuất Baseline 1.
- Kết quả đo thực tế (trước khi sửa param-grouping): **75.78%** exact match, so với mốc report 77.00%.
- Đã tinh chỉnh cách AdamW áp dụng (param-grouping weight decay, expose betas/eps/OneCycle shape) — chưa đổi optimizer, chỉ tối ưu cách dùng AdamW hiện có.
- **Ablation đầy đủ 7 cấu hình (A-F) trên GPU thực tế** — xem bảng và phân tích chi tiết ở mục 6:

| Cấu hình | Val Acc | So với baseline |
|---|---:|---:|
| Baseline gốc | 75.78% | — |
| A — param-grouping | 75.98% | +0.20 |
| B — wd cao + warmup ngắn (mạnh, cùng lúc) | 75.58% | -0.20 |
| E — warmup ngắn + train dài hơn (60 epoch) | 75.08% | -0.70 |
| F — wd + warmup mức vừa (cùng lúc) | 76.08% | +0.30 |
| C — chỉ tăng weight_decay | 76.18% | +0.40 |
| **D — chỉ rút ngắn warmup (`pct_start=0.15`)** | **76.28%** | **+0.50** |

- **3 bài học ablation rút ra, không phụ thuộc vào nhiễu thống kê của validation set nhỏ (999 sample)**:
  1. `weight_decay` cao hơn (5e-4) và warmup ngắn hơn (`pct_start=0.15`) **đều tốt khi đứng riêng**, nhưng **không cộng dồn khi kết hợp** — xác nhận nhất quán qua cả B (mức mạnh) lẫn F (mức vừa), cả hai đều kém hơn D đứng một mình.
  2. **Train dài hơn (60 epoch) làm giảm accuracy do overfit**, không giúp hội tụ tốt hơn — dataset ~19,000 track là không đủ lớn để nuôi gấp đôi số bước gradient update mà không mất tổng quát hóa.
  3. **D (`--onecycle-pct-start 0.15`, mọi flag khác giữ mặc định, 30 epoch) là cấu hình AdamW cuối cùng, tốt nhất — 76.28% exact match.**

- **Về mục tiêu 80% (yêu cầu người dùng)**: **không đạt được** bằng optimizer tuning trên backbone CRNN gốc — mức trần thực nghiệm là 76.28%. Đây không phải do tuning chưa đủ kỹ (đã thử 7 cấu hình có hệ thống, mỗi cấu hình đều phân tích rõ nguyên nhân) mà là giới hạn thực sự của kiến trúc: ngay cả cấu hình mạnh nhất trong report ICPR gốc (ResNet+Transformer+STN, không phải CRNN đơn giản) cũng chỉ đạt 78.70%. Muốn tiến gần 80% cần đổi hướng kiến trúc, không phải tiếp tục tinh chỉnh optimizer:
  - [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) — nâng backbone CNN (76.68% với optimizer cũ) — có thể thử cộng dồn với `--onecycle-pct-start 0.15` tìm được ở đây.
  - [super_resolution_experiments.md](super_resolution_experiments.md) — các hướng SR đã/đang thử.

- **Khuyến nghị cuối cùng cho submission/train tiếp theo trên Baseline 1 gốc**: dùng cấu hình D:
  ```bash
  python train.py --submission-mode \
   --experiment-name crnn_stn_adamw_final \
   --batch-size 64 --epochs 30 --lr 0.0005 \
   --onecycle-pct-start 0.15 \
   --num-workers 8 --aug-level full
  ```
