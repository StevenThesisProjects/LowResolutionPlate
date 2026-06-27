# LowResolutionPlate OCR Project

Project OCR biển số cho dataset ICPR 2026 LRLPR, tập trung vào pipeline **CRNN + STN + CTC** và bản nâng cấp backbone **ResBlock kiểu Super-Resolution** để cải thiện độ ổn định khi train trên dữ liệu low-resolution, 5-frame.

## Mục tiêu chính

- Giữ nguyên luồng OCR chuẩn: `STN → Backbone → Attention Fusion → BiLSTM → CTC`.
- Tăng khả năng giữ chi tiết ký tự nhỏ bằng backbone ResBlock không BatchNorm, có residual scaling nhỏ.
- Hỗ trợ train dài hơn, batch lớn hơn, gradient accumulation và early stopping theo exact match.
- Cung cấp preset CLI để chạy nhanh các cấu hình `debug`, `stable`, `strong`.

## Những gì đã được nâng cấp

- **`src/models/components.py`**
  - Thêm backbone ResBlock OCR-friendly.
  - Hỗ trợ `res_scale` và `Squeeze-Excitation` tùy chọn.
  - Giữ feature map ổn định cho chuỗi 5 frame.

- **`src/models/crnn.py`**
  - Gắn backbone mới vào `MultiFrameCRNN`.
  - Giữ nguyên STN, attention fusion, BiLSTM và CTC head.

- **`src/training/trainer.py`**
  - Warmup + cosine decay.
  - Gradient accumulation.
  - AMP + gradient clipping.
  - Early stopping và checkpoint theo validation exact match.

- **`train.py`**
  - Thêm preset `debug`, `stable`, `strong`.
  - Cho phép override backbone và tham số train ngay trên CLI.
  - Hỗ trợ `--submission-mode` để train full data và tạo file dự đoán.

- **`configs/config.py`**
  - Bổ sung các tham số backbone và training ổn định hơn.

- **`src/utils/common.py`** và **`src/utils/postprocess.py`**
  - Làm rõ seed/reproducibility.
  - Bổ sung normalize text, edit distance, CER, exact match và decode utilities.

## Cấu trúc thư mục chính

```text
crnn_and_stn/
├── train.py
├── configs/
│   └── config.py
├── src/
│   ├── models/
│   │   ├── components.py
│   │   └── crnn.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       ├── common.py
│       └── postprocess.py
└── summary/
    ├── resblock_ocr_upgrade_report.md
    └── run_gpu.md
```

## Cách chạy nhanh

### 1. Chạy preset ổn định

```bash
python crnn_and_stn/train.py --preset stable
```

### 2. Chạy preset mạnh hơn

```bash
python crnn_and_stn/train.py --preset strong
```

### 3. Chạy debug nhanh

```bash
python crnn_and_stn/train.py --preset debug
```

### 4. Custom backbone

```bash
python crnn_and_stn/train.py \
  --experiment-name crnn_resblock_custom \
  --batch-size 64 \
  --epochs 100 \
  --lr 0.0006 \
  --backbone-base-channels 64 \
  --backbone-blocks 2,2,3,3,4 \
  --backbone-stage-channels 64,128,256,256,512 \
  --backbone-res-scale 0.08 \
  --fusion-dropout 0.05 \
  --frame-dropout 0.08
```

### 5. Train full data và xuất submission

```bash
python crnn_and_stn/train.py \
  --submission-mode \
  --preset strong \
  --experiment-name crnn_resblock_submission
```

## Ý nghĩa của các preset

- **`debug`**: chạy rất nhanh để kiểm tra pipeline.
- **`stable`**: cấu hình an toàn, phù hợp cho training chính.
- **`strong`**: train lâu hơn, sâu hơn, hiệu quả hơn khi có đủ tài nguyên.

## Vì sao dùng ResBlock kiểu Super-Resolution

Ảnh biển số low-resolution thường mất nhiều chi tiết nhỏ. Backbone ResBlock không BatchNorm giúp:

- giữ biên và nét ký tự tốt hơn,
- ổn định hơn khi batch nhỏ,
- phù hợp với train dài và dữ liệu nhiều nhiễu,
- kết hợp tốt với STN và attention fusion trong pipeline 5-frame.

## Kết quả kỳ vọng

Sau nâng cấp này, mô hình kỳ vọng:

- exact match tốt hơn trên track khó,
- train ổn định hơn khi dùng batch lớn hoặc gradient accumulation,
- dễ ablation hơn khi cần tắt STN hoặc SE,
- có cấu hình rõ ràng để tái lập kết quả.

## Tài liệu bổ sung

- `crnn_and_stn/summary/resblock_ocr_upgrade_report.md`
- `crnn_and_stn/summary/run_gpu.md`

## Ghi chú

Repo này là nhánh OCR chuyên cho bài toán biển số low-resolution, nên ưu tiên các quyết định thiết kế giúp tăng độ chính xác và độ ổn định hơn là tối ưu cho mô hình quá nhỏ.
