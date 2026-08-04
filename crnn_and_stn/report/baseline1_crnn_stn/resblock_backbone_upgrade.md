# Nâng cấp CRNN + STN bằng backbone ResBlock kiểu Super-Resolution

> Cải tiến đầu tiên áp dụng lên [Baseline 1 (CRNN + STN)](baseline1_architecture.md). Đây là hướng đang cho kết quả tốt nhất hiện tại của project: **~76.68%** exact match (so với ~75-77% của CRNN/CRNN+STN gốc) — xem log chạy thực tế tại [training_runs/run_gpu.md](../training_runs/run_gpu.md).

## 1. Bối cảnh và mục tiêu

Bài toán OCR biển số trong project này có đầu vào rất nhỏ, mờ, nhiễu và thường quan sát qua nhiều frame liên tiếp. Với đặc thù đó, pipeline CRNN + STN + CTC truyền thống vẫn đúng hướng, nhưng backbone CNN đơn giản dễ gặp hai hạn chế chính:

- khó giữ lại chi tiết mảnh của ký tự khi ảnh đầu vào bị low-resolution;
- training dễ dao động khi chạy dài, batch lớn hơn, hoặc khi dùng gradient accumulation.

Vì vậy, lần nâng cấp này giữ nguyên kiến trúc tổng thể của CRNN + STN + CTC, nhưng thay backbone CNN bằng một backbone ResBlock ổn định hơn, lấy cảm hứng từ các mô hình Super Resolution.

## 2. Ý tưởng chính

Mục tiêu không phải thay toàn bộ mô hình, mà là tăng chất lượng trích xuất đặc trưng ở phần backbone. Trong OCR low-resolution, phần quan trọng nhất là giữ được biên ký tự, nét nhỏ và cấu trúc chữ/số sau khi qua nhiều bước xử lý.

Backbone kiểu SR phù hợp vì nó thường ưu tiên: phục hồi chi tiết nhỏ, giữ tín hiệu biên tốt hơn, học theo hướng tăng cường feature thay vì làm chúng bị "trơn" quá sớm.

## 3. Cách thực hiện

### 3.1. Backbone ResBlock không dùng BatchNorm
Trong `src/models/components.py`, backbone được thiết kế lại theo hướng:
- dùng `ResidualBlock` gồm 2 lớp conv 3x3;
- bỏ BatchNorm trong residual path;
- thêm `res_scale` nhỏ để giới hạn biên độ cập nhật;
- hỗ trợ bật/tắt `Squeeze-Excitation` để ablation.

Lý do chính là BatchNorm thường không lý tưởng khi batch size nhỏ hoặc khi dữ liệu biến thiên mạnh. Trong OCR biển số, batch nhỏ và tín hiệu mờ làm cho feature normalization dễ gây dao động. Residual scaling giúp block học chậm hơn nhưng an toàn hơn, đặc biệt khi train dài.

### 3.2. Nối backbone mới vào `MultiFrameCRNN`
Trong `src/models/crnn.py`, pipeline tổng thể vẫn giữ nguyên:
1. 5 frame đầu vào đi qua STN để căn chỉnh hình học;
2. shared backbone trích feature từ từng frame;
3. attention fusion gộp các frame thành một biểu diễn duy nhất;
4. BiLSTM đọc chuỗi feature theo trục width;
5. linear head xuất log-probabilities cho CTC.

Điểm mới là backbone giờ có thể cấu hình linh hoạt theo: `backbone_base_channels`, `backbone_stage_blocks`, `backbone_stage_channels`, `residual_scale`, `use_se`. Nhờ đó, model có thể chạy ở các chế độ khác nhau: nhẹ để debug, ổn định để train chuẩn, hoặc mạnh hơn cho thực nghiệm dài hạn.

### 3.3. Mở rộng CLI trong `train.py`
CLI được mở rộng để hỗ trợ: preset `stable`, `strong`, `debug`; override trực tiếp backbone từ dòng lệnh; batch size, epochs, gradient accumulation, warmup, min LR ratio, patience; bật/tắt STN, SE, AMP, cudnn benchmark. Điều này giúp chạy ablation nhanh hơn và tái lập một cấu hình train mạnh chỉ bằng một lệnh.

### 3.4. Tăng độ ổn định cho training loop
Trong `src/training/trainer.py`, training loop được tinh chỉnh theo hướng phù hợp với các run dài hơn: thay scheduler cứng bằng warmup + cosine decay; hỗ trợ gradient accumulation; giữ AMP để tiết kiệm bộ nhớ; clip gradient để giảm nguy cơ nổ gradient; early stopping theo validation exact match; lưu checkpoint tốt nhất theo metric chính.

Với OCR, exact match là metric quan trọng nhất vì sai một ký tự là coi như sai cả biển số. Vì vậy, ưu tiên checkpoint theo exact match hợp lý hơn chỉ theo loss.

### 3.5. Bổ sung tiện ích hậu xử lý và reproducibility
Trong `src/utils/postprocess.py`: normalize text, tính edit distance, CER, exact match theo batch, greedy CTC decode kèm confidence. Trong `src/utils/common.py`: hàm seed rõ ràng hơn để hỗ trợ reproducibility khi so sánh các ablation.

## 4. Tóm tắt các thay đổi và tác động

| Thành phần | Thay đổi | Tác động |
|---|---|---|
| `components.py` | ResBlock không BN, residual scaling, optional SE | Giữ chi tiết tốt hơn, backbone ổn định hơn |
| `crnn.py` | Backbone mới gắn vào pipeline 5-frame + STN + fusion + BiLSTM | Tăng khả năng trích feature cho OCR low-res |
| `train.py` | CLI preset, override backbone, batch/epoch/accumulation | Dễ chạy experiment mạnh và ablation |
| `trainer.py` | Warmup, cosine decay, early stopping, gradient clipping | Train dài ổn định hơn, giảm dao động |
| `postprocess.py` | Normalize text, CER, exact match, decode utility | Đánh giá nhất quán hơn |
| `common.py` | Seed rõ ràng hơn | Tăng reproducibility |

## 5. Vì sao các thay đổi này có ích

- **Backbone kiểu SR giúp giữ chi tiết tốt hơn**: biển số LR thường chỉ có vài pixel/ký tự, backbone không BN giúp giữ biên/nét nhỏ, giảm feature bị "mềm" quá sớm.
- **Residual scaling làm train an toàn hơn**: backbone sâu hơn + residual scaling nhỏ giúp output không nhảy quá mạnh khi train dài, batch lớn, dùng gradient accumulation, hoặc xử lý 5-frame có độ rõ khác nhau.
- **Attention fusion giữ lại frame hữu ích**: không phải frame nào cũng rõ; attention fusion học frame nào đáng tin hơn thay vì trộn đều toàn bộ 5 frame.
- **STN vẫn rất quan trọng**: giúp căn chỉnh hình học cho frame bị xoay/méo/lệch, làm input sạch hơn để backbone tập trung vào chi tiết ký tự.
- **Warmup + accumulation giúp train dài ổn định**: preset `strong` và warmup/min-LR-ratio/gradient-accumulation/patience cho phép train lâu hơn mà ít dao động loss hoặc tụt chất lượng giữa chừng.

## 6. Trade-off cần lưu ý

- Backbone sâu hơn tốn compute và VRAM hơn.
- Preset `strong` cần thời gian train lâu hơn.
- Gradient accumulation tăng effective batch size nhưng làm mỗi epoch chạy lâu hơn.
- Bật SE có thể tăng nhẹ chi phí tính toán.

Tuy nhiên, với bài toán OCR biển số low-resolution, đánh đổi này thường xứng đáng nếu mục tiêu là tăng exact match và độ ổn định của mô hình.

## 7. Cách dùng nhanh

- `--preset stable`: cấu hình mặc định an toàn, phù hợp để train chính.
- `--preset strong`: train dài hơn, mạnh hơn, dùng khi muốn tối ưu kết quả.
- `--preset debug`: kiểm tra pipeline nhanh.
- `--no-stn`: ablation để đo lợi ích của STN.
- `--backbone-blocks 2,2,3,3,4 --backbone-stage-channels 64,128,256,256,512`: chạy backbone sâu hơn theo ý muốn.

Xem đầy đủ các lệnh CLI thực tế đã chạy (kèm kết quả) tại [training_runs/run_gpu.md](../training_runs/run_gpu.md).

## 8. Kết luận

Nâng cấp này giữ nguyên triết lý của CRNN + STN cho OCR, nhưng làm backbone tốt hơn theo hướng ResBlock/SR, đồng thời bổ sung các điều kiện train thực tế như warmup, accumulation và early stopping. Đây là bước cải thiện quan trọng nhất đã đo được cho tới nay trên bài toán biển số low-resolution nhiều frame (~76.68% exact match), và là baseline mới để so sánh các hướng Super Resolution — xem [super_resolution_experiments.md](../../backup/report/summary_project/document/super_resolution_experiments.md).
