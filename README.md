## Phân loại PR:
- [x] Feature
- [ ] Bugs
- [ ] Hotfix

# Đối chiếu tính năng đã làm theo yêu cầu:

- [x] #1: Áp dụng kiến thức mới về Super Resolution vào pipeline train
- [x] #2: Tích hợp/chuẩn bị các hướng SR đề xuất như ResBlock, Channel Attention, PixelShuffle, Stacked Inputs

# Cách thực hiện để xử lý mỗi yêu cầu:

- #1: cập nhật thiết kế model/pipeline để có thể thử nghiệm SR như một thành phần hỗ trợ OCR
- #2: chuẩn bị cấu trúc kiến trúc và tham số để dễ ablation các biến thể SR khác nhau

# Phạm vi ảnh hưởng:

- Ảnh hưởng đến hướng thử nghiệm model `CRNN + STN + SR`
- Ảnh hưởng đến chiến lược train và so sánh baseline
- Ảnh hưởng đến các module liên quan đến SR trong model pipeline

<img width="714" height="740" alt="image" src="https://github.com/user-attachments/assets/c44cf3b5-a7a5-44b4-a436-d98c94ab390b" />

<img width="1171" height="784" alt="image" src="https://github.com/user-attachments/assets/de97c15f-a2c0-46f7-9ce8-a719da2c3f78" />
