## Phân Tích Nguyên Nhân

**Sinh viên chưa xác định đúng nguyên nhân** vì sao mô hình bị plateau ở mức **76.78%** và chưa thể vượt qua Baseline $78.70\%$ hoặc tiệm cận $82\%+$. Dưới đây là cây chẩn đoán 5 cấp (5-Level Cause Tracing):

```
[Triệu chứng C1] Val Accuracy bị kẹt ở 76.78%, không tăng sau Epoch 18 & kém hơn Baseline 78.70%
       │
[Nguyên nhân gần C2] Feature maps bị nhòe sau Fusion & Module Super-Resolution (SR) không học được chi tiết nét chữ
       │
[Nguyên nhân kỹ thuật C3] STN căn chỉnh độc lập từng frame gây lệch góc giữa 5 frames + Module SR thiếu Supervision Loss (L1/Perceptual)
       │
[Nguyên nhân thiết kế C4] Thiết kế khối SR dạng LR-to-LR residual mà không upsample + Thiếu liên kết thời gian (Temporal Alignment) trước Attention Fusion
       │
[Nguyên nhân gốc rễ C5] Nhầm lẫn khái niệm giữa Image Restoration và Feature Extraction; coi SR là "cơ chế phụ" end-to-end thay vì task có giám sát trực tiếp.
```

### Chi tiết 4 Nguyên nhân Cốt lõi:

1. **Nguyên nhân #1: Module Super-Resolution (SR) bị "Mù Giám Sát" (Unsupervised Bottleneck)**
   - _Hiện trạng trong Code:_ Module `SuperResolutionBlock` được đặt đầu pipeline nhưng chỉ nhận Loss gián tiếp từ CTC Loss thông qua toàn bộ mạng.
   - _Bản chất toán học:_ CTC loss chỉ quan tâm đến tính phân biệt chuỗi ký tự ở lớp cuối (logits), không cung cấp gradient độ phân giải không gian (spatial gradient) ở các tần số cao (high-frequency edges). Do đó, khối SR không học được cách khôi phục nét chữ mà ngược lại biến đổi ảnh thành các feature mờ làm mất thông tin gốc.

2. **Nguyên nhân #2: Spatial Transformer Network (STN) Gây Mất Đồng Bộ Đa Khung Hình (Frame-Disjoint Alignment)**
   - _Hiện trạng trong Code:_ `STNBlock` nhận `x_flat` có shape `[B*F, 3, H, W]` và tính ma trận Affine $\theta \in \mathbb{R}^{2 \times 3}$ độc lập cho từng frame trong 5 frames.
   - _Hậu quả:_ Mỗi frame trong cùng 1 track bị xoay/co giãn theo một góc khác nhau. Khi đưa vào `AttentionFusion` để tính tổng trọng số $\sum w_i \cdot f_i$, các đặc trưng bị lệch vị trí điểm ảnh (pixel misalignment), dẫn đến hiện tượng bóng ma (ghosting artifact) làm nhòe nét chữ trước khi qua Transformer.

3. **Nguyên nhân #3: Chiến lược Synthetic Degradation chưa khớp với Nhiễu Nén Nặng ở Scenario-B**
   - _Hiện trạng trong Dataset:_ Data loader tạo ảnh LR nhân tạo từ ảnh HR bằng các phép biến đổi làm mờ nhẹ (Gaussian Blur, Downsampling).
   - _Hậu quả:_ Dữ liệu Scenario-B chịu nhiễu nén JPEG nặng (Quantization Noise, Blocking 8x8). Mô hình học trên ảnh mờ mịn (smooth blur) nhưng khi validate trên Scenario-B thì thất bại trước nhiễu nén góc cạnh (JPEG artifacts).

4. **Nguyên nhân #4: Transformer Encoder Bị Thiếu Warmup & Epoch Training Quá Ngắn**
   - _Hiện trạng trong Training:_ Số epoch cố định = 30, Learning Rate = 5e-4 không có bước Warmup cho Transformer.
   - _Hậu quả:_ Các lớp Self-Attention trong Transformer rất nhạy cảm ở giai đoạn đầu. Việc không có LR Warm-up làm các ma trận trọng số $W_Q, W_K, W_V$ bị dao động mạnh ở 5 epoch đầu, dẫn đến việc hội tụ sớm (premature convergence) ở Epoch 18 và bị plateau.

---

## Audit Code và Kiến Trúc Mô Hình

Sau khi kiểm tra toàn bộ file source code, dưới đây là các gợi ý:

### Trace Kiến trúc Forward Pass & Tensor Shape Analysis

Pipeline hiện tại trong `ResTranOCR.forward`:
$$X \in \mathbb{R}^{B \times F \times 3 \times 32 \times 128} \xrightarrow{\text{reshape}} X_{\text{flat}} \in \mathbb{R}^{(B\cdot F) \times 3 \times 32 \times 128}$$
$$X_{\text{sr}} = \text{SuperResolutionBlock}(X_{\text{flat}}) \in \mathbb{R}^{(B\cdot F) \times 3 \times 32 \times 128} \quad (\text{Vẫn là độ phân giải } 32 \times 128!)$$
$$\theta = \text{STN}(X_{\text{sr}}) \in \mathbb{R}^{(B\cdot F) \times 2 \times 3} \implies X_{\text{align}} = \text{GridSample}(X_{\text{sr}}, \theta)$$
$$H_{\text{conv}} = \text{ResNet34}(X_{\text{align}}) \in \mathbb{R}^{(B\cdot F) \times 512 \times 1 \times 32} \quad (\text{Height bị nén về 1, Width = 32})$$
$$H_{\text{fused}} = \text{AttentionFusion}(H_{\text{conv}}) \in \mathbb{R}^{B \times 512 \times 1 \times 32}$$
$$Z = \text{TransformerEncoder}(\text{PositionalEncoding}(H_{\text{fused}})) \in \mathbb{R}^{B \times 32 \times 512}$$
$$\text{Logits} = \text{Linear}(Z) \in \mathbb{R}^{B \times 32 \times 37} \xrightarrow{\text{log\_softmax}} P_{\text{CTC}}$$

### Nhận xét Kiến trúc

- **Điểm mạnh:** Lựa chọn ResNet34 tùy chỉnh stride (layer3 và layer4 có stride `(2,1)`) giúp giữ nguyên độ dài chuỗi feature map (Width=32) là một thiết kế chuẩn xác cho OCR sequence modeling.
- **Điểm yếu:**
  1. KHÔNG nâng độ phân giải (Upsampling factor $r=1$ thay vì $r=2$ hoặc $r=4$).
  2. STN không chia sẻ tham số biến đổi giữa 5 frames.
  3. Spatial score trong `AttentionFusion` dùng Conv2D 1x1 trên feature map $1 \times 32$ (chiều cao đã bị nén về 1), do đó `score_net` thực chất chỉ là 1D attention theo chiều ngang, không còn khả năng chú ý không gian 2D nữa!

---

# Phân Tích Phương Pháp Training & Pipeline Dữ Liệu

### Data Leakage & Sampling Imbalance trong `dataset.py`

Trong `_index_samples`:

```python
# Sample 1: ảnh LR thật từ camera
self.samples.append({"paths": lr_files, "label": label, "is_synthetic": False})
# Sample 2: synthetic LR (degrade từ HR) — chỉ khi training
if self.mode == "train" and hr_files:
    self.samples.append({"paths": hr_files, "label": label, "is_synthetic": True})
```

- **Vấn đề:** Trong 1 epoch, một track xuất hiện 2 lần dưới dạng 2 sample độc lập trong `DataLoader`. Điều này làm tỷ lệ sample thực tế tăng gấp đôi (36,000 samples thay vì 18,000 tracks).
- **Hậu quả:** Model xem 36,000 samples là 36,000 tracks khác nhau, dẫn đến số step per epoch tăng gấp đôi, làm sai lệch việc tính toán Cosine Annealing Learning Rate Scheduler theo epoch.

### Super Resolution Loss Missing

Trong `trainer.py`, nếu `use_learnable_sr=True`, loss tổng cộng chỉ bao gồm CTC Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}}$$
thiếu hoàn toàn auxiliary loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{sr}} \mathcal{L}_{1}(I_{\text{SR}}, I_{\text{HR}})$$

---

## Hướng Giải Quyết Đề Xuất

Dựa trên việc chẩn đoán nguyên nhân, dưới đây là giải pháp kỹ thuật cụ thể theo thứ tự ưu tiên:

### Giải pháp 1: Thiết kế lại Khối Super-Resolution với Supervision Loss chuẩn (Sửa ERR-01)

- **Kiến trúc:** Chuyển `SuperResolutionBlock` thành dạng **Joint Super-Resolution & Recognition**. Sử dụng khối PixelShuffle để upsample kích thước ảnh từ $32 \times 128 \to 64 \times 256$ ($r=2$).
- **Loss Function:** Tích hợp auxiliary Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}} \cdot \left( \mathcal{L}_{1}(I_{\text{SR}}, I_{\text{HR}}) + \mu \mathcal{L}_{\text{perceptual}}(I_{\text{SR}}, I_{\text{HR}}) \right)$$
  với $\lambda_{\text{SR}} = 0.1$, $\mu = 0.01$.

```python
# Code sửa đổi gợi ý cho ResTranOCR forward pass:
if self.use_learnable_sr:
    # x_flat: [B*F, 3, 32, 128] -> sr_out: [B*F, 3, 64, 256]
    sr_out = self.sr_block(x_flat)
    x_for_backbone = sr_out
else:
    sr_out = None
    x_for_backbone = x_flat
```

### Giải pháp 2: Chuẩn hóa STN Đồng nhất Khung hình (Shared Alignment STN - Sửa ERR-02)

- **Cơ chế:** Thay vì dự đoán 5 ma trận affine riêng lẻ, truyền toàn bộ chuỗi 5 frames qua 1 khối STN dự đoán **1 ma trận affine duy nhất** $\theta_{\text{track}} \in \mathbb{R}^{2 \times 3}$ đại diện cho cả track, HOẶC dùng 1 frame làm Anchor Frame (khung hình chuẩn) và căn chỉnh 4 frames còn lại theo Anchor Frame.

### Giải pháp 3: Tái cấu trúc Attention Fusion ở không gian 2D (Sửa ERR-04)

- **Cơ chế:** Chuyển `AttentionFusion` lên **TRƯỚC** khi nén chiều cao $H$.
- Thực hiện Attention Fusion ngay trên Feature Map $2D$ dạng $[B \cdot F, C, H', W']$ (ví dụ: $C=256, H'=8, W'=32$). Nhờ đó, mô hình có thể kết hợp vùng ký tự nét rõ từ Frame 1 và vùng ký tự bị mờ ở Frame 2 trên cùng một tọa độ $2D$.

### Giải pháp 4: Bổ sung JPEG Degradation Simulation vào Pipeline (Sửa ERR-03)

- Sử dụng Albumentations `ImageCompression(quality_lower=15, quality_upper=50, p=0.7)` để mô phỏng chính xác nhiễu nén của Scenario-B trong quá trình sinh ảnh Synthetic LR từ HR.

---
