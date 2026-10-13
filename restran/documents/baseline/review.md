## 2. Phân tích Nguyên nhân Gốc rễ (Root Cause Analysis - RCA)

Áp dụng phương pháp Chẩn đoán 5 Cấp (5-Level Cause Tracing) để giải mã vì sao toàn bộ các cải tiến kiến trúc (SR, EDVR, Distillation) đều dừng bước trước ngưỡng $\approx 77.5\%$:

```
[Triệu chứng C1] Mọi nỗ lực thêm SR / Alignment / Distillation đều plateau ở 76.8% - 77.6% (kém baseline BTC 78.70%).
       │
[Nguyên nhân gần C2] Feature Distillation & Khối SR không tái tạo được tần số cao mang tính phân biệt ký tự từ ảnh LR 48x18.
       │
[Nguyên nhân kỹ thuật C3] Tín hiệu giám sát CTC là "weak sequence-level supervision", không có gradient ràng buộc hình học/pixel; trong khi Feature Distillation MSE bị chi phối bởi nền biển số thay vì nét chữ.
       │
[Nguyên nhân thiết kế C4] Cố gắng giải bài toán "Image Restoration / Alignment" ở không gian pixel trước Backbone, thay vì "Task-Driven Spatio-Temporal Feature Aggregation" và "Language/Syntax-Constrained Decoding".
       │
[Nguyên nhân gốc rễ C5] Nhầm lẫn giữa "khôi phục ảnh đẹp thị giác" và "tối ưu hóa năng lực phân biệt ký tự cho OCR"; bỏ qua miền tri thức tiền định cứng (Syntax Priors) của bài toán Biển số xe.
```

### Chi tiết 4 Kết luận Cốt lõi về Nguyên nhân Gốc rễ:

1. **RCA-01: Bế tắc của Super-Resolution không giám sát điểm ảnh (Pixel-Unsupervised SR Bottleneck)**
   - _Bản chất:_ Khi huấn luyện SR nội suy từ LR $\to$ LR với CTC Loss duy nhất, CTC Loss chỉ đo divergence giữa alignment chuỗi và ground truth text. Khối SR 0.21M tham số không có inductive bias hay ground-truth pixel HR để học cách "tái tạo nét ký tự". Thêm identity-skip chỉ giúp nó học thành ánh xạ đồng nhất (Identity function) để tránh phá hỏng gradient, dẫn đến việc hòa vốn so với baseline.

2. **RCA-02: Bất cập của Feature Distillation không trọng số (Unweighted Feature Distillation)**
   - _Bản chất trong Run #17:_ `distill_loss` giảm $8\times$ (1.44 $\to$ 0.175) nhưng CTC Loss xấu đi ($0.0330 \to 0.0434$). Lý do: hàm tổn thất $L_2$ trên toàn bộ feature map $\mathbb{R}^{B \times 512 \times 1 \times 32}$ bị chi phối bởi các vùng nền (background) chiếm $70\% - 80\%$ diện tích biển số. Student học cách giả lập feature nền của Teacher, nhưng hy sinh các gradient nhạy cảm về biên ký tự.

3. **RCA-03: Chi phí và Rủi ro của Deformable Convolution trên Biển số Nhỏ (EDVR Limitations on Tiny ROIs)**
   - _Bản chất trong Run #22:_ Biển số xe là vật thể cứng (rigid body). Độ dịch chuyển giữa 5 frame trong 1 track chủ yếu là tịnh tiến nhỏ và rung camera. Deformable Conv trong EDVR với offset tự do có xu hướng làm méo dạng các nét thẳng/cong của chữ cái (nhầm `8` $\leftrightarrow$ `B`, `D` $\leftrightarrow$ `0`), trong khi cơ chế Temporal Attention Fusion đa điểm trên ResNet features vốn đã là phép chiếu affine ngầm định tối ưu.

4. **RCA-04: CTC Decoder Thiếu Ràng Buộc Cú Pháp (Unconstrained Greedy Decoding)**
   - _Bản chất:_ Cả Mercosur (`LLLNLNN` hoặc `LLLNNNN`) và Brazilian (`LLLNNNN`) đều tuân theo cấu trúc cú pháp cố định nghiêm ngặt theo từng vị trí ký tự. Hiện tại model sử dụng Greedy CTC Decoding thuần túy. Khi ảnh mờ (nhóm Q1 gây $50.2\%$ lỗi), CTC dễ dự đoán nhầm ký tự chữ thành số hoặc ngược lại (ví dụ vị trí 1–3 là Chữ nhưng model dự đoán Số).

---

## 3. Audit Toàn vẹn Code & Hệ Thống (Code & System Integrity Audit)

Dưới đây là các lỗi kỹ thuật và điểm nghẽn được phát hiện trong [dataset.py](file:///Users/steven/Library/CloudStorage/GoogleDrive-stevenhuynh87@gmail.com/My%20Drive/FSB%20-%20Thesis/2026_Phat/dataset.py) và cấu trúc training hiện tại:

### 3.1 Template Báo cáo Lỗi Kỹ thuật Chi tiết

#### [ISSUE #1] — Phép tạo mẫu Synthetic LR làm lệch tỷ lệ epoch và không mô phỏng nhiễu nén Scenario-B

- **Location:** [dataset.py:183-199](file:///Users/steven/Library/CloudStorage/GoogleDrive-stevenhuynh87@gmail.com/My%20Drive/FSB%20-%20Thesis/2026_Phat/dataset.py#L183-L199)
- **Severity:** 🟠 Major
- **Category:** Experimental / Data Pipeline

```python
# Original Text:
# Sample 1: ảnh LR thật từ camera
self.samples.append({
    "paths": lr_files,
    "label": label,
    "is_synthetic": False,
    "track_id": track_id,
})
# Sample 2: synthetic LR (degrade từ HR) — chỉ khi training
if self.mode == "train" and hr_files:
    self.samples.append({
        "paths": hr_files,
        "label": label,
        "is_synthetic": True,
        "track_id": track_id,
    })
```

- **Problem Description:**
  1. Trong 1 epoch, mỗi track được nhân đôi thành 2 sample độc lập, làm tổng sample tăng từ 18,001 lên 36,002. Điều này làm số step tối ưu hóa mỗi epoch tăng gấp 2 lần, làm OneCycleLR scheduler duy trì vùng LR cao lâu gấp đôi dự kiến.
  2. Phép `self.degrade` chỉ áp dụng khi load HR sample, nhưng phép degrade hiện tại thiếu mô phỏng nhiễu nén JPEG nén khối ($8\times 8$ block artifacts) vốn là đặc trưng của Scenario-B.
- **Suggested Fix:** Gom việc chọn LR thật vs LR synthetic vào bên trong `__getitem__` với xác suất $p=0.5$ (hoặc tạo paired data), giữ nguyên độ dài epoch chuẩn bằng đúng số track (18,001 tracks), đồng thời bổ sung `A.ImageCompression(quality_range=(15, 50))` vào transform.
- **Revised Version:**

```python
# Sửa trong dataset.py:
def _index_samples(self, tracks: List[str]) -> None:
    for track_path in tqdm(tracks, desc=f"Index {self.mode}"):
        json_path = os.path.join(track_path, "annotations.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = data[0]
            label = data.get("plate_text", data.get("license_plate", data.get("text", ""))).strip()
            if not label:
                continue
            track_id = os.path.basename(track_path)
            lr_files = sorted(glob.glob(os.path.join(track_path, "lr-*.png")) + glob.glob(os.path.join(track_path, "lr-*.jpg")))
            hr_files = sorted(glob.glob(os.path.join(track_path, "hr-*.png")) + glob.glob(os.path.join(track_path, "hr-*.jpg")))

            self.samples.append({
                "lr_paths": lr_files,
                "hr_paths": hr_files if hr_files else lr_files,
                "label": label,
                "track_id": track_id,
            })
        except Exception:
            pass

def __getitem__(self, idx: int):
    item = self.samples[idx]
    # Khi train, 50% dùng LR thật, 50% dùng Synthetic LR từ HR với JPEG compression
    use_synthetic = (self.mode == "train") and (random.random() < 0.5) and (len(item["hr_paths"]) > 0)
    paths = item["hr_paths"] if use_synthetic else item["lr_paths"]
    # ... load và augment nhất quán ...
```

---

#### [ISSUE #2] — Thiếu Text-Mask Weighting trong Feature Distillation

- **Location:** `src/training/trainer.py` (Run #17 logic)
- **Severity:** 🟠 Major
- **Category:** Technical / Optimization
- **Problem Description:** Feature Distillation tính toán khoảng cách Euclidean toàn cục: $\mathcal{L}_{\text{distill}} = \|F_{\text{student}} - F_{\text{teacher}}\|^2$. Do $80\%$ diện tích feature map là nền biển số, student tối ưu hóa vào nền thay vì ký tự, làm tăng CTC Loss từ 0.0330 lên 0.0434.
- **Suggested Fix:** Tích hợp Cross-Attention Mask hoặc CTC Peak Alignment Mask để chỉ chưng cất đặc trưng tại các time-steps $t \in [1, 32]$ mà Teacher có xác suất phát hiện ký tự (non-blank).

---

#### [ISSUE #3] — Greedy CTC Decoding bỏ qua ràng buộc cú pháp biển số (Syntax Constraints)

- **Location:** `src/utils/postprocess.py` (hoặc inference logic)
- **Severity:** 🟠 Major
- **Category:** Technical / Inference
- **Problem Description:** Mô hình giải mã bằng `argmax` trên logits $[B, 32, 37]$ mà không kiểm tra tính hợp lệ của định dạng biển số. Trong khi đó, $50.2\%$ lỗi xảy ra ở ảnh mờ (Q1) với lỗi phổ biến là nhầm ký tự chữ $\leftrightarrow$ số ở các vị trí cố định.
- **Suggested Fix:** Triển khai **Prefix Beam Search with Syntax Masking (Lexicon/Regex Prior Constraint)** hoàn toàn không cần train lại model.
- **Revised Code (`src/utils/constrained_decoder.py`):**

```python
import torch
import numpy as np

MERCOSUR_PATTERN = ["ALPHA", "ALPHA", "ALPHA", "DIGIT", "ALPHA", "DIGIT", "DIGIT"] # LLLNLNN
BRAZILIAN_PATTERN = ["ALPHA", "ALPHA", "ALPHA", "DIGIT", "DIGIT", "DIGIT", "DIGIT"] # LLLNNNN

def decode_with_plate_syntax(logits: torch.Tensor, idx2char: dict, blank_idx: int = 0) -> str:
    """
    Constrained CTC Beam Search Decoder áp dụng cú pháp biển số Mercosur & Brazilian.
    logits: Tensor shape [T, num_classes] (ví dụ: [32, 37])
    """
    probs = torch.softmax(logits, dim=-1).cpu().numpy() # [32, 37]
    # Phân loại tập chỉ số ký tự
    alpha_indices = [idx for idx, ch in idx2char.items() if ch.isalpha() and idx != blank_idx]
    digit_indices = [idx for idx, ch in idx2char.items() if ch.isdigit() and idx != blank_idx]

    # Beam Search kết hợp kiểm tra template (giữ lại các path thỏa mãn template)
    # Triển khai thuật toán CTC prefix beam search với syntax prior penalty...
```

---

## 5. Đánh giá Training Dynamics & Lịch Trình Tối Ưu

### 5.1 Bệnh lý Đường cong Loss & Hiện tượng Nổ Gradient

Qua phân tích so sánh Run #1, #3, #4, #5, #6, #7, #8:

- **Hiện tượng:** Cùng `--lr 3e-4`, lịch 30 epoch hoàn thành trọn vẹn (Run #3, #6: 77.18%), nhưng lịch 60 epoch luôn nổ ở epoch 22–25 (Run #4, #5, #7, #8).
- **Cơ chế Toán học:** `OneCycleLR` với `pct_start=0.3` dời đỉnh LR từ epoch 9 (trong lịch 30 epoch) sang epoch 18 (trong lịch 60 epoch). Đến epoch 24–25, mô hình đã hội tụ sâu ($\text{loss} \approx 0.145$), các phân phối xác suất CTC $P(c_t | X)$ trở nên cực kỳ tự tin (cận kề 1.0 hoặc 0.0), làm cho đạo hàm $\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_t} = y_t - P(c_t)$ có độ cong cực lớn (sharp loss landscape). Với $\text{LR} \approx 2.85 \times 10^{-4}$ ở thời điểm đó, bước nhảy tối ưu vọt qua khỏi cực tiểu địa phương, làm gradient norm bùng nổ lên 1399.
- **Quy tắc Vàng cho Lịch trình Huấn luyện:**
  $$\text{Nếu } \text{Epochs} \ge 60 \implies \text{Max LR} \le 1.8 \times 10^{-4} \quad \text{hoặc dùng CosineAnnealingWarmRestarts với warmup 5 epochs.}$$

### 5.2 Bảng Cấu hình Hyperparameter Tối ưu Khuyến nghị

| Tham số               | Giá trị Cũ                | Giá trị Đề xuất                           | Lý do & Cơ sở Kỹ thuật                                                            |
| --------------------- | ------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------- |
| **Optimizer**         | AdamW                     | **AdamW (weight_decay=1e-4)**             | Giữ ổn định cho Transformer weights                                               |
| **LR Scheduler**      | OneCycleLR (pct=0.3)      | **CosineAnnealingLR + 3-epoch Warmup**    | Tránh giữ LR cao ở pha cuối khi loss đã giảm sâu                                  |
| **Max LR**            | 3e-4 (30ep) / 3e-4 (60ep) | **2.5e-4 (30ep) / 1.5e-4 (60ep)**         | Tránh tuyệt đối "Vùng Nguy hiểm" $\text{loss} < 0.165, \text{LR} > 2.8\text{e-}4$ |
| **Gradient Clipping** | 1.0                       | **0.5**                                   | Siết chặt ngưỡng bảo vệ CTC gradient                                              |
| **Batch Size**        | 64                        | **64**                                    | Cân bằng bộ nhớ VRAM và độ ổn định thống kê batch                                 |
| **Augmentation**      | ReplayCompose             | **Consistent Track Augment + JPEG Noise** | Mô phỏng chính xác Scenario-B                                                     |

---

## 7. Đề Xuất 3 Hướng (Novel Contributions)

---

### 💡 ĐỀ XUẤT 1: Syntax-Constrained CTC Prefix Beam Search (SC-CTC)

- **Bản chất Novelty:** Cơ chế Thuật toán Suy luận (Inference Algorithm / Problem Formulation).
- **Ý tưởng:** Biển số xe tuân theo quy tắc hữu hạn trạng thái (Deterministic Finite Automaton — DFA). Tại mỗi bước giải mã $t \in [1, 7]$, không gian tìm kiếm được chiếu qua một **Ma trận Mặt nạ Cú pháp (Syntax Mask)** $M_t \in \{0, 1\}^{37}$:
  $$M_t(c) = \begin{cases} 1 & \text{nếu ký tự } c \text{ hợp lệ với cú pháp tại vị trí } t \\ 0 & \text{ngược lại} \end{cases}$$
- **Delta so với SOTA:** Các mô hình LPR hiện nay (CRNN, ResTran) decode hoàn toàn tự do. Bằng cách tích hợp Syntax Mask vào quá trình beam search, ta loại bỏ $100\%$ các lỗi nhầm chữ $\leftrightarrow$ số ở các vị trí cố định mà **KHÔNG cần train lại mô hình, không tốn thêm tham số**.
- **Kỳ vọng cải thiện:** $+1.5\% \to +3.0\%$ trên tập Val, đặc biệt cứu vãn phần lớn lỗi ở nhóm ảnh mờ Q1.

---

### 💡 ĐỀ XUẤT 2: Confidence-Weighted Temporal Attention with Laplacian Prior (CW-TA)

- **Bản chất Novelty:** Kiến trúc Khối Fusion Đa Khung hình (Architectural Mechanism).
- **Ý tưởng:** Hiện tại `AttentionFusion` học trọng số $\alpha_f$ thuần túy từ feature ResNet. Ta đưa trực tiếp **Độ nét Laplacian** $\sigma_{\text{Lap}}^2(x_f)$ của từng frame vào làm **Inductive Bias** cho trọng số attention:
  $$\alpha_f = \text{Softmax}\left( \text{Linear}(F_f) + \gamma \cdot \log(\sigma_{\text{Lap}}^2(x_f) + \epsilon) \right)$$
- **Delta so với SOTA:** Giúp mô hình tự động gán trọng số cao cho frame ít mờ nhất trong track 5 frame ngay cả khi feature của frame đó bị nhiễu do nén JPEG.
- **Kỳ vọng cải thiện:** $+0.8\% \to +1.5\%$ trên Brazilian plates và nhóm Q1.

---

### 💡 ĐỀ XUẤT 3: Character-Centric Contrastive Feature Alignment (CC-CFA)

- **Bản chất Novelty:** Chiến lược Huấn luyện / Regularization (Training Paradigm).
- **Ý tưởng:** Thay vì Feature Distillation toàn cục kiểu MSE (thất bại ở Run #17), sử dụng **Contrastive Loss** có hướng dẫn từ CTC Alignment:
  - Anchor: Feature vector $z_t^{\text{student}}$ tại vị trí có ký tự $c_k$.
  - Positive: Feature vector $z_t^{\text{teacher}}$ của ảnh HR tương ứng tại đúng ký tự $c_k$.
  - Negative: Feature vectors của các ký tự khác $c_{j \neq k}$.
- **Delta so với SOTA:** Bắt Student học chính xác biểu diễn tần số cao của _nét chữ_ từ Teacher HR mà bỏ qua hoàn toàn sự khác biệt ở vùng nền biển số.
- **Kỳ vọng cải thiện:** $+1.0\% \to +2.0\%$ vượt trần baseline.

---
