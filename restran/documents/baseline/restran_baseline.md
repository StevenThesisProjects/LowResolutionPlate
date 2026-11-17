# Phân tích Baseline ResTran (ResNet34 + Transformer) — ICPR 2026 LRLPR

**Phạm vi tài liệu:** chỉ tập trung vào **ResTran** với vai trò là **baseline chính**. Các kiến trúc khác được trình bày trong slide (CRNN) và phần Future of Work (Super-Resolution) **không** đưa vào đây.

**Nguồn:**
- Slide `Documents_2026-0_ICPR Challenge - Training_AIO2025_ICPR_TRAINING.pdf` (89 trang, nhóm Truong-Binh Duong & Anh-Khoi Nguyen — STA, 2025)
- Code thực tế trong repo: [src/models/restran.py](src/models/restran.py), [src/models/components.py](src/models/components.py), [src/data/dataset.py](src/data/dataset.py), [src/training/trainer.py](src/training/trainer.py), [configs/config.py](configs/config.py)
- Số liệu dataset đã đo: [documents/baseline/dataset.md](documents/baseline/dataset.md)

Mọi thông số kiến trúc trong tài liệu này đã được **kiểm chứng bằng cách chạy thật model** (shape đầu ra, số tham số), không chỉ đọc slide.

---

## 1. Bối cảnh bài toán (theo slide)

### 1.1. Định nghĩa

**Low-Resolution License Plate Recognition** — nhận dạng biển số từ ảnh độ phân giải cực thấp, bị nén mạnh do giới hạn băng thông/lưu trữ. Ký tự bị méo, nhoè lẫn vào nền, hoặc chồng lấn nhau.

> **Các phương pháp SOTA hiện tại chỉ đạt 50–60% accuracy** (slide 6).

Ban tổ chức khuyến khích 3 hướng: (1) Super-resolution, (2) **Temporal modeling — khai thác chuỗi frame**, (3) OCR mạnh. **ResTran đi theo hướng (2) + (3)**: hợp nhất 5 frame bằng attention rồi giải mã bằng Transformer + CTC.

### 1.2. Problem Statement

```
Input: 5 ảnh LR của cùng một track  →  Model  →  Output: "BAI8068", 0.95
```

Tức mỗi track trả về **một chuỗi ký tự duy nhất + một điểm confidence**.

### 1.3. Metric chính thức

| | Công thức | Ý nghĩa |
|---|---|---|
| **Recognition Rate** (xếp hạng chính) | `Số track đúng / Tổng số track trong test set` | Một track chỉ được tính đúng khi **toàn bộ ký tự khớp tuyệt đối** với ground truth. Sai 1 ký tự = sai cả track. |
| **Confidence Gap** (tie-break) | `Mean confidence của các dự đoán ĐÚNG − Mean confidence của các dự đoán SAI` | Gap **càng lớn xếp hạng càng cao**: phản ánh khả năng model "biết mình đang sai". |

Xếp hạng chính thức tính trên **Blind Test Set**.

> **Hệ quả thiết kế:** Confidence không phải phần trang trí — nó là tiêu chí xếp hạng thứ hai. Cách tính confidence hiện tại (§4.2) và việc hiệu chỉnh (calibration) nó là một hướng cải thiện thứ hạng **không cần tăng accuracy**.

### 1.4. Submission

- Nền tảng: **Codabench**.
- File bắt buộc: **`predictions.txt`**, mỗi dòng một track:
  ```
  track_id,plate_text;confidence
  track_00001,ABC1234;0.9876
  ```
  Dấu **phẩy** ngăn `track_id` và `plate_text`; dấu **chấm phẩy** ngăn `plate_text` và `confidence`.
- Nén **trực tiếp file** (không nén thư mục): `zip submission.zip predictions.txt`.
- Giới hạn public test: **5 lượt nộp/ngày, tối đa 25 lượt** toàn giải.

> ⚠️ Repo hiện ghi ra `results/submission_{experiment_name}.txt` ([trainer.py:185](src/training/trainer.py#L185)). **Định dạng dòng đã đúng**, nhưng phải **đổi tên thành `predictions.txt`** trước khi zip. Giới hạn 25 lượt nộp cũng có nghĩa: không được dùng public leaderboard làm tập validation — phải dựa vào val nội bộ.

### 1.5. Timeline (theo slide 12)

| Mốc | Ngày |
|---|---|
| Registration opens | 15/12/2025 |
| Training set releases | 18/12/2025 |
| Public test set releases (~30% dữ liệu blind test, dùng cho leaderboard) | 19/01/2026 |
| Registration ends | 15/02/2026 |
| Blind test set releases (dùng cho xếp hạng cuối) | 25/02/2026 |
| **Submission deadline** | **01/03/2026** |
| Trình bày tại ICPR 2026 | 08/2026 |

> Kiểm chứng với dữ liệu thật: public test có 1.000 track, blind test 3.000 track → **33%**, và 1.000 track của public là **tập con byte-identical** của blind (xác nhận bằng MD5). Khớp với mô tả "~30% of the blind test data".

---

## 2. Kiến trúc ResTran

### 2.1. Sơ đồ tổng thể (slide 41)

```
5 LR frames ──▶ STN Block ──▶ ResNet34 (weight sharing) ──▶ Attention Fusion ──▶ Transformer Layers ──▶ FC ──▶ CTC ──▶ "BAI8068"
                                                              Fused Feature Map
```

Điểm mấu chốt: **ResNet34 dùng chung trọng số cho cả 5 frame** (weight sharing) — 5 frame được gộp vào chiều batch, đi qua đúng một backbone, rồi mới hợp nhất ở mức đặc trưng. Nhờ vậy số tham số không tăng theo số frame.

### 2.2. Shape thực tế qua từng khối (đã chạy kiểm chứng)

| Bước | Tensor | Ghi chú |
|---|---|---|
| Input | `[B, 5, 3, 32, 128]` | 5 frame, ảnh resize 32×128, normalize về [-1, 1] |
| `view` | `[B*5, 3, 32, 128]` | Gộp frame vào batch → weight sharing |
| STN: `stn(x)` → `theta` | `[B*5, 2, 3]` | Ma trận affine 2×3 cho **từng frame riêng** |
| `affine_grid` + `grid_sample` | `[B*5, 3, 32, 128]` | Warp ảnh, hoàn toàn khả vi |
| ResNet34 (sửa stride) | `[B*5, 512, 1, 16]` | Chiều cao bị ép về 1, chiều rộng giữ 16 |
| AttentionFusion | `[B, 512, 1, 16]` | Softmax trên trục 5 frame |
| `squeeze(2).permute(0,2,1)` | `[B, 16, 512]` | Chuyển thành chuỗi 16 token |
| PositionalEncoding | `[B, 16, 512]` | Sinusoidal, cộng trực tiếp |
| TransformerEncoder ×3 | `[B, 16, 512]` | 8 head, ff 2048, GELU, batch_first |
| FC head + `log_softmax` | **`[B, 16, 37]`** | 36 ký tự + 1 blank |

**T = 16 timestep cho nhãn 7 ký tự** → tỉ lệ ~2,3 step/ký tự. Thoả điều kiện CTC (T ≥ L) nhưng biên hẹp; nếu giảm `IMG_WIDTH` xuống dưới 128 thì phải kiểm tra lại.

### 2.3. Ngân sách tham số (đo thật)

| Khối | Tham số |
|---|---|
| STN Block | 0,28 M |
| **ResNet34 backbone** | **21,28 M (68%)** |
| Attention Fusion | 0,03 M |
| Transformer (3 lớp) | 9,46 M (30%) |
| FC head | 0,02 M |
| **Tổng (có STN)** | **31,08 M** |
| Tổng (không STN) | 30,79 M |

STN chỉ chiếm **0,9%** tham số nhưng mang lại **+2,90 điểm accuracy** (§6) — đây là module có tỉ lệ lợi ích/chi phí cao nhất trong toàn bộ kiến trúc.

### 2.4. Chi tiết từng khối

#### a) STN Block — căn chỉnh hình học

Slide 36–38 (theo [Spatial Transformer Networks, arXiv:1506.02025](https://arxiv.org/pdf/1506.02025)):
- Trích đặc trưng để **phát hiện méo hình không gian**;
- Ánh xạ đặc trưng đó thành **6 tham số của ma trận affine**;
- Tự động nắn chỉnh xoay / co giãn / nghiêng **không cần nhãn giám sát**, khả vi hoàn toàn → train end-to-end bằng backprop thông thường.

Trong repo ([components.py:10-50](src/models/components.py#L10-L50)): localization net = `Conv(3→32, k5, s2)` → `MaxPool(2)` → `Conv(32→64, k3)` → `AdaptiveAvgPool(4×8)` → `FC(2048→128→6)`. Lớp FC cuối được **khởi tạo bằng biến đổi đồng nhất** (`weight=0`, `bias=[1,0,0,0,1,0]`) — chi tiết quan trọng: giúp model khởi động như thể không có STN rồi mới học lệch dần, tránh phá huỷ ảnh ở những epoch đầu.

Lưu ý: STN sinh `theta` cho **từng frame độc lập**, nên nó vừa nắn méo, vừa gián tiếp **căn chỉnh 5 frame về cùng hệ quy chiếu** — điều kiện cần để Attention Fusion cộng đặc trưng có ý nghĩa.

#### b) ResNet34 — backbone chia sẻ trọng số

`ResNetFeatureExtractor` ([components.py:114-166](src/models/components.py#L114-L166)) lấy `resnet34` từ torchvision, **`pretrained=False`** (train from scratch), và **sửa stride của `layer3[0]` và `layer4[0]` từ `(2,2)` thành `(2,1)`**.

Ý nghĩa: đây là tinh chỉnh chuẩn cho OCR — **chỉ giảm chiều cao, giữ nguyên chiều rộng**, để chuỗi đặc trưng đủ dài cho CTC.

Truy vết chiều rộng: `128 → conv1(s2): 64 → maxpool(s2): 32 → layer2(s2): 16 → layer3(2,1): 16 → layer4(2,1): 16`.
Truy vết chiều cao: `32 → 16 → 8 → 8 → 4 → 2`, cuối cùng `adaptive_avg_pool2d(1, None)` ép về 1.

#### c) Attention Fusion — hợp nhất 5 frame

Slide 29: `Fused Feature Map ← ∑ (Attention Weights × Features)`, trong đó Attention Weights = `Softmax(Score Net(features))`.

Repo ([components.py:53-87](src/models/components.py#L53-L87)): Score Net là `Conv1×1(512→64) → ReLU → Conv1×1(64→1)`, cho ra một **bản đồ điểm chất lượng theo từng vị trí không gian**, softmax **trên trục 5 frame**, rồi tổng có trọng số.

Đây là điểm cốt lõi về mặt ý tưởng: trọng số fusion là **per-pixel, không phải per-frame**. Nghĩa là model có thể lấy nửa trái của biển từ frame 2 và nửa phải từ frame 4 — đúng với thực tế, vì mỗi frame nhoè ở một vùng khác nhau.

> Ràng buộc cứng: `num_frames = 5` được **hard-code** ([components.py:75](src/models/components.py#L75)). Toàn bộ 24.000 track (train + test) đều đúng 5 frame nên hiện tại an toàn, nhưng module sẽ vỡ nếu số frame thay đổi.

#### d) Transformer Encoder

`PositionalEncoding` sinusoidal (chuẩn *Attention Is All You Need*) + `nn.TransformerEncoder` 3 lớp: `d_model=512`, `nhead=8`, `dim_feedforward=2048`, `dropout=0.1`, `activation='gelu'`, `batch_first=True`.

Vai trò so với BiLSTM: mô hình hoá quan hệ **giữa các vị trí ký tự** trên chuỗi 16 token bằng self-attention toàn cục thay vì hồi quy tuần tự. Với chuỗi chỉ dài 16, self-attention rẻ (16×16 attention matrix) và cho phép mọi vị trí nhìn thấy nhau ngay ở lớp đầu tiên — hữu ích vì biển số có **cấu trúc pattern cố định** (§5.3), tức ký tự ở vị trí này ràng buộc kiểu ký tự ở vị trí khác.

---

## 3. Hàm mất mát: CTC

### 3.1. Vì sao CTC

Không cần phân đoạn ký tự tường minh, không cần biết ký tự nằm ở timestep nào — **học không cần căn chỉnh** (alignment-free). Đây là điều kiện bắt buộc với ảnh LR, nơi ranh giới ký tự gần như không xác định được.

### 3.2. Cơ chế (ví dụ slide 32)

Với ground truth `"cat"`, tập ký tự `{c, a, t, -}` (`-` = blank), CTC tính **tổng xác suất của mọi đường đi hợp lệ** rồi collapse về `"cat"`:

```
P1: c a - t = 0.8 × 0.6 × 0.5 × 0.9 = 0.216
P2: c a a t = 0.8 × 0.6 × 0.4 × 0.9 = 0.1728
P3: c - a t = 0.8 × 0.2 × 0.4 × 0.9 = 0.0576
P_total = 0.4464  →  Loss = −log(0.4464) = 0.806
```

Tối thiểu hoá loss ⇔ tối đa hoá tổng xác suất của **tất cả** các đường đi cho ra chuỗi đúng.

### 3.3. Trong repo

`nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')` ([trainer.py:44](src/training/trainer.py#L44)). Chỉ số 0 dành cho blank, ký tự map sang 1…36 (`NUM_CLASSES = 37`). `input_lengths` được đặt bằng `preds.size(1) = 16` cho mọi mẫu; `target_lengths` lấy từ độ dài nhãn thật (luôn = 7).

---

## 4. Giải mã và Confidence

### 4.1. Greedy CTC decoding (slide 31)

```
Raw prediction (argmax mỗi timestep):  a a - b b
Bước 1 — Gộp ký tự lặp liên tiếp:      a - b
Bước 2 — Xoá blank:                    a b
```

### 4.2. Confidence (slide 49)

```
t1    t2    t3    t4    t5
H     H     -     I     I      ← lớp dự đoán
0.80  0.92  0.99  0.60  0.74   ← xác suất

→ H lấy max(0.80, 0.92) = 0.92 ;  I lấy max(0.60, 0.74) = 0.74
→ "HI", confidence = (0.92 + 0.74) / 2 = 0.83
```

Tức: **max trong mỗi nhóm ký tự đã gộp, rồi trung bình trên các ký tự**. Code [postprocess.py:40-53](src/utils/postprocess.py#L40-L53) triển khai **đúng** công thức này.

> **Đây là điểm nối trực tiếp với tie-break Confidence Gap.** Lấy `max` trong nhóm là lựa chọn *lạc quan* — nó đẩy confidence lên cao kể cả khi model đang phân vân, làm **thu hẹp** khoảng cách giữa dự đoán đúng và sai. Các phương án đáng thử: dùng trung bình (thay vì max) trong nhóm, dùng **min trên các ký tự** (chuỗi chỉ đúng khi ký tự yếu nhất đúng — hợp với metric exact match), hoặc tích xác suất. Đây là thay đổi **vài dòng code, không cần train lại**, và tác động trực tiếp lên thứ hạng khi hoà accuracy.

---

## 5. Xây dựng dữ liệu cho baseline

### 5.1. Theo slide (43–44)

```
Scenario A (10.000 tracks) ─┐
                            ├─▶ Training Set = A + 80% B = 18.000 tracks
Scenario B (10.000 tracks) ─┘                     Validation Set = 20% B = 2.000 tracks
                                                  (Split 9:1)

Training Set 18.000 tracks
  ├── LR images         18.000 tracks
  └── HR images 18.000 ──[Degradation]──▶ Synthetic LR 18.000
                                          Total Train: 36.000 samples
```

Điểm quan trọng: **validation lấy hoàn toàn từ Scenario B**, và **HR không dùng làm target super-resolution mà dùng để sinh thêm dữ liệu train** thông qua degradation.

### 5.2. ⚠️ Khác biệt giữa slide và code hiện tại

| | Slide (43) | Code hiện tại (`SPLIT_RATIO = 0.9`) |
|---|---|---|
| Tỉ lệ val lấy từ Scenario B | **20%** | **10%** |
| Số track train | 18.000 | 19.001 |
| Số track val | **2.000** | **999** |
| Tổng mẫu train | 36.000 | 38.002 |

Trong [dataset.py:148](src/data/dataset.py#L148), `val_size = max(1, int(len(scenario_b_tracks) × (1 − split_ratio)))`. Nếu muốn tái lập đúng con số của slide (và đúng bộ kết quả ở §6) thì phải đặt `SPLIT_RATIO = 0.8`.

> Con số **999** (thay vì 1.000) là do sai số dấu phẩy động: `1 − 0.9 = 0.09999999999999998` → `int(10000 × 0.0999…) = 999`. Với `SPLIT_RATIO = 0.8` sẽ ra 1.999 thay vì 2.000. Đã kiểm chứng bằng cách chạy thật pipeline; lệch 1 track không ảnh hưởng thực tế.

**Hệ quả đo lường:** val ~1.000 mẫu có sai số chuẩn ≈ ±1,5 điểm ở mức accuracy ~78%; val ~2.000 mẫu giảm còn ≈ ±0,9 điểm. Với khoảng cách STN mang lại là 2,90 điểm thì cả hai đều phát hiện được, nhưng các cải tiến nhỏ hơn (~1 điểm) sẽ chìm trong nhiễu nếu chỉ dùng ~1.000 mẫu.

### 5.3. Kiểm chứng với dữ liệu thật

Số liệu slide khớp với dữ liệu đo trực tiếp (chi tiết ở [dataset.md](documents/baseline/dataset.md)):

- Slide 21: biểu đồ tần suất ký tự ghi **"Based on 16,946 Unique Plates"** — đo thực tế: **16.947** biển unique / 20.000 track. Khớp (chênh 1 do một nhãn có dấu cách thừa).
- Slide 22 (Character Distribution by Position) khớp tuyệt đối với dữ liệu:

| Vị trí | Mercosur | Brazilian |
|---|---|---|
| 1 | Chữ (A, B chiếm ưu thế; tập thực tế A–U) | Chữ (tập thực tế A–R) |
| 2 | Chữ A–Z | Chữ A–Z |
| 3 | Chữ A–Z (phân bố gần đều) | Chữ A–Z |
| 4 | Số 0–9 | Số 0–9 |
| 5 | **Chữ, chỉ dùng A–J** | Số 0–9 |
| 6 | Số 0–9 | Số 0–9 |
| 7 | Số 0–9 | Số 0–9 |

→ **Ràng buộc mạnh chưa được khai thác:** vị trí 5 của biển Mercosur chỉ nhận **10 chữ cái A–J**; vị trí 4, 6, 7 luôn là số; vị trí 1–3 luôn là chữ. Greedy decode hiện tại không áp bất kỳ ràng buộc nào, nên hoàn toàn có thể sinh ra chuỗi dài ≠ 7 hoặc sai kiểu ký tự — đây là **lỗi miễn phí có thể loại bỏ** bằng constrained decoding.

- Slide 18–20 (LR/HR size): scatter Width–Height cho thấy LR tập trung ở **W ≈ 25–68, H ≈ 12–28**, HR ở **W ≈ 55–163, H ≈ 26–61** — khớp với thống kê đo được (LR trung bình 47,9×17,2 ở Scenario B; HR 129,8×45,7).

### 5.4. Degradation HR → Synthetic LR

Pipeline ([transforms.py:47-60](src/data/transforms.py#L47-L60)): `GaussianBlur | MotionBlur` (p=0,7) → `GaussNoise | MultiplicativeNoise` (p=0,7) → `ImageCompression` quality 20–50 (p=0,5) → `Downscale` scale 0,3–0,5 (p=0,5).

Đối chiếu với tỉ lệ HR/LR thật đo được (1,96× ở Scenario A, 2,53× ở Scenario B), dải `Downscale(0.3–0.5)` ⇒ 2,0–3,3× là **được chọn khớp với dữ liệu**, không phải tham số tuỳ tiện.

> **Nhưng slide 45 (HR original | Real LR | Synthetic LR) cho thấy khoảng cách domain rõ rệt:** ảnh Real LR nhoè mượt, mất gần hết nét ký tự; ảnh Synthetic LR vẫn giữ được cạnh sắc và chỉ bị vỡ hạt kiểu nén. Nói cách khác **synthetic LR "dễ" hơn real LR**. 50% dữ liệu train là synthetic, nên đây là một nguồn sai lệch train/test cần theo dõi — đáng làm ablation: (a) chỉ real LR, (b) real + synthetic (hiện tại), (c) real + synthetic với degradation mạnh hơn.

### 5.5. Data Augmentation

Slide 46 minh hoạ 8 phép biến đổi: Affine, Perspective, Random Brightness Contrast, Hue Saturation Value, Coarse Dropout, Rotate — và **Horizontal Flip / Vertical Flip bị gạch chéo đỏ**, tức slide đã chủ động loại hai phép này (lật ảnh phá huỷ khả năng đọc chữ).

Repo ([transforms.py:6-35](src/data/transforms.py#L6-L35)) dùng: `Resize(32,128)` → `Affine(scale 0.95–1.05, translate 5%, rotate ±5°, p=0.5)` → `Perspective(0.02–0.05, p=0.3)` → `RandomBrightnessContrast(p=0.5)` → `HueSaturationValue(p=0.3)` → `Rotate(±10°, p=0.3)` → `ChannelShuffle(p=0.3)` → `CoarseDropout(2–5 lỗ, 4–8 px, p=0.3)` → `Normalize` → `ToTensorV2`.

Đối chiếu: repo **khớp với slide** ở cả 6 phép được giữ lẫn 2 phép flip bị loại. Khác biệt thật sự duy nhất là **`ChannelShuffle`** — repo có, slide không nêu.

Hai điểm đáng đưa vào ablation:
1. **`ChannelShuffle(p=0.3)`** hoán vị kênh màu ở 30% số lần. Màu nền biển là tín hiệu phân biệt layout Mercosur/Brazilian, kéo theo pattern ký tự ở §5.3 → augmentation này phá tín hiệu có ích.
2. **Augmentation được áp dụng độc lập cho từng frame** trong `__getitem__` → 5 frame cùng track nhận affine/perspective/rotate/dropout **khác nhau**, phá vỡ sự nhất quán không gian giữa các frame. Điều này đi ngược lại chính giả định mà STN + Attention Fusion dựa vào. Nên thử dùng chung tham số augment cho cả track (`A.ReplayCompose` hoặc `additional_targets`).

---

## 6. Cấu hình huấn luyện và kết quả

### 6.1. Config (slide 48, khớp với `configs/config.py`)

| Tham số | Giá trị |
|---|---|
| Batch size | 64 |
| Learning rate | 5e-4 |
| Epochs | 30 |
| Seed | 42 |
| ResNet layers | 34 |
| Transformer heads | 8 |
| Transformer layers | 3 |
| Transformer dim (feed-forward) | 2048 |
| Transformer dropout | 0.1 |
| Metric | Exact Match |

Repo bổ sung (không có trên slide): `AdamW` với `weight_decay=1e-4`, `OneCycleLR` (`max_lr = LEARNING_RATE`), gradient clipping `5.0`, mixed precision (`GradScaler` + `autocast`) và bước scheduler chỉ chạy khi optimizer thực sự step ([trainer.py:104-112](src/training/trainer.py#L104-L112)).

### 6.2. Kết quả baseline (slide 50)

| Model | Accuracy (%) |
|---|---|
| ResNet + Transformer (ResTran) | **75,80** |
| ResNet + Transformer + STN | **78,70** |

**STN đóng góp +2,90 điểm** với chi phí chỉ 0,28M tham số (0,9% model). Đây là con số baseline cần vượt qua.

Đặt trong bối cảnh slide 6 ("SOTA hiện đạt 50–60%"): con số 78,70% **không so sánh trực tiếp được** với SOTA vì đây là accuracy trên **val nội bộ (Scenario B)**, còn 50–60% là trên test set của bài báo challenge. Không nên coi baseline đã vượt SOTA.

---

## 7. Đánh giá tổng hợp và hướng cải tiến (bám ResTran)

### 7.1. Những gì baseline làm đúng

1. **Weight sharing + fusion ở mức đặc trưng** thay vì nối 5 frame theo kênh — giữ số tham số cố định và cho phép trọng số fusion phụ thuộc nội dung.
2. **Fusion theo từng pixel** (softmax trên trục frame cho mỗi vị trí không gian) đúng với bản chất nhoè cục bộ của ảnh LR.
3. **Sửa stride ResNet thành (2,1)** — chi tiết nhỏ nhưng quyết định: nếu giữ stride mặc định, chuỗi chỉ còn 4 timestep, không đủ cho 7 ký tự.
4. **Khởi tạo STN bằng identity** — tránh phá ảnh ở giai đoạn đầu.
5. **Split val chỉ từ Scenario B** — đã xác nhận bằng số liệu rằng test cùng phân phối với Scenario B (JPEG, AR 2,50–3,50, ~6,8 px/ký tự), khác hẳn Scenario A.

### 7.2. Hướng cải tiến, xếp theo tỉ lệ lợi ích/chi phí

| # | Đề xuất | Chi phí | Kỳ vọng |
|---|---|---|---|
| 1 | **Constrained decoding theo pattern** (§5.3): ép độ dài 7, ép kiểu ký tự theo vị trí, vị trí 5 Mercosur chỉ A–J | Vài chục dòng, **không train lại** | Loại bỏ trực tiếp một lớp lỗi hệ thống |
| 2 | **Đổi công thức confidence** (§4.2) sang min/tích thay vì mean-of-max | Vài dòng, không train lại | Cải thiện Confidence Gap → thứ hạng tie-break |
| 3 | **Augment nhất quán theo track** (`ReplayCompose`) | Sửa `__getitem__` | Không phá giả định của STN + Fusion |
| 4 | **Bỏ `ChannelShuffle`**, giảm kích thước `CoarseDropout` | 1 dòng | Giữ tín hiệu màu/nét ký tự |
| 5 | **Dùng `corners` của 10.000 track Scenario A** để giám sát STN (loss phụ trên `theta`) hoặc warp sẵn ảnh | Trung bình | Nguồn giám sát sẵn có đang bị bỏ hoàn toàn |
| 6 | **Dùng `plate_layout`** làm head phụ (multi-task) | Trung bình | Giúp model chọn đúng pattern ký tự |
| 7 | **Đặt `SPLIT_RATIO = 0.8`** để val 2.000 mẫu như slide | 1 dòng | Giảm nhiễu đo từ ±1,5 xuống ±0,9 điểm |
| 8 | **Bổ sung metric CER** bên cạnh exact match | Nhỏ | Tín hiệu mịn hơn để so ablation |
| 9 | **Kiểm tra `pretrained=True` cho ResNet34** | 1 dòng | Backbone đang train from scratch trên 38.002 mẫu ảnh 32×128 |

### 7.3. Rủi ro cần lưu ý

- **`--submission-mode` train trên toàn bộ 20.000 track và không có validation** → checkpoint "best" thực chất là checkpoint cuối cùng ([trainer.py:232-236](src/training/trainer.py#L232-L236)). Phải chốt số epoch bằng chế độ có val trước, rồi mới chạy submission-mode.
- **Giới hạn 25 lượt nộp** toàn giải → mọi quyết định phải dựa trên val nội bộ, không dùng leaderboard để dò tham số.
- **`AttentionFusion` hard-code 5 frame** — ràng buộc cứng giữa data và model, cần nhớ nếu mở rộng.
- ~~Đường dẫn trong config chưa khớp dữ liệu thực tế~~ — **đã sửa trong baseline này**: `DATA_ROOT = "dataset/data/train"`, `TEST_DATA_ROOT = "dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public"`, `VAL_SPLIT_FILE = "splits/val_tracks.json"`. Dữ liệu nằm ngay trong `dataset/` của folder này.

---

*Ghi chú phạm vi: slide deck còn khoảng 38 trang cuối (52–89) trình bày một module Super-Resolution như Future of Work. Phần đó là một model riêng biệt, nằm ngoài phạm vi baseline ResTran và không được đưa vào tài liệu này.*
