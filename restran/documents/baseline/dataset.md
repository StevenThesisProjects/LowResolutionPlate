# Phân tích tập dataset dùng cho ResTranOCR

Tài liệu này mô tả **dữ liệu thực tế trên đĩa** (`dataset/`) và **cách code trong `src/data/` biến dữ liệu đó thành input của `ResTranOCR`**. Mọi con số thống kê đều được đo trực tiếp từ dữ liệu (toàn bộ 20.000 track cho nhãn; mẫu ngẫu nhiên 400 track/nhóm cho kích thước ảnh).

Nguồn code liên quan:
- [src/data/dataset.py](src/data/dataset.py) — `MultiFrameDataset`
- [src/data/transforms.py](src/data/transforms.py) — augmentation / degradation
- [configs/config.py](configs/config.py) — đường dẫn, charset, kích thước ảnh
- [src/models/restran.py](src/models/restran.py), [src/models/components.py](src/models/components.py) — model tiêu thụ dữ liệu

---

## 1. Tổng quan bài toán dữ liệu

ICPR 2026 LRLPR: mỗi mẫu là một **track = 5 frame liên tiếp** của cùng một biển số, độ phân giải rất thấp. Nhãn là **chuỗi 7 ký tự** của biển số. Đây là bài toán *multi-frame OCR*, không phải single-image OCR: 5 frame cùng nhãn, khác nhau về blur / góc nhìn / nhiễu nén, và model phải hợp nhất (fusion) chúng.

---

## 2. Cấu trúc dữ liệu thật trên đĩa

```
dataset/
├── data/train/
│   ├── Scenario-A/
│   │   ├── Mercosur/   track_XXXXX/   (5.000 track)
│   │   └── Brazilian/  track_XXXXX/   (5.000 track)
│   └── Scenario-B/
│       ├── Mercosur/   track_XXXXX/   (8.000 track)
│       └── Brazilian/  track_XXXXX/   (2.000 track)
├── Pa7a3Hin-test-public/Pa7a3Hin-test-public/   (1.000 track, chỉ lr-*)
└── TKzFBtn7-test-blind/TKzFBtn7-test-blind/     (3.000 track, chỉ lr-*)
```

Một track train:

```
track_04378/
├── annotations.json
├── hr-001.png … hr-005.png   (5 frame high-resolution)
└── lr-001.png … lr-005.png   (5 frame low-resolution — cùng cảnh, cùng biển)
```

`annotations.json` (Scenario-A):

```json
{
  "plate_layout": "Mercosur",
  "plate_text": "ACT0F03",
  "corners": { "lr-001.png": {"top-left":[2,2], "top-right":[36,3],
                              "bottom-right":[35,17], "bottom-left":[2,16]}, ... }
}
```

Scenario-B chỉ có `plate_layout` + `plate_text`, **`corners` là dict rỗng**.

### 2.1. Dữ liệu này được sinh ra như thế nào (slide 14–17)

Trước khi đọc các con số, cần biết cơ chế thu thập — nó giải thích gần hết những khác biệt đo được ở §2.2.

**Slide 14 — LR và HR là hai lần chụp khác nhau, không phải một ảnh ở hai độ phân giải.** Pipeline của ban tổ chức: YOLOv11 detect biển số trong video, nhưng crop ở **hai vùng khác nhau của khung hình** — `LR region` (đoạn đường xa camera) và `HR region` (đoạn gần camera). Mỗi track vì thế gồm 5 frame LR lấy lúc xe còn ở xa và 5 frame HR lấy lúc xe đã tới gần.

Kiểm chứng trên 1.200 track (300 track/nhóm) — bề rộng frame 5 trừ frame 1:

| Nhóm | LR | HR |
|---|---|---|
| A / Mercosur | **+3,59 px** (tăng ở 98% track) | **−7,52 px** (tăng ở 1% track) |
| A / Brazilian | +3,76 px (95%) | −7,49 px (0%) |
| B / Mercosur | +2,27 px (84%) | −7,15 px (1%) |
| B / Brazilian | +2,43 px (87%) | −7,24 px (1%) |

Xu hướng **ngược dấu** xác nhận hai chuỗi thuộc hai khoảng thời gian khác nhau. Hệ quả trực tiếp: **không có tương ứng `lr-00i` ↔ `hr-00i`** (đo bằng PSNR: ghép đúng chỉ số 12,75 dB vs ghép chéo 12,72 dB ở Scenario-B — không khác nhau). Điều này quan trọng với mọi phương án dùng HR làm target pixel-wise — xem [super-reslution.md §2.4](documents/baseline/super-reslution.md).

**Slide 15–16 — vì sao A và B khác domain.** Scenario A lấy từ một dataset đã công bố (Nascimento et al., 2025), quay trong điều kiện **có kiểm soát: ban ngày, không mưa**, và có sẵn annotation `corners`. Scenario B thu mới riêng cho cuộc thi, **cùng camera nhưng đặt hướng khác**, dải điều kiện môi trường rộng hơn hẳn (slide có cảnh đêm và cảnh mưa), và **không cung cấp `corners`**. Đây là nguyên nhân gốc của mọi chênh lệch đo được ở §2.2 — khác góc camera kéo theo khác aspect ratio và cách crop, khác điều kiện môi trường kéo theo khác mức nhiễu.

**Slide 17 và 43 — test lấy từ đâu.** Ban tổ chức nói thẳng: *"Public Test Set (1,000 Tracks): tracks sourced from Scenario B"* và *"Blind Test Set (3,000+ Tracks): tracks sourced from Scenario B"*. Tức kết luận ở §2.2 không chỉ là suy luận từ phân phối ảnh — nó được ghi rõ trong tài liệu chính thức.

Slide 43 còn nêu hai điểm ảnh hưởng tới thiết kế giải pháp:
- *"HR images are provided exclusively for training, so participants may explore image enhancement techniques."*
- *"Participants may use any method to aggregate predictions across the five LR images (e.g., majority voting, confidence-based selection, temporal modeling)."* → hợp nhất 5 frame **không bắt buộc** phải ở mức đặc trưng như ResTran đang làm; OCR từng frame rồi vote là một họ giải pháp hợp lệ.

### 2.2. Khác biệt giữa hai kịch bản (rất quan trọng)

| Thuộc tính | Scenario-A | Scenario-B | Test (public + blind) |
|---|---|---|---|
| Số track | 10.000 | 10.000 | 1.000 + 3.000 |
| Định dạng ảnh | `.png` | `.jpg` | `.jpg` |
| Có `corners` | ✅ 10.000/10.000 | ❌ 0/10.000 | — |
| Có frame HR | ✅ | ✅ | ❌ (chỉ `lr-*`) |
| Aspect ratio LR | 1,47 – 3,16 (TB **2,16**) | 2,50 – 3,47 (TB **2,77**) | 2,50 – 3,50 (TB **2,78**) |
| Bề rộng LR (px) | 25 – 60 (TB 40,7) | 35 – 60 (TB 48,1) | 38 – 60 (TB 48,0) |
| px / ký tự (LR) | ≈ 5,8 | ≈ 6,8 | ≈ 6,8 |
| Nhãn trùng lặp | Nhiều (xem §3.2) | Không (100% unique) | — |

**Kết luận then chốt:** phân phối ảnh của tập test **trùng khít Scenario-B** (cùng đuôi `.jpg`, cùng dải AR bị chặn ở 2,50–3,50, cùng bề rộng ≤ 60px, cùng px/ký tự). Scenario-A là một domain khác (crop lỏng hơn, AR thấp hơn, PNG không mất mát). Điều này **biện minh trực tiếp** cho logic split trong code: validation chỉ lấy từ Scenario-B ([dataset.py:141](src/data/dataset.py#L141)) → val là proxy trung thực cho test.

### 2.3. Quan hệ giữa test-public và test-blind

- `test-public` (1.000 track) là **tập con byte-identical** của `test-blind` (3.000 track): 250/250 file kiểm tra bằng MD5 đều giống hệt, 1.000/1.000 track_id trùng nhau.
- ID track không đụng nhau giữa train và test: `public ∩ train = 0`, `blind ∩ train = 0`.
- Toàn bộ không gian ID là 1…23.000 cho 24.000 track (train 20.000 + blind 3.000, public nằm trong blind), **basename `track_XXXXX` là duy nhất toàn cục** — nên việc `val_tracks.json` chỉ lưu basename ([dataset.py:161](src/data/dataset.py#L161)) là an toàn, không có va chạm.

---

## 3. Thống kê nhãn

### 3.1. Charset và pattern

- Charset thực tế: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 ký tự) → khớp đúng `Config.CHARS`, `NUM_CLASSES = 37` (36 + blank CTC).
- **Độ dài nhãn = 7 cho 19.999/20.000 track.**
- Hai layout tương ứng hai pattern cố định:

| Layout | Pattern | Số track | Ví dụ |
|---|---|---|---|
| Mercosur | `LLL D L DD` (chữ-chữ-chữ-số-chữ-số-số) | 13.000 | `ACT0F03` |
| Brazilian | `LLL DDDD` | 7.000 | `AYA1516` |

Pattern này **tuyệt đối nhất quán** (13.000/13.000 và 6.999/7.000) → có thể khai thác làm ràng buộc hậu xử lý (xem §7).

- Tần suất ký tự lệch mạnh: `A` (9.526) và `B` (8.694) áp đảo vì ký tự đầu biển số Brazil chủ yếu là A/B; các chữ hiếm nhất `K` (1.156), `N` (1.206), `L` (1.228). Chữ số phân bố khá đều (6.135 – 7.023).

### 3.2. Trùng lặp nhãn (rủi ro leakage)

| Nhóm | Track | Nhãn unique | Trùng nhiều nhất |
|---|---|---|---|
| Scenario-A / Mercosur | 5.000 | 4.344 | `BDN5E52` ×4 |
| Scenario-A / Brazilian | 5.000 | 2.603 | `BCN4394` ×11 |
| Scenario-B / Mercosur | 8.000 | 8.000 | — |
| Scenario-B / Brazilian | 2.000 | 2.000 | — |

- Tổng: 20.000 track / **16.947 biển số unique**. Toàn bộ phần trùng nằm ở Scenario-A (một biển số được ghi lại nhiều lượt).
- **Giao nhãn giữa 4 nhóm = 0 cho mọi cặp.** Vì val lấy từ Scenario-B (nhãn unique, không giao với A), **không có label leakage train→val** — kể cả khi Scenario-A trùng lặp nội bộ nặng. Nếu sau này đổi sang split ngẫu nhiên trên toàn bộ dữ liệu thì leakage sẽ xuất hiện ngay.

### 3.3. Nhãn bất thường

- `data/train/Scenario-A/Brazilian/track_06546` có `plate_text = "ARO3383 "` (thừa một dấu cách cuối). Code lọc ký tự ngoài `char2idx` ([dataset.py:265](src/data/dataset.py#L265)) nên dấu cách bị bỏ → target vẫn đúng 7. Không cần sửa, nhưng nên biết khi thống kê độ dài nhãn.

---

## 4. Thống kê ảnh

### 4.1. Kích thước (mẫu 400 track/nhóm, frame đầu tiên)

| Nhóm | LR: W × H (TB) | HR: W × H (TB) | Tỉ lệ HR/LR (bề rộng) |
|---|---|---|---|
| A / Mercosur | 40,3 × 18,4 | 85,9 × 36,7 | **1,96×** |
| A / Brazilian | 41,1 × 19,1 | 88,1 × 39,0 | ~2,0× |
| B / Mercosur | 47,9 × 17,2 | 129,8 × 45,7 | **2,53×** |
| B / Brazilian | 48,3 × 17,5 | 126,2 × 44,5 | ~2,5× |
| test-public | 47,8 × 17,3 | — | — |
| test-blind | 48,3 × 17,4 | — | — |

- **Mỗi ký tự chỉ chiếm ~6,8 px bề ngang trong ảnh LR** — đây chính là độ khó cốt lõi của bài toán.
- Kích thước biến thiên **ngay trong một track**: bề rộng LR dao động trung bình 6–9,5% giữa 5 frame (xe đang tiến lại gần camera). Fusion vì thế phải xử lý được lệch scale nhẹ giữa các frame — đúng vai trò của `STNBlock`.
- Ảnh là **ảnh màu thật** (chỉ ~1/100 frame gần như grayscale) → giữ 3 kênh là hợp lý; nhưng xem cảnh báo `ChannelShuffle` ở §7.

### 4.2. Ảnh HR dùng để làm gì

HR **không được dùng làm target super-resolution**. Code dùng HR như một **nguồn tăng cường dữ liệu**: HR được làm suy biến (degradation) thành "LR tổng hợp" rồi huấn luyện như một mẫu độc lập ([dataset.py:199-206](src/data/dataset.py#L199-L206)).

Pipeline degradation ([transforms.py:47-60](src/data/transforms.py#L47-L60)): `GaussianBlur | MotionBlur` (p=0,7) → `GaussNoise | MultiplicativeNoise` (p=0,7) → `ImageCompression` quality 20–50 (p=0,5) → `Downscale` scale 0,3–0,5 (p=0,5).

Đối chiếu với dữ liệu thật: `Downscale(0.3–0.5)` ứng với hệ số thu nhỏ 2,0–3,3×, trong khi tỉ lệ kích thước HR/LR đo được là **1,96× (Scenario-A)** và **2,53× (Scenario-B)**. Tức là **dải degradation được chọn nằm đúng thang kích thước của dataset** — đây là một điểm mạnh của pipeline, không phải tham số đặt bừa. `ImageCompression` cũng hợp lý vì test/Scenario-B là JPEG.

> ⚠️ Nhưng đừng đọc tỉ lệ 1,96× / 2,53× như "mức suy biến" theo nghĩa LR = HR bị thu nhỏ. Theo §2.1, LR và HR là **hai lần chụp ở hai vị trí khác nhau trên đường**, nên đó là tỉ lệ **khoảng cách tới camera**, không phải một phép downscale. Sự trùng khớp về con số vẫn là lý do tốt để giữ dải `Downscale(0.3–0.5)`, nhưng nó không chứng minh rằng ảnh synthetic giống ảnh LR thật — slide 45 cho thấy ngược lại (§7, mục synthetic dễ hơn real).

---

## 5. Pipeline dữ liệu trong code

### 5.1. Quét và lập chỉ mục

`MultiFrameDataset.__init__` glob `root_dir/**/track_*` (đệ quy, đã `sorted`) nên tự động gom cả 4 nhóm A/B × Mercosur/Brazilian. Với mỗi track, `_index_samples` đọc nhãn theo thứ tự ưu tiên `plate_text` → `license_plate` → `text`, rồi tạo:

| Chế độ | Mẫu sinh ra mỗi track |
|---|---|
| `train` | 2 mẫu: (a) 5 frame `lr-*`, `is_synthetic=False`; (b) 5 frame `hr-*`, `is_synthetic=True` |
| `val` | 1 mẫu: 5 frame `lr-*` |
| `test` (`is_test=True`) | 1 mẫu: 5 frame `lr-*`, nhãn rỗng |

→ **Số mẫu train = 2 × số track train.** Với split mặc định: 19.001 track train → **38.002 mẫu/epoch → ~190.000 lượt đọc ảnh mỗi epoch** (đây là điểm nghẽn I/O chính, `NUM_WORKERS=10`).

Toàn bộ 20.000 track đều có đúng 5 `lr-*` và 5 `hr-*` (đã kiểm tra 100%), nên giả định "luôn 5 frame" của code là đúng với dữ liệu hiện tại.

### 5.2. Chia train/val (Scenario-B priority)

`_load_or_create_split` ([dataset.py:102-163](src/data/dataset.py#L102-L163)):

1. Nếu `full_train=True` (cờ `--submission-mode`) → toàn bộ 20.000 track vào train, không có val.
2. Nếu tồn tại `splits/val_tracks.json` → nạp lại; nhưng **tự động tạo lại** nếu val rỗng hoặc val không chứa track Scenario-B nào (và tổng > 100 track).
3. Khi tạo mới: lọc các track có `"Scenario-B"` trong đường dẫn (10.000 track), shuffle với `random.Random(seed)`, lấy `(1 - split_ratio)` làm val.

Với `SPLIT_RATIO = 0.9`, `SEED = 42`:

| | Track | Mẫu |
|---|---|---|
| Train | 19.001 (toàn bộ A + 9.001 của B) | **38.002** (19.001 LR thật + 19.001 HR-degraded) |
| Val | **999** (100% Scenario-B) | **999** (chỉ LR thật, không augment) |

(Số liệu đo bằng cách chạy thật pipeline, không phải suy luận.)

Split có tính tất định (glob đã `sorted` + seed cố định) và được lưu ra JSON để tái sử dụng.

> **Vì sao 999 chứ không phải 1.000:** `val_size = max(1, int(len(scenario_b) * (1 - split_ratio)))` ([dataset.py:148](src/data/dataset.py#L148)) gặp sai số dấu phẩy động — `1 − 0.9 = 0.09999999999999998`, nên `int(10000 × 0.0999…) = 999`. Tương tự `SPLIT_RATIO = 0.8` cho 1.999 chứ không phải 2.000. Lệch 1 track không ảnh hưởng thực tế; giữ nguyên để trùng hành vi baseline gốc.

### 5.3. Biến đổi ảnh

Thứ tự trong `__getitem__`: đọc BGR → RGB → **degradation (nếu là mẫu synthetic)** → **transform** → stack.

| Pipeline | Nội dung |
|---|---|
| `get_train_transforms` (mặc định, `full`) | Resize 32×128 → Affine(scale 0,95–1,05; translate 5%; rotate ±5°, p=0,5) → Perspective(0,02–0,05, p=0,3) → RandomBrightnessContrast(p=0,5) → HueSaturationValue(p=0,3) → Rotate(±10°, p=0,3) → **ChannelShuffle(p=0,3)** → CoarseDropout(2–5 lỗ, 4–8 px, p=0,3) → Normalize(0,5/0,5) → ToTensor |
| `get_light_transforms` (`--aug-level light`) | Resize + Normalize + ToTensor |
| `get_val_transforms` (val & test) | Resize + Normalize + ToTensor |

Chuẩn hoá `mean=std=0,5` → đưa về khoảng [-1, 1].

### 5.4. Output của Dataset và collate

`__getitem__` trả `(images [5,3,32,128], target [L], target_len, label_text, track_id)`.
`collate_fn` stack ảnh thành `[B,5,3,32,128]`, **nối phẳng target thành 1-D** và trả `target_lengths` — đúng định dạng `nn.CTCLoss` cần ([trainer.py:44](src/training/trainer.py#L44), `blank=0`, `zero_infinity=True`).

`char2idx` map ký tự → 1…36, chỉ số 0 dành cho blank. Ở chế độ test, target được đặt là `[0]` giả và bị bỏ qua.

---

## 6. Dữ liệu chảy qua ResTranOCR như thế nào

```
[B, 5, 3, 32, 128]
  → view → [B*5, 3, 32, 128]
  → STN (affine_grid + grid_sample, khởi tạo identity)   # căn chỉnh từng frame
  → ResNet34 (layer3/layer4 stride đổi thành (2,1))      # giảm chiều cao, giữ chiều rộng
  → [B*5, 512, 1, 16]
  → AttentionFusion (softmax trên trục 5 frame)          # [B, 512, 1, 16]
  → squeeze + permute → [B, 16, 512] + PositionalEncoding
  → TransformerEncoder (3 lớp, 8 head, ff 2048)
  → Linear(512 → 37) → log_softmax → [B, 16, 37]
```

Vài hệ quả về mặt dữ liệu:

- **Chuỗi CTC có T = 16 timestep** (128 → conv1 s2 → 64 → maxpool s2 → 32 → layer2 s2 → 16 → layer3/layer4 giữ nguyên bề rộng). Với nhãn 7 ký tự, T = 16 thoả điều kiện CTC (T ≥ L + số ký tự lặp liền kề) nhưng **biên khá hẹp: ~2,3 timestep cho mỗi ký tự**. Nếu tăng độ dài nhãn hoặc giảm `IMG_WIDTH` thì cần kiểm tra lại.
- **`AttentionFusion` hard-code `num_frames = 5`** ([components.py:75](src/models/components.py#L75)); mọi track trong dataset đều đúng 5 frame nên hiện tại an toàn, nhưng đây là ràng buộc cứng giữa data và model.
- **Resize 32×128 (AR 4,0) từ ảnh gốc AR ~2,8** → ảnh bị **kéo ngang ~1,4×**. Việc này nhất quán giữa train/val/test nên model học được, và còn có lợi: kéo ngang làm tăng số timestep trên mỗi ký tự. Chỉ cần lưu ý STN phải học bù méo này.

---

## 7. Nhận xét, rủi ro và hướng khai thác

**Điểm mạnh của pipeline hiện tại**

1. Split val chỉ từ Scenario-B là lựa chọn đúng — ban tổ chức nói rõ test lấy từ Scenario-B (§2.1) và số liệu đo được cũng xác nhận cùng phân phối (§2.2); nhãn B không giao với A nên không leakage (§3.2).
2. Dải degradation HR→LR khớp với tỉ lệ HR/LR thật đo được (§4.2).
3. Nhân đôi dữ liệu train bằng HR-degraded là cách tận dụng miễn phí 20.000 track HR mà test không có.

**Rủi ro / điểm cần cân nhắc**

1. **Augmentation được áp dụng độc lập cho từng frame.** Trong `__getitem__`, mỗi frame đi qua `self.transform` riêng → 5 frame của cùng một track nhận affine/perspective/rotate/dropout **khác nhau**. Điều này phá vỡ sự nhất quán không gian giữa các frame — đúng cái mà `AttentionFusion` và STN đang cố khai thác. Nên thử: dùng chung một tham số augment cho cả 5 frame (Albumentations `additional_targets` hoặc `ReplayCompose`).
2. **`ChannelShuffle(p=0,3)`** hoán vị kênh màu 30% số lần. Ảnh là ảnh màu thật, và màu nền biển số là tín hiệu phân biệt layout Mercosur/Brazilian (kéo theo pattern ký tự). Đây là augmentation phá tín hiệu, nên đưa vào ablation.
3. **`CoarseDropout` 2–5 lỗ kích thước 4–8 px trên ảnh 32×128**: một ký tự chỉ rộng ~18 px sau resize, nên một lỗ 8 px có thể che gần nửa ký tự. Kết hợp với nhãn CTC không có cơ chế "ký tự bị che" → có thể tạo nhiễu nhãn. Cân nhắc giảm kích thước lỗ.
4. **`corners` của 10.000 track Scenario-A chưa được dùng.** Đây là dữ liệu giám sát sẵn có cho việc rectify 4 góc biển số — có thể (a) huấn luyện STN có giám sát (loss phụ trên `theta`) thay vì để STN tự học, hoặc (b) warp sẵn ảnh về hình chữ nhật chuẩn. Đây là nguồn tín hiệu chưa khai thác lớn nhất trong dataset.
5. **Pattern ký tự cố định 100%** (`LLLDLDD` / `LLLDDDD`) nhưng greedy CTC decode ([postprocess.py](src/utils/postprocess.py)) không ràng buộc gì — có thể sinh chuỗi dài ≠ 7 hoặc sai kiểu ký tự. Hậu xử lý theo pattern (hoặc beam search có ràng buộc) là cải thiện gần như miễn phí. Có thể phân loại layout bằng chính pattern dự đoán, hoặc thêm một head phụ dự đoán `plate_layout` (nhãn này có sẵn trong `annotations.json` nhưng hiện đang bị bỏ qua).
6. **Mất cân bằng layout**: Mercosur 13.000 vs Brazilian 7.000; riêng Scenario-B là 8.000 vs 2.000, nên val 999 track (rút từ B) sẽ có tỉ lệ ~80/20 nghiêng về Mercosur. Nên báo cáo accuracy tách theo layout thay vì chỉ một con số tổng.
7. ~~Đường dẫn trong config chưa khớp dữ liệu thực tế~~ — **đã sửa trong baseline này**: `DATA_ROOT = "dataset/data/train"`, `TEST_DATA_ROOT = "dataset/Pa7a3Hin-test-public/Pa7a3Hin-test-public"`, `VAL_SPLIT_FILE = "splits/val_tracks.json"`. Dữ liệu nằm ngay trong `dataset/` của folder này.
8. **Nếu một track thiếu file `lr-*`**, `_index_samples` vẫn thêm mẫu với `paths = []` → `torch.stack([])` sẽ crash. Hiện tại không track nào rơi vào trường hợp này (đã kiểm tra 20.000/20.000), nên đây chỉ là rủi ro tiềm ẩn khi thêm dữ liệu mới.
9. **Metric hiện chỉ có exact-match accuracy**; `validate()` trả key `cer` khi không có val_loader nhưng không bao giờ tính CER. Với biển 7 ký tự, CER cho tín hiệu mịn hơn nhiều để so sánh các ablation.

**Đánh giá về đo lường**

- Val ~1.000 mẫu → sai số chuẩn của accuracy ≈ 1,5 điểm phần trăm ở mức 50%. Chênh lệch < 3 điểm giữa hai cấu hình ablation **không kết luận được**. Nếu cần so sánh tinh, tăng tỉ lệ val hoặc dùng CER.
- `--submission-mode` train trên toàn bộ 20.000 track và **không có val** → checkpoint "best" thực chất là checkpoint cuối cùng ([trainer.py:232-236](src/training/trainer.py#L232-L236)). Nên chốt số epoch bằng chế độ có val trước, rồi mới chạy submission-mode.
