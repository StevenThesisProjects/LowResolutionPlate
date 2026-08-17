# Super-Resolution cho ResTran — phân tích và phương án tích hợp

**Mục tiêu:** lấy phần Super-Resolution (SR) từ slide `Documents_2026-0_ICPR Challenge - Training` (trang 52–89) và xác định cách áp dụng vào **ResTran hiện tại mà không phá vỡ kiến trúc đang chạy**.

**Nguồn:** slide trang 52–89 (phần *Future of Work*); code ResTran trong repo này; số liệu dataset ở [dataset.md](dataset.md); kết quả 21 thí nghiệm ở [RUNS.md](../RUNS.md).

Toàn bộ kiến trúc SR trong tài liệu này đã được **dựng lại từ slide và chạy thật** để đo shape/số tham số — không phải ước lượng.

> 📌 **Nếu bạn đang chọn kiến trúc SR cho bài báo, đọc [§6](#6-chọn-kiến-trúc-sr-cho-bài-báo-nghiên-cứu) trước.** Các mục 1–5 viết cho mục tiêu cũ (ghép SR vào ResTran mà không phá kiến trúc đang chạy); §6 viết sau khi đã có dữ liệu thực nghiệm và trả lời câu hỏi khác: kiến trúc nào đúng về mặt khoa học.
>
> Hai đính chính so với bản đầu: **§1.4** (phần lớn "failure case" trong slide không phải hallucination) và **§2.4** (cặp LR–HR không đăng ký).

---

## 1. SR trong slide có gì

### 1.1. Ba thành phần và lý do chọn

| Thành phần | Nguồn | Vấn đề được giải quyết (theo slide) |
|---|---|---|
| **ResBlock không BatchNorm** | EDSR — [arXiv:1707.02921](https://arxiv.org/pdf/1707.02921) | BatchNorm chuẩn hoá về mean=0/std=1, **giới hạn dải động** mà SR cần để tái tạo. Bỏ BN → tăng chất lượng + **giảm 40% bộ nhớ**. |
| **Channel Attention (CALayer)** | RCAN — [arXiv:1807.02758](https://arxiv.org/pdf/1807.02758) | Các kênh không quan trọng như nhau: một số kênh mang **chi tiết tần số cao (cạnh, nét chữ)** — thứ quyết định trong SR; số khác chỉ mang cấu trúc tần số thấp. CA học cách **rescale kênh theo ngữ cảnh**. |
| **PixelShuffle** | ESPCN — [arXiv:1609.05158](https://arxiv.org/pdf/1609.05158) | Upsample trước rồi conv (bilinear/bicubic) là **tính toán lãng phí ở độ phân giải cao**. PixelShuffle trích đặc trưng ở không gian LR rồi học upsample → **rẻ hơn, nhanh hơn**, chất lượng giữ nguyên. |

### 1.2. Kiến trúc thật (từ code trên slide 66, 79, 81, 82)

```
StackedSRNet
 └─ x: [B, 5, 3, H, W] ──view──▶ [B, 15, H, W]        # 5 frame ghép theo trục KÊNH
     └─ EDSRLite
         ├─ head:        Conv(15 → nf, 3×3)
         ├─ body:        16 × ResBlock(nf) + Conv(nf,nf)      ── skip connection toàn cục ──┐
         │                 ResBlock: Conv → ReLU → Conv → CALayer(r=16) → ×res_scale + skip │
         ├─ upsampler:   Conv(nf → nf×4) → PixelShuffle(2) → ReLU        # ×2               │
         ├─ upsample_conv: Conv(nf, nf, 3×3)                                                │
         ├─ interpolate → target_size = (43, 121)                                           │
         └─ tail:        Conv(nf → 3, 3×3)  ──▶ [B, 3, 43, 121]   # MỘT ảnh HR ◀────────────┘
```

Tham số mặc định trên slide: `num_features=64`, `num_blocks=16`, `res_scale=0.1`, `num_frames=5`, `target_size=(43,121)`, CA `reduction=16`.

**Đo thật:**

| Cấu hình | Input | Output | Tham số |
|---|---|---|---|
| Mặc định trong code slide (nf=64) | `(B,5,3,17,48)` | `(B,3,43,121)` | **1,42 M** |
| Cấu hình dùng khi train (nf=128) | `(B,5,3,17,48)` | `(B,3,43,121)` | **5,66 M** |

`target_size=(43,121)` khớp đúng kích thước HR trung bình đo được của Scenario-B (129,8 × 45,7 px) → SR đang học đưa LR về đúng thang HR (~2,5×).

### 1.3. Huấn luyện và kết quả (slide 83–89)

```
num_features = 128
epochs       = 100
batch_size   = 16
Loss = w0 · MSE + w1 · Perceptual + w2 · Edge
→ 2 giờ 30 phút training
```

Đường cong (slide 83):

| Chỉ số | Train | Val |
|---|---|---|
| Loss | 0,667 (vẫn giảm) | 0,708 (**phẳng từ ~epoch 55**) |
| PSNR | 14,85 dB (vẫn tăng) | **~14,05 dB (phẳng từ ~epoch 40)** |
| SSIM | — | **~0,63** |

Kết quả định tính:
- **Thành công** (slide 84–85): 5 frame LR gần như không đọc được → SR ra `BAT·1550` sắc nét, khớp HR target.
- **Cần cân nhắc** (slide 86): `FFR4157` đọc được nhưng mềm và nhạt hơn HR rõ rệt.

### 1.4. ⚠️ Đọc lại slide 87–89: phần lớn KHÔNG phải hallucination

> Bản đầu của tài liệu này viết rằng slide 87–89 cho thấy SR "bịa lại nét chữ". Đọc kỹ từng ảnh thì **sai** — và đây là điểm quan trọng vì lập luận "SR nguy hiểm vì hallucination" đã được dùng để loại bỏ cả hướng SR.

| Slide | SR output | HR target | Thực chất |
|---|---|---|---|
| 87 | `DZW4167` | `DZW4167` | **SR khớp HR hoàn toàn.** Nhãn ghi `OZW4167` → lệch giữa **nhãn và ảnh**, không phải lỗi SR |
| 88 | `ARZ·2863` | `ARZ·2863` | Cả hai đọc đúng nhãn; khác biệt chỉ ở nét chữ, cực nhỏ |
| 89 | `BEA7028` | `BEA7G23` | **Hallucination thật** — SR đổi `G→0` và `3→8` |

Chỉ **1 trong 3** ca là hallucination thật. Hai ca còn lại SR tái tạo trung thực ảnh HR; cái "sai" nằm ở nhãn, hoặc ở chỗ ký tự vốn đã nhập nhằng ngay trên ảnh HR.

**Hệ quả:** rủi ro hallucination của SR bị đánh giá quá cao trong bản đầu. Đồng thời xuất hiện một quan sát đáng viết vào bài báo: **một phần lỗi bị quy cho SR thực ra là nhiễu nhãn của dataset**.

---

## 2. Bốn lý do KHÔNG thể ghép thẳng SR vào ResTran

Đây là phần quan trọng nhất. Nếu nối `StackedSRNet → ResTranOCR` theo cách hiển nhiên, model sẽ **hỏng** vì:

### 2.1. SR trả về 1 frame, ResTran cần đúng 5

`StackedSRNet` gộp 5 frame vào trục kênh và xuất **một** ảnh. Trong khi đó:

- `ResTranOCR.forward` nhận `[B, 5, 3, H, W]` rồi `view(b*f, c, h, w)`;
- `AttentionFusion` **hard-code `num_frames = 5`** ([components.py:75](../../src/models/components.py#L75)) và tính `batch_size = total_frames // 5`.

Đưa 1 frame vào → `batch_size` tính sai, tensor reshape sai hình. **Đây chính là chỗ "phá vỡ model"**: SR 5→1 đã làm mất chức năng của Attention Fusion — module lõi của ResTran.

### 2.2. Resize 32×128 xoá phần lớn thành quả của SR

ResTran resize **mọi** ảnh về 32×128 trước khi vào model ([transforms.py](../../src/data/transforms.py)). Nếu SR xuất 43×121 rồi mới resize:

```
SR: 17×48  ──▶  43×121  ──resize──▶  32×128
                 ↑                     ↓
          chiều cao tăng 2,5×    lại bị nén xuống 0,74×
```

**Khoảng 26% độ phân giải dọc mà SR vừa tạo ra bị vứt đi ngay ở bước resize.** Đây là lỗi tích hợp dễ mắc nhất và nó âm thầm làm SR "vô dụng" mà không báo lỗi gì.

**Cách sửa:** đặt thẳng `target_size = (32, 128)` — để SR **thay thế** phép resize bilinear, thay vì nối tiếp nó.

### 2.3. PSNR 14 dB + hallucination vs. metric exact-match

- PSNR val ~14 dB và SSIM ~0,63 là **thấp** cho một bài SR. Val phẳng từ epoch 40 trong khi train vẫn giảm → **overfit**, và giới hạn chất lượng tái tạo là có thật.
- Metric của challenge là **exact match toàn chuỗi**: SR bịa sai **một** ký tự → **cả track sai**. Slide 89 cho thấy SR có thể bịa ra nét chữ trông rất thuyết phục (`G→0`, `3→8`).

> Mức độ rủi ro này đã bị bản đầu **thổi phồng**: xem [§1.4](#14--đọc-lại-slide-8789-phần-lớn-không-phải-hallucination) — chỉ 1 trong 3 "failure case" của slide là hallucination thật, hai ca còn lại SR tái tạo trung thực ảnh HR.

⚠️ **Hệ quả:** không được dùng PSNR/SSIM để quyết định SR có tốt hay không. **Chỉ dùng accuracy exact-match trên val Scenario-B (999 mẫu)** làm tiêu chí. §2.4 cho thấy con số PSNR ở đây còn không phản ánh đúng chất lượng SR.

### 2.4. ⚠️ Cặp LR–HR **không đăng ký với nhau** — đây là lỗi tích hợp nguy hiểm nhất

Slide 14 mô tả cách sinh dữ liệu: YOLOv11 detect biển số ở **hai vùng khác nhau của khung hình** — `LR region` (xa camera) và `HR region` (gần camera) — rồi crop riêng từng vùng. Nghĩa là **`lr-*` và `hr-*` là hai lần chụp ở hai thời điểm/vị trí khác nhau**, không phải cùng một ảnh ở hai độ phân giải.

Đo trên 1.200 track (300 track mỗi nhóm), bề rộng frame 5 trừ frame 1:

| Nhóm | LR | HR |
|---|---|---|
| A / Mercosur | **+3,59 px** (tăng ở 98% track) | **−7,52 px** (tăng ở 1% track) |
| A / Brazilian | +3,76 px (95%) | −7,49 px (0%) |
| B / Mercosur | +2,27 px (84%) | −7,15 px (1%) |
| B / Brazilian | +2,43 px (87%) | −7,24 px (1%) |

Hai xu hướng **ngược dấu**: trong đoạn LR xe tiến lại gần, trong đoạn HR xe đi xa dần. Kiểm tra trực tiếp giả thiết "frame i khớp frame i" bằng PSNR giữa LR phóng to và HR:

| | Ghép theo chỉ số (i↔i) | Ghép chéo (i↔j, i≠j) |
|---|---|---|
| Scenario-A | 9,68 dB | 9,67 dB |
| Scenario-B | 12,75 dB | 12,72 dB |

**Ghép đúng chỉ số không tốt hơn ghép bừa** → không tồn tại tương ứng frame-to-frame giữa `lr-*` và `hr-*`.

Ba hệ quả:

1. **Loss pixel-wise (MSE / Perceptual) đang học trên cặp lệch.** Mọi phương án dùng `(LR → HR)` làm cặp giám sát (Mức 0 và Mức 1 ở §3) đều phải **đăng ký ảnh trước**, nếu không SR chỉ học được ánh xạ trung bình mờ.
2. **Giải thích con số PSNR 14 dB của slide 83.** Bicubic thô trên Scenario-B đã cho ~12,75 dB, nên SR chỉ hơn ~1,3 dB. Không phải vì SR yếu, mà vì **target không đăng ký được với input** — PSNR ở bài này gần như vô nghĩa. (Số 12,75 dB đo thô trên ảnh xám cùng cỡ 96×32, khác giao thức với slide nên chỉ mang tính chỉ báo về bậc độ lớn.)
3. **Về lý thuyết củng cố cho Mức 2** (§3): distillation từ teacher-HR so khớp ở không gian **đặc trưng**, không phải pixel, nên miễn nhiễm với vấn đề lệch này.

> ⚠️ **Đã thử và thất bại** (run #17, [RUNS.md](../RUNS.md)): teacher HR đạt 98,45%, `distill_loss` giảm 8 lần, nhưng accuracy **−0,90 điểm** so với mốc và CTC train loss xấu đi. Student khớp được phần cấu trúc chung của đặc trưng teacher, còn phần tần số cao phân biệt ký tự thì không — vì thông tin đó không tồn tại trong ảnh LR. Miễn nhiễm với lệch đăng ký không có nghĩa là hữu ích.

Cách đăng ký khả thi: dùng `corners` sẵn có của 10.000 track Scenario-A để warp cả LR lẫn HR về cùng một hình chữ nhật chuẩn — đây cũng là cách duy nhất tạo được cặp đăng ký cho Scenario-B (vốn không có `corners`) nếu học một mô hình dò 4 góc từ A rồi áp sang B.

---

## 3. Ba mức tích hợp — xếp theo mức độ "đụng vào model"

### Mức 0 — SR offline, **không sửa một dòng nào của ResTran**

Ý tưởng: train SR riêng, chạy inference một lần trên toàn dataset, ghi ảnh SR ra đĩa theo **đúng cấu trúc track**, rồi train ResTran như cũ với `--data-root` mới.

```
dataset_sr/data/train/Scenario-B/Mercosur/track_12602/
├── annotations.json      (copy nguyên)
├── lr-001.png … lr-005.png   ← 5 ảnh SR 32×128 (KHÔNG phải 1 ảnh)
└── hr-001.png … hr-005.png   (copy nguyên, để nhánh synthetic vẫn chạy)
```

```bash
python train.py --data-root dataset_sr/data/train -n restran_sr
```

Điều kiện bắt buộc: SR phải xuất **5 ảnh** để giữ contract. Sửa 1 dòng trong `StackedSRNet`:

```python
# out_channels = in_channels           →  ra 1 frame  (phá contract)
out_channels = in_channels * num_frames  #  ra 5 frame  (giữ contract)
...
return out.view(b, num_frames, c, H, W)
```

Dataset có sẵn `hr-001..005` cho **cả 20.000 track**, nhưng ⚠️ **không được giả định `lr-00i` tương ứng `hr-00i`** — xem §2.4. Muốn dùng HR làm target pixel-wise thì phải đăng ký (register) ảnh trước.

| Ưu | Nhược |
|---|---|
| ResTran nguyên vẹn tuyệt đối, không rủi ro | Tốn đĩa (100.000 ảnh SR, nhưng chỉ 32×128 nên rất nhẹ) |
| Chạy SR 1 lần, train ResTran bao nhiêu lần cũng được | SR và OCR tối ưu **rời nhau** — SR không biết OCR cần gì |
| Dễ so sánh A/B: cùng code, chỉ khác `--data-root` | Lỗi SR bị "đóng băng" vào dữ liệu |

> **Mẹo test nhanh 5→1:** nếu muốn thử ngay `StackedSRNet` bản gốc (5→1) mà không sửa gì, hãy **nhân bản ảnh SR ra 5 lần**. Contract `[B,5,3,32,128]` được giữ, code chạy bình thường; Attention Fusion khi đó thành phép trung bình vô hại (softmax trên 5 đặc trưng giống hệt nhau = 0,2 mỗi frame). Đây là cách đo **rất nhanh** xem SR 5→1 có giúp gì không — nhưng nhớ rằng bạn đang tắt Attention Fusion, nên kết quả kém hơn không có nghĩa SR vô dụng.

---

### Mức 1 — SR làm front-end, **giữ nguyên contract `[B,5,3,32,128]`**

`ResTranOCR` không sửa dòng nào; chỉ bọc thêm một lớp ngoài:

```python
class ResTranSR(nn.Module):
    """SR front-end + ResTran. Contract vào/ra không đổi."""
    def __init__(self, sr, restran, freeze_sr=True):
        super().__init__()
        self.sr, self.restran = sr, restran
        if freeze_sr:
            for p in self.sr.parameters():
                p.requires_grad = False

    def forward(self, x):          # x: [B, 5, 3, 16, 64]
        x = self.sr(x)             # [B, 5, 3, 32, 128]  ← 5 vào, 5 ra
        return self.restran(x)     # [B, 16, 37]  y hệt như cũ
```

**Thiết kế gọn nhất — để SR gánh luôn việc upscale:**

| | Hiện tại | Với SR |
|---|---|---|
| Dataset resize về | 32×128 (bilinear) | **16×64** |
| Upscale lên 32×128 bởi | `A.Resize` (bilinear, không học) | **PixelShuffle ×2 (có học)** |
| ResTran nhận | 32×128 | 32×128 — **không đổi** |

Vì `16×64 ×2 = 32×128` đúng chằn chặn nên **không cần `F.interpolate`**, PixelShuffle ra thẳng kích thước đích. LR thật trung bình 17×48 nên resize về 16×64 gần như không mất mát.

Chi phí đo thật (5 vào → 5 ra, target 32×128):

| Cấu hình SR | Tham số | So với ResTran 31,08 M |
|---|---|---|
| nf=64, 16 blocks | 1,43 M | +4,6% |
| nf=32, 16 blocks | 0,36 M | +1,2% |
| **nf=32, 8 blocks** | **0,21 M** | **+0,7%** |

→ Ngay cả bản đầy đủ cũng chỉ tốn thêm ~4,6% tham số. Nên **bắt đầu từ nf=32/8 blocks**: SR nặng dễ overfit (chính slide 83 đã cho thấy overfit với nf=128).

**Quy trình 3 bước (không giai đoạn nào phá model):**
1. Train SR riêng bằng cặp `(5 LR thật → 5 HR)` **đã đăng ký** (§2.4 — không ghép theo chỉ số frame), val trên Scenario-B. Nếu bỏ qua được bước đăng ký thì bỏ luôn bước 1: đi thẳng bước 3, để CTC loss tự dạy SR.
2. Đóng băng SR, train ResTran → đo accuracy. Đây là mốc so sánh.
3. Mở băng SR, fine-tune cả cụm bằng **CTC loss** với LR nhỏ (vd. 1e-5 cho SR). Lúc này SR mới học "làm sắc nét thứ mà OCR cần", chứ không phải "tối ưu PSNR".

Bước 3 là chỗ SR thực sự có giá trị: SR tối ưu theo PSNR làm ảnh *đẹp*, SR tối ưu theo CTC làm ảnh *dễ đọc* — hai thứ này khác nhau, và slide 87–89 chính là bằng chứng.

---

### Mức 2 — SR chỉ tồn tại lúc train, **inference giống hệt bây giờ**

Đây là phương án **an toàn tuyệt đối**: thêm một head SR phụ trong lúc train, **bỏ đi khi inference**. Model đem đi nộp bài y hệt ResTran hiện tại, **0 tham số phát sinh, 0 chi phí inference**.

```python
# --- chỉ khi train ---
feat   = model.backbone(x_aligned)        # đặc trưng ResNet sẵn có
sr_out = sr_head(feat)                    # decoder nhỏ, tái tạo về 32×128
loss   = ctc_loss + lam * (mse(sr_out, hr) + w_edge * edge_loss(sr_out, hr))

# --- khi inference: không gọi sr_head, ResTranOCR nguyên bản ---
```

Cơ chế: SR loss ép các tầng đầu của ResNet **giữ lại chi tiết tần số cao của nét chữ** thay vì bỏ qua chúng. Đây là regularizer, không phải thành phần kiến trúc.

**Biến thể mạnh hơn — chưng cất từ teacher HR (không cần sinh ảnh SR nào):**

1. Train một ResTran trên chính ảnh **HR** (`hr-*`, không degrade) → *teacher*.
2. Train ResTran trên ảnh LR → *student*, với loss `CTC + β · ‖feat_student − feat_teacher‖²`.
3. Nộp bài bằng student — kiến trúc không đổi một chút nào.

Cách này giải quyết đúng vấn đề gốc (LR thiếu thông tin) mà **không** phải đi qua không gian pixel, nên **miễn nhiễm với hallucination**.

> **Thí nghiệm rẻ nên làm trước mọi thứ:** train ResTran trực tiếp trên ảnh HR để biết **trần accuracy** là bao nhiêu. Nếu HR-ResTran chỉ đạt ~85% thì dù SR hoàn hảo bạn cũng không vượt được mốc đó — biết sớm để không đầu tư nhầm. Chi phí: một lần train, không viết thêm code SR nào.

---

## 4. So sánh và thứ tự đề xuất

| | Mức 0 (offline) | Mức 1 (front-end) | Mức 2 (aux loss / distill) |
|---|---|---|---|
| Sửa `ResTranOCR` | không | không | không |
| Sửa data layer | có (đường dẫn) | có (16×64) | có (trả thêm HR) |
| Tham số inference thêm | 0 | +0,21…1,43 M | **0** |
| Chi phí inference | 0 | +~5% | **0** |
| Rủi ro hallucination | cao | trung bình (thấp sau bước 3) | **không có** |
| SR biết OCR cần gì | không | có (sau fine-tune) | có |
| Cần đăng ký cặp LR–HR (§2.4) | **có, bắt buộc** | có ở bước 1 (bỏ được nếu train thẳng bằng CTC) | **không** |

**Thứ tự nên làm:**

1. **Đo trần**: train ResTran trên HR → biết mức tối đa SR có thể mang lại.
2. **Mức 2 (distillation)**: rủi ro bằng 0, không đổi model nộp bài. Nếu ăn điểm thì dừng ở đây.
3. **Mức 1**: SR nhẹ (nf=32, 8 blocks, 16×64 → 32×128), đóng băng → đo → fine-tune bằng CTC.
4. **Mức 0**: chỉ khi cần tách hẳn SR ra để tái sử dụng hoặc chạy pipeline 2 giai đoạn.

---

## 5. Rủi ro và checklist trước khi tin vào kết quả

- [ ] **ĐĂNG KÝ (register) cặp LR–HR trước khi train SR** — việc phải làm đầu tiên. `lr-00i` và `hr-00i` là hai lần chụp khác thời điểm, **không khớp nhau theo chỉ số** (§2.4, đã kiểm chứng bằng số đo). Bỏ qua bước này thì mọi loss pixel-wise đều học trên cặp lệch. Nếu không đăng ký được, **đừng dùng Mức 0/Mức 1 — chuyển thẳng sang Mức 2**.
- [ ] **Train SR bằng cặp LR thật → HR**, không phải synthetic-LR → HR. Slide 45 và [dataset.md §5.4](dataset.md) đã chỉ ra synthetic LR **dễ hơn** LR thật; SR học trên synthetic sẽ hụt khi gặp test.
- [ ] **Val SR trên Scenario-B**, vì test cùng phân phối với B (JPEG, AR 2,50–3,50, ~6,8 px/ký tự), khác Scenario-A.
- [ ] **Không dùng PSNR/SSIM để chọn model.** Chỉ dùng exact-match accuracy trên 999 mẫu val. Nhớ sai số ±1,5 điểm ở mức ~78% — cải thiện dưới 3 điểm là chưa kết luận được. Càng đúng vì cặp LR–HR không đăng ký nên PSNR gần như vô nghĩa (§2.4).
- [ ] **Kiểm tra không có resize kép**: SR phải xuất thẳng 32×128, hoặc tăng `IMG_HEIGHT/IMG_WIDTH` tương ứng. Đây là lỗi âm thầm giết chết mọi lợi ích của SR (§2.2).
- [ ] **Giữ đúng 5 frame** xuyên suốt, nếu không `AttentionFusion` sẽ tính sai batch (§2.1).
- [ ] **Theo dõi overfit của SR**: slide 83 cho thấy val PSNR phẳng từ epoch 40 với nf=128. Bắt đầu bằng model nhỏ.
- [ ] **Đối chiếu chi phí**: SR chạy trên 5 frame × 38.002 mẫu mỗi epoch. Nếu train chậm hẳn, chuyển sang Mức 0 (chạy SR một lần rồi cache).
- [ ] Nếu dùng Perceptual loss (VGG) như slide: nhớ VGG được train trên ảnh tự nhiên 224×224, áp lên ảnh biển số 32×128 có thể không hợp. **Edge loss thường hữu ích hơn cho nét chữ** — cân nhắc tăng `w2` và giảm `w1`.

---

## 6. Chọn kiến trúc SR cho bài báo nghiên cứu

> Phần này viết sau khi đã chạy 21 thí nghiệm (xem [RUNS.md](../RUNS.md)). Mục tiêu khác các phần trên: không phải "ghép SR vào ResTran mà không phá vỡ gì", mà **chọn kiến trúc SR đúng cho một đóng góp khoa học**.

### 6.1. Vấn đề cốt lõi: slide dùng linh kiện SR đơn ảnh cho bài toán đa khung

| Thành phần | Nguồn | Bản chất |
|---|---|---|
| ResBlock bỏ BN | EDSR (2017) | **đơn ảnh** |
| Channel Attention | RCAN (2018) | **đơn ảnh** |
| PixelShuffle | ESPCN (2016) | **đơn ảnh** |

Yếu tố "đa khung" duy nhất là **nối 5 frame theo trục kênh ở đầu vào** — tức *early fusion*, và nó ngầm giả định 5 frame **đã căn chỉnh không gian**.

Chúng không căn chỉnh. Đo trên 1.200 track ([dataset.md §2.1](dataset.md)): bề rộng biển tăng **+2,3 đến +3,8 px** qua 5 frame ở **84–98%** số track. Một conv 3×3 ở lớp đầu không thể căn chỉnh đặc trưng lệch nhau vài pixel.

Đây là lý do bản dựng lại theo slide cho kết quả trung tính (run #19–#21): nó **không có cơ chế nào khai thác dịch chuyển dưới-pixel** — thứ duy nhất khiến đa khung có giá trị.

### 6.2. Bằng chứng thực nghiệm: đa khung đáng +20,66 điểm

| | Accuracy | Nguồn |
|---|---|---|
| 1 frame (nhân bản ×5) | **56,53%** | run #18 |
| 5 frame, fusion mức đặc trưng, **không căn chỉnh** | **77,19%** | run #9 |
| 5 frame ảnh HR (trần) | **98,45%** | run #15 |

Attention Fusion lấy được +20,66 điểm **mà không hề căn chỉnh tường minh**, chỉ bằng softmax theo từng vị trí không gian. Câu hỏi nghiên cứu tự nhiên và chưa ai trả lời trên dataset này: **căn chỉnh tường minh có lấy thêm được không?**

### 6.3. Họ kiến trúc phù hợp

Không phải EDSR/RCAN mà là họ **video / burst SR có căn chỉnh**:

| Kiến trúc | Cơ chế căn chỉnh | Đánh giá |
|---|---|---|
| **EDVR** | PCD — deformable conv kim tự tháp, xếp tầng; + TSA attention thời gian–không gian | **Phù hợp nhất**: thiết kế cho chuỗi ngắn, căn chỉnh ngầm nên không cần optical flow |
| **BasicVSR++** | lan truyền hai chiều + deformable dẫn hướng bằng flow | Mạnh, nhưng thiết kế cho chuỗi dài — 5 frame chưa tận dụng hết |
| **TDAN** | deformable alignment thuần, không flow | Lựa chọn nhẹ hơn EDVR |
| **Burst SR (DBSR)** | fusion có attention cho burst lệch dưới-pixel | Gần cơ chế thu thập của dataset này nhất |

Cho biển số cụ thể, slide 7 đã dẫn **PLNET, LCOFL, SR3** từ bài báo challenge ([arXiv 2505.06393](https://arxiv.org/pdf/2505.06393)) — nên trích dẫn. Họ SR cho văn bản (TSRN / TextZoom) có loss nhận biết chuỗi ký tự, đáng tham khảo cho phần thiết kế loss.

### 6.4. Rào cản riêng của dataset — và là một đóng góp

Đo được (§2.4): ghép `lr-00i` với `hr-00i` cho PSNR **12,75 dB**, ghép chéo `i≠j` cho **12,72 dB**. Không có tương ứng chỉ số. Nguyên nhân ở slide 14: LR và HR crop từ **hai vùng khác nhau của khung hình**, tức hai thời điểm khác nhau khi xe chạy qua.

**Mọi bài SR đều giả định cặp LR–HR đã đăng ký. Dataset này thì không.**

| Cách xử lý | Nhận xét |
|---|---|
| **Đăng ký bằng homography** | Biển số là **vật thể phẳng** → homography là mô hình đúng, và Scenario-A có sẵn nhãn `corners` cho 10.000 track. **Khuyến nghị chính** — biến SR có giám sát thành hợp lệ, và bản thân quy trình đăng ký là một đóng góp |
| Tự giám sát đa khung | Dùng 5 frame LR giám sát lẫn nhau, không cần HR. Tránh hẳn vấn đề nhưng trần thấp hơn |
| Chỉ dùng CTC dẫn hướng | Đã thử ở run #19–#21 → trung tính. Tín hiệu quá gián tiếp để dạy một module đặt ở đầu mạng |

### 6.5. Đề xuất cụ thể

**Kiến trúc:** EDVR rút gọn — PCD alignment (căn 4 frame về frame giữa) → TSA fusion → khối tái tạo EDSR + PixelShuffle. Giữ nguyên phần tái tạo của slide, **chỉ thay early-fusion bằng alignment tường minh**. Đó là biến đổi tối thiểu và đúng biến số cần khảo sát.

**Huấn luyện:** đăng ký cặp LR–HR bằng homography từ `corners`, rồi `L1 + edge loss`, cộng `CTC loss` dẫn hướng.

**Đánh giá:** ⚠️ **không dùng PSNR/SSIM làm tiêu chí chính.** Bicubic thô đã cho ~12,75 dB còn SR của slide chỉ ~14,05 dB — chênh 1,3 dB gần như vô nghĩa khi cặp không đăng ký. Dùng **recognition rate** làm metric chính, PSNR/SSIM chỉ báo cáo phụ.

### 6.6. Đóng góp thực nghiệm đã có sẵn

Từ 21 thí nghiệm đã chạy ([RUNS.md](../RUNS.md)):

1. **Trần HR = 98,45%** — định lượng mức tối đa mà bất kỳ phương pháp cải thiện ảnh nào có thể đạt tới.
2. **Giá trị đa khung = +20,66 điểm** — chứng minh 5 frame chứa thông tin bổ sung thật.
3. **Cặp LR–HR không đăng ký** — phát hiện và định lượng bằng PSNR ghép chéo.
4. **Khoảng cách layout là hiện tượng của độ phân giải**, không phải bản chất biển số: Mercosur/Brazilian chênh 27,3 điểm ở ảnh LR nhưng chỉ 4,9 điểm ở ảnh HR.
5. **Bốn kết quả âm tính có kiểm soát**: early-fusion SR (−5,10 và −2,85 sau khi tách biến), distillation từ teacher HR (−0,90), ImageNet pretraining (−2,80), cân bằng lớp theo layout (−0,95).
6. **Nhiễu nhãn trong đánh giá định tính** (§1.4).

---

## 7. Tóm tắt một câu

SR trong slide là **EDSR (bỏ BN) + Channel Attention (RCAN) + PixelShuffle**, ghép 5 frame theo trục kênh để xuất **một** ảnh HR 43×121 — và chính hai điểm đó (**5→1** và **43×121**) là hai chỗ sẽ phá vỡ ResTran nếu nối thẳng. Muốn dùng SR mà giữ nguyên model: **cho SR xuất 5 frame ở đúng 32×128**, hoặc tốt hơn nữa là **chỉ dùng SR như một loss phụ lúc train rồi bỏ đi khi inference** — khi đó model đem nộp không khác gì bản hiện tại. Và trước tất cả: **cặp `lr-*`/`hr-*` không đăng ký với nhau** (§2.4), nên hướng pixel-wise chỉ đáng theo nếu bạn chịu bỏ công warp ảnh về hệ quy chiếu chung; nếu không, Mức 2 là lựa chọn duy nhất có cơ sở.
