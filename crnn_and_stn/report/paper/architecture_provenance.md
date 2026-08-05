# Truy vết kiến trúc S1 — mỗi khối trong hình/paper đến từ dòng code nào

> **Mục đích**: khi viết Method hoặc vẽ lại hình, không được mô tả theo trí nhớ.
> Tài liệu này neo từng khối của **S1 (phương pháp đề xuất)** vào **file:dòng**
> trong `src/`, kèm **lý do thiết kế** trích từ chính comment trong code.
>
> Dùng cùng lúc với:
> - Hình: [`paperLatex/Methonology/make_system_overview.py`](../../paperLatex/Methonology/make_system_overview.py)
> - Bài: `paperLatex/2026_fisat_NhutMinh.tex` §2 Methodology
> - Công thức loss: [loss_formula_corrected.md](loss_formula_corrected.md)
> - Số liệu: [multi_seed_results.md](../baseline1_crnn_stn/multi_seed_results.md)

---

## 0. 🚨 Nguồn chân lý là banner log, không phải report

Report dễ lẫn giữa **J1 / S1 / S4** vì chúng dùng chung phần lớn cờ. Trước khi
mô tả kiến trúc, luôn mở banner in ở đầu log của đúng run:

```
results/multi-seed/s1_mf_sr_ocr/log_s1_seed42.txt
```

```
STN     : True
SE      : True
SR      : True (scale=2, lambda_sr=0.1, edge=0.0, perceptual=0.0, multi_frame=True)
DCN     : True | Fusion: attention | Frames: 5
STN pool: (4, 8) | Width /8 -> T=32
Decode  : constrained (layouts=LLLNLNN,LLLNNNN, beam=16)
EMA     : True (decay=0.999)
Image size : 32x128 | LR domain match: True
Epochs  : 60 | Batch: 32 | LR: 0.0008 | Grad Accum: 2 | WeightDecay: 0.0001
Seed    : 42 | cudnn.benchmark: False (deterministic: True)
Backbone: base=64, blocks=(2,2,2,2,2), channels=(64,128,256,256,512),
          res_scale=0.1, norm=group
📊 Model params: 29,577,214
```

Ba chỗ **khác biệt** giữa 3 cấu hình — nhầm là mô tả sai model:

| | SR | `sr_scale` | `width_downsample` | `T` | Params |
|---|:---:|:---:|:---:|---:|---:|
| **S1** 🎯 | ✅ | **2** | 8 (mặc định) | **32** | 29,577,214 |
| **J1** | ❌ | — | 8 | **16** | 29,442,700 |
| **S4** | ✅ | **1** | **4** | 32 | 29,549,470 |

---

## 1. Bảng truy vết theo từng khối

### Stage 1 — STN (affine θ, identity init)

| | |
|---|---|
| **Code** | [`src/models/components.py:399`](../../src/models/components.py) lớp `STNBlock`; `pool_size=(4,8)` mặc định L411; identity init L442–446; áp dụng ở [`src/models/crnn.py:138-142`](../../src/models/crnn.py) qua `affine_grid` + `grid_sample` |
| **Vì sao** | Comment L400–409: xoay và tịnh tiến là **đại lượng không gian**; descriptor pool về `1×1` không biểu diễn được *"biển nghiêng 5° và lệch trái"*, nên FC head gần như chỉ học được zoom. Vì thế giữ lưới `4×8`, conv stride 1 + max-pool. |
| **Bẫy** | `pool_size=(1,1)` là **topology cũ**, chỉ để checkpoint trước khi sửa vẫn eval đúng. J1 *lịch sử* (76.88%) chạy `(1,1)`; J1 *mới* (80.45%) chạy `(4,8)`. Không đặt 2 số này chung cột. |

### Stage 2 — DCNv2 align

| | |
|---|---|
| **Code** | `components.py:249` lớp `DCNAlignment`; frame giữa làm reference L305; offset/mask head zero-init L282–285; **deform kernel identity-init** L292–300; gọi ở `crnn.py:171-174` |
| **Vì sao** | L250–257: STN chỉ áp **một** affine toàn cục cho mỗi frame, nên lệch cục bộ (motion blur, rolling shutter, biển cong) còn nguyên. L286–289: **chỉ zero offset là chưa đủ** — kernel `3×3` ngẫu nhiên sẽ thay frame RGB bằng tổ hợp ngẫu nhiên ngay forward đầu tiên, tức module bắt đầu bằng việc **phá ảnh** thay vì truyền qua. |

### Stage 3 — Multi-frame SR ×2

| | |
|---|---|
| **Code** | `components.py:319` lớp `FrameSR`; temporal mean qua 5 frame rồi concat L382–391; PixelShuffle L361–365; cộng residual học được lên **bilinear base** L376–379 và L395; gọi ở `crnn.py:179-184` với `scale=2` |
| **Vì sao** | L320–333: ở ~2 px/ký tự, SR **đơn ảnh** không thể thêm thông tin nó chưa từng nhận — chỉ có thể **bịa**, mà bịa thì không phân biệt được với đọc sai. 5 frame lệch sub-pixel mới thật sự mang thêm tín hiệu, và đó là chỗ tín hiệu đó đi vào. Bilinear base giúp module khởi đầu **gần identity**, giống STN. |

### Stage 4 — ResBlock backbone (GroupNorm + SE)

| | |
|---|---|
| **Code** | `components.py:98` lớp `ResBackbone`; mặc định stage `(64,128,256,256,512)` và `(2,2,2,2,2)` ở `crnn.py:27` và `crnn.py:38`; stride schedule cho `width_downsample=8` L139–142; `make_norm` L40–55; `ResidualBlock` L58 với `res_scale=0.1`; `SqueezeExcitation` L19 |
| **SE bật hay tắt** | **BẬT.** Preset `stable` set `BACKBONE_USE_SE=True` ([`train.py:149`](../../train.py)), mặc định `config.py:57` cũng True, và banner S1 in `SE : True`. |
| **Vì sao GroupNorm** | `make_norm` L41–50: EDSR bỏ BatchNorm vì BN ghìm dải động mà decoder SR cần. Lập luận đó **không chuyển sang được** backbone OCR — đầu ra của nó là chuỗi feature nhiều chiều nuôi BiLSTM: **không có normalization nào thì scale feature trôi và gradient nổ thành NaN** ngay khi SR head nhân đôi kích thước không gian. GroupNorm chuẩn hoá **theo từng mẫu** nên khôi phục được ổn định mà không dính vấn đề batch nhỏ của BN. |

### Stage 5 — Attention fusion

| | |
|---|---|
| **Code** | `components.py:190` lớp `AttentionFusion`; scorer MLP L211–216; softmax qua frame + weighted sum L233–236 |
| **Vì sao** | Các frame trong cùng track khác nhau về độ mờ/nén, nên trọng số tin cậy **học được** hơn hẳn `avg`/`max` — hai chế độ không tham số giữ ở L225–228 để làm ablation. |

### Stage 6 — BiLSTM + CTC head

| | |
|---|---|
| **Code** | `crnn.py:124-131` (2 lớp, hidden 256, bidirectional); head L132–136; `log_softmax` L200; height pool về 1 dòng ở `components.py:183-186` để width thành trục thời gian |
| **`T = 32`** | width 128 → SR ×2 thành 256 → chia `width_downsample=8`. Xem `config.py:69` và [s1_proposed_mf_sr_ocr.md §1](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md) |

### CTC decode có ràng buộc layout

| | |
|---|---|
| **Code** | [`src/utils/postprocess.py:28`](../../src/utils/postprocess.py) `DEFAULT_PLATE_LAYOUTS`; `constrained_beam_decode:283`; `BEAM_WIDTH=16` ở `config.py:112`, trainer đọc ở `trainer.py:174` |
| **Vì sao** | Cả 20.000 nhãn đều **7 ký tự** và chỉ thuộc 2 layout, khác nhau **duy nhất ở vị trí 5** (`postprocess.py:221-224`) → khoá cứng được **6/7** vị trí lớp chữ/số. Chỉ can thiệp lúc decode: **không thêm tham số, không train lại**. |
| **Đóng góp đo được** | 0–2 track ([s1_proposed_mf_sr_ocr.md §6](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md)) — nhỏ nhưng miễn phí compute. |

---

## 2. Hàm mất mát

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}} \cdot \mathcal{L}_{\text{SR}}, \qquad \lambda_{\text{SR}} = 0.1$$

$$\mathcal{L}_{\text{SR}} = \frac{1}{N}\sum_{i=1}^{N} \left\| I_{\text{SR}}^{(i)} - \mathrm{Warp}_{\mathrm{sg}[\theta_i]}\!\left(I_{\text{HR}}^{(i)}\right) \right\|_1$$

| Thành phần | Code | Vì sao viết như vậy |
|---|---|---|
| Nhân λ cho **cả cụm** | [`trainer.py:336`](../../src/training/trainer.py), λ từ L158 | Viết 3 số hạng song song chỉ **tương đương** khi $\lambda_{\text{Perc}} = \lambda_{\text{SR}} \times \alpha$ — xem [loss_formula_corrected.md](loss_formula_corrected.md) "Lỗi 4" |
| **ℓ₁**, không phải Smooth L1 | [`losses.py:84`](../../src/training/losses.py) gọi `F.l1_loss` | Thuần tuý là gọi sai tên trong bản thảo cũ. Giữ ℓ₁, **không đổi code** — đổi là phải chạy lại 39 giờ GPU |
| Chỉ warp **vế target** | `trainer.py:269-272` | `trainer.py:265-268`: SR chạy **sau** STN nên $I_{\text{SR}}$ đã nằm trong khung đã nắn; warp nó lần nữa là biến đổi hình học **hai lần** |
| **stop-gradient** `sg[θ]` | `.detach()` tại `trainer.py:270` | Không chặn thì nhánh SR có thể giảm $\mathcal{L}_{\text{SR}}$ bằng cách **kéo STN** về hình học dễ tái tạo nhất, tức để mục tiêu phụ chi phối mục tiêu chính |
| Mốc bilinear | `trainer.py:278-285`, log thành `sr_loss_bilinear` | Chỉ mình `sr_loss` không trả lời được câu duy nhất đáng hỏi: **nhánh học có hơn nội suy không** |

### ⚠️ Hai số hạng KHÔNG được vẽ / không được nêu như đang bật

| Số hạng | Code | Trạng thái |
|---|---|---|
| Perceptual VGG ($\alpha$) | `losses.py:87-88` | $\alpha = 0$ ở **mọi** cấu hình có error bar — banner in `perceptual=0.0` |
| Edge Sobel ($\beta$) | `losses.py:86` | $\beta = 0$ ở mọi cấu hình báo cáo — banner in `edge=0.0` |

Vẽ hoặc in chúng mà không chú thích → reviewer hiểu nhầm **79.95% có perceptual loss**.

---

## 3. Dữ liệu và siêu tham số huấn luyện

| Mục | Giá trị | Nguồn |
|---|---|---|
| Ảnh vào | `32 × 128` | `config.py:28-29` |
| Track | 5 LR (+5 HR ở train) | [dataset_overview.md](../summary_project/dataset/dataset_overview.md) |
| Lớp CTC | 37 = 36 ký tự + blank | dataset_overview.md §5 |
| Train / Val | 19.001 / 999 (val 100% Scenario-B) | dataset_overview.md §2 |
| Optimizer | AdamW, lr `8e-4`, wd `1e-4` | `config.py:35,39` |
| Schedule | warmup 5%, cosine → 5% | `config.py:120`, preset `train.py:139-140` |
| Grad clip | 2.0 | `config.py:44` |
| Batch | 32 × accum 2 = **64 hiệu dụng** | banner |
| Epoch | 60, early stop patience 18 | banner, `config.py:122` |
| EMA | decay 0.999, **eval trên trọng số EMA** | `config.py:117`, `trainer.py:176`, L375 |
| Chọn checkpoint | theo **Val Exact Match**, không theo val loss | [s1_proposed_mf_sr_ocr.md §8](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md) — val loss chạm đáy ep 16–20 rồi tăng, còn val acc còn lên tới ep 24–32 |

---

## 4. Checklist trước khi in bất kỳ mô tả kiến trúc nào

- [ ] Mở banner log của **đúng run** (S1 ≠ J1 ≠ S4)
- [ ] Đối chiếu `Model params` với số trong report
- [ ] Kiểm `SE`, `sr_scale`, `width_downsample`, `T` — 4 chỗ dễ chép nhầm nhất
- [ ] Xác nhận `edge=0.0, perceptual=0.0` trước khi in công thức loss
- [ ] Nếu sửa hình: sửa `make_system_overview.py` rồi chạy lại, **không** sửa file ảnh
