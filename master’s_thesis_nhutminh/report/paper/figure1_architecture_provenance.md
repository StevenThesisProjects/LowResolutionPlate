# Figure 1 (System Overview) — truy vết kiến trúc về code

> Tài liệu này trả lời đúng một câu hỏi: **mỗi khối trong hình kiến trúc của paper lấy từ đâu ra, và tại sao công thức lại viết như vậy.**
>
> Dùng khi: reviewer hỏi "cái này ở đâu ra", hoặc khi sửa hình/sửa Method mà cần kiểm tra có còn khớp code không.
>
> - Hình: `paperLatex/Methonology/SystemOverview.png` (+ `.pdf`)
> - Script sinh hình: `paperLatex/Methonology/make_system_overview.py` (block `PROVENANCE` ở đầu file là bản rút gọn của tài liệu này)
> - Bài: `paperLatex/2026_fisat_NhutMinh.tex`, mục 2 (Methodology)

---

## 1. 🚨 Nguồn chuẩn là banner log, KHÔNG phải report

Hình vẽ cấu hình **S1** — phương pháp đề xuất. J1 và S4 dùng **gần hết** cùng bộ cờ, chỉ khác vài chỗ, nên đọc report rồi vẽ theo trí nhớ là rất dễ vẽ nhầm sang model khác.

Nguồn đúng duy nhất: banner in ở đầu `results/multi-seed/s1_mf_sr_ocr/log_s1_seed42.txt`

```
STN        : True
SE         : True
SR         : True (scale=2, lambda_sr=0.1, edge=0.0, perceptual=0.0, multi_frame=True)
DCN        : True | Fusion: attention | Frames: 5
STN pool   : (4, 8) | Width /8 -> T=32
Decode     : constrained (layouts=LLLNLNN,LLLNNNN, beam=16)
EMA        : True (decay=0.999)
Image size : 32x128 | LR domain match: True
Epochs     : 60 | Batch: 32 | LR: 0.0008 | Grad Accum: 2
WeightDecay: 0.0001
Seed       : 42 | cudnn.benchmark: False (deterministic: True)
Backbone   : base=64, blocks=(2,2,2,2,2), channels=(64,128,256,256,512), res_scale=0.1, norm=group
📊 Model params: 29,577,214
```

`29,577,214` khớp chính xác với con số trong [../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md) → xác nhận đúng checkpoint đã sinh ra **79.95% ± 0.15**.

### Ba chỗ dễ vẽ nhầm sang J1 / S4

| | S1 (vẽ trong hình) | J1 | S4 |
|---|---|---|---|
| Nhánh SR | ×2, multi-frame | **không có** | ×1 |
| `T` | **32** | 16 | 32 (qua `width_downsample=4`) |
| `width_downsample` | 8 (mặc định) | 8 | **4** |

---

## 2. Bảng truy vết từng khối

| # | Khối trong hình | Code | Vì sao thiết kế như vậy |
|---|---|---|---|
| 1 | **STN** — affine θ, identity init | `src/models/components.py:399` (`STNBlock`); `pool_size=(4,8)` L411; identity init L442–446; áp dụng ở `src/models/crnn.py:138-142` (`affine_grid` + `grid_sample`) | L400–409: xoay và tịnh tiến là **đại lượng không gian**. Pool localization map về `1×1` thì descriptor mất thông tin vị trí, FC head gần như chỉ học được zoom. `(1,1)` chỉ giữ lại để chạy checkpoint cũ. |
| 2 | **DCNv2 align** — về frame giữa | `components.py:249` (`DCNAlignment`); reference = frame giữa L305; offset/mask head zero-init L282–285; **deform kernel identity-init L292–300**; gọi ở `crnn.py:171-174` | L250–257: một affine toàn cục cho mỗi frame không diễn tả được lệch **cục bộ** (motion blur, rolling shutter, biển cong). L286–289: chỉ zero offset là chưa đủ — kernel 3×3 ngẫu nhiên sẽ thay ảnh RGB bằng một tổ hợp ngẫu nhiên ngay forward đầu tiên, tức module bắt đầu bằng việc **phá ảnh**. |
| 3 | **Multi-frame SR ×2** | `components.py:319` (`FrameSR`); temporal mean + concat L382–391; PixelShuffle L361–365; cộng lên bilinear base L376–379, L395; gọi ở `crnn.py:179-184` với `scale=2` | L320–333: ở ~2 px/ký tự, SR **đơn ảnh** không thể thêm thông tin nó chưa từng nhận — chỉ có thể bịa, mà bịa thì không phân biệt được với đọc sai. 5 frame lệch sub-pixel mới thực sự mang thêm tín hiệu. Cộng lên bilinear base để module khởi đầu ≈ identity, giống STN. |
| 4 | **ResBlock backbone** — GroupNorm + SE | `components.py:98` (`ResBackbone`); stage `(64,128,256,256,512)` & `(2,2,2,2,2)` mặc định ở `crnn.py:27,38`; stride schedule L139–142; `make_norm` L40–55; `ResidualBlock` L58 (`res_scale=0.1`); `SqueezeExcitation` L19 | `make_norm` L41–50: bỏ BatchNorm theo lối EDSR (BN bó hẹp dynamic range mà decoder SR cần), **nhưng** backbone OCR nuôi một BiLSTM — không chuẩn hoá gì thì scale đặc trưng trôi và gradient nổ thành **NaN** ngay khi SR nhân đôi kích thước. GroupNorm chuẩn hoá theo từng mẫu nên không dính vấn đề batch nhỏ vốn là lý do bỏ BN. |
| 5 | **Attention fusion** — trọng số theo frame | `components.py:190` (`AttentionFusion`); scorer MLP L211–216; softmax + weighted sum L233–236 | Các frame trong cùng track khác nhau về mờ/nén/ánh sáng. Trọng số học được thắng avg/max — hai chế độ không tham số vẫn giữ ở L225–228 làm ablation. |
| 6 | **BiLSTM + CTC head** | `crnn.py:124-131` (2 lớp, hidden 256, bidirectional); head L132–136; `log_softmax` L200; pool height → 1 dòng ở `components.py:183-186` | Height ép về 1 nên **trục width trở thành trục thời gian** của CTC. `T = 128×2/8 = 32` — xem `configs/config.py:69` và [s1_proposed_mf_sr_ocr.md](../baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md) mục 1. |
| — | **CTC decode** — layout-constrained, beam 16 | `src/utils/postprocess.py:28` (`DEFAULT_PLATE_LAYOUTS`); `constrained_beam_decode:283`; `BEAM_WIDTH=16` ở `configs/config.py:112`, đọc ở `trainer.py:174` | 20.000 nhãn đều dài 7 ký tự và chỉ thuộc 2 layout, **khác nhau đúng một vị trí thứ 5** (`postprocess.py:221-224`) → khoá cứng 6/7 vị trí lớp chữ/số. Chỉ can thiệp ở decode: không thêm tham số, không train lại. Đóng góp đo được: **0–2 track**. |

---

## 3. Công thức loss — vì sao viết như trong hình

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}} \cdot \mathcal{L}_{\text{SR}}, \qquad \lambda_{\text{SR}} = 0.1$$

$$\mathcal{L}_{\text{SR}} = \frac{1}{N}\sum_{i=1}^{N} \left\| I_{\text{SR}}^{(i)} - \mathrm{Warp}_{\mathrm{sg}[\theta_i]}\!\left(I_{\text{HR}}^{(i)}\right) \right\|_1$$

| Thành phần | Code | Vì sao |
|---|---|---|
| `λ_SR` nhân **cả cụm** SR | `trainer.py:336`, λ đọc ở L158 (`--lambda-sr 0.1`) | Nếu paper viết 3 số hạng song song thì chỉ tương đương khi `λ_Perc = λ_SR × α`. Xem [loss_formula_corrected.md](loss_formula_corrected.md) mục "Lỗi 4". |
| **ℓ₁ thuần**, không phải Smooth L1 | `src/training/losses.py:84` (`F.l1_loss`) | Thuần tuý là sai tên trong bản thảo cũ. Giữ ℓ₁, **không** đổi code — đổi hàm mục tiêu là phải chạy lại toàn bộ 39 giờ GPU. |
| **Chỉ warp vế target** `I_HR` | `trainer.py:269-272` | SR chạy **sau** STN nên `I_SR` **đã nằm trong khung đã nắn**. Warp nó lần nữa là biến đổi hình học **hai lần**; không warp vế nào thì so hai hệ toạ độ khác nhau, SR head chỉ học được cách làm mờ. |
| **Stop-gradient** `sg[θ]` | `.detach()` ở `trainer.py:270` | Nếu để gradient chảy, nhánh SR có thể giảm `L_SR` bằng cách **kéo STN về phép biến đổi dễ tái tạo nhất** thay vì phép nắn đúng — tức mục tiêu phụ chi phối mục tiêu chính. Có stop-gradient, nhánh SR chỉ được làm ảnh nét hơn **trong khung mà STN đã chọn**. |
| Mốc bilinear (không vẽ) | `trainer.py:278-285`, log thành `sr_loss_bilinear` | Chỉ `sr_loss` thì không biết nhánh học được có hơn nội suy hay không — đó là câu hỏi duy nhất biện minh cho chi phí của nhánh SR. |

---

## 4. Hai thứ CỐ TÌNH không vẽ

| Thành phần | Có trong code | Vì sao không vẽ |
|---|---|---|
| Perceptual loss VGG (`α`) | `losses.py:87-88` | Banner ghi `perceptual=0.0`. Vẽ vào → reviewer hiểu nhầm 79.95% được tạo ra **có** perceptual loss. Chỉ khảo sát ở 1 ablation 1-seed, chưa xác nhận multi-seed. |
| Edge loss Sobel (`β`) | `losses.py:86` | Banner ghi `edge=0.0`. Không ảnh hưởng bất kỳ con số nào được báo cáo. |

Nguyên tắc: **đã bỏ thì bỏ nhất quán** — không để lại dấu vết ở Method hay hình. Xem [loss_formula_corrected.md](loss_formula_corrected.md) mục 3.

---

## 5. ⚠️ Lỗi đã phát hiện và sửa (2026-08-05)

**Thiếu Squeeze-and-Excitation.** Bản hình đầu tiên và mục Stage 4 của Methodology đều mô tả backbone chỉ có "ResBlock + GroupNorm".

Sự thật: preset `stable` bật SE (`train.py:149`; mặc định `configs/config.py:57` cũng `True`), và **cả 3 log seed đều in `SE : True`**. Module là `SqueezeExcitation` ở `components.py:19`, gắn vào `ResidualBlock` qua tham số `use_se` (L83, L92).

Đã sửa: hình ghi "GroupNorm + SE, shared", Methodology thêm "squeeze-and-excitation channel re-weighting".

📌 **Bài học**: mô tả kiến trúc phải đối chiếu **banner log**, không viết theo trí nhớ về pipeline. Cùng loại lỗi với [../../report/baseline1_crnn_stn/multi_seed_results.md](../baseline1_crnn_stn/multi_seed_results.md) mục 2 (J1 lịch sử vs J1 mới).

---

## 6. Sinh lại hình

```bash
python3 paperLatex/Methonology/make_system_overview.py
```

Xuất `SystemOverview.png` (300 dpi, đang được `.tex` dùng) và `SystemOverview.pdf` (vector, nét hơn khi in — đổi đuôi trong `\includegraphics` là dùng được).

Chữ nhỏ khi ghép vào bài → tăng `FONT_SCALE` ở đầu script rồi chạy lại.

**Quy ước vẽ** (giữ khi sửa): mọi đường là thẳng hoặc gấp khúc vuông góc, không đường chéo cắt qua block; ràng buộc stop-gradient ghi thành chú thích dưới khối `Warp` thay vì kéo đường đứt dài từ STN (bản đầu làm vậy và rất rối); mũi tên **xám** = luồng tổng hợp loss, mũi tên **đậm** = luồng dữ liệu forward.