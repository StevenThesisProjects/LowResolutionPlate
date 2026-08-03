# Nhật ký Tối ưu hóa MultiFrame-LPR (ICPR 2026 LRLPR Challenge)

Tài liệu này ghi chú lại toàn bộ những thay đổi kiến trúc, siêu tham số (hyperparameters), và kết quả huấn luyện trong nỗ lực vượt qua giới hạn độ chính xác của bài báo gốc (tác giả).

---

## 1. Tối ưu Baseline & Regularization (V100 Preset)
**Mục tiêu:** Cải thiện cấu hình huấn luyện gốc của tác giả (vốn dễ bị overfitting và thiếu regularization).
**Các thay đổi chính:**
- Sửa file `configs/config.py`: Tạo `apply_v100_preset`.
- Đổi scheduler từ `onecycle` sang `cosine_restarts` để mô hình hội tụ từ từ.
- Thêm các kỹ thuật chống Overfitting: `WEIGHT_DECAY = 3e-4`, `LABEL_SMOOTHING = 0.04`, `HEAD_DROPOUT = 0.08`, `TRANSFORMER_DROPOUT = 0.15`.
- Gỡ bỏ phép thuật tăng nét (sharpening) của OpenCV vì nó tạo ra nhiễu (artifacts) có hại cho chữ lờ mờ.

**Kết quả ghi nhận:**
- Mô hình đạt đỉnh ở mức **Val Acc: 74.47% - 75.08%**.
- Nhận xét: Các phương pháp SR truyền thống (OpenCV) không mang lại hiệu quả trên tập dữ liệu này. Cần một phương pháp SR học sâu (Deep Learning).

---

## 2. Tích hợp Supervised Super-Resolution (Lần 1)
**Mục tiêu:** Thay thế bộ lọc OpenCV bằng một mạng SR có khả năng học, được giám sát bởi ảnh gốc chất lượng cao (HR images).
**Các thay đổi chính:**
- `src/models/components.py`: Thêm `SuperResolutionBlock` (4 khối Residual, 64 channels).
- `src/models/restran.py`: Tích hợp khối SR vào luồng dữ liệu trước khi qua STN.
- `src/data/dataset.py`: Load thêm ảnh `hr-*.png` làm nhãn mục tiêu (target) cho SR.
- `src/training/trainer.py`: Cập nhật hàm loss tổng = `CTC_Loss + λ_sr * L1_Loss` (trong đó λ giảm dần theo Cosine Decay).

**Kết quả ghi nhận:**
- Epoch 23: Train Loss: 0.4369 | Val Loss: 0.2795 | **Val Acc: 74.77%**
- Nhận xét: Train loss giảm tốt nhưng Val Loss tăng chậm (Overfitting). Việc dùng Pixel L1 Loss khiến ảnh SR có xu hướng bị mờ (safe average blur) thay vì tạo ra góc cạnh rõ nét. Không bứt phá qua mốc 75%.

---

## 3. Thử nghiệm Perceptual Loss & Sửa lỗi Beam Search
**Mục tiêu:** Ép mạng SR học các chi tiết quan trọng (cạnh, nét chữ) thay vì chỉ so màu pixel, đồng thời bật các kỹ thuật tối đa hóa điểm số lúc đánh giá.
**Các thay đổi chính:**
- `src/training/trainer.py`: Thêm Feature Loss (Self-Distillation), cho ảnh qua mạng ResNet đang train để so sánh đặc trưng.
- `src/data/transforms.py`: Tăng cường Augmentation (`GridDistortion`, tăng kích thước lỗ của `CoarseDropout`).
- `configs/config.py`: Bật `USE_VAL_TTA = True` và `USE_BEAM_DECODE = True`.

**Sự cố 1: Val Acc rớt xuống 0.00%**
- Nguyên nhân: Lỗi toán học nghiêm trọng trong mã nguồn thuật toán Beam Search cũ (`_beam_search_single` trong `src/utils/postprocess.py`). Thuật toán thực hiện cộng đại số xác suất thay vì dùng `Log-Sum-Exp` và đảo lộn logic của Blank token.
- Khắc phục: Viết lại chuẩn xác logic CTC Prefix Beam Search.

**Sự cố 2: Val Acc phục hồi lên 40% rồi rớt thê thảm xuống 20%**
- Nguyên nhân (Catastrophic Forgetting): Ý tưởng Self-Distillation bị sai sót ở chỗ *dòng chảy gradient (gradient flow)*. Khi ép SR tính loss dựa trên ResNet, gradient đã chạy ngược làm hỏng chính trọng số của ResNet. ResNet tự biến đổi thành "đồ ngốc" chỉ để làm giảm loss của SR.
- Khắc phục: **Gỡ bỏ hoàn toàn Perceptual Feature Loss**. Quay trở lại sử dụng Pixel L1 Loss thuần túy.

---

## 4. Trạng thái Hiện tại (Supervised SR v2 Tối ưu)
**Cấu hình chốt lại:**
1. **Kiến trúc:** ResTran34 + STN + SuperResolutionBlock.
2. **Loss:** CTC Loss (Label smoothing 0.04) + Pixel L1 Loss (trọng số giảm dần).
3. **Data Augmentation:** Rất mạnh (Affine, Perspective, GridDistortion, CoarseDropout lớn).
4. **Decoding:** Test-Time Augmentation (TTA) + Beam Search (Width=10) đã được sửa lỗi hoàn toàn.

**Mục tiêu kỳ vọng:** Với Data Augmentation đủ mạnh để kìm hãm Overfitting, cùng với TTA và Beam Search giúp "vắt kiệt" độ chính xác, mô hình được kỳ vọng sẽ phá vỡ mốc 75% ban đầu. 

*(Chờ kết quả chạy lệnh `python train.py --v100 --experiment-name restran34_sr_pixel_l1` để cập nhật tiếp...)*

---

## 5. Phân tích lại dữ liệu & Khai thác prior 7-ký tự (thay đổi mới)

**Phát hiện khi soi dữ liệu thực tế (20.000 track):**
- Nhãn **luôn đúng 7 ký tự**, charset đúng `0-9A-Z` (không có ký tự bị rớt).
- Ảnh LR chỉ **~30×18px** (≈4px/ký tự) → resize 32×128 là kéo giãn ~4×. Đây là trần thông tin thật.
- ~~Chỉ ~52% track có HR khớp~~ **[ĐÍNH CHÍNH: 100% track có 5 HR khớp — 200k ảnh (100k lr + 100k hr). SR được giám sát đầy đủ. Con "52%" trước là lỗi lấy mẫu].** Việc SR chỉ đẩy 76.8→77.4% là do giới hạn thông tin ảnh ~30px, không phải thiếu giám sát.
- Test set chỉ có 5 ảnh LR `.jpg`, **không có corners/HR** ⇒ buộc giữ STN học, không rectify được.

**Thay đổi đã áp dụng:**
1. **Decode ràng buộc độ dài 7** (`PLATE_LENGTH=7`) trong `src/utils/postprocess.py`: beam search rerank ưu tiên ứng viên collapse đúng 7 ký tự, fallback về best toàn cục nếu không có. Sửa được lỗi off-by-one của CTC (nuốt/tách ký tự) — đòn bẩy lớn nhất, chi phí train = 0.
2. **`BEAM_WIDTH` 10 → 16**: đủ rộng để ứng viên độ dài 7 sống sót cho prior.
3. **`SYNTHETIC_SAMPLE_PROB` 0.4 → 0.85**: khôi phục domain-randomization cho ảnh mờ (v100 preset trước đó vô tình cắt mất 60% tín hiệu chống overfit).
4. **`BACKBONE_LR_RATIO` 0.5 → 0.7**: backbone pretrained trước bị undertrain.
5. **`SR_LOSS_WEIGHT_MIN` 0.05 → 0.1**: giữ SR như regularizer deblur ổn định thay vì tắt dần về cuối.

**Kết quả ghi nhận:** Val Acc đỉnh **76.78%** (epoch 18) — vượt cấu hình cũ, tiệm cận tác giả.

---

## 6. Layout-Aware Positional Constrained Decoding (đóng góp khoa học chính)

**Phát hiện cấu trúc (phân tích toàn bộ 20.000 nhãn):**
- Mercosur (13.000 mẫu): pattern **`LLLDLDD`** (100% thuần).
- Brazilian (6.999 mẫu): pattern **`LLLDDDD`** (100% thuần).
- Hai layout **chỉ khác nhau ở vị trí index 4**. ⇒ 6/7 vị trí có lớp ký tự (chữ/số) cố định **bất kể layout**:
  `pos: L L L D ? D D` (? = L hoặc D).

**Ý tưởng:** Vì ảnh mờ độ phân giải thấp, lỗi chủ yếu là nhầm giữa **chữ cái ↔ chữ số** giống hình dạng: `0↔O, 1↔I, 5↔S, 8↔B, 2↔Z, 4↔A`. Biết trước mỗi vị trí là chữ hay số ⇒ loại bỏ hoàn toàn nhóm lỗi này. Vì pattern đúng 100%, ràng buộc **không bao giờ loại mất đáp án đúng**.

**Triển khai (`src/utils/postprocess.py`):** CTC prefix beam search có ràng buộc lớp-vị trí — khi một beam "mọc" sang vị trí collapse thứ k, chỉ cho phép ký tự thuộc lớp hợp lệ tại vị trí k, và không vượt quá 7 ký tự. Layout-agnostic (không cần biết layout lúc test). Bật qua `USE_POS_CLASS_DECODE=True`, template `PLATE_POS_CLASSES=('L','L','L','D','LD','D','D')`.

**Đặc điểm:** Đây là thay đổi **chỉ ở khâu decode** → áp dụng ngay trên checkpoint đã train, không cần train lại. Unit test xác nhận sửa đúng `0BC1D23→ABC1D23` và `...S→...7`.

*(Chờ đo lại Val Acc với ràng buộc lớp-vị trí...)*

**Phase 1 — Công cụ đo (không train lại):** `eval_ablation.py` chấm **cùng một checkpoint** dưới các biến thể decode trên tập val, xuất bảng ablation (greedy → beam → +len7 → +posclass → +TTA). Chạy:
```
python eval_ablation.py --checkpoint results/restran34_sr_pixel_l1_best.pth --v100
```

---

## 7. Positional Head — Fixed-Length Parallel-Attention (đóng góp kiến trúc, Phase 2)

**Ý tưởng:** Ràng buộc ở mục 6 chỉ khai thác prior *lúc decode*. Ở đây model **học** prior cấu trúc: thêm một đầu nhận dạng độ dài cố định (7 vị trí) huấn luyện song song với CTC (multi-task).

**Kiến trúc (`PositionalHead` trong `components.py`):** 7 learnable position queries → MultiheadAttention over chuỗi Transformer (W'=16) → LayerNorm(residual) → Linear 36 lớp. Output `[B,7,36]` (không blank). Kiểu parallel-attention (ABINet/PARSeq) nhưng gọn cho biển định dạng cứng.

**Huấn luyện:** `Loss = CTC + λ_sr·L1 + λ_pos·CE`, với `λ_pos=0.5` (`POS_LOSS_WEIGHT`). Target vị trí dựng từ chuỗi nhãn 7 ký tự.

**Suy luận (fusion):** posterior mỗi vị trí của positional head được **mask theo `PLATE_POS_CLASSES`** (chữ/số bất khả thi → xác suất 0) rồi argmax; kết hợp với kết quả CTC beam (`fuse_ctc_positional`): đồng thuận → giữ; bất đồng → chọn đầu tự tin hơn (ưu tiên positional vì bị ràng buộc cấu trúc). Bật qua `USE_POSITIONAL_HEAD`, `POS_FUSE_AT_INFERENCE`.

**Tương thích:** `predict.py` và `eval_ablation.py` **tự phát hiện** positional head trong checkpoint → checkpoint cũ (chưa có head) vẫn load bình thường. `eval_ablation.py` bổ sung 2 dòng `positional-only` và `FUSED` khi checkpoint có head.

**Train Phase 2:**
```
python train.py --v100 --experiment-name restran34_poshead --epochs 45 --output-dir results
```

**Kết quả Phase 2:** Val Acc đỉnh **76.68%** (epoch 21) — KHÔNG cải thiện so với Phase 1 (76.78%). Positional head + fusion không phá được trần. Kết luận: model đã ngầm học format từ 20k mẫu nên ràng buộc cấu trúc gần như không kích hoạt; lỗi còn lại là **same-class trên ảnh quá mờ** → nút thắt chuyển sang **biểu diễn đặc trưng + overfitting**. (Dùng `error_analysis.py` để định lượng.)

---

## 8. Real Multi-Frame 2× Super-Resolution (Phase 3 — centerpiece)

**Vấn đề gốc:** SR cũ giám sát HR **sau khi downscale về 32×128** → chỉ là khử mờ cùng độ phân giải, vứt bỏ chi tiết tần số cao thật của HR (LR ~30px, HR ~70px). Backbone luôn đọc ảnh 32×128 nội suy.

**Thay đổi (SR thật):**
- `SuperResolutionBlock` thêm `scale`: body → **PixelShuffle upsample 2×** → tail; cộng vào base bicubic-upsample (chỉ học residual tần số cao). Output **64×256**.
- `dataset.py`: HR target resize về **kích thước gốc 2× (64×256)**, không downscale nữa; zeros sentinel cũng ở 64×256.
- Backbone giờ nhận diện ở **64×256** → seq length W'=32 (gấp đôi), ~36px/ký tự thay vì ~18px.
- `SR_UPSAMPLE=2`; `BATCH_SIZE 64→32` (2× SR ~4× activation memory trên V100 32GB).

**Cờ ablation mới:** `--no-positional-head`, `--sr-upsample {1,2}` trong `train.py` để tách riêng đóng góp.

**Train Phase 3 (SR 2× thật, sạch — tắt positional head vốn vô ích):**
```
python train.py --v100 --no-positional-head --experiment-name restran34_sr2x --epochs 45 --output-dir results
```

**Phase 3 lần 1 — phân kỳ:** Học tốt tới epoch 8 (**73.47%**, nhanh hơn các lần trước) rồi **diverge** epoch 9-10 (train 0.69→3.4→7.3, val→0%). Nguyên nhân: front-end SR 2× (PixelShuffle+residual) nhạy với LR cao; 3e-4 + grad clip 3.0 + fp16 làm trọng số trôi rồi cascade.

**Ổn định hóa (Phase 3 lần 2):**
- `LEARNING_RATE` 3e-4 → **2e-4**.
- `GRAD_CLIP` 3.0 → **1.0**.
- `WARMUP_EPOCHS` 2 → **3**.
- `MAX_TRAIN_LOSS` 50 → **15** (skip batch phân kỳ sớm).
- Trainer: **bỏ qua step khi grad norm không hữu hạn** (chặn spike fp16 làm hỏng trọng số).

Chạy lại:
```
python train.py --v100 --no-positional-head --experiment-name restran34_sr2x --epochs 45 --output-dir results
```

**Kết quả Phase 3 lần 2:** Ổn định (spike epoch 29 được grad-guard bắt và phục hồi). Val Acc đỉnh **77.38%** (epoch 21) — **phá trần 76.8%**, xác nhận SR 2× thật có tác dụng. Còn cách tác giả (78.70%) ~1.3%.

---

## 9. Stochastic Weight Averaging (Phase 4 — vắt nốt khoảng cách)

**Quan sát:** Val dao động 75-77% suốt epoch 18-32 rồi overfit — đúng "sân nhà" của SWA. Tác giả đạt 78.70% với kiến trúc đơn giản hơn ⇒ lợi thế nằm ở recipe, không phải kiến trúc.

**Triển khai (`trainer.py`):**
- Gom trung bình đều trọng số raw qua nửa cuối training (`SWA_START_FRAC=0.5` → từ epoch 22).
- Cuối training: nạp trọng số SWA → **tính lại BN running stats** (`_bn_update`, 200 batch) → đánh giá val; nếu tốt hơn best thì ghi đè `_best.pth`. Luôn lưu `_swa.pth`.
- Tắt early-stopping trong pha SWA để gom đủ cửa sổ.
- Config: `USE_SWA`, `SWA_START_FRAC`, `SWA_BN_BATCHES`.

**Train Phase 4 (SWA tự bật trong preset):**
```
python train.py --v100 --no-positional-head --experiment-name restran34_sr2x_swa --epochs 45 --output-dir results
```

*(Chờ dòng `🧪 SWA Val Acc: ...` cuối training — kỳ vọng +0.5-1.5%, vượt 78.70%...)*

---

## 10. Sửa phân kỳ triệt để (fp16 → bf16 + chặn SR)

**Sự cố:** Phase 4 phân kỳ lại ở epoch 16-17 (train loss 0.60→1.10→9.59→kẹt 10, val 0%, KHÔNG phục hồi) dù đã hạ LR 2e-4 / clip 1.0. Ngưỡng skip=15 nằm trên mức loss lúc sụp (~10) nên vô dụng.

**Nguyên nhân gốc (2 cơ chế):** (1) **fp16 overflow** ở activations 64×256; (2) **SR block không bị chặn** trên ~48% mẫu không có HR (chỉ chịu gradient CTC) → CTC đẩy output SR ra cực trị → nổ backbone.

**Thử bf16 — THẤT BẠI:** bf16 autocast tạo NaN ngay epoch 1 (`Train Loss: inf`) trên V100 (Volta không có HW bf16; grid_sample/PixelShuffle/CTC dưới bf16 sinh NaN). Val loss vẫn hữu hạn vì val-TTA chạy fp32 → xác nhận lỗi nằm ở bf16 autocast. **Bỏ bf16.**

**Khắc phục chốt lại (fp16 + clamp là fix gốc thật):**
1. **Giữ fp16** (đã chạy ổn tới epoch 15-21 nhiều lần) + GradScaler bật.
2. **Chặn output SR block `clamp(±6)`** — ĐÂY là fix gốc: mọi lần fp16 phân kỳ trước đều CHƯA có clamp. Clamp chặn "SR nổ → backbone tràn fp16". Dù ~48% mẫu không HR, SR không thể vượt ±6.
3. **Grad-norm guard** (bỏ step khi grad không hữu hạn) + **MAX_TRAIN_LOSS=15**.

**Fallback nếu vẫn phân kỳ:** fp32 (`USE_AMP=False`, `--batch-size 16`) — chậm nhưng chắc chắn.

**Chạy lại:**
```
python train.py --v100 --no-positional-head --experiment-name restran34_sr2x_fp16clamp --epochs 45 --output-dir results
```

**Kết quả fp16+clamp:** ỔN ĐỊNH hết training — spike epoch 24-25 (train 1.10→2.27) **tự phục hồi** (epoch 26-27 về 0.57/75.58%). Clamp là fix đúng. Đỉnh **77.08%** (epoch 22) rồi overfit về ~75%.

---

## 11. Sửa SWA gating + Giảm over-regularization (Phase 5)

**Phát hiện 1 — SWA bị nhiễm độc:** SWA gom cả epoch 24-25 hỏng (val 71%/63%) → trung bình bị kéo tụt. **Fix:** gate SWA chỉ gom epoch có val acc ≥ best − `SWA_ACC_MARGIN(3.0)`.

**Phát hiện 2 — OVER-REGULARIZED (lý do kẹt ~77% vs author 78.70%):** Tác giả đạt 78.70 với **cùng kiến trúc** nhưng recipe nhẹ hơn nhiều. Ta regularize quá mạnh → ghìm đỉnh. SR-2× + SWA đã tự tạo generalization nên giảm reg là an toàn:

| | Cũ | Mới (gần author) |
|---|---|---|
| head_dropout | 0.08 | **0.03** |
| transformer_dropout | 0.15 | **0.10** |
| label_smoothing | 0.04 | **0.02** |
| weight_decay | 3e-4 | **2e-4** |
| synthetic_prob | 0.85 | **1.0** |
| CoarseDropout | lỗ (4-16)px, p0.5 | **lỗ (3-10)px, p0.3** |
| GridDistortion | limit0.2, p0.3 | **limit0.12, p0.15** |

**Chạy lại:**
```
python train.py --v100 --no-positional-head --experiment-name restran34_sr2x_light --epochs 45 --output-dir results
```

**Kết quả reg nhẹ:** Đỉnh **77.58%** (epoch 21) — cao nhất từ trước, nhưng reg nhẹ → **overfit mạnh hơn** (train 0.25 vs 0.45; val loss 0.32). Trần ~77.5% rất lì qua MỌI cấu hình. SWA cho 76.18% (< đỉnh) vì cosine LR giảm về ~0 ở pha gom → các epoch cuối chỉ overfit đứng yên, kéo trung bình xuống. **Bài học:** (1) over-reg không phải trần chính; (2) SWA cần LR hằng/cyclic ở pha gom mới hiệu quả.

---

## 12. Ensemble (Phase 6 — đòn bẩy chắc chắn nhất)

Trần ~77.5% quá lì qua mọi cấu hình đơn model ⇒ chuyển sang **ensemble** (trung bình posterior nhiều model), cách đáng tin cậy nhất để vượt (+1-2%). Công cụ `ensemble_eval.py` nạp ≥2 checkpoint cùng kiến trúc (cùng `SR_UPSAMPLE`), trung bình xác suất rồi decode với len7+pos-class.

**Dùng ngay checkpoint đã có (không train lại):**
```
python ensemble_eval.py --v100 --tta --checkpoints \
  results/restran34_sr2x_light_best.pth \
  results/restran34_sr2x_best.pth \
  results/restran34_sr2x_fp16clamp_best.pth
```

**Lưu ý:** các checkpoint này đều SEED=42 → hơi tương quan, gain ensemble hạn chế. Để ensemble mạnh, train thêm 2 seed khác rồi ensemble 3:
```
python train.py --v100 --no-positional-head --seed 1 --experiment-name sr2x_s1 --epochs 45 --output-dir results
python train.py --v100 --no-positional-head --seed 7 --experiment-name sr2x_s7 --epochs 45 --output-dir results
```

**Caveat quan trọng:** cần xác nhận 78.70% của tác giả đo trên **CÙNG val split** (val của ta = 0.1×Scenario-B, seed 42). Nếu tác giả báo trên test set chính thức thì so sánh không trực tiếp.

*(Chờ kết quả ensemble...)*

---

## 13. Single-model boosts: SWA đúng chuẩn + Lookahead (Phase 7)

Quyết định viết paper bằng **single model** (so single-vs-single với tác giả cho công bằng; ensemble chỉ để leaderboard). Hai kỹ thuật đều cho ra **1 model** (hợp lệ):

**1. Sửa SWA → LR hằng số:** SWA cũ hỏng vì cosine giảm về ~0 ở pha gom (chỉ trung bình điểm overfit). Giờ giữ **LR hằng `SWA_LR=5e-5`** suốt pha gom (từ epoch 22) → model "nảy" quanh vùng 76-77% → trung bình rơi vào cực tiểu phẳng hơn. Citeable (Izmailov 2018). Kỳ vọng +0.5-1%.

**2. Lookahead(AdamW)** (Zhang 2019): slow weights kéo về fast weights mỗi `k=5` step (`α=0.5`) → cập nhật mượt, generalize tốt hơn, gần như miễn phí. Triển khai trainer-side (không bọc optimizer → không đụng GradScaler/scheduler).

**Ghi chú về LR 5e-4 gốc tác giả (giải đáp thắc mắc):** Acc→0 sau 10 epoch do 3 cơ chế ở LR cao: (a) fp16 overflow, (b) STN sinh affine thoái hóa, (c) CTC blank-collapse. Tác giả tránh được nhờ OneCycle warmup + grad_clip 5 (+ có thể fp32).

**Chạy (single model tốt nhất, seed 42):**
```
python train.py --v100 --no-positional-head --seed 42 --experiment-name sr2x_swa_la --epochs 45 --output-dir results
```

**Kết quả Phase 7 — REGRESSION:** Lookahead làm tụt val −3% ngay epoch 21 (77.58→74.27, cùng seed 42); SWA_LR=5e-5 quá thấp làm tail overfit đứng yên. **Kết luận: optimizer tricks KHÔNG phải đòn bẩy — gỡ bỏ.**

---

## 14. Cross-Frame Attention Fusion + review của thầy (Phase 8)

**Gỡ bỏ:** `USE_LOOKAHEAD=False`, `USE_SWA=False` → về recipe 77.58 sạch.

**Review của thầy (guide.md):** phần lớn là các điểm ta ĐÃ SỬA (SR upsample 2× + L1 loss, warmup, epochs) → xác nhận đúng hướng. Điểm còn hợp lệ & mới:
- **AttentionFusion chỉ "chọn" frame, không "kết hợp"** → triển khai **CrossFrameFusion** (attention pooling qua 5 frame mỗi vị trí width → tổng hợp chi tiết bổ sung). `USE_CROSS_FRAME_FUSION=True`, cờ ablation `--no-cross-frame-fusion`.
- **JPEG degradation chưa đủ nặng** cho Scenario-B → tăng `jpeg_p` (0.15→0.35 base), hạ `quality_low` (35→18) để mô phỏng nhiễu nén nặng.
- (Chưa làm) Shared/anchor STN — để lần sau nếu cross-frame fusion chưa đủ.

**Điểm thầy hiểu sai:** "data leakage 36000 samples" — không phải leakage (synthetic chỉ ở train, cùng label); scheduler đã dùng len(loader).

**K-Fold (Q4):** để đánh giá độ tin cậy (mean±std), KHÔNG tăng accuracy, tốn K× compute → chưa làm; ưu tiên vượt số trước.

**Chạy (cross-frame fusion, seed 42):**
```
python train.py --v100 --no-positional-head --seed 42 --experiment-name sr2x_crossfusion_s42 --epochs 45 --output-dir results
```
Ablation đối chứng (fusion cũ): thêm `--no-cross-frame-fusion`.

**Kết quả cross-frame fusion:** đỉnh **76.98%** < 77.58% (old fusion) → KHÔNG giúp. Đã trả default về old AttentionFusion + JPEG gốc. Module giữ lại làm ablation (kết quả âm tính có giá trị cho paper).

---

## 15. Nhìn nhận trần & Đổi hướng đo lường (Phase 9)

**Mẫu hình qua ~10 cấu hình:** mọi thành phần (SR 2×, positional head, SWA, Lookahead, cross-frame fusion, reg tuning) đều đáp về **~77%**. Kết luận dựa trên dữ liệu: **nút thắt không còn ở thành phần model** — mà là (1) giới hạn thông tin ảnh 30px, hoặc (2) bản thân val split (0.1×Scenario-B).

**Best single model: 77.58%** (`restran34_sr2x_light_best.pth`: SR 2× + STN + old fusion + reg nhẹ, seed 42).

**Đổi hướng then chốt:** con 78.70 của tác giả gần như chắc chắn đo trên **test set chính thức** (leaderboard), KHÔNG phải val nội bộ Scenario-B của ta. → Phải đo trên **cùng test set** mới so được:
```
python predict.py --checkpoint results/restran34_sr2x_light_best.pth --mode test --v100
```
→ nộp file submission lên leaderboard → lấy số CHÍNH THỨC so với 78.70.

**Định hướng paper:** đóng góp = phương pháp (SR 2× native-HR, layout-aware decoding) + **bảng ablation trung thực** (gồm cả kết quả âm tính: positional head, cross-fusion, SWA không giúp — đây là giá trị khoa học thật, hiếm paper dám báo).

---

## 16. HR Distillation + RCAN Channel Attention (Phase 10)

**Đính chính dữ liệu:** 100% track có 5 HR khớp (200k ảnh) — SR giám sát đầy đủ (con "52%" trước là sai).

**Hai hướng thử (tận dụng 100% HR + đề xuất của user):**

1. **RCAN-style Channel Attention trong SR block** (`SR_CHANNEL_ATTENTION`): thay `_ResidualConvBlock` bằng `_RCAB` (conv-relu-conv → SE channel attention → skip, bỏ BN theo chuẩn EDSR/RCAN). Tái trọng số kênh feature quan trọng cho việc phục hồi nét chữ.

2. **HR Privileged-Information Distillation** (`USE_HR_DISTILL`): forward HR (hạ về 32×128) qua CHÍNH model ở eval-mode + no-grad → "teacher" sạch; student (LR) bị kéo về teacher bằng KL. Teacher đóng băng (detach) nên KHÔNG lặp lại lỗi self-distillation cũ (catastrophic forgetting). Tận dụng 100% HR. λ_kd=0.5.

**Lưu ý compute:** HR distill thêm 1 forward/batch → epoch chậm ~gấp đôi (đã 2× SR sẵn).

**Chạy (cả hai, seed 42):**
```
python train.py --v100 --no-positional-head --seed 42 --experiment-name sr2x_ca_kd --epochs 45 --output-dir results
```
Ablation: `--no-hr-distill` (chỉ channel attention), `--no-sr-channel-attention` (chỉ distill).

*(Chờ kết quả — HR distill/channel attention có phá trần 77.5%?...)*

**Kết quả HR distill + CA:** BẤT ỔN (train loss spike 0.6→1.48, val loss nhảy loạn, đỉnh chỉ 74.97%). HR-teacher online dùng BN stats kém ở đầu → target nhiễu. **Tắt HR distill.**

---

## 17. GroupNorm backbone (Phase 11 — đóng gap recipe)

**Bối cảnh:** So với bài của bạn cùng lớp (CRNN+STN, thua author 1%), best của ta (77.58) cũng chỉ thua ~1.1% — NGANG, không phải 5-6% (con 5-6% là từ run HR distill hỏng). Audit kiến trúc: ta có ĐỦ thành phần author (STN+ResNet+Transformer+CTC) + dư (multi-frame, SR). Không thiếu kiến trúc.

**Khác biệt thật = recipe:** ta buộc phải hạ LR 5e-4→2e-4 + giữ BatchNorm vì **BN+fp16+LR cao gây nổ** (đúng lúc user chạy config gốc 5e-4 → acc 0). Bạn cùng lớp dùng `--backbone-norm group` → ổn định ở LR cao → tái tạo recipe author.

**Triển khai:** `_bn_to_gn` chuyển BatchNorm→GroupNorm trong backbone (giữ conv pretrained, seed affine từ BN, bỏ running stats). GroupNorm độc lập batch + fp16-stable → chịu được LR cao. `BACKBONE_NORM='group'`, **nâng LR 2e-4→3e-4**. Tắt channel attention + HR distill để cô lập biến. Cờ `--backbone-norm {batch,group}`.

**Chạy:**
```
python train.py --v100 --no-positional-head --seed 42 --experiment-name sr2x_groupnorm --epochs 45 --output-dir results
```

*(Chờ kết quả — GroupNorm + LR cao có đóng được 1% gap tới 78.70?...)*

---

## 18. RTX 4090 preset — mở khóa bfloat16 (Phase 12)

Chuyển sang RTX 4090 (Ada Lovelace, có bf16 phần cứng). `apply_rtx4090_preset`: kế thừa v100 preset + `AMP_DTYPE='bfloat16'` (Ada chạy bf16 native, khác V100/Volta fail) + `BATCH_SIZE=24` (24GB). GradScaler tự tắt cho bf16.

**Ý nghĩa:** bf16 dải mũ như fp32 → **hết overflow → hết phân kỳ**. Kết hợp GroupNorm → training cực ổn định → **giờ có thể đẩy LR lên như author (4-5e-4)** để đóng nốt gap. Cờ `--rtx4090`.

**Chạy:**
```
python train.py --rtx4090 --no-positional-head --seed 42 --experiment-name sr2x_gn_bf16 --epochs 45 --output-dir results
```
Đẩy LR sát author: thêm `--lr 5e-4`.

*(Chờ kết quả trên 4090...)*

**Kết quả 4090 lần 1 — CTC blank collapse:** GroupNorm (chuyển từ backbone pretrained-BN) + LR 3e-4 làm train loss kẹt ~3.0, val 0% (dự đoán toàn blank). SR loss vẫn giảm → chỉ phần nhận dạng sụp. Conv pretrained co-adapt với BatchNorm; đổi norm quá đột ngột → backbone xuất rác.

**Sửa (Phase 12b):**
- **Revert GroupNorm → BatchNorm** + LR 3e-4 → **2e-4** (đúng recipe 77.58). GroupNorm code giữ lại nhưng cần xử lý riêng (LR thấp/warmup dài) nếu muốn dùng — future work.
- **Val chậm:** tắt `USE_VAL_TTA` (TTA chạy model 3× × raw+EMA → val gấp ~6× chi phí, chỉ +0.3%). Giữ beam+pos-class. TTA đầy đủ chỉ dùng lúc inference cuối.
- Giữ **bf16** (lợi ích thật của 4090: hết phân kỳ fp16).

**RTX 4090 preset chốt = recipe 77.58 + bf16 + batch 24 + workers 16 + val không TTA.**

**Chạy:**
```
python train.py --rtx4090 --no-positional-head --seed 42 --experiment-name sr2x_bf16 --epochs 45 --output-dir results
```

**Kết quả 4090 + bf16:** ỔN ĐỊNH hoàn toàn, không spike. Epoch 19/45 đạt **76.68%** (EMA, KHÔNG TTA) và vẫn đang lên. Tốc độ ~2:12/epoch (vs V100 chậm hơn nhiều). Val nhanh hẳn sau khi tắt TTA. → bf16 là lựa chọn đúng cho 4090.

---

## 19. Phân tích ảnh thật + Shared STN (Phase 13)

**Đo 400 track ngẫu nhiên:**
| | median | range | aspect W/H |
|---|---|---|---|
| LR (input test) | 38×19 | 27-54 × 15-26 | **2.11** (1.4–2.8) |
| HR (target SR) | 82×38 | 56-125 × 26-60 | **2.20** |
| **Model input** | 32×128 | cố định | **4.00** |

**Phát hiện:** ảnh thật AR≈2.1 nhưng model resize về AR=4.0 → **kéo giãn ngang ~1.9×**. Ngoài ra AR nguồn biến thiên 1.4–2.8 nên mức méo **không đồng nhất** giữa các mẫu. HR/LR đúng **2.0×** → xác nhận `SR_UPSAMPLE=2` là lựa chọn chính xác.

**Trace feature map (input SR 64×256):** conv1→32×128, maxpool→16×64, layer2→8×32, layer3→4×32, layer4→**2×32**, pool→1×32 (32 timesteps = 4.6/ký tự).

**Đối chiếu 4 giải pháp của thầy:**
| Giải pháp | Trạng thái |
|---|---|
| GP1: SR PixelShuffle 2× + L1 loss | ✅ ĐÃ LÀM (Phase 3) — chính là thứ phá trần 76.8→77.4 |
| **GP2: Shared/Anchor STN** | ✅ **LÀM MỚI** (dưới đây) |
| GP3: Fusion 2D trước khi nén H | ⚠️ Thầy giả định H'=8, **thực tế H'=2** → gần như vô nghĩa. Bản 1D đã thử (cross-frame) = 76.98% < 77.58%. Muốn 2D thật phải fuse sớm ở layer2 (H'=8) — thay đổi lớn. |
| GP4: JPEG nặng | ⚠️ Đã thử nhưng lẫn với thay đổi khác → chưa kết luận, có thể test lại sạch |

**Shared STN (GP2):** `STNBlock.forward(x, num_frames)` — trung bình đặc trưng localization qua 5 frame rồi mới hồi quy **1 affine duy nhất/track**, broadcast cho cả 5 frame ⇒ các frame được warp **giống hệt nhau**, giữ pixel-alignment trước fusion (tránh ghosting). Cờ `--stn-shared` (train + predict + eval; KHÔNG auto-detect được vì không thêm tham số).

**Thêm cờ `--img-height/--img-width`** để test giả thuyết tỷ lệ khung hình.

**Chạy:**
```
# GP2: shared STN
python train.py --rtx4090 --no-positional-head --stn-shared --seed 42 --experiment-name sr2x_stnshared --epochs 45 --output-dir results
# Giả thuyết aspect-ratio (AR=2 khớp ảnh thật, nhiều pixel dọc hơn)
python train.py --rtx4090 --no-positional-head --img-height 48 --img-width 96 --seed 42 --experiment-name sr2x_ar2 --epochs 45 --output-dir results
```

**KẾT QUẢ Aspect-Ratio 48×96 (V100, fp16):** đỉnh **77.48% @ep23 — KHÔNG TTA**.
So sánh phải đồng nhất: baseline 77.58% đo **CÓ TTA**; run này **KHÔNG TTA** ⇒ cùng điều kiện AR2 ước ~77.8-78.0% ⇒ **có vẻ VƯỢT baseline**. Giả thuyết từ đo ảnh thật (AR 2.1 vs model 4.0) được ủng hộ. Cần xác nhận bằng `eval_ablation.py` (có TTA).

**Quan sát khác:**
- Bất ổn fp16 vẫn còn: ep12 (18 batch skip), **ep21 raw sụp 0%** nhưng **EMA giữ 73.27%**, ep28. Clamp+guard+EMA cứu được, nhưng…
- **Khoảng cách raw vs EMA rất lớn (~8 điểm:** 69.37 vs 77.18) → trọng số raw rất nhiễu, EMA đang gánh phần lớn. Dấu hiệu fp16 + LR còn hơi nóng cho cấu hình 48×96 (nhiều pixel hơn → gradient lớn hơn).

**Đã thêm `--img-height/--img-width` cho `eval_ablation.py`, `predict.py`, `error_analysis.py`, `ensemble_eval.py`** (bắt buộc khớp lúc train, nếu không dataset resize sai → accuracy sụp về 0%; đã thêm dòng in INPUT SIZE + cảnh báo tự động khi acc < 5%).

---

## 20. ⚠️ ĐÍNH CHÍNH QUAN TRỌNG: Layout-aware decoding đóng góp 0.00%

**Đo sạch trên `sr2x_ar2_v100_best.pth`** (999 mẫu val, CÙNG logits, chỉ khác cách decode):

| Decode variant | Val Acc | Correct |
|---|---|---|
| greedy | **77.58%** | 775/999 |
| beam16 | 77.58% | 775/999 (**+0.00**) |
| beam16+len7 | 77.58% | 775/999 (**+0.00**) |
| beam16+len7+posclass | 77.58% | 775/999 (**+0.00**) |
| beam16+len7+posclass+TTA | 77.38% | 773/999 (**−0.20**) |

**Kết luận (lật lại nhận định trước đây):**
- Trước đây (mục 5-6) tôi kết luận layout-aware decoding là "đòn bẩy lớn nhất / đóng góp cốt lõi" khi val nhảy 74.87→76.78. **SAI.** Lần đó đổi ĐỒNG THỜI nhiều thứ (`SYNTHETIC_SAMPLE_PROB` 0.4→0.85, `BACKBONE_LR_RATIO`, `SR_LOSS_WEIGHT_MIN`) — mức tăng đến từ những thay đổi kia.
- Phép đo có kiểm soát cho thấy beam/len-7/pos-class **chính xác không sửa được gì** (775 = 775 = 775 = 775). Nguyên nhân: model đã ngầm học format từ 20k mẫu nên **không bao giờ vi phạm** ràng buộc độ dài 7 / lớp chữ-số ⇒ prior không có gì để sửa.
- **TTA không có lợi** (−0.20%, 2 mẫu ~ nhiễu) mà tốn 3× forward ⇒ `USE_INFERENCE_TTA = False`.

**Bài học phương pháp:** mọi kết luận trước đây rút ra từ các run đổi-nhiều-biến-cùng-lúc đều **không đáng tin**. Chỉ các phép đo có kiểm soát (như bảng trên) mới dùng được cho paper.

**Hệ quả cho paper:** không được claim layout decoding là đóng góp — phải báo cáo trung thực là **+0.00%** (kết quả âm tính). Đóng góp thật còn lại: **SR 2× có giám sát HR gốc** (đo được: 76.8→77.4) và **phân tích trần**.

---

## 21. Error analysis + phát hiện domain gap PNG/JPEG (Phase 14)

**`error_analysis.py` trên `sr2x_ar2` (999 mẫu):**
- Greedy 77.58% | +pos-class 77.58% (**headroom +0.00**)
- Length-mismatch 1.10% (11) | **Class-violation 0.00% (0)** ⇒ giải thích dứt điểm vì sao prior vô dụng: model **không bao giờ vi phạm** template.
- **Confusion: same-class 99.5%, cross-class 0.5%** ⇒ **REPRESENTATION-bound**. Top: `6→8, M→H, 6→5, 6→4, 8→6, D→O, V→W, O→Q` — toàn nhầm hình dạng ở độ phân giải thấp, decode không cứu được.
- Lỗi theo vị trí: chữ cái khó hơn số (pos2 9.51%, pos1 7.39% vs pos3-6 ~4.3-5.4%).
- **🔥 Brazilian 55.24% (116/210) vs Mercosur 83.52% (659/789) — chênh 28 điểm.** Nếu Brazilian đạt mức Mercosur ⇒ tổng **~83.5% (+6 điểm)**. Đây là đòn bẩy lớn nhất còn lại.

**Kiểm chứng dữ liệu (soi trực tiếp file):**
| Scenario/Layout | Định dạng | W×H | AR |
|---|---|---|---|
| A/Brazilian | **.png** | 44×19 | 2.18 |
| A/Mercosur | **.png** | 37×18 | 2.13 |
| B/Brazilian | **.jpg** | 47×17 | **2.72** |
| B/Mercosur | **.jpg** | 48×17 | **2.75** |

**Ba đính chính/phát hiện:**
1. **⚠️ Đo AR trước đây SAI:** mục 19 glob `lr-*.png` ⇒ **chỉ lấy Scenario-A**. Miền val thật (Scenario-B) có **AR≈2.75**, không phải 2.11. Lý do đưa ra cho 48×96 (AR=2.0) là sai — khớp với việc nó **không cải thiện** (77.58 = 77.58).
2. **✅ Thầy ĐÚNG về JPEG (guide.md #3 / Giải pháp 4):** Scenario-A = PNG (lossless), Scenario-B = JPEG. Val/test thuộc miền JPEG ⇒ **domain gap thật**, trước đây tôi đánh giá thấp.
3. **Brazilian KHÔNG do ảnh nhỏ hơn** (47×17 vs 48×17, area 813 vs 816) ⇒ chênh 28 điểm đến từ **mất cân bằng dữ liệu** (Scenario-B: 2000 vs 8000; toàn train 7000 vs 13000) hoặc font Brazilian khó hơn.

**Đã triển khai (2 hướng dựa trên dữ liệu):**
- **`JPEG_DOMAIN_AUG`**: chèn `ImageCompression(quality 30-75, p=0.8)` vào mẫu **miền PNG (Scenario-A)** ở độ phân giải gốc, để khớp miền JPEG của val/test. Cờ `--no-jpeg-aug`.
- **`LAYOUT_BALANCE`**: oversample layout thiểu số (Brazilian) cho bằng đa số. Cờ `--no-layout-balance`.
- **`error_analysis.py`**: thêm bảng lỗi **theo từng layout** + phân tích **quyết định lớp ký tự ở slot 4** (slot duy nhất 2 layout khác nhau) để biết lỗi Brazilian tập trung ở đâu.

**Chạy:**
```
python train.py --v100 --seed 42 --experiment-name sr2x_jpeg_balance --epochs 45 --output-dir results
python error_analysis.py --checkpoint results/sr2x_ar2_v100_best.pth --v100 --img-height 48 --img-width 96
```

---

## 22. Nguyên nhân gốc của gap Brazilian: DẤU CHẤM phân cách (Phase 15)

**Bác bỏ giả thuyết slot-4 bias:** Brazilian sai lớp ký tự ở slot 4 chỉ **0.97% (2/206)**, Mercosur **0.00%**. Model nhận diện layout gần như hoàn hảo ⇒ **KHÔNG cần layout-classifier head** (may là đã đo trước khi xây).

**Lỗi theo vị trí × layout:**
| slot | Brazilian | Mercosur | tỷ lệ |
|---|---|---|---|
| pos0 [L] | 5.34% | 3.71% | 1.4× |
| pos1 [L] | 11.17% | 6.39% | 1.8× |
| pos2 [L] | 16.99% | 7.54% | 2.3× |
| pos3 [D] | **13.11%** | 3.32% | **3.9×** |
| pos4 | **12.62%** | 2.05% | **6.2×** |
| pos5 [D] | **11.17%** | 3.07% | **3.6×** |
| pos6 [D] | **10.68%** | 2.56% | **4.2×** |

Chữ cái kém 1.4-2.3×, **chữ số kém 3.6-4.2×**.

**Soi ảnh thật ⇒ tìm ra nguyên nhân:**
- Brazilian: `BCL·4302`, `AYL·1894`, `EIT·2065` — **có DẤU CHẤM phân cách** giữa chữ và số; biển cũ, tối/bạc màu, nền đa dạng.
- Mercosur: `MFI7J49`, `TBX3A11` — liền mạch, nền trắng + dải xanh nhất quán.

⇒ Brazilian nhồi **8 ký hiệu** (7 ký tự + dấu chấm) vào cùng bề rộng ⇒ **~5.9 px/ký tự vs 6.9 px** của Mercosur (**−15%**). Dấu chấm nằm **ngay trước pos3** — đúng chỗ lỗi chữ số tăng vọt.

**Phát hiện kéo theo — thí nghiệm AR2 tự bắn vào chân:** timestep CTC cho 8 ký hiệu:
| Input | W′ (timesteps) | timestep/ký hiệu |
|---|---|---|
| 32×128 (baseline) | 32 | 4.0 |
| **48×96 (AR2)** | 24 | **3.0** ← chật, hại Brazilian |
| **48×160 (đề xuất)** | 40 | **5.0** |

AR2 tăng dọc nhưng **cắt ngang 128→96** ⇒ chỉ 3.0 timestep/ký hiệu ⇒ giải thích vì sao AR2 chỉ hòa (77.58 = 77.58).

**Hướng tiếp theo (theo thứ tự ưu tiên):**
1. `--img-height 48 --img-width 160` (nhiều pixel dọc + 40 timesteps)
2. `LAYOUT_BALANCE` + `JPEG_DOMAIN_AUG` (đã bật sẵn)

**Xác nhận ràng buộc 7 ký tự VẪN ĐÚNG:** nhãn của cả 2 layout đều đúng 7 ký tự, charset sạch `0-9A-Z`, **không chứa dấu chấm** (dấu chấm chỉ là chi tiết thị giác trên biển, không phải token trong nhãn). GT của `BCL·4302` là `BCL4302`. ⇒ **Không gỡ ràng buộc.**

---

## 23. 🐛 BUG: guard loss tuyệt đối làm skip 100% batch ở width 160

**Triệu chứng:** `--img-width 160` → `skip=2036/2036`, `Train Loss: inf`, dừng ngay epoch 1.

**Nguyên nhân (lỗi thiết kế của guard, không phải model/dữ liệu):** `MAX_TRAIN_LOSS` là ngưỡng **tuyệt đối**, nhưng loss CTC **tỉ lệ với số timestep**:
| Width | Timesteps | Loss init ~ | vs ngưỡng 15 |
|---|---|---|---|
| 96 | 24 | ~12 | ✅ qua |
| 128 | 32 | ~16 | ⚠️ sát |
| **160** | **40** | **~20** | ❌ **vượt → skip sạch** |

**Sửa — guard TƯƠNG ĐỐI:** skip khi loss không hữu hạn, HOẶC `loss > max(MAX_TRAIN_LOSS, LOSS_SPIKE_MULT × median(200 loss gần nhất))`; trong 50 step đầu (chưa đủ lịch sử) chỉ skip khi không hữu hạn. Thêm `LOSS_SPIKE_MULT=10.0`, hạ floor `15.0 → 8.0`.

**Kiểm chứng bằng mô phỏng:**
- Kịch bản width-160 (loss init ~20): **skip 0/400** (trước: 2036/2036) ✅
- Kịch bản phân kỳ sau hội tụ (0.4 → 10.0 → inf): floor 8.0 bắt được **10.0 và inf**; floor 15.0 chỉ bắt inf ⇒ chọn 8.0.

**Bài học:** mọi ngưỡng tuyệt đối trên loss đều không portable khi đổi độ dài chuỗi/kích thước input.

---

## 24. Width 160 lần 1: raw weights phân rã dần → lưới an toàn EMA rollback

**Kết quả:** đỉnh **76.28% @ep14**, sau đó suy giảm rồi sụp hoàn toàn (ep22 val 0.90%, ep23-24 val 0.00%).

**Thứ tự sụp đổ (quan trọng — đổi chẩn đoán):**
| ep | Train | SR loss | Val raw | Val EMA |
|---|---|---|---|---|
| 20 | 0.51 | 0.50 | 59.16 | 73.67 |
| 21 | 1.26 | 0.53 | 21.32 | 67.47 |
| 22 | 1.34 | 0.51 | **0.00** | 0.90 |
| 23 | 2.97 | **3.25** | 0.00 | 0.00 |

⇒ **Nhận dạng chết TRƯỚC (ep22), SR nổ SAU (ep23)** ⇒ SR là **triệu chứng**, không phải nguyên nhân (khác với suy đoán ban đầu).

**Nguyên nhân:** trọng số **raw** phân rã dần từ ~ep16 (chênh raw-vs-EMA: 8.5 → 14.5 → 46.2), EMA che giấu tới khi không che nổi. Width 160 ⇒ **40 timestep** (thay vì 32) ⇒ gradient CTC tích lũy qua nhiều bước hơn ⇒ **LR 2e-4 quá nóng**; batch 24 (nhỏ hơn) càng nhiễu.

**Guard spike KHÔNG bắt được** vì đây là **trôi dần** chứ không phải spike: 0.5→1.3→2.9 chưa từng vượt ngưỡng 8.

**Lưới an toàn mới — EMA rollback:** khi `val_EMA − val_raw > COLLAPSE_ROLLBACK_MARGIN (15 điểm)` ⇒ nạp lại trọng số EMA vào model **và** hạ LR × `COLLAPSE_LR_DECAY (0.5)`. Kèm `_scale_lr()` hạ cả `base_lrs` của scheduler (nếu không, `scheduler.step()` kế tiếp sẽ khôi phục LR cũ — đã kiểm chứng bằng test). Cơ chế này sẽ cứu được run ở ep21 thay vì mất trắng.

**Chạy lại (LR thấp hơn cho 40 timestep):**
```
python train.py --v100 --img-height 48 --img-width 160 --batch-size 24 --lr 1.2e-4 \
  --seed 42 --experiment-name sr2x_48x160_lr12 --epochs 45 --output-dir results
```

---

## 25. 🔍 ĐỐI CHIẾU VỚI SOURCE GỐC CỦA TÁC GIẢ — tìm ra hồi quy lớn

**Kết luận về lo ngại "xoá nhầm khi remove CRNN": KHÔNG có gì bị mất.** Đường ResTran nguyên vẹn — `restran.py`, `STNBlock`, `AttentionFusion`, `ResNetFeatureExtractor`, `PositionalEncoding` đều khớp baseline (ta chỉ **thêm** SR block, frame_quality, positional head…). CRNN được gỡ sạch, không ảnh hưởng.

**NHƯNG tìm ra hồi quy nghiêm trọng do CHÍNH TA gây ra — degradation quá yếu:**

| Phép | **Tác giả** | **Ta (trước sửa, mạnh nhất)** |
|---|---|---|
| Blur p | 0.70 | 0.55 |
| Noise p | 0.70 | 0.55 |
| JPEG p | 0.50 | 0.40 |
| **JPEG quality** | **20-50** | 35-65 |
| Downscale p | 0.50 | 0.35 |
| **Downscale scale** | **0.30-0.50** | 0.55-0.90 |

Tệ hơn: curriculum của ta khởi đầu ở `strength=0.25` (epoch 0 → JPEG q=53-76, downscale 0.74-0.94 ≈ **gần như không degrade**), trong khi **tác giả luôn dùng full strength từ epoch 1**. Ảnh synthetic LR chính là dữ liệu dạy model đọc biển mờ nặng — ta đã làm nó dễ đi rất nhiều.

**Đã sửa:**
- `get_degradation_transforms`: tại `strength=1.0` **khớp CHÍNH XÁC** công thức tác giả (đã verify bằng assert).
- Sàn curriculum `0.25 → 0.6`, warmup `40% → 25%` số epoch.
- Thêm lại `A.Rotate(limit=10, p=0.3)` (tác giả có, ta bỏ nhầm khi thêm GridDistortion).
- `ChannelShuffle` `0.15 → 0.3` (giá trị tác giả).

**Khác biệt còn lại (cố ý, không phải lỗi):** ta dùng `pretrained=True` (tác giả `False`), cosine+warmup (tác giả OneCycle 5e-4/30ep), grad_clip 1.0 (tác giả 5.0), batch 32 (tác giả 64), discriminative LR, EMA, SR 2×.

⚠️ **Lưu ý normalization:** ta bật `pretrained=True` nhưng vẫn normalize `mean=std=0.5` thay vì thống kê ImageNet ⇒ lợi ích pretrain bị giảm.

---

## 26. Công cụ chẩn đoán: NaN detection + CSV log mỗi epoch

**1. Phát hiện & ĐỊNH VỊ NaN** (trước đây chỉ biết loss non-finite mà không biết vì sao):
- `_first_nonfinite(named_tensors)` — trả về tensor đầu tiên chứa NaN/Inf kèm số lượng (`input images (nan=1, inf=0, numel=3)`).
- `_check_weights_finite()` — quét toàn bộ parameters + buffers của model.
- Khi loss non-finite: in ngay nguồn gốc (`input images` / `hr targets` / `param:...` / `loss computation`), giới hạn 3 lần/epoch để log không bị ngập.
- Cuối mỗi epoch: báo tổng `nan_batches` và cảnh báo nếu **trọng số model** đã hỏng.
- Phân biệt rõ **`nan_batches`** (NaN thật) với **`skipped_batches`** (gồm cả spike hữu hạn bị bỏ) — trước đây gộp làm một nên không biết đang gặp loại nào.

**2. CSV log mỗi epoch** → `results/<experiment>_history.csv`, ghi ngay sau mỗi epoch (không mất dữ liệu nếu run bị kill).

Cột: `epoch, train_loss, val_loss, val_acc, val_raw_acc, val_ema_acc, used_ema, lr, sr_loss, sr_weight, pos_loss, distill_loss, skipped_batches, nan_batches, weights_nonfinite, best_acc, epoch_seconds`.

Đáng chú ý: có **`val_raw_acc` và `val_ema_acc` riêng** — chính khoảng cách này là dấu hiệu sớm của sụp đổ raw weights (mục 24), giờ theo dõi được bằng đồ thị thay vì phải đọc log thủ công.

---

## 27. Kết quả `sr2x_authordegrade`: fix đúng một nửa + xác định thủ phạm thật

**Đỉnh 76.78% (ep14/16) — THẤP hơn baseline 77.58%.** Nhưng error analysis cho thấy bức tranh tinh tế hơn:

| | `sr2x_ar2` | `authordegrade` | Δ |
|---|---|---|---|
| **Brazilian** | 55.24% | **59.52%** | **+4.28** ✅ |
| **Mercosur** | 83.52% | **81.37%** | **−2.15** ❌ |
| Length-mismatch | 1.10% | **0.40%** | tốt hơn ✅ |
| Tổng | 77.58% | 76.78% | −0.80 |

⇒ Degradation mạnh + layout balance **đúng hướng cho Brazilian** và giảm nửa lỗi độ dài, nhưng **oversample Brazilian làm Mercosur tụt**; val có 79% Mercosur nên net = `0.21×4.28 + 0.79×(−2.15) = −0.80`. **Đã tắt `LAYOUT_BALANCE`** (giữ flag `--layout-balance` cho ablation; có thể bật lại nếu test set cân bằng hơn Scenario-B).

**THỦ PHẠM THẬT — bất ổn tối ưu hoá, không phải kiến trúc:**
- **`nan_batches = 0` toàn run** ⇒ KHÔNG hề tràn số. Đây không phải vấn đề fp16.
- Rollback kích hoạt **3 lần** (ep18/20/21), cắt LR 1.5e-4 → 1.5e-5, **vẫn không cứu nổi**.
- **Gap raw-vs-EMA luôn 5-12 điểm ngay cả epoch khoẻ** ⇒ trọng số raw **luôn nhiễu**, EMA gánh toàn bộ. Đây là bất thường và là gốc rễ của mọi lần sụp quanh ep18-22.

**Đối chiếu lệnh của bạn cùng lớp (CRNN+STN đạt 82.8%)** — hai thứ ta thiếu hoàn toàn:
| Bạn ấy | Ta | Tác dụng |
|---|---|---|
| `--grad-accum-steps 2` → **batch hiệu dụng 64** | batch 32 | Giảm nhiễu gradient |
| `--backbone-norm group` | BatchNorm | Độc lập batch size |

**Mấu chốt về GroupNorm:** ta thử và nó sụp vì **chuyển đổi ResNet pretrained ImageNet (conv co-adapt với BN) sang GroupNorm**. Bạn ấy dùng CRNN **train from scratch** nên GroupNorm là native. **Tác giả cũng dùng `pretrained=False`.** ⇒ GroupNorm không sai, **cách áp dụng của ta mới sai**.

**Đã triển khai — Gradient Accumulation:** `GRAD_ACCUM_STEPS=2` → batch hiệu dụng **64** (khớp tác giả). Loss chia cho `accum` để gradient là trung bình; zero_grad/clip/step chỉ ở cuối cửa sổ; batch bị skip vẫn giữ đúng biên cửa sổ. **Đã verify bằng test số học: accum 2×4 cho gradient TRÙNG KHỚP 1 batch 8.** Cờ `--grad-accum-steps`.

---

## 28. 🚨 ĐÍNH CHÍNH PHƯƠNG PHÁP LUẬN: phần lớn kết luận trước đây là NHIỄU

Tài liệu của dự án bạn cùng lớp (`baseline1_crnn_stn/`) ghi rõ 2 điều đo được trên **CÙNG tập val 999 track Scenario-B** và **cùng kích thước 32×128** (nên hoàn toàn so sánh được với ta):
- **Biên nhiễu thống kê: ±13 track = ±1.3 điểm.**
- **`cudnn.benchmark=True` gây lệch tới 6.5 điểm giữa 2 lần chạy CÙNG SEED.**

**Preset của ta đang bật `USE_CUDNN_BENCHMARK=True`** ⇒ mọi so sánh đều nhiễm non-determinism.

| Run | Val Acc |
|---|---|
| sr2x_light | 77.58% |
| sr2x_ar2 | 77.58% |
| gn_scratch | 77.28% |
| sr2x_crossfusion | 76.98% |
| authordegrade | 76.78% |
| sr2x_accum64 | 75.78% |

**Toàn bộ dải này nằm trong biên nhiễu.** Các kết luận kiểu "cross-fusion kém hơn", "AR2 hoà", "degradation làm tệ đi" — **đều không đủ bằng chứng**. Đây là sai sót phương pháp của tôi: rút kết luận từ chênh lệch 0.5-1.5 điểm khi nhiễu đo lường lớn hơn thế.

**Đã sửa: `USE_CUDNN_BENCHMARK = False`** (deterministic). Từ giờ mọi so sánh mới ít nhất loại được nguồn nhiễu 6.5 điểm; muốn khẳng định cải tiến vẫn cần **multi-seed** và chênh lệch **> 1.3 điểm**.

**Con số thật của bạn cùng lớp:** tài liệu ghi **S1 = 79.78%** (797/999) — không phải 82.8%. Và họ tự nhận đây là kết quả **DUY NHẤT** vượt biên nhiễu trong 10 cấu hình đã thử, vẫn **chưa multi-seed**.

---

## 29. Kết quả `accum64` vs `gn_scratch`: tìm ra thứ KHÔNG phải nhiễu

| | `sr2x_accum64` (BN+pretrained) | `gn_scratch` (GroupNorm+scratch) |
|---|---|---|
| Epoch sống được | **17** (chết) | **37** (vẫn đang lên) |
| Rollback (gap>15) | **3 lần** (ep12,15,17) | **1 lần** (ep4) |
| Gap raw-EMA trung bình | **13.51** điểm | **6.71** điểm |
| Đỉnh | 75.78% @ep16 | **77.28% @ep35** |

Accuracy giữa 2 run nằm trong biên nhiễu, **nhưng ĐỘ ỔN ĐỊNH thì không** — đây là khác biệt về động lực huấn luyện, đo bằng số epoch sống sót và tần suất sụp đổ, không phải chênh lệch 1 điểm.

**Kết luận:** Gradient accumulation **KHÔNG** cứu được bất ổn (accum64 vẫn sụp 3 lần). **GroupNorm + train from scratch mới là thuốc** — đúng như setup của bạn cùng lớp, và cũng là điều **tác giả gốc làm** (`pretrained=False`).

**Đã chốt làm mặc định:** `BACKBONE_NORM='group'`, `USE_PRETRAINED_BACKBONE=False`.

---

## 30. 🐛🐛 BUG LỚN NHẤT ĐÃ TÌM RA: SR target lệch hình học

**Nguồn:** tài liệu S1 của dự án bạn cùng lớp ghi bảng J2→S1 có dòng
*"SR target alignment | **lệch geometry (bug)** | đã sửa — augment 1 lần ở cỡ target"*.
Kiểm tra code ta ⇒ **ta mắc ĐÚNG bug đó.**

| | Augment hình học |
|---|---|
| Ảnh LR (input SR) | ✅ Affine (dịch 6%, xoay ±6°), Perspective, GridDistortion, Rotate ±10° |
| Ảnh HR (target SR) | ❌ **KHÔNG** — chỉ `get_val_transforms` (Normalize) |

SR block nhận LR **đã xoay/dịch/méo ngẫu nhiên** rồi bị bắt tái tạo HR **không biến dạng**. Dịch 6% ở width 128 = **~8px, rộng hơn một ký tự**. L1 loss **bất khả thi về hình học** ⇒ SR chỉ học được ảnh mờ trung bình.

**Bằng chứng khớp hoàn hảo:** `sr_loss` của ta đứng im ở **0.42-0.52 trong MỌI run** (xem cột `sr_loss` các CSV), không bao giờ giảm sâu. Giải thích trọn vẹn vì sao SR chưa từng giúp gì suốt ~15 run.

**Đã sửa:**
- `get_paired_train_augment()` — dùng `additional_targets={'hr':'image'}` để albumentations **rút tham số 1 lần rồi áp cho CẢ HAI** ảnh.
- Tách `get_normalize_transform()` (Normalize+ToTensor) khỏi phần augment.
- `__getitem__`: khi train + có HR ⇒ resize LR lên **cỡ target**, augment CẶP, rồi hạ LR về cỡ input model (`INTER_AREA`); val/test giữ nguyên đường cũ.

**Cũng chỉnh:** `λ_SR = 0.1 cố định` (đúng S1; tài liệu họ ghi λ=0.5 kém hơn 3 track). Trước đây `gn_lam01` **KHÔNG hề chạy λ=0.1** — cột `sr_weight` bắt đầu ở 0.3, tức config chưa từng được sửa (thiếu sót của tôi).

**Bằng chứng phương sai tình cờ:** `gn_scratch_60` và `gn_lam01` có **config giống hệt nhau** (cả hai λ bắt đầu 0.3) nhưng: đỉnh 76.48% vs 76.68%, rollback **4 lần vs 1 lần**, kết cục **sụp còn 31.23% vs ổn định 75.88%**. Cùng seed, cùng config, kết quả rất khác ⇒ củng cố kết luận mục 28.

**Chỉ số cần theo dõi ở run tới:** `sr_loss` phải **giảm xuống dưới 0.4** — nếu vẫn đứng ~0.45 thì fix chưa ăn.

---

## 31. ✅ XÁC NHẬN fix SR alignment có tác dụng + đo được phương sai seed

**`sr_loss` — bằng chứng trực tiếp (độc lập với accuracy):**
| Run | ep1 | min | Giảm |
|---|---|---|---|
| `gn_scratch_60` (còn bug) | 0.4757 | 0.4334 | 8.9% — kẹt |
| `gn_lam01` (còn bug) | 0.4756 | 0.4275 | 10.1% — kẹt |
| **`sr_aligned` (đã sửa)** | **0.3753** | **0.3042** | **19.0%, vẫn giảm ở ep40** |

Vừa bắt đầu thấp hơn vừa xuống sâu hơn ⇒ mục tiêu SR giờ khả thi về hình học, SR đang học thật.

**Ổn định cải thiện rõ** (gap raw-EMA hội tụ): ep1-10 **6.11** → ep11-20 6.15 → ep21-30 4.89 → ep31-40 **2.15**. Chỉ 1 rollback (ep8) trong 40 epoch — run ổn định nhất từ trước tới nay.

**Nhưng accuracy vẫn 76.28%** (@ep29) — trong dải cũ.

**Phương sai seed (seed 42 vs 2026, cùng config, so từng epoch ep7-17):**
- Chênh trung bình **+1.33 điểm**, lớn nhất **+3.00 điểm**; seed 2026 nhỉnh hơn ở hầu hết epoch.
- ⇒ **Seed 42 KHÔNG "hỏng"**, nhưng phương sai seed (~1.3) đúng bằng biên nhiễu thống kê (±1.3). Xác nhận lần nữa: mọi so sánh 0.5-1 điểm trước đây đều vô nghĩa. **Con số này nên đưa vào paper.**
- `val_tracks.json` load từ file nên đổi seed **không đổi tập val** ⇒ so sánh vẫn hợp lệ.

---

## 32. Multi-Frame Super-Resolution (mảnh ghép cuối so với S1)

**Lý do:** ablation của dự án bạn cùng lớp cho thấy **J2 (SR per-frame) 77.18% → S1 (SR đa khung) 79.78%** — bước nhảy **duy nhất** vượt biên nhiễu ±13 track trong 10 cấu hình của họ. Ta đang ở đúng vị trí J2.

**`MultiFrameSuperResolutionBlock`:** head per-frame → gộp đặc trưng **mean + max qua 5 frame của cùng track** → concat vào từng frame → `fuse` 1×1 → residual body → PixelShuffle ×2 → tail (+ base bicubic, clamp ±6).

Nguyên lý: 5 khung là **cùng một biển số**, nên nhiễu cảm biến độc lập sẽ **triệt tiêu khi lấy trung bình** trong khi nét chữ **cộng hưởng** — thông tin mà SR per-frame vứt bỏ hoàn toàn.

**Giữ SR TRƯỚC STN** (không đổi thứ tự như S1): tránh phải warp HR target theo `theta`, giữ nguyên fix alignment vừa được kiểm chứng. Các khung trong một track vốn đã gần trùng nhau nên fuse trước STN vẫn hợp lý.

Output vẫn **per-frame** `[B*F,3,sH,sW]` ⇒ AttentionFusion phía sau và giám sát HR per-frame đều không phải đổi.

**Đã test:** đổi frame 1 của track 0 ⇒ output frame 0 **cùng track thay đổi** (0.0778) nhưng frame 0 của **track khác KHÔNG đổi** (0.000000) ⇒ đúng multi-frame, không rò rỉ giữa track. Auto-detect qua key `sr_block.fuse.*`.

**Chạy:**
```
python train.py --v100 --seed 42 --experiment-name mfsr --epochs 60 --output-dir results
```
Ablation đối chứng: `--no-multi-frame-sr`.

**KẾT QUẢ — âm tính:** `mfsr` đỉnh **76.18% @ep34** vs `sr_aligned` (per-frame) **76.28% @ep29** ⇒ chênh **0.10 điểm**, sâu trong biên nhiễu ±1.3. `val_loss` tăng đều từ ep34 (0.2372→0.2563) ⇒ đã qua đỉnh, chạy thêm không cứu được. `sr_loss` cũng không tốt hơn (0.3095 vs 0.3042). **Multi-frame SR không mang lại lợi ích đo được** → tắt mặc định (giữ code làm ablation).

---

## 33. 🔴 SR target ×2 có 81% là pixel NỘI SUY — nguyên nhân nghi ngờ chính

Tài liệu `run_gpu.md` của dự án bạn cùng lớp (thí nghiệm S4) ghi:
> *"HR gốc ~115×42px, target ×2 (256px) có **55% là nội suy bicubic thuần**. ×1 giữ output 32×128 ≈ đúng độ phân giải HR thật."*

**HR của ta còn nhỏ hơn nhiều** (đo được median **82×38**):
| | Số pixel | Pixel THẬT | **Nội suy** |
|---|---|---|---|
| HR gốc | 3,116 | — | — |
| **SR ×2 target (64×256)** | 16,384 | **19.0%** | **81.0%** |
| SR ×1 target (32×128) | 4,096 | 76.1% | 23.9% |

⇒ Suốt ~20 run, SR loss chủ yếu dạy model **tái tạo phép nội suy bicubic** chứ không phải khôi phục chi tiết thật. Giải thích rất tự nhiên vì sao SR chưa từng giúp, **kể cả sau khi đã sửa bug alignment** (mục 30-31).

**Đánh đổi của SR ×1:** timestep CTC giảm **32 → 16**. Chật hơn cho biển Brazilian 8 ký hiệu (2.0/ký hiệu) nhưng **đúng bằng cấu hình baseline của tác giả** (đạt 78.70%) nên chắc chắn hợp lệ (CTC cần ≥ 2L−1 = 13).

**Đã đổi:** `SR_UPSAMPLE = 1`, `SR_MULTI_FRAME = False` (giữ 1 biến thay đổi so với mốc `sr_aligned` 76.28%). Phụ lợi: target = cỡ input ⇒ nhánh paired-augment bỏ được một lần nội suy, và bộ nhớ giảm ~4×.

**Thứ CHƯA làm (user chọn ưu tiên 2 trước):** đối chứng `sr_loss_bilinear` — tài liệu họ dùng nó để phán quyết *"`sr_loss ≥ sr_loss_bilinear` kéo dài = SR học không hơn nội suy, tốn 3.66× compute"*. Ta chưa có mốc này nên **chưa biết SR của ta có hơn bicubic thuần hay không**.

**Chạy:**
```
python train.py --v100 --seed 42 --experiment-name sr1x --epochs 60 --output-dir results
```

---

## 34. Thêm 6 cột chẩn đoán vào CSV

| Cột mới | Trả lời câu hỏi gì |
|---|---|
| **`sr_loss_bilinear`** | Mốc "không học gì" = chính `base` mà SR cộng residual lên (nội suy bicubic ở ×2, hoặc chính input ở ×1). |
| **`sr_beats_interp`** | 1/0 — `sr_loss < sr_loss_bilinear` không. **Nếu = 0 kéo dài ⇒ SR vô dụng, nên bỏ.** |
| **`acc_brazilian`** | Layout yếu (đã liên tục thấp hơn Mercosur ~25 điểm). Trước đây phải chạy `error_analysis.py` riêng mới biết. |
| **`acc_mercosur`** | Layout đa số — phát hiện trường hợp cải thiện Brazilian nhưng làm hỏng Mercosur (đã xảy ra ở mục 27). |
| **`acc_greedy`** | Tách đóng góp của decode: `val_acc − acc_greedy`. |
| **`grad_norm`** | Độ lớn gradient TB mỗi epoch — chẩn đoán bất ổn sớm hơn cả gap raw-EMA. |

Đã test: bilinear resize đúng cỡ target ở cả ×1 và ×2; phân loại layout đúng (`gt[4].isalpha()` ⇒ Mercosur); cờ `sr_beats_interp` đúng.

**Cách đọc khi chạy:** nếu `sr_beats_interp` = 1 và khoảng cách `sr_loss_bilinear − sr_loss` **nới rộng theo thời gian** ⇒ SR học được chi tiết thật (đúng tiêu chí dự án bạn cùng lớp dùng để kết luận S1 thành công).
