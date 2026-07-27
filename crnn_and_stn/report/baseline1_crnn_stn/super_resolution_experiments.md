# Super Resolution cho CRNN + STN — Phân tích, Kế hoạch & Triển khai

Tài liệu này gộp toàn bộ nội dung liên quan tới Super Resolution (SR) từng nằm rải rác trong `summary/` (`file.md` phần SR, `sr_conclusion.md`, `sr_ocr_improvement_plan.md`, `sr_stacked_plan.md`, `light_edge_sr_implementation.md`), tổ chức lại theo dòng thời gian: **report → kế hoạch → triển khai thực tế**.

## 0. Mốc kết quả đã đo (tính tới thời điểm viết tài liệu này)

| Cấu hình | Exact Match | Ghi chú |
|---|---:|---|
| CRNN thường | ~75% | Baseline gốc |
| CRNN + STN (Baseline 1, report) | 77.00% | Số liệu gốc trong report — xem [baseline1_architecture.md](baseline1_architecture.md) |
| CRNN + STN (đo trên dataset thực tế của project) | 75.78% | batch 64, epochs 30, lr 5e-4 |
| **ResBlock stable** | **76.68–75.78%** | Backbone tốt nhất hiện tại — xem [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) |
| SR stacked-input v1 | 49.25% | Gộp 5 frame → 1 frame SR rồi nhân bản — **kém hơn nhiều so với baseline** |
| SR stacked-input v2 | 55.06% | Vẫn kém hơn baseline đáng kể |
| LightEdgeSR (edge-preserving, per-frame) | *chưa có kết quả* | Đã cài đặt, đang chờ chạy — xem [training_runs/run_gpu.md](../training_runs/run_gpu.md) |

**Kết luận nhanh**: hướng SR duy nhất đã đo được tới nay (stacked-input) làm giảm accuracy rất mạnh so với baseline. Nguyên nhân nhiều khả năng là thiết kế gộp 5 frame thành 1 trước khi nhân bản lại (xem mục 3), khiến Attention Fusion mất hết đa dạng giữa các frame để khai thác. `LightEdgeSR` được thiết kế lại để sửa đúng vấn đề này (áp dụng SR độc lập theo từng frame), nhưng **chưa có số liệu thực nghiệm xác nhận có cải thiện hay không**.

---

## 1. Bốn hướng SR theo report (mục "Future of Work", slide 52–83)

Report đặt SR như hướng phát triển tương lai, **chưa có số liệu thực nghiệm trong report** — toàn bộ phần dưới đây là suy diễn/đề xuất áp dụng cho bài toán biển số, không phải kết quả đã đo.

### 1.1 Backbone ResBlock (slide 53, 56–59)
- Dùng kiến trúc residual để giữ thông tin đầu vào tốt hơn qua nhiều tầng convolution.
- Vấn đề của ResNet gốc: dùng Batch Normalization, không phù hợp cho SR vì BN chuẩn hoá feature (mean=0, std=1) và giới hạn khả năng tái tạo — trong khi SR cần output có dynamic range lớn.
- Kết quả trong paper gốc (không phải project này): cải thiện hiệu năng đáng kể + giảm 40% memory.
- **Nhận xét áp dụng**: phù hợp làm khối nền cho các biến thể SR đơn giản, nhẹ, dễ tích hợp vào `CRNN + STN`, giữ biên ký tự tốt hơn nội suy đơn giản. → Đây là hướng SR nên thử đầu tiên.

### 1.2 Channel Attention (slide 54, 60–66)
- Học trọng số ưu tiên theo từng kênh đặc trưng, làm nổi bật channel chứa thông tin quan trọng.
- **Nhận xét áp dụng**: có thể hữu ích khi ảnh vẫn giữ một phần cấu trúc ký tự nhưng bị mờ/nhiễu nhẹ — nhấn đúng kênh có thể giữ lại cạnh ký tự, khoảng trống giữa ký tự, độ tương phản biên. Cần kiểm chứng thực nghiệm vì attention không phải lúc nào cũng tăng OCR accuracy (có thể chỉ tăng chất lượng hiển thị).

### 1.3 PixelShuffle (slide 55, 67–76)
- Cơ chế tái sắp xếp feature map để tăng kích thước đầu ra — upsampling phổ biến trong SR, gọn và hiệu quả hơn các cách tăng kích thước truyền thống.
- **Nhận xét áp dụng**: có ưu thế khi cần cân bằng độ phân giải/chi phí tính toán/tốc độ suy luận — pipeline OCR cần chạy nhanh nên đáng cân nhắc nếu muốn tăng độ nét mà vẫn giữ hệ nhẹ. Cần theo dõi artifact vì có thể làm sai lệch nét ký tự.

### 1.4 Stacked Inputs (slide 82)
- Thay vì dùng 1 ảnh đơn lẻ, model nhận nhiều frame của cùng track, kết hợp thông tin bổ sung giữa các frame để phục hồi chi tiết tốt hơn xử lý từng frame độc lập.
- **Nhận xét áp dụng**: đặc biệt phù hợp với dataset 5-frame/track của project này — có thể tận dụng chênh lệch nhỏ giữa các frame để bù chi tiết bị mất (frame này blur nhưng frame khác rõ hơn, biên ký tự rõ hơn ở một số frame). Tiềm năng cao hơn SR một frame nhưng phức tạp hơn trong thiết kế dữ liệu/huấn luyện/đánh giá.

### So sánh tổng hợp 4 hướng

| Hướng | Mục tiêu chính | Lợi thế | Hạn chế |
|---|---|---|---|
| `CRNN + STN` | Baseline nhận dạng trực tiếp | Đơn giản, ổn định, dễ đo lường | Kém hơn khi ảnh quá mờ |
| `ResBlock SR` | Giữ chi tiết cơ bản | Dễ triển khai, an toàn | Có thể cải thiện chưa nhiều |
| `Channel Attention` | Nhấn mạnh đặc trưng quan trọng | Có thể tăng khả năng tập trung vào ký tự | Tăng độ phức tạp mô hình |
| `PixelShuffle` | Upsampling hiệu quả | Gọn, nhanh, phù hợp pipeline nhẹ | Có nguy cơ sinh artifact |
| `Stacked Inputs` | Tận dụng nhiều frame | Tiềm năng cải thiện cao nhất | Khó thiết kế hơn |

### Thứ tự triển khai khuyến nghị theo report

1. `CRNN + STN` làm baseline.
2. `ResBlock SR` để kiểm tra tác động SR cơ bản.
3. `Channel Attention` để đánh giá khả năng nhấn mạnh đặc trưng.
4. `PixelShuffle` để kiểm tra hướng upsampling hiệu quả.
5. `Stacked Inputs` nếu muốn khai thác mạnh thông tin nhiều frame.

**Vị trí trong report** (để tra cứu lại slide gốc khi cần):

| Thành phần | Slide |
|---|---|
| Backbone ResBlock | 53, 56–59 |
| Channel Attention | 54, 60–66 |
| PixelShuffle | 55, 67–76 |
| Stacked Inputs | 82 |

---

## 2. Kế hoạch cải thiện OCR chi tiết

### Prompt gốc của người dùng (lưu nguyên văn để tra cứu lại bối cảnh)

> @crnn_and_stn/results/crnn_stn @crnn_and_stn/results/crnn_stn_resblock_stable @crnn_and_stn/run_ablation.py @crnn_and_stn/train.py @crnn_and_stn/summary @crnn_and_stn/src @crnn_and_stn/summary/sr_conclusion.md Ngoài 4 cái super resolution bạn có thể gợi ý cách làm hay 1 vài thuộc tính SR để caie thiện tối đa việc OCR biển số, làm rõ ảnh ở dộ phân giải thấp hay không, hiện tại tôi đã áp dụng resblock stable được 76.68% crnn thường thì được tầm 75, còn stacked-inputđyocjw cao nấht 55%@crnn_and_stn/summary/run_gpu.md hãy áp dụng, đề xuất những cách cải thiện độ chính xác của model với bộ dataset, tạo pla phân tích theo từng step lưu vào file .md và gợi ý những best SR các bước thực hiện dự kiến vào file .md lưu luôn cả prompt của tôi

### Mục tiêu
- Tăng độ chính xác OCR biển số trên ảnh độ phân giải thấp.
- Ưu tiên cải thiện **độ đúng ký tự** hơn là chỉ làm ảnh nhìn "đẹp".
- Mốc hiện tại: CRNN thường ~75%, ResBlock stable ~76.68%, Stacked-input SR ~55%.

### Nhận định nhanh
- **ResBlock stable đang là hướng tốt nhất**: tăng ~1.5% so với CRNN thường, cho thấy backbone/residual design giúp ích thật. Hướng tối ưu tiếp theo: giữ kiến trúc ổn định, giảm artifact, tăng khả năng giữ nét ký tự.
- **Stacked-input SR đang chưa phù hợp nếu dùng nguyên bản** (55%): có thể đang tạo artifact, làm lệch nét ký tự, hoặc tăng độ phức tạp đầu vào quá mức so với lợi ích. Kết luận thực dụng: không nên coi SR hiện tại là bước chắc chắn giúp OCR — phải test theo hướng "SR tốt cho OCR" thay vì "SR đẹp".

### 4 thuộc tính SR nên ưu tiên cho OCR biển số

1. **Giữ biên ký tự sắc nét** — ký tự biển số có cạnh thẳng, nét mỏng, tương phản cao; SR tốt phải bảo toàn biên, không làm tròn cạnh hay tạo texture giả.
2. **Ít hallucination** — tuyệt đối tránh SR "vẽ thêm" chi tiết không có thật, vì rất dễ làm CRNN đọc sai ký tự.
3. **Tối ưu cho text/plate domain** — SR cho ảnh tự nhiên chưa chắc tốt cho biển số; nên ưu tiên loss/thiết kế thiên về text preservation hoặc edge preservation.
4. **Upscale vừa đủ** — không phải scale càng lớn càng tốt; với biển số, 2x thường an toàn hơn, 4x chỉ nên thử khi model đã thật sự ổn định.

### Các hướng SR nên thử ngoài 4 hướng của report

- **A. SR dạng nhẹ, ưu tiên OCR**: CNN/ResBlock SR nhỏ gọn, ít layer, tránh overfit/artifact — phù hợp làm baseline SR thực chiến. *(→ đây chính là hướng `LightEdgeSR` đã triển khai, xem mục 4.)*
- **B. SR có edge-aware loss**: thêm Sobel edge loss / gradient loss / Charbonnier + edge consistency — giữ nét ký tự thay vì chỉ giảm MSE.
- **C. SR có attention nhẹ**: channel attention hoặc residual attention, hữu ích nếu cần tập trung vùng chứa ký tự; không nên quá nặng vì dễ chậm/overfit.
- **D. Multi-frame SR có chọn lọc**: thay vì stack thẳng tất cả frame, chọn frame tốt nhất hoặc fuse có trọng số theo chất lượng — chỉ hữu ích nếu alignment giữa frame tốt.
- **E. SR tiền xử lý bằng denoise + deblur nhẹ**: denoise nhẹ, deblur nhẹ, contrast normalization trước SR — đưa ảnh LR về dạng dễ đọc hơn trước khi OCR.

### Cải thiện accuracy ngoài SR

1. **Augmentation theo domain biển số**: motion blur nhẹ, Gaussian blur nhẹ, JPEG compression, low-light/brightness shift, perspective transform nhẹ, random crop/resize mô phỏng camera xa.
2. **Tối ưu sequence-length và label quality**: kiểm tra mẫu nhãn sai/format không đồng nhất, loại sample lỗi quá mức nếu gây nhiễu lớn.
3. **Regularization có kiểm soát**: dropout nhẹ hơn ở frame fusion/RNN nếu overfit, label smoothing nhỏ để ổn định, giữ early stopping + warmup.
4. **Tuning backbone thay vì thay toàn bộ model**: số block, số channel, residual scale, SE on/off, frame dropout.
5. **Ensemble/late fusion**: bước cuối cùng sau khi đã tối ưu một model đơn, có thể tăng thêm vài phần trăm nhỏ.

### Kế hoạch thí nghiệm theo step

- **Step 1 — Chốt baseline tốt nhất**: CRNN thường vs. ResBlock stable vs. ResBlock stable + regularization tối ưu. Đánh giá bằng accuracy toàn tập, exact match, per-length accuracy nếu có. Kết quả kỳ vọng: chọn baseline ổn định làm mốc so sánh cho mọi SR mới.
- **Step 2 — Test SR nhẹ, không stacked-input**: SR 2x nhẹ bằng ResBlock / SR nhẹ + channel attention / SR nhẹ + edge-preserving loss. Quy tắc: không dùng kiến trúc quá nặng ngay từ đầu; nếu SR làm accuracy giảm, loại ngay.
- **Step 3 — Test SR theo hướng OCR-aware**: loss theo biên, loss theo gradient, giữ upsample vừa phải, giảm hallucination. Đánh giá thêm: độ ổn định trên sample tối, trên biển số mờ/nhoè.
- **Step 4 — Tối ưu input pipeline**: chọn frame tốt nhất thay vì gộp tất cả, align frame trước khi fuse, normalize contrast, tăng resize policy hợp lý.
- **Step 5 — Ablation có hệ thống**: mỗi lần chỉ thay 1 biến — có/không SE, res_scale khác nhau, frame_dropout khác nhau, fusion_dropout khác nhau, augmentation full/light, có/không SR, scale 2x vs 4x.

### Best SR nên ưu tiên theo thứ tự

1. **SR nhẹ, giữ nét ký tự** — ResBlock nhỏ gọn + loss thiên về edge preservation + scale 2x + ít artifact. Lý do: phù hợp OCR hơn SR "đẹp".
2. **SR có attention nhẹ** — channel attention, không dùng attention quá nặng, phù hợp khi cần nhấn vào vùng ký tự.
3. **Multi-frame SR có chọn lọc** — chỉ dùng nếu frame alignment tốt, tránh stack thô gây nhiễu, ưu tiên fusion theo chất lượng frame.
4. **PixelShuffle nếu cần tốc độ** — phù hợp làm upsampling cuối, nhẹ và phổ biến, nên kết hợp với backbone ổn định.

### Cấu hình thực nghiệm dự kiến

- **Cấu hình A — ổn định, làm baseline cuối**: ResBlock stable, augmentation full, LR vừa phải, early stopping, giữ SE.
- **Cấu hình B — SR OCR-aware nhẹ**: ResBlock nhẹ + edge loss, scale 2x, input sạch hơn, so sánh trực tiếp với A.
- **Cấu hình C — SR + attention**: channel attention nhẹ, giữ model nhỏ, đánh giá artifact và exact match.
- **Cấu hình D — multi-frame có chọn lọc**: chọn frame rõ nhất theo score, fusion nhẹ, chỉ thử khi A/B đã ổn.

### Tiêu chí chọn mô hình cuối

Chọn mô hình có: OCR accuracy cao nhất, ít lỗi ký tự gần giống nhau, ổn định trên ảnh mờ, không phụ thuộc quá nhiều vào artifact SR, dễ train và lặp lại.

### Kết luận ngắn

- Hiện tại hướng tốt nhất là **ResBlock stable**.
- SR chỉ nên dùng nếu SR đó giúp **giữ nét ký tự** và **không sinh artifact**.
- Không nên ưu tiên SR "đẹp ảnh" mà nên ưu tiên SR "đọc được biển số".
- Nên thử theo thứ tự: **SR nhẹ → SR + edge-aware loss → SR + attention → multi-frame chọn lọc**.
- Với dataset hiện tại, khả năng cải thiện lớn hơn có thể đến từ **thiết kế input + regularization + augmentation domain-specific** hơn là chỉ thay SR.
- Checklist thực thi: mỗi lần chạy chỉ thay một yếu tố; luôn so sánh trực tiếp với ResBlock stable ~76.68%; nếu SR không vượt baseline hoặc làm giảm accuracy, loại sớm.

---

## 3. SR kết hợp Stacked Inputs — phân tích khi chạy GPU

### Thiết kế ban đầu được đề xuất
- SR nhẹ (`LightweightSR`) theo hướng ResBlock + Channel Attention + PixelShuffle, áp dụng **theo từng frame** trước STN/CNN.
- Giữ nguyên input dạng `[B, 5, 3, 32, 128]`, Attention Fusion tiếp tục gộp 5 frame.
- Vị trí SR trong pipeline được đề xuất ban đầu: `5 LR frames → SR → STN → CNN → Attention Fusion → BiLSTM → CTC` (SR **trước** STN).

> **Lưu ý đối chiếu với code thực tế**: bản triển khai đầu tiên (`StackedSRNet`, xem phần 3.1 bên dưới) đặt SR trước STN đúng như đề xuất này, nhưng lại **gộp cả 5 frame thành 1 frame SR duy nhất rồi nhân bản ra 5 lần** trước khi vào STN — khác với ý tưởng "SR theo từng frame" ban đầu. Bản `LightEdgeSR` sau này (mục 4) đổi lại thứ tự thành **STN → SR** và áp dụng SR độc lập trên từng frame, sửa đúng vấn đề mất đa dạng frame.

### 3.1 Vì sao ban đầu đề xuất SR trước STN
- SR không thay thế STN: nếu ảnh biển số bị lệch góc/crop chưa chuẩn, SR chỉ làm ảnh rõ hơn chứ không tự căn chỉnh hình học.
- Lập luận: SR trước giúp tăng chi tiết đầu vào, STN sau đó vẫn có thể căn lại vùng biển số trên ảnh đã rõ hơn, CRNN nhận feature ổn định hơn nếu SR không làm méo ký tự.
- (Bản triển khai thực tế sau này — `LightEdgeSR` — lại chọn **STN trước, SR sau**, với lý do: chỉnh hình học trước rồi mới tăng cường biên sẽ tránh SR khuếch đại artifact từ việc ảnh chưa được căn chỉnh. Cả hai thứ tự đều có lý lẽ hợp lý riêng; đây là một biến cần ablation nếu muốn xác định thứ tự tối ưu.)

### 3.2 Chi phí GPU khi bật SR theo từng frame
Vì SR áp dụng cho từng frame trước khi vào STN/CNN, chi phí inference tăng gần theo số frame:
- tốc độ train có thể giảm rõ rệt so với baseline không SR;
- VRAM tiêu thụ tăng vì phải giữ thêm activation của SR;
- nếu batch size đang sát ngưỡng GPU, dễ gặp out-of-memory khi bật `--use-sr`.

### 3.3 Multi-frame làm bài toán khó hơn nhưng đáng thử hơn
Dataset có 5 frame/track nên stacked inputs là lợi thế lớn, nhưng khi chạy GPU model phải xử lý 5 frame + SR trên từng frame + fusion sau đó → pipeline nặng hơn, đổi lại có cơ hội khai thác thông tin bù trừ giữa các frame (frame mờ được bù bởi frame rõ, motion blur khác nhau, biên ký tự rõ hơn ở một số frame).

### 3.4 Các mode train nên chạy (đề xuất ban đầu)

```bash
# A. Baseline — CRNN + STN, không SR, dùng làm mốc đo thời gian/VRAM/accuracy
python train.py

# B. SR nhẹ — bật SR trước STN/CNN, mode quan trọng nhất để kiểm tra SR có giúp OCR không
python train.py --use-sr

# C. Tắt STN để ablation — kiểm tra STN đóng góp bao nhiêu
python train.py --no-stn

# D. Submission / train full data — chỉ chạy sau khi đã chốt cấu hình tốt nhất
python train.py --submission-mode
```

### 3.5 Ghi chú về stacked inputs / stacked SR mở rộng

Nếu SR được mở rộng sang multi-frame thật sự (không chỉ gộp-rồi-nhân-bản), model có thể học ưu tiên frame chất lượng tốt hơn, gom chi tiết từ nhiều frame để bù phần mất nét, làm ký tự ổn định hơn giữa các frame. Rủi ro: nếu frame lệch quá nhiều, fusion có thể làm nhiễu feature; nếu SR tạo artifact giống ký tự thật, CRNN có thể đọc sai; nếu chất lượng 5 frame chênh lệch lớn, model có thể bị "lẫn" thông tin.

### 3.6 Thứ tự ablation đề xuất
1. Baseline CRNN + STN
2. CRNN + STN + SR nhẹ
3. CRNN + STN + SR nhẹ + stacked ablation
4. CRNN + STN + SR nhẹ + multi-frame stacked SR

### 3.7 Kỳ vọng khi chạy thực tế (trước khi có số liệu)
- Nếu ảnh LR quá nhỏ/nén mạnh, SR nhẹ có thể cải thiện OCR accuracy.
- Nếu SR tạo artifact, accuracy có thể giảm dù ảnh nhìn sắc nét hơn.
- Nếu stacked inputs hoạt động tốt, accuracy thường tăng rõ hơn single-frame SR vì model có thêm thông tin theo thời gian.
- Đánh giá bằng **Exact Match accuracy** trên validation Scenario-B, đồng thời theo dõi loss hội tụ, VRAM, thời gian train/inference khi bật SR.

**Kết quả thực tế đo được sau đó** (xem mục 0 và [training_runs/run_gpu.md](../training_runs/run_gpu.md)): SR stacked-input v1 = 49.25%, v2 = 55.06% — **thấp hơn nhiều so với kỳ vọng và thấp hơn baseline ResBlock (~76.68%)**. Điều này xác nhận rủi ro "SR tạo artifact / làm nhiễu feature" đã nêu ở trên là có thật, cụ thể hơn: nguyên nhân chính là gộp 5 frame → 1 frame SR → nhân bản, làm mất đa dạng frame cho Attention Fusion khai thác.

---

## 4. Đã triển khai: LightEdgeSR (edge-preserving, per-frame)

Đây là hướng SR nhẹ **đã cài đặt trong code hiện tại** (`src/models/components.py`, `src/models/crnn.py`, `train.py`, `configs/config.py`), thuộc nhóm "A. SR dạng nhẹ, ưu tiên OCR" ở mục 2 và được thiết kế để sửa lỗi của SR stacked-input (mục 3).

### Mục tiêu
- Áp dụng một hướng Super Resolution nhẹ, ổn định hơn SR stacked-input trước đó.
- Tập trung giữ biên ký tự biển số thay vì chỉ làm ảnh nhìn đẹp hơn.
- Đưa SR vào pipeline theo cách không phá sequence length của CTC.

### Những gì đã xử lý

1. **Thêm adapter SR nhẹ, edge-preserving** — module `LightEdgeSR` trong `src/models/components.py`. Đặc điểm: upscale tạm thời bằng bilinear 2x → trích biên bằng Sobel → refine bằng conv nhẹ → trả ảnh về lại đúng resolution đầu vào để không đổi cấu trúc pipeline CRNN + CTC.
2. **Tích hợp vào model CRNN + STN** — `src/models/crnn.py` hỗ trợ `use_sr`, `sr_scale`, `sr_hidden_channels`. SR được đặt **sau STN, trước backbone CNN** (khác thứ tự đề xuất ban đầu ở mục 3.1), nhằm: chỉnh hình học trước → tăng độ rõ biên sau → rồi mới trích feature cho OCR.
3. **Mở CLI** — `train.py` có `--use-sr`, `--sr-scale`, `--sr-hidden-channels`.
4. **Mở config mặc định** — `configs/config.py` có `USE_SR`, `SR_SCALE`, `SR_HIDDEN_CHANNELS`.

### Pipeline hiện tại

```
1. Input 5 frame
2. STN căn chỉnh từng frame
3. LightEdgeSR tăng cường biên ở mức nhẹ (áp dụng ĐỘC LẬP theo từng frame)
4. Backbone ResBlock trích feature
5. Attention fusion qua frames
6. BiLSTM + CTC decoding
```

Mục tiêu: giữ nguyên độ ổn định của ResBlock stable, thêm lợi ích từ SR mà không đưa artifact quá mạnh vào OCR — và quan trọng nhất, **giữ đa dạng giữa 5 frame** (khác hẳn lỗi thiết kế của SR stacked-input ở mục 3).

### Lý do chọn hướng này
- SR stacked-input trước đó cho kết quả thấp hơn nhiều baseline (mục 0, mục 3.7).
- Với OCR biển số, quan trọng nhất là đường biên ký tự rõ và ít nhiễu.
- Adapter SR nhẹ này không thay toàn bộ backbone → rủi ro thấp hơn.
- Output được đưa về lại size ban đầu → không đổi sequence layout của CTC.

### Cách chạy thử

```bash
python train.py \
  --preset stable \
  --experiment-name crnn_resblock_edge_sr \
  --batch-size 64 \
  --epochs 80 \
  --lr 0.0008 \
  --num-workers 8 \
  --aug-level full \
  --use-sr \
  --sr-scale 2 \
  --sr-hidden-channels 32
```

Lệnh sanity-check (chạy trước, 5 epoch) và lệnh so sánh đầy đủ (80 epoch, khớp hyperparameter với `crnn_resblock_stable` để so sánh táo-với-táo) đã có sẵn tại [training_runs/run_gpu.md](../training_runs/run_gpu.md).

### Lưu ý quan trọng
- Đây là hướng SR nhẹ ưu tiên OCR, không phải SR phục vụ ảnh đẹp.
- Nếu accuracy tăng, giữ nguyên để ablation tiếp theo.
- Nếu không tăng, tiếp tục thử: giảm `sr_hidden_channels`, giữ `sr_scale=2`, hoặc tắt SR với từng nhóm augmentation để xác định tác động thật.

### Điểm cần theo dõi khi có kết quả thực nghiệm (đánh giá kỹ thuật bổ sung)

- Forward pass của `LightEdgeSR` hiện tại: upsample bilinear 2x → refine (Sobel edge + conv) → **downsample bilinear ngay lập tức về lại kích thước gốc**. Downsample bilinear là phép low-pass, có thể làm mờ lại một phần chi tiết vừa được refine, nên lợi ích thực tế lọt qua tới backbone có thể nhỏ hơn kỳ vọng. Đây là điểm nên kiểm chứng bằng ablation nếu kết quả không đạt kỳ vọng — ví dụ thử **không downsample về lại kích thước gốc** (chấp nhận sequence CTC dài hơn ~2x, vì CTC chỉ cần `seq_len >= label_len`, không cần seq_len cố định).
- `res_scale = 0.2` trong `LightEdgeSR` hiện đang hardcode, chưa expose ra CLI như `sr_scale`/`sr_hidden_channels` — có thể bổ sung nếu cần ablation riêng cho tham số này.

---

## 5. Kết luận tổng hợp

- Bốn hướng SR trong report (ResBlock / Channel Attention / PixelShuffle / Stacked Inputs) đều là đề xuất **chưa có số liệu thực nghiệm** trong report gốc — project này là nơi kiểm chứng thực tế.
- Trong các hướng đã thử nghiệm thực tế: **ResBlock backbone upgrade (không phải SR, mà là thay backbone chính)** đang là cải tiến duy nhất đã xác nhận vượt baseline (~76.68% so với ~75-77%).
- **SR stacked-input (gộp 5 frame → 1 → nhân bản) đã thử và cho kết quả kém hẳn baseline** (49-55%), nhiều khả năng do phá vỡ đa dạng frame cần thiết cho Attention Fusion.
- **LightEdgeSR (per-frame, edge-preserving) đã cài đặt xong, sửa đúng lỗi thiết kế của stacked-input, nhưng chưa có kết quả thực nghiệm** — cần chạy theo lệnh ở mục 4 và log kết quả vào `run_gpu.md` trước khi kết luận có nên giữ SR trong pipeline hay không.
- Nguyên tắc xuyên suốt cho mọi hướng SR tiếp theo: đánh giá bằng **OCR exact-match accuracy**, không phải PSNR/SSIM; ưu tiên SR "đọc được biển số" hơn SR "đẹp ảnh"; nếu SR không vượt baseline ResBlock stable, loại sớm và không đầu tư thêm effort vào nhánh đó.
