# So sánh tổng hợp — toàn bộ cấu hình đã chạy (Baseline 1: CRNN + STN)

> File điều hướng: gom kết quả từ 9 tài liệu trong thư mục này +
> [../training_runs/run_gpu.md](../training_runs/run_gpu.md) thành 1 bảng duy nhất.
> Không lặp lại phân tích chi tiết — mỗi dòng trỏ tới đúng file để đọc sâu.
> Toàn bộ Val Acc đo trên cùng 1 tập **999 track Scenario-B** (`dataset/val_tracks.json`),
> trừ dòng "report gốc" (đo trên tập của tác giả report, không phải tập này).

## 1. Bảng tổng hợp theo trình tự thời gian

| # | Cấu hình | Track đúng | Val Acc | Δ vs mốc trước | Trong biên nhiễu ±13? | Tài liệu |
|---|---|---:|---:|---:|:---:|---|
| 0 | CRNN + STN (report ICPR gốc) | — | 77.00% | — (mốc tham chiếu chuẩn) | — | [baseline1_architecture.md](baseline1_architecture.md) |
| 1 | CRNN + STN (đo lại trên dataset project) | 757 | 75.78% | −1.22 vs #0 | có | [baseline1_architecture.md](baseline1_architecture.md), [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 2 | — aug light (augmentation nhẹ) | 739 | 73.97% | −18 track vs #1 | **không** (kém hơn rõ) | chỉ có trong `run_gpu.md` |
| 3 | SR-v1 — stacked-input SR (bản lỗi, gộp 5 frame→1) | 492 | 49.25% | −265 track vs #1 | **không** (thất bại nặng) | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |
| 4 | SR-v2 — stacked-input SR, lr thấp + aug light | 550 | 55.06% | −207 track vs #1 | **không** (vẫn thất bại) | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |
| 5 | ResBlock backbone (`norm=none`, PR #8) | 766 | 76.68% | +9 track vs #1 | có | [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) |
| 6 | J1 — ResBlock + GroupNorm, không SR | 768 | 76.88% | +2 track vs #5 | có | [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md) |
| 7 | J2 — + SR per-frame (single-frame) có giám sát | 771 | 77.18% | +3 track vs #6 | có | [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md) |
| 8 | J3 — + DCNv2 (kernel init ngẫu nhiên, bug) | 762 | 76.28% | −9 track vs #7 | có | [s1_proposed_mf_sr_ocr.md §2](s1_proposed_mf_sr_ocr.md#2-kết-quả-chính) |
| 9 | **S1 — Joint MF-SR-OCR, λ_SR=0.1 (đề xuất)** | **797** | **79.78%** | **+26 track vs #7 (J2)** | **KHÔNG — vượt rõ ràng** | [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) |
| 10 | S2 — S1 + `λ_SR=0.5` thay vì 0.1 | 794 | 79.48% | −3 track vs #9 | có (không cải thiện) | [s2_lam05_mf_sr_ocr.md](s2_lam05_mf_sr_ocr.md) |
| 11 | S3 — S1 + L_Perceptual (α=0.1) | 805 | 80.58% | +8 track vs #9 | có (chưa đủ bằng chứng, nhưng đồng hạng cao nhất) | [s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md) |
| 12 | **S4 — SR scale=1 (T=32 qua `width_downsample=4`)** | **805** | **80.58%** | **+8 track vs #9, hoà #11** | có (nhưng 2 hướng độc lập cùng hội tụ) | [s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md) |

**Đọc bảng này thế nào**: cột "Trong biên nhiễu ±13?" quan trọng hơn cột Val Acc.
Val 999 track cho biên nhiễu thống kê **±13 track (±1.3 điểm)**
(xem [groupnorm_sr_ablation_j1_j2.md §4](groupnorm_sr_ablation_j1_j2.md#4-caveat-thống-kê--quan-trọng-hơn-mọi-con-số-ở-trên)).
Mọi chênh lệch nhỏ hơn ngưỡng đó (#1→#5→#6→#7→#8, #9→#10, #9→#11, #9→#12) là
nhiễu đo lường, không phải bằng chứng cải tiến thật **giữa các dòng đó với nhau**.
Nhưng so với mốc xa hơn (#0 hoặc #7), cả #9, #11, #12 đều vượt hẳn biên nhiễu —
tức "S1/S3/S4 tốt hơn baseline/J2" đáng tin, còn thứ hạng nội bộ giữa S1/S3/S4
(hay S1 so với S2) thì chưa. Đáng chú ý: **#11 và #12 đạt đúng cùng 1 điểm số
(805/999) bằng 2 cách hoàn toàn độc lập** (S3: giữ SR ×2, thêm perceptual loss;
S4: bỏ upsampling của SR, đổi cách lấy T=32) — 2 điểm hội tụ độc lập là tín hiệu
đáng tin hơn 1 điểm đơn lẻ vượt biên nhiễu, dù không thay thế được multi-seed.

## 2. Đánh giá so sánh giữa các model

**Xếp hạng theo điểm đo được** (999 track, cùng decode constrained trừ khi ghi chú):

1. 🥇 **S3 = S4** (80.58%, 805 track, đồng hạng) — điểm cao nhất từng đo, đạt bằng 2 cách độc lập.
2. 🥉 **S1** (79.78%, 797 track) — kết quả tin cậy nhất (duy nhất vượt rõ biên nhiễu so với J2).
3. **S2** (79.48%, 794 track) — bằng S1 trong biên nhiễu, không có lý do chọn.
4. J2 (77.18%, 771 track) — mốc cũ trước khi có S-series.
5. Baseline report gốc (77.00%) — mốc tham chiếu chuẩn của toàn bộ project.

**Nhận xét chính:**

- **S1, S2, S3, S4 dùng chung 1 kiến trúc nền** (STN+DCN+MFSR+domain-match+decode
  constrained+EMA), mỗi lần chỉ đổi cách cấu hình nhánh SR (trọng số loss, thêm
  loss, hoặc bỏ upsampling) — đây là 3 ablation tách từ S1, không phải 3 hướng độc lập.
- **Tăng λ_SR (S2) không giúp gì** — kết quả âm tính rõ ràng, loại khỏi cân nhắc tiếp.
- **Thêm L_Perceptual (S3) và bỏ upsampling giữ T=32 (S4) đều cho +8 track so với
  S1** — từng cái riêng lẻ vẫn trong biên nhiễu ±13 nên không "chắc chắn", nhưng
  2 con đường độc lập cùng dừng lại ở đúng 805/999 là tín hiệu đồng thuận mạnh hơn
  một điểm số đơn lẻ. **Phát hiện quan trọng nhất từ S4**: khi giữ `T=32` bằng
  cách khác (không phóng to ảnh) thay vì để SR ×2 tự động tăng T, độ chính xác
  không hề giảm — nghiêng về giả thuyết `T=32` (nhiều bước CTC hơn), chứ không
  phải bản thân việc ảnh được phóng to, mới là yếu tố chính đứng sau lợi ích của
  "SR" đo được ở S1. Chưa có ablation "T=32 mà không SR" để khẳng định dứt điểm.
- **Không có cấu hình nào miễn phí** — S1/S2/S3 tốn ~3.85x compute so với ResBlock
  trơn (mục 4); **S4 nhiều khả năng rẻ hơn hẳn** (xử lý ảnh 32×128 thay vì 64×256
  qua backbone) nhưng chưa benchmark thật. S3 tốn thêm vì phải forward qua VGG16
  mỗi batch, đổi lại hội tụ nhanh nhất (đỉnh epoch 25).
- **Nếu phải chọn 1 cấu hình để nộp submission ngay bây giờ**: **S4** — điểm bằng
  S3 nhưng confidence cao nhất (0.9735), 0/999 track confidence thấp (<0.55, so
  với 9 của S3), và nhiều khả năng rẻ hơn về compute. S3 là lựa chọn thứ hai hợp
  lý nếu ưu tiên 0/999 track lỗi độ dài. Đây là lựa chọn "tốt nhất trong điều kiện
  chưa multi-seed", không phải kết luận thống kê chắc chắn — xem mục 6.

## 2b. Metrics mức ký tự — CER / NED (review Bước 2)

Tính lại từ `submission_*.txt` + nhãn gốc, **không cần train lại**. CER là mức corpus
(tổng edit distance / tổng ký tự nhãn), NED là trung bình mỗi track
(`ED / max(len_ref, len_hyp)`). Cả hai **thấp = tốt**; paper hay ghi `1−NED`.

| Cấu hình | Exact Match | CER ↓ | NED ↓ | 1−NED ↑ |
|---|---:|---:|---:|---:|
| S1 (λ=0.1) | 79.78% | 0.0562 | 0.0562 | 0.9438 |
| S2 (λ=0.5) | 79.48% | 0.0549 | 0.0549 | 0.9451 |
| **S3 (+perceptual)** | **80.58%** | **0.0522** | **0.0522** | **0.9478** |
| S4 (SR ×1) | 80.58% | 0.0532 | 0.0532 | 0.9468 |

Hai điều exact-match không cho thấy được:

1. **S2 kém S1 về exact match nhưng TỐT hơn về CER** (0.0549 vs 0.0562) — khi S2 sai,
   nó sai "gần đúng" hơn. Exact match phạt sai 1 ký tự như sai cả biển nên giấu mất
   điều này.
2. **CER phá được thế hoà S3 ↔ S4.** Hai cấu hình bằng nhau tuyệt đối ở exact match
   (805/999) nhưng S3 có CER tốt hơn (0.0522 vs 0.0532) — tiêu chí đầu tiên tách được
   hai ứng viên tốt nhất. Chênh lệch nhỏ, vẫn cần multi-seed xác nhận.

CER và NED trùng nhau vì gần như mọi dự đoán đều đúng 7 ký tự, nên
`max(len_ref, len_hyp)` luôn bằng `len_ref`.

### PSNR / SSIM — chất lượng tái tạo ảnh của nhánh SR

Đo bằng `tools/eval_sr_quality.py` trên 999 track val (4.995 ảnh). `base` = ảnh chưa
qua SR. **Con số đáng đọc là cột "Chênh"**, không phải giá trị tuyệt đối.

| Cấu hình | PSNR (SR) | PSNR (base) | Chênh | SSIM (SR) | SSIM (base) | Chênh |
|---|---:|---:|---:|---:|---:|---:|
| S1 (×2) | 16.6827 | 15.6110 | +1.0717 dB | 0.4179 | 0.3481 | +0.0698 |
| **S2** (×2, λ=0.5) | 18.5228 | 16.2349 | **+2.2879 dB** | 0.5461 | 0.3780 | **+0.1681** |
| S3 (×2, perceptual) | 16.9582 | 15.8727 | +1.0855 dB | 0.4234 | 0.3569 | +0.0664 |
| S4 (×1) | 17.5249 | 16.5011 | +1.0238 dB | 0.5027 | 0.4408 | +0.0618 |

⚠️ **Không so PSNR tuyệt đối giữa S4 và S1/S2/S3** — S1–S3 xuất ảnh 64×256, S4 xuất
32×128, hai thang khác nhau. Chỉ cột "Chênh" so được, vì mỗi cấu hình so với base của
chính nó.

**Phát hiện quan trọng — PSNR/SSIM NGHỊCH với khả năng đọc biển số:**

Ở mức từng track (n=999, ghép PSNR với đúng/sai từng track):

| Cấu hình | PSNR track **đọc đúng** | PSNR track **đọc sai** | r(PSNR, đúng) |
|---|---:|---:|---:|
| S1 | 16.288 | 18.240 | **−0.362** |
| S2 | 18.102 | 20.154 | **−0.346** |
| S3 | 16.492 | 18.894 | **−0.412** |
| S4 | 17.080 | 19.371 | **−0.400** |

**Track đọc SAI lại có PSNR CAO hơn ~2 dB**, tương quan âm nhất quán ở cả 4 cấu hình.
Ở mức cấu hình cũng cùng chiều: S2 có PSNR tốt nhất (+2.29 dB, hơn gấp đôi mọi cấu
hình khác) nhưng OCR **kém nhất** (794/999).

Giả thuyết: PSNR bị chi phối bởi *nội dung ảnh* — ảnh mờ/tương phản thấp có sai khác
pixel nhỏ nên PSNR cao, nhưng đúng là loại khó đọc nhất. Tức PSNR đo "ảnh này dễ tái
tạo tới đâu", còn OCR cần "ảnh này chứa bao nhiêu chi tiết đọc được".

→ Tối ưu theo PSNR **đẩy model đi sai hướng**, không chỉ là vô ích. Đây là câu trả lời
cho câu hỏi "sao không tối ưu theo PSNR", và là bằng chứng số cho nguyên tắc đặt ra từ
đầu project: đánh giá SR bằng OCR exact-match, không phải PSNR/SSIM.

Chi tiết + 3 chỗ lệch so với review: [../buoc2_metrics.md](../buoc2_metrics.md).

### Hình định tính (Figure 4) — đã sinh cho cả 4 cấu hình

`results/mf_sr_ocr/<cấu hình>/paper_figures/` — mỗi cấu hình có
`figure4_qualitative_grid.png` (grid 4 cột `I_LR → I_SR → Attention → Prediction`,
5 case đúng + 5 case sai) và 10 ảnh từng track riêng.

**Track xuất hiện ở nhiều cấu hình** — tiện để so trực tiếp khi chọn hình:

| Track | Xuất hiện ở | Dùng để minh hoạ |
|---|---|---|
| `track_22161`, `track_19095` | **sai ở cả 4** | giới hạn thật của dữ liệu, không phải điểm yếu của một model |
| `track_21455` | đúng ở S1/S2/S3 | case dễ, đọc chắc chắn |
| `track_12478`, `track_17959` | sai ở S2 và S3 | |
| `track_14442` | chỉ sai ở S4 | |

⚠️ Danh sách "10 track tiêu biểu" **không tái lập chính xác khi đổi phần cứng** — thứ
tự xếp theo confidence lệch nhau ở các track có confidence gần bằng nhau (khác biệt số
thực GPU vs CPU). Nên chốt một bộ hình và giữ nguyên, đừng chạy lại rồi lấy bộ khác.

## 3. Nhánh phụ — AdamW tuning trên backbone cũ (lịch sử, không so trực tiếp được)

Chạy trên backbone CNN+BatchNorm gốc, nay không còn tồn tại trong code (đã bị
thay bởi ResBlock ở dòng #5):

| Cấu hình | Val Acc | Δ vs baseline (75.78%) |
|---|---:|---:|
| Baseline (bug: weight-decay áp sai lên bias/BN) | 75.78% | — |
| A — sửa bug param-grouping | 75.98% | +0.20 |
| B — wd cao + warmup ngắn (2 biến cùng lúc) | 75.58% | −0.20 |
| F — wd + warmup mức vừa | 76.08% | +0.30 |
| C — chỉ tăng weight_decay | 76.18% | +0.40 |
| D — chỉ rút ngắn warmup (tốt nhất) | 76.28% | +0.50 |
| E — D + train dài hơn (60 epoch) | 75.08% | −0.70 |

Toàn bộ A-F nằm trong biên nhiễu. Trần 76.28% thấp hơn ResBlock (#5, 76.68%) đạt
được mà không cần tuning — kết luận: giới hạn nằm ở backbone, không phải
optimizer. Chi tiết: [optimizer_adamw_verification.md](optimizer_adamw_verification.md).

## 4. Chi phí compute (kiến trúc hiện tại, STN pool `(4,8)`)

Đo bằng `tools/benchmark.py --all` (CPU). Nguồn: [run_gpu.md §2](../training_runs/run_gpu.md).

| Cấu hình | Params | GFLOPs/track | Latency (ms) | vs base |
|---|---:|---:|---:|---:|
| ResBlock (không SR) | 29,427,468 | 26.14 | 50.03 | 1.00x |
| + GroupNorm (J1) | 29,442,700 | 26.14 | 51.30 | 1.03x |
| + SR per-frame (J2 kiến trúc) | 29,558,255 | 108.31 | 184.43 | 3.69x |
| + SR + DCNv2 = **S1 / S2 / S3** (cùng kiến trúc, chỉ khác loss) | 29,577,214 | 109.08 | 192.75 | 3.85x |
| S4 (SR scale=1, `width_downsample=4`) | 29,549,470 | chưa đo | chưa đo | **có thể < 3.85x** |

Bảng trên **chưa tính thêm chi phí VGG16 của S3** (forward-only, không train,
nhưng vẫn tốn thêm compute mỗi batch) — chưa benchmark riêng. Ở J2 (kiến trúc
cũ), SR đổi 3.66x compute lấy 3 track (trong biên nhiễu — đổi chác tệ). **Ở S1,
3.85x compute đổi lấy 26 track** (ngoài biên nhiễu) — tỷ lệ đổi chác tốt hơn hẳn.
**S4 chưa được `tools/benchmark.py` đo**, nhưng vì SR scale=1 không phóng to ảnh,
backbone/BiLSTM của S4 xử lý ảnh 32×128 thay vì 64×256 như S1/S3 — nhiều khả năng
rẻ hơn đáng kể trong khi điểm số bằng S3 (xem [s4_sr_scale1_mf_sr_ocr.md §3](s4_sr_scale1_mf_sr_ocr.md#3-trả-lời-câu-hỏi-t-confound--kết-quả-chính-của-s4)) — nên benchmark thật trước khi coi đây là kết luận.

## 5. Bản đồ tài liệu — đọc gì khi cần gì

| Muốn biết... | Đọc file |
|---|---|
| Kiến trúc gốc, đối chiếu với report ICPR | [baseline1_architecture.md](baseline1_architecture.md) |
| SR thất bại lúc đầu vì sao (stacked-input bug) | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |
| AdamW/weight-decay ablation (lịch sử, backbone cũ) | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| ResBlock backbone thay BatchNorm bằng gì, vì sao | [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) |
| GroupNorm giúp gì, SR per-frame có giám sát lần đầu | [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md) |
| Cấu hình đề xuất đầy đủ (MFSR+DCN+domain-match+decode+EMA) | [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) |
| λ_SR=0.5 có tốt hơn 0.1 không | [s2_lam05_mf_sr_ocr.md](s2_lam05_mf_sr_ocr.md) |
| L_Perceptual có giúp gì không | [s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md) |
| SR ×1 (không phóng to ảnh) có bằng SR ×2 không, "T confound" là gì | [s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md) |
| Lệnh chạy GPU đầy đủ + bảng compute + hướng dẫn vẽ chart | [../training_runs/run_gpu.md](../training_runs/run_gpu.md) |
| Cấu trúc dataset, plate layout, corners annotation | [../summary_project/dataset/dataset_overview.md](../summary_project/dataset/dataset_overview.md) |

## 6. Bước tiếp theo (chưa chạy)

Cả 4 cấu hình đề xuất theo kế hoạch ban đầu (S1-S4) đã chạy xong. Việc còn lại:

1. **Ablation "T=32 không SR"** — `--width-downsample 4` **không** `--use-sr` —
   chưa nằm trong kế hoạch gốc, nhưng cần để tách dứt điểm câu hỏi S4 đặt ra (mục
   2): liệu module SR có đóng góp gì ngoài việc cho phép T=32, hay gần như không.
2. **Benchmark compute cho S4** — `tools/benchmark.py --all` chưa có dòng cho cấu
   hình `sr-scale=1`, cần đo để xác nhận giả thuyết "S4 rẻ hơn" ở mục 4.
3. **O1 — multi-seed** (42/100/2026), giờ nên chạy trên **cả S1, S3, S4** (không
   chỉ S1) vì S3/S4 hiện là điểm ước lượng tốt nhất — điều kiện bắt buộc trước
   khi chốt cấu hình cuối để báo cáo/nộp submission.
4. Ablation tách 6 thành phần của S1 (đóng góp riêng của DCN/domain-match/EMA) —
   giá trị thông tin cao nhưng tốn nhiều run nhất, làm sau cùng nếu còn thời gian.
