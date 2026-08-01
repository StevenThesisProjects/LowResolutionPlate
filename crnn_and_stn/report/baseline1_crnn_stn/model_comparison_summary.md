# So sánh tổng hợp — toàn bộ cấu hình đã chạy (Baseline 1: CRNN + STN)

> File điều hướng: gom kết quả từ 8 tài liệu trong thư mục này +
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
| 11 | **S3 — S1 + L_Perceptual (α=0.1)** | **805** | **80.58%** | **+8 track vs #9** | có (chưa đủ bằng chứng, nhưng cao nhất đã đo) | [s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md) |

**Đọc bảng này thế nào**: cột "Trong biên nhiễu ±13?" quan trọng hơn cột Val Acc.
Val 999 track cho biên nhiễu thống kê **±13 track (±1.3 điểm)**
(xem [groupnorm_sr_ablation_j1_j2.md §4](groupnorm_sr_ablation_j1_j2.md#4-caveat-thống-kê--quan-trọng-hơn-mọi-con-số-ở-trên)).
Mọi chênh lệch nhỏ hơn ngưỡng đó (#1→#5→#6→#7→#8, #9→#10, #9→#11) là nhiễu đo
lường, không phải bằng chứng cải tiến thật **giữa các dòng đó với nhau**. Nhưng
so với mốc xa hơn (#0 hoặc #7), cả #9 và #11 đều vượt hẳn biên nhiễu — tức "S1 và
S3 tốt hơn baseline/J2" đáng tin, còn "S3 tốt hơn S1" hay "S1 tốt hơn S2" thì chưa.

## 2. Đánh giá so sánh giữa các model

**Xếp hạng theo điểm đo được** (999 track, cùng decode constrained trừ khi ghi chú):

1. 🥇 **S3** (80.58%, 805 track) — điểm cao nhất từng đo, nhưng chỉ 1 run.
2. 🥈 **S1** (79.78%, 797 track) — kết quả tin cậy nhất (duy nhất vượt rõ biên nhiễu so với J2).
3. 🥉 **S2** (79.48%, 794 track) — bằng S1 trong biên nhiễu, không có lý do chọn.
4. J2 (77.18%, 771 track) — mốc cũ trước khi có S-series.
5. Baseline report gốc (77.00%) — mốc tham chiếu chuẩn của toàn bộ project.

**Nhận xét chính:**

- **S1, S2, S3 dùng chung 1 kiến trúc** (STN+DCN+MFSR+domain-match+decode
  constrained+EMA), chỉ khác 1 tham số loss mỗi lần (λ_SR hoặc perceptual weight)
  — đây là 2 ablation sạch tách từ S1, không phải 2 hướng độc lập.
- **Tăng λ_SR (S2) không giúp gì** — kết quả âm tính rõ ràng, nên loại khỏi cân nhắc tiếp.
- **Thêm L_Perceptual (S3) là tín hiệu tích cực nhất đã đo** (+8 track so với S1)
  nhưng chưa đủ để tuyên bố "tốt hơn S1" — 8 track vẫn nằm trong biên nhiễu ±13.
  Cần multi-seed để phân định thật giữa S1 và S3.
- **Không có cấu hình nào miễn phí** — cả 3 (S1/S2/S3) tốn ~3.85x compute so với
  ResBlock trơn (mục 4). S3 tốn thêm nữa vì phải forward qua VGG16 mỗi batch,
  nhưng đổi lại hội tụ nhanh hơn hẳn (đỉnh ở epoch 25 so với 37-38 của S1/S2).
- **Nếu phải chọn 1 cấu hình để nộp submission ngay bây giờ**: S3, vì có điểm cao
  nhất và không có bằng chứng nào cho thấy nó tệ hơn S1 ở bất kỳ mặt nào khác
  (0/999 track lỗi độ dài, tốt hơn cả S1). Nhưng đây là lựa chọn "tốt nhất trong
  điều kiện chưa multi-seed", không phải kết luận thống kê chắc chắn.

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

Bảng trên **chưa tính thêm chi phí VGG16 của S3** (forward-only, không train,
nhưng vẫn tốn thêm compute mỗi batch) — chưa benchmark riêng. Ở J2 (kiến trúc
cũ), SR đổi 3.66x compute lấy 3 track (trong biên nhiễu — đổi chác tệ). **Ở S1,
3.85x compute đổi lấy 26 track** (ngoài biên nhiễu) — tỷ lệ đổi chác tốt hơn hẳn.

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
| Lệnh chạy GPU đầy đủ + bảng compute + hướng dẫn vẽ chart | [../training_runs/run_gpu.md](../training_runs/run_gpu.md) |
| Cấu trúc dataset, plate layout, corners annotation | [../summary_project/dataset/dataset_overview.md](../summary_project/dataset/dataset_overview.md) |

## 6. Bước tiếp theo (chưa chạy)

1. **S4** — `--sr-scale 1 --width-downsample 4` (kiểm tra giới hạn ~55% pixel nội
   suy do HR gốc chỉ ~115×42px).
2. **O1** — multi-seed (42/100/2026), giờ nên chạy trên **cả S1 và S3** (không
   chỉ S1) vì S3 hiện là điểm ước lượng tốt nhất — điều kiện bắt buộc trước khi
   chốt cấu hình cuối để báo cáo/nộp submission.
3. Ablation tách 6 thành phần của S1 (đóng góp riêng của DCN/domain-match/EMA) —
   giá trị thông tin cao nhưng tốn nhiều run nhất, làm sau cùng nếu còn thời gian.
