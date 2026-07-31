# So sánh tổng hợp — toàn bộ cấu hình đã chạy (Baseline 1: CRNN + STN)

> File điều hướng: gom kết quả từ 7 tài liệu trong thư mục này +
> [../training_runs/run_gpu.md](../training_runs/run_gpu.md) thành 1 bảng duy nhất.
> Không lặp lại phân tích chi tiết — mỗi dòng trỏ tới đúng file để đọc sâu.
> Toàn bộ Val Acc đo trên cùng 1 tập **999 track Scenario-B** (`dataset/val_tracks.json`),
> trừ dòng "report gốc" (đo trên tập của tác giả report, không phải tập này).

## 1. Bảng tổng hợp theo trình tự thời gian

| # | Cấu hình | Track đúng | Val Acc | Δ vs mốc trước | Trong biên nhiễu ±13? | Tài liệu |
|---|---|---:|---:|---:|:---:|---|
| 0 | CRNN + STN (report ICPR gốc) | — | 77.00% | — (mốc tham chiếu chuẩn) | — | [baseline1_architecture.md](baseline1_architecture.md) |
| 1 | CRNN + STN (đo lại trên dataset project) | 757 | 75.78% | −1.22 vs #0 | có | [baseline1_architecture.md](baseline1_architecture.md), [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 2 | — aug light (augmentation nhẹ) | 739 | 73.97% | −18 track vs #1 | **không** (kém hơn rõ) | chỉ có trong `run_gpu.md`, không có phân tích riêng |
| 3 | SR-v1 — stacked-input SR (bản lỗi, gộp 5 frame→1) | 492 | 49.25% | −265 track vs #1 | **không** (thất bại nặng) | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |
| 4 | SR-v2 — stacked-input SR, lr thấp + aug light | 550 | 55.06% | −207 track vs #1 | **không** (vẫn thất bại) | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |
| 5 | ResBlock backbone (`norm=none`, PR #8) | 766 | 76.68% | +9 track vs #1 | có | [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) |
| 6 | J1 — ResBlock + GroupNorm, không SR | 768 | 76.88% | +2 track vs #5 | có | [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md) |
| 7 | J2 — + SR per-frame (single-frame) có giám sát | 771 | 77.18% | +3 track vs #6 | có | [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md) |
| 8 | J3 — + DCNv2 (kernel init ngẫu nhiên, bug) | 762 | 76.28% | −9 track vs #7 | có | [s1_proposed_mf_sr_ocr.md §2](s1_proposed_mf_sr_ocr.md#2-kết-quả-chính) |
| 9 | **S1 — Joint MF-SR-OCR, λ_SR=0.1 (đề xuất)** | **797** | **79.78%** | **+26 track vs #7 (J2)** | **KHÔNG — vượt rõ ràng** | [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) |
| 10 | S2 — Joint MF-SR-OCR, λ_SR=0.5 | 794 | 79.48% | −3 track vs #9 | có (không cải thiện) | [s2_lam05_mf_sr_ocr.md](s2_lam05_mf_sr_ocr.md) |

**Đọc bảng này thế nào**: cột "Trong biên nhiễu ±13?" là thứ quan trọng nhất, không
phải cột Val Acc. Val 999 track cho biên nhiễu thống kê **±13 track (±1.3 điểm)**
(xem [groupnorm_sr_ablation_j1_j2.md §4](groupnorm_sr_ablation_j1_j2.md#4-caveat-thống-kê--quan-trọng-hơn-mọi-con-số-ở-trên)) —
mọi chênh lệch nhỏ hơn ngưỡng đó (#1→#5→#6→#7→#8, #9→#10) là nhiễu đo lường, không
phải bằng chứng cải tiến thật. **Chỉ có #9 (S1) vượt ngưỡng này một cách rõ ràng**
kể từ mốc gốc #0 — đây là kết quả duy nhất trong toàn bộ bảng đủ điều kiện gọi là
"cải thiện đã đo được", dù vẫn cần multi-seed để xác nhận chắc chắn (xem mục 5).

## 2. Nhánh phụ — AdamW tuning trên backbone cũ (không nằm trong timeline chính)

Chạy song song, **trên backbone CNN+BatchNorm gốc** — backbone đó không còn tồn
tại trong code hiện tại (đã bị thay bởi ResBlock ở dòng #5). Kết quả chỉ còn giá
trị tham khảo phương pháp luận, không so sánh trực tiếp được với bảng trên:

| Cấu hình | Val Acc | Δ vs baseline (75.78%) |
|---|---:|---:|
| Baseline (bug: weight-decay áp sai lên bias/BN) | 75.78% | — |
| A — sửa bug param-grouping | 75.98% | +0.20 |
| B — wd cao + warmup ngắn (2 biến cùng lúc) | 75.58% | −0.20 |
| F — wd + warmup mức vừa | 76.08% | +0.30 |
| C — chỉ tăng weight_decay | 76.18% | +0.40 |
| D — chỉ rút ngắn warmup (tốt nhất) | 76.28% | +0.50 |
| E — D + train dài hơn (60 epoch) | 75.08% | −0.70 |

Toàn bộ A-F nằm trong biên nhiễu ±13 track (±2.7 điểm ở thời điểm đo — ước lượng
ban đầu, sau này tinh chỉnh thành ±1.3 điểm). Trần 76.28% của nhánh này thấp hơn
ResBlock (#5, 76.68%) đạt được mà không cần tuning — kết luận cuối: giới hạn nằm
ở backbone, không phải optimizer. Chi tiết: [optimizer_adamw_verification.md](optimizer_adamw_verification.md).

## 3. Chi phí compute (kiến trúc hiện tại, STN pool `(4,8)`)

Đo bằng `tools/benchmark.py --all` (CPU). Nguồn: [run_gpu.md §2](../training_runs/run_gpu.md).

| Cấu hình | Params | GFLOPs/track | Latency (ms) | vs base |
|---|---:|---:|---:|---:|
| ResBlock (không SR) | 29,427,468 | 26.14 | 50.03 | 1.00x |
| + GroupNorm (J1) | 29,442,700 | 26.14 | 51.30 | 1.03x |
| + SR per-frame (J2 kiến trúc) | 29,558,255 | 108.31 | 184.43 | 3.69x |
| + SR + DCNv2 = **S1 / S2** (cùng kiến trúc, chỉ khác λ_SR) | 29,577,214 | 109.08 | 192.75 | 3.85x |

Ở J2 (kiến trúc cũ), SR đổi 3.66x compute lấy 3 track (trong biên nhiễu — đổi
chác tệ). **Ở S1, 3.85x compute đổi lấy 26 track** (ngoài biên nhiễu) — tỷ lệ
đổi chác tốt hơn hẳn, dù vẫn tốn gần 4x compute so với ResBlock trơn.

## 4. Cấu hình tốt nhất hiện tại và mức độ tin cậy

**S1 (λ_SR=0.1)** là cấu hình tốt nhất đã đo: 79.78% (797/999), vượt mốc gốc
77.00% tới 2.78 điểm và là kết quả duy nhất vượt rõ biên nhiễu. Nhưng:

- **Chưa multi-seed** — 1 run duy nhất; dự án đã tự đo được `cudnn.benchmark=True`
  gây lệch tới 6.5 điểm giữa 2 lần chạy cùng seed, nên "vượt biên nhiễu track-level"
  chưa loại trừ hết rủi ro này.
- **Gộp 6 thay đổi trong 1 run** (MFSR, DCN, domain-match, STN pool, decode, EMA)
  — chỉ tách được phần decode (+2 track) khỏi phần kiến trúc/training (+24 track),
  chưa tách sâu hơn được DCN/domain-match/EMA đóng góp bao nhiêu mỗi phần.
- **S2 đã loại trừ được 1 giả thuyết**: tăng `λ_SR` không phải hướng cải thiện
  tiếp — 0.1 vẫn là lựa chọn tốt nhất trong 2 điểm đã đo.

Xem đầy đủ giới hạn cần nêu khi báo cáo tại
[s1_proposed_mf_sr_ocr.md §6](s1_proposed_mf_sr_ocr.md#6-giới-hạn-cần-nêu-khi-báo-cáo).

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
| Lệnh chạy GPU đầy đủ + bảng compute + hướng dẫn vẽ chart | [../training_runs/run_gpu.md](../training_runs/run_gpu.md) |
| Cấu trúc dataset, plate layout, corners annotation | [../summary_project/dataset/dataset_overview.md](../summary_project/dataset/dataset_overview.md) |

## 6. Bước tiếp theo (chưa chạy)

Theo đúng thứ tự đã đặt ra trong `run_gpu.md` §5, chưa bị thay đổi bởi kết quả S2:

1. **S3** — `--sr-perceptual-weight 0.1` (thêm L_Perceptual vào L_SR).
2. **S4** — `--sr-scale 1 --width-downsample 4` (kiểm tra giới hạn ~55% pixel nội
   suy do HR gốc chỉ ~115×42px).
3. **O1** — multi-seed (42/100/2026) trên cấu hình λ=0.1 (S1) — điều kiện bắt buộc
   trước khi khẳng định S1 là kết quả cuối để báo cáo/nộp submission.
4. Ablation tách 6 thành phần của S1 (đóng góp riêng của DCN/domain-match/EMA) —
   nếu còn thời gian, giá trị thông tin cao nhưng tốn nhiều run nhất.
