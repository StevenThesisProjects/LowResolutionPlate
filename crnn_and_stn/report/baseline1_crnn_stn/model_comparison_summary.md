# So sánh tổng hợp — TOÀN BỘ mô hình đã train (Baseline 1: CRNN + STN)

> File điều hướng: gom kết quả từ 9 tài liệu trong thư mục này +
> [../training_runs/run_gpu.md](../training_runs/run_gpu.md) thành 1 bảng duy nhất.
> Không lặp lại phân tích chi tiết — mỗi dòng trỏ tới đúng file để đọc sâu.
> Toàn bộ Val Acc đo trên cùng 1 tập **999 track Scenario-B** (`dataset/val_tracks.json`),
> trừ 2 dòng "report gốc" (đo trên tập của tác giả report, không phải tập này).
> Cập nhật: 2026-08-03 — **phạm vi multi-seed chốt cuối: J1 + S1 + S4**
> ✅ **đã xong cả 3** — J1 đạt điểm cao nhất (80.45% ± 0.45).

## 0. 🚨 Đọc trước: có 2 loại số trong file này, đừng trộn

| Loại | Cách chạy | Dùng để làm gì |
|---|---|---|
| **Multi-seed** (§1c) | 3 seed 42/100/2026, `--no-cudnn-benchmark` (deterministic) | ✅ **Số chính thức đưa vào paper** |
| **1 seed** (§1, §2) | 1 lần chạy, `cudnn.benchmark=True` | ⚠️ Chỉ **exploratory** — xếp thứ tự sơ bộ, chọn cấu hình đáng multi-seed |

Multi-seed cho thấy hiệu ứng **khác nhau tuỳ cấu hình** — không phải quy luật chung:

| Cùng seed 42, chỉ đổi chế độ cudnn | 1-seed (`benchmark=True`) | 3-seed (deterministic) | Chênh ở seed 42 |
|---|---:|---:|---:|
| **S1** | 797/999 (79.78%) | **79.95% ± 0.15** | **0 track — giống hệt** |
| **S4** | 805/999 (80.58%) | **79.48% ± 0.44** | **−16 track / −1.60 điểm** |

Với **S4**, 16 track lệch **vượt biên nhiễu ±13** — riêng tính không xác định của
cudnn đã đủ tạo chênh lệch lớn hơn ngưỡng dùng để phán xét cải thiện, và 80.58% hoá
ra là một lần chạy may mắn (cao hơn cả 3 seed). Với **S1** thì ngược lại: cùng seed
42 ra đúng cùng con số ở cả 2 chế độ.

**Hệ quả**: số 1-seed của J1/J2/J3/S2/S3 **có thể** bị thổi phồng, nhưng cũng có thể
không — chưa multi-seed thì chưa biết. Bảng §1 vẫn đầy đủ để nhìn toàn cảnh, nhưng
thứ hạng trong đó **không phải kết luận**. Chi tiết đầy đủ:
[multi_seed_results.md](multi_seed_results.md).

---

## 1. Bảng tổng hợp — TOÀN BỘ 19 lần train, xếp theo Val Acc

Mọi cấu hình đã từng train trong project, không bỏ dòng nào. Cột **Loại** cho biết dòng
đó thuộc nhóm nào; cột **Seeds** cho biết con số đáng tin tới đâu.

| # | Cấu hình | Loại | Track đúng | Val Acc | Seeds | Tài liệu |
|---|---|---|---:|---:|:---:|---|
| 1 | **S3 — S1 + L_Perceptual (α=0.1)** | Đề xuất (S) | **805** | **80.58%** | 1 ⚠️ | [s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md) |
| 2 | **S4 — SR scale=1, T=32 qua `width_downsample=4`** | Đề xuất (S) | **805** | **80.58%** | 1 ⚠️ | [s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md) |
| 3 | **S1 — Joint MF-SR-OCR, λ_SR=0.1 (đề xuất chính)** | Đề xuất (S) | **797** | **79.78%** | 1 ⚠️ | [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) |
| 4 | S2 — S1 + λ_SR=0.5 | Đề xuất (S) | 794 | 79.48% | 1 ⚠️ | [s2_lam05_mf_sr_ocr.md](s2_lam05_mf_sr_ocr.md) |
| — | 🥇 **J1 — 3 seed deterministic (KHÔNG SR)** | **CHÍNH THỨC** | 799/808/804 | **80.45% ± 0.45** | **3 ✅** | [multi_seed_results.md](multi_seed_results.md) |
| — | 🥈 **S1 — 3 seed deterministic** | **CHÍNH THỨC** | 797/800/799 | **79.95% ± 0.15** | **3 ✅** | [multi_seed_results.md](multi_seed_results.md) |
| — | 🥉 **S4 — 3 seed deterministic** | **CHÍNH THỨC** | 789/797/796 | **79.48% ± 0.44** | **3 ✅** | [multi_seed_results.md](multi_seed_results.md) |
| 5 | ResNet + Transformer + STN (report ICPR gốc)\* | Mốc tham chiếu | — | 78.70% | — | [baseline1_architecture.md](baseline1_architecture.md) |
| 6 | J2 — ResBlock + GroupNorm + SR per-frame (kèm `edge=0.5`) | Ablation (J) | 771 ✅ | 77.18% | 1 ⚠️ | [groupnorm_sr_ablation_j1_j2.md §3b](groupnorm_sr_ablation_j1_j2.md) |
| 7 | **CRNN + STN (report ICPR gốc)** | **Mốc chuẩn** | — | **77.00%** | — | [baseline1_architecture.md](baseline1_architecture.md) |
| 8 | J1 — ResBlock + GroupNorm, không SR | Ablation (J) | 768 ✅ | 76.88% | 1 ⚠️ | [groupnorm_sr_ablation_j1_j2.md §3b](groupnorm_sr_ablation_j1_j2.md) |
| 9 | ResBlock backbone (`norm=none`, PR #8) | Backbone | 766 | 76.68% | 1 ⚠️ | [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) |
| 10 | J3 — J2 + DCNv2 (kernel init ngẫu nhiên — bug) | Ablation (J) | 762 | 76.28% | 1 ⚠️ | [s1_proposed_mf_sr_ocr.md §2](s1_proposed_mf_sr_ocr.md#2-kết-quả-chính) |
| 11 | D — AdamW: chỉ rút ngắn warmup (tốt nhất nhóm) | Optimizer† | — | 76.28% | 1 ⚠️ | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 12 | C — AdamW: chỉ tăng weight_decay | Optimizer† | — | 76.18% | 1 ⚠️ | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 13 | F — AdamW: wd + warmup mức vừa | Optimizer† | — | 76.08% | 1 ⚠️ | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 14 | A — AdamW: sửa bug param-grouping | Optimizer† | — | 75.98% | 1 ⚠️ | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 15 | CRNN + STN (đo lại trên dataset project) | Baseline | 757 | 75.78% | 1 ⚠️ | [baseline1_architecture.md](baseline1_architecture.md) |
| 16 | B — AdamW: wd cao + warmup ngắn (2 biến cùng lúc) | Optimizer† | — | 75.58% | 1 ⚠️ | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 17 | E — AdamW: D + train dài hơn (60 epoch) | Optimizer† | — | 75.08% | 1 ⚠️ | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| 18 | aug light (augmentation nhẹ) | Baseline | 739 | 73.97% | 1 ⚠️ | chỉ có trong `run_gpu.md` |
| 19 | SR-v2 — stacked-input SR, lr thấp + aug light | SR hỏng | 550 | 55.06% | 1 ⚠️ | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |
| 20 | SR-v1 — stacked-input SR (bản lỗi, gộp 5 frame→1) | SR hỏng | 492 | 49.25% | 1 ⚠️ | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |

\* Kiến trúc khác hẳn (ResNet + Transformer), **không so trực tiếp được** với nhánh CRNN
đang làm — để ở đây chỉ như mốc "cao nhất report gốc từng đạt".
✅ = đã **chấm lại trực tiếp** từ `submission_*.txt` + nhãn gốc (2026-08-03), không
chỉ chép số cũ. J1/J2 còn được xác minh kiến trúc từ `state_dict`.
† Nhóm Optimizer (A–F) chạy trên **backbone CNN+BatchNorm cũ, nay không còn trong code**
— xem §5, không so trực tiếp được với các dòng còn lại.

## 1b. Biểu đồ so sánh trực quan

Thang: 1 ký tự ≈ 0.2 điểm, gốc 73.5%. `█` = số multi-seed (chính thức) ·
`▒` = số 1-seed (exploratory) · `░` = mốc tham chiếu từ report gốc.

```
                                          73.5%                                81.0%
                                            ├────┬────┬────┬────┬────┬────┬────┤
 S3  +L_Perceptual              80.58% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 S4  SR ×1                      80.58% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
▶J1  3 SEED — KHÔNG SR 🥇       80.45%±0.45 ███████████████████████████████████[██]
▶S1  3 SEED DETERMINISTIC       79.95%±0.15 ████████████████████████████████[▪]
 S1  Joint MF-SR-OCR λ=0.1      79.78% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 S2  λ_SR=0.5                   79.48% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
▶S4  3 SEED DETERMINISTIC       79.48%±0.44 ████████████████████████████[███]
 R2  ResNet+Transf. (report)    78.70%  ref ░░░░░░░░░░░░░░░░░░░░░░░░░░
 J2  +SR per-frame              77.18% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
▶R1  CRNN+STN (report, CHUẨN)   77.00%  ref ░░░░░░░░░░░░░░░░░░
 J1  +GroupNorm, không SR       76.88% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 —   ResBlock backbone          76.68% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 J3  +DCNv2 (bug init)          76.28% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 D   AdamW warmup ngắn †        76.28% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 C   AdamW +weight_decay †      76.18% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒
 F   AdamW wd+warmup vừa †      76.08% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒▒
 A   AdamW sửa param-group †    75.98% (1s) ▒▒▒▒▒▒▒▒▒▒▒▒
 —   CRNN+STN (đo lại)          75.78% (1s) ▒▒▒▒▒▒▒▒▒▒▒
 B   AdamW wd cao+warmup ngắn † 75.58% (1s) ▒▒▒▒▒▒▒▒▒▒
 E   AdamW D + 60 epoch †       75.08% (1s) ▒▒▒▒▒▒▒▒
 —   aug light                  73.97% (1s) ▒▒
                                            ├────┬────┬────┬────┬────┬────┬────┤
                                          73.5   75   76   77   78   79   80  81.0
```

**Ngoài thang** (thất bại nặng, không vẽ được cùng khung):
`SR-v2 = 55.06%` · `SR-v1 = 49.25%` — hướng stacked-input gộp 5 frame → 1, đã bỏ.

**Cách đọc biểu đồ này:** khoảng cách **±13 track = ±1.3 điểm ≈ 6.5 ký tự** trên thang
trên. Hai thanh chênh nhau **dưới 6.5 ký tự là không phân biệt được**. Nhìn theo cách
đó, cả cụm từ `aug light` tới `J2` (18 dòng dưới) gần như là **một khối duy nhất**, và
chỉ cụm S mới thật sự tách khỏi mốc chuẩn 77.00%.

Hai thanh `█` (S1 và S4 multi-seed) chênh nhau **2.5 ký tự** — đúng như kiểm định
thống kê ở §1c cho biết: **không phân biệt được**. Đáng chú ý là 2 thanh `▒` cao nhất
(S3, S4 ở 80.58%) **đều cao hơn cả 2 thanh `█`** — minh hoạ trực quan cho việc số
1-seed có xu hướng lạc quan, và là lý do không được xếp hạng bằng bảng §1.

Chart vector chuẩn cho paper: `tools/plot_results.py ablation`
(lệnh ở [run_gpu.md §6](../training_runs/run_gpu.md#6-chart-cho-báo-cáo)).

## 1c. Kết quả multi-seed — số chính thức cho paper

**✅ BƯỚC 1 HOÀN THÀNH — đủ 3 model có Mean ± Std** (42/100/2026, deterministic).
📄 Phân tích đầy đủ 9 run: **[multi_seed_results.md](multi_seed_results.md)**.

| Model | SR | `T` | GFLOPs | **Mean ± Std** | Hạng |
|---|---|---:|---:|---:|:---:|
| **J1** — GroupNorm, **không SR** | ❌ | 16 | **26.14** | **80.45% ± 0.45** | 🥇 |
| **S1** — Joint MF-SR-OCR ×2 (đề xuất) | ×2 MFSR+DCN | 32 | 109.08 | **79.95% ± 0.15** | 🥈 |
| **S4** — SR ×1, T=32 qua backbone | ×1 MFSR+DCN | 32 | chưa đo | **79.48% ± 0.44** | 🥉 |
| ~~S2~~ · ~~S3~~ · ~~J2~~ · ~~J3~~ | — | — | — | 1 seed | phụ lục |

### 🚨 Kết quả chính — cấu hình KHÔNG SR đạt điểm cao nhất

| Cặp | Chênh | Sai số hiệu | Kết luận |
|---|---:|---:|---|
| **J1 vs S1** | +0.50 | ±0.28 | ⚠️ trong nhiễu → **hoà** |
| **J1 vs S4** | **+0.97** | ±0.36 | ✅ **vượt 2× sai số → J1 tốt hơn thật** |
| S1 vs S4 | +0.47 | ±0.27 | ⚠️ trong nhiễu → **hoà** |

**S1 tốn 3.76× compute so với J1 để đổi lấy −0.50 điểm.**

**Hai giả thuyết trung tâm bị bác bỏ:**

1. **"SR đóng góp vào độ chính xác"** — ❌. J1 và S1 dùng chung toàn bộ cụm cờ nền
   (GroupNorm + STN pool `(4,8)` + domain-match + constrained decode + EMA); S1 chỉ
   thêm SR/DCN/MFSR. Thêm cụm đó **không cải thiện**. Phần +3.57 điểm mà J1-mới hơn
   J1-**lịch sử** (76.88% → 80.45%) đến từ **cụm cờ nền**, không phải SR.
2. **"`T=32` là yếu tố chính"** — ❌. J1 chạy **`T=16`** mà vẫn ngang/hơn S1 và S4
   (đều `T=32`).

> ⚠️ **J1 multi-seed KHÔNG dùng cờ lịch sử.** Run thật dùng đúng bộ cờ nền của
> S1/S4 (STN pool `(4,8)`, domain-match, constrained, EMA), chỉ bỏ SR/DCN — tức là
> **ablation 1-cụm-biến sạch so với S1**, tốt hơn về mặt khoa học. Nhưng vì thế
> **không đặt chung cột với 76.88%** của J1 lịch sử (STN pool `(1,1)`, params
> 29,313,452 vs 29,442,700). Chi tiết:
> [groupnorm_sr_ablation_j1_j2.md §3c](groupnorm_sr_ablation_j1_j2.md).

### ✅ Kết luận chính thức đầu tiên có error bar: S1 ≈ S4

```
Chênh lệch   : +0.47 điểm (S1 79.95% so với S4 79.48%)
Sai số hiệu  : ±0.27
⚠️ Chênh lệch NẰM TRONG biên độ nhiễu → chưa đủ bằng chứng kết luận.
```

**S1 và S4 không khác biệt có ý nghĩa thống kê.** Claim *"S4 bằng S1 nhưng rẻ hơn
2.3×"* — trước đây là so sánh khập khiễng (1 bên multi-seed, 1 bên 1-seed) — **nay
đã được xác nhận đúng cách**. Cả hai đều vượt rõ biên nhiễu so với J2 1-seed
(+2.77 và +2.30 điểm) và baseline gốc 77.00% (+2.95 và +2.48 điểm).

### 🚨 Phát hiện mới: `cudnn.benchmark` KHÔNG luôn thổi phồng số

| Cùng seed 42, chỉ đổi chế độ cudnn | 1-seed cũ (`benchmark=True`) | multi-seed (deterministic) | Chênh |
|---|---:|---:|---:|
| **S1** | 797/999 (79.78%) | 797/999 (79.78%) | **0 track — giống hệt** |
| **S4** | 805/999 (80.58%) | 789/999 (78.98%) | **−16 track (−1.60 điểm)** |

Mức nhạy cảm với tính không xác định của cudnn **phụ thuộc cấu hình cụ thể**, không
phải quy luật chung của project. Con số headline S1 hoá ra là con số **ổn định nhất**
đã đo — không đổi dù bật hay tắt `benchmark`. Chi tiết + các con số ổn định dự đoán
(22% track nhạy seed, 18% track khác nhau giữa S1/S4): [multi_seed_results.md](multi_seed_results.md).

### ⚠️ J1 là run MỚI, số sẽ KHÁC 76.88%

J1/J2 lịch sử chạy khi `STN_POOL` mặc định còn là `(1,1)`; code hiện tại mặc định
`(4,8)` ([`configs/config.py:68`](../../configs/config.py)), và J1/J2 cũ **không** có
`--lr-domain-match`, `--decode constrained`, `--use-ema`.

Bản chạy lại (**J1**) dùng **đúng bộ cờ nền của S1/S4**, chỉ khác đúng biến
SR đang đo — nhờ vậy J1 → S1 → S4 mới là một ablation đọc được:

| | SR | DCN | GroupNorm | domain-match | constrained | EMA | T |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **J1** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 16 |
| **S1** | multi-frame ×2 | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |
| **S4** | multi-frame ×1 | ✅ | ✅ | ✅ | ✅ | ✅ | 32 |

→ Trong paper, **đừng ghép J1 vào cùng cột với 76.88%**. Hai bộ số này
đến từ hai kiến trúc khác nhau. Số cũ giữ lại ở §2 như lịch sử; số mới vào bảng chính.

Lệnh chạy: [run_gpu.md §0 B4](../training_runs/run_gpu.md#-b4--multi-seed-j1-j2-s1-ladder-sạch).

---

## 2. Bảng theo trình tự thời gian (1 seed — exploratory, giữ để tra lịch sử)

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

---

## 3. Đánh giá so sánh giữa các model

**Xếp hạng CHÍNH THỨC** (chỉ dòng có error bar — 999 track, decode constrained):

1. 🥇 **S1 = S4** — **79.95% ± 0.15** và **79.48% ± 0.44**, chênh +0.47 điểm
   **nằm trong biên nhiễu** (sai số hiệu ±0.27) → không phân biệt được về thống kê.
   Tách nhau bằng chi phí: **S4 rẻ hơn 2.30× khi train**.

**Xếp hạng tham khảo** (1 seed, `benchmark=True` — **không phải kết luận**):

2. S3 = S4 (80.58%, 805 track) — điểm 1-seed cao nhất từng đo; riêng S4 đã được
   multi-seed chứng minh là lạc quan (mức thật 79.48%). S3 sẽ không được multi-seed.
3. S1 (79.78%, 797 track) — số 1-seed **trùng khớp** với seed 42 của multi-seed.
4. S2 (79.48%, 794 track) — bằng S1 trong biên nhiễu, không có lý do chọn.
5. J2 (77.18%, 771 track) — mốc cũ trước khi có S-series.
6. Baseline report gốc (77.00%) — mốc tham chiếu chuẩn của toàn bộ project.

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
  "SR" đo được ở S1. ⚠️ **Vẫn chưa tách được dứt điểm**: cần thêm run
  `--width-downsample 4` **không** `--use-sr` (T=32, không SR) — J1 **không** thay
  thế được (J1 dùng T=16). Ghi vào Limitations.
- **Không có cấu hình nào miễn phí** — S1/S2/S3 tốn ~3.85x compute so với ResBlock
  trơn (§6); **S4 rẻ hơn hẳn khi train** (2.34×/epoch, đã đo) nhưng chưa benchmark
  latency inference. S3 tốn thêm vì phải forward qua VGG16 mỗi batch, đổi lại hội
  tụ nhanh nhất (đỉnh epoch 25).
- **Nếu phải chọn 1 cấu hình để nộp submission ngay bây giờ**: **S4 nếu ưu tiên chi
  phí, S1 nếu ưu tiên độ ổn định.** Nay đã có error bar cho cả hai nên lựa chọn này
  dựa trên số thật, không còn là phỏng đoán:
  - **S4** — rẻ hơn **2.30×** khi train (4.09 vs 9.39 phút/epoch), độ chính xác
    tương đương về mặt thống kê.
  - **S1** — mean cao hơn 0.47 điểm (không đáng kể) nhưng **std nhỏ hơn gần 3 lần**
    (0.15 vs 0.44), tức ít rủi ro "gặp seed xấu" hơn; và là cấu hình duy nhất cho
    **cùng một kết quả ở cả 2 chế độ cudnn**.

  Chi tiết đánh đổi: [multi_seed_results.md §8](multi_seed_results.md#8-kết-luận--cập-nhật-cho-paper).

## 3b. Metrics mức ký tự — CER / NED (review Bước 2)

Tính lại từ `submission_*.txt` + nhãn gốc, **không cần train lại**. CER là mức corpus
(tổng edit distance / tổng ký tự nhãn), NED là trung bình mỗi track
(`ED / max(len_ref, len_hyp)`). Cả hai **thấp = tốt**; paper hay ghi `1−NED`.
Cả 4 dòng đều là **1 seed**.

| Cấu hình | Exact Match | CER ↓ | Track sai độ dài | Nguồn |
|---|---:|---:|---:|---|
| **S1 — 3 seed ✅** | **79.95% ± 0.15** | **0.0541 ± 0.0016** | 0–1 | multi-seed |
| **S4 — 3 seed ✅** | **79.48% ± 0.44** | **0.0543 ± 0.0001** | 0–1 | multi-seed |
| S1 (λ=0.1, 1 seed) | 79.78% | 0.0562 | 2 | 1 seed |
| S2 (λ=0.5, 1 seed) | 79.48% | 0.0549 | 1 | 1 seed |
| S3 (+perceptual, 1 seed) | 80.58% | 0.0522 | 0 | 1 seed |
| S4 (SR ×1, 1 seed) | 80.58% | 0.0532 | 1 | 1 seed |
| **J2 (lịch sử, 1 seed)** | 77.18% | **0.0645** | **18** ⚠️ | ✅ chấm lại 2026-08-03 |
| **J1 (lịch sử, 1 seed)** | 76.88% | **0.0608** | **0** | ✅ chấm lại 2026-08-03 |

> 🆕 **J1/J2 đã được chấm lại từ checkpoint + submission còn lưu trên đĩa**
> (`results/crnn_resblock_groupnorm_nosr_j1/`, `results/crnn_resblock_sr_supervised_j2/`)
> — exact match khớp chính xác 76.88% / 77.18% đã báo cáo. **Phát hiện mới: J2 có
> CER TỆ HƠN J1** (0.0645 vs 0.0608) và **18/999 track sai độ dài** (J1: 0) dù exact
> match nhỉnh hơn 3 track. Tức "SR giúp ở J1→J2" còn mong manh hơn những gì biên
> nhiễu ±13 đã cảnh báo. Chi tiết:
> [groupnorm_sr_ablation_j1_j2.md §3b](groupnorm_sr_ablation_j1_j2.md).

**Multi-seed cũng phá luôn thế hoà CER**: S1 (0.0541 ± 0.0016) và S4
(0.0543 ± 0.0001) gần như bằng nhau — chênh 0.0002, nhỏ hơn cả std của S1. Nhưng
**S4 ổn định hơn hẳn về CER** (std 0.0001 vs 0.0016, nhỏ hơn 16 lần) dù kém ổn định
hơn về exact match (std 0.44 vs 0.15) — hai đại lượng **không đi cùng chiều**, một
quan sát chỉ multi-seed mới thấy được.

Hai điều exact-match không cho thấy được:

1. **S2 kém S1 về exact match nhưng TỐT hơn về CER** (0.0549 vs 0.0562) — khi S2 sai,
   nó sai "gần đúng" hơn. Exact match phạt sai 1 ký tự như sai cả biển nên giấu mất
   điều này.
2. **CER phá được thế hoà S3 ↔ S4.** Hai cấu hình bằng nhau tuyệt đối ở exact match
   (805/999) nhưng S3 có CER tốt hơn (0.0522 vs 0.0532) — tiêu chí đầu tiên tách được
   hai ứng viên tốt nhất. Chênh lệch nhỏ, và **S3 sẽ không có multi-seed** nên đây
   vẫn là quan sát 1 seed, không phải kết luận.

CER và NED trùng nhau vì gần như mọi dự đoán đều đúng 7 ký tự, nên
`max(len_ref, len_hyp)` luôn bằng `len_ref`.

> J1 sẽ có `val_cer` / `val_ned` **tự động trong `history_*.csv`** (2 cột thêm từ
> 2026-08-01) → sau multi-seed, bảng này mở rộng được thành 6 dòng có CER kèm error bar.

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
chính nó. **J1 không có dòng ở bảng này** — không có nhánh SR nên không đo được.

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

Đây cũng là **bằng chứng độc lập thay thế cho S2 multi-seed**: kết luận "tăng λ_SR
không giúp" đứng vững nhờ tương quan âm trên n=999 ở 4 cấu hình, không phụ thuộc
thứ hạng 1-seed của S2.

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

---

## 4. Ma trận cấu hình — cái gì bật ở model nào

Tra nhanh khi cần biết một cấu hình khác cấu hình kia đúng chỗ nào.

| | GroupNorm | SR | DCNv2 | MFSR | **edge** | STN pool | domain-match | decode | EMA | T |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CRNN+STN (đo lại) | ❌ (BN) | ❌ | ❌ | ❌ | — | (1,1) | ❌ | greedy | ❌ | 16 |
| ResBlock `norm=none` | ❌ | ❌ | ❌ | ❌ | — | (1,1) | ❌ | greedy | ❌ | 16 |
| J1 (lịch sử) ✅ verify | ✅ | ❌ | ❌ | ❌ | — | (1,1) | ❌ | greedy | ❌ | 16 |
| J2 (lịch sử) ✅ verify | ✅ | ×2 | ❌ | ❌ | **0.5** ⚠️ | (1,1) | ❌ | greedy | ❌ | 32 |
| J3 (lịch sử) | ✅ | ×2 | ✅ bug init | ❌ | ? | (1,1) | ❌ | greedy | ❌ | 32 |
| **J1** (sẽ multi-seed) | ✅ | ❌ | ❌ | ❌ | — | (1,1) | ❌ | greedy | ❌ | 16 |
| **S1** | ✅ | ×2 | ✅ | ✅ | **0.0** | (4,8) | ✅ | constrained | ✅ | 32 |
| S2 | ✅ | ×2 | ✅ | ✅ | 0.0 | (4,8) | ✅ | constrained | ✅ | 32 |
| S3 | ✅ | ×2 +VGG | ✅ | ✅ | 0.0 | (4,8) | ✅ | constrained | ✅ | 32 |
| **S4** | ✅ | **×1** | ✅ | ✅ | **0.0** | (4,8) | ✅ | constrained | ✅ | 32\* |

\* S4 đạt T=32 qua `--width-downsample 4` (đổi backbone), không qua phóng to ảnh.
⚠️ **Cột `edge` (Sobel edge loss) mới thêm 2026-08-03**: J2 chạy `edge=0.5` — chi
tiết này **thiếu trong mọi tài liệu trước đó**, phát hiện khi đọc banner `log_j2.txt`.
Nghĩa là J2 khác S1 ở **7 biến**, không phải 6. J1/J2 đã được xác minh trực tiếp từ
`state_dict` (params 29,313,452 / 29,426,895 — khớp bảng compute STN pool `(1,1)`).

📌 **J1 multi-seed dùng ĐÚNG cấu hình lịch sử** — đó là lý do nó giữ
`(1,1)` / `greedy` / không EMA. Nghĩa là ladder J1→J2→S1→S4 **không phải ablation
1-biến**: mỗi bậc đổi nhiều thứ cùng lúc (ablation *tích luỹ*, không phải *tách biến*).
Cần ghi rõ cách đọc này trong paper để reviewer không hiểu nhầm.

Nhìn bảng này thấy rõ **vì sao J1/J2 lịch sử không so 1-biến được với S1**: chúng lệch
S1 ở 5–7 cột cùng lúc — đây là **ablation tích luỹ**, không phải tách 1 biến.
Cách đọc đúng: *"đi từ cấu hình J1 sang S1, gộp tất cả thay đổi, được +X track"*,
chứ không phải *"SR đóng góp +X track"*.

---

## 5. Nhánh phụ — AdamW tuning trên backbone cũ (lịch sử, không so trực tiếp được)

Chạy trên backbone CNN+BatchNorm gốc, nay không còn tồn tại trong code (đã bị
thay bởi ResBlock ở §2 dòng #5):

| Cấu hình | Val Acc | Δ vs baseline (75.78%) |
|---|---:|---:|
| Baseline (bug: weight-decay áp sai lên bias/BN) | 75.78% | — |
| A — sửa bug param-grouping | 75.98% | +0.20 |
| B — wd cao + warmup ngắn (2 biến cùng lúc) | 75.58% | −0.20 |
| F — wd + warmup mức vừa | 76.08% | +0.30 |
| C — chỉ tăng weight_decay | 76.18% | +0.40 |
| D — chỉ rút ngắn warmup (tốt nhất) | 76.28% | +0.50 |
| E — D + train dài hơn (60 epoch) | 75.08% | −0.70 |

Toàn bộ A-F nằm trong biên nhiễu. Trần 76.28% thấp hơn ResBlock (76.68%) đạt
được mà không cần tuning — kết luận: giới hạn nằm ở backbone, không phải
optimizer. Chi tiết: [optimizer_adamw_verification.md](optimizer_adamw_verification.md).

## 6. Chi phí compute (kiến trúc hiện tại, STN pool `(4,8)`)

Đo bằng `tools/benchmark.py --all` (CPU). Nguồn: [run_gpu.md §2](../training_runs/run_gpu.md).

| Cấu hình | Params | GFLOPs/track | Latency (ms) | vs base |
|---|---:|---:|---:|---:|
| ResBlock (không SR) ≈ **J1** | 29,427,468 | 26.14 | 50.03 | 1.00x |
| + GroupNorm (J1) | 29,442,700 | 26.14 | 51.30 | 1.03x |
| + SR per-frame (J2 kiến trúc cũ) | 29,558,255 | 108.31 | 184.43 | 3.69x |
| + SR + DCNv2 = **S1 / S2 / S3** (cùng kiến trúc, chỉ khác loss) | 29,577,214 | 109.08 | 192.75 | 3.85x |
| S4 (SR scale=1, `width_downsample=4`) | 29,549,470 | chưa đo | chưa đo | **có thể < 3.85x** |

Bảng trên **chưa tính thêm chi phí VGG16 của S3** (forward-only, không train,
nhưng vẫn tốn thêm compute mỗi batch) — chưa benchmark riêng. Ở J2 (kiến trúc
cũ), SR đổi 3.66x compute lấy 3 track (trong biên nhiễu — đổi chác tệ). **Ở S1,
3.85x compute đổi lấy 26 track** (ngoài biên nhiễu) — tỷ lệ đổi chác tốt hơn hẳn.
**J1 multi-seed sẽ đo lại đại lượng này có error bar**: 3.85x compute của
S1 đổi được bao nhiêu track so với mốc không SR, tính cả phương sai giữa seed.

**S4 chưa được `tools/benchmark.py` đo**, nhưng đã có số **train xác nhận qua
multi-seed** (trung bình 3 seed, chế độ deterministic):

| | Phút/epoch (3 seed det.) | 1-seed cũ (`benchmark=True`) |
|---|---:|---:|
| S1 | 9.39 | 9.21 |
| S4 | 4.09 | 3.94 |
| **Tỷ lệ S1/S4** | **2.30×** | 2.34× |

Chế độ deterministic chỉ chậm hơn 2–4% (không phải +25% như ước lượng ban đầu), và
**tỷ lệ 2.3× giữ nguyên ở cả 2 chế độ** — claim "S4 rẻ hơn" nay vững cả 2 vế
(accuracy tương đương + tốc độ nhanh hơn, cả hai từ multi-seed). Latency inference
vẫn cần benchmark thật ([multi_seed_results.md §7](multi_seed_results.md#7-compute--xác-nhận-lại-tỷ-lệ-23-dưới-chế-độ-deterministic)).

## 7. Bản đồ tài liệu — đọc gì khi cần gì

| Muốn biết... | Đọc file |
|---|---|
| **Số chính thức Mean ± Std, so sánh S1 vs S4 có kiểm định** | **[multi_seed_results.md](multi_seed_results.md)** |
| Kiến trúc gốc, đối chiếu với report ICPR | [baseline1_architecture.md](baseline1_architecture.md) |
| SR thất bại lúc đầu vì sao (stacked-input bug) | [super_resolution_experiments.md](../summary_project/document/super_resolution_experiments.md) |
| AdamW/weight-decay ablation (lịch sử, backbone cũ) | [optimizer_adamw_verification.md](optimizer_adamw_verification.md) |
| ResBlock backbone thay BatchNorm bằng gì, vì sao | [resblock_backbone_upgrade.md](resblock_backbone_upgrade.md) |
| GroupNorm giúp gì, SR per-frame có giám sát lần đầu, **và vì sao J1 phải chạy lại** | [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md) |
| Cấu hình đề xuất đầy đủ (MFSR+DCN+domain-match+decode+EMA) | [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) |
| λ_SR=0.5 có tốt hơn 0.1 không | [s2_lam05_mf_sr_ocr.md](s2_lam05_mf_sr_ocr.md) |
| L_Perceptual có giúp gì không | [s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md) |
| SR ×1 (không phóng to ảnh) có bằng SR ×2 không, "T confound" là gì, multi-seed đầu tiên | [s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md) |
| CER/NED/PSNR/SSIM + bằng chứng PSNR nghịch OCR | [../buoc2_metrics.md](../buoc2_metrics.md) |
| Tiến độ theo review, phạm vi multi-seed và lý do | [../checklist_review.md](../checklist_review.md) |
| Lệnh chạy GPU đầy đủ + bảng compute + hướng dẫn vẽ chart | [../training_runs/run_gpu.md](../training_runs/run_gpu.md) |
| Cấu trúc dataset, plate layout, corners annotation | [../summary_project/dataset/dataset_overview.md](../summary_project/dataset/dataset_overview.md) |

## 8. Trạng thái & việc còn lại

✅ **Bước 1 (multi-seed) đã HOÀN THÀNH** — đủ 3 model J1/S1/S4 có Mean ± Std trên
3 seed deterministic. Không còn run GPU nào trong phạm vi đã chốt.

Việc còn lại (**không thuộc phạm vi 3 model**, chỉ liệt kê để tra cứu):

1. **Benchmark compute cho S4 và J1** — `tools/benchmark.py --all` chưa có dòng cho
   `sr-scale=1`, và phút/epoch của J1 mới là ước tính từ tỷ lệ GFLOPs (log seed 42
   bị thiếu). Cần số thật trước khi in bảng chi phí.
2. **Ablation tách cụm cờ nền** — J1 vs S1 cho biết SR không giúp, nhưng chưa tách
   được đóng góp riêng của STN pool `(4,8)` / domain-match / constrained decode /
   EMA. Đây là câu hỏi mở **quan trọng nhất** còn lại về mặt khoa học.
3. **Ablation "T=32 không SR"** (`--width-downsample 4` không `--use-sr`) — nay ít
   cấp thiết hơn vì J1 (`T=16`) đã cho thấy `T=32` không phải yếu tố quyết định.
4. **Chạy test** — mọi số hiện tại đều là validation, chưa chạy test lần nào.
