# Bước 2 — Metrics CER, NED, PSNR/SSIM

> Review yêu cầu: bổ sung CER + NED (mức ký tự) và PSNR + SSIM (chất lượng ảnh SR),
> xuất ra console và log validation.
> Trạng thái: **đã xong**, có số cho cả S1–S4. Ba chỗ lệch so với nguyên văn review
> nêu ở mục 4 — cần ghi vào paper.
>
> ⚠️ **Toàn bộ số dưới đây là 1 seed** (`benchmark=True`). Phạm vi multi-seed đã
> cuối cùng (2026-08-03) là **3 model J1 + S1 + S4**
> ([../checklist_review.md](checklist_review.md)) — S2 và S3 sẽ **không** có
> CER/NED/PSNR multi-seed; đọc số của 2 cấu hình đó như tham khảo, không phải kết
> luận cuối. J1 sẽ tự động có `val_cer`/`val_ned` trong `history_*.csv` khi
> chạy xong.
>
> ✅ **CER multi-seed đã có cho S1 và S4** — thay thế 2 dòng tương ứng ở mục 1:
>
> | Cấu hình | Exact Match | CER ↓ |
> |---|---:|---:|
> | S1 — 3 seed | **79.95% ± 0.15** | **0.0541 ± 0.0016** |
> | S4 — 3 seed | **79.48% ± 0.44** | **0.0543 ± 0.0001** |
>
> Phát hiện chỉ multi-seed mới thấy: **S4 ổn định hơn 16× về CER** (std 0.0001 vs
> 0.0016) nhưng **kém ổn định hơn 3× về exact match** (std 0.44 vs 0.15) — hai đại
> lượng không đi cùng chiều. Chi tiết:
> [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).

---

## 1. CER và NED

**Là gì**: đo sai ở mức **ký tự**, khác exact match vốn phạt sai 1 ký tự như sai cả biển.

- `CER` = tổng edit distance / tổng ký tự nhãn (mức corpus)
- `NED` = trung bình `edit_distance / max(len_nhãn, len_dự_đoán)` mỗi track
- Cả hai **thấp = tốt**. Paper thường ghi `1−NED` (cao = tốt).

**Đã cài**: `src/training/trainer.py::validate()` → in ra console mỗi epoch và ghi 2 cột
`val_cer`, `val_ned` vào `history_*.csv`.

| Cấu hình | Exact Match | CER ↓ | NED ↓ | 1−NED ↑ |
|---|---:|---:|---:|---:|
| S1 (λ_SR=0.1) | 79.78% | 0.0562 | 0.0562 | 0.9438 |
| S2 (λ_SR=0.5) | 79.48% | 0.0549 | 0.0549 | 0.9451 |
| **S3 (+perceptual)** | **80.58%** | **0.0522** | **0.0522** | **0.9478** |
| S4 (SR ×1) | 80.58% | 0.0532 | 0.0532 | 0.9468 |
| **J2 (lịch sử)** 🆕 | 77.18% | **0.0645** | 0.0645 | 0.9355 |
| **J1 (lịch sử)** 🆕 | 76.88% | **0.0608** | 0.0608 | 0.9392 |

🆕 **Bổ sung 2026-08-03 — J1/J2 tính được mà không cần train lại**, vì `submission_*.txt`
của 2 run đó vẫn còn trên đĩa. Kết quả đáng chú ý: **J2 có CER TỆ hơn J1** (0.0645 vs
0.0608) dù exact match cao hơn 3 track, và J2 có **18/999 track sai độ dài** so với
**0/999** của J1 (J2 dùng greedy decode, `T` tăng 16→32 nên CTC dễ chèn ký tự thừa).
→ Đây là ví dụ thứ hai (sau S2) cho thấy **exact match giấu mất thông tin** — và lần
này CER chỉ **ngược chiều** với exact match. Chi tiết:
[baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md §3b](baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md).

**Hai điều exact match không cho thấy:**

1. **S2 thua S1 ở exact match nhưng thắng ở CER** (0.0549 vs 0.0562) — khi S2 sai, nó
   sai "gần đúng" hơn, đọc đúng nhiều ký tự hơn.
2. **CER phá được thế hoà S3 ↔ S4.** Hai cấu hình bằng nhau tuyệt đối ở exact match
   (805/999) nhưng S3 có CER tốt hơn — tiêu chí đầu tiên tách được hai ứng viên tốt nhất.

CER và NED trùng số vì gần như mọi dự đoán đều đúng 7 ký tự.

---

## 2. PSNR và SSIM

**Là gì**: đo ảnh SR tái tạo **giống ảnh HR gốc** đến mức nào. PSNR tính bằng dB
(cao = tốt), SSIM trong khoảng 0–1 (cao = tốt). Đây là **con số**, không phải hình vẽ.

**Đã cài**: `tools/eval_sr_quality.py` — chạy hậu kỳ trên checkpoint, dùng
`skimage.metrics`, không đụng vào training loop.

Luôn báo cáo kèm mốc **`base`** = ảnh chưa qua SR (nội suy với `sr_scale=2`, giữ
nguyên với `sr_scale=1`). **Con số đáng đọc là cột "Chênh"**, không phải PSNR tuyệt đối.

| Cấu hình | PSNR (SR) | PSNR (base) | **Chênh** | SSIM (SR) | SSIM (base) | **Chênh** |
|---|---:|---:|---:|---:|---:|---:|
| S1 (×2) | 16.6827 | 15.6110 | +1.0717 dB | 0.4179 | 0.3481 | +0.0698 |
| **S2** (×2, λ=0.5) | 18.5228 | 16.2349 | **+2.2879 dB** | 0.5461 | 0.3780 | **+0.1681** |
| S3 (×2, perceptual) | 16.9582 | 15.8727 | +1.0855 dB | 0.4234 | 0.3569 | +0.0664 |
| S4 (×1) | 17.5249 | 16.5011 | +1.0238 dB | 0.5027 | 0.4408 | +0.0618 |

999 track validation, mỗi track 5 frame (4.995 ảnh).

⚠️ **Con số dao động ~±0.05 dB giữa các lần chạy.** Pipeline degradation (blur/noise/JPEG)
là ngẫu nhiên; cờ `--seed` chỉ seed tiến trình chính, còn `--num-workers > 0` thì mỗi
worker của DataLoader có trạng thái ngẫu nhiên riêng mà albumentations không nhận seed
đó. Muốn tái lập tuyệt đối thì chạy `--num-workers 0` (chậm hơn). Biên dao động này
nhỏ hơn nhiều so với khoảng cách giữa các cấu hình nên không đổi kết luận nào.

⚠️ **Không so PSNR tuyệt đối giữa S4 và S1/S2/S3**: S1–S3 xuất ảnh 64×256, S4 xuất
32×128 — hai thang khác nhau. Chỉ so được cột "Chênh" vì mỗi cấu hình so với base của
chính nó.

---

## 3. Kết quả quan trọng nhất — PSNR/SSIM **NGHỊCH** với khả năng đọc biển số

### 3.1. Bằng chứng mức từng track (n = 999) — mạnh nhất

Ghép PSNR từng track (`sr_quality_*.csv`) với kết quả đúng/sai từng track
(`submission_*.txt`), tính hệ số tương quan điểm-nhị phân:

| Cấu hình | PSNR ở track **đọc đúng** | PSNR ở track **đọc sai** | Chênh | r(PSNR, đúng) |
|---|---:|---:|---:|---:|
| S1 | 16.288 | 18.240 | **−1.952 dB** | **−0.362** |
| S2 | 18.102 | 20.154 | **−2.052 dB** | **−0.346** |
| S3 | 16.492 | 18.894 | **−2.402 dB** | **−0.412** |
| S4 | 17.080 | 19.371 | **−2.291 dB** | **−0.400** |

**Track mà model đọc SAI lại có PSNR CAO hơn ~2 dB so với track đọc đúng.** Tương quan
âm rõ rệt (r ≈ −0.35 đến −0.41) và **nhất quán ở cả 4 cấu hình**.

Nghĩa là PSNR không chỉ *vô dụng* với OCR — nó **gây hiểu lầm**: tối ưu theo PSNR sẽ
đẩy model đi sai hướng.

**Giải thích khả dĩ (giả thuyết, chưa kiểm chứng riêng)**: PSNR bị chi phối bởi *nội
dung ảnh* hơn là chất lượng SR. Ảnh biển số mờ nhoè / tương phản thấp có sai khác pixel
nhỏ → **PSNR cao**, nhưng đúng là loại ảnh khó đọc nhất. Ngược lại ảnh sắc nét, nhiều
chi tiết tần số cao thì khó tái tạo → PSNR thấp → nhưng lại dễ đọc. Tức PSNR đang đo
"ảnh này dễ tái tạo tới đâu", còn OCR cần "ảnh này chứa bao nhiêu chi tiết đọc được" —
hai thứ nghịch nhau.

### 3.2. Bằng chứng mức cấu hình (4 điểm) — nhất quán với trên

| | Chất lượng ảnh | Đọc biển số |
|---|---|---|
| **S2** (λ_SR=0.5) | **Tốt nhất** (+2.29 dB, +0.168 SSIM) | **Kém nhất** (794/999) |
| **S3** (+perceptual) | +1.09 dB, +0.066 | **Tốt nhất** (805/999, CER 0.0522) |
| **S4** (SR ×1) | Thấp nhất (+1.02 dB, +0.062) | Đồng hạng nhất (805/999) |

Tăng trọng số SR (S2) **làm ảnh đẹp hơn thật** — hơn gấp đôi mọi cấu hình khác — nhưng
**không làm biển số dễ đọc hơn**, thậm chí còn kém đi. Nhất quán với tương quan âm ở
mục 3.1.

Đây là bằng chứng số cho nguyên tắc đã đặt ra từ đầu project trong
[super_resolution_experiments.md](summary_project/document/super_resolution_experiments.md):
*"đánh giá SR bằng OCR exact-match, không phải PSNR/SSIM; ưu tiên đọc được biển số hơn
đẹp ảnh"*. Trước đây đó mới là lập luận thiết kế; nay có bằng chứng ở **cả hai mức** —
4 cấu hình và 999 track.

**Dùng khi viết paper**: đây là câu trả lời cho câu hỏi "sao không tối ưu theo PSNR" —
không chỉ vì PSNR không giúp gì, mà vì **nó nghịch chiều với mục tiêu**. Đã thử tối ưu
PSNR (S2) và cho OCR tệ nhất.

---

## 4. Ba chỗ lệch so với nguyên văn review

**4.1. Không dùng thư viện `editdistance`.** `src/utils/postprocess.py` đã có sẵn hàm
`edit_distance` (Levenshtein có cache) từ trước — dùng lại, khỏi thêm dependency.
Kết quả tương đương.

**4.2. Không đo PSNR/SSIM trên Scenario-A — đây là chỗ nên phản hồi lại reviewer.**
Hai lý do:
- Validation **không có track Scenario-A nào** (0/999, toàn bộ là Scenario-B).
- Quan trọng hơn: **cả 10.000 track Scenario-A đều nằm trong tập TRAIN**. Đo PSNR ở đó
  là đo trên dữ liệu model đã học → con số bị thổi phồng, **không hợp lệ về phương pháp**.

Đo trên 999 track Scenario-B (đã tách riêng, chưa từng train) mới đúng.

**4.3. PSNR/SSIM không nằm trong log validation mỗi epoch**, mà là tool chạy hậu kỳ.
Lý do: nhét vào training loop thì validation phải sinh thêm ảnh HR (đổi hành vi val,
chậm mỗi epoch) và **multi-seed đang chạy sẽ phải khởi động lại**. Xu hướng theo epoch
vẫn theo dõi được qua 2 cột `sr_loss` vs `sr_loss_bilinear` có sẵn trong CSV.

---

## 5. Cách tạo lại số liệu

```bash
D=results/mf_sr_ocr

# PSNR/SSIM — S1/S2/S3 dùng mặc định; S4 BẮT BUỘC --width-downsample 4
python tools/eval_sr_quality.py --checkpoint $D/s1_mf_sr_ocr/mf_sr_ocr.pth \
  --lr-domain-match --output-csv $D/s1_mf_sr_ocr/sr_quality_s1.csv

python tools/eval_sr_quality.py --checkpoint $D/s4_sr_scale1/s4_sr_scale1_best.pth \
  --width-downsample 4 --lr-domain-match \
  --output-csv $D/s4_sr_scale1/sr_quality_s4.csv
```

⚠️ zsh không tách từ khi expand biến — đừng gom cờ vào biến rồi truyền, viết thẳng.

CER/NED thì tự động có trong `history_*.csv` của mọi run mới (2 cột `val_cer`,
`val_ned`). Với S1–S4 chạy trước ngày 2026-08-01 thì CSV chưa có 2 cột đó — số trong
mục 1 được tính lại từ `submission_*.txt` + nhãn gốc.

> ⚠️ `results/` bị **gitignore** → toàn bộ dữ liệu S1–S4 (CSV, log, submission, .pth)
> không được git theo dõi. Số liệu tổng hợp phải được chép vào các file `.md` trong
> `report/` — chỉ những file đó mới được commit.
