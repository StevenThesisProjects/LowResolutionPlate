# Bước 2 — Metrics CER, NED, PSNR/SSIM

> Review yêu cầu: bổ sung CER + NED (mức ký tự) và PSNR + SSIM (chất lượng ảnh SR),
> xuất ra console và log validation. Trạng thái: **đã xong**.
> Ba chỗ lệch so với nguyên văn review nêu ở mục 4 — cần ghi vào paper.
>
> **Phạm vi**: chỉ J1 · S1 · S4. Các ablation 1-seed khác nằm ở `backup/report/`.

## 0. Số chính thức — CER Mean ± Std trên 3 seed

| Cấu hình | Exact Match | CER ↓ | Track sai độ dài |
|---|---:|---:|---:|
| **J1** (không SR) | 80.45% ± 0.45 | **0.0525 ± 0.0007** 🥇 | **0/999 cả 3 seed** |
| **S1** 🎯 (đề xuất) | **79.95% ± 0.15** | 0.0541 ± 0.0016 | 1 (seed 2026) |
| **S4** (SR ×1) | 79.48% ± 0.44 | 0.0543 ± **0.0001** | 1 (seed 100) |

📌 **Phát hiện chỉ multi-seed mới thấy**: S4 **ổn định hơn 16× về CER**
(std 0.0001 vs 0.0016) nhưng **kém ổn định hơn 3× về exact match** (0.44 vs 0.15) —
hai đại lượng **không đi cùng chiều**. Đáng nêu trong paper.

Chi tiết: [baseline1_crnn_stn/multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).

---

## 1. CER và NED

**Là gì**: đo sai ở mức **ký tự**, khác exact match vốn phạt sai 1 ký tự như sai cả biển.

- `CER` = tổng edit distance / tổng ký tự nhãn (mức corpus)
- `NED` = trung bình `edit_distance / max(len_nhãn, len_dự_đoán)` mỗi track
- Cả hai **thấp = tốt**. Paper thường ghi `1−NED` (cao = tốt).

**Đã cài**: `src/training/trainer.py::validate()` → in ra console mỗi epoch và ghi 2 cột
`val_cer`, `val_ned` vào `history_*.csv`.

**Số 3 seed** (bảng đầy đủ ở mục 0):

| Cấu hình | Exact Match | CER ↓ | NED ↓ | 1−NED ↑ |
|---|---:|---:|---:|---:|
| **J1** (không SR) | 80.45% ± 0.45 | **0.0525** | 0.0525 | **0.9475** |
| **S1** 🎯 | 79.95% ± 0.15 | 0.0541 | 0.0541 | 0.9459 |
| **S4** (SR ×1) | 79.48% ± 0.44 | 0.0543 | 0.0543 | 0.9457 |

CER và NED **trùng số** vì gần như mọi dự đoán đều đúng 7 ký tự.

**Điều exact match không cho thấy**: J1 có **CER tốt nhất** (0.0525) *và* là cấu hình
duy nhất đạt **0/999 track sai độ dài ở cả 3 seed** — tức khi J1 sai, nó vẫn sai "đúng
độ dài", không chèn/thiếu ký tự. Đây là tiêu chí phụ tách được các cấu hình mà exact
match cho gần bằng nhau.

### 🔬 Vì sao CER thấp (0.054) mà exact match chỉ 79.95% — phân bố lỗi

Phân tích **202 track sai** của S1 (seed 42), đếm số ký tự sai trên tổng 7:

| Số ký tự sai (/7) | Số track | Tỷ lệ |
|---:|---:|---:|
| **1** | **114** | **56.4%** |
| 2 | 37 | 18.3% |
| 3 | 30 | 14.9% |
| 4 | 12 | 5.9% |
| 5–7 | 9 | 4.5% |

📌 **56.4% lỗi chỉ sai ĐÚNG 1 ký tự** — đây chính là lý do CER (mức ký tự) chỉ 0.054
trong khi exact match (phạt sai 1 ký tự như sai cả biển) tụt xuống 79.95%. Con số này
**giải thích trực tiếp khoảng cách giữa 2 chỉ số**, rất đáng đưa vào paper khi lập luận
*"vì sao cần báo cáo cả CER lẫn exact match"*.

🔬 **Cảnh báo hiệu chuẩn**: **103/202 track sai (51%) có confidence > 0.9** — model
**tự tin sai** ở một nửa số lỗi (conf trung bình của track sai = 0.868, cao nhất 0.9993).
→ **Không dùng confidence làm ngưỡng lọc lỗi được** — sẽ bỏ sót quá nửa.

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
| **S1** (×2) | 16.5882 ± 3.1938 | 15.8717 ± 2.7812 | +0.7166 dB | 0.3971 ± 0.1952 | 0.3536 ± 0.1787 | +0.0434 |
| **S4** (×1) | 17.4613 ± 3.1172 | 16.4241 ± 2.8200 | +1.0372 dB | 0.4955 ± 0.2060 | 0.4199 ± 0.1961 | +0.0756 |
| **J1** (không SR) | — | — | — | — | — | — |

999 track validation, mỗi track 5 frame (4.995 ảnh). ✅ **Đo trên checkpoint
multi-seed seed 42** (`results/multi-seed/`), cùng nguồn với bảng accuracy Mean ± Std
— đã chạy lại 2026-08-05, không còn dùng số của checkpoint 1-seed cũ.

> 📌 **Số đổi so với bản 1-seed trước đó** (S1: +1.0717→+0.7166 dB, S4: +1.0238→
> +1.0372 dB) — vì là checkpoint khác (multi-seed seed42 deterministic, không phải
> 1-seed `benchmark=True`). Val acc in ra lúc chạy (S1 797/999, S4 789/999) khớp
> chính xác với [multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md),
> xác nhận đúng checkpoint.
>
> 🚨 **J1 không có PSNR/SSIM và không thể có** — J1 không có nhánh SR nên `I_SR`
> **không tồn tại**; [`eval_sr_quality.py:131`](../tools/eval_sr_quality.py) chặn thẳng.
> Đây là **giới hạn cấu trúc**, không phải thiếu sót. Hệ quả: bảng PSNR **không phủ
> được cấu hình có điểm trung bình cao nhất** — phải nêu trong Limitations.
>
> ✅ **Bảng tương quan ở mục 3.1 cũng đã tính lại** từ `sr_quality_s1_seed42.csv` /
> `sr_quality_s4_seed42.csv` — toàn bộ số PSNR/SSIM trong tài liệu này nay cùng một
> nguồn (checkpoint multi-seed seed 42).

⚠️ **Con số dao động ~±0.05 dB giữa các lần chạy.** Pipeline degradation (blur/noise/JPEG)
là ngẫu nhiên; cờ `--seed` chỉ seed tiến trình chính, còn `--num-workers > 0` thì mỗi
worker của DataLoader có trạng thái ngẫu nhiên riêng mà albumentations không nhận seed
đó. Muốn tái lập tuyệt đối thì chạy `--num-workers 0` (chậm hơn). Biên dao động này
nhỏ hơn nhiều so với khoảng cách giữa các cấu hình nên không đổi kết luận nào.

⚠️ **Không so PSNR tuyệt đối giữa S4 và S1**: S1 xuất ảnh 64×256, S4 xuất 32×128 —
hai thang khác nhau. Chỉ so được cột "Chênh" vì mỗi cấu hình so với base của chính nó.

---

## 3. Kết quả quan trọng nhất — PSNR/SSIM **NGHỊCH** với khả năng đọc biển số

### 3.1. Bằng chứng mức từng track (n = 999) — mạnh nhất

Ghép PSNR từng track (`sr_quality_*_seed42.csv`) với kết quả đúng/sai từng track
(`submission_*_seed42.txt`), tính hệ số tương quan điểm-nhị phân:

| Cấu hình | Track đúng | PSNR ở track **đọc đúng** | PSNR ở track **đọc sai** | Chênh | r(PSNR, đúng) |
|---|---:|---:|---:|---:|---:|
| **S1** | 797/999 | 16.124 | 18.419 | **−2.295 dB** | **−0.4106** |
| **S4** | 789/999 | 16.996 | 19.210 | **−2.215 dB** | **−0.4044** |

✅ **Tính trên checkpoint multi-seed seed 42** (2026-08-05), cùng nguồn với bảng
accuracy Mean ± Std. Track đúng chấm lại từ `submission_*_seed42.txt` với `plate_text`
gốc — ra đúng **797/999** và **789/999**, khớp chính xác
[multi_seed_results.md](baseline1_crnn_stn/multi_seed_results.md).

**Track mà model đọc SAI lại có PSNR CAO hơn ~2.2 dB so với track đọc đúng.** Tương quan
âm rõ rệt (**r ≈ −0.41** ở cả hai), **nhất quán ở cả hai cấu hình có nhánh SR**, mỗi cấu
hình đo trên **n = 999 track độc lập**.

> 📌 So với bản đo trên checkpoint 1-seed cũ (S1: r = −0.362, chênh −1.952 dB; S4:
> r = −0.400, chênh −2.291 dB) — hiệu ứng nghịch **mạnh hơn một chút** ở S1 và gần như
> không đổi ở S4. Kết luận không đổi, chỉ vững thêm.

Nghĩa là PSNR không chỉ *vô dụng* với OCR — nó **gây hiểu lầm**: tối ưu theo PSNR sẽ
đẩy model đi sai hướng.

**Giải thích khả dĩ (giả thuyết, chưa kiểm chứng riêng)**: PSNR bị chi phối bởi *nội
dung ảnh* hơn là chất lượng SR. Ảnh biển số mờ nhoè / tương phản thấp có sai khác pixel
nhỏ → **PSNR cao**, nhưng đúng là loại ảnh khó đọc nhất. Ngược lại ảnh sắc nét, nhiều
chi tiết tần số cao thì khó tái tạo → PSNR thấp → nhưng lại dễ đọc. Tức PSNR đang đo
"ảnh này dễ tái tạo tới đâu", còn OCR cần "ảnh này chứa bao nhiêu chi tiết đọc được" —
hai thứ nghịch nhau.

### 3.2. Bằng chứng mức kiến trúc — mạnh nhất, không cần tới PSNR

| | Chất lượng ảnh SR | Đọc biển số (3 seed) |
|---|---|---|
| **J1** | **không có nhánh SR** — không tái tạo ảnh gì cả | **80.45% ± 0.45** — cao nhất |
| **S1** | +0.72 dB, +0.043 SSIM | 79.95% ± 0.15 |
| **S4** | +1.04 dB, +0.076 SSIM | 79.48% ± 0.44 |

📌 Thêm một tầng nghịch nữa: **S4 tái tạo ảnh tốt hơn S1** (+1.04 vs +0.72 dB) nhưng
**đọc kém hơn** (79.48% vs 79.95%) — cùng chiều với tương quan âm ở mục 3.1.

**Cấu hình bỏ hẳn việc tái tạo ảnh lại đọc biển số tốt nhất.** Đây là bằng chứng mạnh
hơn mọi hệ số tương quan: không cần đo PSNR để thấy rằng chất lượng ảnh tái tạo **không
phải** yếu tố quyết định khả năng đọc.

**Dùng khi viết paper**: đây là câu trả lời cho câu hỏi *"sao không tối ưu theo PSNR"* —
không chỉ vì PSNR không giúp, mà vì **nó nghịch chiều với mục tiêu** (mục 3.1, r ≈ −0.4
trên 999 track), và vì **cấu hình không có SR nào lại đọc tốt nhất** (mục này).

> 📌 Hai ablation 1-seed khác (λ_SR=0.5 và perceptual loss, nay ở `backup/report/`) cho
> cùng chiều: cấu hình có PSNR cao nhất toàn dự án (+2.29 dB) lại có exact match **kém
> nhất**. Không đưa vào bảng chính vì không có error bar.

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

> ✅ **Chạy trên checkpoint multi-seed** (seed 42), không dùng `backup/mf_sr_ocr/` (1
> seed cũ) nữa — để số PSNR khớp nguồn với bảng accuracy Mean ± Std. Chạy sau khi
> multi-seed đã xong.

```bash
# S1 — BẮT BUỘC. --num-workers 0 để tái lập tuyệt đối (mặc định gây dao động ±0.05 dB)
python tools/eval_sr_quality.py --lr-domain-match --num-workers 0 \
  --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
  --output-csv results/multi-seed/s1_mf_sr_ocr/sr_quality_s1_seed42.csv

# S4 — NÊN chạy. sr_scale=1 => BẮT BUỘC --width-downsample 4 (J1 không chạy được)
python tools/eval_sr_quality.py --lr-domain-match --num-workers 0 --width-downsample 4 \
  --checkpoint results/multi-seed/s4_sr_scale1/s4_seed42_best.pth \
  --output-csv results/multi-seed/s4_sr_scale1/sr_quality_s4_seed42.csv
```

⚠️ zsh không tách từ khi expand biến — đừng gom cờ vào biến rồi truyền, viết thẳng.

CER/NED tự động có trong `history_*.csv` của **cả 9 run multi-seed** (2 cột `val_cer`,
`val_ned`). Các run 1-seed cũ chạy trước 2026-08-01 chưa có 2 cột đó — số của chúng
được tính lại từ `submission_*.txt` + nhãn gốc.

> ⚠️ `*.pth`, `submission_*.txt`, `log*.txt` bị **gitignore**. Số liệu tổng hợp phải
> được chép vào các file `.md` trong `report/` — chỉ những file đó mới chắc chắn commit được.
