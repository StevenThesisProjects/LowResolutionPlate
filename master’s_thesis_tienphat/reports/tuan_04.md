# Tuần 4 (26/10–01/11): Đánh giá LGTF

## Roadmap yêu cầu

1. Train LGTF (`score = ScoreNet(F) + γ·log Var_Laplacian(x)`, γ học được) 3 seed trên cấu hình tốt nhất.
2. Log γ theo epoch → `gamma_curve.png`.
3. Phân tích theo tứ phân vị độ nét Q1–Q4 và theo layout → `slice_sharpness.csv`, `results_lgtf.csv`.

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng | Lý do |
|---|---|---|
| LGTF 3 seed | **Đóng hướng LGTF.** Ý tưởng này trong code mới là cờ `--sharpness-prior` (`AttentionFusion`, γ khởi tạo 0). Run #33 (1 seed, dev 1.999): best 76,69 / TB6 76,16, thấp hơn mốc #13 (best 77,59) và cả lần chạy lại #13 là run #25 (best 77,04 / TB6 76,60). Thêm vào đó, paper tổng kết ghi đội hạng 5 đã dùng trọng số frame theo chất lượng (roadmap, mục Lưu ý 2) | Kết quả âm, và novelty trùng top-5 |
| Nghiên cứu fusion (mục đích của LGTF) | **Ba thí nghiệm cơ chế có kiểm soát** theo review của thầy, chạy trên giao thức lock-box: **E3** (accuracy theo số frame), **E4** (vai trò STN), **N1** (kiểu che frame) | Trả lời câu hỏi fusion học được gì, thay vì thêm module |
| `gamma_curve.png` | Không có: không train LGTF | — |
| `slice_sharpness.csv` | `artifacts/slice_sharpness.csv`: baseline so với công thức chốt, theo tứ phân vị độ nét × layout, trên lock-box. Dùng lại kết quả chấm ngày 25/10, không suy luận thêm | Câu hỏi "lợi ích nằm ở ảnh mờ hay ảnh nét" vẫn đáng trả lời, nhưng cho công thức chốt |

## Kết quả

**E3: accuracy theo số frame lúc test** (lock-box, epoch cuối, 4 seed, trung bình trên mọi tập con N frame):

| N | Baseline | Công thức | Chênh [CI 95%] |
|---|---|---|---|
| 1 | 50,59 | 55,59 | **+5,00** [+4,32; +5,69] |
| 2 | 66,14 | 69,02 | +2,88 [+2,29; +3,48] |
| 3 | 72,18 | 74,13 | +1,95 [+1,27; +2,63] |
| 4 | 75,31 | 77,00 | +1,69 [+0,94; +2,43] |
| 5 | 77,60 | 79,12 | +1,52 [+0,53; +2,52] |

Kết quả rơi vào dạng B đã đăng ký trước: công thức làm model bền hơn khi thiếu frame hoặc frame kém, chứ không khai thác thêm thông tin bổ sung giữa 5 frame.

**E4: vai trò của STN** (lock-box, Last-6, 3 seed): STN học được 78,29 ± 0,26; crop cố định bằng θ trung bình 76,60 ± 0,15 (−1,68 [−2,78; −0,59]); không STN 74,94 ± 0,74 (−3,35 [−4,69; −2,02]). STN không phải chỉ là một phép zoom cố định: phần zoom và phần thích nghi theo từng ảnh mỗi phần đóng góp khoảng 1,7 điểm.

**N1: kiểu che frame** (lock-box, Last-6, 3 seed): không che 78,38; tube +0,33; độc lập +0,11; bổ sung −0,19; bổ sung − tube = −0,52 [−1,39; +0,36]. Không kiểu che nào có tác dụng đo được.

Chi tiết: `documents/REPORT_E4_N1.md`, RUNS.md run #71–#91.

**Theo độ nét × layout** (`scripts/slice_analysis.py`; lock-box; epoch cuối; trung bình 4 seed mỗi bên; độ nét = phương sai Laplacian của ảnh xám 32×128, trung bình 5 frame; ranh giới tứ phân vị 181 / 353 / 553):

| Lát cắt | n | Baseline | Công thức | Chênh [CI 95% theo track] |
|---|---|---|---|---|
| Q1 (mờ nhất) | 250 | 53,00 | 56,40 | +3,40 [+0,90; +6,00] |
| Q2 | 250 | 79,20 | 81,60 | +2,40 [+0,30; +4,60] |
| Q3 | 250 | 84,10 | 84,50 | +0,40 [−1,10; +1,90] |
| Q4 (nét nhất) | 250 | 94,10 | 94,00 | −0,10 [−1,60; +1,40] |
| Mercosur | 808 | 81,62 | 83,23 | +1,61 [+0,59; +2,66] |
| Brazilian | 192 | 60,68 | 61,85 | +1,17 [−1,43; +3,91] |

- Mức lợi nằm ở nửa ảnh mờ (Q1, Q2), còn ở nửa ảnh nét gần bằng 0. Kết quả này khớp với E3: công thức giúp khi thông tin trong frame ít.
- **Đây là phân tích thăm dò, không đăng ký trước.** Có 6 lát cắt, CI chưa hiệu chỉnh cho so sánh bội, và mỗi tứ phân vị chỉ có 250 track. Chỉ nên viết là "xu hướng nhất quán với E3", không dùng làm claim chính.
- Bảng theo layout × tứ phân vị nằm trong `artifacts/slice_sharpness.csv`. Mỗi ô Brazilian chỉ có 36–57 track, nên không đủ lực thống kê.

## Deliverable

| Roadmap | File thay thế | Trạng thái |
|---|---|---|
| `results_lgtf.csv` | Không có. Thay bằng E3/E4/N1 (`results_new/lockbox_eval.json`, `results_new/arms_eval_e4.json`, `results_n1/arms_eval_n1.json`) | ✅ (thay thế) |
| `gamma_curve.png` | Không có: hướng LGTF đã đóng | — |
| `slice_sharpness.csv` | `artifacts/slice_sharpness.csv` (+ `.json`) | ✅ |
