# Báo cáo kết quả lock-box

Ngày 25/10/2026. Trả lời ISSUE #1 trong review ngày 23/10.
- Protocol đăng ký trước: [`documents/lockbox_protocol.md`](lockbox_protocol.md).
- Số liệu từng run: [`documents/RUNS.md`](RUNS.md), run #63–#70.
- Báo cáo tổng, đã cập nhật theo toàn bộ review: [`documents/REPORT.md`](REPORT.md).

---

## Tóm tắt

- Đã làm theo phương án dự phòng trong ISSUE #1:
  - tách 1.000 track lock-box chưa từng dùng cho bất kỳ quyết định nào;
  - viết protocol trước khi chạy;
  - train lại 4 seed baseline và 4 seed phương pháp, không dùng các track này;
  - đóng băng checkpoint, rồi chấm đúng một lần.
- **Phương pháp hơn baseline +1,52 điểm** ở epoch cuối (thước đo chính) và **+1,64 điểm** ở Last-6 (thước đo quyết định). CI 95% bootstrap theo track lần lượt là [+0,53; +2,52] và [+0,78; +2,48].
- Theo quy tắc đăng ký trước, kết quả chọn **hướng (b)**: điều chuẩn fusion là trọng tâm bài viết, SR là một mục phân tích.
- Mức lợi trên lock-box ngang mức lợi trên val 999 (+1,38 đến +1,54). Việc chọn cấu hình trên val không làm phồng kết quả, tức giả thuyết H2 trong review không được ủng hộ.
- **Điểm yếu:** ở tầng seed, Welch p = 0,06–0,09, chưa đạt 0,05, vì phương pháp dao động giữa các seed nhiều hơn baseline.
- **E3 cho dạng B:** mức lợi lớn nhất khi chỉ có 1 frame (+5,00) và giảm dần đến 5 frame (+1,52). Phương pháp làm model bền hơn khi thiếu frame. Kết quả này chưa chứng minh model khai thác thông tin bổ sung giữa các frame tốt hơn.

---

## 1. Cách làm

| Bước | Nội dung |
|---|---|
| Protocol | Viết ngày 23/10, trước khi chạy run nào: tập lock-box, hai cấu hình, thước đo, thống kê, quy tắc (a)/(b) và cách diễn giải E3 |
| Kiểm tra code (ISSUE #9) | Test tự động: đánh giá cho kết quả tất định; frame dropout và che frame không chạy lúc val/test; chuyển ảnh xám đúng; thống kê tự viết khớp scipy. Kiểm tra hồi quy: khi tắt mọi cờ mới, code train giống từng bit code gốc (`duongtruongbinh/MultiFrame-LPR`, commit `3ca05c0`) |
| Tập lock-box | 1.000 track Scenario-B, lấy tất định (`random.Random(2027)`) từ 8.001 track trước nay chỉ dùng để train. Không trùng val 999 và val 1.999. Gồm 192 Brazilian, 808 Mercosur |
| Huấn luyện | Baseline đã ổn định (`--lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100`) và phương pháp (cấu hình #50, giữ nguyên theo ISSUE #3). Mỗi bên 4 seed (42, 43, 7, 2026), 30 epoch. Train 18.001 track (20.000 − 999 val − 1.000 lock-box), mỗi track kèm một mẫu synthetic như code gốc. Val 999 chỉ dùng để theo dõi và chọn checkpoint "best" |
| Đóng băng | Trước khi chấm, ghi SHA-256 của 24 file code, 3 file split, 56 checkpoint (6 epoch cuối và best của mỗi run) cùng thông tin môi trường vào `freeze_manifest.json` |
| Chấm | Một lần duy nhất, ngày 25/10, trên Tesla V100-SXM2 32 GB, PyTorch 2.14.1+cu126, AMP fp16 |

**Thước đo** (accuracy khớp tuyệt đối theo track):
- **Epoch cuối:** thước đo chính, vì không có bước chọn epoch nào.
- **Last-6:** trung bình 6 epoch cuối, dùng cho quy tắc quyết định.
- **Best:** checkpoint chọn trên val 999, để so với cách tính số công bố 78,70.

**Quy tắc quyết định** (đăng ký trước): nếu chênh Last-6 ≥ +1,0 điểm và cận dưới CI 95% bootstrap > 0 thì chọn (b), ngược lại chọn (a).

---

## 2. Kết quả

**Theo từng seed** (accuracy % trên 1.000 track lock-box):

| Seed | Baseline: cuối / Last-6 / best | Phương pháp: cuối / Last-6 / best |
|---|---|---|
| 42 | 77,40 / 76,50 / 77,60 | 79,90 / 79,28 / 80,00 |
| 43 | 77,40 / 77,40 / 77,80 | 80,30 / 80,05 / 80,60 |
| 7 | 77,90 / 77,17 / 77,90 | 78,10 / 77,88 / 78,00 |
| 2026 | 77,70 / 77,23 / 77,20 | 78,20 / 77,63 / 78,20 |

**So sánh:**

| Thước đo | Baseline | Phương pháp | Chênh | CI 95% bootstrap theo track | Welch theo seed | Cohen's d |
|---|---|---|---|---|---|---|
| **Epoch cuối** (chính) | 77,60 ± 0,24 | 79,12 ± 1,14 | **+1,52** | [+0,53; +2,52] | p = 0,072 | 1,85 |
| **Last-6** (quyết định) | 77,07 ± 0,40 | 78,71 ± 1,15 | **+1,64** | [+0,78; +2,48] | p = 0,059 | 1,90 |
| Best theo val 999 | 77,62 ± 0,31 | 79,20 ± 1,30 | +1,57 | [+0,57; +2,55] | p = 0,090 | 1,67 |

Bootstrap lấy mẫu lại 1.000 track 10.000 lần, trên chênh lệch accuracy từng track (đã lấy trung bình 4 seed).

McNemar chính xác ở epoch cuối, từng cặp seed (phương pháp đúng mà baseline sai / ngược lại):

| Seed | 42 | 43 | 7 | 2026 |
|---|---|---|---|---|
| Track | +60 / −35 | +59 / −30 | +49 / −47 | +51 / −46 |
| p | 0,013 | 0,003 | 0,92 | 0,68 |

**E3: accuracy theo số frame.** Đo ở epoch cuối, lấy trung bình mọi tập con N frame, trên 4 seed. Model được train với 5 frame; lúc test chỉ đưa vào N frame.

| N | Baseline | Phương pháp | Chênh [CI 95%] |
|---|---|---|---|
| 1 | 50,59 | 55,59 | **+5,00** [+4,32; +5,69] |
| 2 | 66,14 | 69,02 | +2,88 [+2,29; +3,48] |
| 3 | 72,18 | 74,13 | +1,95 [+1,27; +2,63] |
| 4 | 75,31 | 77,00 | +1,69 [+0,94; +2,43] |
| 5 | 77,60 | 79,12 | +1,52 [+0,53; +2,52] |

**Phân tích phụ** (đăng ký trước, chỉ để mô tả; epoch cuối, trung bình 4 seed):
- **Theo loại biển:** Brazilian (n = 192) 60,68 → 61,85 (+1,17); Mercosur (n = 808) 81,62 → 83,23 (+1,61). Brazilian quá ít track nên chưa đủ lực thống kê để so mức lợi giữa hai loại biển.
- **Confidence Gap** (độ tin cậy trung bình của dự đoán đúng trừ dự đoán sai): phương pháp 0,1135, baseline 0,1060; phương pháp cao hơn ở cả 4 seed. Với cấu hình chốt, phương pháp không đánh đổi độ tin cậy (MINOR 12).
- **Batch bị bỏ qua:** baseline 8–11, phương pháp 8–9, trên 16.890 batch mỗi run (MINOR 13). Tất cả đều là bước có gradient không hữu hạn của fp16; ngưỡng grad norm 100 chưa lần nào kích hoạt.
- **Val 999 của chính 8 run này:** baseline best 76,78–77,08; phương pháp best 76,88–79,48. Seed yếu trên val (42, 2026) khác seed yếu trên lock-box (7, 2026). Đây là nhiễu ngẫu nhiên quanh mức lợi trung bình, không phải một seed kém cố định.

---

## 3. Diễn giải

1. **Quy tắc đăng ký trước chọn (b):** chênh Last-6 +1,64 ≥ +1,0, và cận dưới CI +0,78 > 0.
2. **Mức lợi không bị thổi phồng bởi việc chọn cấu hình trên val.**

   | | Best | Last-6 | Epoch cuối |
   |---|---|---|---|
   | Val 999 (4 seed, đã dùng để chọn cấu hình) | +1,38 | +1,54 | +1,48 |
   | Lock-box (4 seed, chưa từng dùng) | +1,57 | +1,64 | +1,52 |

3. **Hai tầng thống kê cho kết luận khác nhau, nên sẽ báo cáo cả hai.**
   - Theo track: CI không chứa 0 ở cả 3 thước đo.
   - Theo seed: p = 0,06–0,09. Ở epoch cuối, seed 42 và 43 tăng 2,5–2,9 điểm, còn seed 7 và 2026 chỉ tăng 0,2–0,5 điểm.
   - Dù vậy, cả 4 seed phương pháp đều cao hơn cả 4 seed baseline ở mọi thước đo. *Phân tích bổ sung, không đăng ký trước:* kiểm định hoán vị chính xác giữa hai nhóm seed cho p hai phía = 2/70 = 0,029.
4. **E3 rơi vào dạng B đã đăng ký:** mức lợi lớn nhất khi có ít frame và giảm dần, mọi CI đều không chứa 0.
   - Theo protocol, dạng B nghĩa là phương pháp làm model bền hơn khi thiếu frame hoặc frame kém. Điều này khớp với thiết kế: frame dropout và che frame đều buộc model đọc được khi có ít thông tin.
   - Đây không phải dạng A (khai thác thông tin bổ sung giữa các frame). Thí nghiệm N1 (các kiểu che frame) sẽ kiểm tra riêng cơ chế đó.
   - Kết quả này cũng liên quan trực tiếp đến tính năng nhận ít hơn 5 ảnh của app: với 1–2 ảnh, phương pháp hơn baseline khoảng 3–5 điểm.

---

## 4. Kiểm chứng và thay đổi so với protocol

**Kiểm chứng sau khi chấm:**
- 56/56 checkpoint, 24/24 file code và 3/3 file split khớp SHA-256 với manifest.
- Tính lại CI với seed bootstrap khác: Last-6 [+0,80; +2,52]; quyết định không đổi.
- Chấm lại 14 checkpoint trên máy khác (CPU và Apple GPU): 3/28.000 dự đoán đổi kết quả. Cả 3 là trường hợp hai ký tự gần hoà nhau (chênh log-prob ≤ 0,008). Lệch lớn nhất 0,1 điểm trên một checkpoint; kết luận không đổi.

**Thay đổi so với protocol:**
- Log huấn luyện của `restran_lockbox_base_s42` không có trên server. File summary của run này vẫn ghi `Train samples: 36002` (18.001 track × 2), giống 7 run còn lại, nên run này đã loại lock-box đúng.
- Kiểm định hoán vị ở mục 3 là phân tích bổ sung, không đăng ký trước.

**Khả năng tái lập:**
- Huấn luyện không tái lập được từng bit trên GPU. Ngoài tính không tất định của GPU, albumentations 2.x lấy số ngẫu nhiên từ một bộ sinh riêng mà `--seed` không điều khiển được (đã kiểm chứng).
- Vì vậy mỗi bên dùng 4 seed, và kiểm định tính cả dao động giữa các seed lẫn dao động giữa các track.

---

## 5. Tiến độ các mục khác

| Mục | Trạng thái |
|---|---|
| N1 (các kiểu che frame), E4 (STN so với crop cố định) | Đang thực hiện, sẽ báo cáo khi có kết quả |
| Prototype (API `/predict`) | Chưa hoàn thành. Thời gian qua ưu tiên huấn luyện và đánh giá mô hình; sẽ báo cáo tiến độ riêng |
| Tập test chính thức | Bản đang có không kèm nhãn; đang xin cấp bản có nhãn. Khi có nhãn, 8 bộ checkpoint lock-box sẽ được chấm trên tập test với cùng thước đo và thống kê, như một xác nhận thêm |
| E2 (leave-one-out), E5 (cấu hình gốc fp32), E6 (SR thêm seed), N3 (hiệu chỉnh độ tin cậy) | Chưa làm |

---

## 6. Xin ý kiến thầy

1. **Phạm vi claim của hướng (b).** E3 cho dạng B, nên đề xuất claim chính là "phương pháp giúp mô hình bền hơn khi thiếu frame hoặc frame kém". Claim "khai thác thông tin bổ sung giữa các frame" chỉ đưa vào nếu N1 ủng hộ. Thầy thấy cách đặt claim này có hợp lý không?
2. **Tầng seed chưa đạt p < 0,05.** Dự định báo cáo đúng kết quả của cả hai tầng và không chạy thêm seed để chấm lại trên lock-box, vì lock-box chỉ được chấm một lần. Tập test chính thức, khi có nhãn, sẽ là phép xác nhận độc lập. Thầy có đồng ý với cách xử lý này không?
