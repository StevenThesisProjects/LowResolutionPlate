# Báo cáo kết quả: cải thiện ResTran cho nhận dạng biển số độ phân giải thấp đa khung hình (ICPR 2026 LRLPR)

Bản sửa ngày 25/10/2026, theo review của thầy ngày 23/10 (`documents/Teacher_Review.md`). Bản trước: 19/10/2026. Lệnh chạy, số liệu từng epoch và phân tích chi tiết nằm trong [`documents/RUNS.md`](RUNS.md). Số thứ tự `#N` trong báo cáo trùng với số run trong nhật ký đó. Protocol lock-box đăng ký trước: [`documents/lockbox_protocol.md`](lockbox_protocol.md).

---

## Kết quả chính

**Trên tập lock-box 1.000 track chưa từng dùng cho bất kỳ quyết định nào**, chấm đúng một lần theo protocol đã đăng ký trước:

| | Epoch cuối (thước đo chính) | Last-6 (thước đo quyết định) | Best theo val 999 |
|---|---|---|---|
| Baseline ResTran + STN đã ổn định (4 seed) | 77,60 ± 0,24 | 77,07 ± 0,40 | 77,62 ± 0,31 |
| **Phương pháp đề xuất** (4 seed) | **79,12 ± 1,14** | **78,71 ± 1,15** | **79,20 ± 1,30** |
| **Chênh lệch** | **+1,52** | **+1,64** | **+1,57** |
| CI 95% bootstrap theo track | [+0,53; +2,52] | [+0,78; +2,48] | [+0,57; +2,55] |
| Welch t-test theo seed | p = 0,072 | p = 0,059 | p = 0,090 |

- **Một mô hình duy nhất, train một lần:** không ensemble, không dùng dữ liệu tập test, kiến trúc ResTran giữ nguyên (31,08M tham số), chi phí suy luận như ResTran gốc.
- **Quy tắc đăng ký trước** (chênh Last-6 ≥ +1,0 và cận dưới CI > 0) cho ra **hướng (b)**: điều chuẩn fusion là trọng tâm bài viết.
- Cả 4 seed của phương pháp đều cao hơn cả 4 seed của baseline ở mọi thước đo. Tầng seed chưa đạt p < 0,05 vì phương pháp dao động giữa các seed nhiều hơn (mục 4.1).
- Mức lợi trên lock-box (+1,52 đến +1,64) **ngang** mức lợi trên val 999 (+1,38 đến +1,54), nên việc chọn cấu hình trên val không làm phồng kết quả.

**Run cao nhất** (chỉ để tham khảo; con số đại diện là trung bình 4 seed ở trên):

| Mô hình | Tập đo | Accuracy |
|---|---|---|
| ResTran + STN, số công bố (1 run) | val 999 (theo cách chia của code gốc) | 78,70, epoch tốt nhất |
| Phương pháp, seed 43 | lock-box 1.000 track | **80,30** ở epoch cuối (80,60 với checkpoint best) |
| Phương pháp, seed 42, 40 epoch (#59) | val 999 | **79,68**, epoch tốt nhất, cùng cách đo với số công bố |
| Phương pháp, seed 42, 30 epoch (#51) | val 999 | 79,28, epoch tốt nhất |

- Mỗi dòng là run cao nhất trong nhóm của nó: seed 43 trong 4 run lock-box; #59 trong 8 run val 999 (4 seed × 30 và 40 epoch); #51 trong 4 run 30 epoch.
- Hai dòng val 999 chọn epoch trên chính tập val nên lạc quan.
- **Ghi chú về số công bố:** 78,70% không ứng với số nguyên track nào trên 999 track (786/999 = 78,68%; 787/999 = 78,78%), nhưng đúng bằng 787/1000. Vậy tập val của số công bố có thể có 1.000 track, hoặc slide đã làm tròn 78,68% thành 78,7%. Chênh lệch này nhỏ hơn nhiều so với sai số lấy mẫu của tập val (khoảng ±1,3).

---

## Thay đổi so với bản 19/10

| Review | Thay đổi |
|---|---|
| ISSUE #1 | Thêm tập lock-box, protocol đăng ký trước, freeze manifest, chấm một lần. Lock-box là kết quả chính |
| ISSUE #2 | Thống kê hai tầng: theo seed (Welch, Cohen's d) và theo track (paired bootstrap, McNemar từng cặp seed) |
| ISSUE #4, #5, #6, #7 | Viết lại các claim với phạm vi đúng (mục 2, 3, 6) |
| ISSUE #8 | E3: accuracy theo số frame (mục 4.1). N1 xong: không kiểu che nào có tác dụng đo được; cơ chế "thông tin bổ sung giữa các frame" không được ủng hộ (mục 9) |
| ISSUE #9 | 14 test tự động. Kiểm tra hồi quy: khi tắt mọi cờ mới, code train **giống từng bit** với code gốc |
| ISSUE #10 | Đo phép biến đổi mà STN học được (mục 8). E4 xong: STN hơn crop cố định 1,68 điểm, crop cố định hơn không STN 1,67 điểm (mục 9) |
| MINOR 12–19 | Xử lý hết, chi tiết ở mục 10 |

---

## Tóm tắt

**Mục tiêu.** Cải thiện ResTran (STN + ResNet34 + Attention Fusion + Transformer + CTC) cho bài toán nhận dạng biển số từ 5 khung hình độ phân giải thấp. Ràng buộc: một mô hình duy nhất, train một lần, không ensemble, không dùng dữ liệu tập test để train.

**Phương pháp.** Giữ nguyên kiến trúc; chỉ đổi cách huấn luyện và dữ liệu: ổn định hoá tối thiểu, augment nhất quán giữa các frame, sửa degradation của nhánh synthetic, CTC phụ cho từng frame, che độc lập từng frame, frame dropout, input ảnh xám (mục 3).

**Kết quả.**
- **Lock-box** (kết quả chính, không có thiên lệch chọn lọc): +1,52 điểm ở epoch cuối (79,12 so với 77,60), CI 95% theo track [+0,53; +2,52].
- **Val 999** (giao thức của số công bố, tập đã dùng để chọn cấu hình): +1,48 điểm ở epoch cuối (78,73 so với 77,25), Welch p = 0,003.
- **Theo số frame (E3):** mức lợi lớn nhất khi chỉ có 1 frame (+5,0) và giảm dần đến 5 frame (+1,5). Theo protocol, kết quả này nghĩa là phương pháp làm model **bền hơn khi thiếu frame hoặc frame kém**. Nó chưa chứng minh model khai thác thông tin bổ sung giữa các frame tốt hơn, và N1 cũng không ủng hộ cơ chế đó (mục 9).
- **Cơ chế (mục 9):** E4: lợi ích của STN gồm phóng to vùng ký tự và điều chỉnh theo từng ảnh, mỗi phần khoảng +1,7 điểm. N1: che frame không có tác dụng đo được trong cấu hình chốt, dù che kiểu nào.
- **So với số công bố 78,70:** con số này nằm trong khoảng dao động giữa các seed của phương pháp trên val 999 (best từ 78,48 đến 79,28).
- **Chi phí:** không thêm tham số, không đổi chi phí suy luận. Huấn luyện chậm hơn: augment nhất quán làm mỗi epoch chậm thêm 46%, CTC phụ làm mỗi batch chậm thêm khoảng 8% (mục 3).
- **Super-Resolution đa khung hình** đã được kiểm tra có đối chứng. Với kiến trúc SR và loss pixel đã thử, SR dựng ảnh tốt hơn bilinear 9,3% nhưng accuracy nhận dạng giảm 1,36 điểm (1 seed; mục 6).

**Còn thiếu:** tập test chính thức 3.000 track (bản đang có không kèm nhãn, đang xin cấp); leave-one-out nhiều seed (E2), nay quan trọng hơn vì cần biết mức lợi và độ bền khi thiếu frame đến từ thành phần nào.

---

## 1. Dữ liệu và giao thức đánh giá

**Dữ liệu (LRLPR-26).** 20.000 track train, mỗi track có 5 ảnh LR và 5 ảnh HR.
- Scenario-A: 10.000 track, điều kiện ổn định, ảnh PNG; LR khoảng 34×18 px.
- Scenario-B: 10.000 track, có mưa và ban đêm, ảnh **JPEG lấy mẫu màu 4:2:0**; LR khoảng 47×17 px, HR khoảng 124×44 px.
- Hai loại biển: Mercosur (`LLLNLNN`) và Brazilian cũ (`LLLNNNN`), khoảng 80/20 trong Scenario-B và tập test.

**Ba giao thức.** Mọi tập val và lock-box đều lấy từ Scenario-B, không trùng nhau ở cấp track. Mỗi track train có thêm một mẫu synthetic (ảnh HR bị degrade), như code gốc.

| Giao thức | Train | Đánh giá | Dùng cho |
|---|---|---|---|
| Phát triển | 18.001 track | val 1.999 track (chứa trọn val 999) | Chọn cấu hình, ablation 1 seed (mục 5–8) |
| Của số công bố | 19.001 track | val 999 track, cách chia mặc định của code gốc (seed 42, tỉ lệ 0,9) | So sánh với 78,70 (mục 4.2) |
| **Lock-box** | 18.001 track (20.000 − 999 − 1.000) | **1.000 track lock-box**; val 999 chỉ để theo dõi | **Kết quả chính** (mục 4.1) |

- **Lock-box** gồm 1.000 track Scenario-B lấy tất định (`random.Random(2027)`) từ 8.001 track trước nay chỉ dùng để train, không trùng hai tập val. Thành phần: 192 Brazilian, 808 Mercosur. SHA-256 của file: `09f90d2b…`.
- **Quy trình đăng ký trước:** viết protocol trước khi chạy run nào → train 8 run (4 seed mỗi bên) → đóng băng (SHA-256 của 24 file code, 3 file split, 56 checkpoint) → chấm đúng một lần.

**Thước đo** (accuracy khớp tuyệt đối theo track):
- **Epoch cuối:** thước đo chính trên lock-box, không có bước chọn epoch nào (MINOR 17).
- **Last-6:** trung bình 6 epoch cuối (bản trước viết là TB6). Đây là thước đo của quy tắc quyết định.
- **Best:** epoch tốt nhất theo val, cùng cách tính với số công bố 78,70. Khi chấm trên chính tập val đó thì lạc quan.

**Thống kê hai tầng** (ISSUE #2):
- *Theo seed* (dao động do huấn luyện): mean ± std, Welch t-test, Cohen's d.
- *Theo track* (dao động do lấy mẫu track): paired bootstrap 10.000 lần trên chênh lệch accuracy từng track (đã lấy trung bình các seed), và McNemar chính xác cho từng cặp seed.
- Ablation trên giao thức phát triển chỉ có 1 seed mỗi cấu hình. Chạy lặp lại cùng cấu hình cho độ lệch chuẩn khoảng 0,33 (best) và 0,37 (Last-6). Vì vậy chênh dưới khoảng 1 điểm giữa hai run đơn được coi là nằm trong nhiễu, trừ khi McNemar cho thấy khác.

**Môi trường:** Tesla V100-SXM2 32 GB, Python 3.10.13, PyTorch 2.14.1+cu126, CUDA 12.6, cuDNN 9.10.2, AMP fp16, batch 64.

**Khả năng tái lập** (MINOR 14):
- Huấn luyện không tái lập được từng bit trên GPU. Ngoài tính không tất định của GPU, albumentations 2.x lấy số ngẫu nhiên từ một bộ sinh riêng mà `--seed` không điều khiển được (đã kiểm chứng). Ví dụ: run #3 và #6 cùng seed 42 lệch nhau 0,5 điểm. Vì vậy mọi so sánh chính đều dùng 4 seed mỗi bên.
- Chấm lại checkpoint trên phần cứng khác: 3/28.000 dự đoán đổi kết quả. Cả 3 đều là trường hợp hai ký tự gần hoà nhau (chênh log-prob giữa lớp nhất và lớp nhì ≤ 0,008). Lệch lớn nhất là 0,1 điểm trên một checkpoint. Mọi số lock-box đều chấm trên cùng một server.
- Khi tắt mọi cờ mới, code train giống từng bit với code gốc (`duongtruongbinh/MultiFrame-LPR`, commit `3ca05c0`). Kiểm tra trên 64 track, 2 epoch, CPU, albumentations được cố định seed: loss từng epoch trùng nhau, và sau 1 lẫn 2 batch không tensor trọng số nào khác nhau (0/267).

---

## 2. Tái lập baseline 78,70%

Cấu hình công bố: LR 5e-4, grad clip 5.0, OneCycleLR, 30 epoch. Cấu hình Transformer khớp với tác giả: 3 layer, 8 head, FFN 2048, dropout 0,1; batch 64, weight decay 1e-4.

**Trong môi trường của chúng tôi (V100, PyTorch 2.14.1, AMP fp16), cấu hình công bố diverge ở 2/2 seed**, đúng lúc OneCycle đưa LR lên đỉnh 5e-4, khi CTC loss đã xuống khoảng 0,3:
- Seed 42 (#1): đạt 61,76% ở epoch 10, diverge ở epoch 12.
- Seed 43 (#55): đạt 54,25% ở epoch 9, diverge ở epoch 10. Grad norm 49.348; accuracy về 0% và train loss kẹt ở 3,65 tới hết.
- **Chưa loại trừ được tràn số fp16.** Run #55 có 21 batch có gradient inf/nan. Bằng chứng "trên 96% batch bị loại có gradient hữu hạn" lấy từ #7 và #8, với lịch 60 epoch khác. Một run cấu hình gốc ở fp32 (E5) sẽ trả lời câu hỏi này; lệnh đã sẵn, chưa có kết quả.

**Baseline đã ổn định**: giữ nguyên kiến trúc và pipeline dữ liệu, chỉ đổi 3 điểm tối thiểu, áp cho cả baseline lẫn phương pháp:
- LR 3e-4.
- Grad clip 1.0.
- Bỏ qua batch có grad norm trước clip > 100, kèm cơ chế dừng sớm khi quá 20% batch trong một epoch bị bỏ.

Thực tế trong 8 run lock-box, ngưỡng 100 chưa lần nào kích hoạt. Mọi batch bị bỏ đều là bước có gradient không hữu hạn mà GradScaler của AMP vẫn tự bỏ qua. Số batch bị bỏ của hai bên tương đương (MINOR 13): baseline 8–11, phương pháp 8–9, trên 16.890 batch mỗi run.

Kết quả baseline ổn định qua 4 seed trên val 999: best 77,58 ± 0,28. Chúng tôi báo cáo cả hai con số (78,70 và 77,58), và so sánh chính với baseline chạy lại cùng giao thức.

---

## 3. Phương pháp đề xuất

Kiến trúc ResTran giữ nguyên. Các thay đổi, theo thứ tự đã đưa vào:

| Thành phần | Nội dung | Lý do |
|---|---|---|
| Ổn định huấn luyện | LR 3e-4, grad clip 1.0, bỏ qua batch có gradient bất thường | Cấu hình gốc diverge (mục 2) |
| Augment nhất quán giữa các frame | Cùng tham số augment cho cả 5 frame của một track | Augment độc lập từng frame phá vỡ quan hệ không gian giữa các frame mà STN và fusion dựa vào |
| Bỏ ChannelShuffle | Bỏ phép hoán vị kênh màu ngẫu nhiên | Lý do ban đầu là màu nền giúp phân biệt loại biển. **Lý do này chỉ đúng với input RGB**; cấu hình chốt dùng ảnh xám (ISSUE #5). Với ảnh xám, ChannelShuffle trước bước chuyển luma vẫn có tác dụng: nó biến thành phép đổi độ sáng ngẫu nhiên ở các vùng có màu (ví dụ dải xanh của biển Mercosur). Việc bỏ nó còn có ích khi đã dùng ảnh xám hay không sẽ được đo trong E2 |
| Sửa degradation của nhánh synthetic | `GaussNoise` của albumentations 2.x mặc định làm lệch pixel khoảng 52/255, phá nét chữ. Đặt lại `std_range=(0.02, 0.08)` | Lỗi cấu hình trong pipeline gốc |
| CTC phụ từng frame (λ = 0,2) | Mỗi frame được đọc riêng bằng chính Transformer + head dùng chung, thêm CTC loss phụ | Giám sát sâu cho đặc trưng từng frame trước fusion; không thêm tham số |
| **Che độc lập từng frame** (p = 0,2) | Mỗi frame, với xác suất 0,2, bị xoá một dải dọc (nửa đến một ký tự) hoặc dải ngang (1/8–1/4 chiều cao). **Mỗi frame tự rút ngẫu nhiên** | Giả thuyết thiết kế: phần bị che ở frame này vẫn thấy ở frame khác, buộc fusion lấy thông tin từ frame khác. CoarseDropout sẵn có thì bị lặp y hệt trên cả 5 frame, nên phần bị che mất khỏi cả track. **N1 không ủng hộ giả thuyết này:** trong cấu hình chốt, che frame không có tác dụng đo được, dù che kiểu nào (mục 9). Thành phần này được giữ vì cấu hình đã đóng băng trước lock-box |
| **Frame dropout** (p = 0,1) | Thay đặc trưng của một frame bằng trung bình các frame còn lại, ngay trước fusion | Chống việc fusion dựa quá nhiều vào một frame |
| **Input luma (ảnh xám)** | Chuyển input sang kênh sáng BT.601 ở bước cuối, áp cho train, val và test | Ảnh Scenario-B và test là JPEG 4:2:0: kênh màu chỉ còn ½ độ phân giải mỗi chiều (khoảng 9 px chiều cao trên biển 17 px) nên chủ yếu là nhiễu nén |

**Chi phí** (MINOR 19): **không tăng chi phí suy luận**, vì mọi thành phần chỉ hoạt động lúc train và không thêm tham số. Huấn luyện thì chậm hơn:
- Augment nhất quán (`ReplayCompose`) làm mỗi epoch chậm thêm 46% (70,9 so với 47,0 s/epoch, run #13 so với run #9).
- CTC phụ làm mỗi batch train chậm thêm khoảng 8% (đo trên CPU, batch 16).
- Thời gian mỗi epoch của các run lock-box không so được với nhau, vì 4 run chạy song song. Luận văn sẽ dùng số đo từ một run chạy riêng trên V100.

Lệnh của cấu hình chốt nằm ở Phụ lục A.

---

## 4. Kết quả chính

### 4.1 Lock-box (kết quả chính)

Giao thức lock-box (mục 1), 4 seed mỗi bên, chấm một lần ngày 25/10/2026.

**Theo từng seed** (accuracy trên 1.000 track lock-box):

| Seed | Baseline: cuối / Last-6 / best | Phương pháp: cuối / Last-6 / best |
|---|---|---|
| 42 | 77,40 / 76,50 / 77,60 | **79,90** / 79,28 / 80,00 |
| 43 | 77,40 / 77,40 / 77,80 | **80,30** / 80,05 / 80,60 |
| 7 | 77,90 / 77,17 / 77,90 | 78,10 / 77,88 / 78,00 |
| 2026 | 77,70 / 77,23 / 77,20 | 78,20 / 77,63 / 78,20 |

**So sánh:**

| Thước đo | Baseline | Phương pháp | Chênh | CI 95% bootstrap (theo track) | Welch (theo seed) | Cohen's d |
|---|---|---|---|---|---|---|
| **Epoch cuối** (chính) | 77,60 ± 0,24 | 79,12 ± 1,14 | **+1,52** | [+0,53; +2,52] | p = 0,072 | 1,85 |
| **Last-6** (quyết định) | 77,07 ± 0,40 | 78,71 ± 1,15 | **+1,64** | [+0,78; +2,48] | p = 0,059 | 1,90 |
| Best theo val 999 | 77,62 ± 0,31 | 79,20 ± 1,30 | +1,57 | [+0,57; +2,55] | p = 0,090 | 1,67 |

McNemar ở epoch cuối, từng cặp seed: 42: +60/−35 (p = 0,013); 43: +59/−30 (p = 0,003); 7: +49/−47 (p = 0,92); 2026: +51/−46 (p = 0,68).

**Đọc kết quả:**
- **Theo track:** CI 95% không chứa 0 ở cả 3 thước đo.
- **Theo seed:** Welch p = 0,06–0,09, chưa đạt 0,05. Lý do là phương pháp dao động giữa các seed nhiều hơn baseline (std 1,1–1,3 so với 0,2–0,4). Ở epoch cuối, seed 42 và 43 tăng 2,5–2,9 điểm, còn seed 7 và 2026 chỉ tăng 0,2–0,5 điểm.
- Dù vậy, cả 4 seed phương pháp đều cao hơn cả 4 seed baseline ở mọi thước đo. *Phân tích bổ sung, không đăng ký trước:* kiểm định hoán vị chính xác cho p hai phía = 2/70 = 0,029.
- Cohen's d lớn (1,7–1,9) một phần vì độ lệch chuẩn của baseline nhỏ. Nên đọc d cùng với chênh lệch tuyệt đối.
- Hai tầng thống kê cho kết luận không giống nhau, nên bài viết sẽ báo cáo cả hai.

**E3: accuracy theo số frame** (epoch cuối, trung bình mọi tập con N frame, 4 seed). Mỗi model được train với 5 frame; lúc test chỉ đưa vào N frame.

| N | Baseline | Phương pháp | Chênh [CI 95%] |
|---|---|---|---|
| 1 | 50,59 | 55,59 | **+5,00** [+4,32; +5,69] |
| 2 | 66,14 | 69,02 | +2,88 [+2,29; +3,48] |
| 3 | 72,18 | 74,13 | +1,95 [+1,27; +2,63] |
| 4 | 75,31 | 77,00 | +1,69 [+0,94; +2,43] |
| 5 | 77,60 | 79,12 | +1,52 [+0,53; +2,52] |

- Kết quả rơi vào **dạng B** đã đăng ký trước: mức lợi lớn nhất khi chỉ có 1 frame và giảm dần khi thêm frame, mọi CI đều không chứa 0.
- Theo protocol, dạng B nghĩa là phương pháp làm model **bền hơn khi thiếu frame hoặc frame kém**. N1 (mục 9) cho thấy độ bền này **không đến từ che frame**: bỏ che frame vẫn giữ nguyên accuracy ở N = 1. Ứng viên còn lại là frame dropout và CTC phụ từng frame; E2 sẽ đo.
- Đây **không** phải dạng A (khai thác thông tin bổ sung giữa các frame), vốn là bằng chứng mạnh nhất cho (b). N1 kiểm tra riêng cơ chế đó qua kiểu che bổ sung và **không ủng hộ** nó.
- E3 liên quan trực tiếp đến tính năng nhận ít hơn 5 ảnh của app: với 1–2 ảnh, phương pháp hơn baseline 3–5 điểm.

**Phân tích phụ** (lock-box, epoch cuối, trung bình 4 seed):
- **Theo loại biển:** Brazilian (n = 192) 60,68 → 61,85 (+1,17); Mercosur (n = 808) 81,62 → 83,23 (+1,61). Brazilian quá ít track nên chưa đủ lực thống kê để nói mức lợi khác nhau giữa hai loại biển.
- **Confidence Gap**: phương pháp 0,1135 (0,112–0,116), baseline 0,1060 (0,103–0,108); phương pháp cao hơn ở cả 4 seed. Confidence Gap là độ tin cậy trung bình của dự đoán đúng trừ dự đoán sai, cao hơn là tốt. Như vậy với cấu hình chốt, phương pháp **không** đánh đổi độ tin cậy (MINOR 12; xem thêm mục 8).
- **Val 999 của chính 8 run lock-box** (train 18.001 track): baseline best 76,78–77,08; phương pháp best 76,88–79,48. Seed yếu trên val (42, 2026) khác seed yếu trên lock-box (7, 2026). Tức là đây là nhiễu ngẫu nhiên quanh mức lợi trung bình, không phải một seed kém cố định.
- **Thêm dữ liệu train giúp đáng kể.** Các run train 18.001 track có accuracy trên val 999 thấp hơn các run train 19.001 track khoảng 0,7 điểm (baseline) và 0,9 điểm (phương pháp). Phần chênh này đến từ 1.000 track Scenario-B đã tách ra làm lock-box.

### 4.2 Val 999 (giao thức của số công bố)

Train 19.001 track, val 999. Đây là tập val đã dùng (gián tiếp, qua val 1.999 chứa nó) để chọn cấu hình, nên chỉ có giá trị đối chiếu với số công bố. Chỉ có tầng seed; tầng track đã được đo trên lock-box.

| Seed | Baseline ổn định: cuối / Last-6 / best | Phương pháp, 30 epoch: cuối / Last-6 / best | Phương pháp, 40 epoch: cuối / Last-6 / best |
|---|---|---|---|
| 42 | 76,58 / 75,98 / 77,18 | 78,48 / 78,33 / 79,28 | 79,48 / 79,08 / 79,68 |
| 43 | 77,38 / 77,21 / 77,78 | 78,48 / 77,83 / 78,48 | 78,58 / 78,55 / 78,88 |
| 7 | 77,58 / 76,99 / 77,58 | 78,88 / 78,41 / 78,88 | 78,68 / 78,08 / 78,68 |
| 2026 | 77,48 / 77,14 / 77,78 | 79,08 / 78,91 / 79,18 | 78,68 / 78,75 / 79,18 |
| **Trung bình** | **77,25 ± 0,46 / 76,83 ± 0,58 / 77,58 ± 0,28** | **78,73 ± 0,30 / 78,37 ± 0,44 / 78,95 ± 0,36** | 78,85 ± 0,42 / 78,61 ± 0,42 / 79,10 ± 0,44 |

- **Phương pháp so với baseline:** +1,48 / +1,54 / +1,38 (Welch p = 0,003 / 0,006 / 0,001), ngang mức lợi trên lock-box.
- **So với số công bố 78,70** (MINOR 16): 78,70 nằm trong khoảng dao động giữa các seed của phương pháp (best từ 78,48 đến 79,28; 3/4 seed cao hơn). Không thể kết luận phương pháp vượt số công bố, vì số đó chỉ đến từ một run.
- **Theo loại biển** (epoch tốt nhất, trung bình các seed): Mercosur 84,16 so với 82,81; Brazilian 59,40 so với 58,57. Brazilian chỉ khoảng 200 track trên val 999 (sai số khoảng ±3,5), nên đây chỉ là xu hướng.
- **Train 40 epoch** (LR 2,5e-4): +0,24 Last-6 so với 30 epoch (paired t-test p = 0,46); 2 seed tăng, 2 seed giảm. Val loss cuối tăng (0,21–0,23 so với khoảng 0,20), tức overfit quay lại. Tiêu chí giữ 30 epoch được đặt trước khi chạy, nên **không dùng 40 epoch**.

---

## 5. Ablation (giao thức phát triển, val 1999, 1 seed mỗi cấu hình)

| # | Cấu hình (cộng dồn từ trên xuống) | Best | Last-6 |
|---|---|---|---|
| 9 | Baseline đã ổn định | 77,19 | 76,25 |
| 13 / 25 / 29 | + augment nhất quán, bỏ ChannelShuffle (3 lần chạy) | 77,59 / 77,04 / 77,64 | 76,75 / 76,60 / 77,31 |
| 38 | + sửa degradation synthetic | 77,94 | 77,30 |
| 42 | + CTC phụ từng frame | 77,74 | 77,53 |
| 45 | #42 + frame dropout | 78,69 | 78,28 |
| 46 | #42 + che độc lập từng frame | **78,89** | **78,47** |
| 47 | #42 + input luma | 78,64 | 78,02 |
| 49 | #42 + che frame + frame dropout | 78,74 | 78,02 |
| 50 | #42 + che frame + frame dropout + luma (**cấu hình chốt**) | 78,89 | 78,22 |

- **Kiểm định McNemar** trên cùng 1.999 track:
  - #46 so với baseline #9: đúng thêm 105 track, sai thêm 71 (p = 0,013).
  - So với #29: #46 p = 0,06; #45 p = 0,12; #47 p = 0,14.
- **Cơ chế:** che frame và frame dropout làm giảm overfit. Train loss cao hơn, val loss cuối thấp hơn (0,208–0,210 so với 0,222 của #42). Ở #42, val loss tăng dần từ epoch 24; ở hai cấu hình này thì không. Cải thiện trải khắp các nhóm ảnh. Mức tăng ở nhóm ảnh mờ nhất (60,6 lên khoảng 63%) nằm trong sai số của nhóm (mỗi nhóm khoảng 500 track, sai số chuẩn khoảng 2,2), nên chỉ là xu hướng.
- **Ghép các thành phần không cộng dồn** (#49, #50 ngang #45–#47): tác dụng điều chuẩn đã bão hoà. Chênh giữa #46 và #50 nằm trong nhiễu. Theo ISSUE #3, **#50 được giữ nguyên làm cấu hình đóng băng**: đổi cấu hình lúc này sẽ là thêm một vòng chọn trên val.
- **Bảng này là ablation cộng dồn, 1 seed.** Đóng góp riêng của từng thành phần sẽ được đo bằng E2 (leave-one-out, 3 seed), chỉ để mô tả. Dòng "bỏ che frame" của E2 đã có từ N1: trên lock-box, bỏ che frame khỏi cấu hình chốt không làm giảm accuracy (mục 9). Điều này khớp với nhận xét "ghép không cộng dồn" ở trên.

---

## 6. Super-Resolution đa khung hình

**Hai phép đo định hướng:**
- **Trần khi có ảnh HR:** train và val trên ảnh HR đạt **98,45%** (Mercosur 99,44, Brazilian 94,57). Kiến trúc đọc được biển; giới hạn nằm ở lượng thông tin trong ảnh LR.
- **Giá trị của đa khung hình:** model train với 1 frame đạt 56,53%; với 5 frame đạt 77,19% (+20,66).

**Các biến thể SR đã thử** (đặt trước STN, input 16×64 phóng ×2 lên 32×128, hoặc giữ 32×128 khi scale 1):

| # | Biến thể | Giám sát | Best | Last-6 |
|---|---|---|---|---|
| 9 | (baseline, không SR) | — | 77,19 | 76,25 |
| 19 | SR xếp chồng 5 frame, ×2 | chỉ CTC | 72,09 | 71,50 |
| 20 | SR xếp chồng, scale 1 | chỉ CTC | 74,34 | 73,52 |
| 21 | SR scale 1 + identity skip (khởi tạo zero) | chỉ CTC | 76,89 | 76,64 |
| 22 | EDVR-lite (căn PCD bằng deformable + TSA fusion), ×2 | chỉ CTC | 74,89 | 74,08 |
| 23 | EDVR-lite, scale 1 | chỉ CTC | 72,24 | 71,80 |
| 24 | SR + L1 với HR đăng ký bằng ECC (NCC ≥ 0,8) | pixel (cặp thật) | 75,99 | 75,57 |
| 27 | SR pretrain trên cặp đăng ký + L1 khi train chính | pixel (cặp thật) | 71,04 | diverge ep21 |
| 28 | SR pretrain, bỏ L1 khi train chính | pretrain | 76,99 | 76,36 |
| 39 | (đối chứng của #40: pipeline synthetic native, không SR) | — | 78,34 | 77,36 |
| 40 | SR ×2 + L1 với **cặp synthetic khớp pixel tuyệt đối** | pixel (cặp synthetic) | 76,44 | 76,00 |

**Chẩn đoán các lần thất bại:**
1. Chỉ có CTC thì SR không có lý do để khôi phục ảnh.
2. HR và LR là hai lần chụp khác nhau. Sau khi đăng ký ECC, NCC chỉ khoảng 0,70, nên target bị nhiễu.
3. Ở #27, input được augment còn target thì không, tức bài toán vô nghiệm, và run diverge.

**#40 sửa cả ba điểm trên.** Ảnh HR được augment hình học một lần ở cỡ target, rồi input được suy ra bằng cách hạ về đúng cỡ LR gốc và degrade, nên input và target khớp pixel tuyệt đối. Kết quả:
- SR **học được** tái tạo: tới epoch 30, L1 so với HR thấp hơn nội suy bilinear 9,3%. Ở epoch 3–15 SR còn tệ hơn bilinear tới 21%, vì CTC kéo SR lệch khỏi HR.
- Nhưng accuracy nhận dạng **thấp hơn 1,36 điểm Last-6** so với pipeline giống hệt mà không có SR (#39).

**Kết luận, với phạm vi đúng** (ISSUE #6): với kiến trúc SR và loss pixel chúng tôi thử, cải thiện tái tạo (L1 thấp hơn bilinear 9,3%) không chuyển thành cải thiện nhận dạng (−1,36 điểm Last-6, 1 seed). Điều này phù hợp với giả thuyết rằng trong chế độ này, nút thắt là lượng thông tin chứ không phải độ phân giải. Nhưng chưa đủ để kết luận "SR không giúp LRLPR":
- mới có 1 seed (E6 sẽ thêm 2 seed cho #39 và #40);
- mới thử 1 kiến trúc SR và chỉ loss L1;
- chưa so với SR chuyên cho ảnh chữ (TSRN, TATT, TPGSR).

**Các thử nghiệm khác về căn chỉnh và fusion:**

| Thử nghiệm | Kết quả |
|---|---|
| Một STN dùng chung cho 5 frame (#30) | 76,34 / Last-6 75,64 (thấp hơn) |
| Fusion 2D, giữ chiều cao 2 (#31) | 76,69 / 76,19 (thấp hơn) |
| Fusion Transformer giữa các frame (#44) | thấp hơn 1,1 điểm so với #42 khi so cùng epoch |
| Căn frame bằng phase correlation lúc test | −1,40 điểm |
| **Gộp ở mức đặc trưng so với trung bình dự đoán từng frame, trong cùng một model** | 78,89 so với 76,19 (#46); 78,69 so với 75,24 (#45); 78,64 so với 75,29 (#47) |

Hàng cuối so sánh hai cách đọc output **của cùng một model đã train cho fusion** (ISSUE #7). Nó chưa so với một model train để đọc từng frame rồi bỏ phiếu, nên không dùng để kết luận về cách các đội trong cuộc thi gộp frame.

---

## 7. Các hướng khác không hiệu quả

| Hướng | Run | Kết quả (best / Last-6, val 1999) | Ghi chú |
|---|---|---|---|
| Decode CTC ràng buộc cú pháp biển số (beam 16) | — | +0,05 (#9), +0,00 (#29) | 0/456 lỗi vi phạm cú pháp; ở 456/456 lỗi, chuỗi đúng có xác suất thấp hơn chuỗi model chọn |
| Cosine 60 epoch, LR 1,5e-4, EMA | #41 | 77,24 / 76,45 | Overfit: val loss thấp nhất 0,196 ở ep20, tăng lên 0,247 |
| 40 epoch với cấu hình chốt | #59–#62 | +0,24 Last-6, p = 0,46 | Mục 4.2 |
| Input 64×256 | #43 | 77,84 / 77,33 | Học nhanh hơn, overfit nhiều hơn |
| Input 32×192 | #16 | 77,59 / 76,86 | |
| Degrade synthetic ở đúng cỡ LR gốc | #39 | 78,34 / 77,36 | Thu hẹp domain gap nhưng accuracy gần như không đổi (+0,06 Last-6) |
| Degrade synthetic mạnh | #34 | diverge ep22 | |
| Blur/noise/JPEG cả cho ảnh LR thật (nhẹ / vừa) | #36, #37 | 77,24 / 76,71; 76,94 / 76,71 | |
| Augment hình học "sạch" (bicubic, nhân bản viền) | #48 | 77,44 / 77,16 | Sửa đúng lệch pha đo được nhưng không tăng accuracy |
| Letterbox (giữ tỉ lệ khung) | #26 | 77,14 / 76,66 | |
| Lấy mẫu cân bằng Brazilian/Mercosur | #10 | 76,24 / 75,67 | |
| ResNet34 pretrain ImageNet | #12 | 74,39 / 73,98 | |
| Chưng cất đặc trưng từ teacher HR (MSE) | #17 | 76,29 / 75,80 | |
| EMA trọng số với OneCycle | #29 | EMA 77,59, bản thường 77,64 | |

---

## 8. Phân tích lỗi và dữ liệu

**Loại lỗi** (model #9, 456 lỗi):
- 446 lỗi có đúng 7 ký tự; 57% chỉ sai 1 ký tự.
- Các cặp nhầm hay gặp đều **cùng lớp ký tự**: 8↔6 (46 lần), V↔Y, 3↔1, O↔Q, D↔B, M↔H, A↔B.
- Vị trí 2–3 (chữ cái thứ 2 và 3) sai nhiều nhất.

**Lỗi theo điều kiện ảnh** (model #29, val 1999):
- Nhóm 1/4 tương phản thấp nhất đọc đúng 57,2% và chứa 47,9% số lỗi. Nhóm 1/4 mờ nhất đọc đúng 54,8% và chứa 50,6% số lỗi.
- Tương phản và độ nét (đã chuẩn hoá theo tương phản) gần như không tương quan (Spearman 0,10). Bảng chéo: thấp/mờ 60,7%; thấp/nét 75,3%; cao/mờ 82,8%; cao/nét 92,0%. Hiệu ứng tương phản có trong từng loại biển: Mercosur 75,2 so với 90,6; Brazilian 47,5 so với 66,5.
- Kéo giãn tương phản lúc test làm giảm 1,0 điểm, và nhóm tương phản thấp còn giảm từ 57,2 xuống 54,4. Tức là giới hạn nằm ở tỉ lệ tín hiệu/nhiễu, không phải ở biên độ.

**Domain gap của dữ liệu synthetic** (model #29, 600 track val, tạo synthetic từ chính ảnh HR của các track đó):

| Input | Accuracy |
|---|---|
| LR thật | 78,33% |
| Synthetic theo pipeline gốc (đã sửa noise) | 98,33% |
| HR hạ đúng về cỡ LR thật | 98,00% |
| HR hạ về cỡ LR thật + blur + noise + JPEG | 93,17% |

Nửa synthetic của dữ liệu train dễ hơn ảnh thật nhiều. Thu hẹp khoảng cách này (#39) không làm tăng accuracy.

**Căn chỉnh giữa các frame và vai trò của STN:**
- Độ lệch ngang so với frame giữa: trung vị 1,63 px; 19,4% số cặp lệch quá 4 px (nửa cột đặc trưng).
- STN không giảm được độ lệch này (sau STN: 1,60 px; 17,9%).
- **Phép biến đổi mà STN học được** (đo trên val 999 bằng checkpoint epoch 30 của 4 model phương pháp lock-box): trung bình θ = [[0,782; 0,025; 0,005], [0,062; 0,606; 0,169]], tức phóng ngang 1,28×, phóng dọc 1,65× và dịch xuống. Giữa các ảnh, θ chỉ dao động với độ lệch chuẩn 0,04 (a, d, tx) và 0,11 (ty). Baseline học được phép biến đổi tương tự.
- Nhìn θ thì STN gần như là một phép crop/zoom cố định. Tuy vậy, E4 (mục 9) cho thấy thay STN bằng đúng crop trung bình này làm giảm 1,68 điểm, nên phần dao động nhỏ theo từng ảnh vẫn có ích.

**Confidence Gap** (MINOR 12):
- Trên giao thức phát triển (val 1999, 1 seed): 0,115 (#29), so với 0,093–0,107 cho các model chỉ thêm *một* thành phần điều chuẩn (#45–#47).
- Trên lock-box, với cấu hình chốt (4 seed): phương pháp **0,1135**, cao hơn baseline 0,1060 ở cả 4 seed.
- Vậy mức giảm ở bản trước không lặp lại với cấu hình chốt. Hiệu chỉnh độ tin cậy (ECE) và đường cong risk–coverage cho tính năng "cần kiểm tra" của app (N3) chưa làm.

---

## 9. Thí nghiệm cơ chế: E4 và N1 (đã xong)

Cả hai dùng giao thức lock-box (train 18.001, val 999), cấu hình nền là cấu hình chốt, 3 seed (42, 43, 7). Mỗi nhánh chỉ khác cấu hình chốt ở đúng một điểm. Server chấm lock-box đã trả, nên E4 chạy trên một server mới, cùng loại GPU và cùng phiên bản thư viện, với nhánh tham chiếu là cấu hình chốt **chạy lại trên chính server đó**. N1 chạy trên một server thứ ba; mọi so sánh của N1 là giữa 4 nhánh N1 với nhau. Tổng cộng 21 run: 9 run E4, 12 run N1. Kết quả N1 và E4 **không** làm thay đổi cấu hình chốt; chúng chỉ dùng để mô tả cơ chế.

**N1: các kiểu che frame** (ISSUE #8, đề xuất N1). Ở cả ba kiểu che, mỗi frame bị che với xác suất 0,2 và bề rộng dải che có cùng phân bố. Các kiểu chỉ khác nhau ở quan hệ giữa các frame của cùng một track:

| Nhánh | Cách che |
|---|---|
| Không che | Bỏ che frame |
| Tube | Chọn track với xác suất 0,2, che **cùng một dải** trên cả 5 frame. Vùng bị che mất hẳn khỏi track (giống tube masking của VideoMAE) |
| Độc lập | Mỗi frame tự rút: có bị che không, che ở đâu (như cấu hình chốt) |
| Bổ sung | Chọn track với xác suất 0,2. Mỗi frame che một dải ở một ô riêng (bề ngang chia thành 5 ô), nên mỗi cột còn thấy ở ít nhất 4 frame |

- Cả ba kiểu che đều chỉ dùng dải dọc, vì kiểu bổ sung chỉ định nghĩa được với dải dọc. Cấu hình chốt (độc lập, dải dọc hoặc ngang) là nhánh đối chiếu thứ năm.
- **So sánh chính** là bổ sung − tube: hai nhánh có cùng tỷ lệ track bị che, cùng số frame và cùng kích thước dải; chỉ khác vị trí tương đối của các dải giữa các frame.
- **Cách đọc kết quả, viết trước khi chạy:**
  - *Bổ sung > tube* (CI không chứa 0): che có tác dụng vì buộc fusion lấy phần bị che từ frame khác. Đây là cơ chế đặc thù đa khung hình cho hướng (b).
  - *Tube ≈ độc lập ≈ bổ sung > không che:* che chỉ là điều chuẩn chung, giống Random Erasing.
  - *Tube > bổ sung:* che có tác dụng vì buộc model suy luận từ ngữ cảnh, không phải nhờ thông tin bổ sung giữa các frame.
- **Kết quả** (30/10, run #80–#91; lock-box, 3 seed):

  | Nhánh | Cuối | Last-6 |
  |---|---|---|
  | Không che | 78,60 ± 0,89 | 78,38 ± 0,42 |
  | Tube | 79,10 ± 0,26 | 78,71 ± 0,44 |
  | Độc lập (dọc) | 78,70 ± 0,95 | 78,49 ± 0,97 |
  | Bổ sung | 78,40 ± 0,46 | 78,19 ± 0,05 |

  - **Không rơi vào trường hợp nào đã đăng ký.** So sánh chính bổ sung − tube = −0,52 Last-6, CI 95% theo track [−1,39; +0,36]: không có bổ sung > tube. Không kiểu che nào hơn không che (ước lượng từ −0,19 đến +0,33, mọi CI chứa 0). Val 999 cho cùng hình ảnh.
  - **Kết luận:** trong cấu hình chốt, che frame dải dọc không có tác dụng đo được, dù che kiểu nào. CI theo track loại được tác dụng lớn hơn khoảng +1,2 điểm. **Cơ chế "che buộc fusion lấy phần bị che từ frame khác" không được ủng hộ.**
  - Đối chiếu tham khảo với cấu hình chốt chạy ở E4 (khác server, không có CI): không che 78,38 so với 78,29 Last-6. Vậy bỏ che frame khỏi cấu hình chốt không làm giảm accuracy (dòng "bỏ che frame" của E2). Ở N = 1, không che vẫn đạt 55,75 (cấu hình chốt 55,48): **độ bền khi thiếu frame (E3) không đến từ che frame.**
  - Khớp với ablation phát triển (mục 5): che frame giúp khi đứng một mình (#46), nhưng không cộng thêm khi đã có frame dropout (#49, #50).
  - Hạn chế: server N1 khác server E4 và không ghi lại được môi trường; mọi so sánh giữa 4 nhánh N1 vẫn cùng một server.

**E4: STN so với crop cố định** (ISSUE #10). **Đã xong** (28/10, run #71–#79):

| Nhánh | | Lock-box Last-6 (3 seed) |
|---|---|---|
| STN học được | Cấu hình chốt, chạy lại trên máy mới | **78,29 ± 0,26** |
| Crop cố định | Thay STN bằng phép biến đổi trung bình đã đo ở mục 8, áp giống nhau cho mọi ảnh. Đây là phép thử mạnh hơn "crop 1,3× ở tâm" | 76,60 ± 0,15 |
| Không STN | `--no-stn` | 74,94 ± 0,74 |

- **Cách đọc kết quả, viết trước khi chạy:**
  - *Crop cố định ≈ STN:* chưa thấy tính thích nghi theo từng ảnh của STN đóng góp gì.
  - *Không STN < crop cố định ≈ STN:* phần có ích là phóng to vùng ký tự, không phải căn chỉnh.
  - *Crop cố định < STN:* tính thích nghi theo từng ảnh có ích.
  - *Không STN ≈ STN:* STN không cần thiết.
- **Kết quả: crop cố định < STN**, tức tính thích nghi theo từng ảnh có ích. Crop cố định − STN = −1,68 Last-6, CI 95% theo track [−2,78; −0,59], Welch p = 0,002. Epoch cuối và best cho cùng kết luận.
- Crop cố định cũng hơn không STN: +1,67 [+0,39; +2,96]; theo seed p = 0,055, mới là xu hướng. Vậy lợi ích của STN (+3,35 so với không STN, cùng cỡ +2,90 mà slide của tác giả báo cáo) chia gần đều: **phóng to vùng ký tự khoảng +1,7, điều chỉnh theo từng ảnh khoảng +1,7**. Val 999 cho cùng thứ tự.
- *Phân tích thêm, không đăng ký trước:* phần điều chỉnh theo từng ảnh lớn nhất khi chỉ có 1 frame (+4,2) và giảm dần tới 5 frame (+1,7); phần phóng to gần như không đổi (+1,1 đến +1,5). Giả thuyết: khi có nhiều frame, fusion bù được một phần cho việc từng frame không được căn chỉnh riêng.
- **Giới hạn:**
  - "Thích nghi" gồm hai thứ chưa tách: θ khác nhau cho từng ảnh lúc suy luận, và STN đổi dần từ phép đồng nhất trong lúc train. Phép thử tách được mà không cần train lại: chấm model STN với θ trung bình thay cho θ từng ảnh.
  - Chỉ thử một crop cố định (trung bình θ của chính STN).
- **Lặp lại trên máy mới** (khác máy vật lý nhưng cùng loại GPU V100 và cùng phiên bản PyTorch/CUDA/cuDNN): cấu hình chốt đạt 78,90 (cuối) / 78,29 (Last-6), so với 79,12 / 78,71 trên máy cũ. Mức chênh này nhỏ hơn độ lệch giữa các seed (1,1). Cả 3 seed mới đều cao hơn cả 4 seed baseline cũ. Baseline chưa chạy lại trên máy mới, nên đây chỉ là kiểm tra tham khảo.

**Giới hạn:** với 3 seed mỗi nhánh, tầng seed chỉ phát hiện được chênh lệch lớn (độ lệch chuẩn giữa các seed của cấu hình chốt khoảng 1,1). Chênh dưới khoảng 1 điểm chỉ nên đọc như xu hướng.

---

## 10. Đối chiếu với review ngày 23/10

| Mục | Đã làm | Kết quả / trạng thái |
|---|---|---|
| ISSUE #1: không có tập đánh giá độc lập | Lock-box 1.000 track, protocol đăng ký trước, 8 run, freeze manifest, chấm một lần | Hướng (b): Last-6 +1,64 [+0,78; +2,48] (mục 4.1) |
| ISSUE #2: kiểm định chỉ theo seed | Thêm paired bootstrap theo track, McNemar từng cặp seed, Cohen's d | Theo track: CI không chứa 0 ở cả 3 thước đo. Theo seed: p = 0,06–0,09. Báo cáo cả hai |
| ISSUE #3: ablation cộng dồn 1 seed | Giữ #50 làm cấu hình đóng băng | Dòng "bỏ che frame" (từ N1): không giảm accuracy. Các dòng còn lại của E2 chưa chạy |
| ISSUE #4: tái lập 78,70 | Viết lại claim có phạm vi môi trường; đối chiếu cấu hình Transformer: khớp | E5 (cấu hình gốc ở fp32): lệnh đã sẵn, chưa có kết quả |
| ISSUE #5: lý do bỏ ChannelShuffle | Viết lại (mục 3) | E2 sẽ đo khi đã dùng ảnh xám |
| ISSUE #6: claim SR quá rộng | Hạ phạm vi (mục 6) | E6 (thêm 2 seed cho #39/#40) chưa chạy |
| ISSUE #7: so sánh với bỏ phiếu | Bỏ phần gán cho các đội; ghi rõ "cùng một model" (mục 6) | — |
| ISSUE #8: novelty, đặc thù đa khung hình | E3 xong: dạng B (mục 4.1). N1: 4 nhánh × 3 seed trên lock-box | Bổ sung − tube −0,52 [−1,39; +0,36]; không kiểu che nào hơn không che. Cơ chế đặc thù đa khung hình qua che frame không được ủng hộ (mục 9) |
| ISSUE #9: code chưa review | 14 test tự động (đánh giá tất định; frame dropout/che frame không chạy lúc val; luma; thống kê khớp scipy; loại lock-box; kiểu che; crop cố định). Kiểm tra hồi quy: giống từng bit với code gốc | [Phát bổ sung: đã gửi diff 5 file chưa] |
| ISSUE #10: vai trò STN | Đo θ (mục 8); E4: 3 nhánh × 3 seed trên lock-box | Crop cố định < STN (−1,68 [−2,78; −0,59]); không STN < crop cố định (−1,67 [−2,96; −0,39]) (mục 9) |
| ISSUE #11: tiến độ prototype | — | API `/predict` chưa hoàn thành: thời gian qua ưu tiên huấn luyện và đánh giá mô hình. Sẽ báo cáo tiến độ riêng |
| MINOR 12: Confidence Gap | Đo lại trên lock-box | Phương pháp cao hơn ở 4/4 seed. N3 chưa làm |
| MINOR 13: số batch bị bỏ | Đếm trên 8 run lock-box | 8–11 so với 8–9 trên 16.890 batch; đều là bước gradient không hữu hạn |
| MINOR 14: không tất định | Mục "Khả năng tái lập" (mục 1) | — |
| MINOR 15: bảng cũ trong RUNS.md | Đánh dấu SUPERSEDED | — |
| MINOR 16: so với 78,70 | "78,70 nằm trong khoảng dao động giữa các seed" | — |
| MINOR 17: metric chính | Epoch cuối là thước đo chính trên lock-box | — |
| MINOR 18: "đội hạng 3 không dùng SR" | Đã bỏ | — |
| MINOR 19: "không tăng chi phí" | "Không tăng chi phí suy luận"; báo cáo chi phí huấn luyện (mục 3) | Cần đo lại bằng một run chạy riêng trên V100 |
| Nhãn tập test | Bản đang có không kèm nhãn | Đang xin bản có nhãn |

Đối chiếu với review ngày 10/09 nằm ở Phụ lục C.

---

## 11. Hạn chế và bước tiếp theo

1. **Chưa có kết quả trên tập test chính thức.** Khi có nhãn, 8 bộ checkpoint lock-box sẽ được chấm trên tập test với cùng thước đo và thống kê, như một phân tích xác nhận thêm. Quyết định (b) vẫn dựa trên lock-box như đã đăng ký. Bảng xếp hạng cuộc thi để tham khảo: top 20 từ 76,47% đến 82,13%; các đội dẫn đầu dùng ensemble 3–4 model và/hoặc dữ liệu ngoài.
2. **Tầng seed chưa đạt p < 0,05 trên lock-box**, vì phương pháp dao động giữa các seed lớn hơn baseline. Giảm dao động này là một hướng cải thiện.
3. **E3 cho dạng B**, nên claim chính là "bền hơn khi thiếu frame". **N1 không ủng hộ** claim khai thác thông tin bổ sung giữa các frame, và cho thấy che frame không đóng góp gì đo được trong cấu hình chốt. Bài báo không được trình bày che frame như nguồn gốc của mức lợi.
4. **Số công bố 78,70% chưa tái lập được với cấu hình công bố** trong môi trường của chúng tôi. E5 (fp32) sẽ loại trừ hoặc xác nhận nguyên nhân tràn số fp16.
5. **Thứ tự việc tiếp theo:** E2 (leave-one-out, 3 seed; ưu tiên bỏ frame dropout và bỏ CTC phụ) → E6 (SR thêm 2 seed) và E5 → N3 (hiệu chỉnh độ tin cậy, risk–coverage cho app).
6. **Hướng viết bài:** theo quy tắc đăng ký trước là (b), "điều chuẩn fusion cho nhận dạng biển số đa khung hình độ phân giải thấp", với SR là một mục phân tích có đối chứng.

---

## Phụ lục A — Lệnh của cấu hình chốt

Giao thức của số công bố (val 999):

```bash
python -u train.py -n restran_combo3_999 --seed 42 \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --aux-frame-ctc 0.2 --frame-mask 0.2 --frame-dropout 0.1 --grayscale
```

- Baseline đã ổn định dùng cùng lệnh nhưng chỉ giữ 4 cờ đầu sau tên run: `--epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100`, cùng 2 cờ giao thức.
- Cấu hình công bố là `--lr 5e-4 --grad-clip 5.0`, không có skip guard.
- Giao thức lock-box thêm `--exclude-tracks splits/lockbox_tracks.json --save-last-k 6`. Chấm bằng `python tools/eval_lockbox.py --frames-analysis`.

## Phụ lục B — Các thay đổi trong code

Mọi tính năng mới đều nằm sau cờ dòng lệnh, mặc định tắt. Đã kiểm chứng bằng `tools/regression_check.py`: khi không bật cờ nào, code train giống từng bit với code gốc.

| Cờ | File chính |
|---|---|
| `--grad-clip`, `--skip-grad-norm`, `--skip-abort-ratio` | `train.py`, `src/training/trainer.py` |
| `--consistent-aug`, `--no-channel-shuffle`, `--degrade-strength` | `src/data/dataset.py`, `src/data/transforms.py` |
| `--aux-frame-ctc`, `--frame-dropout` | `src/models/restran.py`, `src/training/trainer.py` |
| `--frame-mask`, `--frame-mask-mode`, `--frame-mask-orient` (N1) | `src/training/trainer.py` |
| `--fixed-stn` (E4) | `src/models/restran.py` |
| `--grayscale` | `src/data/dataset.py` |
| `--exclude-tracks`, `--save-last-k`, `--amp-dtype` | `train.py`, `src/data/dataset.py`, `src/training/trainer.py` |
| Các biến thể SR, EDVR, fusion và các ablation khác | `src/models/sr.py`, `src/models/edvr.py`, `src/models/components.py` |

| Công cụ | Việc |
|---|---|
| `tools/make_lockbox.py` | Lấy tập lock-box tất định; `--verify` kiểm tra file không bị đổi |
| `tools/freeze_manifest.py` | Ghi SHA-256 của code, split, checkpoint và môi trường trước khi chấm |
| `tools/eval_lockbox.py` | Chấm lock-box theo protocol: 3 thước đo, thống kê hai tầng, E3 |
| `tools/eval_arms.py` | Chấm nhiều nhánh (N1, E4) và so với nhánh tham chiếu |
| `tools/regression_check.py` | So code này với code gốc khi tắt mọi cờ |
| `tests/test_eval_determinism.py` | 14 test (ISSUE #9) |

Toàn bộ số liệu trong báo cáo được trích trực tiếp từ `results/*_metrics.csv`, `results/lockbox_eval.json` và các file kết quả theo từng track. Nhật ký đầy đủ: [`documents/RUNS.md`](RUNS.md).

## Phụ lục C — Đối chiếu với review ngày 10/09

| Mục trong review | Đã làm | Kết quả | Kết luận |
|---|---|---|---|
| RCA-01: SR không có giám sát pixel | Thêm giám sát pixel: cặp đăng ký ECC (#24, #27, #28) và cặp synthetic khớp pixel tuyệt đối (#40) | SR tái tạo tốt hơn bilinear 9,3% nhưng accuracy −1,36 (#40) | Giám sát pixel sửa được việc tái tạo nhưng không sửa được nhận dạng (phạm vi: mục 6) |
| RCA-02 / ISSUE #2: distillation không có mask | Chưa thử | — | HR và LR là hai lần chụp khác nhau (NCC khoảng 0,70 sau đăng ký), nên ghép từng time-step không chắc chắn. Để làm sau |
| RCA-03: EDVR làm méo nét chữ | #22, #23 | 74,89 / 72,24 | Khớp nhận định. Căn frame lúc test (phase correlation) cũng làm giảm 1,4 điểm |
| RCA-04 / ISSUE #3 / Đề xuất 1: decode ràng buộc cú pháp | Prefix beam search với mặt nạ cú pháp (2 layout) | +0,00 đến +0,05 | Model đã tự học cú pháp: 0/456 lỗi vi phạm cú pháp. Lỗi thực là nhầm ký tự cùng lớp |
| ISSUE #1: synthetic thiếu JPEG, nhân đôi epoch | Kiểm tra pipeline | JPEG(20–50) vốn đã có; lỗi thật là `GaussNoise` mặc định quá mạnh. Sửa xong: Last-6 76,89 lên 77,30 (#38) | Đã sửa lỗi thật. Phương án lấy mẫu thật/synthetic với p = 0,5 chưa thử |
| §5.1–5.2: lịch huấn luyện | Cosine + warmup 60 epoch, LR 1,5e-4 (#41); 40 epoch (#59–#62) | −0,85 Last-6; +0,24 (không có ý nghĩa) | Train dài hơn làm overfit quay lại; giữ 30 epoch OneCycle |
| Đề xuất 2: CW-TA (prior độ nét Laplacian) | #33 | 76,69 / Last-6 76,16 (so với 77,59 / 76,75) | Không cải thiện |
| Đề xuất 3: CC-CFA (contrastive theo ký tự) | Chưa thử | — | Để làm sau (= N2 trong review 23/10) |
