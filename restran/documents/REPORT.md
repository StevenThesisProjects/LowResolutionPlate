# Báo cáo kết quả: cải thiện ResTran cho nhận dạng biển số độ phân giải thấp đa khung hình (ICPR 2026 LRLPR)

Ngày 19/10/2026. Toàn bộ lệnh chạy, số liệu từng epoch và phân tích chi tiết nằm trong [`documents/RUNS.md`](RUNS.md). Số thứ tự `#N` trong báo cáo này trùng với số run trong nhật ký đó.

---

## Kết quả tốt nhất đạt được

**79,68%** (796/999 track) trên tập val 999 track Scenario-B, theo cùng cách tính với số công bố 78,70% của ResTran + STN. Đây là **một mô hình duy nhất, train một lần**: không ensemble, không dùng dữ liệu tập test, kiến trúc ResTran giữ nguyên.

| Mô hình | Số epoch train | Val 999, epoch tốt nhất | Mercosur / Brazilian |
|---|---|---|---|
| ResTran + STN, số công bố (1 run) | 30 | 78,70 | — |
| **Tốt nhất của chúng tôi** (#59, cấu hình đề xuất, seed 42) | 40 | **79,68** (+0,98) | 84,79 / 60,48 |
| Tốt nhất với cùng 30 epoch như số công bố (#51, seed 42) | 30 | **79,28** (+0,58) | 84,16 / 60,95 |
| Trung bình 4 seed của cấu hình đề xuất (con số đại diện) | 30 | 78,95 ± 0,36 | 84,16 / 59,40 |

- **Cùng cách đo với số công bố**: epoch tốt nhất của một lần chạy, chọn trên chính tập val.
- **Đây là run cao nhất trong 8 run** của cấu hình đề xuất (4 seed × 30 và 40 epoch). Con số đại diện để so sánh khoa học là trung bình 4 seed (mục Tóm tắt): nó tránh việc trình bày một run được chọn lọc như kết quả kỳ vọng.
- Checkpoint: `results/restran_combo3_999_40ep_s42_best.pth` (epoch 39). Kiến trúc 31,08M tham số, chi phí suy luận như ResTran gốc.
- **Ghi chú về số công bố:** 78,70% không ứng với số nguyên track nào trên 999 track (786/999 = 78,68%; 787/999 = 78,78%), nhưng đúng bằng 787/1000. Vậy tập val của số công bố có thể có 1000 track, hoặc slide đã làm tròn 78,68% thành 78,7%. Chênh lệch này nhỏ hơn nhiều so với sai số chuẩn của tập val (khoảng ±1,3).

---

## Tóm tắt

**Mục tiêu.** Cải thiện mô hình ResTran (STN + ResNet34 + Attention Fusion + Transformer + CTC) cho bài toán nhận dạng biển số từ 5 khung hình độ phân giải thấp. Ràng buộc: một mô hình duy nhất, train một lần, không ensemble, không dùng dữ liệu tập test để train.

**Kết quả chính** (đúng giao thức của mốc công bố: train 19.001 track, val 999 track Scenario-B, 30 epoch; trung bình ± độ lệch chuẩn qua 4 seed 42, 43, 7, 2026):

| | Best epoch | TB 6 epoch cuối | Epoch cuối |
|---|---|---|---|
| ResTran + STN, số công bố (1 run) | 78,70 | — | — |
| ResTran + STN, cấu hình công bố, chạy lại | **nổ 2/2 lần** (ep10, ep12) | — | — |
| ResTran + STN, baseline đã ổn định, chạy lại | 77,58 ± 0,28 | 76,83 ± 0,58 | 77,25 ± 0,46 |
| **Phương pháp đề xuất** | **78,95 ± 0,36** | **78,37 ± 0,44** | **78,73 ± 0,30** |
| Chênh so với baseline ổn định | **+1,38** (p = 0,001) | **+1,54** (p = 0,006) | **+1,48** (p = 0,003) |

- Phương pháp hơn baseline chạy lại khoảng **+1,4 đến +1,5 điểm**, có ý nghĩa thống kê (Welch t-test), không chồng lấn: seed tệ nhất của phương pháp vẫn cao hơn run baseline tốt nhất.
- So với số công bố 78,70%: **ngang hoặc nhỉnh hơn** (3/4 seed vượt), nhưng chưa có ý nghĩa thống kê (p = 0,25).
- **Không đổi kiến trúc, không thêm tham số, không tăng chi phí suy luận** (31,08M tham số, như ResTran gốc). Mọi thay đổi nằm ở huấn luyện và xử lý dữ liệu.
- Hướng **Super-Resolution đa khung hình** đã được kiểm tra có đối chứng: khi được giám sát đúng, SR dựng ảnh tốt hơn nội suy bilinear 9,3% nhưng làm accuracy nhận dạng **giảm 1,4 điểm**.
- **Còn thiếu**: đánh giá trên tập test chính thức 3.000 track. Bản LRLPR-26 có nhãn test đang được xin cấp.

---

## 1. Dữ liệu và giao thức đánh giá

**Dữ liệu (LRLPR-26).** 20.000 track train, mỗi track có 5 ảnh LR và 5 ảnh HR.
- Scenario-A: 10.000 track, điều kiện ổn định, ảnh PNG; LR khoảng 34×18 px.
- Scenario-B: 10.000 track, có mưa và ban đêm, ảnh **JPEG lấy mẫu màu 4:2:0**; LR khoảng 47×17 px, HR khoảng 124×44 px.
- Hai loại biển: Mercosur (`LLLNLNN`) và Brazilian cũ (`LLLNNNN`), khoảng 80/20 trong Scenario-B và tập test.

**Giao thức chính** (dùng cho bảng tóm tắt và mục 4):
- Val gồm 999 track Scenario-B, theo cách chia mặc định của code gốc (seed 42, tỉ lệ 0.9). Chạy lại logic này cho ra đúng 999 track đó, nên nhiều khả năng đây là tập val của số công bố 78,70%. Tuy vậy, xem ghi chú ở mục "Kết quả tốt nhất đạt được": 78,70% đúng bằng 787/1000, không ứng với số nguyên track nào trên 999.
- Train 19.001 track. Mỗi track tạo thêm một mẫu synthetic (HR bị degrade), như code gốc.
- Chỉ số: accuracy khớp tuyệt đối theo track. Báo cáo 3 con số:
  - **Best epoch**: như số công bố, chọn trên chính tập val nên hơi lạc quan.
  - **TB6**: trung bình 6 epoch cuối, ít thiên lệch hơn.
  - **Epoch cuối.**
- Mỗi cấu hình chạy 4 seed; so sánh bằng Welch t-test.

**Giao thức phát triển** (dùng cho ablation, mục 5–8):
- Train 18.001 track, val 1.999 track Scenario-B (chứa trọn 999 track ở trên), 1 seed mỗi cấu hình.
- **Độ nhiễu đo được** giữa các lần chạy lặp lại cùng cấu hình: std khoảng 0,33 (best) và 0,37 (TB6). Vì vậy chênh lệch dưới khoảng 1 điểm giữa hai run đơn được coi là nằm trong nhiễu, trừ khi có kiểm định theo cặp (McNemar).

Phần cứng: Tesla V100 32 GB, AMP fp16, batch 64.

---

## 2. Tái lập baseline 78,70%

Cấu hình công bố: LR 5e-4, grad clip 5.0, OneCycleLR, 30 epoch. Khi chạy lại:
- Seed 42 (#1): đạt 61,76% ở epoch 10 rồi **nổ** ở epoch 12.
- Seed 43 (#55): đạt 54,25% ở epoch 9 rồi **nổ** ở epoch 10. Grad norm 49.348; accuracy về 0% và train loss kẹt ở 3,65 tới hết.
- Cả hai đều nổ đúng lúc OneCycle đưa LR lên đỉnh 5e-4, trong khi CTC loss đã xuống khoảng 0,3. Ở các lần nổ đã phân tích chi tiết (#7, #8: lịch 60 epoch), trên 96% số batch bị loại có gradient hữu hạn, tức đây là bất ổn của quá trình tối ưu, không phải tràn số fp16.

**Baseline đã ổn định**: giữ nguyên kiến trúc và pipeline dữ liệu, chỉ đổi 3 điểm tối thiểu:
- LR 3e-4.
- Grad clip 1.0.
- Bỏ qua batch có grad norm trước clip > 100, kèm cơ chế dừng sớm khi quá 20% batch trong epoch bị bỏ.

Kết quả qua 4 seed: best 77,58 ± 0,28.

Khoảng cách 78,70 so với 77,58 cho thấy số công bố có thể là một run thuận lợi, hoặc do khác môi trường chạy. Báo cáo này giữ cả hai con số và so sánh chính với baseline chạy lại cùng giao thức.

---

## 3. Phương pháp đề xuất

Kiến trúc ResTran giữ nguyên. Các thay đổi, theo thứ tự đã đưa vào:

| Thành phần | Nội dung | Lý do |
|---|---|---|
| Ổn định huấn luyện | LR 3e-4, grad clip 1.0, bỏ qua batch có gradient bất thường | Cấu hình gốc nổ (mục 2) |
| Augment nhất quán giữa các frame | Cùng tham số augment cho cả 5 frame của một track; bỏ ChannelShuffle | Augment độc lập phá vỡ quan hệ không gian giữa các frame mà STN và fusion dựa vào. Màu nền là tín hiệu phân biệt loại biển |
| Sửa degradation của nhánh synthetic | `GaussNoise` của albumentations 2.x mặc định làm lệch pixel khoảng 52/255 (phá nét chữ). Đặt lại `std_range=(0.02, 0.08)` | Lỗi cấu hình trong pipeline gốc |
| Aux CTC từng frame (λ = 0,2) | Mỗi frame được đọc riêng bằng chính Transformer + head đã chia sẻ, thêm CTC loss phụ | Giám sát sâu cho đặc trưng từng frame trước fusion; không thêm tham số |
| **Che độc lập từng frame** (p = 0,2) | Mỗi frame, với xác suất 0,2, bị xoá một dải dọc (nửa đến một ký tự) hoặc ngang (1/8–1/4 chiều cao); **mỗi frame tự rút ngẫu nhiên riêng** | Phần bị che ở frame này vẫn thấy được ở frame khác, buộc fusion phải khai thác thông tin bổ sung giữa các frame. (CoarseDropout sẵn có bị lặp y hệt trên cả 5 frame, nên phần bị che mất khỏi cả track) |
| **Frame dropout** (p = 0,1) | Thay đặc trưng của một frame bằng trung bình các frame còn lại, ngay trước fusion | Chống việc fusion dựa quá nhiều vào một frame |
| **Input luma (ảnh xám)** | Chuyển input sang kênh sáng BT.601 ở bước cuối, áp cho train, val và test | Ảnh Scenario-B và test là JPEG 4:2:0: kênh màu chỉ còn ½ độ phân giải mỗi chiều (khoảng 9 px chiều cao trên biển 17 px) nên chủ yếu là nhiễu nén |

Lệnh của cấu hình chốt nằm ở Phụ lục A.

---

## 4. Kết quả chính

Giao thức chính: val 999, train 19.001 track.

| Seed | Baseline ổn định: best / TB6 / cuối | Phương pháp, 30 epoch: best / TB6 / cuối | Phương pháp, 40 epoch: best / TB6 / cuối |
|---|---|---|---|
| 42 | 77,18 / 75,98 / 76,58 | 79,28 / 78,33 / 78,48 | 79,68 / 79,08 / 79,48 |
| 43 | 77,78 / 77,21 / 77,38 | 78,48 / 77,83 / 78,48 | 78,88 / 78,55 / 78,58 |
| 7 | 77,58 / 76,99 / 77,58 | 78,88 / 78,41 / 78,88 | 78,68 / 78,08 / 78,68 |
| 2026 | 77,78 / 77,14 / 77,48 | 79,18 / 78,91 / 79,08 | 79,18 / 78,75 / 78,68 |
| **Trung bình** | **77,58 ± 0,28 / 76,83 ± 0,58 / 77,25 ± 0,46** | **78,95 ± 0,36 / 78,37 ± 0,44 / 78,73 ± 0,30** | 79,10 ± 0,44 / 78,61 ± 0,42 / 78,85 ± 0,42 |

- **Phương pháp so với baseline:** +1,38 / +1,54 / +1,48 (p = 0,001 / 0,006 / 0,003).
- **Theo loại biển** (epoch tốt nhất, trung bình các seed): Mercosur 84,16 so với 82,81; Brazilian 59,40 so với 58,57. Phần tăng chủ yếu đến từ Mercosur. Brazilian chỉ khoảng 200 track trên tập 999 nên sai số khoảng ±3,5.
- **Train 40 epoch** (LR 2,5e-4): +0,24 TB6 so với 30 epoch (paired t-test p = 0,46), 2 seed tăng và 2 seed giảm. Val loss cuối tăng (0,21–0,23 so với khoảng 0,20), tức overfit quay lại. **Không dùng.**
- **So với số công bố 78,70:** best trung bình 78,95 (p = 0,25), TB6 trung bình 78,37 (p = 0,23). Kết luận là ngang mức công bố.

---

## 5. Ablation (giao thức phát triển, val 1999, 1 seed mỗi cấu hình)

| # | Cấu hình (cộng dồn từ trên xuống) | Best | TB6 |
|---|---|---|---|
| 9 | Baseline đã ổn định | 77,19 | 76,25 |
| 13 / 25 / 29 | + augment nhất quán, bỏ ChannelShuffle (3 lần chạy) | 77,59 / 77,04 / 77,64 | 76,75 / 76,60 / 77,31 |
| 38 | + sửa degradation synthetic | 77,94 | 77,30 |
| 42 | + aux CTC từng frame | 77,74 | 77,53 |
| 45 | #42 + frame dropout | 78,69 | 78,28 |
| 46 | #42 + che độc lập từng frame | **78,89** | **78,47** |
| 47 | #42 + input luma | 78,64 | 78,02 |
| 49 | #42 + che frame + frame dropout | 78,74 | 78,02 |
| 50 | #42 + che frame + frame dropout + luma (**cấu hình chốt**) | 78,89 | 78,22 |

- **Kiểm định McNemar** trên cùng 1.999 track:
  - #46 so với baseline #9: đúng thêm 105 track, sai thêm 71 (p = 0,013).
  - So với #29: #46 p = 0,06; #45 p = 0,12; #47 p = 0,14.
- **Cơ chế:** che frame và frame dropout làm giảm overfit. Train loss cao hơn, val loss cuối thấp hơn (0,208–0,210 so với 0,222 của #42). Ở #42, val loss tăng dần từ epoch 24; ở hai cấu hình này thì không. Cải thiện trải đều nhưng nhỉnh hơn ở nhóm ảnh mờ nhất (60,6 lên khoảng 63%).
- **Ghép các thành phần không cộng dồn** (#49, #50 ngang #45–#47): tác dụng điều chuẩn đã bão hoà. Cấu hình chốt giữ cả ba vì #50 cho val loss thấp và Brazilian tốt nhất (58,52), nhưng chênh giữa #46 và #50 nằm trong nhiễu.

---

## 6. Super-Resolution đa khung hình

**Hai phép đo định hướng:**
- **Trần khi có ảnh HR:** train và val trên ảnh HR đạt **98,45%** (Mercosur 99,44, Brazilian 94,57). Kiến trúc đọc được biển; giới hạn nằm ở lượng thông tin trong ảnh LR.
- **Giá trị của đa khung hình:** 1 frame đạt 56,53%; 5 frame đạt 77,19% (+20,66).

**Các biến thể SR đã thử** (đặt trước STN, input 16×64 phóng ×2 lên 32×128, hoặc giữ 32×128 khi scale 1):

| # | Biến thể | Giám sát | Best | TB6 |
|---|---|---|---|---|
| 9 | (baseline, không SR) | — | 77,19 | 76,25 |
| 19 | SR xếp chồng 5 frame, ×2 | chỉ CTC | 72,09 | 71,50 |
| 20 | SR xếp chồng, scale 1 | chỉ CTC | 74,34 | 73,52 |
| 21 | SR scale 1 + identity skip (khởi tạo zero) | chỉ CTC | 76,89 | 76,64 |
| 22 | EDVR-lite (căn PCD bằng deformable + TSA fusion), ×2 | chỉ CTC | 74,89 | 74,08 |
| 23 | EDVR-lite, scale 1 | chỉ CTC | 72,24 | 71,80 |
| 24 | SR + L1 với HR đăng ký bằng ECC (NCC ≥ 0,8) | pixel (cặp thật) | 75,99 | 75,57 |
| 27 | SR pretrain trên cặp đăng ký + L1 khi train chính | pixel (cặp thật) | 71,04 | nổ ep21 |
| 28 | SR pretrain, bỏ L1 khi train chính | pretrain | 76,99 | 76,36 |
| 39 | (đối chứng của #40: pipeline synthetic native, không SR) | — | 78,34 | 77,36 |
| 40 | SR ×2 + L1 với **cặp synthetic khớp pixel tuyệt đối** | pixel (cặp synthetic) | 76,44 | 76,00 |

**Chẩn đoán các lần thất bại:**
1. Chỉ có CTC thì SR không có lý do để khôi phục ảnh.
2. HR và LR là hai lần chụp khác nhau. Sau khi đăng ký ECC, NCC chỉ khoảng 0,70, nên target bị nhiễu.
3. Ở #27, input được augment còn target thì không, tức bài toán vô nghiệm, và run nổ.

**#40 sửa cả ba điểm trên.** Ảnh HR được augment hình học một lần ở cỡ target, rồi input được suy ra bằng cách hạ về đúng cỡ LR gốc và degrade, nên input và target khớp pixel tuyệt đối. Kết quả:
- SR **học được** tái tạo: tới epoch 30, L1 so với HR thấp hơn nội suy bilinear 9,3%. Ở epoch 3–15 SR còn tệ hơn bilinear tới 21%, vì CTC kéo SR lệch khỏi HR.
- Nhưng accuracy nhận dạng **thấp hơn 1,36 điểm TB6** so với pipeline giống hệt mà không có SR (#39).

**Kết luận:** ảnh tái tạo trung thực hơn ở mức pixel không chuyển thành khả năng phân biệt ký tự tốt hơn. Với 5 frame cỡ khoảng 47×17 px, nút thắt là lượng thông tin và tỉ lệ tín hiệu/nhiễu, không phải độ phân giải. Đòn bẩy đa khung hình hiệu quả nhất tìm được là **điều chuẩn quá trình fusion** (mục 5), không phải khôi phục ảnh ở mức pixel. Điều này phù hợp với bài tổng kết cuộc thi: đội hạng 3 (80,17%) không dùng SR; điểm chung của các đội dẫn đầu là khai thác tốt 5 frame.

**Các thử nghiệm khác về căn chỉnh và fusion:**

| Thử nghiệm | Kết quả |
|---|---|
| Một STN dùng chung cho 5 frame (#30) | 76,34 / TB6 75,64 (thấp hơn) |
| Fusion 2D, giữ chiều cao 2 (#31) | 76,69 / 76,19 (thấp hơn) |
| Fusion Transformer giữa các frame (#44) | thấp hơn 1,1 điểm so với #42 khi so cùng epoch |
| Căn frame bằng phase correlation lúc test | −1,40 điểm |
| **Gộp ở mức đặc trưng so với trung bình dự đoán từng frame** (cùng model) | 78,89 so với 76,19 (#46); 78,69 so với 75,24 (#45); 78,64 so với 75,29 (#47) |

Hàng cuối cho thấy gộp ở mức đặc trưng hơn kiểu "bỏ phiếu theo frame" (cách đội hạng 1 và hạng 3 dùng) khoảng 2,7–3,5 điểm.

---

## 7. Các hướng khác không hiệu quả

| Hướng | Run | Kết quả (best / TB6, val 1999) | Ghi chú |
|---|---|---|---|
| Decode CTC ràng buộc cú pháp biển số (beam 16) | — | +0,05 (#9), +0,00 (#29) | 0/456 lỗi vi phạm cú pháp; ở 456/456 lỗi, chuỗi đúng có xác suất thấp hơn chuỗi model chọn |
| Cosine 60 epoch, LR 1,5e-4, EMA | #41 | 77,24 / 76,45 | Overfit: val loss thấp nhất 0,196 ở ep20, tăng lên 0,247 |
| 40 epoch với cấu hình chốt | #59–#62 | +0,24 TB6, p = 0,46 | Mục 4 |
| Input 64×256 | #43 | 77,84 / 77,33 | Học nhanh hơn, overfit nhiều hơn |
| Input 32×192 | #16 | 77,59 / 76,86 | |
| Degrade synthetic ở đúng cỡ LR gốc | #39 | 78,34 / 77,36 | Thu hẹp domain gap nhưng accuracy gần như không đổi (+0,06 TB6) |
| Degrade synthetic mạnh | #34 | nổ ep22 | |
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

**Căn chỉnh giữa các frame:**
- Độ lệch ngang so với frame giữa: trung vị 1,63 px; 19,4% số cặp lệch quá 4 px (nửa cột đặc trưng).
- STN không giảm được độ lệch này (sau STN: 1,60 px; 17,9%). STN chủ yếu học zoom vào khoảng 1,3 lần.

**Confidence Gap** (tiêu chí phụ của cuộc thi, val 1999): 0,115 (#29) so với 0,093–0,107 cho các model đã điều chuẩn (#45–#47). Điều chuẩn làm khoảng cách độ tin cậy giữa dự đoán đúng và sai giảm nhẹ.

---

## 9. Đối chiếu với review của thầy

| Mục trong review | Đã làm | Kết quả | Kết luận |
|---|---|---|---|
| RCA-01: SR không có giám sát pixel | Thêm giám sát pixel: cặp đăng ký ECC (#24, #27, #28) và cặp synthetic khớp pixel tuyệt đối (#40) | SR tái tạo tốt hơn bilinear 9,3% nhưng accuracy −1,36 (#40) | Giám sát pixel sửa được việc tái tạo nhưng không sửa được nhận dạng. Phù hợp với C5: khôi phục ảnh đẹp khác tối ưu khả năng phân biệt ký tự |
| RCA-02 / ISSUE #2: distillation không có mask | Chưa thử | — | HR và LR là hai lần chụp khác nhau (NCC khoảng 0,70 sau đăng ký), nên ghép từng time-step không chắc chắn. Đề xuất để làm sau |
| RCA-03: EDVR làm méo nét chữ | #22, #23 | 74,89 / 72,24 | Khớp nhận định. Căn frame lúc test (phase correlation) cũng làm giảm 1,4 điểm |
| RCA-04 / ISSUE #3 / Đề xuất 1: decode ràng buộc cú pháp | Đã cài prefix beam search với mặt nạ cú pháp (2 layout) | +0,00 đến +0,05 | Model đã tự học cú pháp: 0/456 lỗi vi phạm cú pháp, và ở mọi lỗi chuỗi đúng có xác suất thấp hơn. Lỗi thực là nhầm ký tự cùng lớp (8↔6, V↔Y, 3↔1) |
| ISSUE #1: synthetic thiếu JPEG, nhân đôi epoch | Kiểm tra pipeline | JPEG(20–50) vốn đã có; lỗi thật là `GaussNoise` mặc định quá mạnh. Sửa xong: TB6 76,89 lên 77,30 (#38). Degrade ở cỡ LR gốc (#39): ≈ | Đã sửa lỗi thật. Phương án lấy mẫu thật/synthetic với p = 0,5 chưa thử |
| §5.1–5.2: lịch huấn luyện | Cosine + warmup 60 epoch, LR 1,5e-4 (#41); 40 epoch (#59–#62) | −0,85 TB6; +0,24 (không có ý nghĩa) | Train dài hơn làm overfit quay lại; giữ 30 epoch OneCycle. Grad clip 0,5 chưa thử (dùng 1,0) |
| Đề xuất 2: CW-TA (prior độ nét Laplacian) | #33 | 76,69 / TB6 76,16 (so với 77,59 / 76,75) | Không cải thiện |
| Đề xuất 3: CC-CFA (contrastive theo ký tự) | Chưa thử | — | Đề xuất để làm sau |

---

## 10. Hạn chế và bước tiếp theo

1. **Chưa có kết quả trên tập test.** Mọi con số đều trên tập val, mà tập val đã dùng để chọn cấu hình. Bản LRLPR-26 chính thức có nhãn cho 3.000 track test (bộ blind test của cuộc thi) đang được xin cấp. Khi có nhãn, toàn bộ checkpoint 4 seed của baseline và phương pháp sẽ được chấm lại. Kết quả đó cũng cho phép đặt mô hình đơn của mình cạnh bảng xếp hạng cuộc thi (top 20 từ 76,47% đến 82,13%; các đội dẫn đầu dùng ensemble 3–4 model và/hoặc dữ liệu ngoài).
2. **Số công bố 78,70% không tái lập được với cấu hình công bố** (nổ 2/2 lần), nên so sánh chính là với baseline đã ổn định.
3. **Ablation chỉ có 1 seed** mỗi cấu hình. Độ nhiễu đã đo (std khoảng 0,37) và kèm kiểm định McNemar ở những so sánh quan trọng.
4. **Chưa thử:** distillation có mask / contrastive (RCA-02, Đề xuất 3), grad clip 0,5, lấy mẫu thật/synthetic với p = 0,5.
5. **Xin ý kiến thầy về hướng viết bài.** Đóng góp chủ yếu mang tính thực nghiệm. Có hai cách đặt trọng tâm:
   - (a) "SR có giúp nhận dạng biển số đa khung hình không?": nghiên cứu thực nghiệm, kết quả âm tính có đối chứng ở mục 6 là trung tâm.
   - (b) "Điều chuẩn dành cho đa khung hình" (che độc lập từng frame, frame dropout, aux CTC từng frame): mục 4–5 là trung tâm, SR là một ablation.

---

## Phụ lục A — Lệnh của cấu hình chốt

```bash
python -u train.py -n restran_combo3_999 --seed 42 \
    --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --split-ratio 0.9 --val-split-file splits/val_tracks.json \
    --consistent-aug --no-channel-shuffle --degrade-strength calibrated \
    --aux-frame-ctc 0.2 --frame-mask 0.2 --frame-dropout 0.1 --grayscale
```

Baseline đã ổn định dùng cùng lệnh nhưng chỉ giữ 4 cờ đầu sau tên run: `--epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100` (cùng 2 cờ giao thức). Cấu hình công bố là `--lr 5e-4 --grad-clip 5.0`, không có skip guard.

## Phụ lục B — Các thay đổi trong code

Mọi tính năng mới đều nằm sau cờ dòng lệnh, mặc định tắt, nên chạy không cờ thì cho đúng hành vi của code gốc.

| Cờ | File chính |
|---|---|
| `--grad-clip`, `--skip-grad-norm`, `--skip-abort-ratio` | `train.py`, `src/training/trainer.py` |
| `--consistent-aug`, `--no-channel-shuffle`, `--degrade-strength` | `src/data/dataset.py`, `src/data/transforms.py` |
| `--aux-frame-ctc`, `--frame-dropout` | `src/models/restran.py`, `src/training/trainer.py` |
| `--frame-mask` | `src/training/trainer.py` |
| `--grayscale` | `src/data/dataset.py` |
| Các biến thể SR, EDVR, fusion và các ablation khác | `src/models/sr.py`, `src/models/edvr.py`, `src/models/components.py` |

Toàn bộ số liệu trong báo cáo được trích trực tiếp từ các file `results/*_metrics.csv`. Nhật ký đầy đủ: [`documents/RUNS.md`](RUNS.md).
