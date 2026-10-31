# Báo cáo thí nghiệm cơ chế: vai trò của STN (E4) và các kiểu che frame (N1)

Ngày 31/10/2026. Trả lời ISSUE #10 (E4/N4) và ISSUE #8 (N1) trong review ngày 23/10.
- Cách đọc kết quả được viết trước khi chạy: [`documents/RUNS.md`](RUNS.md), mục "Thay đổi code 25/10/2026" và "N1 + E4: lệnh chạy".
- Số liệu từng run: [`documents/RUNS.md`](RUNS.md), run #71–#79 (E4) và #80–#91 (N1).
- Kết quả lock-box (ISSUE #1): [`documents/REPORT_lockbox.md`](REPORT_lockbox.md). Báo cáo tổng đã cập nhật: [`documents/REPORT.md`](REPORT.md), mục 9.

---

## Tóm tắt

- **E4 (ISSUE #10): STN cần thiết, và không chỉ là một phép zoom cố định.**
  - Bỏ STN làm giảm 3,35 điểm Last-6 trên lock-box. Con số này cùng cỡ với +2,90 trong slide của tác giả.
  - Thay STN bằng đúng phép biến đổi trung bình mà nó học được (crop cố định) vẫn giảm 1,68 điểm, CI 95% theo track [−2,78; −0,59].
  - Lợi ích của STN chia gần đều làm hai phần: **phóng to vùng ký tự khoảng +1,7** và **điều chỉnh theo từng ảnh khoảng +1,7**.
- **N1 (ISSUE #8): trong cấu hình chốt, che frame không có tác dụng đo được, dù che kiểu nào.**
  - So sánh chính, bổ sung − tube = −0,52 [−1,39; +0,36]: không có chuyện kiểu bổ sung hơn tube.
  - Không kiểu che nào hơn không che (ước lượng từ −0,19 đến +0,33, mọi CI chứa 0).
  - Vì vậy cơ chế "che độc lập buộc fusion khai thác thông tin bổ sung giữa các frame" **không được ủng hộ**.
- **Hệ quả cho bài báo:**
  - Cấu hình chốt giữ nguyên, theo đúng đăng ký trước.
  - Claim chính vẫn là E3 dạng B: phương pháp làm model bền hơn khi thiếu frame. Thêm vào đó là phân tích vai trò STN.
  - Độ bền khi chỉ có 1 frame không đến từ che frame. E2 (leave-one-out) cần ưu tiên hai thành phần còn lại là frame dropout và CTC phụ.

---

## 1. Cách làm

| Mục | Nội dung |
|---|---|
| Giao thức | Giống lock-box: train 18.001 track (đã loại 1.000 track lock-box), val 999 để theo dõi, chấm trên lock-box. 30 epoch, lưu 6 epoch cuối và best. Cấu hình nền là cấu hình chốt (#50); mỗi nhánh chỉ khác nó đúng một điểm |
| Seed | 3 seed mỗi nhánh (42, 43, 7) |
| Thống kê | Như lock-box: CI 95% bootstrap theo track (10.000 lần) và Welch theo seed. Thước đo chính là Last-6 |
| Vai trò | Kết quả E4 và N1 **không** làm thay đổi cấu hình chốt; chúng chỉ dùng để mô tả cơ chế |
| Cách đọc kết quả | Viết ngày 25/10, trước khi chạy run nào (trích ở mục 2 và 3) |

**Server.**
- Server chấm lock-box đã trả trước khi chạy E4. Vì vậy E4 chạy trên một server mới, có cùng GPU (Tesla V100-SXM2 32 GB), Python 3.10.13, PyTorch 2.14.1+cu126, CUDA 12.6 và cuDNN 9.10.2.
- Để các so sánh không lẫn khác biệt về máy, cấu hình chốt được **chạy lại 3 seed trên chính server đó** làm nhánh tham chiếu.
- N1 chạy trên một server thứ ba. Mọi so sánh của N1 là giữa 4 nhánh N1 với nhau, chạy trên cùng server đó.

**Thiết kế E4.**

| Nhánh | Cách làm | Số tham số |
|---|---|---|
| STN học được | Cấu hình chốt | 31.077.676 |
| Crop cố định | Thay STN bằng phép biến đổi trung bình mà STN đã học, áp giống nhau cho mọi ảnh: θ = [[0,782; 0,025; 0,005], [0,062; 0,606; 0,169]], tức phóng ngang 1,28×, dọc 1,65×, dịch xuống một chút | 30.793.702 |
| Không STN | `--no-stn` | 30.793.702 |

- θ được đo trên val 999 bằng 4 model phương pháp lock-box, không dùng tập lock-box.
- Đây là phép thử mạnh hơn "crop 1,3× ở tâm" trong review: nếu STN chỉ học zoom, crop bằng đúng trung bình của nó phải cho kết quả ngang STN.

**Thiết kế N1.** Ở cả ba kiểu che, xác suất che và bề rộng dải che có cùng phân bố; chỉ khác quan hệ giữa các frame trong một track.

| Nhánh | Cách che |
|---|---|
| Không che | Bỏ che frame |
| Tube | Chọn track với xác suất 0,2, che **cùng một dải** trên cả 5 frame (giống tube masking của VideoMAE) |
| Độc lập | Mỗi frame tự rút: có bị che không, che ở đâu (như cấu hình chốt) |
| Bổ sung | Chọn track với xác suất 0,2. Mỗi frame che một dải ở một ô riêng (bề ngang chia 5 ô), nên mỗi cột còn thấy ở ít nhất 4 frame |

- Cả ba kiểu che chỉ dùng dải dọc, vì kiểu bổ sung chỉ định nghĩa được với dải dọc.
- **So sánh chính là bổ sung − tube.** Hai nhánh có cùng tỷ lệ track bị che, cùng số frame và cùng kích thước dải, chỉ khác vị trí tương đối của các dải giữa các frame.
- Có test tự động cho từng kiểu che: tube giống nhau qua các frame; dải bổ sung không chồng nhau; tỷ lệ che biên 0,2 ± 0,01; kiểu "độc lập" giống từng bit code cũ.

---

## 2. E4: STN so với crop cố định (ISSUE #10)

**Cách đọc kết quả, viết trước khi chạy:**
- *Crop cố định ≈ STN:* chưa thấy tính thích nghi theo từng ảnh đóng góp gì; STN thực chất là một phép zoom cố định.
- *Crop cố định < STN:* tính thích nghi theo từng ảnh có ích.
- *Không STN < crop cố định ≈ STN:* phần có ích là phóng to vùng ký tự, không phải căn chỉnh.
- *Không STN ≈ STN:* STN không cần thiết.

**Theo nhánh** (accuracy % trên 1.000 track lock-box, trung bình ± std 3 seed):

| Nhánh | Epoch cuối | **Last-6** | Best |
|---|---|---|---|
| STN học được | 78,90 ± 0,35 | **78,29 ± 0,26** | 78,87 ± 0,21 |
| Crop cố định | 77,20 ± 0,26 | **76,60 ± 0,15** | 77,20 ± 0,10 |
| Không STN | 75,67 ± 1,00 | **74,94 ± 0,74** | 75,43 ± 1,40 |

**So sánh** (CI 95% bootstrap theo track; Welch theo seed):

| So sánh | Last-6 | Epoch cuối | Best |
|---|---|---|---|
| **Crop cố định − STN** | **−1,68** [−2,78; −0,59], p = 0,002 | −1,70 [−2,97; −0,43], p = 0,003 | −1,67 [−2,93; −0,43], p = 0,001 |
| Không STN − STN | −3,35 [−4,69; −2,02], p = 0,010 | −3,23 [−4,67; −1,80], p = 0,021 | −3,43 [−4,93; −1,97], p = 0,049 |
| Crop cố định − không STN | +1,67 [+0,39; +2,96], p = 0,055 | +1,53 [+0,10; +2,93], p = 0,11 | +1,77 [+0,37; +3,17], p = 0,16 |

Val 999 cho cùng thứ tự (Last-6): STN 77,67; crop cố định 74,64; không STN 73,01.

**Kết quả: "crop cố định < STN"**, tức tính thích nghi theo từng ảnh có ích.
- CI theo track không chứa 0 ở cả 3 thước đo, và theo seed p ≤ 0,003. Chênh 1,7 điểm, không nằm sát ranh giới.
- Crop cố định cũng hơn không STN (+1,67 Last-6; CI theo track không chứa 0, nhưng theo seed p = 0,055 nên mới là xu hướng).
- Vậy lợi ích của STN (+3,35) chia gần đều: **phóng to vùng ký tự khoảng +1,7**, **điều chỉnh theo từng ảnh khoảng +1,7**.

**Theo số frame** (epoch cuối; *phân tích thêm, không đăng ký trước*):

| N | STN | Crop cố định | Không STN | STN − crop [CI 95%] | Crop − không STN [CI 95%] |
|---|---|---|---|---|---|
| 1 | 55,48 | 51,28 | 50,23 | +4,20 [+3,39; +5,01] | +1,05 [+0,23; +1,87] |
| 2 | 68,86 | 65,52 | 64,40 | +3,34 [+2,55; +4,12] | +1,12 [+0,25; +2,01] |
| 3 | 74,33 | 71,46 | 70,30 | +2,87 [+2,00; +3,76] | +1,17 [+0,17; +2,14] |
| 4 | 77,17 | 74,84 | 73,57 | +2,33 [+1,34; +3,32] | +1,27 [+0,14; +2,43] |
| 5 | 78,90 | 77,20 | 75,67 | +1,70 [+0,43; +2,97] | +1,53 [+0,10; +2,93] |

- Phần điều chỉnh theo từng ảnh lớn nhất khi chỉ có 1 frame (+4,2) và giảm dần tới 5 frame (+1,7). Phần phóng to gần như không đổi.
- Giả thuyết: khi có nhiều frame, fusion bù được một phần việc từng frame không được điều chỉnh riêng. Đây là quan sát sau khi thấy số liệu, chỉ nên viết như giả thuyết.

**Giới hạn của E4:**
- "Thích nghi" gộp hai thứ chưa tách được: (i) θ khác nhau cho từng ảnh lúc suy luận; (ii) STN khởi tạo bằng phép đồng nhất và đổi dần trong lúc train, còn crop cố định áp mức zoom cuối ngay từ epoch đầu. Có thể tách bằng một phép thử không cần train lại: chấm model STN với θ trung bình thay cho θ từng ảnh (E4b, chưa làm).
- Chỉ thử một crop cố định. Kết luận đúng là "STN hơn crop cố định bằng trung bình của chính nó".

**Lặp lại cấu hình chốt trên máy khác** (nhánh STN của E4, đặt cạnh kết quả lock-box; không có CI):

| | Server lock-box (4 seed) | Server E4 (3 seed) |
|---|---|---|
| Epoch cuối | 79,12 ± 1,14 | 78,90 ± 0,35 |
| Last-6 | 78,71 ± 1,15 | 78,29 ± 0,26 |
| N = 1 (E3) | 55,59 | 55,48 |

Chênh 0,2–0,4 điểm, nhỏ hơn độ lệch giữa các seed. Cả 3 seed mới (Last-6 thấp nhất 78,05) đều cao hơn cả 4 seed baseline lock-box (Last-6 cao nhất 77,40). Baseline không chạy lại trên server E4, nên đây chỉ là kiểm tra tham khảo.

---

## 3. N1: các kiểu che frame (ISSUE #8)

**Cách đọc kết quả, viết trước khi chạy:**
- *Bổ sung > tube* (CI không chứa 0): che có tác dụng vì buộc fusion lấy phần bị che từ frame khác. Đây là cơ chế đặc thù đa khung hình cho hướng (b).
- *Tube ≈ độc lập ≈ bổ sung > không che:* che chỉ là điều chuẩn chung, giống Random Erasing.
- *Tube > bổ sung:* che có tác dụng vì buộc model suy luận từ ngữ cảnh, không phải nhờ thông tin bổ sung giữa các frame.

**Theo nhánh** (lock-box, trung bình ± std 3 seed):

| Nhánh | Epoch cuối | **Last-6** | Best |
|---|---|---|---|
| Không che | 78,60 ± 0,89 | **78,38 ± 0,42** | 78,23 ± 0,57 |
| Tube | 79,10 ± 0,26 | **78,71 ± 0,44** | 78,90 ± 1,04 |
| Độc lập (dải dọc) | 78,70 ± 0,95 | **78,49 ± 0,97** | 78,67 ± 1,10 |
| Bổ sung | 78,40 ± 0,46 | **78,19 ± 0,05** | 77,93 ± 0,38 |

**So sánh** (Last-6; CI 95% bootstrap theo track; Welch theo seed):

| So sánh | Last-6 | Epoch cuối |
|---|---|---|
| **Bổ sung − tube** (so sánh chính) | **−0,52** [−1,39; +0,36], p = 0,18 | −0,70 [−1,73; +0,33], p = 0,10 |
| Tube − không che | +0,33 [−0,51; +1,16], p = 0,40 | +0,50 [−0,43; +1,47] |
| Độc lập − không che | +0,11 [−0,75; +0,93], p = 0,87 | +0,10 [−0,90; +1,10] |
| Bổ sung − không che | −0,19 [−0,96; +0,57], p = 0,52 | −0,20 [−1,20; +0,77] |
| Bổ sung − độc lập | −0,30 [−1,17; +0,54], p = 0,65 | −0,30 [−1,37; +0,77] |
| Độc lập − tube | −0,22 [−1,06; +0,63], p = 0,75 | −0,40 [−1,43; +0,63] |

- Val 999 cho cùng hình ảnh (Last-6): không che 78,02; tube 78,26; độc lập 77,97; bổ sung 77,70. Mọi CI đều chứa 0.
- Theo số frame, không nhánh nào khác nhánh nào quá 0,9 điểm ở bất kỳ N nào. Ở N = 1: 55,75 / 55,09 / 55,34 / 55,35 (không che / tube / độc lập / bổ sung).

**Kết quả: không rơi vào trường hợp nào đã đăng ký.**
- *Bổ sung > tube:* không. Ước lượng còn ngược chiều và CI chứa 0.
- *Tube ≈ độc lập ≈ bổ sung > không che:* vế đầu đúng, nhưng không kiểu che nào hơn không che.
- *Tube > bổ sung:* ước lượng cùng chiều nhưng CI chứa 0; dưới ngưỡng 1 điểm mà 3 seed phát hiện được.
- **Kết luận:** trong cấu hình chốt, che frame dải dọc không có tác dụng đo được. CI theo track loại được tác dụng lớn hơn khoảng +1,2 điểm. Kết quả không nằm sát ranh giới, nên không chạy thêm seed.

**Đặt cạnh cấu hình chốt** (nhánh STN của E4, khác server, không có CI):

| | Epoch cuối | Last-6 | N = 1 |
|---|---|---|---|
| Cấu hình chốt (server E4) | 78,90 | 78,29 | 55,48 |
| Không che (N1) | 78,60 | 78,38 | 55,75 |
| Độc lập, chỉ dải dọc (N1) | 78,70 | 78,49 | 55,34 |

- **Bỏ che frame khỏi cấu hình chốt không làm giảm accuracy.** Đây chính là dòng "bỏ che frame" của E2: đóng góp ước lượng gần 0.
- **Độ bền khi chỉ có 1 frame (E3, +5,0 so với baseline) không đến từ che frame:** không che vẫn đạt 55,75 ở N = 1.
- Kết quả khớp với ablation cộng dồn trên giao thức phát triển (REPORT mục 5): che frame giúp khi đứng một mình (#46), nhưng không cộng thêm khi đã có frame dropout (#49, #50 ngang #45). Khi đã có frame dropout, che frame thành thừa.

---

## 4. Hệ quả cho bài báo

1. **ISSUE #10 được trả lời theo hướng "STN cần thiết".** Giữ tên mô hình. Phân tích "zoom + thích nghi, mỗi phần khoảng +1,7" và "thích nghi quan trọng nhất khi ít frame" là một mục insight viết được.
2. **ISSUE #8: N1 không cung cấp bằng chứng cho cơ chế đặc thù đa khung hình qua che frame.**
   - Story arc trong review (mục 7.3: "đòn bẩy là buộc fusion khai thác phần bổ sung giữa các frame") không được dữ liệu ủng hộ ở phần che frame.
   - Bài báo sẽ **không** trình bày che frame như nguồn gốc của mức lợi. Che frame vẫn nằm trong cấu hình chốt (đã đóng băng trước lock-box), và được báo cáo là không có đóng góp đo được.
   - Review đã dự liệu trường hợp này (mục 8): nếu các kiểu che ngang nhau thì bài vẫn có một phân tích cơ chế có đối chứng.
3. **Claim chính đề xuất:** phương pháp làm fusion đa khung hình **bền hơn khi thiếu frame hoặc frame kém** (E3 dạng B, +5,0 điểm ở N = 1). Kèm theo là phân tích STN (E4) và kết quả âm có đối chứng của N1.
4. **E2 trở nên quan trọng hơn.** Cần biết +1,6 điểm và độ bền ở N = 1 đến từ thành phần nào. Ứng viên chính là frame dropout (thay đặc trưng một frame bằng trung bình các frame còn lại, nên vẫn là điều chuẩn ở mức fusion) và CTC phụ từng frame.

---

## 5. Kiểm chứng và thay đổi so với kế hoạch

**Kiểm chứng:**
- **Đủ dữ liệu:** 21/21 run đủ 30 epoch và 7 checkpoint. `eval_arms.py` nhận đúng kiến trúc từ checkpoint (STN / crop cố định / không STN).
- **Cấu hình đúng:** so toàn bộ khối cấu hình trong 19 log có được, mỗi nhánh chỉ khác nhánh tham chiếu đúng ở cờ đã định. Cả 19 log đều có dòng "loại 1.000 track lock-box khỏi train". Hai run thiếu log có file summary ghi `Train samples: 36002`, tức đã loại lock-box.
- **Chấm lại khớp lúc train:** accuracy val 999 ở epoch 30 ghi lúc train (log hoặc file metrics) khớp số chấm lại ở cả 21 run.
- **Batch bị bỏ:** 6–10 batch mỗi run do gradient không hữu hạn của fp16. Riêng `tube_s7` có thêm 4 batch vượt ngưỡng grad norm 100. Đây là lần đầu ngưỡng này kích hoạt trong các run lock-box.

**Thay đổi so với kế hoạch:**
- **N1 chạy trên server khác E4** (kế hoạch là cả 21 run trên cùng server). Vì vậy mất so sánh chính thức giữa "độc lập, chỉ dải dọc" và cấu hình chốt (vai trò của dải ngang); chỉ còn đối chiếu tham khảo ở mục 3. Các so sánh chính của N1 không bị ảnh hưởng.
- **Không ghi được môi trường của server N1** (server đã trả).
- **Thiếu 2 log train:** `combo3_rerun_s42` (E4) và `n1_nomask_s42` (N1). Cả hai run vẫn có file summary (`Train samples: 36002`, 30/30 epoch) và metrics theo epoch.
- **E4 dùng θ trung bình của STN** thay cho "crop 1,3× ở tâm" trong review, để phép thử mạnh hơn.
- 3 seed mỗi nhánh, như review đề xuất cho E4 và N1.
- Phân tích theo số frame của E4 là phân tích thêm, không đăng ký trước.

**Đính chính:** báo cáo lock-box ghi cuDNN 9.1.2. Giá trị `torch.backends.cudnn.version()` là 91002, tức **cuDNN 9.10.2**.

---

## 6. Tiến độ các mục khác

| Mục | Trạng thái |
|---|---|
| E1 (lock-box), E3 (theo số frame) | Xong (báo cáo 25/10) |
| E4, N1 | Xong (báo cáo này) |
| E2 (leave-one-out, 3 seed) | Chưa làm. Đề xuất ưu tiên bỏ frame dropout và bỏ CTC phụ; dòng "bỏ che frame" đã có từ N1 |
| E4b (STN với θ trung bình lúc chấm) | Chưa làm. Không cần train; chấm được trên 4 model lock-box có sẵn |
| E5 (cấu hình gốc fp32), E6 (SR thêm seed), N3 (hiệu chỉnh độ tin cậy) | Chưa làm |
| Prototype (API `/predict`) | Chưa hoàn thành; thời gian qua ưu tiên huấn luyện và đánh giá mô hình |

---

## 7. Xin ý kiến thầy

1. **Cách đặt claim sau N1.** N1 không ủng hộ cơ chế "khai thác thông tin bổ sung giữa các frame". Đề xuất claim chính là "bền hơn khi thiếu frame" (E3), kèm phân tích STN (E4) và N1 như một kết quả âm có đối chứng. Che frame giữ trong cấu hình chốt nhưng được báo cáo là không đóng góp. Thầy thấy cách này có hợp lý không, hay nên viết lại phần phương pháp theo hướng khác?
2. **Giao thức cho E2.** Review đề xuất chạy E2 trên giao thức phát triển (val 1.999). N1 và E4 đã chạy trên giao thức lock-box, nên nếu E2 cũng dùng lock-box thì các dòng leave-one-out so được trực tiếp với N1 (dòng "bỏ che frame" đã có sẵn). Đổi lại, lock-box bị dùng thêm cho các phân tích mô tả, dù không có quyết định nào dựa vào đó. Thầy muốn chạy E2 trên giao thức nào?
