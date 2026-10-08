# Hai hình của Section 3 nói gì — và không nói gì

> Cả hai hình **chỉ dùng dữ liệu của S1 seed 42**, không đọc file nào của J1 hay S4.
> Script: [`paperLatex/Result/make_result_figures.py`](../../paperLatex/Result/make_result_figures.py)
> Cách PSNR được tính: [architecture_provenance.md §5](architecture_provenance.md)

---

## Fig. 3 — `SRLossVsBilinear.png`

### Câu hỏi hình này trả lời

> *"Nhánh super-resolution có thật sự học được gì không, hay nó chỉ làm lại
> việc mà phép nội suy bilinear đã làm sẵn?"*

Đây là câu hỏi bắt buộc phải trả lời trước khi kết luận "nhánh SR vô ích". Nếu
không có hình này, người phản biện sẽ nói: *"nhánh SR của các anh không giúp gì
vì nó có học được gì đâu"* — và bài mất luôn kết quả âm tính.

### Đọc hình thế nào

| Panel | Vẽ gì | Kết luận rút ra |
|---|---|---|
| **(a)** | Hai đường loss theo epoch: **cam** = ảnh nội suy bilinear so với HR, **xanh** = ảnh SR do model tạo so với HR | Đường cam gần như **phẳng** (nội suy không học gì, đúng như kỳ vọng). Đường xanh **đi xuống đều** → nhánh SR đang cải thiện thật |
| **(b)** | Khoảng cách tương đối giữa hai đường, tính bằng % | Tăng đơn điệu **2,5% → 13,8%**. Càng train, nhánh học được càng bỏ xa nội suy |

**Vì sao phải tách panel (b)?** Ở panel (a), hai đường nằm trong dải rất hẹp
(0,22–0,27) nên mắt không ước lượng được khoảng cách đang giãn ra bao nhiêu.
Mà chính khoảng cách đó mới là điều Abstract khẳng định. Tách ra thành trục
riêng thì nó tự hiển nhiên.

### ⚠️ Hình này KHÔNG chứng minh điều gì

Nó **không** nói nhánh SR giúp đọc biển số tốt hơn. Nó chỉ nói nhánh SR **tái
tạo ảnh** tốt hơn nội suy. Đây chính là điểm khiến bài thú vị: hai điều đó
tưởng đi cùng nhau nhưng thực tế **không**. Đừng viết caption kiểu "chứng minh
nhánh SR hiệu quả" — nói vậy là vượt quá bằng chứng.

### Dữ liệu

`results/multi-seed/s1_mf_sr_ocr/history_s1_seed42.csv`, hai cột `sr_loss` và
`sr_loss_bilinear`, mỗi dòng một epoch (46 epoch). Cột bilinear do
`trainer.py:278-285` ghi ra mỗi bước — nó tính loss của **phép nội suy thuần**
trên cùng đầu vào, cùng target.

---

## Fig. 4 — `PsnrVsCorrect.png`

### Câu hỏi hình này trả lời

> *"Ảnh tái tạo đẹp hơn thì có đọc đúng hơn không?"*

Trả lời: **ngược lại**. Và đây là phát hiện riêng của bài, không lặp lại của ai.

### Đọc hình thế nào

| Panel | Vẽ gì | Kết luận rút ra |
|---|---|---|
| **(a)** | Phân bố PSNR của 999 track, tách thành **đọc đúng (xanh, n=797)** và **đọc sai (cam, n=202)**. Đường đứt = trung bình mỗi nhóm | Nhóm **đọc sai nằm lệch hẳn sang phải**: 18,42 dB so với 16,12 dB. Track model đọc sai lại có PSNR **cao hơn 2,30 dB** |
| **(b)** | Chia 999 track thành **5 nhóm bằng nhau theo PSNR**, mỗi cột là tỷ lệ đọc đúng của nhóm đó | Giảm đều: **93,5 → 91,5 → 85,5 → 77,5 → 51,0%**. Nhóm tái tạo đẹp nhất đọc sai gần một nửa |

**Vì sao vẽ theo mật độ ở (a)?** Hai nhóm lệch cỡ ~4:1 (797 so với 202). Nếu vẽ
đếm thô, cột xanh cao gấp bốn và át hết cột cam, người đọc tưởng nhóm sai không
đáng kể. Chuẩn hoá mật độ thì so được **vị trí** của hai phân bố, đúng thứ cần so.

**Vì sao thêm panel (b)?** Panel (a) và hệ số `r = −0,41` nói cùng một điều,
nhưng cả hai đều đòi người đọc tin vào thống kê. Panel (b) cho họ **tự kiểm**:
chia nhóm, đếm, thấy giảm đơn điệu. Không cần biết tương quan điểm-nhị phân là gì.

### ⚠️ Hình này KHÔNG nói PSNR là chỉ số tồi nói chung

Nó nói: **trên bộ dữ liệu này**, PSNR nghịch chiều với khả năng đọc. Cách giải
thích khả dĩ (đã viết trong bài): ảnh mờ, tương phản thấp thì ít chi tiết tần số
cao để mất, nên **dễ tái tạo → PSNR cao**, nhưng cũng **khó đọc**. Ảnh nét thì
ngược lại. Tức PSNR đang đo "ảnh này dễ dựng lại tới đâu", không đo "ảnh này
chứa bao nhiêu chi tiết đọc được".

Đây là **giả thuyết**, chưa kiểm chứng riêng. Bài viết đúng như vậy, đừng nâng
lên thành khẳng định.

### Dữ liệu

Ghép ba nguồn, tất cả của S1 seed 42:

```
sr_quality_s1_seed42.csv   → PSNR từng track  (cột psnr_sr)
submission_s1_seed42.txt   → model đọc ra chuỗi gì
annotations.json mỗi track → chuỗi đúng (plate_text)
```

Track được đánh dấu "đọc đúng" khi chuỗi dự đoán **trùng khít** chuỗi đúng —
cùng tiêu chí exact match dùng cho mọi con số accuracy trong bài.

---

## Vì sao trong bài vẫn có `w/o SR` và `SR×1`

Đây **không phải model khác**. Chúng là **chính S1 bị gỡ bớt bộ phận**, tồn tại
chỉ để trả lời câu hỏi về S1:

| Cấu hình | Là gì | Trả lời câu hỏi nào về S1 |
|---|---|---|
| **Full** | S1 đầy đủ | — |
| `w/o SR` | S1 **bỏ** nhánh SR + DCN + temporal averaging | *"Nhánh SR của S1 có đóng góp gì không?"* |
| `SR×1` | S1 **giữ** nhánh SR nhưng bỏ phóng đại ×2 | *"Lợi ích đến từ SR hay chỉ từ việc T tăng gấp đôi?"* |

Bỏ hai cấu hình này khỏi bài thì **mất luôn kết quả âm tính** — một trong ba
đóng góp chính. Không có `w/o SR` thì không ai biết nhánh SR có cần thiết hay
không; không có `SR×1` thì không tách được "lợi ích của SR" khỏi "lợi ích của
chuỗi CTC dài gấp đôi".

Nhưng hai hình thì **thuần S1**, vì chúng mô tả **hành vi bên trong** của model
đề xuất, không phải so sánh giữa các cấu hình. Việc so sánh nằm ở Table 2.

---

## Tái tạo lại hai hình

```bash
python3 paperLatex/Result/make_result_figures.py
```

In ra để tự kiểm:

```
SRLossVsBilinear: below bilinear 46/46 epochs, gap 2.5% -> 13.8%
PsnrVsCorrect: quintile accuracy 93.5 -> 91.5 -> 85.5 -> 77.5 -> 51.0
```

Sửa hình thì sửa script rồi chạy lại, **không sửa file ảnh**.
