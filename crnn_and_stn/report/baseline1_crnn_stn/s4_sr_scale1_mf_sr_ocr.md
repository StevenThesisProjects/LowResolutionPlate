# S4 — MF-SR-OCR với SR scale=1 (tách "T confound" khỏi upsampling)

> Kết quả của cấu hình S4 trong [../training_runs/run_gpu.md](../training_runs/run_gpu.md).
> Dữ liệu nguồn: `report/csv-report-process/mf_sr_ocr/s4_sr_scale1/history_s4_sr_scale1.csv`,
> `log_s4.txt`, `submission_s4_sr_scale1.txt`.
> So sánh trực tiếp với [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) và
> [s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md).

## 1. Cấu hình đã chạy

```bash
python train.py --preset stable --experiment-name s4_sr_scale1 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --width-downsample 4 \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s4.txt
```

Khác S1 ở **2 tham số cùng lúc** (không phải ablation 1-biến sạch như S2/S3):
`--sr-scale 1` (thay 2) **và** `--width-downsample 4` (thay mặc định 8). Hai tham
số này phải đi cùng nhau vì lý do kỹ thuật: SR scale=1 không phóng to ảnh
(output = input size, xem `FrameSR.forward` — `base = x` khi `scale=1` thay vì
`F.interpolate`), nên nếu giữ `width_downsample=8` mặc định thì `T` sẽ tụt về 16
thay vì 32, làm S4 khác S1 tới **2 biến chồng nhau** (SR scale lẫn T) thay vì tách
được đúng 1 biến muốn đo.

**Mục tiêu của S4**: `s1_proposed_mf_sr_ocr.md §4` từng đặt câu hỏi để ngỏ — SR ở
S1 thắng vì học được chi tiết ảnh thật, hay một phần (hoặc toàn bộ) lợi ích chỉ
đến từ việc SR ×2 vô tình tăng gấp đôi `T` (32 thay vì 16)? Code comment trong
`src/models/components.py::ResBackbone` gọi đây là "hidden reason". S4 giữ `T=32`
bằng cách khác (`width_downsample=4`, một thay đổi backbone thuần tuý, không
upsample) trong khi tắt hẳn phần phóng to ảnh của SR (`scale=1`) — nếu S4 vẫn đạt
độ chính xác tương đương/cao hơn S1, nghĩa là `T=32` mới là yếu tố chính, không
phải việc SR có phóng to ảnh lên 2x thật hay không.

Params: 29,549,470 (ít hơn S1 27,744 params — do lớp `PixelShuffle` upsample của
SR chỉ cần `hidden_channels × 1²` kênh đầu ra khi `scale=1`, thay vì `× 2²` ở S1).

## 2. Kết quả chính

| Chỉ số | Giá trị | Epoch |
|---|---:|---:|
| **Best Val Acc (constrained decode)** | **80.58%** (805/999 track) | 40 |
| Best Val Acc (greedy decode, cùng epoch) | 80.58% (805/999 track) | 40 |
| Val Acc epoch cuối (58, early-stopped) | 78.98% (789/999 track) | 58 |
| Val Loss thấp nhất | 0.1868 | 21 |
| Train Loss epoch cuối | 0.0283 | 58 |
| `nan_batches` | 0 mọi epoch | — |

Training dừng ở epoch 58 vì early stopping (`patience=18`, 40+18=58). **Điểm khác
biệt duy nhất so với S1/S2/S3**: constrained decode và greedy decode cho **cùng
một kết quả tuyệt đối** tại epoch tốt nhất (805/999 cả hai) — layout constraint
không sửa thêm được track nào ở đây, khác với S1 (+2 track) hay S3 (+2 track).

### So sánh trực tiếp S1 / S3 / S4 (verify bằng ground truth)

Đối chiếu lại bằng cách chấm trực tiếp `submission_*.txt` so với `plate_text` thật
trong `annotations.json` của 999 track val:

| | S1 (scale=2, λ=0.1) | S3 (scale=2, +perceptual) | S4 (scale=1, T qua width_downsample) |
|---|---:|---:|---:|
| Track đúng | 797/999 | 805/999 | **805/999** |
| Val Acc | 79.78% | 80.58% | **80.58%** |
| Δ vs S1 | — | +8 track | **+8 track** |
| Confidence trung bình | 0.9688 | 0.9610 | **0.9735 (cao nhất)** |
| Track confidence < 0.55 | 7/999 | 9/999 | **0/999 (không có)** |
| Track sai độ dài (≠7 ký tự) | 2/999 | 0/999 | 1/999 |

**S4 và S3 hoà điểm tuyệt đối (805/999)** dù đạt bằng 2 cách hoàn toàn khác nhau
(S3: giữ SR ×2 nguyên bản, thêm perceptual loss; S4: bỏ upsampling của SR, đổi
cách lấy T=32). Nhưng **đây không phải cùng 1 model**: so trực tiếp dự đoán từng
track, S3 và S4 đúng/sai ở các track khác nhau gần như 50/50 (42 track S3 đúng/S4
sai, 42 track ngược lại, 185/999 = 18.5% khác dự đoán) — hai con đường độc lập
hội tụ về cùng 1 điểm số, không phải học ra cùng 1 hàm.

So với S1: 44 track S4 đúng/S1 sai, 36 track ngược lại, 182/999 khác dự đoán —
cùng độ lớn "xáo trộn" như cặp S1/S3 đã thấy ở tài liệu trước.

**+8 track vẫn nằm trong biên nhiễu ±13** nên không thể tuyên bố "S4 tốt hơn S1"
một cách chắc chắn theo tiêu chuẩn thống kê đã áp dụng xuyên suốt các tài liệu
khác. Nhưng việc **2 thay đổi độc lập (S3 và S4) đều hội tụ về đúng 805/999** —
thay vì rải rác ngẫu nhiên quanh 797 — là một tín hiệu đồng thuận đáng chú ý hơn
một điểm số đơn lẻ vượt biên nhiễu.

## 3. Trả lời câu hỏi "T confound" — kết quả chính của S4

Đây là phát hiện quan trọng nhất của S4, trực tiếp trả lời câu hỏi đặt ra ở mục 1:

**S4 (T=32 qua `width_downsample=4`, SR không phóng to ảnh) đạt 805/999 — bằng
hoặc cao hơn S1 (T=32 qua SR ×2 phóng to ảnh, 797/999).** Nếu lợi ích chính của
"SR" trong S1 đến từ việc phục hồi chi tiết ảnh thật (thứ chỉ có được khi thực sự
phóng to pixel), S4 lẽ ra phải kém S1 rõ rệt vì S4 không phóng to ảnh chút nào.
Thực tế S4 không hề kém — bằng chứng này nghiêng về phía **`T=32` (nhiều bước thời
gian hơn cho CTC decode) là yếu tố đóng góp chính**, không phải bản thân việc ảnh
được phóng to.

**Giới hạn cần nêu ngay**: đây **không phải ablation tinh khiết** vì đổi 2 biến
cùng lúc (mục 1) — S4 vẫn bật `--use-sr` (module SR vẫn chạy, chỉ không phóng to
ảnh), nên chưa tách được hoàn toàn "SR có ích" khỏi "T=32 có ích". Để trả lời dứt
điểm, cần thêm 1 run nữa: `--width-downsample 4` **không** `--use-sr` (bỏ hẳn
module SR, chỉ đổi cấu trúc backbone để T=32) — nếu run đó cũng đạt ~80%, kết luận
"T=32 là yếu tố chính, module SR gần như không đóng góp thêm" sẽ chắc chắn hơn
nhiều. Run này **chưa được thực hiện**, nằm ngoài phạm vi S1-S4 hiện tại.

**Hệ quả thực tế nếu giả thuyết T=32 đúng**: S4 xử lý ảnh 32×128 (không phóng to)
trong khi S1/S3 xử lý ảnh 64×256 (đã phóng to ×2) qua toàn bộ backbone + BiLSTM —
tức S4 nhiều khả năng **rẻ hơn đáng kể** về GFLOPs/latency so với S1/S3 dù đạt độ
chính xác tương đương/cao hơn. Chưa benchmark thật (`tools/benchmark.py` chưa
chạy cho cấu hình `scale=1`) nên đây là suy luận định tính, không phải số đo — nên
benchmark trước khi đưa vào bảng chi phí compute chính thức.

## 4. SR loss — "bilinear" ở đây là identity, không phải nội suy

Vì `scale=1`, `base` trong `FrameSR.forward` là chính ảnh đầu vào không đổi
(`base = x`, không qua `F.interpolate`) — nên cột `sr_loss_bilinear` ở S4 đo
"không làm gì cả" (identity) chứ không phải phép nội suy bicubic/bilinear thật
như ở S1/S3.

| | Epoch 1 | Epoch cuối (58) | Đổi |
|---|---:|---:|---:|
| `sr_loss` (SR học được) | 0.2228 | 0.1941 | −12.9% |
| `sr_loss_bilinear` (mốc: giữ nguyên ảnh, không SR) | 0.2262 | 0.2269 | ~không đổi |
| Khoảng cách | 0.0034 | 0.0328 | rộng gấp ~9.8 lần |

`sr_loss < sr_loss_bilinear` toàn bộ 58/58 epoch, khoảng cách nới rộng — module SR
vẫn học được cách tinh chỉnh ảnh tốt hơn "giữ nguyên" ngay cả ở đúng độ phân giải
gốc (không phóng to). Cùng kết luận định tính với S1/S3 nhưng phép so sánh ở đây
dễ hơn nhiều (không có nội suy làm mốc phải vượt qua) nên không nên đọc mức % cải
thiện này là bằng chứng mạnh tương đương S1.

## 5. Chất lượng dự đoán — tốt nhất về mặt calibration trong 4 cấu hình

- **0/999 track có confidence dưới 0.55** — lần đầu tiên trong toàn bộ lịch sử
  thử nghiệm không có track "đáng ngờ" nào theo ngưỡng này (S1: 7, S2: 4, S3: 9).
- Confidence trung bình cao nhất (0.9735) trong 4 cấu hình S1-S4.
- 1/999 track fallback về 6 ký tự (`constrained_beam_decode` không tìm được
  chuỗi 7 ký tự hợp lệ trong khung T) — tốt hơn S1 (2) nhưng kém S3 (0).

Kết hợp với mục 2 (decode constrained = greedy tuyệt đối tại epoch tốt nhất),
S4 là cấu hình có hành vi "ổn định/dễ đoán" nhất trong 4 cấu hình đã chạy, dù
điểm accuracy chỉ ngang S3.

## 6. Giới hạn cần nêu khi báo cáo

1. **Không phải ablation 1-biến sạch** — đổi cả `sr-scale` lẫn `width-downsample`
   cùng lúc (mục 1), nên "S4 tốt hơn S1" không tách được rạch ròi là do bỏ
   upsampling hay do đổi cấu trúc downsample của backbone — dù về mặt lý thuyết
   2 tham số này bắt buộc phải đi cùng nhau để giữ T=32 cố định làm biến kiểm soát.
2. **1 run, không multi-seed** — như mọi cấu hình khác trong nhóm S1-S4.
3. **T confound chưa tách hoàn toàn** — xem mục 3, cần thêm 1 run
   `--width-downsample 4` không `--use-sr` để kết luận chắc chắn.
4. **Compute chưa benchmark** — bảng chi phí trong `run_gpu.md`/
   `model_comparison_summary.md` chưa có dòng cho cấu hình `scale=1`.
5. **Val acc dùng trọng số EMA**, cùng lưu ý như S1/S2/S3.

## 7. Kết luận

S4 đạt **805/999 (80.58%)** — hoà tuyệt đối với S3, cao hơn S1 8 track (trong
biên nhiễu, chưa kết luận chắc). Giá trị lớn nhất của S4 không phải ở điểm số
(vốn đã có S3 đạt trước), mà ở việc **cung cấp bằng chứng gián tiếp mạnh rằng
`T=32` — không phải bản thân việc phóng to ảnh ×2 — là yếu tố chính đứng sau lợi
ích đo được của "SR"** trong toàn bộ nhóm S1-S3. Đây là phát hiện có thể đổi cách
hiểu về vì sao S1 hoạt động tốt, và gợi ý một hướng rẻ hơn (SR scale=1 hoặc thậm
chí bỏ hẳn SR, chỉ cần `width_downsample=4`) để đạt độ chính xác tương đương với
chi phí compute thấp hơn nhiều — cần benchmark + 1 ablation "T=32 không SR" để
xác nhận trước khi coi đây là kết luận cuối.

Bước tiếp theo: chạy ablation "`--width-downsample 4` không `--use-sr`" để tách
dứt điểm T khỏi SR (chưa có trong kế hoạch S1-S4 ban đầu, nên cân nhắc bổ sung);
sau đó multi-seed (O1) trên bộ 3 ứng viên tốt nhất hiện tại — **S1, S3, S4** —
trước khi chốt cấu hình cuối để nộp submission.
