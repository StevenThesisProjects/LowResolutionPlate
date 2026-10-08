# S3 — MF-SR-OCR với L_Perceptual (α=0.1)

> ⚠️ **1 seed, `benchmark=True` — chưa xác nhận.** Phạm vi multi-seed đã chốt
> (2026-08-03) là **3 model J1 + S1 + S4**, không có S3 — lý do chi phí ở
> [../checklist_review.md](../../../report/checklist_review.md). 805/999 vẫn là điểm 1-seed
> cao nhất từng đo, nhưng không có error bar; đừng báo cáo như số cuối cho paper.
>
> Kết quả của cấu hình S3 trong [../training_runs/run_gpu.md](../../../report/training_runs/run_gpu.md).
> Dữ liệu nguồn: `results/mf_sr_ocr/s3_l_perceptual/history_s3_perceptual.csv`,
> `submission_s3_perceptual.txt`. Không có `log_s3.txt` (không chạy `tee` khi train run này).
> So sánh trực tiếp với [s1_proposed_mf_sr_ocr.md](../../../report/baseline1_crnn_stn/s1_proposed_mf_sr_ocr.md) — S3 chỉ khác S1
> đúng 1 tham số, giống cách S2 khác S1.

## 1. Cấu hình đã chạy

```bash
python train.py --preset stable --experiment-name s3_perceptual \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 --sr-perceptual-weight 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full
```

Khác **duy nhất 1 tham số** so với S1: thêm `--sr-perceptual-weight 0.1` (bật
`L_Perceptual` — VGG16 relu2_2 — trong công thức `L_SR = L1 + α·L_Perceptual` của
issue #9, α=0.1). `λ_SR` giữ nguyên 0.1 như S1 (khác S2, vốn giữ nguyên perceptual=0
nhưng đổi λ_SR). Đây là ablation sạch thứ 2 tách từ S1, đo trên đúng 1 biến khác.

## 2. Kết quả chính

| Chỉ số | Giá trị | Epoch |
|---|---:|---:|
| **Best Val Acc (constrained decode)** | **80.58%** (805/999 track) | 25 |
| Best Val Acc (greedy decode, cùng epoch) | 80.38% (803/999 track) | 25 |
| Val Acc epoch cuối (43, early-stopped) | 79.18% (791/999 track) | 43 |
| Val Loss thấp nhất | 0.1822 | 19 |
| Train Loss epoch cuối | 0.0519 | 43 |
| `nan_batches` | 0 mọi epoch | — |

Training dừng ở epoch 43 vì early stopping (`patience=18`, 25+18=43) — cùng cơ
chế như S1/S2. Đỉnh rơi ở epoch 25, **sớm hơn nhiều** so với S1 (epoch 37) và S2
(epoch 38) — thêm perceptual loss khiến model hội tụ nhanh hơn nhưng cũng bắt đầu
overfit sớm hơn tương ứng (xem mục 4).

### So sánh trực tiếp S1 / S2 / S3

Đối chiếu lại bằng cách chấm trực tiếp `submission_*.txt` so với `plate_text` thật
trong `annotations.json` của 999 track val (không chỉ tin số ghi trong CSV):

| | S1 (λ=0.1) | S2 (λ=0.5) | S3 (λ=0.1 + perceptual α=0.1) |
|---|---:|---:|---:|
| Track đúng | 797/999 | 794/999 | **805/999** |
| Val Acc | 79.78% | 79.48% | **80.58%** |
| Δ vs S1 | — | −3 track | **+8 track** |
| Confidence trung bình | 0.9688 | 0.9706 | 0.9610 |
| Confidence trung vị | 0.9996 | 0.9997 | 0.9983 |
| Track confidence < 0.55 | 7/999 (0.7%) | 4/999 (0.4%) | **9/999 (0.9%)** |
| Track dự đoán sai độ dài (≠7 ký tự) | 2/999 | 1/999 | **0/999** |

**S3 là điểm ước lượng cao nhất trong toàn bộ lịch sử thử nghiệm** (805/999), vượt
S1 tới +8 track. Nhưng +8 track **vẫn nằm trong biên nhiễu ±13 track** đã dùng
xuyên suốt các tài liệu khác — nghĩa là **chưa đủ bằng chứng để khẳng định
perceptual loss cải thiện thật**, dù đây là tín hiệu tích cực đáng chạy multi-seed
để xác nhận (khác với S2, vốn cho tín hiệu rõ ràng là không cải thiện).

Đáng chú ý: S3 có **confidence trung bình thấp hơn** S1/S2 (0.961 so với 0.969/0.971)
và **nhiều track confidence thấp hơn** (9 so với 7/4), dù chính xác cao hơn. Model
chính xác hơn nhưng "tự tin" kém hơn — không mâu thuẫn, chỉ là 2 đại lượng đo
khác nhau (confidence ở đây tính từ xác suất CTC per-step, không phải calibration
thật). Điểm tốt duy nhất tuyệt đối: **0/999 track bị fallback về 6 ký tự** (S1 có
2, S2 có 1) — layout constraint luôn tìm được chuỗi 7 ký tự hợp lệ trong khung `T`.

So dự đoán từng track (so với S1): 43 track S3 đúng/S1 sai, 35 track ngược lại —
lệch dương rõ ràng (không chỉ triệt tiêu ngẫu nhiên như cặp S1/S2), 182/999 track
(18%) đổi dự đoán. So với S2: 45 track S3 đúng/S2 sai, 34 track ngược lại,
185/999 khác dự đoán — cùng pattern.

## 3. SR loss — lưu ý về thang đo, không so trực tiếp được với S1/S2

`sr_loss`/`sr_loss_bilinear` của S3 dùng **cùng 1 instance `SRPixelLoss`** cho cả
2 cột (bao gồm cả perceptual term, vì `Trainer.sr_loss_fn` được truyền chung cho
cả nhánh học lẫn nhánh baseline không gradient) — nên trong nội bộ S3, so sánh
`sr_loss` với `sr_loss_bilinear` vẫn hợp lệ (táo với táo). Nhưng **giá trị tuyệt
đối của S3 (~0.35-0.41) không so được với S1/S2 (~0.19-0.26)** vì công thức loss
khác nhau (S3 có thêm `α·L_Perceptual`, S1/S2 chỉ L1 thuần) — chênh lệch độ lớn
không phản ánh SR của S3 "kém" SR của S1/S2 hơn, chỉ là thước đo khác.

| | Epoch 1 | Epoch cuối (43) | Đổi |
|---|---:|---:|---:|
| `sr_loss` | 0.4100 | 0.3553 | **−13.3%** |
| `sr_loss_bilinear` | 0.4144 | 0.4110 | ~không đổi (dao động 0.407-0.414) |
| Khoảng cách (bilinear − SR) | 0.0044 | 0.0557 | **rộng gấp ~12.6 lần** |

`sr_loss < sr_loss_bilinear` ở toàn bộ 43/43 epoch, khoảng cách nới rộng dần —
cùng kết luận định tính như S1/S2 (SR học chi tiết thật). `sr_loss_bilinear` của
S3 dao động quanh một dải hẹp không có xu hướng rõ (giống S1, khác S2 vốn tăng
đều đặn) — cả S1 và S3 đều dùng `λ_SR=0.1`, càng củng cố suy đoán ở
[s2_lam05_mf_sr_ocr.md §3](s2_lam05_mf_sr_ocr.md#3-sr-loss-cùng-bằng-chứng-học-được-thật-nhưng-đường-cong-khác-s1)
rằng chính `λ_SR` cao (không phải perceptual) là thứ kéo quỹ đạo STN đi khác ở S2.

## 4. Hội tụ nhanh hơn, overfit sớm hơn

Đỉnh val acc rơi ở epoch 25 (S1: 37, S2: 38) và val loss thấp nhất ở epoch 19
(S1: 21, S2: 17) — cả 3 cấu hình đều chạm đáy val loss sớm rồi tăng dần trong khi
train loss tiếp tục giảm, nhưng S3 chạm đáy **sớm nhất**. Perceptual loss có thể
đã cung cấp gradient signal "đặc" hơn (VGG feature space) giúp model học nhanh hơn
ở giai đoạn đầu, nhưng cũng khiến overfit bắt đầu sớm hơn tương ứng — nhất quán với
nhận định đã lặp lại ở [groupnorm_sr_ablation_j1_j2.md §5](groupnorm_sr_ablation_j1_j2.md#5-overfit--cùng-pattern-ở-cả-2-run)
rằng dataset ~19,000 track là dư cho 60-80 epoch, dư địa nên nhắm vào chống overfit.

## 4b. Metrics bổ sung theo review Bước 2

| Chỉ số | S3 | Ghi chú |
|---|---:|---|
| Exact Match | 80.58% (805/999) | đồng hạng nhất với S4 |
| CER ↓ | **0.0522** | **tốt nhất trong 4 cấu hình** |
| NED ↓ / 1−NED ↑ | 0.0522 / **0.9478** | |
| PSNR: SR vs base | 16.9582 vs 15.8727 | +1.0855 dB — gần bằng S1 |
| SSIM: SR vs base | 0.4234 vs 0.3569 | +0.0664 |
| r(PSNR, đọc đúng) | **−0.412** | tương quan âm **mạnh nhất** trong 4 cấu hình |

**CER là tiêu chí duy nhất tách được S3 khỏi S4** — hai cấu hình hoà tuyệt đối ở exact
match (805/999) nhưng S3 có CER thấp hơn (0.0522 so với 0.0532), tức khi sai thì sai
ít ký tự hơn. Chênh lệch nhỏ, vẫn cần multi-seed để xác nhận.

Đáng chú ý: perceptual loss **không** cải thiện chất lượng ảnh đo được — PSNR/SSIM của
S3 gần như bằng S1 (+1.086 vs +1.072 dB). Lợi ích của nó (nếu có) nằm ở OCR chứ không ở
chất lượng tái tạo.

Chi tiết: [../buoc2_metrics.md](../../../report/buoc2_metrics.md).
Dữ liệu: `results/mf_sr_ocr/s3_l_perceptual/sr_quality_s3.csv`.
Hình định tính: `results/mf_sr_ocr/s3_l_perceptual/paper_figures/`.

## 5. Giới hạn cần nêu khi báo cáo

1. **Không có `log_s3.txt`** — cùng hạn chế như S2, chỉ dựa vào CSV + submission.
2. **1 run, không multi-seed** — +8 track so với S1 là điểm ước lượng cao nhất đã
   đo nhưng **vẫn trong biên nhiễu ±13**, khác hẳn mức độ tin cậy của S1 so với J2
   (+26 track, ngoài biên nhiễu). Không được báo cáo "S3 tốt hơn S1" như một kết
   luận chắc chắn — chỉ là "tín hiệu tích cực cần xác nhận".
3. **sr_loss không so trực tiếp được với S1/S2** — xem lưu ý thang đo ở mục 3.
4. **Compute tăng thêm so với S1** — bật perceptual nghĩa là tải VGG16 và forward
   qua nó mỗi batch cho cả pred và target; chưa benchmark lại latency/FLOPs thật
   (bảng compute ở `run_gpu.md` chưa có dòng riêng cho S3).
5. **Val acc dùng trọng số EMA**, cùng lưu ý như S1/S2.

## 6. Kết luận

Trong phạm vi 1 run mỗi cấu hình: **thêm `L_Perceptual` (α=0.1) cho điểm ước lượng
cao nhất trong toàn bộ lịch sử thử nghiệm** — 805/999 (80.58%), +8 track so với S1
và +11 so với S2 — nhưng khoảng cách với S1 vẫn nằm trong biên nhiễu ±13 track nên
**chưa thể khẳng định chắc chắn perceptual loss cải thiện thật**, chỉ có thể nói
đây là hướng đáng ưu tiên xác nhận trước khi loại bỏ. Khác biệt so với S2 (kết quả
âm tính rõ ràng): S3 từng là ứng viên hợp lý nhất để đưa vào multi-seed cùng S1.

**Cập nhật 2026-08-02**: phạm vi multi-seed cuối cùng đã đổi thành
**3 model J1 + S1 + S4** (không có S3) — để tiết kiệm chi phí GPU và ưu tiên đóng
lập luận "T=32 confound" (câu hỏi mà chính S4 đặt ra ở §3 file này thực ra thuộc
về S4) bằng mốc nền J1 có error bar, thay vì multi-seed thêm một ablation loss
(S3) vốn đã đồng hạng với S4. Lý do đầy đủ:
[../checklist_review.md](../../../report/checklist_review.md). CER của S3 (0.0522, tốt nhất 4
cấu hình) vẫn là một tín hiệu đáng nêu trong paper, kèm nhãn "1 seed, chưa xác nhận".
