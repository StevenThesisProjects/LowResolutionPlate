# S2 — MF-SR-OCR với λ_SR = 0.5 (ablation trọng số SR loss)

> Kết quả của cấu hình S2 trong [../training_runs/run_gpu.md](../training_runs/run_gpu.md).
> Dữ liệu nguồn: `results/mf_sr_ocr/s2_mf_sr_ocr_lam05/history_s2_lam05.csv`,
> `submission_s2_lam05.txt`. Không có `log_s2.txt` (không chạy `tee` khi train run này).
> So sánh trực tiếp với [s1_proposed_mf_sr_ocr.md](s1_proposed_mf_sr_ocr.md) — S2 chỉ khác S1
> đúng 1 tham số.

## 1. Cấu hình đã chạy

```bash
python train.py --preset stable --experiment-name s2_lam05 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.5 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full
```

Khác **duy nhất 1 tham số** so với S1: `--lambda-sr 0.5` thay vì `0.1` (đầu kia của
khoảng đề xuất trong issue #9, `λ_SR ∈ [0.1, 0.5]`). Toàn bộ kiến trúc, decode, EMA,
domain-match, STN pool giữ nguyên như S1 — đây là ablation sạch, chỉ đổi 1 biến,
nên (khác với chính S1 vốn gộp 6 thay đổi so với J2) kết quả so S1↔S2 tách được
đúng tác động của trọng số SR loss.

## 2. Kết quả chính

| Chỉ số                                   |                    Giá trị | Epoch |
| ---------------------------------------- | -------------------------: | ----: |
| **Best Val Acc (constrained decode)**    | **79.48%** (794/999 track) |    38 |
| Best Val Acc (greedy decode, cùng epoch) |     79.38% (793/999 track) |    38 |
| Val Acc epoch cuối (56, early-stopped)   |     77.58% (776/999 track) |    56 |
| Val Loss thấp nhất                       |                     0.1857 |    17 |
| Train Loss epoch cuối                    |                     0.1124 |    56 |
| `nan_batches`                            |                0 mọi epoch |     — |

Training dừng ở epoch 56 vì early stopping (`patience=18`), đúng 18 epoch liên tiếp
không cải thiện kể từ đỉnh ở epoch 38 (38+18=56) — cùng cơ chế và tham số patience
như S1, chỉ khác điểm đỉnh rơi ở epoch 38 thay vì 37.

### So sánh trực tiếp S1 (λ=0.1) vs S2 (λ=0.5)

Đối chiếu lại bằng cách chấm trực tiếp `submission_*.txt` so với `plate_text` thật
trong `annotations.json` của 999 track val (không chỉ tin số ghi trong CSV):

|                                     |   S1 (λ=0.1) |   S2 (λ=0.5) |   Chênh lệch |
| ----------------------------------- | -----------: | -----------: | -----------: |
| Track đúng                          |      797/999 |      794/999 | **−3 track** |
| Val Acc                             |       79.78% |       79.48% |   −0.30 điểm |
| Confidence trung bình               |       0.9688 |       0.9706 |      +0.0018 |
| Confidence trung vị                 |       0.9996 |       0.9997 |      +0.0001 |
| Track confidence < 0.55             | 7/999 (0.7%) | 4/999 (0.4%) |     −3 track |
| Track dự đoán sai độ dài (≠7 ký tự) |        2/999 |        1/999 |     −1 track |

**−3 track nằm sâu trong biên nhiễu ±13 track** — tăng `λ_SR` từ 0.1 lên 0.5
**không cho thấy cải thiện hay suy giảm nào đáng tin cậy** ở mức tổng. Đây là câu
trả lời (âm tính) cho câu hỏi "S1 đã là điểm tối ưu của `λ_SR` hay chưa" mà mục 7
của `s1_proposed_mf_sr_ocr.md` để ngỏ.

**Nhưng ở mức từng track, S1 và S2 KHÔNG phải hai bản gần giống nhau:**
196/999 track (19.6%) có dự đoán khác nhau giữa 2 model — 43 track S1 đúng/S2 sai,
40 track ngược lại (S2 đúng/S1 sai), phần còn lại của chênh lệch −3 đến từ các
track cả hai cùng sai nhưng sai khác chuỗi nhau. Nói cách khác: kết quả tổng gần
như hoà không phải vì 2 model học ra cùng một hàm, mà vì lượng lỗi mới sinh ra
gần bằng lượng lỗi được sửa — một dạng "triệt tiêu" ngẫu nhiên chứ không phải
bằng chứng 2 cấu hình tương đương. Hệ quả thực tế: 83 track (43+40) là ứng viên
tốt cho khảo sát thêm (ensemble S1+S2, hoặc soi lỗi định tính) vì đại diện đúng
phần khác biệt thật giữa 2 giá trị λ, không phải nhiễu ngẫu nhiên thuần.

## 3. SR loss: cùng bằng chứng học được thật, nhưng đường cong khác S1

|                                         | Epoch 1 | Epoch cuối (56) |                  Đổi |
| --------------------------------------- | ------: | --------------: | -------------------: |
| `sr_loss` (SR học được)                 |  0.2229 |          0.1949 |           **−12.5%** |
| `sr_loss_bilinear` (mốc bilinear thuần) |  0.2344 |          0.2570 |            **+9.6%** |
| Khoảng cách (bilinear − SR)             |  0.0116 |          0.0620 | **rộng gấp 5.4 lần** |

`sr_loss < sr_loss_bilinear` xuyên suốt toàn bộ 56/56 epoch và khoảng cách nới
rộng dần — cùng kết luận định tính như S1 (SR học chi tiết thật, không chỉ tái
tạo bilinear), với độ nới rộng tương đương (5.4x so với 5.7x của S1).

Điểm khác biệt đáng chú ý: ở S1, `sr_loss_bilinear` gần như đi ngang suốt training
(0.2569 → 0.2568, ~không đổi). Ở S2, `sr_loss_bilinear` **tăng đều đặn** từ 0.2344
lên 0.2570 trong toàn bộ 56 epoch (không phải nhiễu cuối batch — đây là trung bình
epoch). Vì `sr_loss_bilinear` được tính trên target đã bị warp theo `theta` hiện
tại của STN (xem `Trainer._sr_loss` trong `trainer.py`), cách đọc hợp lý nhất là
`λ_SR=0.5` kéo STN đi theo một quỹ đạo alignment khác S1 — có thể vì gradient từ
nhánh SR (nay có trọng số gấp 5 lần) ảnh hưởng ngược lên STN mạnh hơn, khiến vùng
được STN chọn để align dần trở nên "khó" hơn cho phép nội suy bilinear thuần. Đây
là suy luận từ tương quan trong CSV, chưa verify trực tiếp bằng cách so `theta`
qua các epoch — cần xem lại nếu muốn khẳng định chắc.

## 3b. Metrics bổ sung theo review Bước 2 — **cấu hình đáng chú ý nhất**

| Chỉ số | S2 | So với S1 |
|---|---:|---|
| Exact Match | 79.48% (794/999) | **−3 track (kém hơn)** |
| CER ↓ | 0.0549 | −0.0013 (tốt hơn) |
| NED ↓ / 1−NED ↑ | 0.0549 / 0.9451 | |
| PSNR: SR vs base | 18.5228 vs 16.2349 | **+2.2879 dB** — gấp hơn 2 lần S1 (+1.07) |
| SSIM: SR vs base | 0.5461 vs 0.3780 | **+0.1681** — gấp 2.4 lần S1 (+0.070) |
| r(PSNR, đọc đúng) | −0.346 | tương quan âm |

**S2 là bằng chứng trực tiếp nhất cho kết luận chính của Bước 2**: tăng `λ_SR` lên 0.5
làm chất lượng tái tạo ảnh **tốt vượt trội** — PSNR gấp hơn 2 lần, SSIM gấp 2.5 lần mọi
cấu hình khác — nhưng OCR lại **kém nhất nhóm S**.

Tức là tối ưu theo PSNR **không** đồng nghĩa với dễ đọc hơn. Khi reviewer hỏi "sao
không tối ưu theo PSNR", S2 chính là thí nghiệm đã trả lời: đã thử, và OCR tệ đi.

Chi tiết + tương quan mức track: [../buoc2_metrics.md](../buoc2_metrics.md).
Dữ liệu: `results/mf_sr_ocr/s2_mf_sr_ocr_lam05/sr_quality_s2.csv`.
Hình định tính: `results/mf_sr_ocr/s2_mf_sr_ocr_lam05/paper_figures/`.

## 4. Giới hạn cần nêu khi báo cáo

1. **Không có `log_s2.txt`** — không tái tạo được banner config gốc hay theo dõi
   sr_loss per-batch như S1; toàn bộ phân tích ở đây chỉ dựa vào
   `history_s2_lam05.csv` + `submission_s2_lam05.txt`, đủ để so sánh epoch-level
   và track-level nhưng không có log chi tiết từng step.
2. **1 run, không multi-seed** — giống mọi giới hạn đã nêu ở S1 (mục 6 của
   `s1_proposed_mf_sr_ocr.md`); kết luận "không khác biệt" ở đây cũng chưa loại
   trừ được nhiễu do `cudnn.benchmark=True`.
3. **Compute không đổi so với S1** — `λ_SR` chỉ là trọng số loss, không đổi kiến
   trúc/FLOPs, nên bảng chi phí compute của S1 trong `run_gpu.md` áp dụng nguyên
   cho S2 (~3.85x so với ResBlock không SR).
4. **Val acc dùng trọng số EMA**, cùng lưu ý như S1 (mục 6.5 của
   `s1_proposed_mf_sr_ocr.md`).

## 5. Kết luận

Trong phạm vi 1 run mỗi cấu hình: **tăng `λ_SR` từ 0.1 (S1) lên 0.5 (S2) không cải
thiện exact-match** (−3 track, trong biên nhiễu ±13) dù cả hai đều cho bằng chứng
SR học vượt bilinear. `λ_SR = 0.1` (giá trị S1 đang dùng) vẫn là lựa chọn hợp lý
để giữ làm mặc định — không có lý do đổi sang 0.5. Phát hiện phụ đáng chú ý:
196/999 track đổi dự đoán giữa 2 cấu hình dù điểm tổng gần như hoà, cho thấy 2 model
không tương đương ở mức lỗi từng track — đáng cân nhắc cho hướng ensemble thay vì
chỉ chọn 1 trong 2 theo điểm tổng.

Bước tiếp theo không đổi so với kế hoạch trong `s1_proposed_mf_sr_ocr.md`: S3
(+Perceptual loss), S4 (`--sr-scale 1`, kiểm tra giới hạn ~55% nội suy), rồi
multi-seed (O1) trên cấu hình λ=0.1 (S1) trước khi chốt.
