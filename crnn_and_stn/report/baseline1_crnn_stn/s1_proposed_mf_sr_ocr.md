# S1 — Joint End-to-End MF-SR-OCR (Proposed Method)

> Kết quả của cấu hình S1 trong [../training_runs/run_gpu.md](../training_runs/run_gpu.md).
> Dữ liệu nguồn: `report/csv-report-process/history_s1_proposed.csv`,
> `log_s1.txt`, `submission_s1_proposed.txt`.
> 3 ablation tách từ S1 đã chạy — xem
> [s2_lam05_mf_sr_ocr.md](s2_lam05_mf_sr_ocr.md) (λ_SR=0.5: kém S1 3 track, trong
> biên nhiễu, không cải thiện),
> [s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md) (+L_Perceptual: **hơn
> S1 8 track, đồng hạng cao nhất đã đo**) và
> [s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md) (SR scale=1, không phóng
> to ảnh: **hoà tuyệt đối với S3**, bằng chứng nghiêng về `T=32` là yếu tố chính
> chứ không phải bản thân việc phóng to ảnh). Cả S3 lẫn S4 vẫn trong biên nhiễu so
> với S1 nên chưa kết luận chắc chắn tốt hơn. Bảng so sánh đầy đủ:
> [model_comparison_summary.md](model_comparison_summary.md).

## 1. Cấu hình đã chạy

```
python train.py --preset stable --experiment-name s1_proposed \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full
```

Đọc từ header của `log_s1.txt` — đây là toàn bộ khác biệt so với J2:

| Thành phần | J2 | S1 |
|---|---|---|
| SR | per-frame (single-frame), scale 2 | **multi-frame** (`multi_frame=True`), scale 2 |
| DCNv2 | không có | **có**, kernel identity-init |
| STN pool | `(1,1)` global-average | **`(4,8)`** |
| Domain match (Nguyên nhân #4) | không | **có** (`--lr-domain-match`) |
| Decode | greedy | **constrained** (layout `LLLNLNN,LLLNNNN`, beam 16) |
| EMA | không | **có** (decay 0.999) |
| SR target alignment | lệch geometry (bug) | **đã sửa** — augment 1 lần ở cỡ target, warp target theo `theta` của STN |
| T (timestep CTC) | 32 (do SR tăng width) | 32 |
| Params | 29,426,895 | **29,577,214** |

Toàn bộ 5 bước của pipeline đề xuất đều bật: STN per-frame → DCNv2 align → MFSR ×2
(+`L_SR` = L1, không Sobel/Perceptual) → backbone + Attention Fusion → BiLSTM + CTC,
`λ_SR = 0.1`. 19.001 track train (38.002 sample train+synthetic), 999 track val
(Scenario-B), không NaN batch nào trong suốt 55 epoch đã chạy.

---

## 2. Kết quả chính

| Chỉ số | Giá trị | Epoch |
|---|---:|---:|
| **Best Val Acc (constrained decode)** | **79.78%** (797/999 track) | 37 |
| Best Val Acc (greedy decode, cùng epoch) | 79.58% (795/999 track) | 37 |
| Val Acc epoch cuối (55, early-stopped) | 78.88% (789/999 track) | 55 |
| Val Loss thấp nhất | 0.1910 | 21 |
| Train Loss epoch cuối | 0.0306 | 55 |
| `nan_batches` | 0 mọi epoch | — |

Training dừng ở epoch 55 vì early stopping (`patience=18`) — đúng 18 epoch liên
tiếp không cải thiện kể từ đỉnh ở epoch 37, khớp chính xác với cấu hình
`EARLY_STOPPING_PATIENCE=18` trong `configs/config.py`. Không phải log bị cắt.

### So sánh với các cấu hình đã chạy trước

| Run | Cấu hình | Track đúng | Val Acc | So với S1 |
|---|---|---:|---:|---:|
| — | CRNN + STN (report ICPR gốc) | — | 77.00% | **+27 track / +2.78 điểm** |
| — | CRNN + STN (đo lại) | 757 | 75.78% | +40 track |
| SR-v1 | stacked-input SR (bản lỗi) | 492 | 49.25% | +305 track |
| SR-v2 | stacked-input SR, lr thấp + aug light | 550 | 55.06% | +247 track |
| — | ResBlock backbone `norm=none` | 766 | 76.68% | +31 track |
| J1 | + GroupNorm, không SR | 768 | 76.88% | +29 track |
| J2 | + SR per-frame (single-frame) có giám sát | 771 | 77.18% | **+26 track / +2.60 điểm** |
| J3 | + DCNv2 (kernel init ngẫu nhiên — bug cũ) | 762 | 76.28% | +35 track |
| **S1** | **Joint MF-SR-OCR (đề xuất)** | **797** | **79.78%** | — |

Biên nhiễu của val 999 track là **±13 track (±1.3 điểm)** — đã dùng để đánh giá
J1/J2/J3 trước đây (chênh nhau 3-9 track, tức nằm trong nhiễu, không kết luận
được gì). **S1 hơn J2 tới 26 track — gấp đôi biên nhiễu.** Đây là lần đầu tiên
một cấu hình vượt qua ngưỡng ±13 track kể từ baseline gốc 77.00%. Vẫn cần xác
nhận bằng multi-seed (ngoài phạm vi hiện tại) trước khi khẳng định chắc chắn,
nhưng khoảng cách này không còn mong manh như J1→J2→J3.

---

## 3. Phân rã: cải thiện đến từ đâu

Đây là 1 run duy nhất gộp 6 thay đổi cùng lúc (MFSR, DCN, STN pool, domain-match,
decode, EMA) nên **không tách được đóng góp riêng của từng cái** — ablation cho
việc đó nằm ngoài phạm vi hiện tại. Nhưng CSV cho phép tách được **2 nhóm lớn**:

**Nhóm 1 — kiến trúc/training (MFSR + DCN + GroupNorm + domain-match + STN pool + EMA), đo bằng greedy decode để loại yếu tố decode:**
795/999 (79.58%) so với J2 771/999 (77.18%, cũng đo bằng greedy vì code cũ chỉ
có greedy) → **+24 track / +2.40 điểm**, đã tự nó vượt biên nhiễu ±13.

**Nhóm 2 — constrained decode**, đo bằng chênh lệch `val_acc − val_acc_greedy`
ngay trong S1: sau giai đoạn khởi động (epoch 1-2, model còn xuất chuỗi rác nên
chênh tới +7.5 điểm), khoảng cách ổn định quanh **+0.10 → +0.30 điểm** (~1-3
track) suốt phần còn lại của training, và đúng **+2 track (+0.20 điểm)** tại
epoch tốt nhất (797 so với 795). Nhất quán với dự đoán ban đầu: ràng buộc layout
sửa được lỗi sai độ dài / sai lớp ký tự ở vị trí đã biết, nhưng model đã train
tốt thì tự nó ít mắc lỗi dạng này — phần lớn giá trị nằm ở nhóm 1, không phải decode.

---

## 4. SR có thật sự học được gì, hay chỉ tái tạo bilinear?

Đây là câu hỏi trọng tâm đặt ra từ đầu — SR trước đây (J2) chỉ +0.30 track so với
J1 và không rõ có học được gì hay không. Với bản sửa (target căn chỉnh đúng theo
`theta`, augment hình học một lần dùng chung cho input/target), CSV cho câu trả
lời rõ ràng:

| | Epoch 1 | Epoch 55 | Đổi |
|---|---:|---:|---:|
| `sr_loss` (SR học được) | 0.2497 | 0.2168 | **−13.2%** |
| `sr_loss_bilinear` (mốc: bilinear thuần, không học) | 0.2569 | 0.2568 | ~không đổi |
| Khoảng cách (bilinear − SR) | 0.0070 | 0.0396 | **rộng gấp 5.7 lần** |

`sr_loss < sr_loss_bilinear` ở **toàn bộ 55/55 epoch**, và khoảng cách **nới rộng
dần theo thời gian** thay vì co lại hay dao động ngẫu nhiên quanh 0. Đây là bằng
chứng SR học được chi tiết thật vượt quá phép nội suy bilinear — khác hẳn giả
thuyết "SR chỉ tái tạo lại bilinear" đặt ra khi thiết kế thí nghiệm S4. Không có
nghĩa là giới hạn 55% pixel nội suy (do HR gốc chỉ ~115×42px, target ×2 là
256×64px) đã được giải quyết — S4 (`--sr-scale 1`) vẫn cần chạy để đo SR ở đúng
thang thông tin thật của dataset — nhưng bản sửa target-alignment đã cho SR một
tín hiệu học được thay vì học ra bộ lọc nhiễu như PR #7.

---

## 5. Chất lượng dự đoán trên tập validation (`submission_s1_proposed.txt`)

999 dòng, đúng bằng số track validation — đây là bản ghi dự đoán ở epoch tốt
nhất (37), dùng constrained decode.

- **Confidence**: trung bình 0.969, trung vị 0.9996 — phần lớn dự đoán rất chắc
  chắn. Chỉ 7/999 track (0.7%) có confidence dưới 0.55: `track_19095` (0.494 —
  thấp nhất), `track_15594` (0.505), `track_22161` (0.508), `track_19725`
  (0.513), `track_12513` (0.516), `track_12247` (0.513), `track_12478` (0.522).
  Đây là các ứng viên tốt cho phần "ảnh minh hoạ lỗi" (`tools/visualize.py
  --only-errors`) trong báo cáo.
- **Độ dài dự đoán**: 997/999 track đúng 7 ký tự như ràng buộc layout. **2/999
  track (0.2%) có độ dài 6** (`track_14442` → `MKL712`, `track_17959` →
  `BBB581`) — đây là trường hợp beam search không tìm được chuỗi 7 ký tự hợp lệ
  nào trong khung `T=32`, nên `constrained_beam_decode` rơi vào nhánh fallback về
  greedy (xem docstring hàm trong `postprocess.py`). Ràng buộc layout **không
  tuyệt đối 100%** trong thực tế — cần biết để không báo cáo "100% đúng độ dài".

---

## 6. Giới hạn cần nêu khi báo cáo

1. **1 run, không multi-seed** — kết luận "+26 track so với J2" vượt biên nhiễu
   ±13 nên đáng tin hơn nhiều so với chênh lệch J1/J2/J3 trước đây, nhưng chưa
   được xác nhận qua nhiều seed (nằm ngoài phạm vi J1-J3+S1-S4 hiện tại).
2. **Gộp 6 thay đổi trong 1 run** — không tách được đóng góp riêng của MFSR vs
   DCN vs domain-match vs STN pool vs EMA. Mục 3 chỉ tách được decode vs phần còn
   lại, không tách sâu hơn.
3. **best-of-55-epoch** — chọn model theo đúng epoch có val acc cao nhất trên
   cùng 999 track dùng để báo cáo, nên 79.78% có thể lạc quan hơn thực tế một
   chút. Tham khảo thêm epoch cuối (78.88%) như một mốc bảo thủ hơn.
4. **Compute chưa đo lại** — bảng chi phí compute ở `run_gpu.md` đo trên kiến
   trúc J2/J3 cũ (chưa có MFSR + STN pool 4x8). Params đã tăng nhẹ (29.58M so
   với 29.43M của J2) nhưng FLOPs/latency thật của S1 chưa được benchmark.
5. **Val acc dùng trọng số EMA**, không phải trọng số gốc — `Trainer.validate()`
   và `save_model()` đều dùng `self._eval_model()` trả về `ema.module` khi
   `use_ema=True`. Số liệu trong báo cáo là của model đã làm mượt qua EMA.
6. **Constrained decode có nhánh fallback** (mục 5) — không đảm bảo 100% output
   đúng layout trong mọi trường hợp.

---

## 7. Kết luận

Với đúng phạm vi J1→J2→J3→S1: J1/J2/J3 chênh nhau trong biên nhiễu ±13 track,
không đủ để kết luận GroupNorm/SR/DCN có tác dụng thật. **S1 là cấu hình đầu
tiên vượt biên nhiễu đó một cách rõ ràng** (+26 track so với J2, +27 so với
baseline gốc 77.00%), và phần lớn cải thiện đến từ kiến trúc/training (đo được
+24 track ngay cả khi so sánh thuần greedy), không phải từ decode. SR trong S1
cũng cho bằng chứng học được thật (vượt bilinear, khoảng cách nới rộng theo
thời gian) — khác hẳn kết luận "chưa chứng minh được giá trị" đã đặt ra cho J2.

**Cập nhật — S2 (λ_SR=0.5) đã chạy** (xem [s2_lam05_mf_sr_ocr.md](s2_lam05_mf_sr_ocr.md)):
794/999 (79.48%), kém S1 đúng 3 track — nằm sâu trong biên nhiễu ±13, tức tăng
`λ_SR` từ 0.1 lên 0.5 **không cải thiện**. Trả lời được câu hỏi để ngỏ ở trên:
trong 2 điểm đã đo của khoảng đề xuất `λ_SR ∈ [0.1, 0.5]`, đầu thấp (0.1, cấu hình
S1) vẫn tốt hơn hoặc bằng đầu cao — không có lý do đổi sang 0.5. Phát hiện phụ:
196/999 track đổi dự đoán giữa S1/S2 dù điểm tổng gần hoà, gợi ý hướng ensemble
thay vì chỉ chọn 1 cấu hình theo điểm tổng.

**Cập nhật — S3 (+L_Perceptual α=0.1) đã chạy** (xem
[s3_perceptual_mf_sr_ocr.md](s3_perceptual_mf_sr_ocr.md)): 805/999 (80.58%),
**hơn S1 8 track** — điểm cao nhất trong toàn bộ lịch sử thử nghiệm, nhưng 8 track
vẫn nằm trong biên nhiễu ±13 nên chưa đủ để khẳng định chắc chắn tốt hơn S1 (khác
S2, vốn cho kết quả âm tính rõ ràng). Đây là tín hiệu tích cực đầu tiên kể từ S1
đáng chạy multi-seed cùng S1 thay vì chỉ multi-seed riêng S1.

**Cập nhật — S4 (sr-scale 1 + width_downsample 4) đã chạy** (xem
[s4_sr_scale1_mf_sr_ocr.md](s4_sr_scale1_mf_sr_ocr.md)): 805/999 (80.58%) —
**hoà tuyệt đối với S3**, đạt bằng một cơ chế hoàn toàn khác (không phóng to ảnh,
giữ `T=32` qua thay đổi backbone thay vì qua SR). Đây là bằng chứng gián tiếp
quan trọng: `T=32` (nhiều bước CTC hơn), không phải bản thân việc SR phóng to
ảnh, mới có thể là yếu tố chính đứng sau lợi ích đo được của S1 so với J2/J3
(vốn chạy ở `T=16`). Cần thêm 1 ablation "`width_downsample=4` không SR" để tách
dứt điểm — chưa nằm trong phạm vi S1-S4 đã hoàn thành.

Cả 4 cấu hình S1-S4 theo kế hoạch trong `run_gpu.md` đã chạy xong. Bước tiếp
theo: multi-seed (O1) trên **S1, S3, S4** (3 ứng viên tốt nhất hiện tại, không
chỉ S1) để biết cấu hình nào thật sự tốt hơn một cách đáng tin cậy, rồi Chart +
Submission. Bảng so sánh đầy đủ cả 4: [model_comparison_summary.md](model_comparison_summary.md).
