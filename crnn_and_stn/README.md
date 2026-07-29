# GroupNorm + SR Per-Frame — Ablation J1 & J2 (Root Cause #1, #2, #3 — issue #9)

> Gộp từ 2 thí nghiệm liên tiếp trong nhánh fix issue #9. Checklist đầy đủ + lệnh chạy: [../training_runs/run_gpu.md](../training_runs/run_gpu.md).

**Baseline chuẩn theo report ICPR của tác giả gốc là CRNN + STN (77.00%)** — không phải ResBlock backbone (76.68%, cải tiến làm sau ở PR #8). Mọi so sánh dưới đây tách rõ 2 mốc để không nhầm "vượt ResBlock" thành "vượt baseline gốc".

## 1. Vì sao tách J1 khỏi J2

Chẩn đoán NaN trước đó xác nhận: bật SR trên backbone `norm=none` gây NaN toàn epoch; thêm GroupNorm (`--backbone-norm group`) sửa được. Nhưng GroupNorm **tự nó** có thể đã cải thiện accuracy, không liên quan gì tới SR. Nếu không đo riêng, khi cấu hình SR+GroupNorm vượt baseline sẽ không biết công lao thuộc về SR hay GroupNorm.

- **J1** — chỉ GroupNorm, không SR → đo riêng lợi ích của GroupNorm (sửa Root Cause #3).
- **J2** — GroupNorm + SR per-frame + `L_CTC + λ·L_SR` (sửa Root Cause #1 & #2) → đo thêm lợi ích của SR trên nền đã có J1.

## 2. Cấu hình

```bash
# J1 — GroupNorm, không SR (preset stable: batch 64, 80 epoch, lr 8e-4)
python train.py --preset stable --experiment-name crnn_resblock_groupnorm_nosr \
 --backbone-norm group --num-workers 8 --aug-level full

# J2 — + SR per-frame + giám sát (batch 32 + accum 2 = effective batch 64, bắt buộc
# trên GPU 24GB vì SR phóng ảnh 32x128 -> 64x256, gấp 4 lần pixel, gây OOM ở batch 64)
python train.py --preset stable --experiment-name crnn_resblock_sr_supervised \
 --epochs 60 --batch-size 32 --grad-accum-steps 2 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group \
 --num-workers 8 --aug-level full
```

Cùng seed 42; J2 chỉ khác J1 ở 3 flag `--use-sr --sr-scale 2 --lambda-sr 0.1`.

## 3. Kết quả

| Cấu hình                                                         | Val Exact Match | vs ResBlock | vs baseline chuẩn (77.00%) |
| ---------------------------------------------------------------- | --------------: | ----------: | -------------------------: |
| **CRNN + STN (baseline chuẩn, report ICPR)**                     |      **77.00%** |           — |                          — |
| CRNN + STN (đo thực tế trên dataset project)                     |          75.78% |           — |                      −1.22 |
| ResNet + Transformer + STN (report, tốt nhất trong report gốc)\* |          78.70% |           — |                      +1.70 |
| ResBlock backbone (PR #8, `norm=none`, **không phải baseline**)  |          76.68% |           — |                      −0.32 |
| J1 — ResBlock + GroupNorm                                        |          76.88% |       +0.20 |                      −0.12 |
| **J2 — + SR per-frame + giám sát**                               |      **77.18%** |   **+0.50** |                  **+0.18** |

\* Kiến trúc khác hẳn (ResNet+Transformer), không so trực tiếp được với nhánh CRNN đang làm.

J2 > J1 (+0.30) và > ResBlock (+0.50), nhưng so với **baseline chuẩn chỉ nhỉnh hơn +0.18** — và **J1 vẫn chưa vượt được baseline chuẩn**. So với mốc mạnh nhất trong report gốc (ResNet+Transformer+STN, 78.70%) J2 còn cách **1.52 điểm** — dù kiến trúc khác hẳn nên không so trực tiếp được, con số này cho thấy trần hiện tại của nhánh CRNN vẫn còn xa mốc cao nhất report từng đạt. Xem caveat ở mục 4 trước khi kết luận bất cứ điều gì từ các con số này.

## 4. Caveat thống kê — quan trọng hơn mọi con số ở trên

Trước khi chạy được J2 hoàn chỉnh, cùng một cấu hình (cùng seed) đã chạy 2 lần — lệch nhau **6.5 điểm** ở epoch 3 (42.64% vs 49.15%), do `cudnn.benchmark=True` khiến thuật toán convolution không xác định. Cộng thêm validation chỉ 999 sample → CI 95% ≈ ±2.7 điểm.

**Kết luận: chênh lệch +0.18–0.50 nêu ở mục 3 nằm gọn trong biên độ nhiễu đã đo được bằng thực nghiệm — chưa đủ bằng chứng để khẳng định GroupNorm hay SR thực sự cải thiện accuracy.** Cần chạy **O1 (3 seed: 42/100/2026)** cho cả J1 và J2 trước khi kết luận chắc chắn.

## 5. Overfit — cùng pattern ở cả 2 run

| Run | Val loss chạm đáy |      Best Val Acc | Train loss cuối |
| --- | ----------------- | ----------------: | --------------: |
| J1  | epoch 19 (0.2591) | 76.88% @ epoch 60 |          0.0074 |
| J2  | epoch 15 (0.2773) | 77.18% @ epoch 57 |          0.0748 |

Cả 2 run: val loss chạm đáy sớm rồi tăng dần trong khi train loss gần 0 (model thuộc lòng tập train), val acc vẫn nhích lên dù val loss tăng. **60-80 epoch là dư cho dataset ~19,000 track** — dư địa cải thiện nên nhắm vào chống overfit (augmentation mạnh hơn, dropout, weight decay) hơn là train lâu hơn hoặc thêm tham số/module mới.

## 6. SR loss của J2 — giảm thật, chỉ bị che bởi nhiễu batch-cuối

Trung bình `sr_loss` 10 epoch đầu: 0.804 → 10 epoch cuối: 0.674 (giảm ~16%). Module SR học đúng hướng tái tạo ảnh HR; xu hướng giảm chỉ bị che nếu nhìn giá trị batch-cuối từng epoch (dao động 0.61–0.90) thay vì trung bình cả epoch.

## 7. Kết luận & bước tiếp theo

- GroupNorm sửa đúng Root Cause #3 (hết NaN), gần như miễn phí compute (+15K params, FLOPs không đổi).
- SR per-frame + giám sát cho tín hiệu tích cực (+0.30 so với J1) nhưng tốn **3.66x FLOPs / latency** — xem bảng chi phí trong `run_gpu.md`.
- **Chưa cấu hình nào (J1 lẫn J2) vượt rõ ràng baseline chuẩn của tác giả (77.00%)** khi tính đến biên độ nhiễu — đây là điều quan trọng nhất cần nêu khi báo cáo, không nên nói "đã vượt baseline". Càng chưa gần mốc mạnh nhất report gốc (78.70%, ResNet+Transformer+STN — còn cách 1.52 điểm).
- Bước tiếp theo: chạy **O1 (multi-seed)** trước khi đầu tư thêm vào J3 (+DCNv2) — J2 kỹ thuật đủ điều kiện `J2 ≥ J1` để chạy J3, nhưng xây tiếp trên một kết quả 1-seed chưa xác nhận là rủi ro không đáng.
