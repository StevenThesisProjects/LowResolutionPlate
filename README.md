# J1 — GroupNorm Ablation (Root Cause #3, issue #9)

> Đối chứng bắt buộc cho hướng SR per-frame có giám sát — xem tổng quan issue tại [../../report/training_runs/run_gpu.md](../training_runs/run_gpu.md). Không có GroupNorm, backbone ResBlock bỏ BatchNorm mà không thay thế gì cả (Root Cause #3 theo phân tích issue #9), dẫn tới scale drift bị phơi bày rõ nhất khi SR nhân đôi kích thước không gian đầu vào (32×128 → 64×256).

## 1. Vì sao cần thí nghiệm này

Chẩn đoán NaN (I-a/I-b, xem `run_gpu.md`) đã xác nhận: bật SR trên backbone `norm=none` gây NaN toàn epoch. Thêm GroupNorm (`--backbone-norm group`) sửa được NaN. Nhưng GroupNorm **tự nó** có thể đã cải thiện accuracy — nếu không đo riêng, khi J2 (SR + GroupNorm) vượt baseline sẽ không biết công lao thuộc về SR hay GroupNorm. J1 tách bạch điều đó: chạy GroupNorm **một mình, không SR**.

## 2. Cấu hình

```bash
python train.py \
 --preset stable \
 --experiment-name crnn_resblock_groupnorm_nosr \
 --backbone-norm group \
 --num-workers 8 --aug-level full
```

Preset `stable`: batch 64, 80 epoch, lr 8e-4, AdamW + `WarmupCosineScheduler`, patience 18. GPU: RTX 4090, 1:31/epoch.

## 3. Kết quả

| Cấu hình | Val Exact Match | Chênh lệch |
|---|---:|---:|
| Baseline ResBlock (`norm=none`) | 76.68% | — |
| **J1 — + GroupNorm** | **76.88%** | **+0.20** |

Best ở epoch 60/80; early stopping kích hoạt ở epoch 78 (18 epoch không cải thiện).

**+0.20 điểm nằm trong biên độ nhiễu** — validation chỉ 999 sample → CI 95% ≈ ±2.7 điểm. Không đủ để khẳng định GroupNorm cải thiện accuracy một cách có ý nghĩa thống kê. Giá trị thật của J1 là làm **mốc đối chứng**: từ nay J2 phải so với 76.88%, không phải 76.68%.

## 4. Phát hiện quan trọng hơn: overfit rõ rệt

| Epoch | Train Loss | Val Loss | Val Acc |
|---:|---:|---:|---:|
| 19 | 0.0788 | **0.2591** ← đáy | 71.37% |
| 36 | 0.0281 | 0.2933 | 74.97% |
| 60 | 0.0074 | 0.3308 | **76.88%** ← best acc |
| 78 | 0.0042 | 0.3535 | 75.88% |

Val loss chạm đáy từ epoch 19 rồi tăng liên tục 36% tới epoch 78, trong khi train loss gần bằng 0 (model thuộc lòng tập train). Val acc vẫn nhích lên dù val loss tăng — model đoán đúng nhiều hơn nhưng khi sai thì sai với độ tự tin cao hơn.

## 5. Mốc so sánh theo ngân sách epoch

Dùng khi một run khác chạy ít epoch hơn 80 — so với đúng epoch tương ứng, không so thẳng với 76.88%:

| Ngân sách | J1 best trong khoảng đó | Đạt ở epoch |
|---|---:|---:|
| 30 epoch | 73.07% | 29 |
| 50 epoch | 74.97% | 36 |
| **60 epoch** | **76.88%** | **60** |
| 80 epoch (đầy đủ) | 76.88% | 60 |

## 6. Kết luận

- GroupNorm là bản sửa đúng cho Root Cause #3 — không chỉ hết NaN mà accuracy cũng nhích lên (+0.20, dù trong biên độ nhiễu), gần như miễn phí về compute (+15K params, FLOPs không đổi — xem bảng chi phí trong `run_gpu.md`).
- Model overfit rõ sau epoch ~20; 80 epoch là dư cho dataset ~19,000 track. Dư địa cải thiện tiếp theo nên nhắm vào **chống overfit** (augmentation mạnh hơn, dropout, weight decay) hơn là train lâu hơn hoặc thêm tham số.
- J2 (SR + GroupNorm) phải so với **76.88%** ở cùng ngân sách epoch để kết luận công lao có thực sự thuộc về SR hay không.

<img width="1140" height="165" alt="image" src="https://github.com/user-attachments/assets/8b2b3855-5fc2-45f1-9f61-c69dee36f217" />


