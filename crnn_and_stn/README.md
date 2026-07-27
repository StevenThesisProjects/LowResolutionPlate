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

# J2 — SR Per-Frame + Giám Sát Pixel-Level (Root Cause #1 & #2, issue #9)

> Thí nghiệm chính của hướng fix issue #9 — so với đối chứng [J1 (GroupNorm, không SR)](groupnorm_ablation_j1.md). Xem tổng quan checklist tại [../training_runs/run_gpu.md](../training_runs/run_gpu.md).

## 1. Mục tiêu

Đo tác động thật của module SR per-frame + `L_CTC + λ·L_SR` (λ=0.1) khi đã cô lập khỏi lợi ích của GroupNorm (đo riêng ở J1). Đây là câu hỏi cốt lõi: SR có thực sự giúp OCR đọc biển số tốt hơn, hay chỉ tốn thêm 3.66x compute mà không đổi lại gì?

## 2. Cấu hình

```bash
python train.py \
 --preset stable \
 --experiment-name crnn_resblock_sr_supervised \
 --epochs 60 \
 --batch-size 32 --grad-accum-steps 2 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 \
 --backbone-norm group \
 --num-workers 8 --aug-level full
```

Batch 32 + grad-accum 2 = effective batch 64 (giống J1), bắt buộc trên GPU 24GB vì SR phóng ảnh 32×128 → 64×256 (gấp 4 lần pixel) khiến batch 64 gốc bị OOM. Cùng seed 42, cùng preset `stable`, chỉ khác J1 ở `--use-sr --sr-scale 2 --lambda-sr 0.1`.

## 3. Kết quả

| Cấu hình | Val Exact Match | Epoch đạt | Chênh so với baseline |
|---|---:|---:|---:|
| Baseline ResBlock (`norm=none`, không SR) | 76.68% | — | — |
| J1 — + GroupNorm (không SR, mốc 60-epoch) | 76.88% | 60 | +0.20 |
| **J2 — + SR per-frame + giám sát** | **77.18%** | **57** | **+0.50** |

J2 vượt J1 (+0.30) và vượt baseline (+0.50) trong lần chạy này.

## 4. Caveat thống kê — quan trọng hơn con số 77.18%

**Trước khi chạy được J2 hoàn chỉnh, cùng một cấu hình đã chạy 2 lần (lần đầu bị dừng giữa chừng do lỗi hạ tầng, không phải do model) — và 2 lần cho kết quả lệch nhau tới 6.5 điểm ở epoch 3** (42.64% vs 49.15%), dù giống hệt config + seed. Nguyên nhân: `cudnn.benchmark=True` khiến thuật toán convolution không xác định (nondeterministic), cộng thứ tự dataloader.

Validation chỉ 999 sample → CI 95% ≈ ±2.7 điểm. Kết hợp cả 2 nguồn nhiễu này: **chênh lệch +0.30 giữa J2 và J1 không đủ để kết luận SR thực sự giúp ích** — có thể chỉ là may mắn của 1 lần chạy. Cần chạy **O1 (3 seed)** trước khi kết luận chắc chắn.

## 5. Overfit — lặp lại đúng pattern của J1

| Epoch | Train Loss | Val Loss | Val Acc |
|---:|---:|---:|---:|
| 15 | 0.1937 | **0.2773** ← đáy | 73.17% |
| 30 | 0.1144 | 0.3465 | 73.87% |
| 57 | 0.0754 | 0.3804 | **77.18%** ← best acc |
| 60 | 0.0748 | 0.3856 | 76.18% |

Val loss chạm đáy ở epoch 15 (sớm hơn J1 — epoch 19) rồi tăng 39% tới cuối, trong khi train loss gần 0. Cùng kết luận như J1: **80 (hay 60) epoch là dư, dư địa cải thiện nằm ở chống overfit chứ không phải kiến trúc**.

## 6. SR loss có giảm thật, chỉ bị che bởi nhiễu batch-cuối

Trung bình `sr_loss` 10 epoch đầu: **0.804** → 10 epoch cuối: **0.674** (giảm ~16%). Module SR đang học đúng hướng tái tạo ảnh HR, xu hướng giảm chỉ bị che khuất nếu chỉ nhìn giá trị batch-cuối từng epoch (dao động 0.61–0.90) thay vì trung bình cả epoch.

<img width="1552" height="789" alt="image" src="https://github.com/user-attachments/assets/bdd3b69a-d73f-4ff5-953b-7d758f505665" />
<img width="1488" height="974" alt="image" src="https://github.com/user-attachments/assets/4b400f5e-b0f4-40ca-849d-bbc6d7a9dcdd" />
<img width="1565" height="1250" alt="image" src="https://github.com/user-attachments/assets/e208fff5-e603-467b-93e1-2a26ba94a4c6" />


## 7. Kết luận & bước tiếp theo

- J2 (77.18%) thoả điều kiện `J2 >= J1` → **đủ điều kiện chạy J3** (+ DCNv2) theo checklist.
- GroupNorm (J1) và SR (J2) đều cho dấu hiệu tích cực đứng riêng; ưu tiên xác nhận bằng multi-seed hơn là cộng dồn thêm module mới (DCNv2) vào một kết quả chưa chắc chắn.

