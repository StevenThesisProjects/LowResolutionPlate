# AdamW cho Baseline 1 (STN + CRNN) — Xác nhận & Ablation (lịch sử)

> **Trạng thái: đã kết thúc, chỉ còn giá trị tham khảo.** Toàn bộ ablation dưới đây chạy trên **backbone CNN + BatchNorm gốc** (trước khi có ResBlock backbone ở PR #8). Backbone đó không còn tồn tại trong code hiện tại — kết quả tốt nhất tìm được ở đây (76.28%) đã bị ResBlock backbone (76.68%) vượt qua mà không cần tuning này. Xem [resblock_backbone_upgrade.md](../../../report/baseline1_crnn_stn/resblock_backbone_upgrade.md) và [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md) cho hướng đang active.

## 1. Xác nhận kiến trúc & optimizer

Đối chiếu source `crnn_and_stn` với report ICPR (kiến trúc, dataset, augmentation, training config trang 36-49): **khớp 100%**, không sai lệch — chi tiết đối chiếu từng module xem [baseline1_architecture.md](../../../report/baseline1_crnn_stn/baseline1_architecture.md).

Optimizer **AdamW đã có sẵn từ khi trích xuất Baseline 1** (không phải bổ sung mới), kèm `OneCycleLR`, `GradScaler`/`autocast` (mixed precision), `clip_grad_norm_(GRAD_CLIP=5.0)`:

```python
self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
self.optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.LEARNING_RATE,
    steps_per_epoch=len(train_loader), epochs=config.EPOCHS)
```

Đo thực tế trên dataset project (batch 64, epochs 30, lr 5e-4, khớp config report trang 48): **75.78%** — thấp hơn mốc report (77.00%) khoảng 1.2 điểm, trong biên độ hợp lý do khác dataset split/seed/hardware, không do sai kiến trúc.

## 2. Bug tìm thấy: weight decay áp sai lên bias/BatchNorm

`AdamW(model.parameters(), weight_decay=1e-4)` áp weight decay lên **toàn bộ** tham số kể cả bias và BatchNorm — anti-pattern đã biết, làm méo scale/shift mà BatchNorm cần học tự do. Sửa bằng param-grouping (tách theo `ndim <= 1` — bắt đúng mọi bias/BatchNorm 1-D):

```python
def build_optimizer_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (no_decay if param.ndim <= 1 else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
```

Cũng expose `betas`/`eps` (AdamW) và `pct_start`/`div_factor`/`final_div_factor` (OneCycleLR) ra CLI để ablation.

## 3. Ablation 7 cấu hình (A-F) — kết quả cuối

Nguyên tắc: mỗi lần chỉ đổi **một** biến so với baseline, cô lập biến khi kết quả bất ngờ (bài học rút ra và áp dụng lại cho ablation J1/J2 sau này).

| Cấu hình | weight_decay | pct_start | epochs | Val Acc | So với baseline (75.78%) |
|---|---:|---:|---:|---:|---:|
| Baseline gốc (bug: wd áp sai lên bias/BN) | 1e-4 | 0.3 | 30 | 75.78% | — |
| A — param-grouping (sửa bug) | 1e-4 | 0.3 | 30 | 75.98% | +0.20 |
| B — wd cao + warmup ngắn (2 biến cùng lúc) | 5e-4 | 0.15 | 30 | 75.58% | −0.20 |
| F — wd + warmup mức vừa (2 biến cùng lúc) | 2e-4 | 0.2 | 30 | 76.08% | +0.30 |
| C — chỉ tăng weight_decay | 5e-4 | 0.3 | 30 | 76.18% | +0.40 |
| **D — chỉ rút ngắn warmup** | **1e-4** | **0.15** | **30** | **76.28%** | **+0.50 (tốt nhất)** |
| E — D + train dài hơn | 1e-4 | 0.15 | 60 | 75.08% | −0.70 |

## 4. 3 bài học rút ra (vẫn áp dụng được cho công việc sau này)

1. **`weight_decay` cao và warmup ngắn đều tốt khi đứng riêng (C, D), nhưng không cộng dồn khi kết hợp** — cả B (mạnh) lẫn F (vừa) đều kém hơn D đứng một mình. Bài học chung: 2 cải thiện tốt riêng lẻ không tự động cộng dồn, phải test kết hợp thực tế chứ không suy diễn.
2. **Train dài hơn (60 epoch) làm giảm accuracy do overfit** (E: −0.70) — dataset ~19,000 track không đủ lớn để nuôi gấp đôi số bước gradient update. Cùng kết luận này lặp lại ở ablation J1/J2 sau này với ResBlock backbone.
3. **Caveat thống kê**: validation chỉ 999 sample → CI 95% ≈ ±2.7 điểm. Toàn bộ chênh lệch A-F (0.2-0.7 điểm) nằm trong biên độ này — không đủ để khẳng định chắc chắn D tốt hơn C về mặt thống kê, chỉ là D có số đo cao nhất trong 1 lần chạy. Đây là lý do ablation J1/J2 sau này đặt ra yêu cầu multi-seed (O1) trước khi kết luận.

## 5. Vì sao dừng ở đây

Mức trần đạt được bằng optimizer tuning trên backbone CRNN gốc là 76.28% — không tiệm cận 80% (mục tiêu ban đầu), và **thấp hơn ResBlock backbone (76.68%)** vốn không cần tuning này. Kết luận: giới hạn nằm ở kiến trúc backbone, không phải optimizer — hướng đi tiếp theo là nâng backbone (đã làm, xem [resblock_backbone_upgrade.md](../../../report/baseline1_crnn_stn/resblock_backbone_upgrade.md)) rồi SR có giám sát (đang làm, xem [groupnorm_sr_ablation_j1_j2.md](groupnorm_sr_ablation_j1_j2.md)), không phải tiếp tục tinh chỉnh optimizer trên backbone cũ.

Cấu hình D (`--onecycle-pct-start 0.15`, còn lại mặc định) là lựa chọn cuối cho backbone gốc nếu cần tái lập, nhưng **không dùng cho công việc hiện tại** (đã chuyển sang ResBlock + `WarmupCosineScheduler`, flag `--onecycle-pct-start` không tồn tại trong `train.py` hiện tại).
