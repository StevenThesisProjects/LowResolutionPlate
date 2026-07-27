# Lệnh chạy training trên GPU + log kết quả

Chỉ liệt kê lệnh + kết quả. Phân tích/kết luận xem tại [../baseline1_crnn_stn/](../baseline1_crnn_stn/) (đặc biệt [groupnorm_sr_ablation_j1_j2.md](../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md) cho hướng đang làm).

## Setup môi trường GPU

```bash
## giải nén tại chỗ
unzip crnn_and_stn.zip

## giải nén thư mục cùng tên
unzip crnn_and_stn.zip -d crnn_and_stn

## install pip
python -m pip install --upgrade pip

## kiểm tra CUDA driver trên máy GPU thuê trước khi chọn index (góc trên phải "CUDA Version")
nvidia-smi

## install torch — dùng cu124 để khớp pyproject.toml (index "pytorch-cu124", template EZYCLOUDX nvidia/cuda:12.4.1)
## Nếu nvidia-smi báo CUDA driver 12.1-12.3 (thấp hơn 12.4), đổi lại "cu121" trong URL bên dưới.
python -m pip install torch torchvision torchaudio \
 --index-url https://download.pytorch.org/whl/cu124

cd /home/crnn_and_stn

pip install albumentations opencv-python tqdm numpy

apt update
apt install -y libgl1
```

## Baseline 1 gốc (CRNN + STN, trước ResBlock)

```bash
## => 75.78%
python train.py --experiment-name crnn_stn_pytorch \
 --batch-size 64 --epochs 30 --lr 0.0005 --num-workers 8 --aug-level full

## => 75.08%
python train.py --experiment-name crnn_stn_pytorch \
 --batch-size 96 --epochs 50 --lr 0.001 --num-workers 8 --aug-level full

## aug light => 73.97%
python train.py --experiment-name exp_light_aug \
 --batch-size 64 --epochs 30 --lr 0.0005 --num-workers 8 --aug-level light
```

## AdamW tuning trên backbone gốc (archived — flag không còn tồn tại trên code hiện tại)

Chi tiết: [../baseline1_crnn_stn/optimizer_adamw_verification.md](../baseline1_crnn_stn/optimizer_adamw_verification.md).

```bash
## A. param-grouping => 75.98%
python train.py --experiment-name crnn_stn_adamw_paramgroup \
 --batch-size 64 --epochs 30 --lr 0.0005 --num-workers 8 --aug-level full

## B. wd 5e-4 + pct_start 0.15 (2 biến cùng lúc) => 75.58%
python train.py --experiment-name crnn_stn_adamw_tuned \
 --batch-size 64 --epochs 30 --lr 0.0005 \
 --weight-decay 0.0005 --onecycle-pct-start 0.15 --num-workers 8 --aug-level full

## C. chỉ wd 5e-4 => 76.18%
python train.py --experiment-name crnn_stn_adamw_wd_only \
 --batch-size 64 --epochs 30 --lr 0.0005 --weight-decay 0.0005 --num-workers 8 --aug-level full

## D. chỉ pct_start 0.15 => 76.28% (tốt nhất hướng này)
python train.py --experiment-name crnn_stn_adamw_warmup_only \
 --batch-size 64 --epochs 30 --lr 0.0005 --onecycle-pct-start 0.15 --num-workers 8 --aug-level full

## E. D + 60 epoch => 75.08% (overfit)
python train.py --experiment-name crnn_stn_adamw_warmup_long \
 --batch-size 64 --epochs 60 --lr 0.0005 --onecycle-pct-start 0.15 --num-workers 8 --aug-level full

## F. wd 2e-4 + pct_start 0.2 => 76.08%
python train.py --experiment-name crnn_stn_adamw_mild_combo \
 --batch-size 64 --epochs 30 --lr 0.0005 \
 --weight-decay 0.0002 --onecycle-pct-start 0.2 --num-workers 8 --aug-level full
```

## SR stacked-input (bản lỗi — bằng chứng Root Cause #1 issue #9)

```bash
## v1 => 49.25%
python train.py --experiment-name crnn_stn_sr \
 --batch-size 64 --epochs 30 --lr 0.0005 --num-workers 8 --aug-level full \
 --use-sr --sr-scale 2

## v2 => 55.06%
python train.py --experiment-name crnn_stn_sr \
 --batch-size 64 --epochs 30 --lr 0.0003 --num-workers 8 --aug-level light \
 --use-sr --sr-scale 2
```

## ResBlock backbone

```bash
## preset stable => ~76.68%
python train.py --preset stable --experiment-name crnn_resblock_stable \
 --batch-size 64 --epochs 80 --lr 0.0008 --grad-accum-steps 1 \
 --warmup-ratio 0.05 --min-lr-ratio 0.05 --patience 18 --num-workers 8 --aug-level full

## preset strong — chưa chạy
python train.py --preset strong --experiment-name crnn_resblock_strong \
 --batch-size 96 --epochs 120 --lr 0.0006 --grad-accum-steps 2 \
 --warmup-ratio 0.08 --min-lr-ratio 0.03 --patience 24 --num-workers 8 --aug-level full

## submission cuối — chỉ chạy sau khi chốt cấu hình tốt nhất
python train.py --submission-mode --preset strong \
 --experiment-name crnn_resblock_submission \
 --batch-size 96 --epochs 120 --lr 0.0006 --grad-accum-steps 2 \
 --warmup-ratio 0.08 --min-lr-ratio 0.03 --num-workers 8 --aug-level full
```

## J1/J2/J3 — GroupNorm + SR per-frame (đang làm, issue #9)

Phân tích + caveat thống kê đầy đủ: [../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md](../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md).

```bash
## I-a. Chẩn đoán NaN, tắt AMP => Train Loss hữu hạn (3.3354) — xác nhận AMP là nguyên nhân NaN
python train.py --preset stable --experiment-name crnn_resblock_sr_diag_noamp \
 --epochs 3 --use-sr --sr-scale 2 --lambda-sr 0.1 --no-amp --num-workers 8 --aug-level full

## I-b. GroupNorm + AMP => hết NaN, Val Loss epoch1 2.7588
python train.py --preset stable --experiment-name crnn_resblock_sr_supervised_sanity \
 --epochs 5 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group \
 --num-workers 8 --aug-level full

## J1. GroupNorm, không SR (đối chứng) => 76.88% (best epoch 60/80)
python train.py --preset stable --experiment-name crnn_resblock_groupnorm_nosr \
 --backbone-norm group --num-workers 8 --aug-level full

## J2. + SR per-frame + giám sát, GPU >= 32GB => 77.18% (best epoch 57/60)
python train.py --preset stable --experiment-name crnn_resblock_sr_supervised \
 --epochs 60 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee results/log_j2.txt

## J2-alt. Bản GPU 24GB (batch 32 + accum 2, effective batch = 64, tương đương J2) => 77.18%
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
 --preset stable --experiment-name crnn_resblock_sr_supervised \
 --epochs 60 --batch-size 32 --grad-accum-steps 2 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee results/log_j2.txt

## J3. + DCNv2 — chỉ chạy sau khi J2 >= J1 (đã thoả) + đã cân nhắc chạy O1 trước. Chưa chạy.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
 --preset stable --experiment-name crnn_resblock_sr_dcn \
 --epochs 60 --batch-size 32 --grad-accum-steps 2 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --use-dcn --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee results/log_j3.txt
```

## M. Domain match (Root Cause #4) — chưa chạy

```bash
python train.py --preset stable --experiment-name crnn_resblock_domainmatch \
 --backbone-norm group --lr-domain-match --num-workers 8 --aug-level full

python train.py --preset stable --experiment-name crnn_resblock_sr_domainmatch \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group --lr-domain-match \
 --num-workers 8 --aug-level full
```

## N. Ablation issue #9 — chưa chạy

```bash
## N1 — lambda_sr
for lam in 0.01 0.5 1.0; do
python train.py --preset stable --experiment-name "abl1_lambda_${lam}" \
 --use-sr --sr-scale 2 --lambda-sr $lam --backbone-norm group --num-workers 8 --aug-level full
done

## N1b — perceptual loss VGG16
python train.py --preset stable --experiment-name abl1_perceptual \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --sr-perceptual-weight 0.1 \
 --backbone-norm group --num-workers 8 --aug-level full

## N2 — fusion mode
for mode in avg max; do
python train.py --preset stable --experiment-name "abl2_fusion_${mode}" \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --fusion-mode $mode \
 --backbone-norm group --num-workers 8 --aug-level full
done

## N3 — số frame
for nf in 1 2 3 4; do
python train.py --preset stable --experiment-name "abl3_frames_${nf}" \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --num-frames $nf \
 --backbone-norm group --num-workers 8 --aug-level full
done

## N4 — độ phân giải
python train.py --preset stable --experiment-name abl4_res_16x48 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --img-height 16 --img-width 48 \
 --backbone-norm group --num-workers 8 --aug-level full

python train.py --preset stable --experiment-name abl4_res_24x96 \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --img-height 24 --img-width 96 \
 --backbone-norm group --num-workers 8 --aug-level full
```

## O. Multi-seed + benchmark + visualize — chưa chạy

```bash
## O1 — 3 seed cho cấu hình thắng
for seed in 42 100 2026; do
python train.py --preset stable --experiment-name "best_seed${seed}" --seed $seed \
 --use-sr --sr-scale 2 --lambda-sr 0.1 --backbone-norm group \
 --num-workers 8 --aug-level full 2>&1 | tee "results/log_seed${seed}.txt"
done

## O2 — tổng hợp mean ± std
python tools/aggregate_seeds.py --from-logs "results/log_seed*.txt" --label "SR + GroupNorm"

## O3 — params/FLOPs/latency, đã chạy sẵn trên CPU:
python tools/benchmark.py --all
```

| Cấu hình | Params | GFLOPs/track | Latency (ms) | vs baseline |
|---|---:|---:|---:|---:|
| Baseline ResBlock (không SR) | 29,298,220 | 25.90 | 53.35 | 1.00x |
| + GroupNorm | 29,313,452 | 25.90 | 52.84 | 0.99x |
| + SR (per-frame) | 29,426,895 | 107.99 | 195.04 | 3.66x |
| + SR + DCNv2 | 29,445,854 | 108.75 | 201.60 | 3.78x |

```bash
## O4 — minh hoạ định tính (chưa chạy)
python tools/visualize.py --checkpoint results/crnn_resblock_sr_supervised_best.pth \
 --use-sr --backbone-norm group --num-samples 12 --output-dir results/viz

python tools/visualize.py --checkpoint results/crnn_resblock_sr_supervised_best.pth \
 --use-sr --backbone-norm group --only-errors --num-samples 20 --output-dir results/viz_errors
```

## Ghi chú

- File này chỉ chứa lệnh chạy trên GPU (cloud/server), không dùng lệnh local để so kết quả chính thức.
- `--submission-mode` / `--full-train` không giữ validation split → không có accuracy trong lúc train.
