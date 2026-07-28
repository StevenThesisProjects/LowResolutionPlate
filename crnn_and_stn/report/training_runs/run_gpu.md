# Lệnh chạy GPU

Chỉ lệnh + kết quả. Phân tích xem tại [../baseline1_crnn_stn/](../baseline1_crnn_stn/).

> **Đọc trước khi so số:** validation chỉ **999 track** → 1 track = 0.1001 điểm,
> độ lệch chuẩn nhị thức ở mức 77% là **±1.3 điểm (±13 track)**. Mọi chênh lệch
> dưới ngưỡng đó không phân biệt được với nhiễu. Xem [Kết quả đã đo](#kết-quả-đã-đo).

---

## Setup môi trường GPU

```bash
unzip crnn_and_stn.zip -d crnn_and_stn
cd crnn_and_stn

nvidia-smi                       # xem "CUDA Version" góc trên phải trước khi chọn index

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124    # driver 12.1-12.3 -> đổi cu121
pip install albumentations opencv-python tqdm numpy

apt update && apt install -y libgl1
```

Nếu dùng Docker template `pytorch/pytorch` thì **bỏ dòng cài torch** — image đã có
sẵn torch + torchvision, cài đè dễ kéo về bản CUDA lệch driver. Kiểm tra trước:

```bash
python -c "
import torch, torchvision
from torchvision.ops import DeformConv2d      # --use-dcn phụ thuộc cái này
print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
```

**Chỉ khi gặp `CUDA out of memory`** mới thêm tiền tố dưới đây vào lệnh train. Nó
là cờ chống phân mảnh của allocator PyTorch — không đổi kết quả, không đổi tốc độ,
chỉ cần khi VRAM eo hẹp (bản J2 cũ chạy GPU 24GB nên phải dùng). Với V100 32GB ở
batch 32 + accum 2 thì không cần:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py ...
```

---

## Sanity check trước mọi run dài

**Đừng dùng `--preset debug` để kiểm tra sức khoẻ training.** Preset đó dùng batch
8 + LR 1e-3 (tương đương LR ~8e-3 ở batch 64) trên backbone 7.4M param — CTC kẹt ở
blank plateau vì chính hyperparameter, không nói lên điều gì về cấu hình thật.
Dùng đúng cấu hình S1 rút ngắn còn 3 epoch, vì có sẵn mốc đối chiếu từ J2:

| | J2 (mốc tham chiếu) |
|---|---|
| epoch 1 | train 2.885 · val 2.037 · **0.00%** |
| epoch 2 | train 1.204 · val 0.768 · **31.83%** |
| epoch 3 | train 0.566 · val 0.512 · **49.15%** |

```bash
python train.py \
  --preset stable --experiment-name sanity --epochs 3 \
  --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --num-workers 8 --aug-level full
```

Đạt yêu cầu nếu **epoch 2 vượt ~25%** và cột `sr_loss` trong
`results/history_sanity.csv` **giảm** qua từng epoch. Nếu epoch 2 vẫn 0% thì dừng
lại tìm nguyên nhân, đừng chạy tiếp 60 epoch.

---

## Bước 0 — Chấm lại checkpoint đã có (KHÔNG tốn GPU, chạy trước tiên)

Toàn bộ 20.000 nhãn dài đúng 7 ký tự và khớp 1 trong 2 layout Brazil
(`LLLNLNN` 13.000, `LLLNNNN` 7.000) → **6/7 vị trí bị khoá cứng lớp chữ/số**.
Greedy decode hiện tại không hề dùng ràng buộc này. `eval_decode.py` chấm lại
checkpoint cũ với cả hai kiểu decode nên biết ngay ràng buộc đáng bao nhiêu track
trước khi tiêu thêm giờ GPU nào. Kiến trúc được suy ra từ state_dict nên không
cần nhớ đã train bằng flag gì.

```bash
## P0 — greedy vs constrained trên J2 (chưa chạy)
python tools/eval_decode.py --checkpoint results/crnn_resblock_sr_supervised_best.pth

## P0b — ensemble nhiều seed, cũng so luôn 2 kiểu decode (chạy sau O1)
python tools/eval_decode.py --checkpoint "results/best_seed*.pth" \
  --save-submission results/submission_ensemble.txt
```

---

## Bước 1 — Proposed Method: Joint End-to-End MF-SR-OCR

Đúng sơ đồ Step 1-5 của issue #9:

```
5 LR frames [B,5,3,32,128]
   │
   ├─ Step 1  STN per-frame ────────────────────── --stn-pool 4,8 (mặc định)
   ├─ Step 2  DCNv2 alignment ──────────────────── --use-dcn
   ├─ Step 3  MFSR head -> [B*5,3,64,256] ──────── --use-sr --sr-scale 2
   │            └─ L_SR = ‖I_SR − I_HR‖₁ + α·L_Perceptual
   ├─ Step 4  OCR backbone + Attention Fusion
   └─ Step 5  BiLSTM + CTC ──────────────────────  L_Total = L_CTC + λ_SR·L_SR
```

Khác J2 ở những điểm sau, tất cả đều là hệ quả trực tiếp của review:

| Flag | Sửa vấn đề gì |
|---|---|
| `--use-dcn` | Step 2 của sơ đồ. Kernel giờ identity-init; bản J3 cũ init ngẫu nhiên nên phá ảnh RGB ngay epoch 0 |
| *(mặc định)* MFSR | Step 3 nói **Multi-Frame** SR; bản cũ là single-frame SR chạy 5 lần, không thể thêm thông tin |
| *(mặc định)* `--sr-edge-weight 0` | `L_SR` giờ đúng bằng công thức review (L1 + α·Perceptual), không còn số hạng edge tự thêm |
| `--lr-domain-match` | Nguyên nhân #4 — nhiễu synthetic sinh ở sai thang không gian |
| *(mặc định)* `--stn-pool 4,8` | Mục A của review; pool `(1,1)` là global-average, STN mất sạch thông tin không gian |
| `--decode constrained` | Greedy được phép xuất sai độ dài / sai lớp ký tự |
| `--use-ema` | Val acc dao động ~1 điểm giữa các epoch liền nhau |

```bash
## S1 — Proposed Method đầy đủ (λ_SR = 0.1, đầu khoảng review đề xuất)
python train.py \
  --preset stable --experiment-name s1_proposed \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s1.txt

## S2 — λ_SR = 0.5 (đầu kia của khoảng 0.1 → 0.5)
python train.py \
  --preset stable --experiment-name s2_proposed_lam05 \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.5 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s2.txt

## S3 — + L_Perceptual (số hạng α trong công thức L_SR)
python train.py \
  --preset stable --experiment-name s3_proposed_perceptual \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 --sr-perceptual-weight 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s3.txt
```

### Đọc `results/history_*.csv` trong lúc chạy

```
epoch,train_loss,val_loss,val_acc,val_acc_greedy,lr,sr_loss,sr_loss_bilinear,nan_batches
```

| Cột | Đọc thế nào |
|---|---|
| `val_acc` vs `val_acc_greedy` | Khoảng cách = giá trị thật của constrained decode. Ở sanity run 3 epoch: +1.90 (ep2) → +3.71 (ep3), tức nới rộng khi model khá lên |
| **`sr_loss` vs `sr_loss_bilinear`** | Câu hỏi trung tâm của cả chương. `sr_loss_bilinear` là L1 giữa **bản upscale bilinear thuần** và ảnh HR — tức điểm số mà module SR phải vượt qua mới coi là có đóng góp. `sr_loss ≥ sr_loss_bilinear` kéo dài = nhánh SR học được **không hơn nội suy**, trong khi ngốn 3.66x compute |
| `nan_batches` | Phải luôn bằng 0 |

Nếu `sr_loss` không bao giờ xuống dưới `sr_loss_bilinear`, đó là kết luận mạnh cho
báo cáo — mạnh hơn hẳn "SR cho +0.3 điểm nằm trong biên nhiễu" — và `S4`
(`--sr-scale 1`) chuyển từ tuỳ chọn thành run bắt buộc.

### Một ràng buộc của dataset cần biết trước khi đọc kết quả SR

Ảnh HR gốc chỉ ~**115x42 px**, trong khi input model đã là 128x32 và target SR ×2
là 256x64. Tức **55% chiều rộng của target là nội suy bicubic thuần tuý** — không
có thông tin thật nào ở đó để khôi phục. Sơ đồ review viết `[B, 3, 2H, 2W]` nhưng
dataset không có ground truth tương ứng với ×2.

`--sr-scale 1` giữ output ở 32x128, xấp xỉ đúng độ phân giải HR gốc, nên mọi pixel
target đều là tín hiệu thật. Đây không phải thay thế cho S1 mà là phép đo xem giới
hạn đó có thật sự chặn SR hay không:

```bash
## S4 — SR ở đúng thang thông tin thật của dataset (T=32 nhờ --width-downsample 4)
python train.py --preset stable --experiment-name s4_sr_scale1 \
  --use-sr --sr-scale 1 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --width-downsample 4 \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full 2>&1 | tee results/log_s4.txt
```

---

## Bước 2 — Ablation

Mọi run dưới đây tháo **đúng một** thành phần khỏi S1, để chênh lệch quy được về
thành phần đó. Đặt chung `SR_BASE` cho gọn và để không lệch flag giữa các arm.

```bash
SR_BASE="--preset stable --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --decode constrained \
  --num-workers 8 --aug-level full"
```

### Ablation 1 — hàm loss SR (checklist review)

```bash
## A1a — L_CTC thuần: module SR vẫn có nhưng KHÔNG được giám sát pixel.
##       Đây là bản tái hiện có kiểm soát của lỗi PR #7 (Nguyên nhân #2).
python train.py $SR_BASE --lambda-sr 0 --experiment-name abl1_lambda_0

## A1b — quét λ_SR
for lam in 0.01 0.1 0.5 1.0; do
  python train.py $SR_BASE --lambda-sr $lam --experiment-name "abl1_lambda_${lam}"
done

## A1c — thêm số hạng Sobel-edge (KHÔNG có trong công thức review; đo riêng)
python train.py $SR_BASE --sr-edge-weight 0.5 --experiment-name abl1_edge
```

### Ablation 2 — cơ chế fusion / alignment (checklist review)

```bash
## A2a — Average vs Max vs Attention
for mode in avg max; do
  python train.py $SR_BASE --fusion-mode $mode --experiment-name "abl2_fusion_${mode}"
done

## A2b — bỏ DCNv2 (tách riêng đóng góp của Step 2)
python train.py --preset stable --use-sr --sr-scale 2 --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match --decode constrained \
  --num-workers 8 --aug-level full --experiment-name abl2_no_dcn

## A2c — SR per-frame thay vì multi-frame (tách riêng đóng góp của Step 3)
python train.py $SR_BASE --sr-single-frame --experiment-name abl2_sr_single_frame
```

### Ablation 3 — số frame (checklist review)

```bash
for nf in 1 2 3 4 5; do
  python train.py $SR_BASE --num-frames $nf --experiment-name "abl3_frames_${nf}"
done
```

### Ablation 4 — độ phân giải input (checklist review)

```bash
python train.py $SR_BASE --img-height 16 --img-width 48 --experiment-name abl4_res_16x48
python train.py $SR_BASE --img-height 24 --img-width 96 --experiment-name abl4_res_24x96
## 32x128 chính là S1, không cần chạy lại
```

### Ablation bổ sung — hai confound phát hiện khi audit code

```bash
## A5 — số timestep CTC. Bật SR làm input rộng gấp đôi nên T nhảy 16 -> 32;
##      J2 hơn J1 3 track có thể chỉ vì T chứ không vì SR. Run này cho T=32
##      mà KHÔNG bật SR, nên tách được hai hiệu ứng.
python train.py --preset stable --experiment-name abl5_t32_nosr \
  --backbone-norm group --width-downsample 4 --lr-domain-match \
  --decode constrained --num-workers 8 --aug-level full

## A6 — STN pool: bằng chứng định lượng cho lỗi global-average (Mục A review)
python train.py $SR_BASE --stn-pool 1,1 --experiment-name abl6_stn_pool_1x1
```

### Baseline 3 của checklist — chưa triển khai

"Pre-trained EDSR / Real-ESRGAN (upscale ảnh trước) + CRNN + STN" cần tải trọng số
pretrain và một bước tiền xử lý offline ghi ảnh SR ra đĩa; chưa có trong repo.
Lưu ý khi làm: EDSR/Real-ESRGAN pretrain trên ảnh tự nhiên downscale bicubic, còn
LR ở đây là ảnh camera thật ~46x19 đã nén JPEG — sai domain, nên kết quả phải đọc
như một mốc tham chiếu chứ không phải trần trên.

---

## Bước 3 — Multi-seed, ensemble, benchmark

```bash
## O1 — 3 seed cho cấu hình thắng ở Bước 1 (giả định là S1)
for seed in 42 100 2026; do
  python train.py \
    --preset stable --experiment-name "best_seed${seed}" --seed $seed \
    --epochs 60 --batch-size 32 --grad-accum-steps 2 \
    --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
    --backbone-norm group --lr-domain-match \
    --decode constrained --use-ema \
    --num-workers 8 --aug-level full 2>&1 | tee "results/log_seed${seed}.txt"
done

## O2 — mean ± std + kiểm định so với baseline
python tools/aggregate_seeds.py --from-logs "results/log_seed*.txt" \
  --label "Proposed MF-SR-OCR" \
  --baseline-acc 76.88 --baseline-label "J1 GroupNorm no-SR"

## O3 — ensemble 3 seed (trung bình log-prob rồi mới decode)
python tools/eval_decode.py --checkpoint "results/best_seed*.pth" \
  --save-submission results/submission_ensemble.txt

## O4 — params / FLOPs / latency
python tools/benchmark.py --all

## O5 — minh hoạ định tính: LR -> SR -> attention -> prediction
python tools/visualize.py --checkpoint results/s1_proposed_best.pth \
  --use-sr --use-dcn --backbone-norm group --decode constrained \
  --num-samples 12 --output-dir results/viz
python tools/visualize.py --checkpoint results/s1_proposed_best.pth \
  --use-sr --use-dcn --backbone-norm group --decode constrained \
  --only-errors --num-samples 20 --output-dir results/viz_errors
```

---

## Bước 4 — Submission

```bash
python train.py \
  --submission-mode --preset stable --experiment-name submission_final \
  --epochs 60 --batch-size 32 --grad-accum-steps 2 \
  --use-sr --sr-scale 2 --use-dcn --lambda-sr 0.1 \
  --backbone-norm group --lr-domain-match \
  --decode constrained --use-ema \
  --num-workers 8 --aug-level full
```

`--submission-mode` / `--full-train` bỏ validation split → không có accuracy
trong lúc train. Chỉ chạy sau khi đã chốt cấu hình ở Bước 1-3.

---

## Cách báo cáo số liệu cho trung thực

Ba giới hạn phương pháp cần nêu trong phần Limitations, không được lờ đi:

**1. Chọn model và báo cáo trên cùng một tập val.** Checkpoint tốt nhất được chọn
theo val acc qua 60 epoch, rồi chính con số đó được báo cáo — đây là **max của 60
lần rút tương quan**, lệch lạc quan cỡ biên nhiễu (~1-2 điểm với val 999 sample).
Cách sửa miễn phí: lấy từ `history_*.csv` ra **3 con số** thay vì 1 —

```bash
python - <<'PY'
import csv
rows=list(csv.DictReader(open("results/history_s1_proposed.csv")))
acc=[float(r["val_acc"]) for r in rows]
print(f"best        : {max(acc):.2f}%  (epoch {acc.index(max(acc))+1})")
print(f"epoch cuối  : {acc[-1]:.2f}%")
print(f"tb 5 ep cuối: {sum(acc[-5:])/5:.2f}%")
PY
```

**2. `cudnn.benchmark=True` khiến run không tất định.** Cùng seed 42, hai lần chạy
từng lệch **6.5 điểm** ở epoch 3 (ghi nhận trong nhánh J). "Seed 42" không đảm bảo
tái lập; multi-seed (O1) hấp thụ được phần này nhưng phải nói rõ.

**3. Không có test set có nhãn.** Bộ test của challenge không kèm label nên val
999 track phải gánh cả vai chọn model lẫn vai báo cáo. Không tránh được, nhưng
phải thừa nhận thay vì trình bày val acc như test acc.

---

## Kết quả đã đo

Val = 999 track Scenario-B. Cột "track đúng" là con số thật; phần trăm chỉ là nó
chia cho 999. **±13 track = ±1.3 điểm là biên nhiễu**, chênh lệch nhỏ hơn không
kết luận được từ 1 seed.

| Run | Cấu hình | Track đúng | Val Acc |
|---|---|---:|---:|
| — | CRNN + STN (report ICPR của tác giả gốc) | — | 77.00% |
| — | CRNN + STN (đo lại trên dataset project) | 757 | 75.78% |
| — | aug light | 739 | 73.97% |
| SR-v1 | stacked-input SR (bản lỗi, Root Cause #1) | 492 | 49.25% |
| SR-v2 | stacked-input SR, lr thấp + aug light | 550 | 55.06% |
| — | ResBlock backbone, `norm=none` | 766 | 76.68% |
| J1 | + GroupNorm, không SR | **768** | 76.88% |
| J2 | + SR per-frame có giám sát | **771** | **77.18%** |
| J3 | + DCNv2 (kernel init ngẫu nhiên — đã sửa) | **762** | 76.28% |

J2 hơn J1 đúng **3 track**, J3 kém J2 **9 track** — cả hai đều nằm trong biên
nhiễu ±13. Chưa cấu hình nào vượt rõ ràng mốc 77.00% của tác giả gốc.

### Chi phí compute (`tools/benchmark.py --all`, CPU)

| Cấu hình | Params | GFLOPs/track | Latency (ms) | vs baseline |
|---|---:|---:|---:|---:|
| ResBlock (không SR) | 29,298,220 | 25.90 | 53.35 | 1.00x |
| + GroupNorm | 29,313,452 | 25.90 | 52.84 | 0.99x |
| + SR per-frame | 29,426,895 | 107.99 | 195.04 | 3.66x |
| + SR + DCNv2 | 29,445,854 | 108.75 | 201.60 | 3.78x |

Số đo lại sau khi thêm MFSR + STN pool 4x8 sẽ khác chút; chạy lại `--all` sau S1.
Cần nêu rõ trong báo cáo: ở J2, SR đổi **3.66x compute** lấy **3 track** — nếu S1
không kéo được khoảng cách này ra ngoài biên nhiễu ±13 track thì kết luận trung
thực là "SR chưa chứng minh được giá trị trên dataset này", chứ không phải "SR có
hiệu quả nhẹ".

---

## Lệnh cũ đã bỏ

Nhóm tuning AdamW (`--weight-decay`, `--onecycle-pct-start`) dùng flag không còn
tồn tại trên code hiện tại nên không chạy lại được; kết quả giữ ở
[../baseline1_crnn_stn/optimizer_adamw_verification.md](../baseline1_crnn_stn/optimizer_adamw_verification.md).
Tóm tắt: param-grouping 75.98%, wd 5e-4 76.18%, pct_start 0.15 76.28%,
60 epoch 75.08% (overfit) — không hướng nào vượt được mặc định.

Preset `strong` (120 epoch) cũng bỏ: cả J1 lẫn J2 đều cho thấy val loss chạm đáy
ở epoch 15-19 rồi tăng dần trong khi train loss về gần 0. Train lâu hơn chỉ
overfit thêm; dư địa nằm ở chống overfit và ở decode, không ở số epoch.
