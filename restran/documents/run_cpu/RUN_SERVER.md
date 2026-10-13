# Chạy ResTran trên server thuê — CLI

## 0. Cấu trúc gói upload

Code và dataset tách làm 2 gói vì dataset nặng 1,2 GB:

| Gói | Nội dung | Dung lượng |
|---|---|---|
| `restran_code.zip` | toàn bộ code + docs (không có dataset) | ~7 MB |
| `restran_dataset.tar` | thư mục `dataset/` | ~1,2 GB |

Tạo gói (chạy ở máy local, trong `NewClaude/`):

```bash
cd /Users/nguyentienphat/Learning/Paper/NewClaude

# Gói code (loại dataset, cache, kết quả)
zip -r restran_code.zip restran \
    -x "restran/dataset/*" "restran/__pycache__/*" "restran/*/__pycache__/*" \
       "restran/*/*/__pycache__/*" "restran/results/*" "restran/.venv/*"

# Gói dataset — tar không nén vì ảnh PNG/JPG đã nén sẵn, nén lại chỉ tốn thời gian
tar -cf restran_dataset.tar -C restran dataset
```

Upload:

```bash
scp -P <PORT> restran_code.zip restran_dataset.tar root@<SERVER_IP>:~/
```

---

## 1. Giải nén trên server

```bash
unzip restran_code.zip && cd restran
tar -xf ~/restran_dataset.tar          # bung vào restran/dataset/

# Kiểm tra dataset đủ chưa (phải ra đúng 20000 / 1000 / 3000)
find dataset/data/train -type d -name "track_*" | wc -l
find dataset/Pa7a3Hin-test-public -type d -name "track_*" | wc -l
find dataset/TKzFBtn7-test-blind  -type d -name "track_*" | wc -l
```

---

## 2. Cài môi trường

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nproc             # số CPU core -> dùng cho --num-workers
```

Chọn wheel theo **`compute_cap` (kiến trúc GPU)**, không phải theo "CUDA Version" ở góc phải `nvidia-smi`. CUDA Version đó chỉ là mức tối đa driver hỗ trợ; wheel thiếu kernel cho kiến trúc GPU vẫn cài và `import` bình thường rồi mới **sập ở lớp conv đầu tiên** (xem §2.1).

| `compute_cap` | GPU tiêu biểu | Lệnh cài torch |
|---|---|---|
| **12.0** (sm_120) | RTX 5090 / 5080 / RTX PRO Blackwell | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| **10.0** (sm_100) | B200 / GB200 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| **9.0** (sm_90) | H100 / H200 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| **8.9 / 8.6 / 8.0** | RTX 4090, L40S, RTX 3090, A10, A100 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| **7.5 / 7.0** | T4, V100 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |

Điều kiện kèm theo: driver phải **≥** phiên bản CUDA của wheel. Nếu `compute_cap` cao mà driver lại cũ thì phải đổi máy hoặc nhờ nhà cung cấp nâng driver — không có wheel nào cứu được.

Phần còn lại:

```bash
pip install albumentations opencv-python-headless numpy tqdm
```

> Dùng **`opencv-python-headless`** thì **không cần** `apt install libgl1`. Nếu bạn cài `opencv-python` bản thường thì phải thêm:
> ```bash
> apt update && apt install -y libgl1 libglib2.0-0
> ```
>
> Không cần `torchaudio`, `matplotlib`, `scikit-image` — pipeline này không import chúng.

Kiểm tra nhanh trước khi train:

```bash
python -c "
import torch
print('torch       :', torch.__version__, '| built w/ CUDA', torch.version.cuda)
print('GPU         :', torch.cuda.get_device_name(0))
print('GPU cap     : sm_%d%d' % torch.cuda.get_device_capability(0))
print('arch list   :', torch.cuda.get_arch_list())
cap = 'sm_%d%d' % torch.cuda.get_device_capability(0)
assert cap in torch.cuda.get_arch_list(), f'❌ {cap} KHONG co trong wheel -> xem §2.1'
print((torch.randn(8,8,device='cuda') @ torch.randn(8,8,device='cuda')).sum().item())
print('✅ kernel chay duoc tren GPU nay')
"
python -c "
import sys; sys.path.insert(0,'.')
import torch
from configs.config import Config
from src.models.restran import ResTranOCR
c=Config(); m=ResTranOCR(num_classes=c.NUM_CLASSES, use_stn=c.USE_STN).cuda()
y=m(torch.randn(2,5,3,32,128).cuda())
print('forward OK', tuple(y.shape), '| params %.2fM'%(sum(p.numel() for p in m.parameters())/1e6))
"
```
Kết quả đúng: `forward OK (2, 16, 37) | params 31.08M`

> ⚠️ **Đừng dùng `torch.cuda.is_available()` làm bằng chứng môi trường ổn.** Nó chỉ kiểm tra driver khởi tạo được, **không** kiểm tra wheel có kernel cho kiến trúc GPU — nên nó trả `True` rồi train vẫn sập ở conv đầu tiên. Phải chạy một phép toán thật trên GPU như trên.

### 2.1. Lỗi `no kernel image is available for execution on the device`

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Nghĩa là wheel PyTorch **không chứa kernel biên dịch cho kiến trúc GPU này**. Không liên quan tới code, batch size hay VRAM. Đối chiếu `sm_XX` của GPU với `torch.cuda.get_arch_list()`:

| Tình huống | Nguyên nhân | Cách sửa |
|---|---|---|
| `sm_XX` **cao hơn** mọi mục trong arch list | torch quá cũ so với GPU (hay gặp: RTX 50xx / B200 mà cài wheel cu124) | cài lại theo bảng §2, thường là `cu128`: <br>`pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `sm_XX` **thấp hơn** mọi mục trong arch list | GPU quá cũ, bản torch mới đã bỏ hỗ trợ | hạ torch hoặc dùng build cũ: <br>`pip install --force-reinstall "torch==2.4.*" torchvision --index-url https://download.pytorch.org/whl/cu121` |

Trên server thuê có sẵn venv của nhà cung cấp (ví dụ `/opt/jupyter_venv`), nhớ cài vào **đúng** interpreter đang chạy train:

```bash
which python && python -c "import torch, sys; print(sys.executable, torch.__file__)"
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Cài xong **chạy lại khối kiểm tra ở trên** trước khi train — assert sẽ bắt được ngay nếu vẫn lệch.

---

## 3. Smoke test (bắt buộc chạy trước khi để train 30 epoch)

Chạy 1 epoch với batch nhỏ để chắc chắn data + GPU + CTC không lỗi:

```bash
python train.py --epochs 1 --batch-size 32 --num-workers $(nproc) -n smoke --output-dir results_smoke
```

Kỳ vọng thấy:
```
[TRAIN] ... -> Total: 38002 samples.
[VAL]   ... -> Total: 999 samples.
📊 Model (restran): 31,082,... total params
Epoch 1/1: Train Loss: ... (min ... / max ... / std ...) | Val Loss: ... | Val Acc: ...% | LR: ... | Time: ...s (train ...s / val ...s)
🧾 Saved run summary: results_smoke/smoke_summary.txt
```

Kiểm tra các file thống kê đã ra đúng — đồng thời lấy được **thời gian 1 epoch thật** để ước lượng cả run:

```bash
ls results_smoke/
cat results_smoke/smoke_summary.txt
column -s, -t < results_smoke/smoke_metrics.csv
```

Nếu OK thì xoá `results_smoke/` và chạy thật.

---

## 4. Train chính

```bash
# ResTran + STN (cấu hình baseline, mục tiêu ~78,7%) — ghi log ra file
python -u train.py -n restran_stn --num-workers $(nproc) \
    2>&1 | tee results/restran_stn_train.log

# Ablation: tắt STN (baseline ~75,8%)
python -u train.py -n restran_nostn --no-stn --num-workers $(nproc) \
    2>&1 | tee results/restran_nostn_train.log
```

> `-u` bắt buộc khi ghi log: nếu không, stdout bị buffer và `tail -f` sẽ không thấy gì cho tới lúc buffer đầy.

Chạy nền để không mất khi rớt SSH:

```bash
mkdir -p results
nohup python -u train.py -n restran_stn --num-workers $(nproc) \
    > results/restran_stn_train.log 2>&1 &
echo $!                                      # ghi lại PID
tail -f results/restran_stn_train.log        # Ctrl+C để thoát mà không dừng train
```

Hoặc dùng `tmux` (khuyến nghị hơn):

```bash
tmux new -s train
python -u train.py -n restran_stn --num-workers $(nproc) 2>&1 | tee results/restran_stn_train.log
# Ctrl+B rồi D để detach; quay lại: tmux attach -t train
```

Theo dõi GPU:
```bash
watch -n 2 nvidia-smi
```

### 4.1. Cấu hình chống loss nổ (khuyến nghị)

Cấu hình baseline gốc (`--grad-clip 5.0`, fp16, `lr 5e-4`) đã được quan sát thấy **sập giữa chừng**: train ổn tới epoch 11 (val acc 61,76%), rồi grad norm nhảy từ 3,5 lên 899 → 1325, loss vọt lên 3,39 và acc về 0 vĩnh viễn — CTC rơi vào nghiệm suy biến "chỉ xuất blank". Ba cờ dưới đây phòng đúng tình huống đó:

| Cờ | Khuyến nghị | Tác dụng |
|---|---|---|
| `--grad-clip` | **1.0** (mặc định 5.0) | Clip giữ nguyên *hướng* gradient, chỉ co độ lớn — nên ngưỡng lỏng vẫn cho phép một batch bệnh lý kéo model đi sai. Giai đoạn khoẻ grad norm chỉ 3,5–14 nên 5.0 gần như không ràng buộc gì |
| `--amp-dtype` | **fp16** — chỉ đổi sang bf16 sau khi đo (xem cảnh báo dưới) | bf16 có dải mũ như fp32 nên **không tràn số**; fp16 tràn trong transformer là nguồn gradient khổng lồ phổ biến. Khi bật bf16, `GradScaler` tự tắt |
| `--skip-grad-norm` | **100** (mặc định tắt) | Bỏ hẳn batch có grad norm trước clip vượt ngưỡng, thay vì bước theo nó. Cũng bỏ luôn batch có grad norm `inf`/`nan` |
| `--skip-abort-ratio` | 0.2 (mặc định) | Dừng train khi >20% batch của một epoch bị bỏ — lúc đó model đã hỏng chứ không phải một batch xấu, chạy tiếp chỉ tốn GPU |

Khi guard kích hoạt lần đầu trong mỗi epoch, log in ra chẩn đoán để tìm **nguyên nhân thật**:

```
🔍 Spike epoch 25, batch 118: grad_norm=1399.1 (huu han), loss=0.1832
   grad theo khoi (truoc clip): stn=26.3 backbone=45.3 fusion=0.7 transformer=13.0 head=10.0
   track dau batch: ['track_12625', 'track_12604', ...]
```

- `inf/nan` → tràn số fp16, khác hẳn `huu han` → bất ổn định tối ưu (thường là LR quá cao quá lâu).
- Khối nào có norm lớn nhất là khối gây nổ.
- `loss` vẫn nhỏ mà grad lớn → không phải dữ liệu bẩn; loss vọt lên hàng đơn vị → nghi mẫu hỏng, tra `track dau batch`.

```bash
python -u train.py -n restran_stn_v2 --num-workers $(nproc) \
    --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    2>&1 | tee results/restran_stn_v2_train.log
```

> ⚠️ **Đo tốc độ trước khi dùng `--amp-dtype bf16`.** Đã quan sát thấy bf16 chậm hơn fp16 tới **~38×** trên cấu hình này (3,19 s/it so với 0,084 s/it): `seed_everything` bật `cudnn.deterministic = True`, và cuDNN có rất ít thuật toán conv tất định cho bf16 nên rơi xuống đường chạy chậm — chưa kể GPU trước Ampere không có bf16 gốc. Chỉ bật bf16 nếu benchmark dưới đây cho thấy nó không chậm hơn fp16:
>
> ```bash
> python - <<'PY'
> import time, sys; sys.path.insert(0, '.')
> import torch
> from torch.amp import autocast
> from configs.config import Config
> from src.models.restran import ResTranOCR
> c = Config(); m = ResTranOCR(num_classes=c.NUM_CLASSES, use_stn=c.USE_STN).cuda()
> x = torch.randn(64, 5, 3, 32, 128, device='cuda')
> print('GPU:', torch.cuda.get_device_name(0), '| sm_%d%d' % torch.cuda.get_device_capability(0))
> print('bf16 support:', torch.cuda.is_bf16_supported())
> for name, dt in [('fp16', torch.float16), ('bf16', torch.bfloat16)]:
>     for _ in range(3):
>         with autocast('cuda', dtype=dt): m(x)
>     torch.cuda.synchronize(); t = time.perf_counter()
>     for _ in range(10):
>         with autocast('cuda', dtype=dt): m(x)
>     torch.cuda.synchronize()
>     print(f'{name}: {(time.perf_counter()-t)/10*1000:7.1f} ms/batch')
> PY
> ```
>
> Nếu buộc phải dùng bf16 mà nó chậm, đổi `USE_CUDNN_BENCHMARK: bool = True` trong [config.py](configs/config.py) để bỏ chế độ tất định — đánh đổi khả năng tái lập lấy tốc độ.

Hai cột mới trong `_metrics.csv` để theo dõi: **`grad_norm_max`** (đỉnh trong epoch, thứ mà cột trung bình che mất) và **`skipped_batches`**. Vài batch bị bỏ mỗi epoch là bình thường; **hàng trăm batch bị bỏ nghĩa là ngưỡng quá thấp** — xem `grad_norm_max` của epoch 1–2 rồi đặt ngưỡng khoảng 5–10× giá trị đó.

Train tiếp từ một checkpoint đã có thay vì chạy lại từ đầu:

```bash
python -u train.py -n restran_stn_ft --resume results/restran_stn_best.pth \
    --epochs 10 --lr 1e-4 --grad-clip 1.0 --skip-grad-norm 100 \
    --num-workers $(nproc) 2>&1 | tee results/restran_stn_ft_train.log
```

> `--resume` chỉ nạp **trọng số**, không nạp trạng thái optimizer/scheduler — `OneCycleLR` sẽ chạy lại một chu kỳ mới trên số epoch bạn đưa vào. Vì vậy hãy đặt `--lr` thấp hơn hẳn lần chạy đầu.

---

## 5. Nhật ký, checkpoint và thống kê

### 5.1. Các file được xuất ra

Mỗi lần train ghi ra `results/` (hoặc thư mục `--output-dir`) theo tên `{experiment_name}`:

| File | Nội dung | Ghi khi nào |
|---|---|---|
| `{exp}_metrics.csv` | **Thống kê từng epoch**: loss (mean/min/max/std/median), grad norm, val loss, val acc, LR, thời gian train/val/epoch, throughput, VRAM đỉnh, timestamp | tự động, flush **sau mỗi epoch** |
| `{exp}_batches.csv` | Loss / LR / grad norm của **từng batch** | chỉ khi bật `--log-batch-loss` |
| `{exp}_summary.txt` | Tóm tắt cuối run: tổng thời gian, thời gian epoch TB/min/max, loss đầu→cuối, best acc và epoch đạt được | tự động, cuối run |
| `{exp}_best.pth` | **Checkpoint tốt nhất** (state_dict) theo val accuracy | mỗi lần val acc lập đỉnh mới |
| `submission_{exp}.txt` | Dự đoán trên tập val tại epoch tốt nhất, đúng định dạng nộp | cùng lúc với `_best.pth` |
| `submission_{exp}_final.txt` | Dự đoán trên test set | chỉ ở `--submission-mode` |
| `{exp}_train.log` | Toàn bộ stdout/stderr | do `tee`/`nohup` ở §4 tạo ra |

Vì `_metrics.csv` được flush mỗi epoch nên **train có sập giữa chừng vẫn còn đủ số liệu** của các epoch đã chạy.

Bật thêm log mức batch (file ~1–2 MB cho 30 epoch, dùng khi cần soi loss spike):

```bash
python -u train.py -n restran_stn --num-workers $(nproc) --log-batch-loss \
    2>&1 | tee results/restran_stn_train.log
```

### 5.2. Đọc thống kê

```bash
# Xem bảng gọn: epoch, loss, val loss, val acc, thời gian
cut -d, -f1,2,8,9,14 results/restran_stn_metrics.csv | column -s, -t

# Xem toàn bộ cột, cuộn ngang bằng mũi tên
column -s, -t < results/restran_stn_metrics.csv | less -S
```

Thống kê thời gian chạy mỗi epoch:

```bash
awk -F, 'NR>1{n++; t+=$14; if($14>mx)mx=$14; if(mn==""||$14<mn)mn=$14}
  END{printf "%d epoch | TB %.1fs | min %.1fs | max %.1fs | tong %.2fh\n",
      n, t/n, mn, mx, t/3600}' results/restran_stn_metrics.csv
```

Epoch tốt nhất và ETA khi đang chạy dở:

```bash
# Best epoch
awk -F, 'NR==2{b=$9+0;e=$1} NR>2 && $9+0>b {b=$9+0;e=$1}
  END{printf "best val_acc %.2f%% tai epoch %s\n", b, e}' results/restran_stn_metrics.csv

# Còn bao lâu (sửa total=30 cho khớp --epochs)
awk -F, -v total=30 'NR>1{n++; t+=$14}
  END{printf "da xong %d/%d epoch | con lai ~%.1f phut\n",
      n, total, (total-n)*(t/n)/60}' results/restran_stn_metrics.csv
```

Toàn bộ thống kê loss (đầy đủ các cột) bằng pandas — chỉ cần nếu muốn bảng đẹp:

```bash
pip install pandas
python -c "
import pandas as pd
d = pd.read_csv('results/restran_stn_metrics.csv')
print(d[['epoch','train_loss','train_loss_min','train_loss_max','train_loss_std',
         'grad_norm','val_loss','val_acc','epoch_time_s']].to_string(index=False))
print()
print(d[['train_loss','val_loss','val_acc','epoch_time_s','samples_per_s']].describe())
"
```

### 5.3. Lọc log gọn

`tee` giữ cả thanh tiến trình `tqdm` nên file `.log` khá rác. Tách riêng dòng kết quả mỗi epoch:

```bash
tr '\r' '\n' < results/restran_stn_train.log | grep -a '^Epoch' \
    > results/restran_stn_epochs.txt
cat results/restran_stn_epochs.txt
```

Theo dõi trực tiếp lúc đang train, chỉ hiện dòng epoch:

```bash
tail -f results/restran_stn_train.log | tr '\r' '\n' | grep --line-buffered -a '^Epoch'
```

### 5.4. So sánh nhiều lần chạy

```bash
for f in results/*_metrics.csv; do
  awk -F, -v n="$(basename "$f" _metrics.csv)" '
    NR==2{b=$9+0;e=$1} NR>2 && $9+0>b {b=$9+0;e=$1} {if(NR>1){c++; t+=$14}}
    END{printf "%-18s best %.2f%% (ep %s) | %d epoch | %.1fs/epoch\n", n, b, e, c, t/c}' "$f"
done
```

### 5.5. Tải toàn bộ kết quả về máy

```bash
# Trên server: gói mọi thứ trừ checkpoint (nhẹ, vài trăm KB)
tar -czf results_logs.tar.gz results/*.csv results/*.txt results/*.log

# Trên máy local
scp -P <PORT> root@<SERVER_IP>:~/restran/results_logs.tar.gz .
scp -P <PORT> root@<SERVER_IP>:~/restran/results/restran_stn_best.pth .
```

---

## 6. Điều chỉnh theo phần cứng

| Tình huống | Xử lý |
|---|---|
| **`no kernel image is available for execution on the device`** | wheel torch không có kernel cho kiến trúc GPU — **không phải lỗi code**. Xem [§2.1](#21-lỗi-no-kernel-image-is-available-for-execution-on-the-device). |
| **OOM (CUDA out of memory)** | giảm batch: `--batch-size 32` hoặc `16`. Mỗi batch 64 thực chất là 64×5 = 320 ảnh qua ResNet34. |
| **GPU idle, CPU 100%** | nghẽn ở đọc ảnh (190k ảnh/epoch). Tăng `--num-workers` lên bằng `nproc`. |
| **Server ít RAM** | giảm `--num-workers` xuống 4 (mỗi worker giữ 1 bản dataset index). |
| **GPU rất mạnh, muốn nhanh hơn** | tăng `--batch-size 128 --lr 1e-3` (scale LR theo batch). |
| **Disk chậm** | dataset 240k file nhỏ — nếu server có SSD local, copy dataset vào đó thay vì network storage. |

---

## 7. Sinh file nộp

```bash
# Train toàn bộ 20.000 track (không val) rồi infer public test
python -u train.py -n restran_sub --submission-mode --num-workers $(nproc) \
    2>&1 | tee results/restran_sub_train.log
```

> ⚠️ `--submission-mode` **không có validation** → checkpoint "best" chính là checkpoint cuối cùng, và cột `val_loss` / `val_acc` trong `_metrics.csv` bằng 0 (các cột loss train và thời gian vẫn ghi đầy đủ). Hãy chốt số epoch bằng lần chạy có val ở bước 4 trước, rồi mới chạy bước này với `--epochs <số đã chốt>`.

Đóng gói nộp Codabench:

```bash
cp results/submission_restran_sub_final.txt predictions.txt
zip submission.zip predictions.txt     # zip trực tiếp FILE, không zip thư mục
```

Kiểm tra trước khi nộp:
```bash
wc -l predictions.txt        # phải đúng 1000 dòng (public test)
head -3 predictions.txt      # định dạng: track_10005,ABC1234;0.9876
```

Giới hạn: **5 lượt nộp/ngày, tối đa 25 lượt** toàn giải.

Tải kết quả về máy:
```bash
scp -P <PORT> root@<SERVER_IP>:~/restran/submission.zip .
scp -P <PORT> root@<SERVER_IP>:~/restran/results/restran_stn_best.pth .
```

---

## 8. Toàn bộ lệnh, dán một lần

```bash
# --- trên server ---
unzip restran_code.zip && cd restran
tar -xf ~/restran_dataset.tar

nvidia-smi --query-gpu=name,compute_cap --format=csv   # xem compute_cap rồi chọn wheel theo bảng §2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install albumentations opencv-python-headless numpy tqdm

# BẮT BUỘC: kiểm tra wheel có kernel cho GPU này (is_available() KHÔNG đủ)
python -c "
import torch; cap='sm_%d%d'%torch.cuda.get_device_capability(0)
assert cap in torch.cuda.get_arch_list(), f'{cap} khong co trong {torch.cuda.get_arch_list()} -> xem §2.1'
print('OK', cap, (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item())"
python train.py --epochs 1 --batch-size 32 -n smoke --output-dir results_smoke   # smoke test

mkdir -p results
tmux new -s train
python -u train.py -n restran_stn --num-workers $(nproc) 2>&1 | tee results/restran_stn_train.log
# Ctrl+B rồi D để detach

# --- xem thống kê bất cứ lúc nào ---
cut -d, -f1,2,8,9,14 results/restran_stn_metrics.csv | column -s, -t
cat results/restran_stn_summary.txt
```
