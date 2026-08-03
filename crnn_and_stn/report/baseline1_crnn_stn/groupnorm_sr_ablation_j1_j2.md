# GroupNorm + SR Per-Frame — Ablation J1 & J2 (Root Cause #1, #2, #3 — issue #9)

> ⚠️ **J1/J2 dưới đây là bản lịch sử** (`STN_POOL=(1,1)`, không domain-match/
> constrained-decode/EMA — kiến trúc cũ trước S1). Chúng **không so 1-biến được**
> với S1/S4 vì lệch nhau 5-6 tham số cùng lúc, xem ma trận đầy đủ ở
> [model_comparison_summary.md §4](model_comparison_summary.md#4-ma-trận-cấu-hình--cái-gì-bật-ở-model-nào).
> 🔴 **CHỈ J1 ĐƯỢC MULTI-SEED** (phạm vi cuối cùng: **3 model J1 + S1 + S4**),
> chạy **đúng bộ cờ J1 ở [mục 2](#2-cấu-hình)** + ép `--stn-pool 1,1` (mặc định nay
> là `(4,8)`). ✅ **J1 tái lập được ~76.88%** vì không dùng SR, mà mọi thay đổi code
> từ đó tới nay đều nằm ở nhánh SR.
>
> ⏭️ **J2 ra ngoài phạm vi** — con số **77.18% giữ nguyên là 1 seed**, dùng cho phụ
> lục + Limitations (ablation multi-frame vs single-frame không có error bar).
> Lý do và lệnh chạy:
> [../checklist_review.md](../checklist_review.md),
> [../training_runs/run_gpu.md](../training_runs/run_gpu.md).
>
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

> ⚠️ **Lệnh J2 ở trên CHƯA ĐẦY ĐỦ** (phát hiện 2026-08-03 khi đọc lại
> `log_j2.txt`): run thật còn bật **`--sr-edge-weight 0.5`** — banner in
> `SR: True (scale=2, lambda_sr=0.1, edge=0.5, ...)`. Mặc định code hiện tại là
> `0.0`, và S1/S4 đều chạy `edge=0.0`. Muốn tái lập đúng J2 phải thêm cờ đó;
> chi tiết ở [§3b](#3b--xác-minh-lại-từ-checkpoint--submission-2026-08-03).

## 3. Kết quả

| Cấu hình | Val Exact Match | vs ResBlock | vs baseline chuẩn (77.00%) |
|---|---:|---:|---:|
| **CRNN + STN (baseline chuẩn, report ICPR)** | **77.00%** | — | — |
| CRNN + STN (đo thực tế trên dataset project) | 75.78% | — | −1.22 |
| ResNet + Transformer + STN (report, tốt nhất trong report gốc)* | 78.70% | — | +1.70 |
| ResBlock backbone (PR #8, `norm=none`, **không phải baseline**) | 76.68% | — | −0.32 |
| J1 — ResBlock + GroupNorm | 76.88% | +0.20 | −0.12 |
| **J2 — + SR per-frame + giám sát** | **77.18%** | **+0.50** | **+0.18** |

\* Kiến trúc khác hẳn (ResNet+Transformer), không so trực tiếp được với nhánh CRNN đang làm.

J2 > J1 (+0.30) và > ResBlock (+0.50), nhưng so với **baseline chuẩn chỉ nhỉnh hơn +0.18** — và **J1 vẫn chưa vượt được baseline chuẩn**. So với mốc mạnh nhất trong report gốc (ResNet+Transformer+STN, 78.70%) J2 còn cách **1.52 điểm** — dù kiến trúc khác hẳn nên không so trực tiếp được, con số này cho thấy trần hiện tại của nhánh CRNN vẫn còn xa mốc cao nhất report từng đạt. Xem caveat ở mục 4 trước khi kết luận bất cứ điều gì từ các con số này.

## 3b. ✅ Xác minh lại từ checkpoint + submission (2026-08-03)

Cả 2 run vẫn còn đủ artefact trên đĩa, nên đã **chấm lại trực tiếp** thay vì tin số
cũ. Dữ liệu: `results/crnn_resblock_groupnorm_nosr_j1/`,
`results/crnn_resblock_sr_supervised_j2/`.

| Chỉ số | J1 | J2 | Ghi chú |
|---|---:|---:|---|
| Track đúng (chấm lại với `plate_text` gốc) | **768/999** | **771/999** | ✅ khớp chính xác 76.88% / 77.18% đã báo cáo |
| Val Acc | 76.88% | 77.18% | |
| **CER ↓** | **0.0608** | **0.0645** | ⚠️ **J2 KÉM hơn J1** dù exact match cao hơn |
| Confidence trung bình | 0.9811 | 0.9816 | ~bằng nhau |
| Track confidence < 0.55 | 0/999 | 0/999 | |
| **Track sai độ dài (≠7 ký tự)** | **0/999** | **18/999** | ⚠️ J2 tệ hơn hẳn — greedy decode, không có layout constraint |
| Params (từ `state_dict`) | 29,313,452 | 29,426,895 | khớp bảng compute cũ (STN pool `(1,1)`) |

### ⚠️ Phát hiện 1 — "J2 hơn J1" mong manh hơn cả những gì mục 4 đã cảnh báo

+3 track exact match đã nằm sâu trong biên nhiễu ±13. Nhưng nhìn thêm 2 chỉ số khác
thì **J2 còn kém J1**:

- **CER cao hơn** (0.0645 vs 0.0608) — khi J2 sai, nó sai **nhiều ký tự hơn**.
- **18/999 track sai độ dài** so với 0/999 của J1 — J2 phóng ảnh ×2 nên `T` tăng
  16 → 32, cho CTC nhiều chỗ chèn ký tự thừa hơn, trong khi lúc đó **chưa có
  constrained decode** để chặn.

Ở mức từng track: 224/999 (22.4%) khác dự đoán, **51 track J2 đúng/J1 sai** so với
**48 track ngược lại** — gần như triệt tiêu, đúng bản chất của chênh lệch +3.

→ Kết luận mục 7 (*"chưa chứng minh được SR có giá trị ở J2"*) **được củng cố thêm**,
không chỉ bằng lập luận biên nhiễu mà bằng 2 chỉ số độc lập cùng chỉ ngược chiều.

### ⚠️ Phát hiện 2 — J2 bật `edge=0.5`, KHÔNG có trong lệnh ghi ở mục 2

Banner `log_j2.txt` in ra:

```
SR : True (scale=2, lambda_sr=0.1, edge=0.5, perceptual=0.0)
```

Tức J2 chạy kèm **Sobel edge loss trọng số 0.5** (`L_SR = L1 + 0.5·L_edge`), trong
khi lệnh ghi ở [mục 2](#2-cấu-hình) **không có** `--sr-edge-weight`. Mặc định hiện
tại là `SR_EDGE_WEIGHT = 0.0` (`configs/config.py:90`), và S1/S4 đều chạy `edge=0.0`
— đã kiểm tra banner.

**Hệ quả**: J2 khác S1 ở **7 biến**, không phải 6 như
[s1_proposed_mf_sr_ocr.md §1](s1_proposed_mf_sr_ocr.md#1-cấu-hình-đã-chạy) liệt kê —
thiếu đúng dòng `edge loss`. Nghĩa là ladder J1→S1→S4 là **ablation tích luỹ**
(mỗi bậc đổi nhiều thứ), **không phải tách 1 biến** — phải ghi rõ cách đọc này trong
paper. (Chi tiết `edge=0.5` này chỉ còn quan trọng nếu sau này chạy lại J2.)

### So với multi-seed S1/S4 — khoảng cách nhất quán qua cả 3 seed

Net gain (số track cấu hình kia đúng mà J1/J2 sai, trừ đi chiều ngược lại):

| So với | seed 42 | seed 100 | seed 2026 | Trung bình |
|---|---:|---:|---:|---:|
| **S1 vs J1** | +29 | +32 | +31 | **+30.7 track** |
| **S1 vs J2** | +26 | +29 | +28 | **+27.7 track** |
| **S4 vs J1** | +21 | +29 | +28 | **+26.0 track** |
| **S4 vs J2** | +18 | +26 | +25 | **+23.0 track** |

Cả 4 khoảng cách đều **vượt xa biên nhiễu ±13** và **nhất quán ở cả 3 seed** (không
có seed nào làm đảo chiều). Đây là bằng chứng mạnh nhất hiện có cho claim *"nhóm S
vượt nhóm J"* — dù J1/J2 vẫn là 1 seed nên chưa phải kiểm định 2 phía đầy đủ.

## 3c. 🔴 Multi-seed J1 — KHUNG CHỜ KẾT QUẢ (chưa chạy)

> Đây là **run GPU duy nhất còn lại** của cả đợt revision (~10 h). Điền bảng dưới
> ngay sau khi chạy xong, rồi cập nhật
> [model_comparison_summary.md §1c](model_comparison_summary.md) và
> [multi_seed_results.md](multi_seed_results.md).

**Lệnh** (cờ lịch sử — giữ đúng kiến trúc J1 ở [mục 2](#2-cấu-hình)):

```bash
for SEED in 42 100 2026; do
  python train.py --preset stable --experiment-name j1_seed${SEED} --seed ${SEED} \
    --backbone-norm group --stn-pool 1,1 \
    --no-cudnn-benchmark --num-workers 8 --aug-level full \
    2>&1 | tee results/log_j1_seed${SEED}.txt
done
```

🚨 `--stn-pool 1,1` **phải có dấu phẩy** (`1 1` → parse thành `(11,)` → crash).

| Seed | Track đúng | Val Acc | Best epoch | Số epoch | Val Loss | CER ↓ | Conf. TB | conf<0.55 | sai độ dài |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | | | | | | | | | |
| 100 | | | | | | | | | |
| 2026 | | | | | | | | | |
| **Mean ± Std** | | | | | | | | | |

**✅ Điều kiện đạt**: Mean rơi quanh **76.88% ± 1.3**. J1 **không dùng SR**, mà mọi
thay đổi code từ 2026-07 tới nay đều nằm ở nhánh SR → phải tái lập được số cũ. Lệch
nhiều hơn ⇒ kiểm tra lại cờ (đặc biệt `--stn-pool`) trước khi dùng số.

**Mốc để so sau khi có kết quả** (số chính thức, 3 seed deterministic):

| Model | Mean ± Std | Chênh so với J1 (kỳ vọng) |
|---|---:|---|
| J1 (sẽ có) | *…* | — |
| **S1** (đề xuất) | **79.95% ± 0.15** | ~ +3.1 điểm → **claim headline** |
| **S4** (rẻ hơn 2.3×) | **79.48% ± 0.44** | ~ +2.6 điểm |

Sau khi có J1, chạy kiểm định cho claim headline:

```bash
python tools/aggregate_seeds.py \
  --acc 79.7798 80.0801 79.9800 --label "S1 (đề xuất)" \
  --baseline-acc <3 số J1> --baseline-label "J1 (không SR)"
```

⚠️ **Nhắc lại khi viết paper**: J1→S1 đổi **nhiều biến cùng lúc** (SR, DCN, MFSR,
STN pool, domain-match, decode, EMA) và `T` cũng nhảy 16→32. Đây là **ablation tích
luỹ**, không phải tách 1 biến — viết là *"gộp tất cả thay đổi được +X điểm"*, không
phải *"SR đóng góp +X điểm"*.

## 4. Caveat thống kê — quan trọng hơn mọi con số ở trên

Trước khi chạy được J2 hoàn chỉnh, cùng một cấu hình (cùng seed) đã chạy 2 lần — lệch nhau **6.5 điểm** ở epoch 3 (42.64% vs 49.15%), do `cudnn.benchmark=True` khiến thuật toán convolution không xác định. Cộng thêm validation chỉ 999 sample → CI 95% ≈ ±2.7 điểm.

**Kết luận: chênh lệch +0.18–0.50 nêu ở mục 3 nằm gọn trong biên độ nhiễu đã đo được bằng thực nghiệm — chưa đủ bằng chứng để khẳng định GroupNorm hay SR thực sự cải thiện accuracy.** Cần chạy **O1 (3 seed: 42/100/2026)** cho cả J1 và J2 trước khi kết luận chắc chắn.

## 5. Overfit — cùng pattern ở cả 2 run

| Run | Val loss chạm đáy | Best Val Acc | Train loss cuối |
|---|---|---:|---:|
| J1 | epoch 19 (0.2591) | 76.88% @ epoch 60 | 0.0074 |
| J2 | epoch 15 (0.2773) | 77.18% @ epoch 57 | 0.0748 |

Cả 2 run: val loss chạm đáy sớm rồi tăng dần trong khi train loss gần 0 (model thuộc lòng tập train), val acc vẫn nhích lên dù val loss tăng. **60-80 epoch là dư cho dataset ~19,000 track** — dư địa cải thiện nên nhắm vào chống overfit (augmentation mạnh hơn, dropout, weight decay) hơn là train lâu hơn hoặc thêm tham số/module mới.

## 6. SR loss của J2 — giảm thật, chỉ bị che bởi nhiễu batch-cuối

Trung bình `sr_loss` 10 epoch đầu: 0.804 → 10 epoch cuối: 0.674 (giảm ~16%). Module SR học đúng hướng tái tạo ảnh HR; xu hướng giảm chỉ bị che nếu nhìn giá trị batch-cuối từng epoch (dao động 0.61–0.90) thay vì trung bình cả epoch.

## 7. Kết luận & bước tiếp theo

- GroupNorm sửa đúng Root Cause #3 (hết NaN), gần như miễn phí compute (+15K params, FLOPs không đổi).
- SR per-frame + giám sát cho tín hiệu tích cực (+0.30 so với J1) nhưng tốn **3.66x FLOPs / latency** — xem bảng chi phí trong `run_gpu.md`.
- **Chưa cấu hình nào (J1 lẫn J2) vượt rõ ràng baseline chuẩn của tác giả (77.00%)** khi tính đến biên độ nhiễu — đây là điều quan trọng nhất cần nêu khi báo cáo, không nên nói "đã vượt baseline". Càng chưa gần mốc mạnh nhất report gốc (78.70%, ResNet+Transformer+STN — còn cách 1.52 điểm).
- Bước tiếp theo lúc đó: chạy **O1 (multi-seed)** trước khi đầu tư thêm vào J3 (+DCNv2) — J2 kỹ thuật đủ điều kiện `J2 ≥ J1` để chạy J3, nhưng xây tiếp trên một kết quả 1-seed chưa xác nhận là rủi ro không đáng.

**Cập nhật 2026-08-03 — chỉ J1 quay lại đợt multi-seed.**
Nhánh S1-S4 sau này đã trả lời câu hỏi "SR có giúp gì" (S1 vượt J2 tới 26 track,
ngoài biên nhiễu). Phạm vi multi-seed chốt cuối là **J1 + S1 + S4**:

- ✅ **J1** (T=16, không SR, chạy trên đúng nền S1/S4) — mốc nền để claim
  *"S1 vượt cấu hình không SR"* có error bar. J1 lịch sử ở trên **không dùng được**
  cho việc này vì chạy kiến trúc khác (`STN_POOL=(1,1)`, thiếu domain-match/
  constrained/EMA).
- ⏭️ **J2 ra ngoài phạm vi** (tiết kiệm ~27h GPU). Con số **77.18% giữ nguyên là
  1 seed**, dùng cho phụ lục + Limitations. Khi paper cần so multi-frame (S1) với
  single-frame (J2), nêu kèm nhãn *"1 seed; J2 khác S1 ở 7 tham số"* — chênh 26 track
  (gấp đôi biên nhiễu ±13) nên vẫn đáng nêu, chỉ là không có error bar.

⚠️ **Lưu ý cho paper**: J1→S1 **không** tách được `T` khỏi SR — bật
`--use-sr --sr-scale 2` tự động nâng `T` từ 16 lên 32
([`train.py:285`](../../train.py)), nên J1→S1 đổi **cả `T` lẫn SR cùng lúc**. Cấu
hình duy nhất tách được là `--width-downsample 4` **không** `--use-sr` — ngoài phạm
vi, ghi Limitations.
Xem [model_comparison_summary.md §1c](model_comparison_summary.md#1c-kết-quả-multi-seed--số-chính-thức-cho-paper)
cho ma trận đầy đủ và [s4_sr_scale1_mf_sr_ocr.md §3](s4_sr_scale1_mf_sr_ocr.md#3-trả-lời-câu-hỏi-t-confound--kết-quả-chính-của-s4)
cho câu hỏi gốc.
