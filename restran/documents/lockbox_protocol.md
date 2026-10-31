# Protocol đánh giá lock-box (đăng ký trước)

Viết ngày 23/10/2026, **trước khi chạy bất kỳ run lock-box nào**, theo yêu cầu ở mục 7.4 và ISSUE #1 của `documents/Teacher_Review.md`. Sau khi bắt đầu chạy, mọi thay đổi so với protocol này phải ghi vào mục 10, kèm lý do.

---

## 1. Mục đích

Mọi thành phần của cấu hình chốt đã được chọn trên tập val dev 1.999 track (chứa trọn tập val 999). Vì vậy các con số hiện có mang thiên lệch chọn lọc ở cấp track. Lock-box là một tập track **chưa từng được dùng cho bất kỳ quyết định nào**. Nó được chấm **đúng một lần**, sau khi mọi checkpoint đã được đóng băng, để:
1. Đo mức lợi của phương pháp so với baseline mà không có thiên lệch chọn lọc (RQ1).
2. Quyết định hướng viết bài (a) hay (b) theo quy tắc ở mục 7.

## 2. Cấu hình đóng băng

| | Cấu hình | Cờ |
|---|---|---|
| **Baseline** | ResTran + STN, đã ổn định hoá tối thiểu (như run #6) | `--epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100` |
| **Phương pháp** | Cấu hình #50 (combo3), giữ nguyên theo ISSUE #3 | như baseline, thêm `--consistent-aug --no-channel-shuffle --degrade-strength calibrated --aux-frame-ctc 0.2 --frame-mask 0.2 --frame-dropout 0.1 --grayscale` |

Hai bên chung kiến trúc (31,08M tham số), chung giao thức dữ liệu và chung các bước ổn định hoá.

**Kiểm tra code trước khi đóng băng** (ISSUE #9):
- `tests/test_eval_determinism.py`: 7/7 test đạt. Đánh giá cho kết quả xác định; frame dropout và frame mask không bao giờ chạy lúc validate; luma khớp với chuẩn hoá; thống kê tự viết khớp scipy.
- `tools/regression_check.py`: khi không bật cờ nào, code này train **giống từng bit** với code gốc (`duongtruongbinh/MultiFrame-LPR`, commit `3ca05c0`). Đã so trên 64 track, 2 epoch, CPU: train loss 4,1085 → 2,8737; val loss 3,3356 → 2,7804; 0/267 tensor trọng số khác nhau sau 1 và 2 batch.

## 3. Tập lock-box

- **Nguồn:** các track Scenario-B **không** nằm trong `splits/val_tracks_2000.json`. Tập này đã chứa trọn `splits/val_tracks.json`, nên nguồn gồm 8.001 track, trước nay chỉ dùng làm dữ liệu train.
- **Cách lấy:** `python tools/make_lockbox.py`. Lấy 1.000 track bằng `random.Random(2027).sample` trên danh sách ID đã sắp xếp, nên tất định và kiểm chứng lại được bằng `--verify`.
- **File:** `splits/lockbox_tracks.json`, SHA-256 `09f90d2beaeea847d76f152315bc2a4db0e80885e0060e4400f0c30fd332e63f`.
- **Thành phần:** 192 Brazilian, 808 Mercosur. Không trùng track nào với hai tập val.
- **Không được dùng** để train, validate, early stopping, chọn checkpoint, hay bất kỳ quyết định nào trước bước 6.

## 4. Huấn luyện: 8 run

- **Train:** 20.000 track − 999 (val) − 1.000 (lock-box) = **18.001 track**, mỗi track kèm một mẫu synthetic như code gốc.
- **Val 999** (`splits/val_tracks.json`, SHA-256 `8a560294…`): chỉ dùng để theo dõi và chọn checkpoint "best".
- **Seed:** 42, 43, 7, 2026 cho mỗi bên. 30 epoch. `--save-last-k 6` lưu trọng số epoch 25–30.
- **Tên run:** `restran_lockbox_base_s{seed}` và `restran_lockbox_combo3_s{seed}`.
- Lệnh đầy đủ nằm trong `documents/RUNS.md`, mục "Lock-box: lệnh chạy".
- Nếu một run bị dừng bởi cơ chế ngắt an toàn hoặc bị lỗi: chạy lại **một lần**, cùng seed và cùng lệnh, rồi báo cáo cả hai. Không có lý do nào khác để chạy lại.

## 5. Đóng băng trước khi chấm

Chỉ làm sau khi **cả 8 run** đã xong:
1. `python tools/make_lockbox.py --verify` (lock-box không bị đổi).
2. `python tools/freeze_manifest.py --out results/freeze_manifest.json --checkpoints "results/restran_lockbox_*_s*_ep*.pth" "results/restran_lockbox_*_best.pth"`. Lệnh này ghi SHA-256 của code, các file split, 56 checkpoint và thông tin môi trường (phiên bản PyTorch, CUDA, cuDNN, GPU).
3. Ghi digest của manifest vào mục 9.

## 6. Chấm: một lần duy nhất

```bash
python tools/eval_lockbox.py --frames-analysis --out results/lockbox_eval.json
```

**Thước đo:**
- **Thước đo chính** (MINOR 17): accuracy ở **epoch cuối**, không có bước chọn epoch nào.
- **Thước đo dùng cho quyết định:** **Last-6**, tức accuracy trung bình của 6 checkpoint epoch 25–30, như quy tắc ở mục 7.4 của review.
- **Phụ:** checkpoint "best" chọn trên val 999, so được với cách tính số công bố 78,70%.

**Thống kê hai tầng** (ISSUE #2):
- *Giữa các seed:* mean ± std, Welch t-test, Cohen's d. Tầng này đo dao động do huấn luyện.
- *Giữa các track:* paired bootstrap 10.000 lần, lấy mẫu lại track. Đại lượng được lấy mẫu là chênh lệch accuracy theo từng track (đã lấy trung bình 4 seed, và trung bình 6 epoch với Last-6); in ra CI 95%. Thêm McNemar chính xác cho từng cặp seed ở epoch cuối.

**Phân tích phụ, đăng ký trước, chỉ để mô tả:**
- Theo loại biển. Chỉ có 192 track Brazilian nên ghi rõ là chưa đủ lực thống kê.
- **E3:** accuracy theo số frame N = 1..5, lấy trung bình trên mọi tập con N frame, ở checkpoint epoch cuối, kèm CI bootstrap cho từng N.

## 7. Quy tắc quyết định

> Nếu chênh lệch **Last-6** (phương pháp − baseline) **≥ +1,0 điểm** và **cận dưới CI 95% bootstrap > 0** → hướng **(b)**: điều chuẩn fusion là trọng tâm, SR là mục phân tích. Ngược lại → hướng **(a)**: "SR có giúp LRLPR đa khung hình?".

Kết quả được báo cáo đầy đủ dù rơi vào trường hợp nào. Script in ra quyết định tự động theo đúng quy tắc này.

## 8. Diễn giải E3: đăng ký trước để tránh kể chuyện sau khi đã thấy số liệu

| Dạng kết quả | Diễn giải |
|---|---|
| A. Mức lợi gần 0 ở N = 1 và tăng dần đến N = 5 | Phương pháp giúp khai thác thông tin bổ sung **giữa các frame**. Bằng chứng mạnh nhất cho (b) |
| B. Mức lợi lớn nhất ở N nhỏ và giảm dần khi N tăng | Phương pháp làm model **bền hơn khi thiếu frame hoặc frame xấu**. Điều này khớp với thiết kế: frame dropout và che frame đều buộc model đọc được với ít frame. Vẫn ủng hộ (b), nhưng theo hướng "chịu được mất frame", và liên quan trực tiếp tính năng nhận ít hơn 5 ảnh của app |
| C. Mức lợi như nhau ở mọi N | Chỉ là điều chuẩn chung; tính mới của (b) yếu đi |
| CI của các N chồng lấn nhiều | "Không có xu hướng rõ" |

## 9. Ghi nhận khi chạy

| Mục | Giá trị |
|---|---|
| Server, GPU, PyTorch / CUDA / cuDNN | `fierce-ibex`, Tesla V100-SXM2-32GB, Python 3.10.13, PyTorch 2.14.1+cu126, CUDA 12.6, cuDNN 9.10.2 (91002), AMP fp16 |
| Ngày chạy 8 run | 2 đợt (seed 42, 43; rồi seed 7, 2026), xong trước ngày 25/10/2026 |
| Run phải chạy lại | Không có |
| Digest của `results/freeze_manifest.json` | `39c39dfceb86a2d5b9d338ade096c5a00a85bebe88ad0127e049d318b7117824` (24 file code, 3 split, 56 checkpoint) |
| Ngày chấm lock-box | 25/10/2026, chấm một lần |
| Kết quả quy tắc mục 7 | **(b)**: chênh Last-6 +1,64, CI 95% [+0,78; +2,48] |
| Kiểm chứng sau khi chấm | 56/56 checkpoint, 24/24 file code, 3/3 split khớp SHA-256 với manifest. Code trên server trùng code đã test trên máy local. Tính lại thống kê từ kết quả theo từng track với seed bootstrap khác: CI Last-6 [+0,80; +2,52], quyết định không đổi. Chấm lại 2 checkpoint trên máy khác (Apple GPU): 77,40% và 80,30%, trùng với server. Chấm lại rộng hơn (25/10, 14 checkpoint của seed 42 và 43): 3/28.000 dự đoán khác server, đều là trường hợp hai ký tự gần hoà (chênh log-prob giữa lớp nhất và nhì ≤ 0,008); CPU và Apple GPU cho cùng kết quả. Lệch lớn nhất 0,1 điểm trên một checkpoint (`base_s43_ep30`), không đổi kết luận. Các phân tích sau dùng chính server này để chấm |
| Lưu trữ | Server trả ngày 25/10/2026. Mọi kết quả (56 checkpoint, log, CSV, `lockbox_eval.*`, manifest) đã tải về `results_new/`. Code đóng băng dựng lại từ bản sao lưu tự động của Claude Code, lưu ở `results_new/code_lockbox_frozen.tgz`: 27/27 file (24 code + 3 split) khớp SHA-256 trong manifest |
| Dạng E3 (mục 8) | **B**: +5,00 (N = 1) → +1,52 (N = 5), giảm dần, mọi CI không chứa 0 |

## 10. Thay đổi so với protocol

- Log huấn luyện của `restran_lockbox_base_s42` không có trên server, các run khác đều có. **Đã xác nhận** run này loại lock-box qua `results/restran_lockbox_base_s42_summary.txt`: `Train samples : 36002` (18.001 track × 2), giống 7 run còn lại. Bốn run phương pháp có cấu hình giống hệt nhau trong log (AMP fp16, consistent-aug, không ChannelShuffle, frame dropout 0,1, frame mask 0,2, grayscale, aux CTC 0,2, loại lock-box với đúng SHA-256), nên chênh lệch giữa seed 42/43 và seed 7/2026 là dao động huấn luyện, không phải khác cấu hình.
- Phân tích bổ sung **không** đăng ký trước (ghi rõ khi báo cáo): kiểm định hoán vị chính xác giữa các seed, p hai phía = 0,029 (4/4 seed phương pháp cao hơn cả 4 seed baseline ở mọi thước đo).

---

**Ghi chú về khả năng tái lập:**
- Huấn luyện không tái lập được từng bit trên GPU. Ngoài việc GPU không tất định, albumentations 2.x lấy số ngẫu nhiên từ một bộ sinh riêng mà `--seed` không điều khiển được (đã kiểm chứng). Đây là lý do cùng seed 42 mà run #3 và #6 lệch nhau 0,5 điểm.
- Vì vậy bảng chính dùng **4 seed mỗi bên**, và kiểm định tính cả dao động giữa các seed lẫn dao động giữa các track.
- **Nếu nhãn tập test chính thức về:** chấm cùng 8 bộ checkpoint trên tập test, với cùng thước đo và cùng thống kê, như một phân tích xác nhận thêm. Quyết định ở mục 7 vẫn dựa trên lock-box như đã đăng ký.
