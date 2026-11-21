# Mở đầu: roadmap 12 tuần và cách đối chiếu với code mới

Cập nhật ngày 21/11/2026, tuần 7 của roadmap (16–22/11).

**Vị trí thư mục.** Từ 21/11/2026, `masters_thesis_tienphat/` nằm **cùng cấp** với repo `restran/` (trước đó nằm bên trong `restran/`). Trong các báo cáo, đường dẫn bắt đầu bằng `src/`, `configs/`, `tools/`, `results_new/`, `documents/` tính từ `restran/`; các đường dẫn còn lại tính từ thư mục này. Checkpoint mà app phục vụ được chép vào `model/` của thư mục này (cùng SHA-256; xem `model/README.md`).

## 1. Vì sao không làm lại theo roadmap

Roadmap "MF-PlateReader — Roadmap 12 tuần" (`documents/Roadmap.md`, 05/10–27/12) được thầy xây dựng từ **Pull Request cũ**. Lúc đó code và kết quả còn ở giai đoạn trước:

| Roadmap giả định (PR cũ) | Thực tế hiện nay (code mới) |
|---|---|
| Mốc là **run #13** (`--consistent-aug --no-channel-shuffle`, 77,59% trên val 1.999) | Mốc là **baseline đã ổn định hoá** (LR 3e-4, clip 1.0, bỏ bước khi norm > 100). Cấu hình chốt là **công thức huấn luyện #50** ("combo3"), giữ nguyên theo ISSUE #3 trong review ngày 23/10 |
| Đóng góp dự kiến: **SC-CTC** (decode ràng buộc cú pháp) và **LGTF** (fusion có thêm độ nét Laplacian) | Cả hai đã đo và đã đóng. Decode có ràng buộc chỉ thêm +0,00 đến +0,05 điểm (RUNS.md, 01/10). LGTF tương ứng với cờ `--sharpness-prior` (run #33): kém mốc. Đóng góp chuyển sang **điều chuẩn fusion** (aux CTC từng frame, frame dropout, che từng frame), **0 tham số, 0 chi phí suy luận** |
| Đánh giá cuối trên **test 3.000 track có nhãn**, chạy 1 lần ở Tuần 6 | **Thầy quyết định không dùng tập test chính thức** (31/10). Thay bằng **lock-box 1.000 track** tách từ train, có protocol đăng ký trước, chấm đúng một lần ngày 25/10 |
| Mọi so sánh dùng 3 seed (42, 123, 999) | Lock-box dùng 4 seed (42, 43, 7, 2026) cho mỗi bên; E4 và N1 dùng 3 seed (42, 43, 7) |
| Code có `restran_lgtf.py`, `detect_plate_layout()` | Hai thứ này **không còn trong code mới** (đã grep `src/`, `configs/`, `tools/`, `tests/`, `train.py`). Code mới là repo `restran`: code gốc của BTC cộng các cờ mặc định tắt. Khi tắt hết cờ, code mới train giống từng bit code gốc (`tools/regression_check.py`) |

Vì vậy **chúng ta không train lại theo roadmap**. Mỗi hạng mục được xử lý theo một trong ba cách:

1. **Thay bằng phần đã làm** sau review của thầy: lock-box, E3, E4, N1, phân tích lỗi, bài báo ICCIES. Báo cáo từng tuần ghi rõ file bằng chứng.
2. **Làm mới trong thư mục này** bằng model đã đóng băng, không train thêm: so sánh decoder, audit cú pháp, hiệu chỉnh ngưỡng, parity, API, UI, Docker, test E2E, đo latency.
3. **Ghi rõ là chưa làm hoặc không còn cần**, kèm lý do.

## 2. Bảng đối chiếu tổng

| Hạng mục roadmap | Dùng gì thay thế | Bằng chứng |
|---|---|---|
| Mốc run #13 + SHA-256 (T1) | 8 run lock-box (4 baseline + 4 công thức #50), SHA-256 của 56 checkpoint trong manifest; model phục vụ chọn theo quy tắc cố định | `results_new/freeze_manifest.json`, `artifacts/serving_config.json` |
| Unit test decoder ≥ 10 case (T1) | 22 test decoder (tổng cộng 67 test tự động) | `tests/test_decoding.py` |
| % lỗi vi phạm cú pháp (T1) | Đo lại trên 4 model chốt, val 999 | `artifacts/error_syntax_audit.md` |
| 3 seed #13 + so 3 decoder (T2) | Seed của công thức chốt (giao thức công bố + lock-box); so 3 decoder trên 4 model chốt | RUNS.md #52–#70, `artifacts/results_decode.csv` |
| Issue #1 nhân đôi mẫu (T3) | Không phải lỗi: mỗi track = 1 mẫu LR thật + 1 mẫu synthetic, như code gốc. Thí nghiệm bỏ nhánh synthetic nằm trong vòng v2: code xong 31/10 nhưng **chưa chạy**; bài ICCIES nộp với kết quả cũ | `reports/tuan_03.md` |
| LGTF 3 seed + γ (T4) | Đóng (run #33). Thay bằng 3 thí nghiệm cơ chế E3/E4/N1 và phân tích theo độ nét × layout | REPORT_E4_N1.md, `artifacts/slice_sharpness.csv` |
| Gate + chốt cấu hình + ngưỡng (T5) | #50 đóng băng theo ISSUE #3; quy tắc (a)/(b) đăng ký trước; ngưỡng chốt trên val 999 | `documents/lockbox_protocol.md`, `artifacts/threshold_calibration.json` |
| Test 3.000 một lần (T6) | Lock-box 1.000 track, chấm một lần | `results_new/lockbox_eval.json`, `documents/REPORT_lockbox.md` |
| Backend v2, parity, OpenAPI (T7) | `mfplate/api.py`; parity 0/999 track lệch | `artifacts/parity_final.json`, `artifacts/openapi.json`, `docs/api.md` |
| UI v1 (T8) | Giao diện React do API phục vụ ở `/` (bản Gradio giữ làm dự phòng); 3 ảnh chụp màn hình | `webui/`, `artifacts/screenshots/` |
| App hoàn chỉnh + deploy (T9) | SQLite, feedback, xử lý lỗi, Docker, docker-compose có sẵn Cloudflare Tunnel | `mfplate/storage.py`, `docker/`, `artifacts/error_cases.md` |
| Testing (T10) | 30 case E2E, latency p50/p95, 5 request đồng thời; mẫu khảo sát SUS | `artifacts/test_report.md`, `artifacts/latency_summary.json`, `docs/user_test.md` |
| Thesis + demo (T11–T12) | Có kịch bản demo và track mẫu; chương 4–5 do Phát viết | `docs/demo_script.md`, `samples/` |

## 3. Trạng thái theo tuần

| Tuần | Mục tiêu roadmap | Trạng thái | Báo cáo |
|---|---|---|---|
| 1 | Khóa mốc & đo trần SC-CTC | Xong (thay thế) | [tuan_01.md](tuan_01.md) |
| 2 | Đánh giá SC-CTC + seed mốc + API khởi đầu | Xong (thay thế); bảng top-5 còn thiếu đội hạng 4 | [tuan_02.md](tuan_02.md) |
| 3 | Kiểm tra data pipeline (Issue #1) | Đã giải thích; chưa chạy thí nghiệm bỏ synthetic | [tuan_03.md](tuan_03.md) |
| 4 | Đánh giá LGTF | Đóng; thay bằng E3/E4/N1 | [tuan_04.md](tuan_04.md) |
| 5 | Chốt cấu hình + hiệu chỉnh ngưỡng | Xong | [tuan_05.md](tuan_05.md) |
| 6 | Freeze + chấm một lần | Xong (lock-box thay test) | [tuan_06.md](tuan_06.md) |
| 7 | Backend v2 | **Xong** (tuần hiện tại) | [tuan_07.md](tuan_07.md) |
| 8 | UI v1 | Xong (giao diện React, kèm ảnh chụp) | [tuan_08.md](tuan_08.md) |
| 9 | App hoàn chỉnh + deploy | Code xong; Phát cần build Docker, mở tunnel, thử trên iOS/Android | [tuan_09.md](tuan_09.md) |
| 10 | Testing | Phần tự động xong; Phát cần làm user test 5 người + SUS | [tuan_10.md](tuan_10.md) |
| 11 | Thesis + chuẩn bị demo | Có kịch bản demo; chương 4–5 và video do Phát làm | [tuan_11.md](tuan_11.md) |
| 12 | Demo & đóng gói | Chưa đến | [tuan_12.md](tuan_12.md) |

## 4. Tiêu chí MVP (mục 4 của roadmap)

| # | Tiêu chí | Ngưỡng | Kết quả | Bằng chứng |
|---|---|---|---|---|
| 1 | Parity app vs offline trên val | ≤ 2 track lệch | **0/999 track lệch** (cùng máy, CPU). CPU so với GPU: accuracy val 76,68% = 76,68%; lock-box 0/1.000 track khác đúng/sai | `artifacts/parity_final.json` |
| 2 | Test E2E | 30/30 không crash | Xem Tuần 10 | `artifacts/test_report.md` |
| 3 | Latency CPU, 1 track 5 frame | p95 ≤ 1 s | Xem Tuần 10 | `artifacts/latency_summary.json` |
| 4 | Mở được từ điện thoại qua URL | — | Chưa: cần Phát chạy tunnel và thử | `docker/docker-compose.yml` (profile `tunnel`) |
| 5 | Chạy lại từ README trên máy sạch ≤ 30 phút | — | Chưa thử trên máy sạch | `README.md` |

**Đọc tiếp:** mỗi file `tuan_XX.md` có bốn phần: roadmap yêu cầu gì, dùng gì thay thế, kết quả, và việc còn lại.
