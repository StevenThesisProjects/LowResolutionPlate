# Tuần 12 (21–27/12): Demo và đóng gói

Chưa đến lịch.

## Roadmap yêu cầu

1. Dry-run với thầy, sửa lỗi phát hiện khi dry-run.
2. README tái lập trên máy sạch.
3. Tag release `v1.0`.

Deliverable: demo live + video, repo tag `v1.0`, báo cáo kết quả PDF.

## Đã chuẩn bị

| Hạng mục | Trạng thái |
|---|---|
| README tái lập | `README.md` của thư mục này: cài đặt, chạy API/UI, chạy lại toàn bộ phép đo, Docker. **Chưa thử trên máy sạch** (tiêu chí MVP số 5: ≤ 30 phút) |
| Phiên bản | `mfplate.__version__ = "1.0.0"`; `model_version` gắn SHA-256 của checkpoint |
| Danh sách file + md5 | `MANIFEST.md` của thư mục này |
| Tag `v1.0` | ⚠️ Chưa có git. Từ 21/11 thư mục này nằm cùng cấp với `restran/`, nên có thể `git init` riêng rồi tag `v1.0`. App vẫn cần repo `restran/` (code model + checkpoint) đặt cạnh; `restran/` trên máy cũng **không phải git repo** (RUNS.md đã ghi từ 23/10), nên cần ghi rõ bản `restran` đi kèm |

## Việc còn lại (Phát)

1. Thử README trên một máy sạch hoặc container mới, ghi lại thời gian.
2. Dry-run theo `docs/demo_script.md` với thầy.
3. Đưa thư mục vào repo git, commit và tag `v1.0`.
4. Xuất báo cáo kết quả PDF (từ chương 4–5 của luận văn).
