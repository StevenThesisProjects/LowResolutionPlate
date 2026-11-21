# Tuần 9 (30/11–06/12): App hoàn chỉnh và deploy

Code làm trước lịch. Các bước cần thiết bị thật do Phát làm.

## Roadmap yêu cầu

1. SQLite log, nút feedback.
2. Xử lý lỗi (file hỏng, không phải ảnh, > 5 ảnh, ảnh quá lớn) → `error_cases.md` (≥ 8 case lỗi xử lý đúng).
3. Dockerfile CPU, Cloudflare Tunnel → Docker image, URL demo riêng tư.
4. Test trên iOS Safari và Android Chrome.

## Dùng gì thay thế

Không có gì phải thay; làm theo roadmap.

## Kết quả

**SQLite** (`mfplate/storage.py`):
- Bảng `predictions`: request_id, thời điểm, session, SHA-256 của ảnh đầu vào, số frame, biển số, confidence, điểm triage, cờ, decoder, latency, model_version, JSON kết quả.
- Bảng `feedback`: đúng/sai, biển số sửa.
- **Không lưu ảnh.**
- `GET /stats` tính tỉ lệ cần kiểm tra và accuracy theo phản hồi, tách nhóm tự động và nhóm cần kiểm tra. Đây chính là số liệu để đánh giá triage khi có người dùng thật (Chương 5).

**Xử lý lỗi** (`scripts/run_e2e.py` gửi tới server uvicorn thật → `artifacts/error_cases.md`): **14/14 trường hợp trả đúng mã HTTP và mã lỗi**, server không crash:

| # | Trường hợp | HTTP | Mã lỗi |
|---|---|---|---|
| 1 | Không gửi file | 400 | `no_files` |
| 2 | 6 ảnh | 400 | `too_many_files` |
| 3 | File .txt | 415 | `unsupported_type` |
| 4 | GIF đổi đuôi .jpg | 415 | `unsupported_type` |
| 5 | JPEG bị cắt cụt | 422 | `corrupt_image` |
| 6 | File rỗng | 422 | `empty_file` |
| 7 | Ảnh > 2 MB | 413 | `file_too_large` |
| 8 | Ảnh rộng 4000 px | 413 | `image_too_large` |
| 9 | Ảnh 2×3 px | 422 | `image_too_small` |
| 10 | Zip hỏng | 400 | `zip_invalid` |
| 11 | Zip không có ảnh | 400 | `zip_no_images` |
| 12 | Gửi lẫn zip và ảnh | 400 | `mixed_upload` |
| 13 | Decoder không tồn tại | 422 | `invalid_parameter` |
| 14 | Feedback cho request_id không tồn tại | 404 | `not_found` |

Ngoài ra, `tests/test_api.py` có 17 test lỗi chạy trong mỗi lần `pytest`.

**Docker** (`docker/`):
- `Dockerfile`: python:3.11-slim, PyTorch 2.13.0 CPU, chạy bằng user không phải root.
- Build context là thư mục cha chứa cả `restran/` và `masters_thesis_tienphat/` (hai thư mục cùng cấp từ 21/11). Image chỉ chép `src/`, `configs/` (từ `restran/`), checkpoint đã đóng băng (từ `model/` của thư mục này) và 3 file cấu hình phục vụ. `Dockerfile.dockerignore` loại khoảng 20 GB dữ liệu và checkpoint khác khỏi build context.
- `docker-compose.yml`:
  - service `api` (cổng 8000, healthcheck `/health`, volume giữ SQLite);
  - giao diện web React được build ngay trong Dockerfile (stage Node) và do chính service `api` phục vụ ở cổng 8000; giao diện Gradio cũ là service tuỳ chọn `gradio` (profile `gradio`, cổng 7860);
  - service `tunnel` (Cloudflare quick tunnel tới `api:8000`, bật bằng `--profile tunnel`);
  - thư mục `samples/` được mount chỉ-đọc từ máy chủ, không đóng vào image (dữ liệu có giấy phép).
- Biến `MFPLATE_API_KEY` bật khoá API khi mở ra ngoài.

**⚠️ Chưa build thử:** máy này có Docker 28.4 nhưng daemon (Docker Desktop) không chạy lúc làm. Dockerfile chưa được kiểm bằng một lần build thật.

## Việc Phát cần làm

```bash
# 1. Build và chạy (từ thư mục masters_thesis_tienphat/; repo restran/ nằm cùng cấp)
docker compose -f docker/docker-compose.yml up --build
curl http://localhost:8000/health          # phải ra "status": "ok"
# 2. URL riêng tư cho điện thoại
MFPLATE_API_KEY=<chuỗi bí mật> docker compose -f docker/docker-compose.yml --profile tunnel up
docker compose -f docker/docker-compose.yml logs tunnel | grep trycloudflare
```

- Quick tunnel cho một URL ngẫu nhiên, khó đoán nhưng ai có link cũng mở được. Muốn chỉ người được mời mới vào thì cần cấu hình Cloudflare Access (tài khoản Cloudflare).
- Mở URL trên iOS Safari và Android Chrome, chạy 3 track mẫu, chụp màn hình.
- Track mẫu là dữ liệu LRLPR-26: chỉ dùng cho demo riêng tư.

## Deliverable

| Roadmap | File | Trạng thái |
|---|---|---|
| SQLite log + feedback | `mfplate/storage.py`, `POST /feedback`, `GET /stats` | ✅ |
| `error_cases.md` (≥ 8 case) | `artifacts/error_cases.md` (14 case) | ✅ |
| Dockerfile CPU | `docker/Dockerfile`, `docker/docker-compose.yml` | ⚠️ viết xong, chưa build thử |
| URL demo riêng tư | Profile `tunnel` | ❌ Phát chạy |
| Test iOS + Android | — | ❌ Phát làm |
