# Hướng dẫn chạy backend và frontend

Ứng dụng có hai phần:

| Phần | Thư mục | Công nghệ | Vai trò |
|---|---|---|---|
| Backend | `mfplate/` | Python, FastAPI, PyTorch (CPU) | Nạp model đã đóng băng, nhận ảnh, trả biển số qua REST API; phục vụ luôn giao diện web đã build |
| Frontend | `webui/` | React + Vite | Giao diện web; build ra `webui/dist/` |

Có hai cách chạy:
- **Cách A (dùng khi demo):** build frontend một lần, chạy một server ở cổng 8000. Giao diện và API cùng một địa chỉ.
- **Cách B (dùng khi sửa giao diện):** chạy backend ở cổng 8000 và Vite dev server ở cổng 5173. Sửa code React là trang tự tải lại.

Mọi lệnh dưới đây đã chạy thử lại ngày 21/11/2026, sau khi tách thư mục ra cùng cấp với `restran/`, trên macOS (Apple M4), Python 3.12.6, Node 22.21.0.

Bố cục thư mục:

```
NewClaude/
├── restran/                   repo train: src/, configs/ (code mạng), results_new/, dataset/
└── masters_thesis_tienphat/   app: backend mfplate/, frontend webui/, model/ (checkpoint), báo cáo
```

## 1. Yêu cầu

- **Python 3.11 hoặc 3.12.** Trên máy này lệnh là `python3`; không có lệnh `python`.
- **Node.js 18 trở lên** và npm, chỉ cần để build frontend.
- **Repo `restran` đặt cùng cấp** với `masters_thesis_tienphat/` (tức `../restran`), có `src/` và `configs/` (code mạng).
- **Checkpoint `model/restran_lockbox_combo3_s42_ep30.pth`** trong thư mục này. Khi khởi động, backend kiểm SHA-256 của file với giá trị đã đóng băng (`9754ebb1…7bba`). File không có trên git; nếu thiếu, xem [`model/README.md`](../model/README.md) (chép từ `../restran/results_new/` hoặc chạy `python3 scripts/freeze_serving_model.py`).
- `artifacts/serving_config.json`, `decoder_decision.json`, `threshold_calibration.json` (có sẵn).
- **Track mẫu** (`masters_thesis_tienphat/samples/*.zip`): không bắt buộc. Thiếu thì giao diện chỉ không có nút track mẫu.

Backend tự tìm repo theo thứ tự: biến `MFPLATE_REPO_ROOT`, thư mục `../restran`, rồi thư mục cha. Nếu repo nằm ở chỗ khác, thêm `MFPLATE_REPO_ROOT=<đường dẫn tới restran>` trước mọi lệnh `python3`.

Mọi đường dẫn bên dưới tính từ thư mục `masters_thesis_tienphat/`:

```bash
cd /Users/nguyentienphat/Learning/Paper/NewClaude/masters_thesis_tienphat
```

## 2. Cài đặt (làm một lần)

```bash
# Thư viện Python cho backend
python3 -m pip install --user -r requirements.txt

# Thư viện cho frontend, rồi build ra webui/dist/
cd webui
npm install
npm run build
cd ..
```

Chỉ cần build lại frontend (`npm run build`) khi sửa code trong `webui/src/`.

## 3. Cách A: một server cho cả backend và frontend

```bash
python3 -m uvicorn mfplate.api:app --port 8000
```

Dùng `python3 -m uvicorn`, không gõ `uvicorn` trực tiếp. Trên máy này lệnh `uvicorn` thuộc Python 3.13, là bản không cài torch.

Khi terminal hiện `Application startup complete`, mở:

| Địa chỉ | Nội dung |
|---|---|
| http://127.0.0.1:8000 | Giao diện web |
| http://127.0.0.1:8000/?sample=review_track_22259.zip | Mở trang và đọc luôn một track mẫu (tên file trong `samples/`) |
| http://127.0.0.1:8000/docs | Tài liệu API, thử từng endpoint trực tiếp |

Kiểm tra backend bằng dòng lệnh:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"model_version":"restran-recipe-s42-ep30-9754ebb1915e",...}

curl -F files=@samples/easy_track_14372.zip http://127.0.0.1:8000/predict
```

Dừng server: bấm `Ctrl+C` trong terminal. Nếu server chạy nền:

```bash
lsof -ti:8000 | xargs kill
```

## 4. Cách B: chạy riêng frontend để sửa giao diện

Mở hai terminal.

```bash
# Terminal 1: backend
python3 -m uvicorn mfplate.api:app --port 8000
```

```bash
# Terminal 2: frontend (Vite dev server)
cd webui
npm run dev
```

- Mở **http://localhost:5173**. Dùng đúng chữ `localhost`, không dùng `127.0.0.1`: Vite chỉ lắng nghe ở `localhost`.
- Mọi lời gọi `/predict`, `/health`, `/samples`, … được Vite chuyển sang backend ở cổng 8000. Muốn trỏ tới backend khác: `MFPLATE_API_URL=http://<địa chỉ>:<cổng> npm run dev`.
- Sửa file trong `webui/src/` là trang tự tải lại.
- Sửa xong, chạy `npm run build` để cổng 8000 (Cách A) dùng bản mới.

## 5. Thử trên điện thoại cùng mạng Wi-Fi

```bash
python3 -m uvicorn mfplate.api:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0          # in ra IP của máy, ví dụ 192.168.1.23
```

- Trên điện thoại, mở `http://192.168.1.23:8000` (thay bằng IP vừa in).
- Nếu macOS hỏi có cho Python nhận kết nối không, chọn **Allow**.
- Muốn có URL dùng được từ ngoài mạng nhà: dùng Docker kèm Cloudflare Tunnel (mục 7).

## 6. Giao diện Gradio cũ (dự phòng)

Cần backend đang chạy ở cổng 8000.

```bash
python3 ui/app.py               # http://127.0.0.1:7860
```

## 7. Chạy bằng Docker

Chạy từ thư mục `masters_thesis_tienphat/`. Repo `restran/` phải nằm cùng cấp, vì build context là thư mục cha chứa cả hai (`docker/Dockerfile.dockerignore` chỉ gửi đúng các file image cần):

```bash
docker compose -f docker/docker-compose.yml up --build
# Giao diện + API: http://localhost:8000

# Thêm URL HTTPS riêng tư cho điện thoại (nên bật khoá API):
MFPLATE_API_KEY=<chuỗi bí mật> docker compose -f docker/docker-compose.yml --profile tunnel up
docker compose -f docker/docker-compose.yml logs tunnel | grep trycloudflare
```

- Image tự build frontend; máy chạy Docker không cần cài Node.
- **Dockerfile chưa được build thử** (lúc viết, Docker daemon không chạy). Nếu lỗi, dùng Cách A.

## 8. Biến môi trường (tuỳ chọn)

Đặt trước lệnh chạy, ví dụ `MFPLATE_API_KEY=abc python3 -m uvicorn mfplate.api:app --port 8000`.

| Biến | Mặc định | Tác dụng |
|---|---|---|
| `MFPLATE_REPO_ROOT` | `../restran` (nếu có), không thì thư mục cha | Đường dẫn repo train `restran` (code mạng `src/`, `configs/`) |
| `MFPLATE_API_KEY` | (trống = tắt) | Bắt buộc header `X-API-Key`. Giao diện web sẽ hỏi khoá một lần và lưu trong trình duyệt |
| `MFPLATE_DB` | `var/mfplate.sqlite3` | File SQLite lưu lịch sử và phản hồi |
| `MFPLATE_THREADS` | `0` (tự chọn) | Số luồng CPU cho PyTorch |
| `MFPLATE_MODEL` | `model/restran_lockbox_combo3_s42_ep30.pth` | Đường dẫn checkpoint khác (vẫn phải đúng SHA-256) |
| `MFPLATE_WEBUI_DIST` | `webui/dist` | Thư mục frontend đã build |
| `MFPLATE_SAMPLES_DIR` | `samples` | Thư mục track mẫu |
| `MFPLATE_API_URL` | `http://127.0.0.1:8000` | Backend mà Vite dev server (Cách B) và giao diện Gradio gọi tới |

## 9. Chạy test

Không cần bật server:

```bash
python3 -m pytest tests                     # 67 test: decoder, tiền xử lý, API, giao diện
```

Cần backend đang chạy ở cổng 8000:

```bash
python3 scripts/run_e2e.py --url http://127.0.0.1:8000       # 30 case E2E + 14 case lỗi
python3 scripts/bench_latency.py --url http://127.0.0.1:8000 # latency p50/p95
```

## 10. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|
| `command not found: python` | Máy chỉ có `python3` | Dùng `python3` |
| `ModuleNotFoundError: Model code not found: no src/models/restran.py under ...` khi khởi động | Không tìm thấy repo `restran` (không nằm cùng cấp với `masters_thesis_tienphat/`) | Đặt `restran/` cạnh thư mục này, hoặc chạy với `MFPLATE_REPO_ROOT=<đường dẫn tới restran>` |
| `ModuleNotFoundError: No module named 'torch'` (hoặc `fastapi`) khi gõ `uvicorn ...` | Lệnh `uvicorn` thuộc một bản Python khác | Dùng `python3 -m uvicorn mfplate.api:app --port 8000` |
| Mở http://127.0.0.1:8000 nhưng bị chuyển sang trang `/docs` | Frontend chưa được build | `cd webui && npm install && npm run build` |
| Trang trắng ở cổng 8000 sau khi vừa build lần đầu | Thư mục `webui/dist/assets` chưa có lúc server khởi động | Tắt server và chạy lại |
| Giao diện hiện "Model chưa sẵn sàng"; `/health` trả 503 | Thiếu checkpoint trong `model/` (`detail` bắt đầu bằng `FileNotFoundError: checkpoint not found`), hoặc file không khớp SHA-256 đã đóng băng | Xem trường `detail` trong `curl http://127.0.0.1:8000/health`; chép lại file theo [`model/README.md`](../model/README.md) |
| `Address already in use` | Cổng 8000 đang bị một server khác dùng | `lsof -ti:8000 \| xargs kill`, hoặc chạy với `--port 8001` |
| Trang ở cổng 5173 không mở được hoặc báo lỗi kết nối API | Mở bằng `127.0.0.1:5173`, hoặc backend chưa chạy | Mở `http://localhost:5173`; bật backend ở cổng 8000 trước |
| Không có nút track mẫu | Thư mục `samples/` trống. Các file zip không được đưa lên git vì là dữ liệu có giấy phép | Chép lại 3 file zip vào `samples/`, hoặc tạo lại bằng `python3 scripts/make_samples.py` (cần chạy `scripts/offline_cache.py` trước) |
| Giao diện hỏi "API đang bật khoá truy cập" | Server chạy với `MFPLATE_API_KEY` | Nhập đúng khoá đó |
| Điện thoại không mở được trang | Server chỉ nghe trên máy, hoặc khác mạng Wi-Fi | Chạy uvicorn với `--host 0.0.0.0`; cho phép kết nối trong tường lửa macOS; điện thoại và máy phải cùng mạng |
