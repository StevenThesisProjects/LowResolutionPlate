# Tuần 8 (23–29/11): UI v1

Làm trước lịch: code đã xong ngày 20/11.

## Roadmap yêu cầu

Gradio:
- upload ảnh hoặc zip;
- hiển thị 5 frame kèm trọng số;
- hiển thị biển số, top-3 và cờ "cần kiểm tra";
- lịch sử trong phiên.

Deliverable: UI v1 chạy local, 3 ảnh chụp màn hình.

## Cập nhật 20/11: giao diện v2 bằng React

Bản Gradio (v1) chạy đúng nhưng bố cục rối: các component dựng sẵn xếp chồng, không có thứ bậc thông tin rõ ràng. Giao diện chính đổi sang **React + Vite** (`webui/`). Bản build (`webui/dist`) do chính API phục vụ ở `/`, nên demo chỉ cần một server, một cổng (8000), và không phải cấu hình CORS. Bản Gradio được giữ lại làm dự phòng (`ui/app.py`).

| Phần | Giao diện v2 |
|---|---|
| Bố cục | Hai cột trên máy tính (ảnh đầu vào / kết quả), một cột trên điện thoại; lịch sử và thông tin model ở dưới |
| Đầu vào | Kéo thả hoặc chọn 1–5 ảnh hay 1 zip; xem trước các frame theo đúng thứ tự API đọc (zip: chỉ `lr-*`); nút track mẫu; chọn bộ giải mã |
| Biển số | Vẽ thành thẻ biển số (dải màu theo layout), thanh độ tin cậy dưới từng ký tự (xanh ≥ 95%, vàng ≥ 80%, đỏ dưới 80%); ký tự sai so với nhãn trong zip bị gạch đỏ |
| Phân loại | Hộp "Có thể dùng tự động" / "Cần người kiểm tra" kèm lý do và thước điểm so với ngưỡng của đúng số frame |
| Chi tiết | 4 ô số (độ tin cậy, xác suất cả chuỗi, accuracy lock-box với số frame này, thời gian); top-3 với ký tự khác được tô màu; trọng số fusion và độ nét của từng frame, frame đóng góp nhiều nhất được viền |
| Khác | Phản hồi đúng/sai và sửa biển số; lịch sử phiên; chế độ sáng/tối theo hệ điều hành; nhập API key khi API bật khoá; mở thẳng một track mẫu bằng `?sample=<tên file>` |

Kiểm tra:
- `npm run build` thành công.
- 6 test API mới (`/` trả giao diện, `/samples` liệt kê và tải đúng file, chặn tên lạ và đường dẫn `../`); tổng 67/67 test đạt.
- Ảnh chụp bằng Chrome headless: máy tính chế độ sáng và tối, và khung rộng 390 px giả lập điện thoại. Lần chụp đầu ở 390 px phát hiện lỗi tràn ngang (cột lưới không co được dưới chiều rộng của chuỗi phiên bản model); đã sửa.

Ảnh chụp màn hình (deliverable của tuần này):
- `artifacts/screenshots/01_desktop_easy_light.png`: track dễ, tự động;
- `artifacts/screenshots/02_desktop_review_dark.png`: track bị đọc sai E→I, cần kiểm tra;
- `artifacts/screenshots/03_phone_blurry_light.png`: track rất mờ, khung 390 px.

Ảnh 03 chụp trong trình duyệt giả lập chiều rộng điện thoại, chưa phải trên máy thật. Việc thử trên iOS và Android vẫn thuộc Tuần 9.

## Dùng gì thay thế

Không có gì phải thay: làm đúng phương án A (Gradio) của roadmap. Một điểm khác: **UI chỉ là client của API** (`ui/app.py` gọi `POST /predict` qua HTTP và không nạp model). Nhờ vậy:
- kết quả trên UI chính là kết quả API, mà API đã kiểm parity với offline (Tuần 7);
- sau này muốn đổi sang Next.js (phương án C) chỉ cần viết lại UI.

## Chức năng (`ui/app.py`, Gradio 6.32)

| Yêu cầu roadmap / MVP | Có trong UI |
|---|---|
| Upload 1–5 ảnh hoặc 1 zip | `gr.File`, nhiều file, nhận `.jpg/.jpeg/.png/.zip` |
| Biển số + confidence tổng + confidence từng ký tự | Thẻ kết quả (biển số chữ lớn, layout, confidence, xác suất cả chuỗi) và bảng confidence từng ký tự |
| Top-3 | Bảng top-3 kèm xác suất |
| Cờ "cần kiểm tra" | Nhãn đỏ "CẦN KIỂM TRA" hoặc xanh "TỰ ĐỘNG", kèm lý do, điểm và ngưỡng của đúng số frame |
| 5 frame kèm trọng số | Gallery, mỗi frame ghi trọng số fusion và độ nét |
| Lịch sử trong phiên | Bảng lịch sử, lấy từ `GET /history` theo `session_id` (sinh ngẫu nhiên khi mở trang) |
| Nút feedback đúng/sai | Nút ✅/❌ và ô nhập biển số đúng → `POST /feedback` |
| Thông báo phạm vi | Dòng đầu trang: "chỉ biển Brazil/Mercosur 1 dòng…; kết quả chỉ để tham khảo" |
| Thêm | Cảnh báo khi ít hơn 5 frame (kèm accuracy E3); chọn decoder; JSON thô; thông tin model; 3 track mẫu (`samples/`) bấm là chạy |

## Đã kiểm tra

- Gọi các hàm của UI tới API thật (uvicorn, cổng 8000):
  - track mẫu "dễ" → SFB6A67, tự động;
  - track mẫu "cần kiểm tra" → IAR5J37 (nhãn EAR5J37), có cờ, top-2 là IAR5J27;
  - phản hồi "sai → EAR5J37" được ghi lại và hiện trong lịch sử;
  - không chọn file, và file `.txt` rỗng, đều hiện thông báo lỗi tiếng Việt.
- Chạy server Gradio (cổng 7861) và gọi qua `gradio_client`: track mẫu "mờ" → RHP5D14, đúng nhãn, gallery 5 frame.
- **Lỗi tìm ra và đã sửa:** khi chưa có `session_id` (client gọi qua API, hoặc trước khi trang chạy `load`), UI gửi `session_id` rỗng và API trả 422. Giờ UI chỉ gửi khi có giá trị; lịch sử trả rỗng khi chưa có phiên.

## Chạy

```bash
cd masters_thesis_tienphat
python3 -m uvicorn mfplate.api:app --port 8000      # terminal 1
python3 ui/app.py                         # terminal 2 → http://127.0.0.1:7860
```

## Deliverable

| Roadmap | File | Trạng thái |
|---|---|---|
| UI v1 chạy local | `ui/app.py` (Gradio, dự phòng) và `webui/` (React, giao diện chính) | ✅ |
| 3 ảnh chụp màn hình | `artifacts/screenshots/` (3 ảnh) | ✅ (ảnh điện thoại là giả lập 390 px; máy thật thuộc Tuần 9) |
