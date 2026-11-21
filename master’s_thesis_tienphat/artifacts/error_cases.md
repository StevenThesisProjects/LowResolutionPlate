# Các trường hợp lỗi (sinh tự động bởi `scripts/run_e2e.py`, 2026-11-20T21:31:24)

**14/14 trường hợp trả đúng mã HTTP và mã lỗi đã thiết kế.** Server không crash ở trường hợp nào. Mọi lỗi có cùng dạng JSON `{"error": {"code", "message"}, "request_id"}`.

| # | Trường hợp | HTTP mong đợi | Mã lỗi mong đợi | HTTP nhận | Mã lỗi nhận | Thông báo | Đạt |
|---|---|---|---|---|---|---|---|
| 1 | không gửi file nào | 400 | `no_files` | 400 | `no_files` | Cần gửi 1 đến 5 ảnh crop biển số (trường `files`) | ✓ |
| 2 | 6 ảnh (quá 5) | 400 | `too_many_files` | 400 | `too_many_files` | Nhận 6 ảnh, tối đa 5 ảnh của cùng một biển số | ✓ |
| 3 | file .txt | 415 | `unsupported_type` | 415 | `unsupported_type` | plate.txt: chỉ nhận ảnh JPEG hoặc PNG | ✓ |
| 4 | ảnh GIF đổi đuôi .jpg | 415 | `unsupported_type` | 415 | `unsupported_type` | p.jpg: chỉ nhận ảnh JPEG hoặc PNG | ✓ |
| 5 | JPEG bị cắt cụt (file hỏng) | 422 | `corrupt_image` | 422 | `corrupt_image` | cut.jpg: không giải mã được ảnh (file hỏng?) | ✓ |
| 6 | file rỗng | 422 | `empty_file` | 422 | `empty_file` | e.jpg: file rỗng | ✓ |
| 7 | ảnh > 2 MB | 413 | `file_too_large` | 413 | `file_too_large` | big.jpg: vượt 2 MB mỗi ảnh | ✓ |
| 8 | ảnh 4000 px chiều rộng | 413 | `image_too_large` | 413 | `image_too_large` | wide.png: ảnh 4000x8 px vượt giới hạn 2048 px mỗi cạnh | ✓ |
| 9 | ảnh 2×3 px | 422 | `image_too_small` | 422 | `image_too_small` | dot.png: ảnh 3x2 px quá nhỏ | ✓ |
| 10 | zip hỏng | 400 | `zip_invalid` | 400 | `zip_invalid` | t.zip: không đọc được file zip | ✓ |
| 11 | zip không có ảnh | 400 | `zip_no_images` | 400 | `zip_no_images` | t.zip: zip không chứa ảnh .jpg/.png nào | ✓ |
| 12 | gửi lẫn zip và ảnh | 400 | `mixed_upload` | 400 | `mixed_upload` | Gửi 1 file zip HOẶC 1–5 ảnh, không gửi lẫn | ✓ |
| 13 | decoder không tồn tại | 422 | `invalid_parameter` | 422 | `invalid_parameter` | query.decoder: Input should be 'greedy', 'beam' or 'constrained' | ✓ |
| 14 | feedback cho request_id không tồn tại | 404 | `not_found` | 404 | `not_found` | Không có dự đoán với request_id 00000000000000000000000000000000 | ✓ |
