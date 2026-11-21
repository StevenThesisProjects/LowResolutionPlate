# Kịch bản demo 5 phút

Chuẩn bị:
- API và UI đang chạy: `docker compose -f docker/docker-compose.yml up` (từ thư mục `masters_thesis_tienphat/`), hoặc chạy các lệnh ở mục 2 của README;
- mở `http://localhost:8000` (hoặc URL tunnel trên điện thoại). Mở thẳng một track bằng `?sample=<tên file>`, ví dụ `http://localhost:8000/?sample=easy_track_14372.zip`;
- có sẵn 3 file trong `samples/`;
- **phương án dự phòng:** video `demo_backup.mp4` và tab `http://localhost:8000/docs`.

| Thời gian | Làm gì | Nói gì |
|---|---|---|
| 0:00–0:40 | Mở trang, chỉ dòng phạm vi | Bài toán: 5 ảnh crop biển số khoảng 47×17 px của cùng một xe. Model ResTran đã đóng băng, train bằng công thức của luận văn: 0 tham số thêm, 0 chi phí suy luận thêm. Chỉ dùng cho biển Brazil/Mercosur 1 dòng |
| 0:40–1:40 | **Tình huống 1, dễ:** bấm nút mẫu "Dễ đọc" | Đọc SFB6A67, nhãn TỰ ĐỘNG. Confidence từng ký tự gần 100%. Top-2 chỉ có xác suất 0,003%. Mục "Đóng góp của từng frame" cho thấy trọng số fusion của 5 frame |
| 1:40–2:40 | **Tình huống 2, mờ:** bấm "Ảnh rất mờ" | Độ nét trung bình chỉ 17, là track mờ nhất trong số track val 999 được đọc đúng và tự động. Model vẫn đọc đúng RHP5D14, điểm vượt ngưỡng. Ảnh mờ là chỗ công thức giúp nhiều nhất: trên lock-box, công thức hơn baseline +3,4 điểm ở 25% track mờ nhất (phân tích thăm dò) |
| 2:40–3:40 | **Tình huống 3, cần kiểm tra:** bấm "Cần kiểm tra" | Model đọc IAR5J37 nhưng nhãn là EAR5J37. Điểm 0,905 dưới ngưỡng 0,975, nên nhãn đỏ CẦN KIỂM TRA. Ký tự đầu và ký tự thứ 6 có confidence thấp nhất, và top-2 khác ở vị trí 6. Bấm "Sai", sửa thành EAR5J37, bấm "Gửi" → lịch sử ghi lại |
| 3:40–4:20 | Bỏ bớt frame: tải 1–2 ảnh của track dễ | Cảnh báo "Chỉ có N frame", kèm accuracy lock-box với N frame (55,6% / 69,0%). Ngưỡng tự chặt hơn khi ít frame |
| 4:20–5:00 | Tổng kết bằng số | Trên 1.000 track lock-box chưa từng dùng: tự động 74% track với precision 95,4%; nhóm cần kiểm tra giữ lại 83% lỗi. Latency khoảng 22 ms/track trên CPU. App cho cùng kết quả với đánh giá offline (0/999 track lệch) |

Câu hỏi thường gặp:
- **"Sao không dùng decode theo cú pháp biển?"** Có trong app (`constrained`), nhưng chỉ +0,05 điểm. Không lỗi nào của model có chuỗi đúng mang xác suất cao hơn chuỗi model chọn, nên decode không sửa được.
- **"Confidence 99% có nghĩa là đúng 99%?"** Không. Model tự tin quá mức (ECE 0,20). Confidence chỉ dùng để xếp hạng; ngưỡng được chốt để nhóm tự động đạt precision 95%.
- **"Có dùng tập test chính thức không?"** Không, theo quyết định của thầy. Lock-box 1.000 track được tách trước và chấm một lần.
