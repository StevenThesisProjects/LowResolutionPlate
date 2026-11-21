# User test (roadmap Tuần 10): kịch bản và bảng hỏi SUS

**Người tham gia:** 5 người, ưu tiên người từng xem camera giám sát hoặc làm vận hành bãi xe. Mỗi người khoảng 10 phút, dùng điện thoại của chính họ qua URL tunnel.

**Trước khi bắt đầu:** giải thích app đọc biển số Brazil/Mercosur từ ảnh crop; không hướng dẫn cách dùng giao diện.

## Nhiệm vụ

Ghi lại cho từng nhiệm vụ: hoàn thành không, thời gian, số lần hỏi lại.

| # | Nhiệm vụ | Hoàn thành khi |
|---|---|---|
| 1 | Đọc biển số của track mẫu `easy` | Người tham gia nói đúng biển số app trả về |
| 2 | Đọc track `review`, rồi cho biết app có chắc chắn không | Người tham gia nhận ra cờ "cần kiểm tra" |
| 3 | Báo cho app biết kết quả ở nhiệm vụ 2 là sai, biển đúng là EAR5J37 | Lịch sử có dòng "sai → EAR5J37" |
| 4 | Chỉ ra frame nào app tin nhất ở nhiệm vụ 1 | Người tham gia chỉ frame có trọng số cao nhất |

## Bảng hỏi SUS

Trả lời từ 1 (hoàn toàn không đồng ý) đến 5 (hoàn toàn đồng ý). Các câu là bản dịch của 10 câu chuẩn trong Brooke (1996).

1. Tôi nghĩ tôi sẽ muốn dùng hệ thống này thường xuyên.
2. Tôi thấy hệ thống phức tạp một cách không cần thiết.
3. Tôi thấy hệ thống dễ dùng.
4. Tôi nghĩ tôi cần người có chuyên môn kỹ thuật hỗ trợ mới dùng được hệ thống này.
5. Tôi thấy các chức năng của hệ thống được kết hợp tốt với nhau.
6. Tôi thấy hệ thống có quá nhiều chỗ không nhất quán.
7. Tôi nghĩ hầu hết mọi người sẽ học dùng hệ thống này rất nhanh.
8. Tôi thấy hệ thống rất rườm rà, khó thao tác.
9. Tôi cảm thấy rất tự tin khi dùng hệ thống.
10. Tôi cần học nhiều thứ trước khi có thể dùng được hệ thống này.

## Ghi kết quả

Tạo `artifacts/sus_results.csv`:

```csv
participant,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
P1,4,2,4,1,4,2,5,1,4,2
```

Dòng P1 chỉ là ví dụ định dạng, hãy thay bằng câu trả lời thật. Sau đó chạy `python3 scripts/sus_score.py artifacts/sus_results.csv`.
