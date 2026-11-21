# Tuần 10 (07–13/12): Testing

Phần tự động làm trước lịch (20/11). Phần cần người dùng do Phát làm.

## Roadmap yêu cầu

1. 30 test case E2E: 20 track test chia đều layout × độ nét, cộng 10 case biên (1–4 frame, ảnh xám, ảnh xoay, ảnh không phải biển số).
2. Đo latency p50/p95, 5 request đồng thời.
3. User test 5 người + SUS.

Deliverable: `test_report.md`, `latency.csv`, `sus_results.csv`.

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng | Lý do |
|---|---|---|
| 20 track **test** | 20 track **val 999**, chọn tất định (seed 2026): 3 track cho mỗi (Mercosur, tứ phân vị độ nét) và 2 track cho mỗi (Brazilian, tứ phân vị); nửa gửi dạng 5 ảnh, nửa dạng zip | Không dùng tập test chính thức. Không lấy lock-box, để giữ lock-box cho việc đánh giá |

## Kết quả E2E (`scripts/run_e2e.py`, server uvicorn thật → `artifacts/test_report.md`)

- **30/30 case trả 200 với đủ trường JSON** (tiêu chí MVP số 2 ✅).
- 20 track: đọc đúng 16/20. 9 track bị gắn cờ, trong đó có cả 4 track đọc sai. Không track sai nào lọt vào nhóm tự động trong 20 track này.
- 10 case biên đều được trả lời. Ảnh xoay 90°, lộn ngược 180°, nhiễu ngẫu nhiên, thẻ trắng và frame lặp 5 lần đều bị gắn cờ "cần kiểm tra". Ảnh xoay 90° có thêm cảnh báo tỉ lệ khung hình.
- **Điểm cần ghi vào luận văn:** track T01 (nhãn KRB6I42) bị đọc sai thành ARB6I42 ở mọi số frame. Với 1, 2 và 5 frame thì bị gắn cờ, nhưng **với 3 và 4 frame thì không** (E03, E04). Đây là phần lỗi khoảng 5% còn lọt qua nhóm tự động, khớp với thiết kế precision 95%. Triage giảm lỗi chứ không loại hết.

## Latency (`scripts/bench_latency.py` → `artifacts/latency_cpu.csv`, `artifacts/latency_summary.json`)

Máy: Apple M4 (10 nhân), macOS, PyTorch 2.13.0 CPU, uvicorn 1 worker, gọi qua HTTP trên localhost:

| Kịch bản | Số request | p50 | p95 | Thời gian model (p50) |
|---|---|---|---|---|
| 1 track 5 frame, tuần tự | 200 | 21,9 ms | **24,1 ms** | 18,1 ms |
| 1 frame | 50 | 11,9 ms | 12,6 ms | 8,5 ms |
| 2 frame | 50 | 14,3 ms | 14,8 ms | 10,8 ms |
| 3 frame | 50 | 16,2 ms | 17,0 ms | 12,6 ms |
| 4 frame | 50 | 17,8 ms | 18,7 ms | 14,1 ms |
| 5 client đồng thời, 5 frame | 200 | 108,3 ms | 112,1 ms | 21,5 ms |

- Tiêu chí MVP số 3 (p95 ≤ 1 s): **đạt, 24 ms**.
- Khi 5 client gửi cùng lúc, request phải xếp hàng sau lock của model, nên latency khoảng 5 × 21 ms. Thông lượng là 44,8 request/s.
- Các con số này đo trên M4. Server cloud CPU hoặc container Docker giới hạn CPU sẽ chậm hơn. **Cần đo lại trên máy demo thật** bằng cùng lệnh.

## User test + SUS

Đã chuẩn bị:
- `docs/user_test.md`: kịch bản 4 nhiệm vụ, 10 câu SUS tiếng Việt, mẫu `sus_results.csv`;
- `scripts/sus_score.py`: tính điểm SUS theo cách chuẩn.

**Chưa có người tham gia.** Phát cần tổ chức với 5 người.

## Deliverable

| Roadmap | File | Trạng thái |
|---|---|---|
| `test_report.md` (30 case E2E) | `artifacts/test_report.md` + `artifacts/e2e_results.json` | ✅ |
| `latency.csv` | `artifacts/latency_cpu.csv` + `artifacts/latency_summary.json` | ✅ (máy dev; đo lại trên máy demo) |
| `sus_results.csv` | Mẫu trong `docs/user_test.md` | ❌ Phát làm user test |

## Lệnh

```bash
cd masters_thesis_tienphat
python3 -m uvicorn mfplate.api:app --port 8000 &
python3 scripts/bench_latency.py --url http://127.0.0.1:8000
python3 scripts/run_e2e.py --url http://127.0.0.1:8000
python3 scripts/sus_score.py artifacts/sus_results.csv     # sau user test
```
