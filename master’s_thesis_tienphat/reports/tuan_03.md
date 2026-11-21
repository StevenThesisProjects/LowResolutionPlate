# Tuần 3 (19–25/10): Kiểm tra data pipeline (Issue #1)

## Roadmap yêu cầu

1. Xác minh việc nhân đôi số mẫu mỗi epoch. Docstring nói đã sửa, nhưng RUNS.md vẫn ghi 563 batch/epoch.
2. Nếu vẫn nhân đôi: sửa thành 1 mẫu mỗi track (50% LR thật, 50% synthetic + JPEG q 15–50), chạy 3 seed, **chỉ đổi biến này**.
3. Parity test API so với offline (greedy) → `parity_baseline.json`.
4. Deliverable: `pipeline_audit.md`, `results_pipeline.csv`, `parity_baseline.json`.

## Dùng gì thay thế

| Hạng mục roadmap | Thay bằng | Lý do |
|---|---|---|
| Xác minh nhân đôi | Đã xác minh: **đây là thiết kế của code gốc, không phải lỗi** (bằng chứng bên dưới) | Lock-box protocol ghi rõ "mỗi track kèm một mẫu synthetic như code gốc". Bài báo mô tả đúng như vậy |
| Sửa thành 1 mẫu/track, 3 seed | **Chưa chạy.** Câu hỏi tương ứng ("nửa synthetic có giúp không?") là nhánh `nosyn` (`--no-synthetic`) của vòng v2, lập kế hoạch ngày 31/10. **Vòng v2 chưa chạy run nào** (tính đến 21/11), và bài ICCIES đã nộp với kết quả cũ, không dùng vòng này. Code v2 (3 cờ mới) mặc định tắt và vẫn train giống hệt bản đã chạy E4/N1 (regression check 21/11, ghi ở RUNS.md, mục "Thay đổi code 31/10/2026") | Đổi thành 1 mẫu/track làm giảm một nửa số bước tối ưu, nên phải là một run kiểm soát riêng, như roadmap nhắc |
| `parity_baseline.json` | `artifacts/parity_final.json`, đo trên model chốt (Tuần 7) | Không còn mốc #13 để đo parity riêng |

## Kết quả

**Số mẫu mỗi epoch** (`src/data/dataset.py`, hàm `_index_samples`):
- Ở chế độ train, mỗi track thêm **1 mẫu LR thật** (`is_synthetic: False`). Nếu không bật `--no-synthetic`, track thêm **1 mẫu synthetic** lấy từ ảnh HR rồi làm xấu đi.
- Val và test chỉ có mẫu LR thật.
- Log của run lock-box ghi `Train samples : 36002` = 2 × 18.001 track. Với batch 64 là ⌈36.002 / 64⌉ = 563 batch/epoch, khớp con số trong RUNS.md.
- Regression test (`tools/regression_check.py`) xác nhận: khi tắt mọi cờ, code mới train giống từng bit code gốc của BTC (commit `3ca05c0`). Vậy hành vi 2 mẫu/track có từ code gốc.

**Những gì đã làm với nhánh synthetic thay cho thí nghiệm roadmap:**
- Ngày 01/10 đo domain gap: model đọc đúng ảnh LR thật 78,33%, nhưng ảnh synthetic theo công thức cũ thì 97,83% (RUNS.md, mục 01/10 §3). Nửa synthetic dễ hơn ảnh thật khoảng 20 điểm.
- Run #38: hiệu chỉnh nhiễu Gauss của nhánh synthetic (độ lệch chuẩn 0,02–0,08). Mặc định của thư viện làm lệch pixel khoảng 52/255. Đây là thành phần "calibrated degradation" của công thức chốt.
- Nhánh `nosyn` của vòng v2 sẽ trả lời câu hỏi của roadmap ("bỏ hay giữ nửa synthetic"). Tiêu chí đã viết trước: giữ ứng viên nếu Last-6 trên val 999, trung bình 3 seed, hơn `ref` ít nhất +0,6.

**Parity** đo ở Tuần 7 trên model chốt: 0/999 track lệch giữa API và offline.

## Deliverable

| Roadmap | File thay thế | Trạng thái |
|---|---|---|
| `pipeline_audit.md` (batch/epoch trước–sau) | Mục Kết quả ở trên: 36.002 mẫu, 563 batch/epoch, giải thích nguồn gốc | ✅ (giải thích, không sửa) |
| `results_pipeline.csv` (3 seed) | Chưa có: nhánh `nosyn` của vòng v2 chưa có kết quả trong repo | ❌ |
| `parity_baseline.json` | `artifacts/parity_final.json` | ✅ (Tuần 7) |

## Việc còn lại

- Nếu thầy vẫn muốn trả lời câu hỏi này trong luận văn: chạy nhánh `nosyn` 3 seed theo lệnh có sẵn ở RUNS.md, mục "Vòng v2: lệnh chạy". Chi phí khoảng 3 run × 35–40 phút trên V100.
