# MF-SR-OCR — Đề xuất & Xác nhận Tuần 1

> Người trả lời: Minh · Ngày: \_**\_/\_\_**/2026

---

## 1. ĐỀ XUẤT BÀI TOÁN

| Mục                    | Nội dung                                                                                                                                                                          |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên**                | **MF-SR-OCR: End-to-End Joint Multi-Frame Super-Resolution and Recognition for Low-Resolution License Plate** — theo bài đã nộp Hội nghị FISAT ngày 10/08/2026 ([link hội nghị](https://daihoc.fpt.edu.vn/hcm/hoi-nghi-fisat/)). Prototype demo: nhận dạng biển số độ phân giải thấp từ chuỗi 5 frame camera giám sát, có cờ kiểm tra thủ công |
| **Problem**            | Camera giao thông nén mạnh, biển số chỉ ~46×19 px. OCR đơn frame sai nhiều (74.45%). Người vận hành phải đọc tay từng track.                                                      |
| **Đối tượng**          | Nhân viên vận hành/điều tra xem lại footage; bãi xe, trạm thu phí cần tra cứu lại phương tiện                                                                                     |
| **AI**                 | Mô hình S1 (Joint MF-SR-OCR) + ngưỡng tin cậy để tự động chấp nhận hoặc chuyển người kiểm tra (selective prediction)                                                              |
| **Prototype**          | Web app: upload 1–5 ảnh crop biển số (hoặc file zip nhiều track) → chuỗi biển số, confidence từng ký tự, ảnh SR, cờ "cần kiểm tra"; xuất CSV                                      |
| **Khả thi 3 tháng vì** | Model, data pipeline, training CLI đã có. Inference chạy được trên CPU (~0.2 s/track) → deploy không cần GPU. Chỉ thêm: kiểm chứng thống kê, 1 module ngưỡng tin cậy, 1 app mỏng. |

---

## 2. CÔNG NGHỆ & KIẾN TRÚC

| Thành phần      | Lựa chọn                                                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AI/ML model** | S1 checkpoint (EMA weights), PyTorch; decode greedy + constrained beam search; `seed_everything` + `cudnn.deterministic=True`                                                       |
| **Module mới**  | Ngưỡng tin cậy τ chọn trên val: precision ≥ 95% cho track tự động chấp nhận; phần còn lại gắn cờ review                                                                             |
| **Dataset**     | Train: Scenario-A + Scenario-B (trừ val, test). Val: 999 track cũ (giữ nguyên để so với lịch sử). **Test-holdout mới: 1,000 track Scenario-B, tách từ phần train, dùng đúng 1 lần** |
| **Backend**     | FastAPI + Uvicorn; `/health`, `/predict` (1–5 ảnh), `/predict_batch` (zip → CSV)                                                                                                    |
| **Frontend**    | **React (Vite)** — xem bảng lựa chọn                                                                                                                                                |
| **Database**    | SQLite: log request, kết quả, confidence, latency, sửa tay của người review                                                                                                         |
| **Deployment**  | Docker CPU-only + Cloudflare Tunnel (hoặc lab server). Dự phòng: video demo ghi sẵn + 10 track mẫu có sẵn trong app                                                                 |

**Lựa chọn frontend (tối đa 3):**

| Lựa chọn               | Phù hợp khi                  | Trade-off                                                                                                             |
| ---------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| A. Gradio              | Chỉ mạnh Python              | ~1 tuần, có sẵn component ảnh/zip/bảng; đồng bộ stack với RoadBuddy (Quyền) để dùng chung hạ tầng lab; UI ít tùy biến |
| B. Streamlit           | Muốn dashboard batch đẹp hơn | Tương đương A; xử lý state kém hơn cho tab review                                                                     |
| **C. React (đã chọn)** | Có kinh nghiệm React         | UI tốt, tùy biến tab review/batch; tốn thêm 2–3 tuần so với A                                                         |

Chọn **C** vì Minh có kinh nghiệm React/Next.js (xem câu 5). Rủi ro lệch trọng tâm thesis được kiểm soát bằng: <!-- VD: giới hạn UI ở 3 màn hình (Upload / Batch / Review), đóng băng UI sau Tuần __ -->

**Stretch goal (chỉ làm nếu Tuần 1–6 đúng hạn):** nhận video → YOLOv8 phát hiện biển số → IoU tracking lấy 5 crop/phương tiện → đưa vào S1. Không đưa vào tiêu chí MVP.

**Flow hệ thống:**

```
User (laptop/điện thoại)
  → React UI: upload 1–5 crop biển số | zip nhiều track
  → FastAPI /predict | /predict_batch
      1. Validate (jpg/png, 1–5 frame, ≤ 5MB/ảnh; thiếu frame → lặp frame cuối)
      2. Resize 32×128, chuẩn hóa
      3. S1: STN → DCNv2 align → FrameSR (×2) → ResBackbone(GN) → Attention Fusion → BiLSTM → CTC
      4. Decode: greedy | constrained beam (layout Brazil)
      5. So confidence với τ → accepted | needs_review
  → JSON {plate, char_conf[], track_conf, status, sr_images[], latency_ms, model_version}
  → UI hiển thị LR vs SR, kết quả, cờ; người review sửa → SQLite
```

---

## 3. XÁC NHẬN TUẦN 1

### Câu 1 — Sau 30/07 đã chạy multi-seed chưa? PR Joint MF-SR-OCR #12 đã merge chưa?

- **Multi-seed:** ☑ Đã chạy · ☐ Chưa chạy
  - Kết quả S1, 3 seed (val 999 track, best epoch):

    | Seed           | Best val acc      | Epoch đạt best | Số epoch đã chạy (early stop) |
    | -------------- | ----------------- | -------------- | ----------------------------- |
    | 42             | 79.78%            | 28             | 46                            |
    | 100            | 80.08%            | 32             | 50                            |
    | 2026           | 79.98%            | 24             | 42                            |
    | **Mean ± std** | **79.95 ± 0.15%** |                |                               |

  - Thư mục kết quả: `LowResolutionPlate/crnn_and_stn/results/multi-seed/s1_mf_sr_ocr/`

- **PR #12 → đổi tên đề tài:** _MF-SR-OCR: End-to-End Joint Multi-Frame Super-Resolution and Recognition for Low-Resolution License Plate_ (trùng tên bài nộp Hội nghị FISAT, 10/08/2026 — <https://daihoc.fpt.edu.vn/hcm/hoi-nghi-fisat/>)
  - Nhánh code: `feature/run-multi-seeda-and-evulate` (đã push lên `origin`)
  - Trạng thái merge: ☐ Đã merge · ☑ **Chưa merge** — lý do: tách nhánh code riêng (`feature/run-multi-seeda-and-evulate`)
- Ghi chú: **\_\_**

### Câu 2 — 1 run S1 (55 epoch) mất bao nhiêu giờ trên GPU hiện có?

- **GPU:** 1× **Tesla V100-SXM2-32GB** (VRAM 32 GB) — thuê cloud, region `na-01`, SLA 99.9%
  - CPU Intel Xeon Gold 6248 (10 cores) · RAM 94 GB · Disk 890 GB (read 2,555 MB/s · write 1,102 MB/s) · OS Linux
  - Giá thuê: **9,000 ⚡/giờ**
- **Thời gian/epoch:** ~580 s (~9.7 phút) — train ~565 s (1188 batch, bs=32) + val ~16.5 s
- **1 run 55 epoch (full):** ≈ **8.9 giờ**
- **1 run thực tế (early stopping, patience 18):** **6.8–8.0 giờ** (dừng ở epoch 42–50)
- **Chi phí ước tính / run:** full 55 epoch ≈ 8.9 h × 9,000 ≈ **80,000 ⚡** · thực tế (early stop) ≈ **61,000–72,000 ⚡**
- **Hệ quả cho Tuần 2–4:** ~3 run/ngày/GPU nếu chạy liên tục (lý thuyết ≤ ~60 run / 21 ngày) → dự kiến chạy \_\_\_ run (≈ \_\_\_ ⚡)
- Ghi chú: config hiện tại là 60 epoch, không phải 55. Nguồn số liệu: cột `epoch_time_s` trong `history_s1_seed*.csv`.

### Câu 3 — License dataset ICPR 2026 LRLPR có cho phép hiển thị ảnh trên demo có URL public không?

- ☐ Cho phép · ☑ **Không cho phép** (nếu không có văn bản đồng ý của tác giả) · ☐ Chưa rõ
- **Kết luận:** Chỉ được dùng ảnh dataset để minh họa trong **bài báo và buổi thuyết trình học thuật** (thesis, slide bảo vệ, paper). Nếu public URL cho ai cũng mở được và hiển thị/cho tải ảnh LR/HR của dataset thì vượt phạm vi đó và bị tính là _redistribution_ → **không được làm**, trừ khi có sự cho phép bằng văn bản của tác giả.
- **Tên license / điều khoản liên quan:** _ICPR26-LRLPR Dataset – License Agreement_ (bản đã ký khi đăng ký dataset), mục **Terms and Conditions**:

  > "The dataset is available **exclusively** to academic researchers from educational or research institutions for **non-commercial use**."

  > "Without the expressed permission of the authors, any of the following will be considered illegal: **redistribution, modification, and commercial usage** of this dataset in any way or form, either partially or in its entirety;"

  > "For the sake of privacy, all images in this dataset are **only allowed for demonstration in academic publications and presentations**;"

  > "Once the competition paper becomes publicly available, all publications using the ICPR26-LRLPR dataset **must cite it**."

- **Trang chủ cuộc thi** (mục _Get Started_):

  > "The dataset will be made available exclusively for non-commercial use and only to institutional email addresses from educational or research organizations."

  > "Research use and results authorization: the provided dataset is intended solely for research purposes." (mục _Rules_)

- **Link:**
  - Trang chủ: <https://icpr26lrlpr.github.io/>
  - License agreement: <https://icpr26lrlpr.github.io/license-agreement.pdf>
  - Ngày truy cập: 30/09/2026
- **Nếu không cho phép** (áp dụng cho demo):
  - ☑ Demo public **không** chứa ảnh dataset: 10 track mẫu dùng ảnh tự chụp hoặc ảnh có license mở (mục 2 · Deployment cần đổi theo)
  - ☑ Dùng ảnh dataset (LR vs SR) chỉ trong slide bảo vệ, thesis và video demo chiếu tại buổi bảo vệ
  - ☐ Demo có ảnh dataset chỉ chạy trong mạng lab hoặc sau đăng nhập, giới hạn cho hội đồng/GVHD _(vẫn nên hỏi tác giả trước)_
  - ☐ Gửi email xin phép tác giả (UFPR) nếu vẫn muốn public — ngày gửi: \_**\_ · phản hồi: \_\_**
  - Model/checkpoint train trên dataset: license không nói rõ → chỉ dùng phi thương mại; nếu định public checkpoint thì hỏi tác giả trước
- Trích dẫn trong thesis: thêm citation cho competition paper ICPR 2026 LRLPR khi paper được công bố

### Câu 4 — Hạn nộp thesis và ngày bảo vệ chính thức?

- **Trạng thái:** ☑ Chưa có lịch chính thức. Nhà trường sẽ gửi kế hoạch qua email **trước ngày 15/10/2026** cho các học viên dự kiến làm ĐATN trong kỳ này.
- **Nguồn:** phản hồi của nhà trường (ngày nhận: \_**\_/\_\_**/2026)
- **Cập nhật khi có email:**

| Mốc                       | Ngày                 |
| ------------------------- | -------------------- |
| Hạn nộp bản thảo cho GVHD | \_**\_/\_\_**/20\_\_ |
| Hạn nộp thesis chính thức | \_**\_/\_\_**/20\_\_ |
| Ngày bảo vệ               | \_**\_/\_\_**/20\_\_ |

- **Tạm thời:** lập kế hoạch Tuần 2–4 dựa trên mốc 3 tháng trong đề xuất; điều chỉnh lại timeline sau 15/10/2026.
- ☐ Đã nhận email kế hoạch (ngày: \_\_\_\_ ) → điền bảng trên

### Câu 5 — Có làm được React/Next.js không?

- ☑ **Có** → Frontend dùng **React**, Backend **FastAPI** (đã cập nhật mục 2)
- Mức kinh nghiệm: **Cao** — ☑ Đã làm dự án thực tế · ☐ Đã học/làm project cá nhân
- Framework cụ thể: ☑ **React + Vite** · ☐ Next.js
- UI library dự kiến: **Tailwind CSS**
- Thời gian ước lượng cho FE MVP: \_\_\_ tuần
