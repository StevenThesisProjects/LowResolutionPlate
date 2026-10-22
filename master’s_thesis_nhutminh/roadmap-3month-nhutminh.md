# PlateReader LR — Roadmap 12 tuần (MVP + Thesis)

## Lưu ý

1. **Dữ liệu là biển số Brazil, không phải Việt Nam.** Dataset ICPR 2026 LRLPR lấy từ UFPR. Constrained decode đang hard-code 2 layout Brazil (`LLLNLNN`, `LLLNNNN`). App chỉ được tuyên bố hoạt động trên miền dữ liệu này. Biển số VN là ngoài scope; chỉ test định tính 10 mẫu để nêu limitation.
2. **79.78% chưa đủ để kết luận.** Đây là 1 seed, trên tập val 999 track đã dùng để chọn cấu hình qua nhiều lần chạy (J1, J2, J3, S1) → số liệu có xu hướng lạc quan. Cần multi-seed + 1 tập held-out chưa từng dùng.
3. **So sánh với baseline đo lại, không so với con số report.** Baseline CRNN+STN đo trên codebase project là **75.78%**, không phải 77.00%. Mọi Δ phải tính trên cùng split, cùng code.
4. **Ngoài scope 3 tháng:** 3 ý tưởng novelty trong advisor report (FL-ISR, TG-STDA, SupCon), INT8/Jetson, so sánh 5+ SOTA. Nhận định "INT8 giảm 60% FLOPs giữ 95% accuracy" trong report chưa được đo — không đưa vào thesis.

---

## 0. ĐẦU VÀO SINH VIÊN

| Hạng mục          | Trạng thái                      | Nội dung                                                                                                                                                                                 |
| ----------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chuyên môn        | Có                              | Computer Vision, OCR (CRNN/CTC), Super-Resolution, multi-frame fusion                                                                                                                    |
| Kỹ năng lập trình | Có (Python) / **Giả định** (JS) | PyTorch, Albumentations, CLI training có preset. Giả định JS cơ bản → frontend thuần Python                                                                                              |
| Kinh nghiệm AI    | Có                              | STN, DCNv2, FrameSR, GroupNorm, EMA, CTC beam search; biết tự chẩn đoán lỗi (NaN, domain gap)                                                                                            |
| Ý tưởng           | Có                              | Joint Multi-Frame SR-OCR (S1) cho biển số độ phân giải thấp                                                                                                                              |
| Lĩnh vực ứng dụng | Có                              | Giám sát giao thông, tra cứu lại phương tiện từ footage camera                                                                                                                           |
| Dataset           | Có                              | ICPR 2026 LRLPR: Scenario-A 10,000 track (PNG, có HR) + Scenario-B 10,000 track (JPG nén nặng); mỗi track 5 frame LR ~46×19 px; val 999 track từ Scenario-B. Test set ICPR không có nhãn |
| Thiết bị          | Có                              | GPU 24GB (J2 cần batch 32 + grad accum 2). S1 inference CPU 192.75 ms/track                                                                                                              |
| Mục tiêu đầu ra   | Có                              | Thesis thạc sĩ + web app demo được                                                                                                                                                       |

**Hiện trạng kỹ thuật (30/07/2026):**

| Cấu hình                                                                                            |              Val Acc (999) | Ghi chú                                   |
| --------------------------------------------------------------------------------------------------- | -------------------------: | ----------------------------------------- |
| CRNN + STN (đo lại trên project)                                                                    |                     75.78% | Baseline dùng để so                       |
| J2 — GroupNorm + SR per-frame + L_SR                                                                |                     77.18% | 1 seed                                    |
| **S1 — PR #12 (STN → DCNv2 → FrameSR → ResBackbone GN → Fusion → BiLSTM, EMA, constrained decode)** | **79.78%** (greedy 79.58%) | 1 seed; 29.6M params; 109.08 GFLOPs/track |

**Câu hỏi cần Minh xác nhận trong Tuần 1 (tối đa 5):**

1. Sau 30/07 đã chạy multi-seed chưa? PR #12 đã merge chưa?
2. 1 run S1 (55 epoch) mất bao nhiêu giờ trên GPU hiện có? (Quyết định số run ở Tuần 2–4.)
3. License dataset ICPR 2026 LRLPR có cho phép hiển thị ảnh trên demo có URL public không?
4. Hạn nộp thesis và ngày bảo vệ chính thức?
5. Có làm được React/Next.js không? (Quyết định giữ Gradio hay nâng cấp UI.)

---

## 1. ĐỀ XUẤT BÀI TOÁN

| Mục                    | Nội dung                                                                                                                                                                          |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên**                | PlateReader LR — Nhận dạng biển số độ phân giải thấp từ chuỗi 5 frame camera giám sát, có cờ kiểm tra thủ công                                                                    |
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
| **Frontend**    | **Gradio** (khuyến nghị) — xem bảng lựa chọn                                                                                                                                        |
| **Database**    | SQLite: log request, kết quả, confidence, latency, sửa tay của người review                                                                                                         |
| **Deployment**  | Docker CPU-only + Cloudflare Tunnel (hoặc lab server). Dự phòng: video demo ghi sẵn + 10 track mẫu có sẵn trong app                                                                 |

**Lựa chọn frontend (tối đa 3):**

| Lựa chọn                    | Phù hợp khi                  | Trade-off                                                                                                             |
| --------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **A. Gradio (khuyến nghị)** | Chỉ mạnh Python              | ~1 tuần, có sẵn component ảnh/zip/bảng; đồng bộ stack với RoadBuddy (Quyền) để dùng chung hạ tầng lab; UI ít tùy biến |
| B. Streamlit                | Muốn dashboard batch đẹp hơn | Tương đương A; xử lý state kém hơn cho tab review                                                                     |
| C. Next.js                  | Có kinh nghiệm React         | UI tốt; tốn thêm 2–3 tuần, lệch trọng tâm thesis                                                                      |

Chọn **A** vì tiêu chí quyết định là **khả thi vận hành trong 3 tháng**; thời gian dồn cho kiểm chứng model.

**Stretch goal (chỉ làm nếu Tuần 1–6 đúng hạn):** nhận video → YOLOv8 phát hiện biển số → IoU tracking lấy 5 crop/phương tiện → đưa vào S1. Không đưa vào tiêu chí MVP.

**Flow hệ thống:**

```
User (laptop/điện thoại)
  → Gradio UI: upload 1–5 crop biển số | zip nhiều track
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

## 3. ROADMAP 12 TUẦN

| Tuần                | Mục tiêu                              | Công việc chính                                                                                                                                                                                                                   | Deliverable                                                                                                                |
| ------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **1** (05–11/10)    | Khóa scope, split, tính tái lập       | Trả lời 5 câu hỏi Mục 0. Review + merge PR #12. Thêm `seed_everything`, tắt `cudnn.benchmark`. Tách test-holdout 1,000 track Scenario-B (giữ nguyên val 999). Viết scope 1 trang.                                                 | `docs/scope_v1.md`; `splits/{train,val,test_holdout}.csv` + SHA-256; PR #12 merged; 2 run cùng seed cho cùng kết quả (log) |
| **2** (12–18/10)    | Multi-seed baseline vs S1             | Train lại CRNN+STN và S1 trên split mới, 3 seed (42, 100, 2026) = 6 run. Song song: FastAPI skeleton load S1.                                                                                                                     | `multiseed_results.csv` (mean ± std); API `/health`, `/predict` trả đúng 1 track val                                       |
| **3** (19–25/10)    | Metric đầy đủ + kiểm định             | Thêm CER, NED, Confidence Gap vào `validate()`. PSNR/SSIM của SR vs HR trên Scenario-A. McNemar S1 vs baseline (paired, 999 track) + bootstrap CI. **Gate:** S1 là winner nếu ΔAcc CI > 0 và p < 0.05.                            | `eval_report.md`, `stats.json`, `decision_gate.json`                                                                       |
| **4** (26/10–01/11) | Ablation tối thiểu                    | 1 seed mỗi arm: S1 − DCN, S1 − multi-frame (SR single-frame), S1 − EMA, số frame F ∈ {1, 3, 5}. PARSeq 1 run chỉ nếu Tuần 2–3 đúng hạn. Viết nháp Chương 3 (Method).                                                              | `ablation_table.csv`, nháp Chương 3                                                                                        |
| **5** (02–08/11)    | Selective prediction + error analysis | Vẽ đường coverage–precision theo confidence. Chọn τ trên val (precision ≥ 95%). Phân loại 100 track sai theo nhóm: đêm/chói, blur, nén, nhầm cặp ký tự (0/O, 1/I, 8/B).                                                           | `tau.json`, `coverage_precision.png`, `error_analysis.md`                                                                  |
| **6** (09–15/11)    | Freeze + test-holdout one-shot        | Freeze weights/config/τ. Chạy test-holdout **đúng 1 lần**. Đo latency CPU/GPU p50/p95, params, FLOPs. Thử TorchScript; lỗi với DCN thì giữ PyTorch eager.                                                                         | `freeze_manifest.json` (SHA-256), `test_holdout_result.json`, `latency.csv`                                                |
| **7** (16–22/11)    | Backend hoàn chỉnh                    | `/predict_batch` (zip → CSV), trả ảnh SR, confidence từng ký tự, status theo τ. Parity test API vs offline trên 999 track val.                                                                                                    | API v1, `parity_report.json`                                                                                               |
| **8** (23–29/11)    | UI v1                                 | Gradio 3 tab: (1) Single track: upload, LR vs SR, kết quả, cờ; (2) Batch: zip → bảng + CSV; (3) Review: sửa track bị gắn cờ → SQLite.                                                                                             | UI chạy local, 3 tab hoạt động                                                                                             |
| **9** (30/11–06/12) | Deploy + xử lý lỗi                    | Dockerfile CPU. Tunnel. Xử lý lỗi: sai định dạng, 0 hoặc > 5 frame, ảnh không phải biển số, file quá lớn, zip sai cấu trúc. Test trên Chrome Android + Safari iOS.                                                                | Docker image, URL demo, `error_cases.md` (≥ 8 case)                                                                        |
| **10** (07–13/12)   | Testing                               | 30 test case end-to-end (20 track val + 10 track test-holdout). Batch 1,000 track: đo throughput. 5 request đồng thời. 10 ảnh biển số VN (định tính, ghi limitation). User test 5 người + SUS + thời gian review có/không có app. | `test_report.md`, `throughput.csv`, `sus_results.csv`, `vn_plates_qualitative.md`                                          |
| **11** (14–20/12)   | Thesis kết quả + chuẩn bị demo        | Viết Chương 4 (Experiments) và 5 (Prototype & Evaluation). Hình định tính 4 cột: LR → SR → attention → dự đoán (5 đúng, 5 sai). Kịch bản demo 5 phút. Ghi video dự phòng.                                                         | Nháp Chương 4–5, `fig_qualitative.png`, `demo_script.md`, `demo_backup.mp4`                                                |
| **12** (21–27/12)   | Demo & đóng gói                       | Dry-run với Cố vấn. Sửa lỗi dry-run. README tái lập trên máy sạch. Tag release.                                                                                                                                                   | Demo live + video, repo tag `v1.0`, báo cáo kết quả PDF                                                                    |

---

## 4. MVP CUỐI KỲ

**Chức năng bắt buộc**

- Upload 1–5 ảnh crop biển số (jpg/png) cho 1 track
- Upload zip nhiều track → bảng kết quả + tải CSV
- Hiển thị ảnh LR gốc và ảnh SR tương ứng
- Gắn cờ `needs_review` theo τ; người dùng sửa và lưu
- Thông báo phạm vi: "Mô hình huấn luyện trên biển số Brazil (ICPR 2026 LRLPR). Không dùng cho biển số khác định dạng."

**Chức năng AI**

- S1 Joint MF-SR-OCR (hoặc baseline nếu S1 không qua gate Tuần 3)
- Constrained decode theo layout; bật/tắt được trong UI để so với greedy
- Confidence từng ký tự + confidence track + quyết định accept/review

**Input:** 1–5 crop biển số LR (hoặc zip thư mục `track_x/lr-1..5`)

**Output:** chuỗi biển số, confidence, status (accepted/needs_review), ảnh SR, latency, model version; CSV với chế độ batch

**Prototype "chạy được" khi đạt đủ:**

| #   | Tiêu chí                                 | Ngưỡng                                     |
| --- | ---------------------------------------- | ------------------------------------------ |
| 1   | Parity app vs offline trên 999 track val | 0 track lệch (greedy, cùng checkpoint)     |
| 2   | Test case end-to-end                     | 30/30 không crash                          |
| 3   | Latency 1 track (CPU)                    | p95 ≤ 1 s (đo thực tế ở Tuần 6 và 10)      |
| 4   | Batch                                    | 1,000 track chạy xong không lỗi, có CSV    |
| 5   | Truy cập                                 | Mở được từ điện thoại qua URL              |
| 6   | Tái lập                                  | Chạy lại từ README trên máy sạch ≤ 30 phút |

---

## 5. CHECKLIST CUỐI 3 THÁNG

| Hạng mục                     | Bằng chứng                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| [ ] Model chạy được          | `freeze_manifest.json`; kết quả 3 seed; `test_holdout_result.json`                    |
| [ ] API/backend chạy được    | FastAPI `/health`, `/predict`, `/predict_batch`; `parity_report.json`                 |
| [ ] Web/mobile app chạy được | URL demo mở được trên laptop + điện thoại                                             |
| [ ] AI tích hợp vào app      | Kết quả UI khớp offline (Mục 4, tiêu chí 1); cờ review hoạt động theo τ               |
| [ ] Có dữ liệu/test case     | `splits/` + SHA-256; 30 test case; 10 ảnh biển số VN định tính                        |
| [ ] Có demo end-to-end       | Demo live + `demo_backup.mp4`                                                         |
| [ ] Có đánh giá performance  | `eval_report.md`, `stats.json`, `ablation_table.csv`, `latency.csv`, `throughput.csv` |
| [ ] Có báo cáo kết quả       | Chương 3–5 thesis + báo cáo PDF, repo tag `v1.0`                                      |

---
