# Super Resolution cho CRNN + STN — Lịch sử nghiên cứu & bằng chứng root cause

> Tài liệu lịch sử các hướng SR đã thử **trước khi** phát hiện root cause thật (issue #9) và triển khai `FrameSR` thật trong [groupnorm_sr_ablation_j1_j2.md](../../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md). Đọc file đó trước nếu chỉ cần biết trạng thái SR hiện tại — file này chỉ để tra cứu lại vì sao các hướng cũ thất bại.

## 0. Mốc kết quả SR đã đo

| Cấu hình | Exact Match | Ghi chú |
|---|---:|---|
| SR stacked-input v1 | 49.25% | Gộp 5 frame → 1 frame SR rồi nhân bản — bug thiết kế |
| SR stacked-input v2 | 55.06% | Vẫn kém hẳn baseline |
| `LightEdgeSR` (mục 4 cũ, edge-preserving per-frame) | **Chưa từng chạy — code không tồn tại** | Xem cảnh báo ở mục 4 |
| `FrameSR` per-frame + giám sát pixel-level (triển khai thật) | 77.18% (J2) | Xem [groupnorm_sr_ablation_j1_j2.md](../../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md) |

Baseline CRNN+STN/ResBlock để đối chiếu: xem bảng đầy đủ tại [../training_runs/run_gpu.md](../../../../report/training_runs/run_gpu.md) — không lặp lại ở đây để tránh 2 nguồn số liệu.

**Kết luận cốt lõi từ lịch sử này**: SR stacked-input (gộp 5 frame → 1 → nhân bản) làm giảm accuracy rất mạnh — nguyên nhân là phá vỡ đa dạng frame mà `AttentionFusion` cần khai thác. Đây chính là **Root Cause #1** được xác định lại và sửa đúng trong đợt fix issue #9 (`FrameSR` chạy per-frame, không gộp).

---

## 1. Bốn hướng SR theo report (chưa có số liệu thực nghiệm trong report gốc)

Report đặt 4 hướng làm "Future of Work" (slide 52-83), thuần lý thuyết — project này là nơi kiểm chứng thực tế:

| Hướng | Slide | Ghi chú áp dụng |
|---|---|---|
| Backbone ResBlock | 53, 56-59 | → đã triển khai thật, xem [resblock_backbone_upgrade.md](../../../../report/baseline1_crnn_stn/resblock_backbone_upgrade.md) |
| Channel Attention | 54, 60-66 | Chưa thử riêng — cần kiểm chứng có tăng OCR accuracy thật hay chỉ tăng chất lượng hiển thị |
| PixelShuffle | 55, 67-76 | Dùng trong `FrameSR` hiện tại (mục upsample) |
| Stacked Inputs | 82 | Đã thử bản gộp-thô (mục 3 dưới) — thất bại vì lỗi thiết kế, không phải vì ý tưởng sai |

## 2. Kế hoạch cải thiện OCR (lịch sử, phần lớn đã thực hiện ở nhánh J1/J2)

Bối cảnh: người dùng yêu cầu tìm hướng SR cải thiện OCR biển số ngoài 4 hướng report, khi mốc đo được lúc đó là CRNN ~75%, ResBlock stable 76.68%, stacked-input SR 49-55%.

**4 thuộc tính SR ưu tiên cho biển số** (vẫn còn giá trị làm nguyên tắc thiết kế):
1. Giữ biên ký tự sắc nét — không làm tròn cạnh hay tạo texture giả.
2. Ít hallucination — SR không được "vẽ thêm" chi tiết không có thật.
3. Tối ưu cho text/plate domain — ưu tiên loss thiên về edge/text preservation hơn MSE thuần.
4. Upscale vừa đủ — 2x an toàn hơn 4x cho biển số.

**Kết luận đã áp dụng thật vào `FrameSR`** (xem J1/J2): SR nhẹ + per-frame + edge-aware loss (Sobel) + scale 2x — đúng thứ tự ưu tiên đề ra ở đây (SR nhẹ → edge-aware loss → attention → multi-frame chọn lọc), dừng ở bước 2 (chưa làm attention/multi-frame chọn lọc, xem N2/N3 trong `run_gpu.md`).

## 3. SR Stacked-Input — vì sao thất bại (bằng chứng root cause #1)

Bản triển khai đầu tiên (`StackedSRNet`) đặt SR **trước** STN theo đề xuất ban đầu, nhưng thiết kế sai: **gộp cả 5 frame thành 1 frame SR duy nhất rồi nhân bản ra 5 lần** trước khi vào STN — không phải "SR theo từng frame" như ý tưởng gốc.

**Hậu quả**: STN vốn cần warp affine riêng cho từng frame (góc lệch, rung camera khác nhau), nhưng 5 frame sau bước gộp-nhân-bản đã giống hệt nhau — STN mất hết thông tin để căn chỉnh độc lập, `AttentionFusion` cũng không còn gì khác biệt để chọn lọc giữa các frame.

**Kết quả đo được**: v1 = 49.25%, v2 = 55.06% — thấp hơn nhiều baseline (76.68%), xác nhận đúng giả thuyết lỗi thiết kế. Đây là bằng chứng thực nghiệm trực tiếp cho **Root Cause #1** trong phân tích issue #9.

## 4. ⚠️ `LightEdgeSR` — ĐÃ TỪNG GHI NHẬN NHẦM LÀ "ĐÃ TRIỂN KHAI", THỰC TẾ KHÔNG TỒN TẠI

Một phiên làm việc trước đã viết tài liệu mô tả module `LightEdgeSR` (edge-preserving, per-frame, đặt sau STN, upsample bilinear → Sobel edge → refine → downsample về size gốc) như **đã cài đặt trong code**, kèm lệnh chạy G/H trong `run_gpu.md` cũ.

**Xác minh lại (trong đợt fix issue #9)**: grep toàn bộ các branch của repo — class `LightEdgeSR` **không tồn tại ở bất kỳ đâu**. Tài liệu mô tả trước đó là kế hoạch/thiết kế chưa từng được viết thành code, không phải kết quả đã đo. Số liệu G/H không tồn tại vì lệnh đó chưa từng thực sự chạy được (flag `--sr-scale` có tồn tại nhưng trỏ tới `StackedSRNet` lỗi ở mục 3, không phải `LightEdgeSR`).

**Bài học quy trình**: từ nay, mọi tài liệu ghi "đã triển khai" phải kèm bằng chứng kiểm chứng được (đường dẫn file, kết quả log thật) — không suy diễn từ ý định thiết kế.

**Hướng thay thế đã triển khai thật**: `FrameSR` trong `src/models/components.py` — thiết kế tương tự về ý tưởng (per-frame, sau STN, PixelShuffle) nhưng là code mới viết và đã chạy thật, kết quả J2 = 77.18%. Xem [groupnorm_sr_ablation_j1_j2.md](../../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md).

## 5. Kết luận

- SR stacked-input (gộp-nhân-bản) đã thử và thất bại rõ ràng (49-55%) — bằng chứng thực cho Root Cause #1.
- Thiết kế `LightEdgeSR` mô tả ở tài liệu cũ **chưa từng được viết thành code** — đã sửa lại nhận định trong file này.
- `FrameSR` (triển khai thật, per-frame + giám sát pixel-level) đạt 77.18% (J2) — xem chi tiết và caveat thống kê tại [groupnorm_sr_ablation_j1_j2.md](../../baseline1_crnn_stn/groupnorm_sr_ablation_j1_j2.md).
- Nguyên tắc xuyên suốt vẫn đúng: đánh giá SR bằng OCR exact-match, không phải PSNR/SSIM; ưu tiên "đọc được biển số" hơn "đẹp ảnh".
