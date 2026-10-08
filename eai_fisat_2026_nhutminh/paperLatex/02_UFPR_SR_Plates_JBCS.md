# 02 — Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark

## Thông tin bài báo

| Trường | Nội dung |
|---|---|
| **Tiêu đề** | Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark |
| **Tác giả** | Valfride Nascimento, Gabriel E. Lima, Rafael O. Ribeiro, William Robson Schwartz, Rayson Laroca, David Menotti |
| **Tạp chí** | Journal of the Brazilian Computer Society (JBCS), 2025 |
| **DOI** | https://doi.org/10.5753/jbcs.2025.5159 |
| **ResearchGate** | https://www.researchgate.net/publication/393042604_Toward_Advancing_License_Plate_Super-Resolution_in_Real-World_Scenarios_A_Dataset_and_Benchmark |
| **Trang dự án / dataset** | https://valfride.github.io/nascimento2025toward/ · https://github.com/valfride/UFPR-SR-Plates |

> **Phiên bản:** đây là bản tạp chí (JBCS) — bản chính thức nên dùng khi trích dẫn. Công trình này cũng có bản preprint arXiv [2505.06393](https://arxiv.org/abs/2505.06393) (nộp 09/05/2025) với nội dung khoa học trùng hoàn toàn, chỉ khác kênh công bố; dùng link arXiv nếu cần bản truy cập mở.
>
> **Ghi chú nguồn:** ResearchGate trả HTTP 403 với truy cập tự động; nội dung lấy từ bản arXiv/JBCS công khai của cùng bài.

---

## Abstract (nguyên văn)

> Recent advancements in super-resolution for License Plate Recognition (LPR) have sought to address challenges posed by low-resolution (LR) and degraded images in surveillance, traffic monitoring, and forensic applications. However, existing studies have relied on private datasets and simplistic degradation models. To address this gap, we introduce UFPR-SR-Plates, a novel dataset containing 10,000 tracks with 100,000 paired low and high-resolution license plate images captured under real-world conditions. We establish a benchmark using multiple sequential LR and high-resolution (HR) images per vehicle — five of each — and two state-of-the-art models for super-resolution of license plates. We also investigate three fusion strategies to evaluate how combining predictions from a leading Optical Character Recognition (OCR) model for multiple super-resolved license plates enhances overall performance. Our findings demonstrate that super-resolution significantly boosts LPR performance, with further improvements observed when applying majority vote-based fusion techniques. Specifically, the Layout-Aware and Character-Driven Network (LCDNet) model combined with the Majority Vote by Character Position (MVCP) strategy led to the highest recognition rates, increasing from 1.7% with low-resolution images to 31.1% with super-resolution, and up to 44.7% when combining OCR outputs from five super-resolved images. These findings underscore the critical role of super-resolution and temporal information in enhancing LPR accuracy under real-world, adverse conditions.

## Tóm tắt Abstract (tiếng Việt)

Các nghiên cứu siêu phân giải (SR) cho nhận dạng biển số gần đây đều vướng hai vấn đề: dùng **dataset riêng tư** và **mô hình suy giảm quá đơn giản**. Nhóm tác giả giới thiệu **UFPR-SR-Plates**: 10.000 track / 100.000 ảnh biển số cặp LR–HR chụp trong điều kiện thực tế. Benchmark dùng nhiều ảnh LR và HR liên tiếp cho mỗi xe (5 ảnh mỗi loại) với 2 mô hình SR SOTA, cùng 3 chiến lược **fusion** kết quả OCR từ nhiều ảnh SR. Kết quả: SR làm tăng mạnh hiệu năng LPR; **LCDNet + Majority Vote by Character Position (MVCP)** cho kết quả tốt nhất — từ **1,7%** (ảnh LR) lên **31,1%** (ảnh SR) và tới **44,7%** khi hợp nhất đầu ra OCR của 5 ảnh SR. Kết luận: SR và **thông tin thời gian (temporal)** đóng vai trò then chốt.

---

## Introduction — các ý chính

**Động cơ**
- LPR được triển khai rộng rãi cho giám sát giao thông, thu phí. Các báo cáo đạt độ chính xác cao đều đánh giá trên **ảnh HR, ký tự rõ ràng**.
- Thực tế: camera giám sát giá rẻ cho ảnh LR, bị nén, nhiễu → ký tự méo mó hoặc dính vào nhau, độ chính xác nhận dạng sụt nghiêm trọng.

**Khoảng trống nghiên cứu (research gap) — 3 điểm**
1. **Dataset riêng tư**: đa số nghiên cứu dùng dữ liệu độc quyền → không so sánh công bằng được giữa các phương pháp.
2. **Suy giảm đơn giản**: nhiều công trình tạo ảnh LR bằng downsampling cơ bản, không tái hiện được mẫu suy giảm thực tế.
3. **Chỉ số đánh giá không phù hợp**: chủ yếu dùng SSIM/PSNR, vốn "không tương quan tốt với cảm nhận thị giác của con người hay độ chính xác nhận dạng".

**Bộ dữ liệu giới thiệu — UFPR-SR-Plates**
- 100.000 ảnh biển số chụp bằng camera rolling shutter đặt trên đường ở Brazil, chia thành 10.000 track.
- Mỗi track: 5 ảnh LR liên tiếp + 5 ảnh HR liên tiếp của **cùng một biển số**.
- Điều kiện thực tế: ánh sáng và thời tiết thay đổi; cân bằng giữa layout Brazil và Mercosur; 2 mức độ phân giải video.

**Đóng góp**
- Bộ dữ liệu công khai 100.000 ảnh / 10.000 track.
- Thực nghiệm benchmark với các mô hình SR SOTA kèm chiến lược fusion.
- Khai thác **quan hệ thời gian** giữa các ảnh LR liên tiếp trong một track.

---

## Liên hệ với đề tài của mình

- Đây là **tiền thân trực tiếp** của bộ LRLPR-26 dùng trong cuộc thi ICPR 2026 (cùng nhóm tác giả, cùng cấu trúc track 5 LR + 5 HR) → trích dẫn ở phần Related Work / Dataset.
- **MVCP (Majority Vote by Character Position)** là kỹ thuật fusion theo vị trí ký tự — trực tiếp áp dụng được cho pipeline CRNN+STN hiện tại (hợp nhất 5 dự đoán LR trong cùng track). Đây là hướng cải tiến chi phí thấp đáng thử.
- Con số 1,7% → 31,1% → 44,7% là ví dụ tốt để lập luận rằng "khai thác nhiều frame trong track" cho lợi ích lớn hơn cải tiến kiến trúc đơn lẻ.

## BibTeX

```bibtex
@article{nascimento2025toward,
  title   = {Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark},
  author  = {Nascimento, Valfride and Lima, Gabriel E. and Ribeiro, Rafael O. and Schwartz, William Robson and Laroca, Rayson and Menotti, David},
  journal = {Journal of the Brazilian Computer Society},
  year    = {2025},
  doi     = {10.5753/jbcs.2025.5159}
}
```
