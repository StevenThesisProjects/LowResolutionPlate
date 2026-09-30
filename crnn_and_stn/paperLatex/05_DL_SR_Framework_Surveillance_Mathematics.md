# 05 — A Deep Learning Framework of Super Resolution for License Plate Recognition in Surveillance System

## Thông tin bài báo

| Trường | Nội dung |
|---|---|
| **Tiêu đề** | A Deep Learning Framework of Super Resolution for License Plate Recognition in Surveillance System |
| **Tác giả** | Pei-Fen Tsai, Jia-Yin Shiu, Shyan-Ming Yuan |
| **Đơn vị** | Institute of Computer Science and Engineering, Dept. of Electrical and Computer Engineering, National Yang Ming Chiao Tung University, Hsinchu, Taiwan |
| **Tạp chí** | Mathematics (MDPI), 2025, Vol. 13, No. 10, art. 1673 (28 trang) |
| **DOI** | https://doi.org/10.3390/math13101673 |
| **Mốc thời gian** | Nhận 14/04/2025 · Sửa 14/05/2025 · Chấp nhận 16/05/2025 · Đăng 20/05/2025 |
| **Từ khóa** | license plate recognition (LPR); super resolution (SR); perceptual loss; optical character recognition (OCR) |
| **ResearchGate** | https://www.researchgate.net/publication/391905917_A_Deep_Learning_Framework_of_Super_Resolution_for_License_Plate_Recognition_in_Surveillance_System |

> **Ghi chú nguồn:** ResearchGate và trang MDPI đều chặn truy cập tự động (HTTP 403); nội dung lấy trực tiếp từ file PDF open-access (CC BY) của MDPI.

---

## Abstract (nguyên văn)

> Recognizing low-resolution license plates from real-world scenes remains a challenging task. While deep learning-based super-resolution methods have been widely applied, most existing datasets rely on artificially degraded images, and common quality metrics poorly correlate with OCR accuracy. We construct a new paired low- and high-resolution license plate dataset from dashcam videos and propose a specialized super-resolution framework for license plate recognition. Only low-resolution images with OCR accuracy ≥5 are used to ensure sufficient feature information for effective perceptual learning. We analyze existing loss functions and introduce two novel perceptual losses—one CNN-based and one Transformer-based. Our approach improves recognition performance, achieving an average OCR accuracy of 85.14%.

## Tóm tắt Abstract (tiếng Việt)

Nhận dạng biển số độ phân giải thấp từ cảnh thực vẫn khó. Hai vấn đề của nghiên cứu hiện có: (1) hầu hết dataset dùng **ảnh suy giảm nhân tạo**; (2) các **chỉ số chất lượng ảnh thông dụng tương quan kém với độ chính xác OCR**. Nhóm tác giả xây dựng bộ dữ liệu cặp LR–HR mới **từ video dashcam thật** và đề xuất khung siêu phân giải chuyên biệt cho LPR. Chỉ dùng ảnh LR có **OCR accuracy ≥ 5** (đủ thông tin đặc trưng cho học perceptual). Phân tích các hàm mất mát hiện có và giới thiệu **hai perceptual loss mới** — một dựa trên CNN, một dựa trên Transformer. Kết quả: **OCR accuracy trung bình 85,14%**.

---

## Introduction — các ý chính

**Động cơ thực tế**
- Cảnh sát dùng LPR cho điều tra tội phạm: xác định xe gây tai nạn bỏ chạy (hit-and-run), truy vết xe nghi vấn.
- Độ chính xác LPR thường bị phá hỏng bởi **nhòe chuyển động** và **ảnh độ phân giải thấp** do xe di chuyển nhanh.
- SR đóng vai trò **bước tiền xử lý ảnh then chốt**, làm rõ ảnh biển số trước khi nhận dạng.
- Các yếu tố gây suy giảm được liệt kê: nhòe chuyển động, thiếu sáng/bóng đổ/quá sáng, độ phân giải camera thấp, góc nhìn xiên, thời tiết (mưa, sương mù, tuyết).
- Hai kịch bản ứng dụng (Figure 1): giám sát xe nghi vấn; tai nạn có tài xế bỏ chạy.

**Khoảng trống nghiên cứu (research gap)**
1. Dataset công khai cho LPR **chủ yếu là ảnh LR tổng hợp, thiếu tính thực tế** → hạn chế hiệu quả khi triển khai thật.
2. Đánh giá mô hình SR cho LPR **bị bó buộc vào PSNR và SSIM**, vốn "có thể không phản ánh đầy đủ tác động của việc tăng cường ảnh lên độ chính xác nhận dạng".

**Cách xây dựng dữ liệu (Figure 2)**
- Dùng video hộp đen (black box / dashcam) thực tế.
- Ảnh **HR (I^HR)** lấy từ đoạn quay **gần**; ảnh **LR (I^LR)** lấy từ đoạn quay **xa** → tạo cặp huấn luyện, nhãn ground truth là ký tự biển số.
- Chỉ giữ ảnh LR có OCR accuracy ≥ 5 để đảm bảo đủ thông tin đặc trưng.
- Biển số Trung Quốc chuẩn: 1 chữ cái mã tỉnh/thành + 6 ký tự alphanumeric = 7 ký tự.

**Pipeline (Figure 3)**
- Mô hình SR **SwinFIR** được huấn luyện với tổ hợp **MSE theo pixel + perceptual loss**, tối ưu qua backpropagation.
- Ảnh HR phục hồi được đưa vào **bộ nhận dạng OCR dựa trên CRNN đã pretrain**.

**Năm loại perceptual loss được khảo sát**
1. Multi-task MSE loss cho OCR
2. DISTS (Deep Image Structure and Texture Similarity) với VGG16
3. VGG loss với VGG19
4. Loss với **Swin Transformer**
5. Loss với **CRNN**
+ các mô hình **ensemble** kết hợp các loss trên.

**Kết quả chính trong Introduction**
- Chỉ dùng **một** perceptual loss: độ chính xác nhận dạng ký tự tăng từ **75,75% → 82,57%** (+7%).
- **Ensemble Swin Transformer + DISTS**: tăng tiếp lên **85,14%** (+9,75% so với gốc).

**Bốn đóng góp chính (nguyên văn rút gọn)**
1. Khung hệ thống giám sát với dataset mở rộng được, kèm SR biển số và phát hiện biển số.
2. Bộ dữ liệu cặp LR–HR mới từ cảnh lái xe thực tế, có thể chia sẻ cho nghiên cứu tiếp theo.
3. Phân tích chi tiết hiệu năng của các hàm mất mát trong bài toán SR nhắm tới cải thiện nhận dạng chữ trên biển số.
4. Chứng minh cải thiện đáng kể cả về trực quan hóa lẫn kết quả nhận dạng chữ nhờ kết hợp hiệu quả các hàm mất mát.

**Related Work — các dataset được so sánh:** LSV-LP, CCPD, SSIG-SegPlates, UFPR-ALPR, RodoSol.

---

## Liên hệ với đề tài của mình

- **Trùng lập luận cốt lõi với dự án**: "ảnh suy giảm tổng hợp không đại diện cho thực tế" + "PSNR/SSIM tương quan kém với OCR accuracy" → trích dẫn bài này để củng cố lý do chỉ báo cáo recognition rate.
- **Bộ nhận dạng của họ cũng là CRNN** → so sánh trực tiếp được với pipeline CRNN+STN hiện tại.
- **Hướng thử nghiệm cụ thể:** thêm perceptual loss dựa trên CRNN/Transformer vào huấn luyện. Lưu ý bài này áp perceptual loss cho **mô hình SR**, còn dự án hiện tại nhận dạng trực tiếp từ LR — muốn mượn ý tưởng thì phải chuyển thành **feature-consistency loss giữa nhánh LR và nhánh HR**, không copy nguyên xi được.
- Mức tăng +9,75% của họ đến **hoàn toàn từ thiết kế loss**, không đổi kiến trúc — đáng chú ý vì rẻ và dễ thử.

## BibTeX

```bibtex
@article{tsai2025deep,
  title   = {A Deep Learning Framework of Super Resolution for License Plate Recognition in Surveillance System},
  author  = {Tsai, Pei-Fen and Shiu, Jia-Yin and Yuan, Shyan-Ming},
  journal = {Mathematics},
  volume  = {13},
  number  = {10},
  pages   = {1673},
  year    = {2025},
  doi     = {10.3390/math13101673}
}
```
