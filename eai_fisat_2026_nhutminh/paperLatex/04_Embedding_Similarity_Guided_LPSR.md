# 04 — Embedding Similarity Guided License Plate Super Resolution

## Thông tin bài báo

| Trường | Nội dung |
|---|---|
| **Tiêu đề** | Embedding Similarity Guided License Plate Super Resolution |
| **Tác giả** | Abderrezzaq Sendjasni, Mohamed-Chaker Larabi |
| **Tạp chí** | Neurocomputing, Vol. 651, 2025, art. 130657 |
| **DOI** | https://doi.org/10.1016/j.neucom.2025.130657 |
| **arXiv** | [2501.01483](https://arxiv.org/abs/2501.01483) — nộp 02/01/2025, eess.IV; cs.CV |
| **ResearchGate** | https://www.researchgate.net/publication/387744986_Embedding_Similarity_Guided_License_Plate_Super_Resolution |
| **HAL** | https://hal.science/hal-04942266v1 |

> **Ghi chú nguồn:** ResearchGate trả HTTP 403 với truy cập tự động; nội dung lấy từ bản arXiv v2 của cùng bài.

---

## Abstract (nguyên văn)

> Super-resolution (SR) techniques play a pivotal role in enhancing the quality of low-resolution images, particularly for applications such as security and surveillance, where accurate license plate recognition is crucial. This study proposes a novel framework that combines pixel-based loss with embedding similarity learning to address the unique challenges of license plate super-resolution (LPSR). The introduced pixel and embedding consistency loss (PECL) integrates a Siamese network and applies contrastive loss to force embedding similarities to improve perceptual and structural fidelity. By effectively balancing pixel-wise accuracy with embedding-level consistency, the framework achieves superior alignment of fine-grained features between high-resolution (HR) and super-resolved (SR) license plates. Extensive experiments on the CCPD and PKU dataset validate the efficacy of the proposed framework, demonstrating consistent improvements over state-of-the-art methods in terms of PSNR, SSIM, LPIPS, and optical character recognition (OCR) accuracy. These results highlight the potential of embedding similarity learning to advance both perceptual quality and task-specific performance in extreme super-resolution scenarios.

## Tóm tắt Abstract (tiếng Việt)

Bài báo đề xuất khung SR cho biển số kết hợp **loss theo pixel** với **học tương đồng embedding**. Thành phần cốt lõi là **PECL (pixel and embedding consistency loss)**: tích hợp một **mạng Siamese** và áp dụng **contrastive loss** để ép độ tương đồng embedding, qua đó cải thiện độ trung thực về **cảm nhận thị giác** và **cấu trúc**. Nhờ cân bằng giữa độ chính xác pixel và nhất quán ở mức embedding, khung này căn chỉnh tốt hơn các đặc trưng chi tiết giữa ảnh HR và ảnh SR. Thực nghiệm trên **CCPD** và **PKU** cho cải thiện nhất quán so với SOTA về PSNR, SSIM, LPIPS và **độ chính xác OCR** — đặc biệt trong kịch bản siêu phân giải cực đoan (×8).

---

## Introduction — các ý chính

**Động cơ**
- SR thiết yếu cho giám sát/an ninh, nơi nhận dạng chính xác biển số là yêu cầu bắt buộc.
- Biển số thực tế bị **độ phân giải thấp, nhòe chuyển động, nhiễu**. Nguyên nhân chính: chụp từ **khoảng cách xa** → biển số nhỏ, số pixel giảm mạnh, khó đọc cho cả máy lẫn người.

**Hạn chế của công trình trước**
- Mô hình SISR (single image SR) tiêu chuẩn không xử lý tốt đặc thù biển số: **chữ nhỏ, nền phức tạp, ánh sáng thay đổi, font đa dạng**.
- Nội suy truyền thống (bilinear, bicubic) cho ảnh mờ, mất chi tiết.
- CNN/GAN tốt hơn nhưng nghiên cứu **LPSR chuyên biệt vẫn ít** so với SISR tổng quát.
- Đa số phương pháp chỉ nhắm hệ số phóng đại **vừa phải (×2, ×4)**, thất bại ở trường hợp cực đoan **×8** khi chi tiết đã bị hủy nghiêm trọng.

**Phương pháp đề xuất**
- Kiến trúc: **residual dense blocks** + **channel attention**.
- Hàm mất mát **PECL** = MSE ở mức pixel + tương đồng ở mức embedding qua mạng Siamese và contrastive loss.

**Đóng góp (nguyên văn)**
1. "We developed a deep learning framework for extreme LPSR with a scaling factor of ×8, leveraging residual dense blocks and channel attention mechanisms to enhance visual quality and recover fine details."
2. "We introduced the pixel and embedding consistency loss (PECL), which integrates pixel-level and embedding-level similarities. A Siamese network and contrastive loss are employed to align and constrain the similarity between HR and SR embeddings."
3. "We conducted a comprehensive evaluation of the proposed method on the CCPD dataset, demonstrating its robustness and effectiveness across diverse real-world conditions."

---

## Liên hệ với đề tài của mình

- **Ý tưởng dùng được ngay:** loss ở mức embedding (không chỉ pixel/CTC) để ép đặc trưng ảnh LR gần với đặc trưng ảnh HR. Với dataset LRLPR-26 có sẵn 5 ảnh HR mỗi track, hoàn toàn có thể thêm một nhánh **consistency loss LR↔HR** vào CRNN+STN mà không cần mô hình SR riêng.
- Bài này làm **×8** — gần với tỷ lệ suy giảm của LRLPR-26 (ảnh gốc ~46×19 px) → dẫn chứng tốt cho phần "extreme SR".
- Khác biệt cần nêu rõ khi so sánh: bài này dùng **CCPD/PKU (suy giảm tổng hợp)**, còn LRLPR-26 là **suy giảm thật** → không so trực tiếp con số được.

## BibTeX

```bibtex
@article{sendjasni2025embedding,
  title   = {Embedding similarity guided license plate super resolution},
  author  = {Sendjasni, Abderrezzaq and Larabi, Mohamed-Chaker},
  journal = {Neurocomputing},
  volume  = {651},
  pages   = {130657},
  year    = {2025},
  doi     = {10.1016/j.neucom.2025.130657}
}
```
