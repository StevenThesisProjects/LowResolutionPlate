# 03 — MF-LPR²: Multi-frame License Plate Image Restoration and Recognition using Optical Flow

## Thông tin bài báo

| Trường | Nội dung |
|---|---|
| **Tiêu đề** | MF-LPR²: Multi-frame license plate image restoration and recognition using optical flow |
| **Tác giả** | Kihyun Na, Junseok Oh, Youngkwan Cho, Bumjin Kim, Sungmin Cho, Jinyoung Choi, Injung Kim |
| **Tạp chí** | Computer Vision and Image Understanding (CVIU), Vol. 256, 2025, art. 104361 |
| **DOI** | https://doi.org/10.1016/j.cviu.2025.104361 |
| **ResearchGate** | https://www.researchgate.net/publication/390525999_MF-LPR_2_Multi-frame_license_plate_image_restoration_and_recognition_using_optical_flow |

> **Phiên bản:** đây là bản tạp chí CVIU — bản chính thức nên dùng khi trích dẫn. Công trình cũng có bản preprint arXiv [2508.14797](https://arxiv.org/abs/2508.14797) (nộp 19/08/2025); dùng link arXiv nếu cần bản truy cập mở.
>
> **Ghi chú nguồn:** ResearchGate trả HTTP 403 với truy cập tự động; nội dung lấy từ bản arXiv/CVIU công khai của cùng bài.
>
> ⚠️ **Abstract arXiv ≠ abstract CVIU.** Con số "best single-frame LPR" khác nhau giữa hai bản:
> - **arXiv 2508.14797:** `14.04%` (chính là dòng nguyên văn chép bên dưới)
> - **CVIU / ScienceDirect (bản of record):** `16.18%` — kiểm tra tại <https://www.sciencedirect.com/science/article/abs/pii/S1077314225000840>
>
> Table 3 của bản manuscript liệt kê WPOD-Net `13.08%` và AFA-Net `14.04%` (đều single-frame), nên `14.04%` trong bản arXiv nhiều khả năng là AFA-Net, và Elsevier đã sửa lại thành `16.18%` khi xuất bản.
>
> **Quy tắc dùng trong paper:** entry `na2025mflpr2` trong `refs.bib` trỏ tới DOI của CVIU, nên **phải dùng `16.18%`**, không dùng `14.04%`. Hai con số `86.44%` và `82.55%` giống nhau ở cả hai bản, dùng an toàn.

---

## Abstract (nguyên văn — bản arXiv 2508.14797)

> License plate recognition (LPR) is important for traffic law enforcement, crime investigation, and surveillance. However, license plate areas in dash cam images often suffer from low resolution, motion blur, and glare, which make accurate recognition challenging. Existing generative models that rely on pretrained priors cannot reliably restore such poor-quality images, frequently introducing severe artifacts and distortions. To address this issue, we propose a novel multi-frame license plate restoration and recognition framework, MF-LPR², which addresses ambiguities in poor-quality images by aligning and aggregating neighboring frames instead of relying on pretrained knowledge. To achieve accurate frame alignment, we employ a state-of-the-art optical flow estimator in conjunction with carefully designed algorithms that detect and correct erroneous optical flow estimations by leveraging the spatio-temporal consistency inherent in license plate image sequences. Our approach enhances both image quality and recognition accuracy while preserving the evidential content of the input images. In addition, we constructed a novel Realistic LPR (RLPR) dataset to evaluate MF-LPR². The RLPR dataset contains 200 pairs of low-quality license plate image sequences and high-quality pseudo ground-truth images, reflecting the complexities of real-world scenarios. In experiments, MF-LPR² outperformed eight recent restoration models in terms of PSNR, SSIM, and LPIPS by significant margins. In recognition, MF-LPR² achieved an accuracy of 86.44%, outperforming both the best single-frame LPR (14.04%) and the multi-frame LPR (82.55%) among the eleven baseline models. The results of ablation studies confirm that our filtering and refinement algorithms significantly contribute to these improvements.

## Tóm tắt Abstract (tiếng Việt)

Biển số trong ảnh dash cam thường bị **độ phân giải thấp, nhòe chuyển động và lóa sáng**. Các mô hình sinh (generative) dựa vào **prior đã pretrain** không phục hồi đáng tin cậy được ảnh chất lượng kém — thường tạo ra artifact và biến dạng nghiêm trọng (nguy hiểm khi ảnh dùng làm chứng cứ). Nhóm tác giả đề xuất **MF-LPR²**: khung phục hồi + nhận dạng **đa khung hình**, giải quyết mơ hồ bằng cách **căn chỉnh và tổng hợp các frame lân cận** thay vì dựa vào tri thức pretrain. Dùng optical flow estimator SOTA kèm các thuật toán **phát hiện và sửa optical flow sai** dựa trên tính nhất quán không–thời gian của chuỗi ảnh biển số. Xây dựng bộ **RLPR** (200 cặp chuỗi ảnh chất lượng thấp + pseudo ground-truth chất lượng cao). Kết quả: vượt 8 mô hình phục hồi gần đây về PSNR/SSIM/LPIPS; độ chính xác nhận dạng **86,44%**, so với **16,18%** (single-frame tốt nhất, theo bản CVIU — bản arXiv ghi 14,04%) và **82,55%** (multi-frame) trong 11 baseline.

---

## Introduction — các ý chính

**Động cơ**
- LPR thiết yếu cho xử phạt giao thông, điều tra tội phạm, giám sát. Ảnh dash cam: biển số rất nhỏ (thường khoảng **88×32 pixel**), kèm nhòe chuyển động và lóa.
- Yêu cầu đặc thù: phục hồi phải **tăng tính đọc được mà vẫn bảo toàn giá trị chứng cứ** của ảnh gốc — khác hẳn bài toán phục hồi ảnh tổng quát (nơi "đẹp mắt" là đủ).

**Hạn chế của công trình trước — 3 vấn đề**
1. Mức độ suy giảm vượt quá khả năng xử lý của các mô hình phục hồi hiện có.
2. Mô hình ưu tiên **chất lượng thị giác** hơn **bảo toàn nội dung** → sinh ký tự "ảo" (hallucination), phá giá trị chứng cứ.
3. Mô hình huấn luyện trên cặp ảnh **tổng hợp**, trong khi suy giảm thực tế khác biệt đáng kể.
- Ngoài ra, nghiên cứu LPR trước đây chủ yếu tập trung vào **khử nhòe chuyển động**, ít quan tâm độ phân giải thấp và chất lượng ảnh nói chung.

**Phương pháp đề xuất**
- Căn chỉnh + tổng hợp frame lân cận thay cho prior pretrain.
- Optical flow + thuật toán **lọc (filtering)** và **tinh chỉnh (refinement)** khai thác nhất quán không–thời gian.

**Đóng góp (nguyên văn rút gọn)**
1. Chỉ ra sự **kém hiệu quả của các phương pháp phục hồi ảnh thông thường** cho biển số, và đề xuất MF-LPR² tăng độ đọc được mà vẫn giữ nội dung chứng cứ nhờ dùng nhiều frame.
2. Thuật toán **lọc và tinh chỉnh optical flow** căn chỉnh chính xác nhiều frame chất lượng thấp bằng nhất quán không–thời gian.
3. Bộ dữ liệu **RLPR** để đánh giá phục hồi đa khung trong điều kiện thực tế.
4. Chỉ số mới **PDNF-k** (Top-k Percentile average of Distance to the Nearest Frame) — định lượng mức độ artifact giả tạo trong ảnh đã phục hồi.
5. Cải thiện đáng kể chất lượng ảnh và độ chính xác nhận dạng so với 11 baseline.

---

## Liên hệ với đề tài của mình

- **Rất sát** với thiết lập LRLPR-26: mỗi track có 5 ảnh LR → đây là bằng chứng mạnh rằng **hợp nhất đa khung** (chứ không chỉ SR đơn ảnh) là hướng cho lợi ích lớn nhất (14,04% → 86,44%).
- Cảnh báo về **hallucination** rất đáng dẫn trong phần thảo luận: mô hình SR/generative có thể "bịa" ký tự — lý do nên đánh giá bằng recognition rate trên ký tự thật thay vì chỉ số ảnh.
- Ý tưởng dùng optical flow để căn chỉnh 5 frame LR trước khi đưa vào CRNN là hướng mở rộng khả thi cho pipeline hiện tại (dù chi phí tính toán cao hơn MVCP ở file 02).
- Chỉ số **PDNF-k** có thể dùng nếu cần định lượng artifact khi so sánh các biến thể mô hình.

## BibTeX

```bibtex
@article{na2025mflpr2,
  title   = {{MF-LPR}$^2$: Multi-frame license plate image restoration and recognition using optical flow},
  author  = {Na, Kihyun and Oh, Junseok and Cho, Youngkwan and Kim, Bumjin and Cho, Sungmin and Choi, Jinyoung and Kim, Injung},
  journal = {Computer Vision and Image Understanding},
  volume  = {256},
  pages   = {104361},
  year    = {2025},
  doi     = {10.1016/j.cviu.2025.104361}
}
```
