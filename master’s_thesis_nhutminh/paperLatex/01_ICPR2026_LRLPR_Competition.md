# 01 — ICPR 2026 Competition on Low-Resolution License Plate Recognition

## Thông tin bài báo

| Trường | Nội dung |
|---|---|
| **Tiêu đề** | ICPR 2026 Competition on Low-Resolution License Plate Recognition |
| **Tác giả** | Rayson Laroca, Valfride Nascimento, Donggun Kim, Sanghyeok Chung, Subin Bae, Uihwan Seo, Seungsang Oh, Chi M. Phung, Minh G. Vo, Xingsong Ye, Yongkun Du, Yuchen Su, Zhineng Chen, Sunhee Heo, Hyangwoo Lee, Kihyun Na, Khanh V. Vu Nguyen, Sang T. Pham, Duc N. N. Phung, Trong P. Le, Vy N. Vo Tran, David Menotti |
| **Loại** | Competition report (ICPR 2026) |
| **Năm / Ngày** | 24/04/2026 |
| **arXiv** | [2604.22506](https://arxiv.org/abs/2604.22506) — cs.CV |
| **ResearchGate** | https://www.researchgate.net/publication/404217248_ICPR_2026_Competition_on_Low-Resolution_License_Plate_Recognition |
| **Website** | https://icpr26lrlpr.github.io/ · Codabench: https://www.codabench.org/competitions/12259/ |

> **Ghi chú nguồn:** ResearchGate chặn truy cập tự động (HTTP 403), nội dung dưới đây lấy từ bản arXiv chính thức của cùng bài báo.

---

## Abstract (nguyên văn, rút gọn theo bản arXiv)

> Low-Resolution License Plate Recognition (LRLPR) remains a challenging problem in real-world surveillance scenarios, where long capture distances, compression artifacts, and adverse imaging conditions can severely degrade license plate legibility. This paper presents the ICPR 2026 Competition on Low-Resolution License Plate Recognition — the first competition specifically dedicated to LRLPR using real low-quality data collected under operationally relevant conditions. The competition was based on the LRLPR-26 dataset, comprising 20,000 training tracks and 3,000 test tracks, where each training track contains five low-resolution and five high-resolution images of the same license plate. A total of 269 teams from 41 countries registered for the competition, and 99 teams submitted valid entries in the Blind Test Phase. The winning team achieved a Recognition Rate of 82.13%, and four teams surpassed the 80% mark. This paper presents the competition design, evaluation protocol, main results, and summaries of the methods adopted by the top-5 teams, along with discussions on current trends and promising directions for future research on LRLPR.

## Tóm tắt Abstract (tiếng Việt)

Nhận dạng biển số ở độ phân giải thấp (LRLPR) vẫn là bài toán khó trong giám sát thực tế: khoảng cách chụp xa, nén ảnh, điều kiện chụp bất lợi làm biển số gần như không đọc được. Bài báo trình bày cuộc thi ICPR 2026 — cuộc thi **đầu tiên** dành riêng cho LRLPR trên **dữ liệu suy giảm thật** (không phải suy giảm nhân tạo). Dùng bộ **LRLPR-26**: 20.000 track huấn luyện + 3.000 track kiểm thử, mỗi track huấn luyện có 5 ảnh LR và 5 ảnh HR của cùng một biển số. 269 đội từ 41 quốc gia đăng ký, 99 đội nộp bài hợp lệ ở Blind Test Phase. Đội vô địch đạt **Recognition Rate 82,13%**, 4 đội vượt mốc 80%. Bài báo mô tả thiết kế cuộc thi, giao thức đánh giá, kết quả chính, tóm tắt phương pháp của top-5 và thảo luận hướng nghiên cứu tương lai.

---

## Introduction — các ý chính

**Động cơ / bối cảnh**
- ALPR trên ảnh chất lượng cao gần như đã "bão hòa" nhờ các detector hiện đại (họ YOLO) — độ chính xác rất cao.
- Nhưng trong giám sát thực tế, biển số thường bị chụp ở **độ phân giải rất thấp** do hạn chế phần cứng hoặc khoảng cách xe–camera lớn, cộng thêm **artifact do nén video**.

**Khoảng trống nghiên cứu (research gap)**
- LRLPR bị nghiên cứu ít so với tầm quan trọng thực tế của nó.
- Phần lớn công trình hiện có dùng ảnh **suy giảm tổng hợp** (bicubic downsampling và các mô hình đơn giản) → không phản ánh được độ phức tạp của suy giảm thật.
- Trên dữ liệu chất lượng thấp thật, ngay cả phương pháp SOTA cũng chật vật vượt mức **50–60%** độ chính xác.

**Đóng góp của bài báo**
1. Benchmark quốc tế quy mô lớn **đầu tiên** cho LRLPR trên dữ liệu suy giảm thật; bộ LRLPR-26 là bộ dữ liệu LP cặp LR/HR thật công khai lớn nhất.
2. Giao thức đánh giá toàn diện: Recognition Rate + các chỉ số hiệu chỉnh độ tin cậy (confidence calibration).
3. Phân tích phương pháp của top-5 đội, cho thấy nhiều chiến lược thắng cuộc khác nhau.
4. Bằng chứng cho thấy LRLPR **chưa được giải quyết** dù đã có tiến bộ đáng kể.

**Mô tả dữ liệu LRLPR-26**
- Tổng ~200.000 ảnh.
- Train: 20.000 track × (5 ảnh LR + 5 ảnh HR).
- Test: 3.000 track × 5 ảnh LR.
- 2 layout biển số Brazil; 2 kịch bản thu thập:
  - **Scenario A**: ban ngày, điều kiện kiểm soát.
  - **Scenario B**: điều kiện đa dạng — mưa, ban đêm.

**Cấu trúc bài báo:** 1) Introduction — 2) Thiết kế cuộc thi (dữ liệu, đánh giá, các phase) — 3) Kết quả & phương pháp top-5 — 4) Thảo luận — 5) Kết luận & hướng tương lai.

---

## Liên hệ với đề tài của mình

- Đây chính là **bài mô tả cuộc thi và bộ dữ liệu** mà dự án CRNN+STN đang làm việc trên đó → **bắt buộc trích dẫn** ở phần Introduction/Dataset.
- Mốc so sánh: winner 82,13%, 4 đội >80%. Baseline nội bộ 77,00%, cấu hình S1 tốt nhất 79,78% (trên val 999 track Scenario-B) → nằm ở nhóm cận top, nên trình bày kèm cảnh báo về biên nhiễu ±13 track.
- Luận điểm "ảnh suy giảm tổng hợp ≠ suy giảm thật" trong Introduction là lập luận rất tốt để dùng lại khi bảo vệ thiết kế thực nghiệm.

## BibTeX

```bibtex
@article{laroca2026icpr,
  title   = {{ICPR} 2026 Competition on Low-Resolution License Plate Recognition},
  author  = {Laroca, Rayson and Nascimento, Valfride and Kim, Donggun and Chung, Sanghyeok and Bae, Subin and Seo, Uihwan and Oh, Seungsang and Phung, Chi M. and Vo, Minh G. and Ye, Xingsong and Du, Yongkun and Su, Yuchen and Chen, Zhineng and Heo, Sunhee and Lee, Hyangwoo and Na, Kihyun and Nguyen, Khanh V. Vu and Pham, Sang T. and Phung, Duc N. N. and Le, Trong P. and Tran, Vy N. Vo and Menotti, David},
  journal = {arXiv preprint arXiv:2604.22506},
  year    = {2026}
}
```
