## Phân loại PR:

- [x] Feature
- [ ] Bugs
- [ ] Hotfix

---

# Đối chiếu tính năng đã làm theo yêu cầu:

- [x] **#1**: Implement pipeline Multi-Frame CRNN + STN (Baseline 1)
- [x] **#2**: Hỗ trợ ablation study CRNN vs CRNN+STN
- [x] **#4**: Training loop: CTC Loss + AdamW + OneCycleLR + AMP

---

# Cách thực hiện để xử lý mỗi yêu cầu:

- **#1**: Tạo `MultiFrameCRNN` (crnn.py) kết hợp `STNBlock`, `CNNBackbone`, `AttentionFusion` (components.py)
- **#2**: Thêm flag `--no-stn` vào `train.py`; `run_ablation.py` tự chạy 2 experiments tuần tự
- **#4**: Class `Trainer` (trainer.py) với CTC decode + confidence score (postprocess.py)

---

# Phạm vi ảnh hưởng:

- Toàn bộ code nằm trong module mới `crnn_and_stn/`, **không ảnh hưởng** các module khác trong repo
- Thay đổi hyperparameter trong `configs/config.py` sẽ ảnh hưởng tất cả experiments

---

# Liên kết đến các issues liên quan:

- Trích xuất từ: `MultiFrame-LPR-main/` (cùng repo), chỉ giữ Baseline 1, bỏ ResTran
- Branch: `dev/nhutminh` → merge target: `develop`
- Commit: `76c5dff` — `featfeat: baseline crnn combine stn`
