# Audit lỗi cú pháp (sinh tự động bởi `scripts/syntax_audit.py`)

Model: cấu hình chốt #50, 4 run lock-box (seed 42, 43, 7, 2026), epoch cuối, decoder greedy. Tập: val 999 (`splits/val_tracks.json`), các model này chưa từng train trên tập này.

| Run | Accuracy | Số lỗi | Sai độ dài | 7 ký tự nhưng sai mẫu | Đúng mẫu của layout kia | Nhầm ký tự trong đúng layout | p(GT) > p(dự đoán) |
|---|---|---|---|---|---|---|---|
| combo3_s42 | 76.68 | 233 | 1 | 0 | 0 | 232 | 0 |
| combo3_s43 | 79.08 | 209 | 0 | 0 | 3 | 206 | 0 |
| combo3_s7 | 77.98 | 220 | 1 | 0 | 1 | 218 | 0 |
| combo3_s2026 | 76.88 | 231 | 10 | 0 | 1 | 220 | 0 |

- Gộp 4 run: 893 lỗi; **12 lỗi vi phạm cú pháp (1.3%)**. Đây là trần lợi ích của decode có ràng buộc cú pháp (SC-CTC).
- Ở 0 lỗi (0.0%), chuỗi đúng có xác suất CTC cao hơn chuỗi greedy chọn. Chỉ những lỗi này mới có thể sửa bằng bất kỳ cách decode nào.
- 58.7% lỗi chỉ sai đúng 1 ký tự. Phân bố số ký tự sai (lỗi cùng độ dài): {'1': 524, '2': 180, '3': 103, '4': 31, '5': 31, '6': 8, '7': 4}.
- Vị trí ký tự sai (1–7): {'1': 154, '2': 308, '3': 377, '4': 198, '5': 157, '6': 169, '7': 185}.

## Top-10 cặp nhầm (gộp 4 run, không phân biệt chiều)

| Cặp | Số lần |
|---|---|
| 6↔8 | 89 |
| H↔M | 50 |
| 4↔6 | 47 |
| O↔Q | 46 |
| 5↔6 | 44 |
| H↔N | 34 |
| V↔Y | 34 |
| D↔O | 33 |
| 5↔9 | 33 |
| A↔B | 29 |

## Top-10 nhầm theo layout: Mercosur (đúng → đọc thành)

| Nhầm | Số lần |
|---|---|
| 1→3 | 26 |
| 8→6 | 24 |
| 4→6 | 23 |
| H→M | 22 |
| M→H | 22 |
| N→H | 21 |
| 6→8 | 20 |
| D→B | 14 |
| 2→3 | 14 |
| E→C | 14 |

## Top-10 nhầm theo layout: Brazilian (đúng → đọc thành)

| Nhầm | Số lần |
|---|---|
| 6→8 | 30 |
| 6→5 | 22 |
| 5→6 | 21 |
| 5→9 | 20 |
| O→Q | 19 |
| D→O | 17 |
| M→N | 17 |
| 1→7 | 15 |
| 8→6 | 15 |
| Q→O | 14 |
