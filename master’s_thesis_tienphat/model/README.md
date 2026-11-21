# model/: checkpoint mà app phục vụ

Backend nạp model từ thư mục này. Nó chỉ chứa một file:

| File | Byte | SHA-256 |
|---|---|---|
| `restran_lockbox_combo3_s42_ep30.pth` | 134.719.041 | `9754ebb1915eac8284c130cea479393c5695b25ae1587c1bd24950bc128a7bba` |

- **Đây là model gì:** công thức huấn luyện #50 (combo3), run lock-box seed 42, epoch 30, 31.077.676 tham số. Model được chọn theo quy tắc cố định, không chọn theo điểm lock-box. Chi tiết nằm ở `artifacts/serving_config.json`.
- **Nguồn:** bản chép nguyên từ `restran/results_new/restran_lockbox_combo3_s42_ep30.pth`. SHA-256 trùng giá trị trong `restran/results_new/freeze_manifest.json`, ghi ngày 24/10/2026, trước khi chấm lock-box.
- **Kiểm tra lúc chạy:** mỗi lần khởi động, backend tính lại SHA-256 của file. Không khớp thì `/health` trả 503 và app không đọc biển số. `GET /model` trả `"checkpoint": "model/restran_lockbox_combo3_s42_ep30.pth"` và `"sha256_verified": true`.
- **Không đưa lên git:** `.gitignore` bỏ qua `*.pth`. File nặng 135 MB, vượt giới hạn 100 MB mỗi file của GitHub, và được train trên dữ liệu LRLPR-26 có giấy phép.

## Lấy lại file (máy mới, hoặc sau khi clone)

Từ thư mục `masters_thesis_tienphat/`, khi repo `restran/` nằm cùng cấp:

```bash
python3 scripts/freeze_serving_model.py    # kiểm SHA-256 với manifest rồi chép vào model/
# hoặc chép tay:
cp ../restran/results_new/restran_lockbox_combo3_s42_ep30.pth model/
shasum -a 256 model/restran_lockbox_combo3_s42_ep30.pth   # phải ra 9754ebb1…7bba
```

Muốn dùng một file ở chỗ khác (vẫn phải đúng SHA-256): `MFPLATE_MODEL=<đường dẫn .pth> python3 -m uvicorn mfplate.api:app --port 8000`.

Code mạng (`src/models/restran.py`, `configs/`) vẫn được import từ repo `restran`, không chép sang đây.
