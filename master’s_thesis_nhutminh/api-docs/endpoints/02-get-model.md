# GET /model

Trả về metadata của model đang phục vụ: checkpoint kèm SHA-256, kiến trúc, số tham số, chế độ decode và ngưỡng τ.
Mục đích là để mọi kết quả đều truy được về đúng một checkpoint, đúng yêu cầu `freeze_manifest.json` ở Tuần 6.

```
GET /api/v1/model
```

## Request

Không có tham số.

## Response `200 OK`

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `name` | string | Tên hiển thị (`PLATEREADER_MODEL_NAME`) |
| `model_version` | string | `<tên checkpoint>@<sha256[:12]>`, trùng với `model_version` trong mỗi prediction |
| `checkpoint` | string | Tên file `.pth` |
| `checkpoint_sha256` | string | SHA-256 đầy đủ của file checkpoint |
| `parameters` | int | Tổng số tham số |
| `architecture` | object | Cấu hình suy ngược từ `state_dict` (`tools/eval_decode.infer_architecture`) |
| `device` | string | `cpu` hoặc `cuda` |
| `input_size` | [int, int] | [H, W] sau resize = `[32, 128]` |
| `max_frames` | int | Số frame tối đa mỗi track (5) |
| `default_decode` | `"constrained"` \| `"greedy"` | Decode mặc định khi request không truyền `decode` |
| `beam_width` | int | Độ rộng beam của constrained decode |
| `plate_layouts` | string[] | Layout biển số Brazil: `L` = chữ, `N` = số |
| `tau` | number | Ngưỡng τ hiện tại |

```json
{
  "name": "MF-SR-OCR (S1)",
  "model_version": "s1_seed42_best@a1053d892457",
  "checkpoint": "s1_seed42_best.pth",
  "checkpoint_sha256": "a1053d89245770b5af829cdcc502cac3c504d3d48a608bda780a439642476545",
  "parameters": 29577214,
  "architecture": {
    "use_stn": true, "use_sr": true, "use_dcn": true, "sr_multi_frame": true,
    "backbone_norm": "group", "use_se": true, "stn_pool": [4, 8],
    "sr_hidden_channels": 32, "sr_scale": 2, "sr_num_blocks": 4
  },
  "device": "cpu",
  "input_size": [32, 128],
  "max_frames": 5,
  "default_decode": "constrained",
  "beam_width": 16,
  "plate_layouts": ["LLLNLNN", "LLLNNNN"],
  "tau": 0.9
}
```

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 503 | `model_unavailable` | Checkpoint chưa load được |

## Ví dụ

```bash
curl -s http://localhost:8000/api/v1/model
```
