# GET /health

Kiểm tra service còn sống, model đã load và database kết nối được hay chưa.
Docker `HEALTHCHECK` và frontend (khi hiển thị banner "hệ thống đang bảo trì") đều gọi endpoint này.

```
GET /api/v1/health
```

## Request

Không có tham số.

## Response

| HTTP | Khi nào |
| --- | --- |
| `200 OK` | Model đã load **và** DB đang kết nối (`status = "ok"`) |
| `503 Service Unavailable` | Thiếu 1 trong 2 (`status = "degraded"`); body vẫn có cùng cấu trúc |

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `status` | `"ok"` \| `"degraded"` | Tổng hợp trạng thái |
| `model_loaded` | bool | Checkpoint đã load thành công |
| `database` | `"ok"` \| `"unavailable"` | Kết quả `SELECT 1` qua connection pool |
| `uptime_s` | number | Số giây từ khi process khởi động |
| `version` | string | Phiên bản API |
| `detail` | string \| null | Lý do khi `degraded`, ví dụ lỗi load checkpoint |

```json
{
  "status": "ok",
  "model_loaded": true,
  "database": "ok",
  "uptime_s": 59.6,
  "version": "1.0.0",
  "detail": null
}
```

Ví dụ khi thiếu checkpoint:

```json
{
  "status": "degraded",
  "model_loaded": false,
  "database": "ok",
  "uptime_s": 0.2,
  "version": "1.0.0",
  "detail": "FileNotFoundError: [Errno 2] No such file or directory: '/models/s1_seed42_best.pth'"
}
```

## Ví dụ

```bash
curl -s http://localhost:8000/api/v1/health
```
