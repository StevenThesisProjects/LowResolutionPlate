# DELETE /predictions/{id}

Xoá 1 prediction. Toàn bộ lịch sử review của prediction đó cũng bị xoá theo (`ON DELETE CASCADE`).

```
DELETE /api/v1/predictions/{id}
```

## Request — Path

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID prediction |

## Response `204 No Content`

Không có body.

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `prediction_not_found` | Không có ID này |
| 422 | `validation_error` | ID không phải UUID |

## Ví dụ

```bash
curl -s -X DELETE -o /dev/null -w "%{http_code}\n" \
  http://localhost:8000/api/v1/predictions/3eeca7a5-6d41-497f-97ab-b13607bdb144
# 204
```
