# DELETE /batches/{id}

Xoá 1 batch cùng **toàn bộ** prediction và review thuộc batch đó (`ON DELETE CASCADE`).
Không xoá được batch đang chạy.

```
DELETE /api/v1/batches/{id}
```

## Request — Path

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | uuid | ID batch |

## Response `204 No Content`

Không có body.

## Lỗi

| HTTP | `code` | Khi nào |
| --- | --- | --- |
| 404 | `batch_not_found` | Không có ID này |
| 409 | `batch_in_progress` | Batch đang `queued` / `running` |

## Ví dụ

```bash
curl -s -X DELETE -o /dev/null -w "%{http_code}\n" \
  http://localhost:8000/api/v1/batches/3444f665-6ae0-4a31-8caf-c7a2526fd95b
# 204
```
