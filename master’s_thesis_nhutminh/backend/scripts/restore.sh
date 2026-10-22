#!/bin/sh
# Khôi phục DB từ 1 file backup (chạy từ master’s_thesis_nhutminh/):
#
#   sh backend/scripts/restore.sh backups/platereader_20261022T020000Z.dump
#
# Dừng API trước để không có ghi mới trong lúc restore; --clean xoá bảng cũ rồi tạo lại.
set -eu

FILE="${1:?Cách dùng: restore.sh <file.dump>}"
[ -f "$FILE" ] || { echo "Không thấy $FILE" >&2; exit 1; }

docker compose stop api
docker compose exec -T db sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner' < "$FILE"
docker compose start api
echo "Đã khôi phục từ $FILE"
