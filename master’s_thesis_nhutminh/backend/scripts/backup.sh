#!/bin/sh
# Backup định kỳ PostgreSQL — chạy trong service `backup` của docker-compose.
#
#   BACKUP_INTERVAL_HOURS  (mặc định 6)  — chu kỳ pg_dump
#   BACKUP_RETENTION_DAYS  (mặc định 7)  — xoá bản cũ hơn N ngày
#   BACKUP_ONCE=1                        — chụp 1 bản rồi thoát (backup tay / cron host)
#
# Service `backup` nằm trong profile `backup` (mặc định tắt khi dev):
#   docker compose --profile backup up -d                   # chạy định kỳ (từ Tuần 10)
#   docker compose run --rm -e BACKUP_ONCE=1 backup         # 1 bản ngay lúc này
# Nếu chạy trên server cố định và muốn dùng cron của host thay cho vòng lặp:
#   0 */6 * * *  cd /path/master’s_thesis_nhutminh && docker compose run --rm -e BACKUP_ONCE=1 backup
#
# Mỗi lần tạo /backups/platereader_<UTC>.dump (định dạng custom -Fc, đã nén),
# ghi vào file .partial trước rồi mới đổi tên => không bao giờ để lại file hỏng
# mang tên hợp lệ. Khôi phục: xem backend/scripts/restore.sh.
set -eu

INTERVAL_HOURS="${BACKUP_INTERVAL_HOURS:-6}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
DIR=/backups
export PGPASSWORD="$POSTGRES_PASSWORD"

mkdir -p "$DIR"
if [ "${BACKUP_ONCE:-0}" = "1" ]; then echo "[backup] one-shot -> $DIR"; else echo "[backup] every ${INTERVAL_HOURS}h, keep ${RETENTION_DAYS} days -> $DIR"; fi

until pg_isready -h "$PGHOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q; do sleep 2; done

while true; do
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    OUT="$DIR/platereader_${STAMP}.dump"
    if pg_dump -h "$PGHOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -Z 6 -f "$OUT.partial"; then
        mv "$OUT.partial" "$OUT"
        echo "[backup] $(date -u +%FT%TZ) ok $(du -h "$OUT" | cut -f1) $OUT"
    else
        rm -f "$OUT.partial"
        echo "[backup] $(date -u +%FT%TZ) FAILED" >&2
    fi
    find "$DIR" -name 'platereader_*.dump' -mtime "+$RETENTION_DAYS" -print -delete
    [ "${BACKUP_ONCE:-0}" = "1" ] && exit 0
    sleep $((INTERVAL_HOURS * 3600))
done
