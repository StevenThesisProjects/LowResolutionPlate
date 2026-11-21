# Báo cáo test end-to-end (sinh tự động bởi `scripts/run_e2e.py`, 2026-11-20T21:31:24)

Server: `http://127.0.0.1:8000`, model `restran-recipe-s42-ep30-9754ebb1915e`, decoder mặc định `greedy`. Track lấy từ val 999 (model chưa từng train), không lấy từ lock-box.

**Kết quả: 30/30 case trả về 200 với đủ trường JSON.** Tiêu chí MVP số 2 yêu cầu 30/30 không crash.

Đọc đúng 16/20 track; 9 track bị gắn cờ cần kiểm tra, trong đó 4 track đọc sai.

| Case | Input | Track | Layout | Độ nét (Q1 = mờ nhất) | Nhãn | Đọc được | Đúng | Cần kiểm tra | ms | HTTP |
|---|---|---|---|---|---|---|---|---|---|---|
| T01 | 5 ảnh | track_13125 | Mercosur | Q1 | KRB6I42 | ARB6I42 | ✗ | có | 21.17 | 200 |
| T02 | zip track (lr + hr + annotations) | track_20825 | Mercosur | Q1 | HKW7F75 | MKW7F75 | ✗ | có | 23.3 | 200 |
| T03 | 5 ảnh | track_14539 | Mercosur | Q1 | BAT9D82 | BAT9D82 | ✓ | có | 21.25 | 200 |
| T04 | zip track (lr + hr + annotations) | track_16292 | Mercosur | Q2 | SEO8B02 | SEO8B02 | ✓ | không | 21.18 | 200 |
| T05 | 5 ảnh | track_17145 | Mercosur | Q2 | BCJ1E88 | BCJ1E88 | ✓ | không | 19.85 | 200 |
| T06 | zip track (lr + hr + annotations) | track_13296 | Mercosur | Q2 | MGA1B14 | MGA1B14 | ✓ | có | 20.71 | 200 |
| T07 | 5 ảnh | track_20672 | Mercosur | Q3 | TES3I89 | TES3I89 | ✓ | không | 20.93 | 200 |
| T08 | zip track (lr + hr + annotations) | track_20273 | Mercosur | Q3 | BCS8B36 | BCS8B36 | ✓ | không | 21.2 | 200 |
| T09 | 5 ảnh | track_19475 | Mercosur | Q3 | ASY6G57 | ASY6G57 | ✓ | không | 19.77 | 200 |
| T10 | zip track (lr + hr + annotations) | track_21931 | Mercosur | Q4 | RNC7A38 | RNC7A38 | ✓ | không | 21.07 | 200 |
| T11 | 5 ảnh | track_19240 | Mercosur | Q4 | OJZ3E82 | OJZ3E82 | ✓ | không | 20.62 | 200 |
| T12 | zip track (lr + hr + annotations) | track_14372 | Mercosur | Q4 | SFB6A67 | SFB6A67 | ✓ | không | 21.49 | 200 |
| T13 | 5 ảnh | track_10538 | Brazilian | Q1 | EBR0547 | EBR0547 | ✓ | không | 21.19 | 200 |
| T14 | zip track (lr + hr + annotations) | track_12486 | Brazilian | Q1 | APO5432 | ARQ5432 | ✗ | có | 21.21 | 200 |
| T15 | 5 ảnh | track_12233 | Brazilian | Q2 | BAL9903 | BAL9903 | ✓ | có | 20.13 | 200 |
| T16 | zip track (lr + hr + annotations) | track_10687 | Brazilian | Q2 | AWB4811 | AWB4811 | ✓ | có | 20.17 | 200 |
| T17 | 5 ảnh | track_11539 | Brazilian | Q3 | BCO5163 | BCE9163 | ✗ | có | 21.05 | 200 |
| T18 | zip track (lr + hr + annotations) | track_11335 | Brazilian | Q3 | AKE6476 | AKE6476 | ✓ | có | 21.43 | 200 |
| T19 | 5 ảnh | track_11194 | Brazilian | Q4 | ABQ1928 | ABQ1928 | ✓ | không | 20.9 | 200 |
| T20 | zip track (lr + hr + annotations) | track_11702 | Brazilian | Q4 | AWP8930 | AWP8930 | ✓ | không | 20.88 | 200 |
| E01 | 1 frame | track_13125 |  |  | KRB6I42 | ARB6I42 | ✗ | có | 11.11 | 200 |
| E02 | 2 frame | track_13125 |  |  | KRB6I42 | ARB6I42 | ✗ | có | 13.33 | 200 |
| E03 | 3 frame | track_13125 |  |  | KRB6I42 | ARB6I42 | ✗ | không | 15.45 | 200 |
| E04 | 4 frame | track_13125 |  |  | KRB6I42 | ARB6I42 | ✗ | không | 15.73 | 200 |
| E05 | ảnh xám 1 kênh (PNG) | track_13125 |  |  | KRB6I42 | ARB6I42 | ✗ | có | 22.3 | 200 |
| E06 | ảnh xoay 90° | track_13125 |  |  | KRB6I42 | SEF009 | ✗ | có | 24.39 | 200 |
| E07 | ảnh lộn ngược 180° | track_13125 |  |  | KRB6I42 | CNT9G89 | ✗ | có | 26.28 | 200 |
| E08 | không phải biển số: nhiễu ngẫu nhiên | track_13125 |  |  | KRB6I42 | ASC0292 | ✗ | có | 12.13 | 200 |
| E09 | không phải biển số: thẻ trắng có viền | track_13125 |  |  | KRB6I42 | AVK725 | ✗ | có | 11.62 | 200 |
| E10 | cùng 1 frame lặp 5 lần | track_13125 |  |  | KRB6I42 | ARB6I42 | ✗ | có | 22.24 | 200 |

Ghi chú các case biên:

- E01 (1 frame): ARB6I42 — low_confidence; cảnh báo: Chỉ có 1 frame; model được thiết kế cho 5 frame liên tiếp của cùng một xe. Trên lock-box, accuracy trung bình với 1 frame là 55.6% (5 frame: 79.1%).
- E02 (2 frame): ARB6I42 — low_confidence; cảnh báo: Chỉ có 2 frame; model được thiết kế cho 5 frame liên tiếp của cùng một xe. Trên lock-box, accuracy trung bình với 2 frame là 69.0% (5 frame: 79.1%).
- E03 (3 frame): ARB6I42 — không gắn cờ; cảnh báo: Chỉ có 3 frame; model được thiết kế cho 5 frame liên tiếp của cùng một xe. Trên lock-box, accuracy trung bình với 3 frame là 74.1% (5 frame: 79.1%).
- E04 (4 frame): ARB6I42 — không gắn cờ; cảnh báo: Chỉ có 4 frame; model được thiết kế cho 5 frame liên tiếp của cùng một xe. Trên lock-box, accuracy trung bình với 4 frame là 77.0% (5 frame: 79.1%).
- E05 (ảnh xám 1 kênh (PNG)): ARB6I42 — low_confidence
- E06 (ảnh xoay 90°): SEF009 — low_confidence, invalid_layout; cảnh báo: r0.png: tỉ lệ rộng/cao 0.4, không giống ảnh crop biển số 1 dòng (thường khoảng 2–3).
- E07 (ảnh lộn ngược 180°): CNT9G89 — low_confidence
- E08 (không phải biển số: nhiễu ngẫu nhiên): ASC0292 — low_confidence; cảnh báo: Chỉ có 1 frame; model được thiết kế cho 5 frame liên tiếp của cùng một xe. Trên lock-box, accuracy trung bình với 1 frame là 55.6% (5 frame: 79.1%).
- E09 (không phải biển số: thẻ trắng có viền): AVK725 — low_confidence, invalid_layout; cảnh báo: Chỉ có 1 frame; model được thiết kế cho 5 frame liên tiếp của cùng một xe. Trên lock-box, accuracy trung bình với 1 frame là 55.6% (5 frame: 79.1%).
- E10 (cùng 1 frame lặp 5 lần): ARB6I42 — low_confidence; cảnh báo: Có frame trùng nhau hoàn toàn; trọng số fusion sẽ bị lệch về frame đó.
