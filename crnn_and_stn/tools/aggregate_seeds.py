#!/usr/bin/env python3
"""Tổng hợp kết quả nhiều seed thành Mean ± Std + khoảng tin cậy 95%.

Phục vụ mục "Mean ± Std trên 3 independent random seeds" trong issue #9.
Quan trọng với project này vì validation set chỉ 999 sample — chênh lệch
0.2-0.5 điểm giữa các cấu hình nằm trong biên độ nhiễu, không thể kết luận
từ 1 lần chạy đơn lẻ.

Cách dùng — nhập trực tiếp accuracy đã đo:
    python tools/aggregate_seeds.py --acc 76.28 76.05 76.51 --label "SR + GroupNorm"

So sánh 2 cấu hình (kèm kiểm định xem chênh lệch có ý nghĩa không):
    python tools/aggregate_seeds.py \
      --acc 76.28 76.05 76.51 --label "SR + GroupNorm" \
      --baseline-acc 76.68 76.40 76.22 --baseline-label "ResBlock baseline"

Hoặc đọc tự động từ log training đã lưu:
    python tools/aggregate_seeds.py --from-logs results/log_seed*.txt

Xuất bảng tổng hợp ra CSV (yêu cầu "ghi nhận bảng kết quả 3 seeds vào CSV log"),
gom nhiều cấu hình vào cùng một file bằng --append:
    for C in j1p s1 s4; do
      python tools/aggregate_seeds.py --from-logs results/multi-seed/*/log_${C}_seed*.txt \
        --label "${C}" --output-csv results/multi_seed_summary.csv --append
    done
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys

CSV_FIELDS = [
    "label", "n_seeds", "accs", "mean", "std", "min", "max",
    "ci95", "ci_low", "ci_high",
    "delta_vs_baseline", "delta_stderr", "verdict",
]

BEST_ACC_PATTERN = re.compile(r"Best Val Acc:\s*([0-9.]+)%")


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample std
    return mean, math.sqrt(variance)


def wald_ci95(accuracy_pct: float, n_samples: int) -> float:
    """Nửa độ rộng khoảng tin cậy 95% cho tỷ lệ nhị phân (điểm phần trăm)."""
    p = accuracy_pct / 100.0
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n_samples) * 100.0


def summarize(values: list[float], label: str, n_val: int) -> dict[str, object]:
    mean, std = mean_std(values)
    ci = wald_ci95(mean, n_val)
    print(f"\n{label}")
    print(f"  Số seed      : {len(values)}  ({', '.join(f'{v:.2f}%' for v in values)})")
    print(f"  Mean ± Std   : {mean:.2f}% ± {std:.2f}")
    print(f"  Min / Max    : {min(values):.2f}% / {max(values):.2f}%")
    print(f"  CI 95% (n={n_val}) : ±{ci:.2f} điểm  → [{mean - ci:.2f}%, {mean + ci:.2f}%]")
    return {
        "label": label,
        "n_seeds": len(values),
        # Space-separated so the cell stays readable when the CSV is eyeballed.
        "accs": " ".join(f"{v:.4f}" for v in values),
        "mean": f"{mean:.4f}",
        "std": f"{std:.4f}",
        "min": f"{min(values):.4f}",
        "max": f"{max(values):.4f}",
        "ci95": f"{ci:.4f}",
        "ci_low": f"{mean - ci:.4f}",
        "ci_high": f"{mean + ci:.4f}",
        "delta_vs_baseline": "",
        "delta_stderr": "",
        "verdict": "",
    }


def write_csv(rows: list[dict[str, object]], path: str, append: bool) -> None:
    """Ghi bảng tổng hợp ra CSV. `append` để gom nhiều cấu hình vào 1 file."""
    # Header chỉ viết khi tạo file mới, nếu không mỗi lần --append lại chèn thêm
    # một dòng header vào giữa dữ liệu.
    existing = append and os.path.exists(path) and os.path.getsize(path) > 0
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a" if existing else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not existing:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n💾 {'Nối vào' if existing else 'Đã ghi'} {path} ({len(rows)} dòng)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("--acc", type=float, nargs="+", help="Accuracy (%%) của từng seed")
    parser.add_argument("--label", type=str, default="Cấu hình")
    parser.add_argument("--baseline-acc", type=float, nargs="+", default=None)
    parser.add_argument("--baseline-label", type=str, default="Baseline")
    parser.add_argument("--from-logs", type=str, nargs="+", default=None,
                        help="Đọc 'Best Val Acc: X%%' từ các file log")
    parser.add_argument("--val-size", type=int, default=999, help="Cỡ validation set")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Xuất bảng Mean ± Std ra file CSV")
    parser.add_argument("--append", action="store_true",
                        help="Nối vào --output-csv thay vì ghi đè, để gom nhiều "
                             "cấu hình vào cùng một file")
    args = parser.parse_args()
    if args.append and not args.output_csv:
        parser.error("--append chỉ dùng được kèm --output-csv")
    return args


def read_logs(patterns: list[str]) -> list[float]:
    values = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", errors="ignore") as handle:
                matches = BEST_ACC_PATTERN.findall(handle.read())
            if matches:
                values.append(float(matches[-1]))
                print(f"  {path}: {matches[-1]}%")
            else:
                print(f"  {path}: ⚠️ không tìm thấy 'Best Val Acc:'")
    return values


def main() -> None:
    args = parse_args()

    if args.from_logs:
        print("Đọc từ log:")
        values = read_logs(args.from_logs)
    else:
        values = args.acc or []

    if not values:
        print("❌ Không có dữ liệu. Dùng --acc hoặc --from-logs.")
        sys.exit(1)

    row = summarize(values, args.label, args.val_size)
    rows = [row]
    mean, std = mean_std(values)

    if args.baseline_acc:
        base_row = summarize(args.baseline_acc, args.baseline_label, args.val_size)
        rows.append(base_row)
        base_mean, base_std = mean_std(args.baseline_acc)
        delta = mean - base_mean
        # Sai số của hiệu 2 trung bình độc lập.
        pooled = math.sqrt(std**2 / max(len(values), 1) + base_std**2 / max(len(args.baseline_acc), 1))
        print(f"\n{'=' * 60}\nSO SÁNH\n{'=' * 60}")
        print(f"  Chênh lệch   : {delta:+.2f} điểm ({args.label} so với {args.baseline_label})")
        print(f"  Sai số hiệu  : ±{pooled:.2f}")
        if pooled == 0:
            verdict = "single_seed"
            print("  ⚠️ Chỉ 1 seed mỗi bên — không đánh giá được độ tin cậy.")
        elif abs(delta) > 2 * pooled:
            verdict = "significant"
            print(f"  ✅ Chênh lệch LỚN HƠN 2x sai số → nhiều khả năng là cải thiện thật.")
        else:
            verdict = "within_noise"
            print(f"  ⚠️ Chênh lệch NẰM TRONG biên độ nhiễu → chưa đủ bằng chứng kết luận.")
            print(f"     Cần thêm seed hoặc validation set lớn hơn.")
        # Kết quả so sánh gắn vào dòng của cấu hình đang xét; dòng baseline để trống.
        row["delta_vs_baseline"] = f"{delta:+.4f}"
        row["delta_stderr"] = f"{pooled:.4f}"
        row["verdict"] = verdict

    if args.output_csv:
        write_csv(rows, args.output_csv, args.append)


if __name__ == "__main__":
    main()
