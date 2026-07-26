#!/usr/bin/env python3
"""Xuất 3 biểu đồ chính cho báo cáo, từ dữ liệu training đã lưu.

1. training_curves.png — Val Exact Match + Loss theo epoch (so nhiều run)
2. ablation_comparison.png — So sánh accuracy giữa các cấu hình (kèm Mean±Std)
3. accuracy_vs_cost.png — Đánh đổi accuracy với chi phí tính toán

Yêu cầu: pip install matplotlib

Ví dụ:
    # Chart 1 — đọc history CSV do trainer tự sinh ra trong results/
    python tools/plot_results.py curves \\
      --history results/history_crnn_resblock_groupnorm_nosr.csv \\
                results/history_crnn_resblock_sr_supervised.csv \\
      --labels "GroupNorm (không SR)" "SR + giám sát"

    # Chart 2 — nhập accuracy đã đo (thêm ± nếu có nhiều seed)
    python tools/plot_results.py ablation \\
      --names "Baseline ResBlock" "+ GroupNorm" "+ SR" "+ SR + DCNv2" \\
      --acc 76.68 76.90 77.20 77.05 \\
      --std 0 0.21 0.18 0.25 \\
      --baseline 76.68

    # Chart 3 — dùng số từ tools/benchmark.py --all
    python tools/plot_results.py cost \\
      --names "Baseline" "+ GroupNorm" "+ SR" "+ SR + DCNv2" \\
      --acc 76.68 76.90 77.20 77.05 \\
      --latency 53.35 52.84 195.04 201.60 \\
      --params 29298220 29313452 29426895 29445854
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("❌ Cần matplotlib để vẽ biểu đồ:  pip install matplotlib")
    sys.exit(1)

# Bảng màu đã qua kiểm định CVD (xem dataviz/references/palette.md).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DEEMPH = "#c9c8c1"
GOOD = "#006300"


def style_axes(ax, xlabel: str = "", ylabel: str = "") -> None:
    """Trục/lưới phải lùi về sau, để dữ liệu nổi lên trước."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10, labelpad=8)


def new_figure(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE, dpi=160)
    return fig


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✅ Đã lưu: {path}")


EPOCH_LINE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*Train Loss:\s*([\d.naif-]+)\s*\|\s*"
    r"Val Loss:\s*([\d.naif-]+)\s*\|\s*Val Acc:\s*([\d.]+)%\s*\|\s*LR:\s*([\d.e+-]+)",
    re.IGNORECASE,
)


def read_log(path: str) -> dict[str, list[float]]:
    """Dựng lại history từ stdout của training.

    Dùng khi run được thực hiện bằng bản code chưa có phần ghi history CSV —
    log text thì lúc nào cũng có, nên không mất dữ liệu đường cong.
    """
    cols: dict[str, list[float]] = {k: [] for k in
                                    ("epoch", "train_loss", "val_loss", "val_acc", "lr")}
    with open(path, errors="ignore") as handle:
        for line in handle:
            match = EPOCH_LINE.search(line)
            if not match:
                continue
            epoch, train_loss, val_loss, val_acc, lr = match.groups()

            def num(text: str) -> float:
                try:
                    return float(text)
                except ValueError:
                    return float("nan")

            cols["epoch"].append(float(epoch))
            cols["train_loss"].append(num(train_loss))
            cols["val_loss"].append(num(val_loss))
            cols["val_acc"].append(float(val_acc))
            cols["lr"].append(num(lr))
    return cols


def read_history(path: str) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    cols.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    pass
    return cols


def _place_end_labels(ax, items, min_gap_frac: float = 0.085) -> None:
    """Đặt nhãn ở cuối mỗi đường, đẩy giãn theo trục y để không đè nhau.

    items: list các tuple (y_value, x_value, label, color).
    """
    if not ax.get_ylim():
        return
    lo, hi = ax.get_ylim()
    span = max(hi - lo, 1e-9)
    # Sắp giảm dần rồi đẩy xuống dần: giữ đúng thứ tự trên/dưới của các đường.
    ordered = sorted(items, key=lambda item: item[0], reverse=True)
    placed: list[float] = []
    for y_value, _, _, _ in ordered:
        frac = (y_value - lo) / span
        if placed and placed[-1] - frac < min_gap_frac:
            frac = placed[-1] - min_gap_frac
        placed.append(frac)

    for (y_value, x_value, label, color), frac in zip(ordered, placed):
        ax.annotate(
            label,
            xy=(x_value, y_value), xycoords="data",
            xytext=(14, lo + frac * span), textcoords=("offset points", "data"),
            color=INK_2, fontsize=9, va="center", ha="left", zorder=6,
            arrowprops=dict(arrowstyle="-", color=color, linewidth=1.0,
                            shrinkA=2, shrinkB=1, alpha=0.55),
        )


# --------------------------------------------------------------------------
# Chart 1: đường cong training
# --------------------------------------------------------------------------
def plot_curves(args: argparse.Namespace) -> None:
    sources = [(p, False) for p in (args.history or [])] + [(p, True) for p in (args.from_log or [])]
    runs = []
    for idx, (path, is_log) in enumerate(sources):
        if not os.path.exists(path):
            print(f"⚠️ Bỏ qua (không tồn tại): {path}")
            continue
        data = read_log(path) if is_log else read_history(path)
        if not data.get("epoch"):
            print(f"⚠️ Bỏ qua (rỗng/sai format): {path}")
            continue
        label = args.labels[idx] if args.labels and idx < len(args.labels) else \
            os.path.basename(path).replace("history_", "").rsplit(".", 1)[0]
        runs.append((label, data))

    if not runs:
        print("❌ Không đọc được dữ liệu nào.\n"
              "   --history : results/history_*.csv (trainer tự sinh)\n"
              "   --from-log: file log stdout của training")
        sys.exit(1)

    # Accuracy và loss khác đơn vị hoàn toàn -> hai panel riêng, KHÔNG dùng 2 trục y
    # chung một khung (dual-axis khiến người đọc tự bịa ra tương quan không có thật).
    fig = new_figure(9.5, 7.4)
    ax_acc = fig.add_subplot(2, 1, 1)
    ax_loss = fig.add_subplot(2, 1, 2, sharex=ax_acc)

    peaks = [max(data["val_acc"]) for _, data in runs]
    winner = peaks.index(max(peaks))
    end_labels = []

    for idx, (label, data) in enumerate(runs):
        color = SERIES[idx % len(SERIES)]
        epochs = data["epoch"]

        ax_acc.plot(epochs, data["val_acc"], color=color, linewidth=2.0, zorder=3)
        end_labels.append((data["val_acc"][-1], epochs[-1], label, color))

        best_idx = max(range(len(data["val_acc"])), key=lambda i: data["val_acc"][i])
        ax_acc.scatter([epochs[best_idx]], [data["val_acc"][best_idx]],
                       s=42, color=color, edgecolor=SURFACE, linewidth=1.6, zorder=5)
        # Chỉ ghi con số cho run tốt nhất. Ghi mọi đỉnh sẽ tạo một đống số chồng
        # nhau ở vùng các đường hội tụ, và làm loãng đúng con số cần nhớ.
        if idx == winner:
            ax_acc.annotate(
                f"{data['val_acc'][best_idx]:.2f}%",
                xy=(epochs[best_idx], data["val_acc"][best_idx]),
                xytext=(0, 11), textcoords="offset points",
                color=INK, fontsize=10, fontweight="bold", ha="center", zorder=6,
            )

        ax_loss.plot(epochs, data["train_loss"], color=color, linewidth=2.0, zorder=3)
        ax_loss.plot(epochs, data["val_loss"], color=color, linewidth=1.6,
                     linestyle=(0, (4, 2)), alpha=0.85, zorder=3)

    if args.baseline is not None:
        ax_acc.axhline(args.baseline, color=MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
        ax_acc.annotate(
            f"Baseline {args.baseline:.2f}%", xy=(0.012, args.baseline),
            xycoords=("axes fraction", "data"), xytext=(0, -13), textcoords="offset points",
            color=MUTED, fontsize=9,
        )

    style_axes(ax_acc, ylabel="Val Exact Match (%)")
    style_axes(ax_loss, xlabel="Epoch", ylabel="Loss")
    ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_acc.tick_params(labelbottom=False)
    ax_acc.set_xlim(left=min(r[1]["epoch"][0] for r in runs))
    # Chừa chỗ bên phải cho nhãn cuối đường, và chừa phía trên cho con số đỉnh.
    ax_acc.margins(x=0.30)
    ax_acc.set_ylim(top=max(peaks) * 1.10)

    ax_acc.set_title("Độ chính xác validation theo epoch", color=INK,
                     fontsize=13, fontweight="bold", loc="left", pad=14)
    ax_loss.set_title("Loss — nét liền: train, nét đứt: validation", color=INK_2,
                      fontsize=10, loc="left", pad=8)

    # Gọi sau khi ylim đã chốt: phép giãn nhãn dựa trên giới hạn trục thật.
    _place_end_labels(ax_acc, end_labels)

    fig.suptitle(args.title, color=INK, fontsize=15, fontweight="bold", x=0.055, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, os.path.join(args.output_dir, "training_curves.png"))


# --------------------------------------------------------------------------
# Chart 2: so sánh ablation
# --------------------------------------------------------------------------
def plot_ablation(args: argparse.Namespace) -> None:
    if len(args.names) != len(args.acc):
        print("❌ --names và --acc phải cùng số lượng.")
        sys.exit(1)
    stds = args.std or [0.0] * len(args.acc)
    if len(stds) != len(args.acc):
        print("❌ --std phải cùng số lượng với --acc.")
        sys.exit(1)

    order = sorted(range(len(args.acc)), key=lambda i: args.acc[i])
    names = [args.names[i] for i in order]
    accs = [args.acc[i] for i in order]
    errs = [stds[i] for i in order]

    # DOT PLOT, không phải bar. Các cấu hình chỉ chênh nhau vài phần mười điểm
    # trong dải 76-78%, nên trục x buộc phải cắt để nhìn thấy khác biệt — mà bar
    # bị cắt trục thì phóng đại sai lệch (độ dài cột không còn tỉ lệ với giá trị).
    # Chấm mã hoá VỊ TRÍ nên cắt trục là trung thực; đoạn nối về mốc baseline
    # chính là phần "hơn/kém baseline bao nhiêu".
    best = len(accs) - 1
    ref = args.baseline if args.baseline is not None else min(accs)

    fig = new_figure(9.4, 0.62 * len(accs) + 2.8)
    ax = fig.add_subplot(111)
    ypos = list(range(len(accs)))

    for i, (acc, err) in enumerate(zip(accs, errs)):
        color = SERIES[0] if i == best else DEEMPH
        ax.plot([ref, acc], [i, i], color=color, linewidth=2.4, solid_capstyle="round", zorder=3)
        if err:
            ax.errorbar(acc, i, xerr=err, ecolor=INK_2, elinewidth=1.3,
                        capsize=4, capthick=1.3, fmt="none", zorder=4)
        ax.scatter([acc], [i], s=115 if i == best else 85, color=color,
                   edgecolor=SURFACE, linewidth=1.8, zorder=5)

        delta = acc - ref
        text = f"{acc:.2f}%" + (f" ± {err:.2f}" if err else "")
        if args.baseline is not None and abs(delta) >= 0.005:
            text += f"   ({delta:+.2f})"
        # Neo vào mép PHẢI của thanh sai số (toạ độ dữ liệu) rồi mới đẩy ra bằng
        # điểm — offset cố định sẽ khiến text đè lên thanh sai số dài.
        ax.annotate(text, xy=(acc + err, i), xytext=(14, 0),
                    textcoords="offset points", va="center",
                    color=INK if i == best else INK_2,
                    fontsize=10, fontweight="bold" if i == best else "normal", zorder=6)

    if args.baseline is not None:
        ax.axvline(args.baseline, color=MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
        ax.annotate(f"Baseline {args.baseline:.2f}%",
                    xy=(args.baseline, len(accs) - 0.42), xytext=(-8, 0),
                    textcoords="offset points", color=MUTED, fontsize=9,
                    va="center", ha="right")

    style_axes(ax, xlabel="Validation Exact Match (%)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, color=INK_2, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(accs) - 0.3)
    span = max(max(accs) - min(min(accs), ref), 1.0)
    # Chừa đủ chỗ bên phải cho chuỗi "xx.xx% ± y.yy  (+z.zz)" nằm sau thanh sai số.
    ax.set_xlim(min(min(accs), ref) - 0.16 * span,
                max(a + e for a, e in zip(accs, errs)) + 0.95 * span)

    ax.set_title(args.title, color=INK, fontsize=14, fontweight="bold", loc="left", pad=14)
    note = "Trục x cắt để thấy chênh lệch nhỏ — dùng chấm (vị trí) thay vì cột (độ dài) để không phóng đại"
    if any(errs):
        note += "\nThanh sai số = ±1 độ lệch chuẩn qua các seed"
    ax.annotate(note, xy=(0, -0.155 if not any(errs) else -0.20),
                xycoords="axes fraction", color=MUTED, fontsize=8.5, linespacing=1.5)
    fig.tight_layout()
    save(fig, os.path.join(args.output_dir, "ablation_comparison.png"))


# --------------------------------------------------------------------------
# Chart 3: accuracy vs chi phí tính toán
# --------------------------------------------------------------------------
def plot_cost(args: argparse.Namespace) -> None:
    n = len(args.names)
    if not (len(args.acc) == len(args.latency) == n):
        print("❌ --names, --acc, --latency phải cùng số lượng.")
        sys.exit(1)
    params = args.params or [None] * n

    fig = new_figure(9.0, 5.8)
    ax = fig.add_subplot(111)

    sizes = [160.0] * n
    if args.params:
        lo, hi = min(params), max(params)
        span = max(hi - lo, 1)
        sizes = [110 + 320 * (p - lo) / span for p in params]

    for i in range(n):
        # Chỉ 3 màu chuỗi cho scatter (dạng all-pairs), phần còn lại lùi xám.
        color = SERIES[i] if i < 3 else DEEMPH
        ax.scatter(args.latency[i], args.acc[i], s=sizes[i], color=color,
                   edgecolor=SURFACE, linewidth=2.0, zorder=4)
        # Nhãn đặt PHÍA TRÊN chấm: đặt bên phải sẽ bị cắt ở mép phải với điểm
        # latency cao, và đè lên đường baseline nằm ngang với điểm sát baseline.
        ax.annotate(f"{args.names[i]}\n{args.acc[i]:.2f}%",
                    xy=(args.latency[i], args.acc[i]),
                    xytext=(0, 16), textcoords="offset points",
                    color=INK_2, fontsize=9.5, ha="center", va="bottom",
                    linespacing=1.4, zorder=5)

    if args.baseline is not None:
        ax.axhline(args.baseline, color=MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
        ax.annotate(f"Baseline {args.baseline:.2f}%", xy=(0.985, args.baseline),
                    xycoords=("axes fraction", "data"), xytext=(0, -14),
                    textcoords="offset points", color=MUTED, fontsize=9, ha="right")

    style_axes(ax, xlabel="Inference latency (ms / track)", ylabel="Validation Exact Match (%)")
    ax.margins(x=0.16, y=0.30)
    ax.set_title(args.title, color=INK, fontsize=14, fontweight="bold", loc="left", pad=14)
    note = "Góc trên bên trái = tốt nhất (chính xác cao, chi phí thấp)"
    if args.params:
        note += " · kích thước điểm ∝ số tham số"
    ax.annotate(note, xy=(0, -0.16), xycoords="axes fraction", color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, os.path.join(args.output_dir, "accuracy_vs_cost.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xuất biểu đồ kết quả cho báo cáo")
    parser.add_argument("--output-dir", type=str, default="results/charts")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("curves", help="Đường cong training theo epoch")
    p1.add_argument("--history", type=str, nargs="+", default=None,
                    help="File results/history_*.csv do trainer sinh ra")
    p1.add_argument("--from-log", type=str, nargs="+", default=None,
                    help="File log stdout của training (dùng khi chưa có history CSV)")
    p1.add_argument("--labels", type=str, nargs="+", default=None)
    p1.add_argument("--baseline", type=float, default=None)
    p1.add_argument("--title", type=str, default="Diễn biến huấn luyện — CRNN + STN + ResBlock")
    p1.add_argument("--output-dir", type=str, default="results/charts")

    p2 = sub.add_parser("ablation", help="So sánh accuracy giữa các cấu hình")
    p2.add_argument("--names", type=str, nargs="+", required=True)
    p2.add_argument("--acc", type=float, nargs="+", required=True)
    p2.add_argument("--std", type=float, nargs="+", default=None)
    p2.add_argument("--baseline", type=float, default=None)
    p2.add_argument("--title", type=str, default="So sánh độ chính xác giữa các cấu hình")
    p2.add_argument("--output-dir", type=str, default="results/charts")

    p3 = sub.add_parser("cost", help="Accuracy vs chi phí tính toán")
    p3.add_argument("--names", type=str, nargs="+", required=True)
    p3.add_argument("--acc", type=float, nargs="+", required=True)
    p3.add_argument("--latency", type=float, nargs="+", required=True)
    p3.add_argument("--params", type=float, nargs="+", default=None)
    p3.add_argument("--baseline", type=float, default=None)
    p3.add_argument("--title", type=str, default="Đánh đổi độ chính xác và chi phí tính toán")
    p3.add_argument("--output-dir", type=str, default="results/charts")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"curves": plot_curves, "ablation": plot_ablation, "cost": plot_cost}[args.command](args)


if __name__ == "__main__":
    main()
