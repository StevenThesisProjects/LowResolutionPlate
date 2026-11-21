"""Figures of the capstone proposal: system architecture and Gantt chart.

    python3 proposal/figures/make_figures.py
"""
import datetime as dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "pdf.fonttype": 42})
W = 6.3  # text width in inches (A4, 2.5 cm margins)

INK, UI_C, API_C, DATA_C, ML_C, EXT_C = "#2b2b2b", "#e8f1fb", "#dce8f6", "#fdf0d5", "#e6f4ea", "#f2f2f2"


def save(fig, name):
    fig.savefig(os.path.join(HERE, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(HERE, f"{name}.png"), bbox_inches="tight", pad_inches=0.03, dpi=200)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ls="-", fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=INK, lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25,
            fontweight="bold" if bold else "normal")


def group(ax, x, y, w, h, label, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#7a7a7a", lw=0.6, ls="--"))
    ax.text(x + 0.08, y + h - 0.08, label, ha="left", va="top", fontsize=7, style="italic",
            color="#444444")


def arrow(ax, p, q, text=None, both=False, ls="-", off=(0, 0.07)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7,
                                 lw=0.8, color=INK, ls=ls, shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center",
                va="bottom", fontsize=6.3, color="#333333")


def architecture():
    fig, ax = plt.subplots(figsize=(W, 4.1))
    ax.set_xlim(0, W); ax.set_ylim(0, 4.1); ax.axis("off")
    # Access layer
    box(ax, 0.05, 2.95, 0.95, 0.75, "Operator /\nreviewer\n(desktop or\nmobile browser)", EXT_C)
    box(ax, 0.05, 1.95, 0.95, 0.6, "Cloudflare\nTunnel (HTTPS,\noptional)", EXT_C, ls="--")
    arrow(ax, (0.525, 2.95), (0.525, 2.55))
    # Presentation container
    group(ax, 1.15, 2.75, 1.17, 1.2, "container: ui", UI_C)
    box(ax, 1.22, 2.85, 1.03, 0.75, "Web UI\n(Gradio)\nupload, results,\nfeedback, history", "white")
    arrow(ax, (1.0, 2.25), (1.3, 2.85))
    ax.text(1.18, 2.3, "HTTPS", ha="left", va="center", fontsize=6.3, color="#333333")
    arrow(ax, (1.0, 3.4), (1.22, 3.4))
    # API container
    group(ax, 2.75, 1.75, 3.5, 2.2, "container: api (FastAPI, single process, CPU)", API_C)
    steps = [("Upload\nvalidation", 2.85), ("Pre-\nprocessing\n(32×128,\nluma)", 3.53),
             ("Inference\nengine\n(frozen\nResTran)", 4.21), ("CTC\ndecoder", 4.89),
             ("Triage\n(threshold\nper N)", 5.57)]
    for t, x in steps:
        box(ax, x, 2.75, 0.6, 0.85, t, "white", fs=6.6)
    for (_, x0), (_, x1) in zip(steps[:-1], steps[1:]):
        arrow(ax, (x0 + 0.6, 3.17), (x1, 3.17))
    arrow(ax, (2.25, 3.22), (2.85, 3.22), both=True)
    ax.text(2.53, 3.3, "REST/\nJSON", ha="center", va="bottom", fontsize=6.0, color="#333333")
    box(ax, 2.85, 1.85, 1.3, 0.6, "REST endpoints:\n/predict /feedback /history\n/stats /model /health\nOpenAPI 3.1 at /docs", "white", fs=6.1)
    box(ax, 4.95, 1.85, 1.22, 0.6, "Store (SQLite)\npredictions, feedback,\ninput hashes only", DATA_C, fs=6.2)
    arrow(ax, (5.87, 2.75), (5.6, 2.45), ls="-")
    # Offline ML pipeline
    group(ax, 0.05, 0.05, 6.2, 1.5, "offline ML pipeline (training repository, rented V100 GPU)", ML_C)
    ml = [("Training recipe #50\n(6 training-only\ncomponents)", 0.2), ("Pre-registered\nlock-box evaluation\n(1,000 tracks, 4 seeds)", 1.75),
          ("Calibration on val 999\n(decoder gate,\nthresholds per N)", 3.3), ("Frozen artifacts\ncheckpoint + SHA-256,\nserving config", 4.85)]
    for t, x in ml:
        box(ax, x, 0.25, 1.3, 0.85, t, "white", fs=6.6)
    for (_, x0), (_, x1) in zip(ml[:-1], ml[1:]):
        arrow(ax, (x0 + 1.3, 0.67), (x1, 0.67))
    arrow(ax, (4.95, 1.1), (4.51, 2.75))
    ax.text(4.82, 1.3, "loaded at start-up,\nSHA-256 verified", ha="right", va="center", fontsize=6.2,
            color="#333333")
    save(fig, "fig_architecture")


def gantt():
    D = lambda s: dt.date.fromisoformat(s)  # noqa: E731
    today = D("2026-11-20")
    phases = [
        ("P1 Requirements & baseline analysis", "2026-08-10", "2026-09-30"),
        ("P2 Model engineering (training recipe)", "2026-10-01", "2026-10-22"),
        ("P3 Pre-registered validation & mechanisms", "2026-10-23", "2026-10-31"),
        ("P4 Paper writing (ICCIES 2027)", "2026-11-01", "2026-11-15"),
        ("P5 Service design & API", "2026-11-16", "2026-11-22"),
        ("P6 Web UI & deployment", "2026-11-20", "2026-12-06"),
        ("P7 Testing & user evaluation", "2026-11-20", "2026-12-13"),
        ("P8 Thesis writing, demo & release", "2026-12-07", "2026-12-27"),
    ]
    milestones = [("M1", "2026-08-12"), ("M2", "2026-10-23"), ("M3", "2026-10-25"), ("M4", "2026-10-31"),
                  ("M5", "2026-11-20"), ("M6", "2026-11-29"), ("M7", "2026-12-06"), ("M8", "2026-12-13"),
                  ("M9", "2026-12-27")]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for i, (name, a, b) in enumerate(phases):
        a, b = D(a), D(b)
        y = len(phases) - 1 - i
        done_end = min(b, today)
        if done_end > a:
            ax.barh(y, (done_end - a).days, left=a, height=0.55, color="#1f5fae")
        if b > today:
            start = max(a, today)
            ax.barh(y, (b - start).days, left=start, height=0.55, color="#a9c4e8", hatch="///",
                    edgecolor="white", lw=0)
        ax.text(D("2026-08-01"), y, name, ha="right", va="center", fontsize=7)
    for k, (label, d) in enumerate(milestones):
        ax.plot(D(d), len(phases) - 0.25, marker="D", ms=4, color="#b5651d")
        lift = 0.45 if label == "M3" else 0.05                    # M2 and M3 are two days apart
        ax.text(D(d), len(phases) + lift, label, ha="center", va="bottom", fontsize=6.3,
                color="#b5651d")
    ax.axvline(today, color="#b42318", lw=0.8, ls="--")
    ax.text(today + dt.timedelta(days=1), -0.5, "today (20 Nov)", ha="left", va="center",
            fontsize=6.3, color="#b42318")
    ax.set_xlim(D("2026-08-01"), D("2026-12-31"))
    ax.set_ylim(-0.7, len(phases) + 0.9)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.barh(-10, 1, color="#1f5fae", label="completed")
    ax.barh(-10, 1, color="#a9c4e8", hatch="///", edgecolor="white", label="planned")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=2)
    save(fig, "fig_gantt")


if __name__ == "__main__":
    architecture()
    gantt()
    print("ok")
