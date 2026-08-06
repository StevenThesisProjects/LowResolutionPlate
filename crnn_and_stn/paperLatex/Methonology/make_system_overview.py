"""Render the MF-SR-OCR system overview figure for the paper.

Usage:
    python3 paperLatex/Methonology/make_system_overview.py

Writes SystemOverview.png (300 dpi, referenced by the .tex) and
SystemOverview.pdf (vector, sharper in print) next to this script.

Edit this file rather than the image, so the figure stays in sync with the code.
If the text reads too small once scaled to \\textwidth, raise FONT_SCALE and
re-run; the boxes have room for it.

================================================================================
PROVENANCE - what the figure draws and where each claim comes from
================================================================================
The figure depicts configuration **S1**, the proposed model. Its ground truth is
the run banner printed at the top of

    results/multi-seed/s1_mf_sr_ocr/log_s1_seed42.txt

which records the exact flags the reported 79.95% +/- 0.15 was produced with:

    STN     : True
    SE      : True
    SR      : True (scale=2, lambda_sr=0.1, edge=0.0, perceptual=0.0,
                    multi_frame=True)
    DCN     : True | Fusion: attention | Frames: 5
    STN pool: (4, 8) | Width /8 -> T=32
    Decode  : constrained (layouts=LLLNLNN,LLLNNNN, beam=16)
    EMA     : True (decay=0.999)
    Backbone: base=64, blocks=(2,2,2,2,2), channels=(64,128,256,256,512),
              res_scale=0.1, norm=group
    Model params: 29,577,214

Do not redraw this figure from the reports alone: J1 (no SR, T=16) and S4
(sr_scale=1, width_downsample=4) share most of the flags and are easy to
confuse with S1. See report/baseline1_crnn_stn/multi_seed_results.md.

Block-by-block sources
----------------------
The full trace - every block mapped to file:line in src/, with the design
rationale quoted from the code comments - lives in

    report/paper/architecture_provenance.md

Keep that document as the single source; do not duplicate the mapping here, or
the two copies will drift. The short version:

    STN            components.py:399   pool (4,8), identity init
    DCNv2          components.py:249   middle-frame ref, identity kernel
    Multi-frame SR components.py:319   temporal mean, PixelShuffle, bilinear base
    Backbone       components.py:98    GroupNorm (make_norm:40) + SE (:19)
    Fusion         components.py:190   learned per-frame weights
    BiLSTM/CTC     crnn.py:124         2x256, T=32
    Constrained    postprocess.py:283  two layouts, beam 16
    L_SR           losses.py:84 (l1) + trainer.py:269-272 (warp) + :270 (detach)
    L_Total        trainer.py:336      lambda_SR scales the whole SR term

Deliberately NOT drawn: the VGG perceptual term (losses.py:87-88) and the Sobel
edge term (losses.py:86). The banner records `edge=0.0, perceptual=0.0`, so
neither contributes to any reported number; drawing them would imply the 79.95%
was obtained with a perceptual loss.
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent

FONT_SCALE = 1.0
FS_TITLE = 13.0 * FONT_SCALE
FS_SUB = 10.0 * FONT_SCALE
FS_SHAPE = 9.0 * FONT_SCALE

# Print-friendly palette: distinct in colour, still separable in greyscale.
C_INPUT = "#E7ECF1"   # data tensors
C_ALIGN = "#CBDDEF"   # geometric alignment
C_SR = "#FADFC2"      # restoration branch
C_REC = "#D3E6D2"     # recognition branch
C_LOSS = "#F2D5DB"    # loss terms
EDGE = "#37424E"
TXT = "#171E26"
MUTED = "#66727E"

LW = 1.4


def box(ax, x, y, w, h, title, sub=None, shape=None, color=C_REC, tag=None):
    """Rounded block: bold title, optional detail line, optional shape caption."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=color, edgecolor=EDGE, linewidth=LW, zorder=3,
        )
    )
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.63, title, ha="center", va="center",
                fontsize=FS_TITLE, fontweight="bold", color=TXT, zorder=4)
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center",
                fontsize=FS_SUB, color=TXT, zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=FS_TITLE, fontweight="bold", color=TXT, zorder=4)
    if shape:
        ax.text(cx, y - 0.13, shape, ha="center", va="top",
                fontsize=FS_SHAPE, color=MUTED, family="monospace", zorder=4)
    if tag:
        ax.text(x + 0.11, y + h - 0.11, tag, ha="left", va="top",
                fontsize=FS_SUB, fontweight="bold", color=MUTED, zorder=4)


def frames(ax, x, y, w, h, label, n=4, dx=0.06, dy=0.07):
    """A stack of offset rectangles standing for the five frames of a track."""
    for i in range(n, 0, -1):
        ax.add_patch(
            FancyBboxPatch(
                (x + i * dx, y + i * dy), w, h,
                boxstyle="round,pad=0.015,rounding_size=0.06",
                facecolor=C_INPUT, edgecolor=EDGE, linewidth=0.9,
                zorder=2, alpha=0.6,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            facecolor=C_INPUT, edgecolor=EDGE, linewidth=LW, zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=FS_TITLE, fontweight="bold", color=TXT, zorder=4)


def arrow(ax, p0, p1, color=EDGE, lw=LW):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color=color, zorder=2))


def elbow(ax, pts, color=EDGE, lw=LW):
    """Orthogonal polyline; the arrow head sits on the final segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=2)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # ------------------------------------------------------------- row 1
    y1, h1 = 5.30, 1.15

    frames(ax, 0.20, y1, 1.30, h1, "5 LR\nframes")
    ax.text(0.85, y1 - 0.13, "[B, 5, 3, 32, 128]", ha="center", va="top",
            fontsize=FS_SHAPE, color=MUTED, family="monospace")

    box(ax, 2.00, y1, 2.05, h1, "STN", r"affine $\theta$, identity init",
        "[B, 5, 3, 32, 128]", C_ALIGN, "1")
    box(ax, 4.55, y1, 2.05, h1, "DCNv2 align", "to middle frame",
        "[B, 5, 3, 32, 128]", C_ALIGN, "2")
    box(ax, 7.10, y1, 2.25, h1, r"Multi-frame SR $\times$2",
        "temporal mean, PixelShuffle", "[B*5, 3, 64, 256]", C_SR, "3")

    arrow(ax, (1.78, y1 + h1 / 2), (2.00, y1 + h1 / 2))
    arrow(ax, (4.05, y1 + h1 / 2), (4.55, y1 + h1 / 2))
    arrow(ax, (6.60, y1 + h1 / 2), (7.10, y1 + h1 / 2))

    # ------------------------------------------------------------- row 2
    y2, h2 = 2.90, 1.15

    # "SE" is not decoration: preset "stable" turns it on (train.py L149) and the
    # S1 banner records `SE: True`, so leaving it out would misstate the model.
    box(ax, 0.20, y2, 2.00, h2, "ResBlock backbone", "GroupNorm + SE, shared",
        "[B, 5, 512, 1, 32]", C_REC, "4")
    box(ax, 2.60, y2, 1.85, h2, "Attention fusion", "per-frame weights",
        "[B, 512, 1, 32]", C_REC, "5")
    box(ax, 4.85, y2, 1.50, h2, "BiLSTM", r"2 $\times$ 256",
        "[B, 32, 37]", C_REC, "6")
    box(ax, 6.75, y2, 1.70, h2, "CTC decode", "layout-constrained",
        "beam 16", C_REC)

    arrow(ax, (2.20, y2 + h2 / 2), (2.60, y2 + h2 / 2))
    arrow(ax, (4.45, y2 + h2 / 2), (4.85, y2 + h2 / 2))
    arrow(ax, (6.35, y2 + h2 / 2), (6.75, y2 + h2 / 2))

    # Main path folds from the end of row 1 into the top of row 2. Both drops
    # leave the SR block near its edges so neither crosses its shape caption.
    elbow(ax, [(7.30, y1), (7.30, 4.70), (1.20, 4.70), (1.20, y2 + h2)])

    # --------------------------------------------- supervision of the SR head
    frames(ax, 9.75, 4.30, 1.35, 1.00, "5 HR\nframes")
    box(ax, 9.55, 2.95, 1.75, 0.75, r"Warp$_{\mathrm{sg}[\theta]}$",
        color=C_SR)
    ax.text(10.42, 2.82, r"$\theta$ from STN, gradient cut", ha="center",
            va="top", fontsize=FS_SHAPE, color=MUTED, style="italic")
    box(ax, 9.55, 1.60, 1.75, 0.80, r"$\mathcal{L}_{\mathrm{SR}}$",
        r"$\ell_1$ pixel loss", color=C_LOSS)

    arrow(ax, (10.42, 4.30), (10.42, 3.70))
    arrow(ax, (10.42, 2.95), (10.42, 2.40))
    elbow(ax, [(9.20, y1), (9.20, 2.00), (9.55, 2.00)])

    # ------------------------------------------------------------- outputs
    box(ax, 4.60, 1.60, 1.70, 0.80, r"$\mathcal{L}_{\mathrm{CTC}}$",
        color=C_LOSS)
    arrow(ax, (5.45, y2), (5.45, 2.40))

    ax.add_patch(
        FancyBboxPatch(
            (6.75, 1.62), 1.70, 0.68,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white", edgecolor=EDGE, linewidth=LW, zorder=3,
        )
    )
    ax.text(7.60, 1.96, "BCH6E19", ha="center", va="center", family="monospace",
            fontsize=FS_TITLE, fontweight="bold", color=TXT, zorder=4)
    arrow(ax, (7.60, y2), (7.60, 2.30))

    ax.add_patch(
        FancyBboxPatch(
            (5.35, 0.42), 3.30, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="white", edgecolor=EDGE, linewidth=LW, zorder=3,
        )
    )
    ax.text(7.00, 0.845,
            r"$\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{CTC}}"
            r" + \lambda_{\mathrm{SR}} \mathcal{L}_{\mathrm{SR}}$,  "
            r"$\lambda_{\mathrm{SR}} = 0.1$",
            ha="center", va="center", fontsize=FS_TITLE, color=TXT, zorder=4)

    arrow(ax, (5.45, 1.60), (6.05, 1.27), color=MUTED)
    arrow(ax, (10.42, 1.60), (8.65, 1.05), color=MUTED)

    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"SystemOverview.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
