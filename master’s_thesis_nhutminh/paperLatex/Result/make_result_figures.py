"""Render the two Section-3 figures.

Usage:
    python3 paperLatex/Result/make_result_figures.py

Writes SRLossVsBilinear.{png,pdf} and PsnrVsCorrect.{png,pdf} next to this file.

DATA - both figures read raw run artifacts; nothing is copied from the reports
-----------------------------------------------------------------------------
Fig. srloss   results/multi-seed/s1_mf_sr_ocr/history_s1_seed42.csv
              columns sr_loss and sr_loss_bilinear, one row per epoch. The
              bilinear column is the loss of a plain interpolation of the same
              input against the same target, logged by
              src/training/trainer.py:278-285. It is the only reference that
              answers "does the learned path beat interpolation at all".

Fig. psnr     results/multi-seed/s1_mf_sr_ocr/sr_quality_s1_seed42.csv (999 tracks)
              joined with submission_s1_seed42.txt and the plate_text field of
              each track's annotations.json, to label every track correct/misread.

FORM - why each figure has two panels
-------------------------------------
Both findings were unreadable as single panels, for the same reason: the
quantity that carries the message is a *difference*, and a difference between
two nearly-parallel curves - or between two overlapping histograms - is exactly
what the eye estimates worst. Each figure therefore shows the raw quantity on
the left and the derived difference on the right.

  srloss (a)  the two losses. Bilinear is flat, the learned branch descends.
         (b)  their relative gap, growing from 2.5% to 13.8%. This is the claim
              the abstract makes, so it gets an axis of its own.

  psnr   (a)  the two PSNR distributions, drawn as step outlines rather than
              filled bars, and as densities rather than counts, because the
              correct set outnumbers the misread set roughly 4:1.
         (b)  exact-match accuracy per PSNR quintile. Same fact as the
              r = -0.41 correlation, but a reader can check it directly:
              accuracy falls monotonically as reconstruction quality rises.

Palette: slots 1-2 of the validated categorical set (blue #2a78d6, orange
#eb6834); `validate_palette.js "#2a78d6,#eb6834" --mode light` passes all checks.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "results" / "multi-seed" / "s1_mf_sr_ocr"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID = "#16191d", "#4b535c", "#7b838c", "#dfe3e8"
FS_LAB, FS_TICK, FS_NOTE = 10.0, 9.2, 9.0


def style(ax, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK, colors=INK_2, length=0)
    if title:
        ax.set_title(title, fontsize=FS_LAB + 0.5, color=INK, fontweight="bold",
                     loc="left", pad=8)


def fig_sr_loss():
    rows = list(csv.DictReader(open(RUN / "history_s1_seed42.csv")))
    ep = [int(r["epoch"]) for r in rows]
    sr = [float(r["sr_loss"]) for r in rows]
    bl = [float(r["sr_loss_bilinear"]) for r in rows]
    gap = [100 * (b - a) / b for a, b in zip(sr, bl)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))

    ax1.plot(ep, bl, color=ORANGE, lw=2.2, label="bilinear upscale (reference)")
    ax1.plot(ep, sr, color=BLUE, lw=2.2, label="learned SR branch")
    ax1.fill_between(ep, sr, bl, color=BLUE, alpha=0.11, zorder=1)
    ax1.text(ep[-1] - 1, bl[-1] + 0.0025, "bilinear upscale", ha="right",
             fontsize=FS_NOTE + 0.4, color=ORANGE, fontweight="bold")
    ax1.text(ep[-1] - 1, sr[-1] - 0.0050, "learned branch", ha="right",
             fontsize=FS_NOTE + 0.4, color=BLUE, fontweight="bold")
    style(ax1, "(a)  reconstruction loss")
    ax1.set_xlabel("epoch", fontsize=FS_LAB, color=INK_2)
    ax1.set_ylabel(r"$\ell_1$ against the HR target", fontsize=FS_LAB, color=INK_2)
    ax1.legend(fontsize=FS_NOTE, frameon=False, loc="upper right")
    ax1.set_ylim(min(sr) - 0.007, max(bl) + 0.010)

    ax2.fill_between(ep, 0, gap, color=BLUE, alpha=0.16)
    ax2.plot(ep, gap, color=BLUE, lw=2.2, label="learned SR branch")
    ax2.axhline(0, color=ORANGE, lw=1.8, label="bilinear upscale (reference)")
    ax2.legend(fontsize=FS_NOTE, frameon=False, loc="upper left")
    ax2.annotate(f"{gap[-1]:.1f}%", xy=(ep[-1], gap[-1]),
                 xytext=(ep[-1] - 8, gap[-1] + 1.7), fontsize=FS_NOTE + 1,
                 fontweight="bold", color=BLUE, ha="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    style(ax2, "(b)  how far below bilinear")
    ax2.set_xlabel("epoch", fontsize=FS_LAB, color=INK_2)
    ax2.set_ylabel("relative reduction (%)", fontsize=FS_LAB, color=INK_2)
    ax2.set_ylim(0, max(gap) + 5.5)

    fig.tight_layout(pad=0.6, w_pad=2.4)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"SRLossVsBilinear.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"SRLossVsBilinear: below bilinear {sum(a < b for a, b in zip(sr, bl))}/{len(ep)} "
          f"epochs, gap {gap[0]:.1f}% -> {gap[-1]:.1f}%")


def _tracks():
    gt = {os.path.basename(p): json.load(open(p + "/annotations.json"))["plate_text"].strip()
          for p in glob.glob(str(ROOT / "dataset/data/train/Scenario-B/*/track_*"))}
    pred = {}
    for line in open(RUN / "submission_s1_seed42.txt"):
        if line.strip():
            t, rest = line.split(",", 1)
            pred[t] = rest.rsplit(";", 1)[0].strip()
    out = []
    for r in csv.DictReader(open(RUN / "sr_quality_s1_seed42.csv")):
        t = r["track_id"]
        if t in pred and t in gt:
            out.append((float(r["psnr_sr"]), pred[t] == gt[t]))
    return sorted(out)


def fig_psnr():
    rows = _tracks()
    ok = [p for p, c in rows if c]
    bad = [p for p, c in rows if not c]
    m_ok, m_bad = sum(ok) / len(ok), sum(bad) / len(bad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))

    rng = (rows[0][0], rows[-1][0])
    for vals, c, lab in ((ok, BLUE, f"read correctly  (n={len(ok)})"),
                         (bad, ORANGE, f"misread  (n={len(bad)})")):
        ax1.hist(vals, bins=28, range=rng, density=True, histtype="stepfilled",
                 color=c, alpha=0.15)
        ax1.hist(vals, bins=28, range=rng, density=True, histtype="step",
                 color=c, lw=2.0, label=lab)
        ax1.axvline(sum(vals) / len(vals), color=c, lw=1.5, ls=(0, (4, 2)))
    ax1.annotate("", xy=(m_bad, 0.252), xytext=(m_ok, 0.252),
                 arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1.3))
    ax1.text((m_bad + m_ok) / 2, 0.259, f"+{m_bad - m_ok:.2f} dB", ha="center",
             fontsize=FS_NOTE + 0.5, fontweight="bold", color=INK)
    style(ax1, "(a)  where each group sits")
    ax1.set_xlabel("PSNR of the super-resolved track (dB)", fontsize=FS_LAB, color=INK_2)
    ax1.set_ylabel("density", fontsize=FS_LAB, color=INK_2)
    ax1.legend(fontsize=FS_NOTE, frameon=False, loc="upper right")
    ax1.set_ylim(0, 0.30)

    k, n = 5, len(rows)
    accs, labs = [], []
    for i in range(k):
        g = rows[i * n // k:(i + 1) * n // k]
        accs.append(100 * sum(c for _, c in g) / len(g))
        labs.append(f"{g[0][0]:.1f}–\n{g[-1][0]:.1f}")
    # One series, so no legend: the title names it. The earlier version painted
    # the last bar orange for emphasis, but orange already means "misread" in
    # panel (a); reusing it for "worst quintile" would encode two things with
    # one hue.
    ax2.bar(range(k), accs, width=0.62, color=BLUE, zorder=3)
    for i, a in enumerate(accs):
        ax2.text(i, a + 1.8, f"{a:.1f}", ha="center", fontsize=FS_NOTE + 0.5,
                 fontweight="bold", color=INK, zorder=4)
    style(ax2, "(b)  accuracy per PSNR quintile")
    ax2.set_xticks(range(k))
    ax2.set_xticklabels(labs, fontsize=FS_TICK - 0.8)
    ax2.set_xlabel("PSNR range of the quintile (dB)", fontsize=FS_LAB, color=INK_2)
    ax2.set_ylabel("exact match (%)", fontsize=FS_LAB, color=INK_2)
    ax2.set_ylim(0, 106)

    fig.tight_layout(pad=0.6, w_pad=2.4)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"PsnrVsCorrect.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("PsnrVsCorrect: quintile accuracy " + " -> ".join(f"{a:.1f}" for a in accs))


if __name__ == "__main__":
    fig_sr_loss()
    fig_psnr()
