"""Render the qualitative data figure (Fig. 2) for the paper.

Usage:
    python3 paperLatex/Methonology/make_data_samples.py

Writes DataSamples.png / .pdf next to this script.

WHY THIS FIGURE EXISTS
----------------------
Section 2 states how little signal each character carries. That claim is the premise of the whole paper - it is why multi-frame
matters, why a single-image restorer can only hallucinate, and why exact match
sits 15 points below character accuracy. A reviewer should not have to take it on
trust when the corpus is public and one figure settles it.

Frames are drawn at their true pixel grid, upscaled with NEAREST so no
interpolation is introduced by the figure itself: what is printed is what the
network receives, one image pixel per screen block.

DATA
----
Two tracks per plate layout, taken from Scenario-B - the harder of the two
acquisition scenarios and the one the validation split is drawn from. Sizes are
read from the files, not from the report, so they cannot drift.

    dataset/data/train/Scenario-B/<layout>/track_*/lr-00{1..5}.jpg
    dataset/data/train/Scenario-B/<layout>/track_*/hr-001.jpg
    ...                                            /annotations.json

NOTE ON THE VALIDATION SPLIT
----------------------------
These are Scenario-B examples, not validation examples. The validation IDs in
dataset/val_tracks.json (currently 99 entries such as track_21392) do not resolve
against the tracks now on disk, which are numbered track_00001..track_20000; the
runs behind the reported numbers logged "[VAL] 999 tracks". See
report/paper/architecture_provenance.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[1] / "dataset" / "data" / "train" / "Scenario-B"

TRACKS = [
    ("Brazilian", "track_10003"),
    ("Mercosur", "track_12602"),
]

INK = "#16191d"
INK_2 = "#4b535c"
MUTED = "#7b838c"
ACCENT = "#2a78d6"


def load(track_dir: Path):
    ann = json.loads((track_dir / "annotations.json").read_text())
    lrs = [Image.open(track_dir / f"lr-00{i}.jpg").convert("RGB") for i in range(1, 6)]
    hr = Image.open(track_dir / "hr-001.jpg").convert("RGB")
    return ann["plate_text"], ann.get("plate_layout", "?"), lrs, hr


def main() -> None:
    rows = [(lay, ROOT / lay / tid) for lay, tid in TRACKS]

    fig = plt.figure(figsize=(11.0, 4.5), facecolor="white")
    n_rows = len(rows)
    # 5 LR frames | gap | HR
    left, right = 0.075, 0.985
    top, bottom = 0.845, 0.11
    row_h = (top - bottom) / n_rows
    cell_w = 0.118
    gap = 0.030

    fig.text(0.5, 0.945,
             "What the network actually receives",
             ha="center", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.905,
             "Five consecutive low-resolution frames of one track (left) and a "
             "high-resolution capture of the same plate (right). "
             "Pixels are shown unsmoothed, at the true sampling grid.",
             ha="center", fontsize=9.2, color=MUTED)

    for r, (layout, tdir) in enumerate(rows):
        text, layout_name, lrs, hr = load(tdir)
        y = top - (r + 1) * row_h + 0.045
        h = row_h - 0.105

        for i, im in enumerate(lrs):
            ax = fig.add_axes([left + i * cell_w, y, cell_w * 0.90, h])
            ax.imshow(im, interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#c9cfd6"); s.set_linewidth(1.0)
            if r == 0:
                ax.set_title(f"LR frame {i + 1}", fontsize=8.8, color=INK_2, pad=5)
            if i == 0:
                ax.text(-0.10, 0.5, f"{layout_name}\n{text}", transform=ax.transAxes,
                        ha="right", va="center", fontsize=10.2, fontweight="bold",
                        color=INK, family="monospace", linespacing=1.6)

        w0, h0 = lrs[0].size
        fig.text(left + 2.5 * cell_w - cell_w * 0.05, y - 0.052,
                 f"{w0} × {h0} px  →  ≈ {w0 / 7:.1f} px of width per character",
                 ha="center", fontsize=8.8, color=ACCENT, fontweight="bold")

        x_hr = left + 5 * cell_w + gap
        ax = fig.add_axes([x_hr, y, right - x_hr, h])
        ax.imshow(hr, interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c9cfd6"); s.set_linewidth(1.0)
        if r == 0:
            ax.set_title("HR reference (SR target)", fontsize=8.8, color=INK_2, pad=5)
        wh, hh = hr.size
        fig.text(x_hr + (right - x_hr) / 2, y - 0.052, f"{wh} × {hh} px",
                 ha="center", fontsize=8.8, color=MUTED)

    fig.text(left, 0.028,
             "Scenario-B tracks (JPEG, the harder acquisition scenario). "
             "Frames within a track differ in blur, compression and sub-pixel offset — "
             "the redundancy the multi-frame branches exploit.",
             fontsize=8.4, color=MUTED)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"DataSamples.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
