#!/usr/bin/env python3
"""Three demo tracks for the UI and the 5-minute demo (roadmap week 11: easy / blurry /
needs review), chosen from val 999 with the served model's cached outputs:

  * easy   : read correctly, automatic, Mercosur, sharpest quartile, highest confidence;
  * blurry : read correctly and automatic although in the blurriest quartile;
  * review : flagged "cần kiểm tra" AND read wrongly, i.e. the case triage exists for.

Each zip holds the track's lr-*.jpg and annotations.json (the UI shows the label next to
the reading). The tracks are LRLPR-26 data under its licence: keep the zips for the
private demo, do not publish them.

    python3 scripts/make_samples.py
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.decoding import decode  # noqa: E402
from mfplate.settings import ARTIFACTS_DIR, THESIS_DIR  # noqa: E402

CACHE = THESIS_DIR / "var" / "cache"
OUT = THESIS_DIR / "samples"


def main():
    cal = json.loads((ARTIFACTS_DIR / "threshold_calibration.json").read_text())
    t5, score_name = cal["thresholds_by_frames"]["5"], cal["score"]
    meta = np.load(CACHE / "val__meta.npz")
    lp = np.load(CACHE / "val__combo3_s42_ep30.npz")["log_probs"]
    dec = [decode(x, cal["decoder"], top_k=1) for x in lp]
    ok = np.array([d.text == y for d, y in zip(dec, meta["labels"])])
    score = np.array([d.score(score_name) for d in dec])
    sharp = meta["sharpness"].mean(axis=1)
    q1, q3 = np.quantile(sharp, [0.25, 0.75])
    auto = score >= t5
    merc = meta["layouts"] == "Mercosur"
    picks = {
        "easy": np.nonzero(ok & auto & merc & (sharp >= q3))[0],
        "blurry": np.nonzero(ok & auto & (sharp <= q1))[0],
        "review": np.nonzero(~ok & ~auto)[0],
    }
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.zip"):
        old.unlink()
    lines = ["# Track mẫu cho demo (sinh bởi `scripts/make_samples.py`)", "",
             "Lấy từ val 999 (model chưa từng train trên các track này). Dữ liệu LRLPR-26 theo "
             "giấy phép của bộ dữ liệu: chỉ dùng cho demo riêng tư, không công bố.", "",
             "| File | Track | Layout | Nhãn | Model đọc | Điểm | Ngưỡng | Độ nét TB |",
             "|---|---|---|---|---|---|---|---|"]
    for kind, idx in picks.items():
        if len(idx) == 0:
            continue
        # easy: the most confident; blurry: the blurriest; review: the median score among
        # the wrong + flagged tracks (a typical case, not one sitting on the threshold)
        if kind == "easy":
            i = int(idx[np.argmax(score[idx])])
        elif kind == "blurry":
            i = int(idx[np.argmin(sharp[idx])])
        else:
            i = int(idx[np.argsort(score[idx])[len(idx) // 2]])
        tid = str(meta["track_ids"][i])
        folder = Path(meta["paths"][i][0]).parent
        name = f"{kind}_{tid}.zip"
        with zipfile.ZipFile(OUT / name, "w") as z:
            for p in sorted(folder.glob("lr-*")) + [folder / "annotations.json"]:
                z.write(p, f"{tid}/{p.name}")
        lines.append(f"| {name} | {tid} | {meta['layouts'][i]} | {meta['labels'][i]} | "
                     f"{dec[i].text} | {score[i]:.6f} | {t5:.6f} | {sharp[i]:.0f} |")
        print(f"  {kind:7s} {tid} nhãn {meta['labels'][i]} đọc {dec[i].text} điểm {score[i]:.4f}")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"📝 {OUT}")


if __name__ == "__main__":
    main()
