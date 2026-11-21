#!/usr/bin/env python3
"""Accuracy by sharpness quartile and plate layout: baseline vs recipe on the lock-box.

Replaces roadmap week 4 ("slice_sharpness.csv: phân tích theo tứ phân vị độ nét Q1-Q4 và
theo layout"), for the recipe instead of LGTF. No new inference: the per-track
correctness is the lock-box scoring of 25/10 (results_new/lockbox_eval_per_track.npz,
4 seeds per arm, last epoch, GPU). Sharpness = mean over the 5 LR frames of the Laplacian
variance at 32x128 (mfplate.preprocess.sharpness); quartiles are cut on the lock-box itself.
Descriptive only: about 250 tracks per quartile, so a difference below ~3 points is
within sampling noise.

    python3 scripts/slice_analysis.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.settings import ARTIFACTS_DIR, REPO_ROOT, THESIS_DIR  # noqa: E402

CACHE = THESIS_DIR / "var" / "cache"


def boot_ci(a, b, n=10000, seed=0):
    d = (b - a) * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    meta = np.load(CACHE / "lockbox__meta.npz")
    gpu = np.load(REPO_ROOT / "results_new" / "lockbox_eval_per_track.npz")
    base = gpu["baseline_last"].mean(axis=0)            # per-track accuracy over 4 seeds
    meth = gpu["method_last"].mean(axis=0)
    assert np.array_equal(gpu["layouts"], meta["layouts"]), "track order differs"
    sharp = meta["sharpness"].mean(axis=1)
    edges = np.quantile(sharp, [0.25, 0.5, 0.75])
    q = np.digitize(sharp, edges)
    rows = []
    groups = [("all", np.ones_like(q, bool))]
    groups += [(f"Q{k + 1}", q == k) for k in range(4)]
    groups += [(lay, meta["layouts"] == lay) for lay in ("Mercosur", "Brazilian")]
    groups += [(f"{lay} Q{k + 1}", (meta["layouts"] == lay) & (q == k))
               for lay in ("Mercosur", "Brazilian") for k in range(4)]
    for name, m in groups:
        lo, hi = boot_ci(base[m], meth[m])
        rows.append({"slice": name, "n": int(m.sum()),
                     "sharpness_range": (f"{sharp[m].min():.0f}-{sharp[m].max():.0f}"),
                     "baseline": round(100 * base[m].mean(), 2), "recipe": round(100 * meth[m].mean(), 2),
                     "gain": round(100 * (meth[m].mean() - base[m].mean()), 2),
                     "ci95_low": round(lo, 2), "ci95_high": round(hi, 2)})
    out = ARTIFACTS_DIR / "slice_sharpness.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (ARTIFACTS_DIR / "slice_sharpness.json").write_text(json.dumps(
        {"source": "results_new/lockbox_eval_per_track.npz (last epoch, 4 seeds per arm)",
         "quartile_edges": [round(float(e), 1) for e in edges], "rows": rows},
        indent=2, ensure_ascii=False) + "\n")
    for r in rows:
        print(f"  {r['slice']:14s} n={r['n']:4d}  baseline {r['baseline']:6.2f}  recipe "
              f"{r['recipe']:6.2f}  Δ {r['gain']:+5.2f} [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]")
    print(f"📝 {out}")


if __name__ == "__main__":
    main()
