#!/usr/bin/env python3
"""Greedy vs beam vs syntax-constrained beam ("SC-CTC") on the frozen recipe, 4 seeds.

Replaces roadmap week 2 ("3 seed x 3 decoder, by layout, paired McNemar"), measured on
the configuration that was actually frozen (#50, lock-box runs, last epoch) instead of
run #13.

Decision rule, written before this script was first run (20/11/2026):
    A beam decoder becomes the app's default only if, on the 999-track val, its mean
    gain over greedy across the 4 seeds is >= +0.30 points AND it is not below greedy
    on any seed. Otherwise the default stays greedy, the decoder behind every number in
    the paper and the lock-box report. Only val is used for the decision; the lock-box
    table (--lockbox) is descriptive and is computed after the decision is written.

    python3 scripts/eval_decoders.py              # val -> decision
    python3 scripts/eval_decoders.py --lockbox    # descriptive
"""
import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.decoding import decode, layout_of  # noqa: E402
from mfplate.settings import ARTIFACTS_DIR, THESIS_DIR  # noqa: E402

CACHE = THESIS_DIR / "var" / "cache"
RUNS = ["combo3_s42", "combo3_s43", "combo3_s7", "combo3_s2026"]
DECODERS = ["greedy", "beam", "constrained"]
MIN_GAIN = 0.30


def mcnemar(a_ok, b_ok):
    gain = int(np.sum(~a_ok & b_ok)); loss = int(np.sum(a_ok & ~b_ok)); n = gain + loss
    if n == 0:
        return gain, loss, 1.0
    tail = sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / 2.0 ** n
    return gain, loss, min(1.0, 2.0 * tail)


def decode_split(split):
    meta = np.load(CACHE / f"{split}__meta.npz")
    labels, layouts = meta["labels"], meta["layouts"]
    preds = {}
    for run in RUNS:
        lp = np.load(CACHE / f"{split}__{run}_ep30.npz")["log_probs"]
        for d in DECODERS:
            preds[(run, d)] = [decode(x, d, top_k=1, beam=10).text for x in lp]
        print(f"  {run}: xong", flush=True)
    return labels, layouts, preds


def table(split, labels, layouts, preds):
    rows = []
    for run in RUNS:
        g_ok = np.array([p == y for p, y in zip(preds[(run, "greedy")], labels)])
        for d in DECODERS:
            ok = np.array([p == y for p, y in zip(preds[(run, d)], labels)])
            gain, loss, p = mcnemar(g_ok, ok)
            row = {"split": split, "run": run, "seed": run.split("_s")[1], "decoder": d,
                   "n": len(labels), "acc": round(100 * ok.mean(), 2)}
            for lay in ("Mercosur", "Brazilian"):
                m = layouts == lay
                row[f"acc_{lay.lower()}"] = round(100 * ok[m].mean(), 2)
                row[f"n_{lay.lower()}"] = int(m.sum())
            row.update({
                "invalid_syntax": int(sum(layout_of(t) is None for t in preds[(run, d)])),
                "changed_vs_greedy": int(sum(a != b for a, b in zip(preds[(run, d)],
                                                                    preds[(run, "greedy")]))),
                "fixed_vs_greedy": gain, "broken_vs_greedy": loss, "mcnemar_p": round(p, 4)})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lockbox", action="store_true", help="descriptive table on the lock-box")
    args = ap.parse_args()
    split = "lockbox" if args.lockbox else "val"
    decision_path = ARTIFACTS_DIR / "decoder_decision.json"
    if args.lockbox and not decision_path.exists():
        sys.exit("❌ Chạy trên val trước: quyết định phải có trước khi xem lock-box")

    labels, layouts, preds = decode_split(split)
    rows = table(split, labels, layouts, preds)
    out_csv = ARTIFACTS_DIR / ("results_decode.csv" if split == "val" else
                               "results_decode_lockbox.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"📝 {out_csv}")
    for d in DECODERS:
        accs = [r["acc"] for r in rows if r["decoder"] == d]
        print(f"  {d:12s} " + "  ".join(f"{a:6.2f}" for a in accs) + f"  | TB {np.mean(accs):6.2f}")

    if split == "val":
        gains = {}
        for d in ("beam", "constrained"):
            g = [r["acc"] - next(x["acc"] for x in rows if x["run"] == r["run"]
                                 and x["decoder"] == "greedy")
                 for r in rows if r["decoder"] == d]
            gains[d] = {"per_seed": [round(v, 2) for v in g], "mean": round(float(np.mean(g)), 3),
                        "passes": bool(np.mean(g) >= MIN_GAIN and min(g) >= 0)}
        passing = [d for d in ("constrained", "beam") if gains[d]["passes"]]
        default = (max(passing, key=lambda d: gains[d]["mean"]) if passing else "greedy")
        decision = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "rule": (f"beam decoder becomes default only if mean gain over greedy >= +{MIN_GAIN} "
                     "points on val 999 across 4 seeds and no seed below greedy"),
            "split": "splits/val_tracks.json (999 tracks)", "runs": RUNS,
            "gain_over_greedy": gains, "default_decoder": default,
            "table": str(out_csv.relative_to(THESIS_DIR)),
        }
        decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
        print(f"➡️  bộ giải mã mặc định: {default}   ({json.dumps(gains)})")
        print(f"📝 {decision_path}")


if __name__ == "__main__":
    main()
