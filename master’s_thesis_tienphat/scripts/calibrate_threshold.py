#!/usr/bin/env python3
"""Triage threshold ("tự động" vs "cần kiểm tra") for the served model, fixed on val 999.

Replaces roadmap week 5 (threshold_calibration.json: confidence threshold calibrated on
val so that the "automatic" group reaches >= 95% precision) and covers N3 of the review
(confidence calibration + risk-coverage).

Rules, written before the first run (20/11/2026):
  1. Score: among mean_char (the competition's confidence), min_char and seq_prob (exact
     CTC probability of the string), take the one with the lowest AURC (area under the
     risk-coverage curve) on val 999 with 5 frames.
  2. Threshold for N frames: the lowest score t such that the tracks with score >= t have
     precision >= 95% on val 999 (all subsets of N frames pooled for N < 5), i.e. the
     largest automatic group that meets the target. Ties are kept on the automatic side.
  3. The lock-box (--lockbox) is only read after the thresholds are written, to report the
     precision and coverage they actually give on unseen tracks.

    python3 scripts/calibrate_threshold.py
    python3 scripts/calibrate_threshold.py --lockbox
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

from mfplate.decoding import decode  # noqa: E402
from mfplate.settings import ARTIFACTS_DIR, THESIS_DIR  # noqa: E402

CACHE = THESIS_DIR / "var" / "cache"
RUN = "combo3_s42"
SCORES = ("mean_char", "min_char", "seq_prob")
TARGET = 0.95


def scored(lp_batch, labels, decoder):
    """Per track: correct?, and the three confidence scores."""
    ok, s = [], {k: [] for k in SCORES}
    for x, y in zip(lp_batch, labels):
        d = decode(x, decoder, top_k=1)
        ok.append(d.text == y)
        for k in SCORES:
            s[k].append(d.score(k))
    return np.array(ok), {k: np.array(v) for k, v in s.items()}


def risk_coverage(ok, score):
    order = np.argsort(-score, kind="stable")
    ok_sorted = ok[order]
    k = np.arange(1, len(ok) + 1)
    precision = np.cumsum(ok_sorted) / k
    return k / len(ok), 1 - precision, score[order]


def aurc(ok, score):
    _, risk, _ = risk_coverage(ok, score)
    return float(risk.mean())


def wilson_lower(successes, n, z=1.96):
    if n == 0:
        return float("nan")
    p = successes / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    return (centre - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def threshold_for(ok, score, target=TARGET):
    """Lowest threshold whose automatic group (score >= t) has precision >= target."""
    best = None
    for t in np.unique(score)[::-1]:                 # from high to low
        auto = score >= t
        if ok[auto].mean() >= target:
            best = float(t)
        # keep scanning: precision is not monotone in t, take the lowest that passes
    return best


def summarize(ok, score, t):
    auto = score >= t if t is not None else np.zeros_like(ok)
    n_auto = int(auto.sum())
    correct_auto = int(ok[auto].sum())
    return {"n": int(len(ok)), "accuracy": round(100 * ok.mean(), 2),
            "coverage": round(n_auto / len(ok), 4),
            "precision_auto": round(correct_auto / n_auto, 4) if n_auto else None,
            "precision_auto_wilson95_low": round(wilson_lower(correct_auto, n_auto), 4)
            if n_auto else None,
            "accuracy_review_group": round(float(ok[~auto].mean()), 4) if (~auto).any() else None,
            "errors_caught": int((~ok & ~auto).sum()), "errors_total": int((~ok).sum())}


def ece(ok, conf, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def load_split(split):
    meta = np.load(CACHE / f"{split}__meta.npz")
    full = np.load(CACHE / f"{split}__{RUN}_ep30.npz")["log_probs"]
    subs = np.load(CACHE / f"{split}__{RUN}_ep30__subsets.npz")
    return meta["labels"], full, subs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lockbox", action="store_true")
    args = ap.parse_args()
    out_path = ARTIFACTS_DIR / "threshold_calibration.json"
    decision = json.loads((ARTIFACTS_DIR / "decoder_decision.json").read_text())
    decoder = decision["default_decoder"]

    if not args.lockbox:
        labels, full, subs = load_split("val")
        ok5, s5 = scored(full, labels, decoder)
        aurcs = {k: round(aurc(ok5, s5[k]), 5) for k in SCORES}
        name = min(aurcs, key=aurcs.get)
        thresholds, per_n, rc_rows = {}, {}, []
        for n in range(1, 6):
            if n == 5:
                ok, sc = ok5, s5[name]
            else:
                arr = subs[f"N{n}"]                       # [subsets, tracks, T, C]
                pairs = [scored(a, labels, decoder) for a in arr]
                ok = np.concatenate([p[0] for p in pairs])
                sc = np.concatenate([p[1][name] for p in pairs])
            t = threshold_for(ok, sc)
            thresholds[str(n)] = t
            per_n[str(n)] = summarize(ok, sc, t)
            cov, risk, sorted_sc = risk_coverage(ok, sc)
            for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                i = max(0, int(math.ceil(c * len(ok))) - 1)
                rc_rows.append({"n_frames": n, "coverage": c, "precision": round(1 - risk[i], 4),
                                "threshold": round(float(sorted_sc[i]), 6)})
        result = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "model": RUN + "_ep30", "decoder": decoder,
            "rule": ("score = lowest AURC on val 999 (5 frames); threshold_N = lowest t with "
                     f"precision(score >= t) >= {TARGET} on val 999, subsets of N frames pooled"),
            "target_precision": TARGET, "score": name, "aurc_val_5frames": aurcs,
            "ece_val_5frames": {k: round(ece(ok5, s5[k]), 4) for k in SCORES},
            "confidence_gap_val_5frames": round(float(s5["mean_char"][ok5].mean()
                                                      - s5["mean_char"][~ok5].mean()), 4),
            "thresholds_by_frames": thresholds, "val": per_n,
        }
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        with open(ARTIFACTS_DIR / "risk_coverage_val.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rc_rows[0]))
            w.writeheader(); w.writerows(rc_rows)
        _plot(ok5, s5, name, thresholds["5"])
        print(json.dumps({k: result[k] for k in ("score", "aurc_val_5frames", "thresholds_by_frames")},
                         indent=1))
        for n, v in per_n.items():
            print(f"  N={n}: coverage {v['coverage']:.3f}  precision {v['precision_auto']}  "
                  f"(Wilson low {v['precision_auto_wilson95_low']})  bắt {v['errors_caught']}/{v['errors_total']} lỗi")
        print(f"📝 {out_path}")
        return

    # --------------------------------------------------------- lock-box, descriptive
    cal = json.loads(out_path.read_text())
    labels, full, subs = load_split("lockbox")
    name = cal["score"]
    per_n = {}
    for n in range(1, 6):
        if n == 5:
            ok, sc = scored(full, labels, decoder)
            sc = sc[name]
        else:
            pairs = [scored(a, labels, cal["decoder"]) for a in subs[f"N{n}"]]
            ok = np.concatenate([p[0] for p in pairs]); sc = np.concatenate([p[1][name] for p in pairs])
        per_n[str(n)] = summarize(ok, sc, cal["thresholds_by_frames"][str(n)])
    out = {"created": dt.datetime.now().isoformat(timespec="seconds"),
           "note": "descriptive: thresholds were fixed on val 999 before this file was computed",
           "thresholds_from": str(out_path.relative_to(THESIS_DIR)),
           "thresholds_created": cal["created"], "score": name, "lockbox": per_n}
    lb_path = ARTIFACTS_DIR / "triage_lockbox.json"
    lb_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for n, v in per_n.items():
        print(f"  lock-box N={n}: coverage {v['coverage']:.3f}  precision {v['precision_auto']}  "
              f"bắt {v['errors_caught']}/{v['errors_total']} lỗi")
    print(f"📝 {lb_path}")


def _plot(ok, scores, chosen, t5):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for k in SCORES:
        cov, risk, _ = risk_coverage(ok, scores[k])
        ax.plot(cov, 100 * (1 - risk), lw=1.2 if k == chosen else 0.8,
                label=f"{k} (AURC {aurc(ok, scores[k]):.4f})")
    s = scores[chosen]
    c = (s >= t5).mean()
    ax.axhline(100 * TARGET, ls=":", color="grey", lw=0.8)
    ax.axvline(c, ls="--", color="grey", lw=0.8)
    ax.set_xlabel("coverage (tỉ lệ track xử lý tự động)")
    ax.set_ylabel("precision của nhóm tự động (%)")
    ax.set_title("Risk–coverage, val 999, 5 frame", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "risk_coverage_val.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
