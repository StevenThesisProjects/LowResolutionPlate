#!/usr/bin/env python3
"""Score the frozen lock-box runs once, as pre-registered in documents/lockbox_protocol.md.

For every arm (baseline, method) and seed it scores, on the held-out tracks:
  * last epoch  — primary metric, no epoch is chosen anywhere;
  * Last-6      — mean accuracy of the last 6 epoch checkpoints (the decision metric);
  * best        — the checkpoint chosen on the 999-track val (comparable to the published 78.70).

Statistics, at the two levels the review asked for (ISSUE #2):
  * across seeds: mean ± std, Welch t-test, Cohen's d (training variability);
  * across tracks: paired bootstrap of the method - baseline difference of the per-track
    accuracy averaged over seeds (10,000 resamples), and McNemar for each seed pair.
--frames-analysis adds E3: accuracy with N = 1..5 frames, averaged over every subset of N
frames, on the last-epoch checkpoints.

    python tools/eval_lockbox.py --frames-analysis --out results/lockbox_eval.json
"""
import argparse
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs.config import Config  # noqa: E402
from src.data.dataset import MultiFrameDataset  # noqa: E402
from src.models.restran import ResTranOCR  # noqa: E402
from src.utils.postprocess import decode_with_confidence  # noqa: E402


# ---------------------------------------------------------------- statistics (no scipy)
def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 3e-14:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def welch(x, y):
    """Two-sided Welch t-test of mean(y) - mean(x). Returns (t, df, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se2 = vx + vy
    if se2 == 0:
        return float("inf"), float("nan"), 0.0
    t = (y.mean() - x.mean()) / math.sqrt(se2)
    df = se2 ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return t, df, betainc(df / 2.0, 0.5, df / (df + t * t))


def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pooled = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    return (y.mean() - x.mean()) / pooled if pooled > 0 else float("inf")


def mcnemar_exact(a_ok, b_ok):
    """Exact two-sided McNemar. Returns (b_right_a_wrong, a_right_b_wrong, p)."""
    gain = int(np.sum(~a_ok & b_ok))
    loss = int(np.sum(a_ok & ~b_ok))
    n = gain + loss
    if n == 0:
        return gain, loss, 1.0
    tail = sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / 2.0 ** n
    return gain, loss, min(1.0, 2.0 * tail)


def paired_bootstrap(score_a, score_b, n_boot, seed=0):
    """Mean of (score_b - score_a) over tracks, in points, with a 95% percentile CI."""
    d = (np.asarray(score_b, float) - np.asarray(score_a, float)) * 100.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):             # chunks keep memory small
        stop = min(n_boot, start + 1000)
        idx = rng.integers(0, len(d), size=(stop - start, len(d)))
        means[start:stop] = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- scoring
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_loader(split_file, grayscale, batch_size):
    cfg = Config()
    before = sha256(split_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = MultiFrameDataset(root_dir=cfg.DATA_ROOT, mode="val", split_ratio=0.9,
                               char2idx=cfg.CHAR2IDX, val_split_file=split_file,
                               grayscale=grayscale)
    with open(split_file) as fh:
        expected = set(json.load(fh))
    got = {s["track_id"] for s in ds.samples}
    # The split loader rewrites an invalid split file; make sure that never happened
    if sha256(split_file) != before or got != expected:
        sys.exit(f"❌ {split_file} không được nạp đúng (file bị đổi hoặc thiếu track)")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=MultiFrameDataset.collate_fn, num_workers=0)
    return ds, loader


@torch.no_grad()
def score(model, loader, frames=None):
    """Per-track (correct, confidence), in loader order. frames = indices of frames to keep."""
    cfg = Config()
    ok, conf = [], []
    for images, _, _, labels, _, *_ in loader:
        if frames is not None:
            images = images[:, list(frames)]
        lp = model(images.to(DEVICE)).float().cpu()
        for (text, c), gt in zip(decode_with_confidence(lp, cfg.IDX2CHAR), labels):
            ok.append(text == gt)
            conf.append(c)
    return np.array(ok), np.array(conf)


def load(path):
    model = ResTranOCR(num_classes=Config().NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return model.eval()


def main():
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--split-file", default="splits/lockbox_tracks.json")
    ap.add_argument("--baseline", default="restran_lockbox_base",
                    help="Run-name prefix of the baseline arm (runs are {prefix}_s{seed})")
    ap.add_argument("--method", default="restran_lockbox_combo3",
                    help="Run-name prefix of the method arm (luma input)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 7, 2026])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--last-k", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--frames-analysis", action="store_true", help="Also run E3 (N = 1..5 frames)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="results/lockbox_eval.json")
    args = ap.parse_args()
    os.chdir(REPO)
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    arms = {"baseline": (args.baseline, False), "method": (args.method, True)}
    epochs = list(range(args.epochs - args.last_k + 1, args.epochs + 1))
    missing = [p for prefix, _ in arms.values() for s in args.seeds
               for p in [os.path.join(args.results_dir, f"{prefix}_s{s}_best.pth")] +
               [os.path.join(args.results_dir, f"{prefix}_s{s}_ep{e:02d}.pth") for e in epochs]
               if not os.path.exists(p)]
    if missing:
        sys.exit("❌ Thiếu checkpoint:\n  " + "\n  ".join(missing))

    report = {"split_file": args.split_file, "split_sha256": sha256(args.split_file),
              "seeds": args.seeds, "epochs_last_k": epochs, "device": str(DEVICE), "arms": {}}
    per_track = {}          # arm -> metric -> [seeds, tracks] correctness
    order = {}              # arm -> track ids in scoring order (must match across arms)
    frames_track = {}       # arm -> N -> [seeds, tracks] accuracy over subsets
    for arm, (prefix, gray) in arms.items():
        ds, loader = build_loader(args.split_file, gray, args.batch_size)
        order[arm] = [s_["track_id"] for s_ in ds.samples]
        n_frames = 5
        last, last6, best, conf_last = [], [], [], []
        frames_track[arm] = {n: [] for n in range(1, n_frames + 1)}
        for s in args.seeds:
            run = f"{prefix}_s{s}"
            eps = [score(load(os.path.join(args.results_dir, f"{run}_ep{e:02d}.pth")), loader)
                   for e in epochs]
            last.append(eps[-1][0]); conf_last.append(eps[-1][1])
            last6.append(np.mean([ok for ok, _ in eps], axis=0))
            best.append(score(load(os.path.join(args.results_dir, f"{run}_best.pth")), loader)[0])
            print(f"  {arm:8s} seed {s:>4}: cuối {100 * eps[-1][0].mean():6.2f} | "
                  f"Last-6 {100 * last6[-1].mean():6.2f} | best {100 * best[-1].mean():6.2f}", flush=True)
            if args.frames_analysis:
                model = load(os.path.join(args.results_dir, f"{run}_ep{epochs[-1]:02d}.pth"))
                for n in range(1, n_frames + 1):
                    subsets = list(itertools.combinations(range(n_frames), n))
                    acc = np.mean([score(model, loader, sub)[0] for sub in subsets], axis=0)
                    frames_track[arm][n].append(acc)
        per_track[arm] = {"last": np.array(last), "last6": np.array(last6), "best": np.array(best)}
        conf = np.array(conf_last); oks = np.array(last)
        report["arms"][arm] = {
            "prefix": prefix, "grayscale": gray,
            "per_seed": {m: [round(100 * float(v.mean()), 2) for v in per_track[arm][m]]
                         for m in ("last", "last6", "best")},
            "confidence_gap_last": [round(float(c[o].mean() - c[~o].mean()), 4)
                                    for c, o in zip(conf, oks)],
        }
        layouts = np.array([ds.layouts.get(s_["track_id"], "Unknown") for s_ in ds.samples])

    if order["baseline"] != order["method"]:
        sys.exit("❌ Thứ tự track khác nhau giữa hai nhánh — so sánh theo cặp sẽ sai")

    report["comparison"] = {}
    print("\n=== Phương pháp − baseline trên lock-box ===")
    for m, label in (("last", "Epoch cuối (chính)"), ("last6", "Last-6 (quyết định)"),
                     ("best", "Best theo val 999")):
        a = 100 * per_track["baseline"][m].mean(axis=1)
        b = 100 * per_track["method"][m].mean(axis=1)
        t, df, p = welch(a, b)
        diff, lo, hi = paired_bootstrap(per_track["baseline"][m].mean(axis=0),
                                        per_track["method"][m].mean(axis=0), args.bootstrap)
        report["comparison"][m] = {
            "baseline_mean": round(float(a.mean()), 2), "baseline_std": round(float(a.std(ddof=1)), 2),
            "method_mean": round(float(b.mean()), 2), "method_std": round(float(b.std(ddof=1)), 2),
            "diff": round(diff, 2), "bootstrap_ci95": [round(lo, 2), round(hi, 2)],
            "welch_t": round(t, 3), "welch_df": round(df, 2), "welch_p": round(p, 4),
            "cohens_d": round(cohens_d(a, b), 2)}
        print(f"  {label:20s} baseline {a.mean():6.2f} ± {a.std(ddof=1):.2f} | phương pháp "
              f"{b.mean():6.2f} ± {b.std(ddof=1):.2f} | chênh {diff:+.2f} [CI95 {lo:+.2f}, {hi:+.2f}] "
              f"| Welch p = {p:.4f} | d = {cohens_d(a, b):.2f}")

    report["mcnemar_last"] = {}
    for i, s in enumerate(args.seeds):
        gain, loss, p = mcnemar_exact(per_track["baseline"]["last"][i].astype(bool),
                                      per_track["method"]["last"][i].astype(bool))
        report["mcnemar_last"][str(s)] = {"method_right_baseline_wrong": gain,
                                         "baseline_right_method_wrong": loss, "p": round(p, 4)}
        print(f"  McNemar seed {s:>4} (epoch cuối): +{gain} / -{loss}, p = {p:.4f}")

    report["by_layout_last"] = {}
    for lay in sorted(set(layouts)):
        mask = layouts == lay
        report["by_layout_last"][lay] = {
            "n": int(mask.sum()),
            "baseline": round(100 * float(per_track["baseline"]["last"][:, mask].mean()), 2),
            "method": round(100 * float(per_track["method"]["last"][:, mask].mean()), 2)}

    if args.frames_analysis:
        report["frames"] = {}
        print("\n=== E3: accuracy theo số frame (epoch cuối, trung bình mọi tập con N frame) ===")
        for n in range(1, 6):
            a = np.array(frames_track["baseline"][n]); b = np.array(frames_track["method"][n])
            diff, lo, hi = paired_bootstrap(a.mean(axis=0), b.mean(axis=0), args.bootstrap)
            report["frames"][str(n)] = {"baseline": round(100 * float(a.mean()), 2),
                                        "method": round(100 * float(b.mean()), 2),
                                        "diff": round(diff, 2), "bootstrap_ci95": [round(lo, 2), round(hi, 2)]}
            print(f"  N = {n}: baseline {100 * a.mean():6.2f} | phương pháp {100 * b.mean():6.2f} | "
                  f"chênh {diff:+.2f} [CI95 {lo:+.2f}, {hi:+.2f}]")

    dec = report["comparison"]["last6"]
    report["decision"] = "(b)" if dec["diff"] >= 1.0 and dec["bootstrap_ci95"][0] > 0 else "(a)"
    print(f"\nQuy tắc đã đăng ký trước: chênh Last-6 ≥ +1,0 và CI95 bootstrap không chứa 0 "
          f"→ (b), ngược lại → (a).  Kết quả: {report['decision']}")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_per_track.npz",
                        **{f"{arm}_{m}": v for arm in per_track for m, v in per_track[arm].items()},
                        layouts=layouts)
    print(f"📝 {args.out}")


DEVICE = torch.device("cpu")

if __name__ == "__main__":
    main()
