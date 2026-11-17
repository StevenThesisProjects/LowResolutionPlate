#!/usr/bin/env python3
"""Score several trained arms on the same tracks and compare each one with a reference arm.

Used for the descriptive follow-up experiments of the review (N1: how the frame masks of a
track relate across frames; E4: learned STN vs one fixed crop vs no STN). They are trained
under the lock-box protocol, so the 8 lock-box runs serve as reference arms, and they are
scored after the one-shot lock-box evaluation: nothing here changes the frozen method.

Arm spec  NAME:PREFIX[:gray][:SEEDS]   runs are {PREFIX}_s{seed}, with checkpoints
{run}_ep{NN}.pth (last K epochs) and {run}_best.pth. Add :gray for arms trained with
--grayscale, and :42,43,7,2026 to use other seeds than --seeds for that arm (e.g. all 4
lock-box seeds for a reference arm). Whether a checkpoint has a learned STN, a fixed affine
(--fixed-stn) or neither is read from its keys.

Per arm and seed: last epoch, Last-K (mean of the last K epoch checkpoints) and best (chosen
on val 999, hence optimistic when val 999 itself is scored). Each arm is compared with --ref
(and any --pairs) by a paired bootstrap over tracks of the seed-averaged per-track accuracy,
plus Welch t-test and Cohen's d across seeds. --frames-analysis adds accuracy with N = 1..5
frames (mean over every subset of N frames) at the last epoch.

Run it on the machine that scored the lock-box, so the reference arms reproduce
results/lockbox_eval.json exactly: on other hardware a few near-tie predictions flip (3 of
28,000 on Apple GPU and CPU vs the V100, all with a top-2 log-prob gap <= 0.008).

    python tools/eval_arms.py --splits splits/lockbox_tracks.json splits/val_tracks.json \\
        --arm combo3:restran_lockbox_combo3:gray --arm nostn:restran_lockbox_e4_nostn:gray \\
        --ref combo3 --out results/arms_eval.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import eval_lockbox as ev  # noqa: E402  (statistics, split loading and scoring)
from configs.config import Config  # noqa: E402
from src.models.restran import ResTranOCR  # noqa: E402

METRICS = (("last", "cuối"), ("lastk", "Last-K"), ("best", "best"))


def parse_arm(spec, default_seeds):
    parts = spec.split(":")
    gray, seeds = False, list(default_seeds)
    for extra in parts[2:]:
        if extra == "gray" and not gray:
            gray = True
        elif extra and all(x.isdigit() for x in extra.split(",")):
            seeds = [int(x) for x in extra.split(",")]
        else:
            parts = []
            break
    if len(parts) < 2 or len(parts) > 4 or not all(parts[:2]):
        sys.exit(f"❌ --arm phải có dạng NAME:PREFIX[:gray][:42,43,7], nhận được {spec!r}")
    return parts[0], parts[1], gray, seeds


def load_auto(path, device):
    """Rebuild the model a checkpoint was trained with (learned STN / fixed affine / none)."""
    sd = torch.load(path, map_location="cpu")
    fixed = sd.get("fixed_theta")
    use_stn = any(k.startswith("stn.") for k in sd)
    model = ResTranOCR(num_classes=Config().NUM_CLASSES, use_stn=use_stn,
                       fixed_stn=None if fixed is None else fixed.flatten().tolist())
    model.load_state_dict(sd, strict=True)
    kind = "fixed" if fixed is not None else "stn" if use_stn else "none"
    return model.to(device).eval(), kind


def cache_split(split_file, gray, batch_size, device):
    """Load a split once (checked against its file by ev.build_loader) and keep it on device."""
    ds, loader = ev.build_loader(split_file, gray, batch_size)
    batches = [(images.to(device), None, None, labels, None)
               for images, _, _, labels, *_ in loader]
    tracks = [s["track_id"] for s in ds.samples]
    return batches, tracks


def compare(a_tracks, b_tracks, n_boot):
    """b - a. Inputs are [seeds, tracks] correctness; returns a dict of statistics."""
    a = 100 * a_tracks.mean(axis=1)
    b = 100 * b_tracks.mean(axis=1)
    diff, lo, hi = ev.paired_bootstrap(a_tracks.mean(axis=0), b_tracks.mean(axis=0), n_boot)
    out = {"a_mean": round(float(a.mean()), 2), "b_mean": round(float(b.mean()), 2),
           "diff": round(diff, 2), "bootstrap_ci95": [round(lo, 2), round(hi, 2)]}
    if len(a) > 1 and len(b) > 1:
        t, df, p = ev.welch(a, b)
        out.update(welch_p=round(p, 4), cohens_d=round(ev.cohens_d(a, b), 2))
    return out


def fmt_cmp(c):
    s = f"{c['diff']:+.2f} [{c['bootstrap_ci95'][0]:+.2f}, {c['bootstrap_ci95'][1]:+.2f}]"
    if "welch_p" in c:
        s += f" p={c['welch_p']:.3f} d={c['cohens_d']:+.2f}"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME:PREFIX[:gray], repeatable")
    ap.add_argument("--ref", required=True, help="NAME of the arm every other arm is compared with")
    ap.add_argument("--pairs", nargs="*", default=[], help="Extra comparisons A:B (reported as B - A)")
    ap.add_argument("--splits", nargs="+", default=["splits/lockbox_tracks.json", "splits/val_tracks.json"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 7])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--last-k", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--frames-analysis", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Drop arms with missing checkpoints instead of stopping (runs still in progress)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="results/arms_eval.json")
    args = ap.parse_args()
    os.chdir(REPO)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ev.DEVICE = device

    arms, arm_seeds = {}, {}
    for spec in args.arm:
        name, prefix, gray, seeds = parse_arm(spec, args.seeds)
        if name in arms:
            sys.exit(f"❌ Tên nhánh bị trùng: {name}")
        arms[name] = (prefix, gray)
        arm_seeds[name] = seeds
    pairs = [tuple(p.split(":")) for p in args.pairs]
    for p in pairs:
        if len(p) != 2 or not all(x in arms for x in p):
            sys.exit(f"❌ --pairs {':'.join(p)}: cần hai tên nhánh có trong --arm")
    if args.ref not in arms:
        sys.exit(f"❌ --ref {args.ref} không có trong --arm")

    epochs = list(range(args.epochs - args.last_k + 1, args.epochs + 1))

    def ckpts(prefix, seed):
        run = os.path.join(args.results_dir, f"{prefix}_s{seed}")
        return [f"{run}_ep{e:02d}.pth" for e in epochs], f"{run}_best.pth"

    for name in list(arms):
        missing = [p for s in arm_seeds[name] for eps, best in [ckpts(arms[name][0], s)]
                   for p in eps + [best] if not os.path.exists(p)]
        if missing:
            msg = f"Nhánh {name}: thiếu {len(missing)} checkpoint, ví dụ {missing[0]}"
            if not args.allow_missing or name == args.ref:
                sys.exit("❌ " + msg)
            print("⚠️  " + msg + " — bỏ nhánh này")
            del arms[name]
    pairs = [p for p in pairs if all(x in arms for x in p)]

    report = {"epochs_last_k": epochs, "device": str(device), "ref": args.ref,
              "arms": {n: {"prefix": p, "grayscale": g, "seeds": arm_seeds[n]}
                       for n, (p, g) in arms.items()},
              "splits": {}}
    npz = {}
    for split in args.splits:
        print(f"\n=== {split} ===", flush=True)
        caches, order = {}, None
        for gray in sorted({g for _, g in arms.values()}):
            caches[gray], tracks = cache_split(split, gray, args.batch_size, device)
            if order is not None and tracks != order:
                sys.exit("❌ Thứ tự track khác nhau giữa bản RGB và bản xám")
            order = tracks
        per_track = {}                                    # arm -> metric -> [seeds, tracks]
        frames = {}                                       # arm -> N -> [seeds, tracks]
        rep = {"split_sha256": ev.sha256(split), "n_tracks": len(order), "arms": {}, "vs_ref": {},
               "pairs": {}}
        for name, (prefix, gray) in arms.items():
            batches = caches[gray]
            rows = {m: [] for m, _ in METRICS}
            frames[name] = {n: [] for n in range(1, 6)}
            kinds = set()
            for s in arm_seeds[name]:
                eps, best = ckpts(prefix, s)
                oks = []
                for path in eps:
                    model, kind = load_auto(path, device)
                    kinds.add(kind)
                    oks.append(ev.score(model, batches)[0])
                rows["last"].append(oks[-1])
                rows["lastk"].append(np.mean(oks, axis=0))
                model, kind = load_auto(best, device)
                kinds.add(kind)
                rows["best"].append(ev.score(model, batches)[0])
                if args.frames_analysis:
                    model, _ = load_auto(eps[-1], device)
                    for n in range(1, 6):
                        subs = list(itertools.combinations(range(5), n))
                        frames[name][n].append(np.mean([ev.score(model, batches, sub)[0]
                                                        for sub in subs], axis=0))
            if len(kinds) != 1:
                sys.exit(f"❌ Nhánh {name} trộn nhiều kiến trúc: {sorted(kinds)}")
            per_track[name] = {m: np.array(v, dtype=float) for m, v in rows.items()}
            rep["arms"][name] = {"stn": kinds.pop(), "per_seed": {
                m: [round(100 * float(v.mean()), 2) for v in per_track[name][m]] for m, _ in METRICS}}
            cells = []
            for m, label in METRICS:
                v = rep["arms"][name]["per_seed"][m]
                cells.append(f"{label} {np.mean(v):6.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}")
            print(f"  {name:10s} [{rep['arms'][name]['stn']:5s}] " + " | ".join(cells)
                  + f"   (seed {','.join(map(str, arm_seeds[name]))} → cuối "
                  f"{rep['arms'][name]['per_seed']['last']})", flush=True)

        comparisons = [(args.ref, n) for n in arms if n != args.ref] + pairs
        print(f"  --- chênh lệch (B − A), CI 95% bootstrap theo track, Welch p và d theo seed ---")
        for a, b in comparisons:
            res = {m: compare(per_track[a][m], per_track[b][m], args.bootstrap) for m, _ in METRICS}
            if args.frames_analysis:
                res["frames"] = {str(n): compare(np.array(frames[a][n]), np.array(frames[b][n]),
                                                 args.bootstrap) for n in range(1, 6)}
            (rep["vs_ref"] if a == args.ref and (a, b) not in pairs else rep["pairs"])[f"{b}-{a}"] = res
            print(f"  {b} − {a}: " + " | ".join(f"{label} {fmt_cmp(res[m])}" for m, label in METRICS))
            if args.frames_analysis:
                print("      theo N frame: " + "  ".join(
                    f"N={n} {res['frames'][str(n)]['diff']:+.2f}" for n in range(1, 6)))
        if args.frames_analysis:
            rep["frames"] = {name: {str(n): round(100 * float(np.mean(frames[name][n])), 2)
                                    for n in range(1, 6)} for name in arms}
            print("  accuracy theo N frame (epoch cuối): ")
            for name in arms:
                print(f"    {name:10s} " + "  ".join(f"N={n} {rep['frames'][name][str(n)]:6.2f}"
                                                    for n in range(1, 6)))
        if os.path.abspath(split) == os.path.abspath("splits/val_tracks.json"):
            print("  (lưu ý: 'best' được chọn trên chính tập này nên lạc quan)")
        report["splits"][split] = rep
        tag = os.path.splitext(os.path.basename(split))[0]
        npz.update({f"{tag}__{arm}__{m}": v for arm in per_track for m, v in per_track[arm].items()})
        npz[f"{tag}__track_ids"] = np.array(order)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_per_track.npz", **npz)
    print(f"\n📝 {args.out}")


if __name__ == "__main__":
    main()
