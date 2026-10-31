#!/usr/bin/env python3
"""Create the lock-box split: Scenario-B tracks never used to select a configuration.

Every configuration choice so far was made on the dev val set (splits/val_tracks_2000.json,
1,999 Scenario-B tracks, which contains the 999-track val of splits/val_tracks.json). The
lock-box is drawn from the remaining Scenario-B tracks, which have only ever been training
data, and is then held out of training for the final runs (see
documents/lockbox_protocol.md). The draw is deterministic: sorted track ids + a fixed seed.

    python tools/make_lockbox.py                  # writes splits/lockbox_tracks.json
    python tools/make_lockbox.py --verify         # recomputes and compares, writes nothing
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys


def draw(data_root, used_files, n, seed):
    tracks = glob.glob(os.path.join(os.path.abspath(data_root), "**", "track_*"), recursive=True)
    scenario_b = sorted(os.path.basename(t) for t in tracks if "Scenario-B" in t)
    used = set()
    for f in used_files:
        with open(f) as fh:
            used |= set(json.load(fh))
    pool = [t for t in scenario_b if t not in used]
    if len(pool) < n:
        sys.exit(f"❌ Chỉ còn {len(pool)} track Scenario-B chưa dùng, cần {n}")
    return sorted(random.Random(seed).sample(pool, n)), len(scenario_b), len(used), len(pool), tracks


def layout_counts(ids, tracks):
    where = {os.path.basename(t): t for t in tracks}
    counts = {}
    for tid in ids:
        with open(os.path.join(where[tid], "annotations.json")) as fh:
            ann = json.load(fh)
        ann = ann[0] if isinstance(ann, list) else ann
        layout = ann.get("plate_layout", "Unknown")
        counts[layout] = counts.get(layout, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="dataset/data/train")
    ap.add_argument("--used", nargs="+",
                    default=["splits/val_tracks_2000.json", "splits/val_tracks.json"],
                    help="Split files whose tracks were used for model selection")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="splits/lockbox_tracks.json")
    ap.add_argument("--verify", action="store_true",
                    help="Recompute the draw and check it matches --out; write nothing")
    args = ap.parse_args()

    ids, n_b, n_used, n_pool, tracks = draw(args.data_root, args.used, args.n, args.seed)
    payload = json.dumps(ids, indent=2).encode()

    if args.verify:
        with open(args.out, "rb") as fh:
            on_disk = fh.read()
        same = json.loads(on_disk) == ids
        print(f"{'✅ Khớp' if same else '❌ KHÁC'}: {args.out} "
              f"sha256 {hashlib.sha256(on_disk).hexdigest()}")
        sys.exit(0 if same else 1)

    if os.path.exists(args.out):
        sys.exit(f"❌ {args.out} đã tồn tại — không ghi đè lock-box. Dùng --verify để kiểm tra.")
    with open(args.out, "wb") as fh:
        fh.write(payload)
    print(f"Scenario-B: {n_b} track | đã dùng để chọn cấu hình: {n_used} | còn lại: {n_pool}")
    print(f"Lock-box: {len(ids)} track (seed {args.seed}) -> {args.out}")
    print(f"Theo layout: {layout_counts(ids, tracks)}")
    print(f"sha256: {hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
