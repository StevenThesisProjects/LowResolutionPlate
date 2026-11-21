#!/usr/bin/env python3
"""Run checkpoints over a split with the OFFLINE pipeline and cache the log-probabilities.

Offline = the training repository's own data path (MultiFrameDataset in val mode, batch
64), i.e. how every number in the paper was produced. The other analysis scripts read
these caches, and scripts/parity_check.py compares them with what the API returns.

    # the four recipe seeds on the 999-track val (decoders, syntax audit, calibration)
    python3 scripts/offline_cache.py --split val \
        --runs combo3_s42 combo3_s43 combo3_s7 combo3_s2026
    # every subset of N < 5 frames for the served model (thresholds per frame count)
    python3 scripts/offline_cache.py --split val --runs combo3_s42 --subsets

Output: var/cache/{split}__meta.npz (track ids, labels, layouts,
per-frame sharpness) and {split}__{run}_ep30.npz (log-probs [tracks, 16, 37]).
"""
import argparse
import contextlib
import glob
import hashlib
import io
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.settings import REPO_ROOT, THESIS_DIR  # noqa: E402
from mfplate.engine import build_model  # noqa: E402
from mfplate.preprocess import sharpness  # noqa: E402

from configs.config import Config  # noqa: E402
from src.data.dataset import MultiFrameDataset  # noqa: E402

SPLITS = {"val": "splits/val_tracks.json", "lockbox": "splits/lockbox_tracks.json"}
CACHE = THESIS_DIR / "var" / "cache"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loader_for(split_file: str, grayscale: bool, batch_size: int = 64):
    cfg = Config()
    before = sha256(REPO_ROOT / split_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = MultiFrameDataset(root_dir=str(REPO_ROOT / cfg.DATA_ROOT), mode="val",
                               split_ratio=0.9, char2idx=cfg.CHAR2IDX,
                               val_split_file=str(REPO_ROOT / split_file), grayscale=grayscale)
    expected = set(json.loads((REPO_ROOT / split_file).read_text()))
    got = {s["track_id"] for s in ds.samples}
    if sha256(REPO_ROOT / split_file) != before or got != expected:
        sys.exit(f"❌ {split_file} không được nạp đúng")
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     collate_fn=MultiFrameDataset.collate_fn, num_workers=0)
    return ds, dl


def run_path(run: str) -> Path:
    return REPO_ROOT / "results_new" / f"restran_lockbox_{run}_ep30.pth"


@torch.no_grad()
def logprobs(model, dl, frames=None, device="cpu"):
    out = []
    for images, *_ in dl:
        if frames is not None:
            images = images[:, list(frames)]
        out.append(model(images.to(device)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def write_meta(split: str, ds) -> Path:
    path = CACHE / f"{split}__meta.npz"
    if path.exists():
        return path
    ids, labels, layouts, sharp, sizes = [], [], [], [], []
    for s in ds.samples:
        ids.append(s["track_id"]); labels.append(s["label"]); layouts.append(s["layout"])
        row, size = [], []
        for p in s["paths"]:
            import cv2
            rgb = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            row.append(sharpness(rgb)); size.append(rgb.shape[:2])
        sharp.append(row); sizes.append(size)
    np.savez_compressed(path, track_ids=np.array(ids), labels=np.array(labels),
                        layouts=np.array(layouts), sharpness=np.array(sharp, dtype=np.float64),
                        sizes=np.array(sizes), paths=np.array([s["paths"] for s in ds.samples]))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=SPLITS, default="val")
    ap.add_argument("--runs", nargs="+", default=["combo3_s42"],
                    help="combo3_s{seed} (recipe, luma) or base_s{seed} (baseline, RGB)")
    ap.add_argument("--subsets", action="store_true", help="also every subset of 1-4 frames")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)
    split_file = SPLITS[args.split]
    loaders = {}
    for run in args.runs:
        gray = run.startswith("combo3")
        if gray not in loaders:
            loaders[gray] = loader_for(split_file, gray)
        ds, dl = loaders[gray]
        meta = write_meta(args.split, ds)
        out = CACHE / f"{args.split}__{run}_ep30.npz"
        model = build_model(run_path(run), args.device)
        if not out.exists() or args.force:
            t0 = time.time()
            lp = logprobs(model, dl, device=args.device)
            np.savez_compressed(out, log_probs=lp, checkpoint_sha256=sha256(run_path(run)),
                                split_sha256=sha256(REPO_ROOT / split_file), device=args.device)
            print(f"✅ {out.name}: {lp.shape} trong {time.time() - t0:.1f} s")
        if args.subsets:
            sub_out = CACHE / f"{args.split}__{run}_ep30__subsets.npz"
            if sub_out.exists() and not args.force:
                continue
            arrays = {}
            for n in range(1, 5):
                t0 = time.time()
                subs = list(itertools.combinations(range(5), n))
                arrays[f"N{n}"] = np.stack([logprobs(model, dl, s, args.device) for s in subs])
                arrays[f"N{n}_subsets"] = np.array(subs)
                print(f"  N = {n}: {len(subs)} tập con, {time.time() - t0:.1f} s")
            np.savez_compressed(sub_out, **arrays)
            print(f"✅ {sub_out.name}")
    print(f"📝 meta: {meta}")


if __name__ == "__main__":
    main()
