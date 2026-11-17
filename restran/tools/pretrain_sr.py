#!/usr/bin/env python3
"""Supervised pre-training of the SR front-end on ECC-registered LR->HR pairs.

Every earlier SR attempt trained the module jointly with the OCR from scratch, so it
only ever saw the CTC gradient — an indirect signal routed through a 31M-parameter
backbone. That is not how SR networks are trained anywhere in the literature: they are
fitted to registered pairs with a pixel loss first, then attached.

Measured headroom on this data (frames with registration NCC > 0.8):
    L1(LR, registered HR)                = 0.295   <- where the module starts
    L1 of a perfect-but-blurry SR        = 0.127   <- a realistic floor
so roughly 57% of the error is removable in principle.

Only frames whose registration converged are used; a misregistered target teaches
noise, which is precisely what made the first joint attempt look hopeless.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.sr import StackedSRNet          # noqa: E402
from src.models.edvr import EDVRLite            # noqa: E402


class RegisteredPairs(Dataset):
    """Tracks whose 5 LR frames all have a well-registered HR counterpart."""

    def __init__(self, data_root, registration, ncc_min, height, width, val_ids=None,
                 want_val=False):
        d = np.load(registration, allow_pickle=True)
        quality = {str(k): (w, float(q))
                   for k, w, q in zip(d['keys'], d['warps'], d['ncc_after'])}
        self.h, self.w = height, width
        self.items = []
        val_ids = val_ids or set()
        for track in sorted(glob.glob(os.path.join(data_root, "**", "track_*"), recursive=True)):
            tid = os.path.basename(track)
            if (tid in val_ids) != want_val:
                continue
            lr = sorted(glob.glob(os.path.join(track, "lr-*")))
            hr = sorted(glob.glob(os.path.join(track, "hr-*")))
            keep = [quality.get(f"{tid}/{i}") for i in range(len(lr))]
            if len(lr) != 5 or len(hr) != 5 or any(k is None for k in keep):
                continue
            if min(k[1] for k in keep) < ncc_min:   # every frame must be usable
                continue
            self.items.append((lr, hr, [k[0] for k in keep]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        lr_paths, hr_paths, warps = self.items[idx]
        size = (self.w, self.h)
        lrs, hrs = [], []
        for lp, hp, wm in zip(lr_paths, hr_paths, warps):
            lr = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
            hr = cv2.cvtColor(cv2.imread(hp), cv2.COLOR_BGR2RGB)
            lrs.append(cv2.resize(lr, size))
            hrs.append(cv2.warpAffine(cv2.resize(hr, size), wm, size))
        to_t = lambda a: torch.from_numpy(
            np.stack(a).astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2)
        return to_t(lrs), to_t(hrs)


def edge_loss(pred, target):
    """Gradient-domain L1: character strokes are edges, and plain L1 blurs them."""
    def grad(x):
        return (x[..., :, 1:] - x[..., :, :-1]).abs().mean() + \
               (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return (grad(pred) - grad(target)).abs() + F.l1_loss(
        pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="dataset/data/train")
    ap.add_argument("--registration", default="splits/registration.npz")
    ap.add_argument("--val-split-file", default="splits/val_tracks_2000.json")
    ap.add_argument("--ncc-min", type=float, default=0.8)
    ap.add_argument("--arch", choices=["stacked", "edvr"], default="stacked")
    ap.add_argument("--sr-features", type=int, default=32)
    ap.add_argument("--sr-blocks", type=int, default=8)
    ap.add_argument("--img-height", type=int, default=32)
    ap.add_argument("--img-width", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--edge-weight", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="results/sr_pretrained.pth")
    args = ap.parse_args()

    import json
    val_ids = set(json.load(open(args.val_split_file))) if os.path.exists(args.val_split_file) else set()
    common = dict(data_root=args.data_root, registration=args.registration,
                  ncc_min=args.ncc_min, height=args.img_height, width=args.img_width,
                  val_ids=val_ids)
    train_ds = RegisteredPairs(**common, want_val=False)
    val_ds = RegisteredPairs(**common, want_val=True)
    print(f"📐 Track dùng được (mọi frame NCC > {args.ncc_min}): "
          f"train {len(train_ds)}, val {len(val_ds)}")
    if len(train_ds) == 0:
        print("❌ Không có track nào đạt ngưỡng — hạ --ncc-min"); sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    make = EDVRLite if args.arch == 'edvr' else StackedSRNet
    model = make(num_features=args.sr_features, num_blocks=args.sr_blocks, scale=1).to(device)
    print(f"🧱 {args.arch}: {sum(p.numel() for p in model.parameters())/1e6:.3f}M tham số")

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=len(tl), epochs=args.epochs)

    def evaluate():
        model.eval(); l1s, psnrs, base = [], [], []
        with torch.no_grad():
            for lr, hr in vl:
                lr, hr = lr.to(device), hr.to(device)
                out = model(lr)
                l1s.append(F.l1_loss(out, hr).item())
                mse = F.mse_loss(out.clamp(-1, 1), hr).item()
                psnrs.append(10 * np.log10(4.0 / max(mse, 1e-9)))
                base.append(F.l1_loss(lr, hr).item())
        return float(np.mean(l1s)), float(np.mean(psnrs)), float(np.mean(base))

    l1, psnr, base = evaluate()
    print(f"Trước khi train : val L1 {l1:.4f} | PSNR {psnr:.2f} dB | L1 của LR thô {base:.4f}")
    best = float('inf')
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for lr, hr in tl:
            lr, hr = lr.to(device), hr.to(device)
            out = model(lr)
            loss = F.l1_loss(out, hr) + args.edge_weight * edge_loss(out, hr)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        l1, psnr, _ = evaluate()
        flag = ""
        if l1 < best:
            best = l1
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out); flag = " ⭐"
        print(f"Epoch {ep+1}/{args.epochs}: train {run/len(tl):.4f} | "
              f"val L1 {l1:.4f} | PSNR {psnr:.2f} dB{flag}", flush=True)

    print(f"\n✅ L1 tốt nhất {best:.4f} (LR thô {base:.4f}, sàn lý thuyết ~0.127) -> {args.out}")


if __name__ == "__main__":
    main()
