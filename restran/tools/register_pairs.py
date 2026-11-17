#!/usr/bin/env python3
"""Register each LR frame to its HR counterpart with ECC, and cache the warps.

Why this exists: `lr-00i` and `hr-00i` are cropped from different regions of the
video (slide 14), so they are *not* registered — pairing by index scores the same
PSNR as pairing at random (12.75 dB vs 12.72 dB). Every supervised SR method assumes
registered pairs, so without this step a pixel-wise loss trains on mismatched images.

The `corners` annotations do not fix it: they are integer coordinates on ~40x18 px
crops, and warping with them actually *degrades* inter-frame alignment
(NCC 0.842 -> 0.760). Content-based ECC does work, and needs no annotations, so it
also covers Scenario-B where `corners` is absent.

Output: an .npz mapping "<track_id>/<frame_index>" to a 2x3 affine matrix plus the
post-registration NCC, so training can filter to well-registered pairs only.
"""
import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

import cv2
import numpy as np

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)


def _gray(path: str, size) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-9 else 0.0


def register_track(args):
    """Align every LR frame of one track into its HR frame's coordinates."""
    track, height, width = args
    size = (width, height)
    lr_files = sorted(glob.glob(os.path.join(track, "lr-*")))
    hr_files = sorted(glob.glob(os.path.join(track, "hr-*")))
    track_id = os.path.basename(track)
    out = []

    for i, (lr_path, hr_path) in enumerate(zip(lr_files, hr_files)):
        lr, hr = _gray(lr_path, size), _gray(hr_path, size)
        if lr is None or hr is None:
            continue
        # ECC needs comparable frequency content, so the sharp HR is blurred down
        # towards the LR before matching. The blurred copy is only the matching
        # template — quality is always scored against the original HR.
        hr_blur = cv2.GaussianBlur(hr, (5, 5), 1.2)
        before = _ncc(lr, hr)
        warp = np.eye(2, 3, dtype=np.float32)
        after = before
        try:
            _, warp = cv2.findTransformECC(
                hr_blur, lr, warp, cv2.MOTION_AFFINE, CRITERIA, None, 5
            )
            aligned = cv2.warpAffine(
                lr, warp, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            )
            after = _ncc(aligned, hr)
            if after < before:  # ECC converged to a worse solution; keep identity
                warp, after = np.eye(2, 3, dtype=np.float32), before
        except cv2.error:
            pass
        out.append((f"{track_id}/{i}", warp.astype(np.float32), before, after))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="dataset/data/train")
    ap.add_argument("--out", default="splits/registration.npz")
    ap.add_argument("--img-height", type=int, default=32)
    ap.add_argument("--img-width", type=int, default=128)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    tracks = sorted(glob.glob(os.path.join(args.data_root, "**", "track_*"), recursive=True))
    if not tracks:
        print(f"❌ Không tìm thấy track nào trong {args.data_root}")
        sys.exit(1)
    print(f"🔎 {len(tracks)} track, khung chuẩn {args.img_height}x{args.img_width}, "
          f"{args.workers} luồng")

    jobs = [(t, args.img_height, args.img_width) for t in tracks]
    keys, warps, before, after = [], [], [], []
    with Pool(args.workers) as pool:
        for n, rows in enumerate(pool.imap_unordered(register_track, jobs, chunksize=32), 1):
            for key, warp, b, a in rows:
                keys.append(key); warps.append(warp); before.append(b); after.append(a)
            if n % 2000 == 0:
                print(f"   {n}/{len(tracks)} track", flush=True)

    before, after = np.array(before, dtype=np.float32), np.array(after, dtype=np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        keys=np.array(keys), warps=np.stack(warps),
        ncc_before=before, ncc_after=after,
        canvas=np.array([args.img_height, args.img_width]),
    )

    print(f"\n✅ Đã lưu {len(keys)} cặp vào {args.out}")
    print(f"   NCC trung bình : {before.mean():.4f} -> {after.mean():.4f}")
    for thr in (0.7, 0.8, 0.9):
        print(f"   NCC > {thr}      : {(before > thr).mean()*100:5.1f}%  ->  "
              f"{(after > thr).mean()*100:5.1f}%   ({int((after > thr).sum())} cặp)")


if __name__ == "__main__":
    main()
