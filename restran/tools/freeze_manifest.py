#!/usr/bin/env python3
"""Freeze manifest: SHA-256 of code, splits and checkpoints, written BEFORE the lock-box is scored.

Anyone can later re-hash the same files and confirm that the scored checkpoints, the split
and the code are exactly the ones listed here (documents/lockbox_protocol.md, step 4).

    python tools/freeze_manifest.py --out results/freeze_manifest.json \
        --checkpoints "results/restran_lockbox_*_s*_ep*.pth" "results/restran_lockbox_*_best.pth"
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import socket
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_PATTERNS = ["train.py", "configs/*.py", "src/**/*.py", "tools/*.py", "tests/*.py"]
SPLIT_PATTERNS = ["splits/*.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True) if os.path.isfile(f)})
    return {f: {"sha256": sha256(f), "bytes": os.path.getsize(f)} for f in files}


def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "host": socket.gethostname()}
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as exc:  # the manifest must still be written without torch
        env["torch"] = f"unavailable: {exc}"
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/freeze_manifest.json")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="Glob patterns (quote them)")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    args = ap.parse_args()
    os.chdir(REPO)
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"❌ {args.out} đã tồn tại — manifest chỉ ghi một lần (dùng --force nếu thật sự cần)")

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "environment": environment(),
        "code": collect(CODE_PATTERNS),
        "splits": collect(SPLIT_PATTERNS),
        "checkpoints": collect(args.checkpoints),
    }
    if not manifest["checkpoints"]:
        sys.exit("❌ Không có checkpoint nào khớp các pattern đã cho")
    # One digest over everything, convenient to quote in the thesis
    flat = "".join(f"{k}:{v['sha256']}" for sec in ("code", "splits", "checkpoints")
                   for k, v in sorted(manifest[sec].items()))
    manifest["digest"] = hashlib.sha256(flat.encode()).hexdigest()
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"📌 {args.out}: {len(manifest['code'])} file code, {len(manifest['splits'])} split, "
          f"{len(manifest['checkpoints'])} checkpoint | digest {manifest['digest']}")


if __name__ == "__main__":
    main()
