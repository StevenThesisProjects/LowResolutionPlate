#!/usr/bin/env python3
"""Regression check (review ISSUE #9 / A5): with no new flag set, this code must train exactly
like the original code.

Each codebase runs ITS OWN train.py for 2 epochs on CPU, on the same tiny dataset, with
--aug-level light (resize + normalise only) and seed 42. albumentations 2.x draws its random
numbers from a private generator that the global seeds do not reach, so the synthetic-LR
degradation would differ from run to run even for identical code; for this check only,
every albumentations Compose is created with seed=0 (a launcher patches the default before
train.py starts). Both runs create their own val split inside a temporary copy of the code,
so nothing in the real repository (in particular splits/) is touched. The logged losses
must match.

    python tools/regression_check.py run --code-dir .                --out reg_branch.json
    python tools/regression_check.py run --code-dir /path/to/main    --out reg_main.json
    python tools/regression_check.py compare reg_branch.json reg_main.json
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


# Runs train.py with every albumentations Compose seeded (unless the code sets a seed itself)
LAUNCHER = """import runpy, sys
import albumentations as A
_init = A.Compose.__init__
def _seeded(self, *args, **kwargs):
    kwargs.setdefault("seed", 0)
    _init(self, *args, **kwargs)
A.Compose.__init__ = _seeded
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path("train.py", run_name="__main__")
"""


def make_tiny_data(data_root, dest, n_tracks):
    tracks = sorted(glob.glob(os.path.join(os.path.abspath(data_root), "**", "track_*"), recursive=True))
    picked = ([t for t in tracks if "Scenario-A" in t][: n_tracks // 2] +
              [t for t in tracks if "Scenario-B" in t][: n_tracks - n_tracks // 2])
    for t in picked:
        scenario = "Scenario-B" if "Scenario-B" in t else "Scenario-A"
        out = os.path.join(dest, scenario, os.path.basename(t))
        os.makedirs(out)
        for f in glob.glob(os.path.join(t, "lr-*")) + glob.glob(os.path.join(t, "hr-*")) + \
                [os.path.join(t, "annotations.json")]:
            os.symlink(f, os.path.join(out, os.path.basename(f)))
    return len(picked)


def run(args):
    tmp = tempfile.mkdtemp(prefix="regcheck_")
    try:
        n = make_tiny_data(args.data_root, os.path.join(tmp, "data"), args.tracks)
        code = os.path.join(tmp, "code")
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        os.makedirs(code)
        shutil.copy2(os.path.join(args.code_dir, "train.py"), code)
        for sub in ("configs", "src"):
            shutil.copytree(os.path.join(args.code_dir, sub), os.path.join(code, sub), ignore=ignore)
        with open(os.path.join(code, "_regrun.py"), "w") as fh:
            fh.write(LAUNCHER)
        cmd = [sys.executable, "_regrun.py", "-n", "regcheck", "--data-root", os.path.join(tmp, "data"),
               "--epochs", str(args.epochs), "--batch-size", "8", "--num-workers", "0",
               "--aug-level", "light", "--output-dir", os.path.join(tmp, "out"), "--seed", "42"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        proc = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True, timeout=3600)
        log = proc.stdout + proc.stderr
        # Only the per-epoch lines: the two codebases word their closing summary differently
        epoch_lines = [ln for ln in log.splitlines() if re.match(r"\s*Epoch \d+/\d+:", ln)]
        grab = lambda pat: [float(m.group(1)) for ln in epoch_lines for m in [re.search(pat, ln)] if m]
        result = {
            "code_dir": os.path.abspath(args.code_dir), "tracks": n, "epochs": args.epochs,
            "returncode": proc.returncode,
            "train_loss": grab(r"Train Loss: ([0-9.]+)"),
            "val_loss": grab(r"Val Loss: ([0-9.]+)"),
            "val_acc": grab(r"Val Acc: ([0-9.]+)%"),
            "log_tail": log.splitlines()[-25:],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"{args.code_dir}: returncode {result['returncode']} | train_loss {result['train_loss']} | "
          f"val_loss {result['val_loss']} | val_acc {result['val_acc']} -> {args.out}")
    if result["returncode"] != 0 or len(result["train_loss"]) != args.epochs:
        print("\n".join(result["log_tail"]))
        sys.exit(1)


def compare(args):
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    ok = True
    for key in ("train_loss", "val_loss", "val_acc"):
        same = len(a[key]) == len(b[key]) and all(abs(x - y) <= 1e-4 for x, y in zip(a[key], b[key]))
        ok &= same
        print(f"{'OK  ' if same else 'KHÁC'} {key:10s} {a[key]}  vs  {b[key]}")
    print("✅ PASS: hai bản code train giống hệt nhau" if ok else "❌ FAIL: hành vi khác nhau")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--code-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data-root", default="dataset/data/train")
    r.add_argument("--tracks", type=int, default=64)
    r.add_argument("--epochs", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
