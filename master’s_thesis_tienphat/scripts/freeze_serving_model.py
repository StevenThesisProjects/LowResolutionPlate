#!/usr/bin/env python3
"""Pick the checkpoint the app serves and write artifacts/serving_config.json.

Rule, fixed before any app measurement: the recipe (configuration #50, "combo3") trained
for the lock-box with seed 42, LAST epoch (ep30).
  * recipe, not baseline: it is the configuration frozen and confirmed on the lock-box;
  * last epoch: the primary metric of the lock-box protocol, no epoch is chosen;
  * seed 42: the first seed of the protocol. Choosing the seed with the best lock-box
    score (43, 80.30%) would be selection on the lock-box, so it is not done.
The model was trained on 18,001 tracks; the 999-track val and the 1,000-track lock-box
were never used to train it, so both can be used to measure the app.

The SHA-256 of the file must equal the one recorded in results_new/freeze_manifest.json
(written on 24/10/2026 before the lock-box was scored), otherwise the script stops.
The file is then copied (same name) into model/, which is where the app loads it from;
the copy is checked against the same SHA-256.

    python3 scripts/freeze_serving_model.py
"""
import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.settings import ARTIFACTS_DIR, MODEL_DIR, REPO_ROOT  # noqa: E402
from mfplate.engine import sha256_file, build_model  # noqa: E402

RUN = "restran_lockbox_combo3_s42"
EPOCH = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results_new")
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "serving_config.json"))
    args = ap.parse_args()

    ckpt_rel = f"{args.results_dir}/{RUN}_ep{EPOCH:02d}.pth"
    ckpt = REPO_ROOT / ckpt_rel
    manifest_path = REPO_ROOT / args.results_dir / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    key = f"results/{RUN}_ep{EPOCH:02d}.pth"          # path as written on the server
    expected = manifest["checkpoints"][key]["sha256"]
    actual = sha256_file(ckpt)
    if actual != expected:
        sys.exit(f"❌ {ckpt_rel}: SHA-256 {actual} khác manifest {expected}")

    model = build_model(ckpt)
    n_params = sum(p.numel() for p in model.parameters())

    lockbox = json.loads((REPO_ROOT / args.results_dir / "lockbox_eval.json").read_text())
    seeds = lockbox["seeds"]
    i = seeds.index(42)
    frames = {n: v["method"] for n, v in lockbox["frames"].items()}

    config = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model_version": f"restran-recipe-s42-ep30-{actual[:12]}",
        "checkpoint": {
            "path": ckpt_rel,
            "sha256": actual,
            "bytes": ckpt.stat().st_size,
            "manifest": f"{args.results_dir}/freeze_manifest.json",
            "manifest_key": key,
            "manifest_digest": manifest["digest"],
            "matches_manifest": True,
        },
        "selection_rule": ("recipe #50 (combo3), lock-box run, seed 42, last epoch (30): "
                           "fixed rule, not chosen by any lock-box score"),
        "training": {
            "run": RUN, "train_tracks": 18001, "train_samples": 36002,
            "excluded": ["splits/val_tracks.json (999)", "splits/lockbox_tracks.json (1000)"],
            "flags": ("--seed 42 --epochs 30 --lr 3e-4 --grad-clip 1.0 --skip-grad-norm 100 "
                      "--split-ratio 0.9 --val-split-file splits/val_tracks.json "
                      "--consistent-aug --no-channel-shuffle --degrade-strength calibrated "
                      "--aux-frame-ctc 0.2 --frame-mask 0.2 --frame-dropout 0.1 --grayscale "
                      "--exclude-tracks splits/lockbox_tracks.json --save-last-k 6"),
            "source": "documents/RUNS.md, section 'Lock-box: lệnh chạy'; results_new/"
                      f"{RUN}_train.log",
            "environment": manifest["environment"],
        },
        "architecture": {"name": "ResTranOCR (STN + ResNet34 + AttentionFusion + "
                                 "Transformer x3 + CTC)", "parameters": n_params,
                         "num_classes": 37, "time_steps": 16},
        "preprocessing": {"size": [32, 128], "resize": "cv2.INTER_LINEAR",
                          "normalize": "mean 0.5, std 0.5", "grayscale": True,
                          "luma": "BT.601 (0.299, 0.587, 0.114), replicated to 3 channels",
                          "frames": "1-5, fed directly (AttentionFusion accepts any count)"},
        "lockbox_reference": {
            "this_checkpoint_last_epoch_acc": lockbox["arms"]["method"]["per_seed"]["last"][i],
            "recipe_4_seeds_last_epoch_mean": lockbox["comparison"]["last"]["method_mean"],
            "baseline_4_seeds_last_epoch_mean": lockbox["comparison"]["last"]["baseline_mean"],
            "source": f"{args.results_dir}/lockbox_eval.json",
        },
        # E3 on the lock-box: recipe, last epoch, mean over 4 seeds and all subsets of N frames
        "expected_accuracy_by_frames": frames,
        "scope": ("Brazilian/Mercosur single-row plates, 7 characters (LRLPR-26). "
                  "Not for Vietnamese plates. Input = plate crops, no detector."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"✅ {ckpt_rel}  sha256 {actual}  ({n_params:,} tham số) khớp manifest")
    print(f"📝 {out}")

    # The app serves a copy in model/ (same name); copy only when missing or different
    served = MODEL_DIR / ckpt.name
    if not served.is_file() or sha256_file(served) != actual:
        served.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, served)
        if sha256_file(served) != actual:
            sys.exit(f"❌ bản chép {served} không khớp SHA-256")
        print(f"📦 đã chép checkpoint vào {served}")
    else:
        print(f"📦 {served} đã có, khớp SHA-256")


if __name__ == "__main__":
    main()
