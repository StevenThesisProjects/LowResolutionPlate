#!/usr/bin/env python3
"""Parity: does the app give the same answers as the offline evaluation?

Replaces roadmap weeks 3 and 7 (parity_baseline.json / parity_final.json, "<= 2 tracks
differ between app and offline on the validation split"). Three comparisons, all with the
served checkpoint (recipe, lock-box run seed 42, epoch 30) and the greedy decoder:

  1. API vs offline on val 999, same machine (CPU): every track is sent through the HTTP
     API (FastAPI TestClient: multipart upload, validation, decoding, JSON) and compared
     with the training repository's own pipeline (scripts/offline_cache.py). Plate and
     confidence must match for every track.
  2. API on CPU vs training server on GPU, val 999: the API's accuracy against the
     val_acc that the training run logged at epoch 30 (metrics.csv, V100, fp16 autocast).
  3. API on CPU vs lock-box scoring on GPU (--lockbox): per-track correctness against
     results_new/lockbox_eval_per_track.npz (seed 42, last epoch). The lock-box predictions
     were already scored on 25/10; this only re-reads them, nothing is chosen.

    python3 scripts/parity_check.py [--lockbox]
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.decoding import greedy  # noqa: E402
from mfplate.settings import ARTIFACTS_DIR, REPO_ROOT, THESIS_DIR  # noqa: E402

CACHE = THESIS_DIR / "var" / "cache"
RUN = "combo3_s42"


def post_track(client, paths):
    files = [("files", (Path(p).name, Path(p).read_bytes(), "image/jpeg")) for p in paths]
    r = client.post("/predict?decoder=greedy", files=files)
    if r.status_code != 200:
        raise RuntimeError(f"{paths[0]}: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def api_pass(client, split):
    meta = np.load(CACHE / f"{split}__meta.npz")
    out = []
    t0 = time.time()
    for i, (tid, paths) in enumerate(zip(meta["track_ids"], meta["paths"])):
        j = post_track(client, list(paths))
        out.append((str(tid), j["plate"], j["confidence"], j["char_conf"]))
        if (i + 1) % 200 == 0:
            print(f"  {split}: {i + 1}/{len(meta['track_ids'])} ({time.time() - t0:.0f} s)", flush=True)
    return meta, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lockbox", action="store_true")
    args = ap.parse_args()
    os.environ["MFPLATE_DB"] = str(Path(tempfile.mkdtemp()) / "parity.sqlite3")
    from fastapi.testclient import TestClient
    from mfplate.api import app

    report = {"created": dt.datetime.now().isoformat(timespec="seconds"),
              "machine": {"platform": platform.platform(), "python": platform.python_version()},
              "decoder": "greedy"}
    with TestClient(app) as client:
        report["model_version"] = client.get("/model").json()["model_version"]
        import torch
        report["machine"]["torch"] = torch.__version__

        # 1 + 2: val 999
        meta, api = api_pass(client, "val")
        lp = np.load(CACHE / f"val__{RUN}_ep30.npz")["log_probs"]
        labels = meta["labels"]
        mismatches, conf_diff, char_diff = [], 0.0, 0.0
        for (tid, plate, conf, cc), x, y in zip(api, lp, labels):
            ref_text, ref_cc = greedy(x)
            ref_conf = float(np.mean(ref_cc)) if ref_cc else 0.0
            if plate != ref_text:
                mismatches.append({"track": tid, "api": plate, "offline": ref_text, "label": str(y)})
            conf_diff = max(conf_diff, abs(conf - ref_conf))
            if len(cc) == len(ref_cc) and cc:
                char_diff = max(char_diff, float(np.max(np.abs(np.array(cc) - np.array(ref_cc)))))
        api_ok = np.array([p == y for (_, p, _, _), y in zip(api, labels)])
        metrics = list(csv.DictReader(open(REPO_ROOT / "results_new" /
                                           f"restran_lockbox_{RUN}_metrics.csv")))
        gpu_acc = float(next(r for r in metrics if r["epoch"] == "30")["val_acc"])
        report["val_api_vs_offline"] = {
            "split": "splits/val_tracks.json", "tracks": len(api),
            "plate_mismatches": len(mismatches), "mismatch_list": mismatches,
            "max_abs_confidence_diff": conf_diff, "max_abs_char_conf_diff": char_diff,
            "criterion": "<= 2 tracks differ (roadmap, section 4, criterion 1)",
            "passed": len(mismatches) <= 2}
        report["val_cpu_vs_gpu"] = {
            "api_cpu_accuracy": round(100 * api_ok.mean(), 4),
            "api_cpu_correct": int(api_ok.sum()),
            "gpu_val_acc_epoch30": gpu_acc,
            "gpu_correct_implied": int(round(gpu_acc * len(api) / 100)),
            "source": f"results_new/restran_lockbox_{RUN}_metrics.csv (epoch 30)",
            "same_accuracy": bool(abs(100 * api_ok.mean() - gpu_acc) < 0.01)}
        print(f"val 999: API≠offline ở {len(mismatches)} track; |Δconf| max {conf_diff:.2e}; "
              f"API CPU {100 * api_ok.mean():.2f}% vs GPU log {gpu_acc:.2f}%")

        # 3: lock-box against the GPU scoring of 25/10
        if args.lockbox:
            meta_lb, api_lb = api_pass(client, "lockbox")
            gpu = np.load(REPO_ROOT / "results_new" / "lockbox_eval_per_track.npz")
            lb = json.loads((REPO_ROOT / "results_new" / "lockbox_eval.json").read_text())
            seed_idx = lb["seeds"].index(42)
            gpu_ok = gpu["method_last"][seed_idx].astype(bool)
            cpu_ok = np.array([p == y for (_, p, _, _), y in zip(api_lb, meta_lb["labels"])])
            differ = np.nonzero(gpu_ok != cpu_ok)[0]
            report["lockbox_cpu_vs_gpu"] = {
                "tracks": len(api_lb),
                "api_cpu_accuracy": round(100 * cpu_ok.mean(), 2),
                "gpu_lockbox_accuracy": lb["arms"]["method"]["per_seed"]["last"][seed_idx],
                "tracks_with_different_correctness": int(len(differ)),
                "differing_tracks": [str(meta_lb["track_ids"][i]) for i in differ],
                "source": "results_new/lockbox_eval_per_track.npz (method_last, seed 42)"}
            print(f"lock-box: API CPU {100 * cpu_ok.mean():.2f}% vs GPU "
                  f"{report['lockbox_cpu_vs_gpu']['gpu_lockbox_accuracy']}%; "
                  f"{len(differ)} track khác đúng/sai")

    out = ARTIFACTS_DIR / "parity_final.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"📝 {out}")


if __name__ == "__main__":
    main()
