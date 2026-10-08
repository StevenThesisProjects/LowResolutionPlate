#!/usr/bin/env python3
"""
Ablation study cho CRNN + STN (Baseline 1).

Chạy 2 experiment tự động:
  1. crnn_no_stn   — CRNN không STN
  2. crnn_with_stn — CRNN + STN (Baseline 1)

Nguồn gốc: MultiFrame-LPR-main/run_ablation.py (chỉ giữ phần CRNN)
"""
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


def build_command(experiment_config: Dict[str, Any], output_dir: str = "experiments") -> List[str]:
    """Tạo lệnh python train.py từ config experiment."""
    cmd: List[str] = [sys.executable or "python3", "train.py"]

    if "experiment_name" in experiment_config:
        cmd += ["-n", str(experiment_config["experiment_name"])]
    if "aug_level" in experiment_config:
        cmd += ["--aug-level", str(experiment_config["aug_level"])]

    cmd += ["--output-dir", output_dir]

    for flag in experiment_config.get("extra_flags", []):
        cmd.append(str(flag))

    return cmd


def _parse_best_accuracy(log_path: str) -> Optional[float]:
    """Đọc Best Val Acc từ file log."""
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "Best Val Acc:" in line:
                    try:
                        token = line.split("Best Val Acc:")[1].strip().split("%")[0]
                        return float(token)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        pass
    return None


def main() -> None:
    experiments_dir = "experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    # 2 experiment CRNN (report Trang 50)
    experiments: List[Dict[str, Any]] = [
        {
            "name": "crnn_no_stn",
            "experiment_name": "crnn_no_stn",
            "aug_level": "full",
            "extra_flags": ["--no-stn"],
        },
        {
            "name": "crnn_with_stn",
            "experiment_name": "crnn_with_stn",
            "aug_level": "full",
            # Mặc định USE_STN=True trong config
        },
    ]

    results_summary: List[Dict[str, Any]] = []

    for experiment_config in experiments:
        experiment_name = experiment_config["name"]
        log_path = os.path.join(experiments_dir, f"{experiment_name}.log")
        cmd = build_command(experiment_config, experiments_dir)

        print(f"\n=== Experiment: {experiment_name} ===")
        print("Command:", " ".join(cmd))

        try:
            with open(log_path, "w") as log_file:
                process = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

            if process.returncode != 0:
                print(f"[{experiment_name}] FAILED (code {process.returncode}). Xem {log_path}")
                results_summary.append({"name": experiment_name, "best_acc": None})
                continue

            print(f"[{experiment_name}] DONE. Log: {log_path}")
            results_summary.append({
                "name": experiment_name,
                "best_acc": _parse_best_accuracy(log_path),
            })

        except Exception as e:
            print(f"[{experiment_name}] ERROR: {e}")
            results_summary.append({"name": experiment_name, "best_acc": None})

    # In và lưu bảng tổng kết
    if results_summary:
        lines = [
            "=== Ablation Summary — CRNN + STN ===",
            f"{'Experiment':25s} | {'Best Acc (%)':12s}",
            "-" * 40,
        ]
        for row in results_summary:
            acc = (
                f"{row['best_acc']:.2f}"
                if isinstance(row.get("best_acc"), (int, float))
                else "N/A"
            )
            lines.append(f"{row['name']:25s} | {acc:12s}")

        summary_text = "\n".join(lines)
        print("\n" + summary_text)

        summary_file = os.path.join(experiments_dir, "ablation_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary_text + "\n")
        print(f"\n📝 Summary → {summary_file}")


if __name__ == "__main__":
    main()
