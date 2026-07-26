#!/usr/bin/env python3
"""Đo Params / FLOPs / Inference latency cho từng cấu hình model.

Phục vụ mục "Computational Complexity" trong issue #9 — để biết mỗi cải tiến
(SR, DCN, GroupNorm) phải trả giá bao nhiêu về compute, không chỉ nhìn accuracy.

Ví dụ:
    python tools/benchmark.py                                  # baseline ResBlock
    python tools/benchmark.py --use-sr --backbone-norm group   # + SR + GroupNorm
    python tools/benchmark.py --use-sr --use-dcn --backbone-norm group
    python tools/benchmark.py --all                            # so sánh mọi cấu hình
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from src.models.crnn import MultiFrameCRNN


def build_model(config: Config) -> MultiFrameCRNN:
    return MultiFrameCRNN(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        rnn_dropout=config.RNN_DROPOUT,
        use_stn=config.USE_STN,
        backbone_channels=config.BACKBONE_CHANNELS,
        backbone_base_channels=config.BACKBONE_BASE_CHANNELS,
        backbone_blocks=config.BACKBONE_STAGE_BLOCKS,
        backbone_stage_channels=config.BACKBONE_STAGE_CHANNELS,
        use_se=config.BACKBONE_USE_SE,
        residual_scale=config.BACKBONE_RES_SCALE,
        frame_dropout=0.0,
        use_sr=config.USE_SR,
        sr_scale=config.SR_SCALE,
        sr_hidden_channels=config.SR_HIDDEN_CHANNELS,
        sr_num_blocks=config.SR_NUM_BLOCKS,
        sr_res_scale=config.SR_RES_SCALE,
        backbone_norm=config.BACKBONE_NORM,
        use_dcn=config.USE_DCN,
        dcn_hidden_channels=config.DCN_HIDDEN_CHANNELS,
        fusion_mode=config.FUSION_MODE,
    )


def count_flops(model: torch.nn.Module, sample: torch.Tensor) -> float | None:
    """FLOPs cho 1 track. Trả None nếu không đo được (thiếu backend)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    try:
        counter = FlopCounterMode(display=False)
        with counter:
            model(sample)
        return counter.get_total_flops()
    except Exception:
        return None


def measure_latency(model, sample, device, warmup: int = 5, runs: int = 30) -> float:
    """ms trung bình cho 1 track (batch=1)."""
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / runs * 1000.0


def benchmark(config: Config, label: str, device: torch.device) -> dict:
    model = build_model(config).to(device).eval()
    sample = torch.randn(
        1, config.NUM_FRAMES, 3, config.IMG_HEIGHT, config.IMG_WIDTH, device=device
    )
    params = sum(p.numel() for p in model.parameters())
    flops = count_flops(model, sample)
    latency = measure_latency(model, sample, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "params": params, "flops": flops, "latency_ms": latency}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark params/FLOPs/latency")
    parser.add_argument("--use-sr", action="store_true")
    parser.add_argument("--use-dcn", action="store_true")
    parser.add_argument("--backbone-norm", choices=["none", "group"], default=None)
    parser.add_argument("--fusion-mode", choices=["attention", "avg", "max"], default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--preset", choices=["stable", "strong", "debug"], default="stable")
    parser.add_argument("--cpu", action="store_true", help="Ép chạy CPU")
    parser.add_argument("--all", action="store_true", help="So sánh mọi cấu hình chính")
    return parser.parse_args()


def make_config(preset: str, **overrides) -> Config:
    import train as train_module

    config = Config()
    train_module._apply_preset(config, preset)
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config


def print_table(rows: list[dict]) -> None:
    baseline_params = rows[0]["params"]
    baseline_latency = rows[0]["latency_ms"]
    print()
    print(f"{'Cấu hình':<38} {'Params':>12} {'GFLOPs/track':>14} {'Latency ms':>12} {'vs base':>10}")
    print("-" * 92)
    for row in rows:
        gflops = f"{row['flops'] / 1e9:.2f}" if row["flops"] else "n/a"
        ratio = row["latency_ms"] / baseline_latency if baseline_latency else 1.0
        print(
            f"{row['label']:<38} {row['params']:>12,} {gflops:>14} "
            f"{row['latency_ms']:>12.2f} {ratio:>9.2f}x"
        )
    print("-" * 92)
    print(f"Params tăng so với baseline: "
          f"{', '.join(f'{r['label']}={r['params'] - baseline_params:+,}' for r in rows[1:])}")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    common = dict(
        NUM_FRAMES=args.num_frames,
        IMG_HEIGHT=args.img_height,
        IMG_WIDTH=args.img_width,
    )

    if args.all:
        configs = [
            ("Baseline ResBlock (không SR)", dict(USE_SR=False, BACKBONE_NORM="none")),
            ("+ GroupNorm", dict(USE_SR=False, BACKBONE_NORM="group")),
            ("+ SR (per-frame)", dict(USE_SR=True, BACKBONE_NORM="group")),
            ("+ SR + DCNv2", dict(USE_SR=True, USE_DCN=True, BACKBONE_NORM="group")),
        ]
        rows = [
            benchmark(make_config(args.preset, **{**common, **kw}), label, device)
            for label, kw in configs
        ]
    else:
        config = make_config(
            args.preset,
            **common,
            USE_SR=True if args.use_sr else None,
            USE_DCN=True if args.use_dcn else None,
            BACKBONE_NORM=args.backbone_norm,
            FUSION_MODE=args.fusion_mode,
        )
        label = (
            f"SR={config.USE_SR} DCN={config.USE_DCN} "
            f"norm={config.BACKBONE_NORM} fusion={config.FUSION_MODE}"
        )
        rows = [benchmark(config, label, device)]

    print_table(rows)


if __name__ == "__main__":
    main()
