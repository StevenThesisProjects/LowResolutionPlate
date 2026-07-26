#!/usr/bin/env python3
"""Training entry point for the upgraded CRNN + STN OCR model.

This script wires together the data pipeline, the residual backbone presets,
and the trainer. It now supports longer/stabler training runs through CLI
presets, backbone overrides, gradient accumulation, warmup, and early stopping.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse comma-separated stage presets from the command line."""

    cleaned = value.replace("x", ",").replace(";", ",").replace(" ", "")
    parts = [part for part in cleaned.split(",") if part]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRNN + STN with residual backbone")
    parser.add_argument("-n", "--experiment-name", type=str, default=None)
    parser.add_argument("--preset", choices=["stable", "strong", "debug"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", "--learning-rate", type=float, default=None, dest="learning_rate")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--test-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--rnn-dropout", type=float, default=None)
    parser.add_argument("--backbone-base-channels", type=int, default=None)
    parser.add_argument("--backbone-channels", type=int, default=None)
    parser.add_argument("--backbone-blocks", type=_parse_int_tuple, default=None, help="Comma-separated stage block counts, e.g. 2,2,3,3,4")
    parser.add_argument("--backbone-stage-channels", type=_parse_int_tuple, default=None, help="Comma-separated stage channels, e.g. 64,128,256,256,512")
    parser.add_argument("--backbone-res-scale", type=float, default=None)
    parser.add_argument(
        "--backbone-norm", choices=["none", "group"], default=None,
        help="'group' thêm GroupNorm thay BatchNorm đã bỏ — khuyến nghị khi bật --use-sr",
    )
    parser.add_argument("--frame-dropout", type=float, default=None)
    parser.add_argument("--fusion-dropout", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--aug-level", choices=["full", "light"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--use-sr", action="store_true", help="Bật SR per-frame sau STN, có giám sát pixel-level")
    parser.add_argument("--sr-scale", type=int, default=None)
    parser.add_argument("--sr-hidden-channels", type=int, default=None)
    parser.add_argument("--sr-num-blocks", type=int, default=None)
    parser.add_argument("--sr-res-scale", type=float, default=None)
    parser.add_argument("--lambda-sr", type=float, default=None, help="Trọng số L_SR trong L_CTC + lambda*L_SR")
    parser.add_argument("--sr-edge-weight", type=float, default=None)
    parser.add_argument("--no-stn", action="store_true")
    parser.add_argument("--no-se", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    parser.add_argument("--submission-mode", action="store_true")
    parser.add_argument("--full-train", action="store_true", help="Train on all data and skip validation split")
    return parser.parse_args()


def _apply_preset(config: Config, preset: str) -> None:
    # Presets are convenience bundles for common experiment styles.
    if preset == "debug":
        config.EXPERIMENT_NAME = f"{config.EXPERIMENT_NAME}_debug"
        config.EPOCHS = 4
        config.BATCH_SIZE = 8
        config.LEARNING_RATE = 1e-3
        config.NUM_WORKERS = 0
        config.GRAD_ACCUM_STEPS = 1
        config.WARMUP_RATIO = 0.0
        config.MIN_LR_RATIO = 0.2
        config.EARLY_STOPPING_PATIENCE = 2
        config.BACKBONE_STAGE_BLOCKS = (1, 1, 1, 1, 1)
        config.BACKBONE_STAGE_CHANNELS = (32, 64, 96, 128, 256)
        config.BACKBONE_BASE_CHANNELS = 32
        config.BACKBONE_CHANNELS = 256
        config.BACKBONE_RES_SCALE = 0.15
        config.FRAME_DROPOUT = 0.1
        config.RNN_DROPOUT = 0.1
        return

    if preset == "stable":
        config.EXPERIMENT_NAME = f"{config.EXPERIMENT_NAME}_stable"
        config.EPOCHS = 80
        config.BATCH_SIZE = 64
        config.LEARNING_RATE = 8e-4
        config.GRAD_ACCUM_STEPS = 1
        config.WARMUP_RATIO = 0.05
        config.MIN_LR_RATIO = 0.05
        config.EARLY_STOPPING_PATIENCE = 18
        config.BACKBONE_STAGE_BLOCKS = (2, 2, 2, 2, 2)
        config.BACKBONE_STAGE_CHANNELS = (64, 128, 256, 256, 512)
        config.BACKBONE_BASE_CHANNELS = 64
        config.BACKBONE_CHANNELS = 512
        config.BACKBONE_RES_SCALE = 0.1
        config.FRAME_DROPOUT = 0.05
        config.RNN_DROPOUT = 0.25
        config.BACKBONE_USE_SE = True
        return

    if preset == "strong":
        config.EXPERIMENT_NAME = f"{config.EXPERIMENT_NAME}_strong"
        config.EPOCHS = 120
        config.BATCH_SIZE = 96
        config.LEARNING_RATE = 6e-4
        config.GRAD_ACCUM_STEPS = 2
        config.WARMUP_RATIO = 0.08
        config.MIN_LR_RATIO = 0.03
        config.EARLY_STOPPING_PATIENCE = 24
        config.BACKBONE_STAGE_BLOCKS = (2, 2, 3, 3, 4)
        config.BACKBONE_STAGE_CHANNELS = (64, 128, 256, 256, 512)
        config.BACKBONE_BASE_CHANNELS = 64
        config.BACKBONE_CHANNELS = 512
        config.BACKBONE_RES_SCALE = 0.08
        config.FRAME_DROPOUT = 0.08
        config.RNN_DROPOUT = 0.2
        config.BACKBONE_USE_SE = True
        return


def _apply_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.preset is not None:
        _apply_preset(config, args.preset)

    mapping = {
        "experiment_name": "EXPERIMENT_NAME",
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "learning_rate": "LEARNING_RATE",
        "data_root": "DATA_ROOT",
        "test_root": "TEST_DATA_ROOT",
        "seed": "SEED",
        "num_workers": "NUM_WORKERS",
        "hidden_size": "HIDDEN_SIZE",
        "rnn_dropout": "RNN_DROPOUT",
        "backbone_base_channels": "BACKBONE_BASE_CHANNELS",
        "backbone_channels": "BACKBONE_CHANNELS",
        "backbone_res_scale": "BACKBONE_RES_SCALE",
        "backbone_norm": "BACKBONE_NORM",
        "frame_dropout": "FRAME_DROPOUT",
        "fusion_dropout": "FUSION_DROPOUT",
        "sr_scale": "SR_SCALE",
        "sr_hidden_channels": "SR_HIDDEN_CHANNELS",
        "sr_num_blocks": "SR_NUM_BLOCKS",
        "sr_res_scale": "SR_RES_SCALE",
        "lambda_sr": "LAMBDA_SR",
        "sr_edge_weight": "SR_EDGE_WEIGHT",
        "grad_clip": "GRAD_CLIP",
        "grad_accum_steps": "GRAD_ACCUM_STEPS",
        "label_smoothing": "LABEL_SMOOTHING",
        "warmup_ratio": "WARMUP_RATIO",
        "min_lr_ratio": "MIN_LR_RATIO",
        "patience": "EARLY_STOPPING_PATIENCE",
        "output_dir": "OUTPUT_DIR",
    }
    for arg_name, attr_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, attr_name, value)

    if args.backbone_blocks is not None:
        config.BACKBONE_STAGE_BLOCKS = args.backbone_blocks
    if args.backbone_stage_channels is not None:
        config.BACKBONE_STAGE_CHANNELS = args.backbone_stage_channels
        config.BACKBONE_CHANNELS = args.backbone_stage_channels[-1]

    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    if args.use_sr:
        config.USE_SR = True
    if args.no_stn:
        config.USE_STN = False
        if args.experiment_name is None and args.preset is None:
            config.EXPERIMENT_NAME = "crnn_resblock_no_stn"
    if args.no_se:
        config.BACKBONE_USE_SE = False
    if args.no_amp:
        config.USE_AMP = False
    if args.no_cudnn_benchmark:
        config.USE_CUDNN_BENCHMARK = False

    if args.fusion_dropout is not None:
        config.FUSION_DROPOUT = args.fusion_dropout


def main() -> None:
    args = parse_args()
    config = Config()
    _apply_overrides(config, args)

    if args.full_train and not args.submission_mode:
        args.submission_mode = True

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

    print("=" * 72)
    print("CRNN + STN with ResBlock backbone")
    print("=" * 72)
    print(f"Experiment : {config.EXPERIMENT_NAME}")
    print(f"Preset     : {args.preset or 'custom'}")
    print(f"STN        : {config.USE_STN}")
    print(f"SE         : {config.BACKBONE_USE_SE}")
    print(f"SR         : {config.USE_SR} (scale={config.SR_SCALE}, lambda_sr={config.LAMBDA_SR})")
    print(f"Data       : {config.DATA_ROOT}")
    print(f"Epochs     : {config.EPOCHS} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print(f"AMP        : {config.USE_AMP} | Grad Accum: {config.GRAD_ACCUM_STEPS}")
    print(f"Device     : {config.DEVICE}")
    print(f"Submission : {args.submission_mode}")
    print(f"Backbone   : base={config.BACKBONE_BASE_CHANNELS}, blocks={config.BACKBONE_STAGE_BLOCKS}, channels={config.BACKBONE_STAGE_CHANNELS}, res_scale={config.BACKBONE_RES_SCALE}, norm={config.BACKBONE_NORM}")
    print("=" * 72)

    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ Không tìm thấy data: {config.DATA_ROOT}")
        sys.exit(1)

    ds_params = {
        "split_ratio": config.SPLIT_RATIO,
        "img_height": config.IMG_HEIGHT,
        "img_width": config.IMG_WIDTH,
        "char2idx": config.CHAR2IDX,
        "val_split_file": config.VAL_SPLIT_FILE,
        "seed": config.SEED,
        "augmentation_level": config.AUGMENTATION_LEVEL,
    }

    val_loader = None
    test_loader = None

    if args.submission_mode:
        train_ds = MultiFrameDataset(
            config.DATA_ROOT,
            mode="train",
            full_train=True,
            provide_sr_target=config.USE_SR,
            sr_scale=config.SR_SCALE,
            **ds_params,
        )
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                config.TEST_DATA_ROOT,
                mode="val",
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
                char2idx=config.CHAR2IDX,
                is_test=True,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )
        else:
            print(f"⚠️ WARNING: Không tìm thấy test data tại {config.TEST_DATA_ROOT}")
    else:
        train_ds = MultiFrameDataset(
            config.DATA_ROOT,
            mode="train",
            provide_sr_target=config.USE_SR,
            sr_scale=config.SR_SCALE,
            **ds_params,
        )
        # val_ds không cần HR target: validate() chỉ đo CTC/exact-match, không
        # tính SR loss — giữ nguyên 5-tuple, không đổi hành vi baseline cũ.
        val_ds = MultiFrameDataset(config.DATA_ROOT, mode="val", **ds_params)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )
        else:
            print("⚠️ WARNING: Validation dataset rỗng.")

    if len(train_ds) == 0:
        print("❌ Training dataset rỗng!")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Build the full OCR network from the config so command line overrides can
    # change the backbone depth, channels, or regularization in one place.
    model = MultiFrameCRNN(
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
        frame_dropout=config.FRAME_DROPOUT,
        use_sr=config.USE_SR,
        sr_scale=config.SR_SCALE,
        sr_hidden_channels=config.SR_HIDDEN_CHANNELS,
        sr_num_blocks=config.SR_NUM_BLOCKS,
        sr_res_scale=config.SR_RES_SCALE,
        backbone_norm=config.BACKBONE_NORM,
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model params: {total_params:,} total, {trainable_params:,} trainable")

    trainer = Trainer(model, train_loader, val_loader, config, config.IDX2CHAR)
    trainer.fit()

    if args.submission_mode and test_loader is not None:
        best_path = os.path.join(config.OUTPUT_DIR, f"{config.EXPERIMENT_NAME}_best.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
        trainer.predict_test(test_loader, f"submission_{config.EXPERIMENT_NAME}_final.txt")


if __name__ == "__main__":
    main()
