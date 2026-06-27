#!/usr/bin/env python3
"""Entry point for Baseline 1: Multi-Frame CRNN + STN + optional SR."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline 1: Multi-Frame CRNN + STN + optional SR")
    parser.add_argument("-n", "--experiment-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", "--learning-rate", type=float, default=None, dest="learning_rate")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--test-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--rnn-dropout", type=float, default=None)
    parser.add_argument("--aug-level", choices=["full", "light"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-stn", action="store_true")
    parser.add_argument("--use-sr", action="store_true", help="Bật SR nhẹ theo kiến trúc stacked-input")
    parser.add_argument("--sr-scale", type=int, choices=[2, 4], default=None, help="Scale SR (giữ cho tương thích CLI)")
    parser.add_argument("--submission-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    overrides = {
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
        "output_dir": "OUTPUT_DIR",
    }
    for arg, attr in overrides.items():
        val = getattr(args, arg, None)
        if val is not None:
            setattr(config, attr, val)

    if args.aug_level:
        config.AUGMENTATION_LEVEL = args.aug_level
    if args.no_stn:
        config.USE_STN = False
        if args.experiment_name is None:
            config.EXPERIMENT_NAME = "crnn_no_stn"
    if args.use_sr:
        config.USE_SR = True
        if args.experiment_name is None:
            config.EXPERIMENT_NAME = "crnn_stn_sr"
    if args.sr_scale is not None:
        config.SR_SCALE = args.sr_scale

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.SEED)

    print("=" * 60)
    print("Baseline 1: Multi-Frame CRNN + STN + optional SR")
    print("=" * 60)
    print(f"  Experiment : {config.EXPERIMENT_NAME}")
    print(f"  STN        : {config.USE_STN}")
    print(f"  SR         : {config.USE_SR} (scale={config.SR_SCALE})")
    print(f"  Data       : {config.DATA_ROOT}")
    print(f"  Epochs     : {config.EPOCHS} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print(f"  Device     : {config.DEVICE}")
    print(f"  Submission : {args.submission_mode}")
    print("=" * 60)

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

    test_loader = None
    val_loader = None

    if args.submission_mode:
        train_ds = MultiFrameDataset(config.DATA_ROOT, mode="train", full_train=True, **ds_params)
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
        train_ds = MultiFrameDataset(config.DATA_ROOT, mode="train", **ds_params)
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

    model = MultiFrameCRNN(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        rnn_dropout=config.RNN_DROPOUT,
        use_stn=config.USE_STN,
        use_sr=config.USE_SR,
        sr_scale=config.SR_SCALE,
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
