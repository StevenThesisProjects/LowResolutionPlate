#!/usr/bin/env python3
"""Main entry point for the ResTranOCR training pipeline."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ResTranOCR (ResNet34 + Transformer) for Low-Resolution License Plate Recognition"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/submission files (default: from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=None,
        help="Number of transformer attention heads (default: from config)"
    )
    parser.add_argument(
        "--transformer-layers", type=int, default=None,
        help="Number of transformer encoder layers (default: from config)"
    )
    parser.add_argument(
        "--img-height", type=int, default=None,
        help="Input image height (default: 32)"
    )
    parser.add_argument(
        "--img-width", type=int, default=None,
        help="Input image width (default: 128). Sets the CTC sequence length"
    )
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level for training data (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save checkpoints and submission files (default: results/)",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable Spatial Transformer Network (STN) alignment",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on full dataset and generate submission file for test data",
    )
    parser.add_argument(
        "--log-batch-loss",
        action="store_true",
        help="Also write per-batch loss/lr/grad-norm to {experiment}_batches.csv",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=None,
        help="Max gradient norm (default: from config, 5.0). Use 1.0 for CTC stability"
    )
    parser.add_argument(
        "--amp-dtype", type=str, choices=["fp16", "bf16"], default=None,
        help="Mixed precision dtype. bf16 cannot overflow (needs Ampere+); default fp16"
    )
    parser.add_argument(
        "--skip-grad-norm", type=float, default=None,
        help="Skip the optimizer step when pre-clip grad norm exceeds this (0 = off)"
    )
    parser.add_argument(
        "--skip-abort-ratio", type=float, default=None,
        help="Stop training if this fraction of an epoch's batches got skipped (0 = off)"
    )
    parser.add_argument(
        "--split-ratio", type=float, default=None,
        help="Train fraction of Scenario-B (default 0.9 -> val 999; 0.8 -> val 1999). "
             "Only applies when the split file does not already exist"
    )
    parser.add_argument(
        "--val-split-file", type=str, default=None,
        help="Path to the val split JSON (default: splits/val_tracks.json). "
             "Use a new path to create a different split without touching the old one"
    )
    parser.add_argument(
        "--sr",
        action="store_true",
        help="Add the multi-frame SR front-end; implies 16x64 input unless overridden",
    )
    parser.add_argument(
        "--sr-features", type=int, default=None,
        help="SR channel width (default 32)",
    )
    parser.add_argument(
        "--sr-blocks", type=int, default=None,
        help="Number of SR residual blocks (default 8)",
    )
    parser.add_argument(
        "--sr-init", type=str, default=None,
        help="Load a pre-trained SR front-end (from tools/pretrain_sr.py) before training",
    )
    parser.add_argument(
        "--sr-pixel-weight", type=float, default=None,
        help="Weight of the L1 loss against ECC-registered HR frames (0 = off)",
    )
    parser.add_argument(
        "--sr-ncc-min", type=float, default=None,
        help="Minimum registration NCC for a pair to be used as an SR target",
    )
    parser.add_argument(
        "--sr-arch", type=str, choices=["stacked", "edvr"], default=None,
        help="stacked = the slide's channel-concat design; edvr = deformable alignment",
    )
    parser.add_argument(
        "--sr-scale", type=int, choices=[1, 2], default=None,
        help="2 = learned 2x upscale from 16x64; 1 = same-resolution mixing at 32x128",
    )
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="Aspect-preserving resize + pad, so every scenario shares one geometry",
    )
    parser.add_argument(
        "--single-frame",
        action="store_true",
        help="Use one frame replicated 5x — ablation for the value of multi-frame fusion",
    )
    parser.add_argument(
        "--teacher", type=str, default=None,
        help="Path to an HR-trained checkpoint used as a distillation teacher",
    )
    parser.add_argument(
        "--distill-weight", type=float, default=None,
        help="Weight of the feature-distillation loss (needs --teacher)",
    )
    parser.add_argument(
        "--use-hr",
        action="store_true",
        help="Train and validate on undegraded HR frames (measures the accuracy ceiling)",
    )
    parser.add_argument(
        "--consistent-aug",
        action="store_true",
        help="Use the same augmentation parameters for all 5 frames of a track",
    )
    parser.add_argument(
        "--no-channel-shuffle",
        action="store_true",
        help="Drop ChannelShuffle from the training pipeline",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialise ResNet34 from ImageNet weights (default: train from scratch)",
    )
    parser.add_argument(
        "--layout-balance", type=float, default=None,
        help="Sampling weight for Brazilian plates (1.0 = off, ~1.73 = equalise layouts)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .pth checkpoint to load before training"
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Initialize config with CLI overrides
    config = Config()

    # Map CLI arguments to config attributes
    arg_to_config = {
        'experiment_name': 'EXPERIMENT_NAME',
        'epochs': 'EPOCHS',
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'data_root': 'DATA_ROOT',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
        'img_height': 'IMG_HEIGHT',
        'img_width': 'IMG_WIDTH',
        'grad_clip': 'GRAD_CLIP',
        'amp_dtype': 'AMP_DTYPE',
        'skip_grad_norm': 'SKIP_GRAD_NORM',
        'skip_abort_ratio': 'SKIP_ABORT_RATIO',
        'split_ratio': 'SPLIT_RATIO',
        'val_split_file': 'VAL_SPLIT_FILE',
        'layout_balance': 'LAYOUT_BALANCE',
        'distill_weight': 'DISTILL_WEIGHT',
        'sr_features': 'SR_FEATURES',
        'sr_blocks': 'SR_BLOCKS',
        'sr_scale': 'SR_SCALE',
        'sr_arch': 'SR_ARCH',
        'sr_pixel_weight': 'SR_PIXEL_WEIGHT',
        'sr_ncc_min': 'SR_NCC_MIN',
    }

    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    # Special cases
    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level

    if args.no_stn:
        config.USE_STN = False

    config.LOG_BATCH_LOSS = args.log_batch_loss

    if args.pretrained:
        config.PRETRAINED_BACKBONE = True
    if args.use_hr:
        config.USE_HR = True
    if args.letterbox:
        config.LETTERBOX = True
    if args.single_frame:
        config.SINGLE_FRAME = True
    if args.sr:
        config.USE_SR = True
        # The SR module upscales 2x, so halve the pipeline size to land on 32x128.
        if config.SR_SCALE == 2 and args.img_height is None and args.img_width is None:
            config.IMG_HEIGHT, config.IMG_WIDTH = 16, 64
            print("📐 --sr: dataset resize 16x64, SR upscale x2 -> 32x128")
    if args.consistent_aug:
        config.CONSISTENT_AUG = True
    if args.no_channel_shuffle:
        config.CHANNEL_SHUFFLE = False

    # Output directory
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Re-using an experiment name silently overwrites the previous run's checkpoint
    # and metrics — the results become impossible to compare afterwards.
    existing = [
        f for f in (f"{config.EXPERIMENT_NAME}_best.pth",
                    f"{config.EXPERIMENT_NAME}_metrics.csv")
        if os.path.exists(os.path.join(config.OUTPUT_DIR, f))
    ]
    if existing:
        print(f"⚠️  CẢNH BÁO: '{config.EXPERIMENT_NAME}' đã có kết quả cũ trong "
              f"{config.OUTPUT_DIR}/: {', '.join(existing)}")
        print("    Chạy tiếp sẽ GHI ĐÈ. Dùng -n <tên khác> nếu muốn giữ để đối chứng.")

    seed_everything(config.SEED)

    print(f"🚀 Configuration:")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: ResTranOCR (ResNet34 + Transformer)")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   PRETRAINED_BACKBONE: {config.PRETRAINED_BACKBONE}")
    print(f"   CONSISTENT_AUG: {config.CONSISTENT_AUG} | CHANNEL_SHUFFLE: {config.CHANNEL_SHUFFLE}")
    print(f"   IMG_SIZE: {config.IMG_HEIGHT}x{config.IMG_WIDTH} | SINGLE_FRAME: {config.SINGLE_FRAME} | LETTERBOX: {config.LETTERBOX}")
    print(f"   USE_SR: {config.USE_SR}" + (f" ({config.SR_ARCH}, nf={config.SR_FEATURES}, blocks={config.SR_BLOCKS}, scale={config.SR_SCALE})" if config.USE_SR else ""))
    if config.USE_HR:
        print("   ⚠️  USE_HR: train VÀ val đều dùng ảnh HR gốc — đây là phép ĐO TRẦN,")
        print("       không so sánh được với các run thường và không dùng để nộp bài.")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   GRAD_CLIP: {config.GRAD_CLIP}")
    print(f"   AMP_DTYPE: {config.AMP_DTYPE}")
    print(f"   SKIP_GRAD_NORM: {config.SKIP_GRAD_NORM or 'off'}")
    print(f"   DEVICE: {config.DEVICE}")
    print(f"   SUBMISSION_MODE: {args.submission_mode}")

    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # Common dataset parameters
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
        'consistent_aug': config.CONSISTENT_AUG,
        'channel_shuffle': config.CHANNEL_SHUFFLE,
        'use_hr': config.USE_HR,
        'with_hr_pair': args.teacher is not None,
        'single_frame': config.SINGLE_FRAME,
        'letterbox': config.LETTERBOX,
        'registration_file': config.REGISTRATION_FILE if config.SR_PIXEL_WEIGHT > 0 else None,
    }

    # Create datasets based on mode
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        print("   - Training on FULL dataset (no validation split)")
        print("   - Will generate predictions for test data after training\n")

        # Create training dataset with full_train=True
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            full_train=True,
            **common_ds_params
        )

        # Create test dataset if test data exists
        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode='val',
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
                char2idx=config.CHAR2IDX,
                seed=config.SEED,
                is_test=True,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print(f"⚠️ WARNING: Test data not found at {config.TEST_DATA_ROOT}")

        val_loader = None
    else:
        # Normal training/validation split mode
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            **common_ds_params
        )

        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            **common_ds_params
        )

        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print("⚠️ WARNING: Validation dataset is empty.")

        test_loader = None

    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    # Class-balanced sampling over plate layout. Keeps the number of batches per
    # epoch identical, so wall-clock time is unchanged; only the mix shifts.
    sampler = None
    counts = train_ds.layout_counts()
    print(f"📐 Layout trong train: " +
          ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if config.LAYOUT_BALANCE != 1.0:
        weights = train_ds.layout_weights(config.LAYOUT_BALANCE)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(config.SEED),
        )
        total_w = sum(weights)
        braz_w = config.LAYOUT_BALANCE * counts.get('Brazilian', 0)
        print(f"   -> LAYOUT_BALANCE={config.LAYOUT_BALANCE}: "
              f"Brazilian chiếm {braz_w / total_w * 100:.1f}% luồng mẫu "
              f"(trước là {counts.get('Brazilian', 0) / len(train_ds) * 100:.1f}%)")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model
    model = ResTranOCR(
        num_classes=config.NUM_CLASSES,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        use_stn=config.USE_STN,
        pretrained=config.PRETRAINED_BACKBONE,
        use_sr=config.USE_SR,
        sr_features=config.SR_FEATURES,
        sr_blocks=config.SR_BLOCKS,
        sr_scale=config.SR_SCALE,
        sr_arch=config.SR_ARCH,
    ).to(config.DEVICE)

    # Warm-start the SR front-end from supervised pre-training on registered pairs
    if args.sr_init:
        if not config.USE_SR:
            print("❌ ERROR: --sr-init cần --sr"); sys.exit(1)
        if not os.path.exists(args.sr_init):
            print(f"❌ ERROR: Không thấy {args.sr_init}"); sys.exit(1)
        model.sr.load_state_dict(torch.load(args.sr_init, map_location=config.DEVICE))
        print(f"🧱 Nạp SR đã pre-train: {args.sr_init}")

    # Resume from an existing checkpoint (weights only; optimizer state is not saved)
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"❌ ERROR: Checkpoint not found: {args.resume}")
            sys.exit(1)
        model.load_state_dict(torch.load(args.resume, map_location=config.DEVICE))
        print(f"📦 Resumed weights from: {args.resume}")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model (restran): {total_params:,} total params, {trainable_params:,} trainable")

    # Frozen teacher for feature distillation (trained on undegraded HR frames)
    teacher = None
    if args.teacher:
        if not os.path.exists(args.teacher):
            print(f"❌ ERROR: Teacher checkpoint not found: {args.teacher}")
            sys.exit(1)
        if config.DISTILL_WEIGHT <= 0:
            print("❌ ERROR: --teacher cần --distill-weight > 0")
            sys.exit(1)
        teacher = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)
        teacher.load_state_dict(torch.load(args.teacher, map_location=config.DEVICE))
        teacher.eval()
        print(f"🎓 Teacher: {args.teacher} | DISTILL_WEIGHT={config.DISTILL_WEIGHT}")

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
        teacher=teacher,
    )

    trainer.fit()

    # Run test inference in submission mode
    if args.submission_mode and test_loader is not None:
        print("\n" + "="*60)
        print("📝 GENERATING SUBMISSION FILE")
        print("="*60)

        # Load best checkpoint if it exists
        exp_name = config.EXPERIMENT_NAME
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        else:
            print("⚠️ No best checkpoint found, using final model weights")

        # Run inference on test data
        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_final.txt")


if __name__ == "__main__":
    main()
