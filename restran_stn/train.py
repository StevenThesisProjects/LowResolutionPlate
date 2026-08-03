#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    Config,
    apply_author_preset,
    apply_rtx4090_preset,
    apply_v100_preset,
)
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition"
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
        "--no-sr",
        action="store_true",
        help="Disable lightweight super-resolution preprocessing",
    )
    parser.add_argument(
        "--no-learnable-sr",
        action="store_true",
        help="Disable learnable deblur block before STN",
    )
    parser.add_argument(
        "--no-positional-head",
        action="store_true",
        help="Disable the multi-task fixed-length positional head (now the default)",
    )
    parser.add_argument(
        "--positional-head",
        action="store_true",
        help="Enable the multi-task positional head (off by default: it measured worse)",
    )
    parser.add_argument(
        "--no-cross-frame-fusion",
        action="store_true",
        help="Use the old quality-weighted AttentionFusion instead of cross-frame attention",
    )
    parser.add_argument(
        "--multi-frame-sr",
        action="store_true",
        help="Reconstruct each frame from all 5 frames of the track (SR ablation)",
    )
    parser.add_argument(
        "--no-multi-frame-sr",
        action="store_true",
        help="Use per-frame SR instead of multi-frame SR (ablation)",
    )
    parser.add_argument(
        "--no-sr-channel-attention",
        action="store_true",
        help="Disable RCAN-style channel attention in the SR block",
    )
    parser.add_argument(
        "--no-hr-distill",
        action="store_true",
        help="Disable HR privileged-information distillation",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Micro-batches per optimizer step (effective batch = batch x this)",
    )
    parser.add_argument(
        "--backbone-norm",
        type=str,
        choices=["batch", "group"],
        default=None,
        help="Backbone normalization: 'batch' (BatchNorm) or 'group' (GroupNorm)",
    )
    parser.add_argument(
        "--no-jpeg-aug",
        action="store_true",
        help="Disable JPEG-artifact injection on PNG-domain (Scenario-A) samples",
    )
    parser.add_argument(
        "--layout-balance",
        action="store_true",
        help="Oversample the minority plate layout (off by default: measured net-negative)",
    )
    parser.add_argument(
        "--no-layout-balance",
        action="store_true",
        help="Disable oversampling of the minority plate layout",
    )
    parser.add_argument(
        "--stn-shared",
        action="store_true",
        help="Predict one shared affine per track (keeps the 5 frames aligned)",
    )
    parser.add_argument(
        "--img-height", type=int, default=None,
        help="Model input height (default 32; try 48 to match the native 2.1 aspect)",
    )
    parser.add_argument(
        "--img-width", type=int, default=None,
        help="Model input width (default 128; try 96 with --img-height 48)",
    )
    parser.add_argument(
        "--sr-upsample",
        type=int,
        default=None,
        help="Learnable-SR upsampling factor (1=deblur, 2=genuine 2x SR)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained ResNet34 weights",
    )
    parser.add_argument(
        "--author-style",
        action="store_true",
        help="Use original author hyperparameters (OneCycleLR, full synthetic, no head dropout)",
    )
    parser.add_argument(
        "--v100",
        action="store_true",
        help="V100 32GB preset: batch 128, FP16 AMP, cudnn benchmark, retrain-optimized",
    )
    parser.add_argument(
        "--rtx4090",
        action="store_true",
        help="RTX 4090 24GB preset: same recipe but native bfloat16 (no fp16 divergence)",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on full dataset and generate submission file for test data",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Initialize config with CLI overrides
    config = Config()

    if args.v100:
        apply_v100_preset(config)
    if args.rtx4090:
        apply_rtx4090_preset(config)
    if args.author_style:
        apply_author_preset(config)
    
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
    if args.no_sr:
        config.USE_SR = False
    if args.no_learnable_sr:
        config.USE_LEARNABLE_SR = False
    if args.positional_head:
        config.USE_POSITIONAL_HEAD = True
    if args.no_positional_head:
        config.USE_POSITIONAL_HEAD = False
    if args.no_cross_frame_fusion:
        config.USE_CROSS_FRAME_FUSION = False
    if args.multi_frame_sr:
        config.SR_MULTI_FRAME = True
    if args.no_multi_frame_sr:
        config.SR_MULTI_FRAME = False
    if args.no_sr_channel_attention:
        config.SR_CHANNEL_ATTENTION = False
    if args.no_hr_distill:
        config.USE_HR_DISTILL = False
    if args.grad_accum_steps is not None:
        config.GRAD_ACCUM_STEPS = args.grad_accum_steps
    if args.backbone_norm is not None:
        config.BACKBONE_NORM = args.backbone_norm
    if args.no_jpeg_aug:
        config.JPEG_DOMAIN_AUG = False
    if args.layout_balance:
        config.LAYOUT_BALANCE = True
    if args.no_layout_balance:
        config.LAYOUT_BALANCE = False
    if args.stn_shared:
        config.STN_SHARED = True
    if args.img_height is not None:
        config.IMG_HEIGHT = args.img_height
    if args.img_width is not None:
        config.IMG_WIDTH = args.img_width
    if args.sr_upsample is not None:
        config.SR_UPSAMPLE = args.sr_upsample
    if args.no_pretrained:
        config.USE_PRETRAINED_BACKBONE = False
    
    # Output directory
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    seed_everything(config.SEED)
    
    print(f"🚀 Configuration:")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: ResTranOCR")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   USE_SR: {config.USE_SR}")
    print(f"   USE_LEARNABLE_SR: {config.USE_LEARNABLE_SR}")
    print(f"   USE_PRETRAINED_BACKBONE: {config.USE_PRETRAINED_BACKBONE}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   SCHEDULER: {config.SCHEDULER_TYPE}")
    print(f"   AUTHOR_STYLE: {args.author_style}")
    print(f"   V100_PRESET: {args.v100}")
    print(f"   USE_AMP: {config.USE_AMP} ({config.AMP_DTYPE})")
    print(f"   CUDNN_BENCHMARK: {config.USE_CUDNN_BENCHMARK}")
    print(f"   USE_EMA: {config.USE_EMA}")
    print(f"   SYNTHETIC_FROM_LR: {config.SYNTHETIC_FROM_LR}")
    print(f"   WEIGHT_DECAY: {config.WEIGHT_DECAY}")
    print(f"   LABEL_SMOOTHING: {getattr(config, 'LABEL_SMOOTHING', 0.0)}")
    print(f"   TRANSFORMER_DROPOUT: {config.TRANSFORMER_DROPOUT}")
    print(f"   HEAD_DROPOUT: {config.HEAD_DROPOUT}")
    if config.USE_LEARNABLE_SR:
        print(f"   SR_LOSS_WEIGHT: {config.SR_LOSS_WEIGHT} → {config.SR_LOSS_WEIGHT_MIN}")
        print(f"   SR_HIDDEN/BLOCKS: {config.SR_HIDDEN_CHANNELS}/{config.SR_NUM_BLOCKS}")
    print(f"   DEVICE: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
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
        'use_sr': config.USE_SR,
        'sr_scale': config.SR_SCALE,
        'sr_upsample': config.SR_UPSAMPLE,
        'jpeg_domain_aug': config.JPEG_DOMAIN_AUG,
        'layout_balance': config.LAYOUT_BALANCE,
        'use_synthetic_lr': config.USE_SYNTHETIC_LR,
        'synthetic_from_lr': config.SYNTHETIC_FROM_LR,
        'synthetic_sample_prob': config.SYNTHETIC_SAMPLE_PROB,
        'degradation_curriculum': config.DEGRADATION_CURRICULUM,
    }
    
    def _make_loader(dataset, shuffle: bool) -> DataLoader:
        loader_kwargs = {
            "batch_size": config.BATCH_SIZE,
            "shuffle": shuffle,
            "collate_fn": MultiFrameDataset.collate_fn,
            "num_workers": config.NUM_WORKERS,
            "pin_memory": torch.cuda.is_available(),
        }
        if config.NUM_WORKERS > 0 and config.PERSISTENT_WORKERS:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
        return DataLoader(dataset, **loader_kwargs)

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
                use_sr=config.USE_SR,
                sr_scale=config.SR_SCALE,
                sr_upsample=config.SR_UPSAMPLE,
            )
            test_loader = _make_loader(test_ds, shuffle=False)
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
            val_loader = _make_loader(val_ds, shuffle=False)
        else:
            print("⚠️ WARNING: Validation dataset is empty.")
        
        test_loader = None
    
    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    # Create training data loader
    train_loader = _make_loader(train_ds, shuffle=True)

    model = ResTranOCR(
        num_classes=config.NUM_CLASSES,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        head_dropout=config.HEAD_DROPOUT,
        use_stn=config.USE_STN,
        use_learnable_sr=config.USE_LEARNABLE_SR,
        sr_hidden=config.SR_HIDDEN_CHANNELS,
        sr_num_blocks=config.SR_NUM_BLOCKS,
        sr_scale=config.SR_UPSAMPLE,
        sr_channel_attention=config.SR_CHANNEL_ATTENTION,
        sr_multi_frame=config.SR_MULTI_FRAME,
        pretrained_backbone=config.USE_PRETRAINED_BACKBONE,
        use_positional_head=config.USE_POSITIONAL_HEAD,
        num_positions=config.NUM_POSITIONS,
        use_cross_frame_fusion=config.USE_CROSS_FRAME_FUSION,
        backbone_norm=config.BACKBONE_NORM,
        stn_shared=config.STN_SHARED,
    ).to(config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stn_label = "with STN" if config.USE_STN else "without STN"
    print(f"📊 Model (ResTranOCR {stn_label}): {total_params:,} total params, {trainable_params:,} trainable")

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
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
