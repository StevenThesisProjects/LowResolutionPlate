#!/usr/bin/env python3
"""Run inference with a trained ResTranOCR checkpoint."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config, apply_v100_preset
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained ResTranOCR checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pth checkpoint (e.g. results/restran34_with_stn_best.pth)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "val"],
        default="test",
        help="'test' = public test set (no labels), 'val' = validation set with accuracy",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Data root for validation mode (default: from config)",
    )
    parser.add_argument(
        "--test-data-root",
        type=str,
        default=None,
        help="Data root for test mode (default: from config)",
    )
    parser.add_argument(
        "--img-height", type=int, default=None,
        help="Model input height (must match training, e.g. 48)",
    )
    parser.add_argument(
        "--img-width", type=int, default=None,
        help="Model input width (must match training, e.g. 96)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save submission file (default: results/)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Submission filename (default: submission_<checkpoint_stem>.txt)",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable STN (must match how the checkpoint was trained)",
    )
    parser.add_argument(
        "--no-sr",
        action="store_true",
        help="Disable lightweight super-resolution preprocessing",
    )
    parser.add_argument(
        "--no-learnable-sr",
        action="store_true",
        help="Disable learnable deblur block (must match training)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained backbone (must match training)",
    )
    parser.add_argument(
        "--v100",
        action="store_true",
        help="Use V100 inference defaults (num_workers=8, pin_memory)",
    )
    parser.add_argument(
        "--stn-shared",
        action="store_true",
        help="Model was trained with shared per-track STN (must match training)",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable TTA at inference (faster, greedy+beam only)",
    )
    parser.add_argument(
        "--no-beam",
        action="store_true",
        help="Disable beam search at inference (greedy decode only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    config = Config()
    if args.v100:
        apply_v100_preset(config)
    if args.img_height is not None:
        config.IMG_HEIGHT = args.img_height
    if args.img_width is not None:
        config.IMG_WIDTH = args.img_width
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.no_stn:
        config.USE_STN = False
    if args.no_sr:
        config.USE_SR = False
    if args.no_learnable_sr:
        config.USE_LEARNABLE_SR = False
    if args.no_pretrained:
        config.USE_PRETRAINED_BACKBONE = False
    if args.no_tta:
        config.USE_INFERENCE_TTA = False
    if args.no_beam:
        config.USE_INFERENCE_BEAM = False

    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.SEED)

    if args.mode == "test":
        data_root = args.test_data_root or config.TEST_DATA_ROOT
        if not os.path.exists(data_root):
            print(f"❌ Test data not found: {data_root}")
            sys.exit(1)

        dataset = MultiFrameDataset(
            root_dir=data_root,
            mode="val",
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            char2idx=config.CHAR2IDX,
            seed=config.SEED,
            is_test=True,
            use_sr=config.USE_SR,
            sr_scale=config.SR_SCALE,
            sr_upsample=config.SR_UPSAMPLE,
        )
        val_loader = None
    else:
        data_root = args.data_root or config.DATA_ROOT
        if not os.path.exists(data_root):
            print(f"❌ Data root not found: {data_root}")
            sys.exit(1)

        dataset = MultiFrameDataset(
            root_dir=data_root,
            mode="val",
            split_ratio=config.SPLIT_RATIO,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            char2idx=config.CHAR2IDX,
            val_split_file=config.VAL_SPLIT_FILE,
            seed=config.SEED,
            use_sr=config.USE_SR,
            sr_scale=config.SR_SCALE,
            sr_upsample=config.SR_UPSAMPLE,
        )
        val_loader = None

    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        sys.exit(1)

    if args.num_workers is not None:
        num_workers = args.num_workers
    elif args.v100:
        num_workers = 8
    elif not torch.cuda.is_available():
        num_workers = 0
    else:
        num_workers = config.NUM_WORKERS

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if args.mode == "val":
        val_loader = loader

    try:
        ckpt = torch.load(args.checkpoint, map_location=config.DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=config.DEVICE)

    # Auto-detect a positional head in the checkpoint so old (pre-Phase-2) and
    # new checkpoints both load cleanly regardless of the config default.
    has_pos_head = any(k.startswith("positional_head") for k in ckpt.keys())
    config.USE_POSITIONAL_HEAD = has_pos_head
    # Auto-detect fusion type (cross-frame has 'fusion.query'/'fusion.attn').
    has_cross_fusion = any(
        k.startswith("fusion.query") or k.startswith("fusion.attn") for k in ckpt.keys()
    )
    config.USE_CROSS_FRAME_FUSION = has_cross_fusion
    # Auto-detect RCAN channel attention in the SR block ('sr_block.body.*.ca.*').
    has_sr_ca = any(k.startswith("sr_block.body") and ".ca." in k for k in ckpt.keys())
    has_sr_mf = any(k.startswith("sr_block.fuse") for k in ckpt.keys())
    config.SR_CHANNEL_ATTENTION = has_sr_ca
    # GroupNorm backbone has no running_mean buffers under 'backbone.'.
    backbone_has_bn = any(
        k.startswith("backbone.") and "running_mean" in k for k in ckpt.keys()
    )
    backbone_norm = "batch" if backbone_has_bn else "group"
    config.BACKBONE_NORM = backbone_norm

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
        sr_channel_attention=has_sr_ca,
        sr_multi_frame=has_sr_mf,
        pretrained_backbone=config.USE_PRETRAINED_BACKBONE,
        use_positional_head=has_pos_head,
        num_positions=config.NUM_POSITIONS,
        use_cross_frame_fusion=has_cross_fusion,
        backbone_norm=backbone_norm,
        stn_shared=args.stn_shared,
    ).to(config.DEVICE)

    print(f"📦 Loading checkpoint: {args.checkpoint}")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   USE_SR: {config.USE_SR}")
    print(f"   USE_LEARNABLE_SR: {config.USE_LEARNABLE_SR}")
    print(f"   POSITIONAL_HEAD: {has_pos_head}")
    print(f"   INFERENCE_TTA: {config.USE_INFERENCE_TTA}")
    print(f"   INFERENCE_BEAM: {config.USE_INFERENCE_BEAM}")
    print(f"   MODE: {args.mode}")
    print(f"   DATA: {data_root}")
    model.load_state_dict(ckpt)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )

    if args.mode == "val":
        # Checkpoint already contains the weights to evaluate; use inference decode.
        metrics, submission_data = trainer.validate(use_ema=False, for_inference=True)
        print(f"\n✅ Validation Results:")
        print(f"   Val Loss: {metrics['loss']:.4f}")
        print(f"   Val Acc:  {metrics['acc']:.2f}%")

        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_name = args.output_name or f"submission_{ckpt_stem}.txt"
        output_path = os.path.join(config.OUTPUT_DIR, output_name)
        with open(output_path, "w") as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} predictions to {output_path}")
    else:
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_name = args.output_name or f"submission_{ckpt_stem}.txt"
        trainer.predict_test(loader, output_filename=output_name)


if __name__ == "__main__":
    main()
