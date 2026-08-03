#!/usr/bin/env python3
"""Decode-ablation evaluator: score ONE checkpoint on the val set under several
decoding strategies, on identical model logits, to produce a clean ablation
table (greedy vs beam vs +length-7 vs +position-class, with/without TTA).

This measures the effect of the layout-aware constrained decoding WITHOUT any
retraining. Run on the machine that has the trained checkpoint + torchvision.

Example:
    python eval_ablation.py \
        --checkpoint results/restran34_sr_pixel_l1_best.pth --v100
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config, apply_v100_preset
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence, forward_with_tta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode ablation on the val set")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--v100", action="store_true", help="Apply V100 preset config")
    p.add_argument("--data-root", default=None, help="Override val data root")
    p.add_argument("--img-height", type=int, default=None,
                   help="Model input height (must match training, e.g. 48)")
    p.add_argument("--img-width", type=int, default=None,
                   help="Model input width (must match training, e.g. 96)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--stn-shared", action="store_true",
                   help="Model was trained with shared per-track STN")
    p.add_argument("--beam-width", type=int, default=None,
                   help="Override beam width for the beam variants")
    return p.parse_args()


def build_variants(config):
    """CTC decode variants: (name, needs_tta, kwargs for decode_with_confidence)."""
    bw = config.BEAM_WIDTH
    L = getattr(config, "PLATE_LENGTH", 0) or None
    pos = list(getattr(config, "PLATE_POS_CLASSES", ())) or None
    return [
        ("greedy",                       False, dict(beam_width=1)),
        (f"beam{bw}",                     False, dict(beam_width=bw)),
        (f"beam{bw}+len7",                False, dict(beam_width=bw, expected_length=L)),
        (f"beam{bw}+len7+posclass",       False, dict(beam_width=bw, expected_length=L, pos_classes=pos)),
        (f"beam{bw}+len7+posclass+TTA",   True,  dict(beam_width=bw, expected_length=L, pos_classes=pos)),
    ]


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
    if args.beam_width is not None:
        config.BEAM_WIDTH = args.beam_width
    seed_everything(config.SEED)

    data_root = args.data_root or config.DATA_ROOT
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
    if len(dataset) == 0:
        print("❌ Val dataset is empty!")
        sys.exit(1)

    num_workers = args.num_workers if args.num_workers is not None else (
        8 if args.v100 else (0 if not torch.cuda.is_available() else config.NUM_WORKERS)
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Auto-detect whether the checkpoint has a positional head so we can score
    # both pre-Phase-2 and Phase-2 checkpoints with the same script.
    state = torch.load(args.checkpoint, map_location=config.DEVICE)
    has_pos_head = any(k.startswith("positional_head") for k in state.keys())
    has_cross_fusion = any(k.startswith("fusion.query") or k.startswith("fusion.attn") for k in state.keys())
    has_sr_ca = any(k.startswith("sr_block.body") and ".ca." in k for k in state.keys())
    has_sr_mf = any(k.startswith("sr_block.fuse") for k in state.keys())
    backbone_norm = "batch" if any(k.startswith("backbone.") and "running_mean" in k for k in state.keys()) else "group"
    print(f"🔎 Checkpoint positional head: {has_pos_head}")

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
        pretrained_backbone=config.USE_PRETRAINED_BACKBONE,
        use_positional_head=has_pos_head,
        num_positions=config.NUM_POSITIONS,
        use_cross_frame_fusion=has_cross_fusion,
        sr_channel_attention=has_sr_ca,
        sr_multi_frame=has_sr_mf,
        backbone_norm=backbone_norm,
        stn_shared=args.stn_shared,
    ).to(config.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"📦 Loaded {args.checkpoint} | val samples: {len(dataset)} | "
          f"beam={config.BEAM_WIDTH}")
    # The architecture is input-size agnostic (adaptive pooling), so a size
    # mismatch loads cleanly but silently destroys accuracy. Make it visible.
    print(f"🖼️  INPUT SIZE: {config.IMG_HEIGHT}x{config.IMG_WIDTH} "
          f"-> SR x{config.SR_UPSAMPLE} -> "
          f"{config.IMG_HEIGHT * config.SR_UPSAMPLE}x{config.IMG_WIDTH * config.SR_UPSAMPLE}"
          f"   ⚠️ MUST match training (--img-height/--img-width)")

    variants = build_variants(config)
    L = getattr(config, "PLATE_LENGTH", 0) or None
    pos = list(getattr(config, "PLATE_POS_CLASSES", ())) or None
    pos_idx2char = {i: config.CHARS[i] for i in range(len(config.CHARS))}
    # Extra head-based variants (only meaningful when a positional head exists).
    pos_variants = []
    if has_pos_head:
        pos_variants = [
            "positional-only+posclass",
            f"FUSED(beam{config.BEAM_WIDTH}+len7+posclass | positional)",
        ]

    from src.utils.postprocess import decode_positional, fuse_ctc_positional

    correct = {name: 0 for name, _, _ in variants}
    for name in pos_variants:
        correct[name] = 0
    total = 0
    idx2char = config.IDX2CHAR
    t0 = time.time()

    with torch.no_grad():
        for images, _t, _tl, labels_text, _tid, _hr in loader:
            images = images.to(config.DEVICE, non_blocking=True)
            if has_pos_head:
                base_logits, pos_logits = model(images, return_pos=True)
            else:
                base_logits = model(images)
                pos_logits = None
            tta_logits = None  # computed lazily if any TTA variant needs it

            for name, needs_tta, kwargs in variants:
                if needs_tta:
                    if tta_logits is None:
                        tta_logits = forward_with_tta(model, images)
                    logits = tta_logits
                else:
                    logits = base_logits
                decoded = decode_with_confidence(logits, idx2char, **kwargs)
                for i, (pred_text, _c) in enumerate(decoded):
                    if pred_text == labels_text[i]:
                        correct[name] += 1

            if has_pos_head:
                pos_only = decode_positional(pos_logits, pos_idx2char, pos_classes=pos)
                ctc_full = decode_with_confidence(
                    base_logits, idx2char, beam_width=config.BEAM_WIDTH,
                    expected_length=L, pos_classes=pos,
                )
                fused = fuse_ctc_positional(ctc_full, pos_only)
                for i in range(len(labels_text)):
                    if pos_only[i][0] == labels_text[i]:
                        correct[pos_variants[0]] += 1
                    if fused[i][0] == labels_text[i]:
                        correct[pos_variants[1]] += 1

            total += len(labels_text)

    all_names = [n for n, _, _ in variants] + pos_variants

    dt = time.time() - t0
    width = max(44, max(len(n) for n in all_names) + 2)
    print("\n" + "=" * (width + 24))
    print(f"{'Decode variant':<{width}}{'Val Acc':>10}{'Correct':>14}")
    print("-" * (width + 24))
    base_acc = None
    for name in all_names:
        acc = correct[name] / total * 100
        if base_acc is None:
            base_acc = acc
            delta = ""
        else:
            delta = f"  ({acc - base_acc:+.2f})"
        print(f"{name:<{width}}{acc:>9.2f}%{correct[name]:>8}/{total}{delta}")
    print("=" * (width + 24))
    print(f"Total eval time: {dt:.1f}s  (delta vs '{all_names[0]}')")

    best = max(correct[n] / total * 100 for n in all_names)
    if best < 5.0:
        print(
            "\n🚨 Near-zero accuracy — this is almost always a CONFIG mismatch,\n"
            "   not a broken model. Check, in order:\n"
            f"   1. Input size: eval used {config.IMG_HEIGHT}x{config.IMG_WIDTH}. Does that\n"
            "      match the size this checkpoint was TRAINED at? Pass\n"
            "      --img-height/--img-width to match (e.g. 48 / 96).\n"
            "   2. --stn-shared: pass it if the model was trained with shared STN\n"
            "      (it adds no parameters, so it cannot be auto-detected).\n"
            f"   3. SR_UPSAMPLE (currently {config.SR_UPSAMPLE}) must match training."
        )


if __name__ == "__main__":
    main()
