#!/usr/bin/env python3
"""Ensemble evaluation: average the CTC posteriors of several trained
checkpoints (same architecture / sequence length) and decode with the
layout-aware length-7 + position-class prior.

Posterior averaging across independently-trained models is the most reliable
way to cross a stubborn single-model ceiling (+1-2% typical). Works on the
checkpoints you already have — no retraining.

Example:
    python ensemble_eval.py --v100 \
        --checkpoints results/restran34_sr2x_best.pth \
                      results/restran34_sr2x_light_best.pth
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
    p = argparse.ArgumentParser(description="Ensemble val evaluation")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Two or more .pth checkpoints (same architecture)")
    p.add_argument("--v100", action="store_true")
    p.add_argument("--data-root", default=None)
    p.add_argument("--img-height", type=int, default=None,
                   help="Model input height (must match training, e.g. 48)")
    p.add_argument("--img-width", type=int, default=None,
                   help="Model input width (must match training, e.g. 96)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--tta", action="store_true",
                   help="Use brightness TTA per model before averaging")
    p.add_argument("--no-single", action="store_true",
                   help="Skip per-model accuracy (ensemble only)")
    return p.parse_args()


def build_model(config, state):
    has_pos_head = any(k.startswith("positional_head") for k in state.keys())
    has_cross_fusion = any(k.startswith("fusion.query") or k.startswith("fusion.attn") for k in state.keys())
    has_sr_ca = any(k.startswith("sr_block.body") and ".ca." in k for k in state.keys())
    has_sr_mf = any(k.startswith("sr_block.fuse") for k in state.keys())
    backbone_norm = "batch" if any(k.startswith("backbone.") and "running_mean" in k for k in state.keys()) else "group"
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
        pretrained_backbone=False,
        use_positional_head=has_pos_head,
        num_positions=config.NUM_POSITIONS,
        use_cross_frame_fusion=has_cross_fusion,
        sr_channel_attention=has_sr_ca,
        sr_multi_frame=has_sr_mf,
        backbone_norm=backbone_norm,
    ).to(config.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def model_probs(model, images, use_tta):
    """Return probability tensor [B, T, C] for one model."""
    if use_tta:
        log_preds = forward_with_tta(model, images)
    else:
        log_preds = model(images)
    return log_preds.exp()


def main() -> None:
    args = parse_args()
    for c in args.checkpoints:
        if not os.path.exists(c):
            print(f"❌ Checkpoint not found: {c}")
            sys.exit(1)
    if len(args.checkpoints) < 2:
        print("⚠️ Only one checkpoint given — ensemble needs ≥2.")

    config = Config()
    if args.v100:
        apply_v100_preset(config)
    if args.img_height is not None:
        config.IMG_HEIGHT = args.img_height
    if args.img_width is not None:
        config.IMG_WIDTH = args.img_width
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    seed_everything(config.SEED)

    dataset = MultiFrameDataset(
        root_dir=args.data_root or config.DATA_ROOT,
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
        print("❌ Val dataset empty!")
        sys.exit(1)

    nw = args.num_workers if args.num_workers is not None else (8 if args.v100 else 0)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn, num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )

    models = []
    for ckpt in args.checkpoints:
        state = torch.load(ckpt, map_location=config.DEVICE)
        models.append((os.path.basename(ckpt), build_model(config, state)))
    print(f"📦 Loaded {len(models)} models | val samples: {len(dataset)} | "
          f"TTA={args.tta} | beam={config.BEAM_WIDTH}")

    idx2char = config.IDX2CHAR
    L = getattr(config, "PLATE_LENGTH", 0) or None
    pos = list(getattr(config, "PLATE_POS_CLASSES", ())) or None
    dkw = dict(beam_width=config.BEAM_WIDTH, expected_length=L, pos_classes=pos)

    single_correct = [0] * len(models)
    ens_correct = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        for images, _t, _tl, labels_text, _tid, _hr in loader:
            images = images.to(config.DEVICE, non_blocking=True)
            prob_sum = None
            T_ref = None
            for mi, (_name, model) in enumerate(models):
                probs = model_probs(model, images, args.tta)  # [B,T,C]
                if T_ref is None:
                    T_ref = probs.size(1)
                elif probs.size(1) != T_ref:
                    print(f"❌ Sequence-length mismatch between models "
                          f"({probs.size(1)} vs {T_ref}) — need same architecture "
                          f"(same SR_UPSAMPLE). Aborting.")
                    sys.exit(1)
                prob_sum = probs if prob_sum is None else prob_sum + probs
                if not args.no_single:
                    log_p = probs.clamp(min=1e-8).log()
                    dec = decode_with_confidence(log_p, idx2char, **dkw)
                    for i, (txt, _c) in enumerate(dec):
                        if txt == labels_text[i]:
                            single_correct[mi] += 1

            avg_log = (prob_sum / len(models)).clamp(min=1e-8).log()
            ens = decode_with_confidence(avg_log, idx2char, **dkw)
            for i, (txt, _c) in enumerate(ens):
                if txt == labels_text[i]:
                    ens_correct += 1
            total += len(labels_text)

    dt = time.time() - t0
    print("\n" + "=" * 62)
    print(f"{'Model':<46}{'Val Acc':>10}")
    print("-" * 62)
    if not args.no_single:
        for (name, _), c in zip(models, single_correct):
            print(f"{name:<46}{c / total * 100:>9.2f}%")
        print("-" * 62)
    print(f"{'ENSEMBLE (posterior mean, ' + str(len(models)) + ' models)':<46}"
          f"{ens_correct / total * 100:>9.2f}%")
    print("=" * 62)
    print(f"Eval time: {dt:.1f}s")


if __name__ == "__main__":
    main()
