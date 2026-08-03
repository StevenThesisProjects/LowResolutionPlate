#!/usr/bin/env python3
"""Error analysis on the val set: locate WHERE a checkpoint fails, so we invest
in the right lever (decoding vs representation vs overfitting).

Reports, using plain GREEDY CTC decoding (the model's raw behavior):
  * overall greedy accuracy vs position-class-constrained accuracy
    (= how much headroom the layout prior actually has for THIS model)
  * length-mismatch rate (pred length != 7)
  * class-violation rate among 7-length preds (letter where digit expected, ...)
  * per-position error rate (which of the 7 slots fail most)
  * per-layout accuracy (Brazilian LLLDDDD vs Mercosur LLLDLDD, from GT)
  * top character confusions gt->pred, tagged [same-class] / [cross-class]

Example:
    python error_analysis.py --checkpoint results/restran34_poshead_best.pth --v100
"""
import argparse
import os
import sys
from collections import Counter

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config, apply_v100_preset
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


def char_class(ch: str) -> str:
    return 'D' if ch.isdigit() else ('L' if ch.isalpha() else '?')


def layout_of(gt: str) -> str:
    """Infer plate layout from ground truth (differ only at position 4)."""
    if len(gt) != 7:
        return "other"
    return "Mercosur" if gt[4].isalpha() else "Brazilian"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Val error analysis")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--v100", action="store_true")
    p.add_argument("--data-root", default=None)
    p.add_argument("--img-height", type=int, default=None,
                   help="Model input height (must match training, e.g. 48)")
    p.add_argument("--img-width", type=int, default=None,
                   help="Model input width (must match training, e.g. 96)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


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

    state = torch.load(args.checkpoint, map_location=config.DEVICE)
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
        pretrained_backbone=config.USE_PRETRAINED_BACKBONE,
        use_positional_head=has_pos_head,
        num_positions=config.NUM_POSITIONS,
        use_cross_frame_fusion=has_cross_fusion,
        sr_channel_attention=has_sr_ca,
        sr_multi_frame=has_sr_mf,
        backbone_norm=backbone_norm,
    ).to(config.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    idx2char = config.IDX2CHAR
    POS = list(config.PLATE_POS_CLASSES)

    n = 0
    greedy_ok = 0
    posclass_ok = 0
    len_mismatch = 0
    class_violation = 0            # among len==7 greedy preds
    pos_err = [0] * 7              # per-position errors among len==7
    pos_tot = [0] * 7
    layout_ok = Counter()
    layout_tot = Counter()
    confusions = Counter()         # (gt_char, pred_char) among len==7 errors
    same_class_conf = 0
    cross_class_conf = 0
    # Per-layout breakdowns: is the Brazilian deficit concentrated at position 4
    # (the only slot where the two layouts differ) or spread across all slots?
    pos_err_lay = {"Brazilian": [0] * 7, "Mercosur": [0] * 7}
    pos_tot_lay = {"Brazilian": [0] * 7, "Mercosur": [0] * 7}
    pos4_class = Counter()         # (layout, true_class, pred_class) at slot 4

    with torch.no_grad():
        for images, _t, _tl, labels_text, _tid, _hr in loader:
            images = images.to(config.DEVICE, non_blocking=True)
            logits = model(images)
            greedy = decode_with_confidence(logits, idx2char, beam_width=1)
            posc = decode_with_confidence(
                logits, idx2char, beam_width=config.BEAM_WIDTH,
                expected_length=config.PLATE_LENGTH, pos_classes=POS,
            )
            for i, gt in enumerate(labels_text):
                n += 1
                gp = greedy[i][0]
                pp = posc[i][0]
                lay = layout_of(gt)
                layout_tot[lay] += 1
                if gp == gt:
                    greedy_ok += 1
                if pp == gt:
                    posclass_ok += 1
                    layout_ok[lay] += 1
                if len(gp) != 7:
                    len_mismatch += 1
                if len(gp) == 7 and len(gt) == 7:
                    violated = False
                    for k in range(7):
                        pos_tot[k] += 1
                        if lay in pos_tot_lay:
                            pos_tot_lay[lay][k] += 1
                        if gp[k] != gt[k]:
                            pos_err[k] += 1
                            if lay in pos_err_lay:
                                pos_err_lay[lay][k] += 1
                            confusions[(gt[k], gp[k])] += 1
                            if char_class(gp[k]) == char_class(gt[k]):
                                same_class_conf += 1
                            else:
                                cross_class_conf += 1
                        allowed = set(POS[k])
                        if char_class(gp[k]) not in allowed:
                            violated = True
                    # Slot 4 is the ONLY slot where the layouts differ, so track
                    # whether the model picks the right character CLASS there.
                    pos4_class[(lay, char_class(gt[4]), char_class(gp[4]))] += 1
                    if violated:
                        class_violation += 1

    def pct(a, b):
        return (a / b * 100) if b else 0.0

    print("\n" + "=" * 60)
    print(f"VAL ERROR ANALYSIS  ({n} samples)  head={has_pos_head}")
    print("=" * 60)
    print(f"Greedy CTC acc            : {pct(greedy_ok, n):6.2f}%")
    print(f"+pos-class beam acc       : {pct(posclass_ok, n):6.2f}%  "
          f"(headroom from prior: {pct(posclass_ok - greedy_ok, n):+.2f})")
    print(f"Length-mismatch (pred!=7) : {pct(len_mismatch, n):6.2f}%  ({len_mismatch})")
    print(f"Class-violation (len7)    : {pct(class_violation, n):6.2f}%  ({class_violation})")
    print("-" * 60)
    print("Per-position error rate (greedy, len7 preds):")
    for k in range(7):
        cls = POS[k]
        print(f"  pos{k} [{cls:>2}] : {pct(pos_err[k], pos_tot[k]):6.2f}%  "
              f"({pos_err[k]}/{pos_tot[k]})")
    print("-" * 60)
    print("Per-layout accuracy (+pos-class):")
    for lay in ("Brazilian", "Mercosur", "other"):
        if layout_tot[lay]:
            print(f"  {lay:<10}: {pct(layout_ok[lay], layout_tot[lay]):6.2f}%  "
                  f"({layout_ok[lay]}/{layout_tot[lay]})")
    print("-" * 60)
    print("Per-position error rate BY LAYOUT (greedy, len7):")
    print(f"  {'slot':<10}{'Brazilian':>16}{'Mercosur':>16}")
    for k in range(7):
        b = pct(pos_err_lay['Brazilian'][k], pos_tot_lay['Brazilian'][k])
        m = pct(pos_err_lay['Mercosur'][k], pos_tot_lay['Mercosur'][k])
        mark = "  <-- layout-specific slot" if k == 4 else ""
        print(f"  pos{k} [{POS[k]:>2}]{b:>14.2f}%{m:>15.2f}%{mark}")
    print("-" * 60)
    print("Slot-4 character CLASS decision (the only slot layouts differ):")
    for lay in ("Brazilian", "Mercosur"):
        sub = {(t, p): c for (l, t, p), c in pos4_class.items() if l == lay}
        s = sum(sub.values())
        if not s:
            continue
        wrong_cls = sum(c for (t, p), c in sub.items() if t != p)
        print(f"  {lay:<10} (true class '{'D' if lay=='Brazilian' else 'L'}'): "
              f"predicted WRONG class {pct(wrong_cls, s):.2f}% ({wrong_cls}/{s})")
        for (t, p), c in sorted(sub.items(), key=lambda kv: -kv[1]):
            flag = "" if t == p else "   <-- wrong class"
            print(f"      true={t} pred={p}: {c}{flag}")
    print("-" * 60)
    tot_conf = same_class_conf + cross_class_conf
    print(f"Char confusions: same-class={pct(same_class_conf, tot_conf):.1f}%  "
          f"cross-class={pct(cross_class_conf, tot_conf):.1f}%  (total {tot_conf})")
    print("Top 15 confusions gt->pred:")
    for (g, p), c in confusions.most_common(15):
        tag = "same " if char_class(g) == char_class(p) else "CROSS"
        print(f"  {g} -> {p}  x{c:<4} [{tag}]")
    print("=" * 60)
    print("READING GUIDE:")
    print("  * high cross-class + big pos-class headroom -> decoding still helps")
    print("  * dominated by same-class confusions        -> REPRESENTATION bound")
    print("    (need better features / real SR, not decoding)")
    print("  * one position dominates errors             -> targeted fix there")


if __name__ == "__main__":
    main()
