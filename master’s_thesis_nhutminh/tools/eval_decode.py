#!/usr/bin/env python3
"""Chấm lại một checkpoint đã train với greedy vs constrained decode.

Không train lại gì cả — chỉ chạy forward trên validation set rồi decode 2 kiểu,
nên biết ngay ràng buộc layout đáng giá bao nhiêu track trước khi tiêu thêm giờ
GPU nào. Cũng hỗ trợ ensemble nhiều checkpoint (trung bình log-prob).

Kiến trúc được suy ra từ chính state_dict, nên checkpoint cũ (STN pool 1x1,
SR per-frame) vẫn load được mà không cần nhớ đã train bằng flag gì.

Ví dụ:
    python tools/eval_decode.py --checkpoint results/crnn_resblock_sr_supervised_best.pth
    python tools/eval_decode.py --checkpoint results/best_seed*.pth   # ensemble
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.utils.postprocess import PlateLayout, decode_batch


def infer_architecture(state: dict, config: Config) -> dict:
    """Đọc ngược cấu hình model từ shape của state_dict.

    Rẻ hơn nhiều so với bắt người dùng nhớ lại tổ hợp flag của một run cũ, và
    không thể sai lệch — mọi thứ suy ra đều là ràng buộc cứng của tensor.
    """
    kwargs = {
        "use_stn": any(k.startswith("stn.") for k in state),
        "use_sr": any(k.startswith("sr.") for k in state),
        "use_dcn": any(k.startswith("dcn.") for k in state),
        "sr_multi_frame": any(k.startswith("sr.temporal_fuse") for k in state),
        "backbone_norm": "group" if any(".norm1.weight" in k for k in state) else "none",
        "use_se": any(".attn.fc.0.weight" in k for k in state),
    }

    # STN pool: Linear đầu tiên có in_features = 64 * pool_h * pool_w.
    # (1,1) đồng thời chọn lại topology cũ, nên checkpoint trước bản sửa vẫn
    # chạy đúng như lúc train.
    if kwargs["use_stn"]:
        cells = state["stn.fc.0.weight"].shape[1] // 64
        kwargs["stn_pool"] = (1, 1) if cells == 1 else (4, cells // 4)

    if kwargs["use_sr"]:
        hidden = state["sr.head.weight"].shape[0]
        kwargs["sr_hidden_channels"] = hidden
        # PixelShuffle: out_channels = hidden * scale^2
        shuffle_out = state["sr.upsample.0.weight"].shape[0]
        kwargs["sr_scale"] = int(round((shuffle_out / hidden) ** 0.5))
        kwargs["sr_num_blocks"] = len({k.split(".")[2] for k in state if k.startswith("sr.body.")})
    return kwargs


def load_models(paths: list[str], config: Config) -> list[MultiFrameCRNN]:
    models = []
    for path in paths:
        state = torch.load(path, map_location="cpu")
        kwargs = infer_architecture(state, config)
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            backbone_channels=config.BACKBONE_CHANNELS,
            backbone_base_channels=config.BACKBONE_BASE_CHANNELS,
            backbone_blocks=config.BACKBONE_STAGE_BLOCKS,
            backbone_stage_channels=config.BACKBONE_STAGE_CHANNELS,
            residual_scale=config.BACKBONE_RES_SCALE,
            fusion_mode=config.FUSION_MODE,
            width_downsample=config.WIDTH_DOWNSAMPLE,
            **kwargs,
        )
        model.load_state_dict(state)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        print(f"  ✓ {os.path.basename(path)}  [{summary}]")
        models.append(model.to(config.DEVICE).eval())
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy vs constrained decode trên checkpoint đã train")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Một hoặc nhiều .pth (nhiều => ensemble trung bình log-prob)")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--plate-layouts", type=str, default=None)
    parser.add_argument("--width-downsample", type=int, choices=[4, 8], default=8,
                        help="Phải khớp với lúc train (stride không nằm trong state_dict)")
    parser.add_argument("--save-submission", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    config.WIDTH_DOWNSAMPLE = args.width_downsample
    if args.data_root:
        config.DATA_ROOT = args.data_root
    layout = PlateLayout.from_spec(args.plate_layouts or config.PLATE_LAYOUTS)

    paths = sorted({p for pattern in args.checkpoint for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"❌ Không tìm thấy checkpoint nào khớp {args.checkpoint}")

    print(f"📦 Load {len(paths)} checkpoint:")
    models = load_models(paths, config)

    val_ds = MultiFrameDataset(
        config.DATA_ROOT,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        num_frames=config.NUM_FRAMES,
    )
    if len(val_ds) == 0:
        raise SystemExit("❌ Validation set rỗng.")
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=args.num_workers,
    )

    counts = {"greedy": 0, "constrained": 0}
    total = 0
    timings = {"greedy": 0.0, "constrained": 0.0}
    submissions: list[str] = []
    bad_format = 0
    recovered: list[tuple[str, str, str]] = []

    with torch.no_grad():
        for images, _, _, labels_text, track_ids in loader:
            images = images.to(config.DEVICE)
            # Ensemble ở mức xác suất, không phải mức chuỗi: trung bình log-prob
            # giữ nguyên phân phối cho beam search dùng tiếp.
            stacked = torch.stack([m(images) for m in models], dim=0)
            preds = torch.logsumexp(stacked, dim=0) - torch.log(
                torch.tensor(float(len(models)), device=stacked.device)
            )

            decoded = {}
            for mode in ("greedy", "constrained"):
                start = time.time()
                decoded[mode] = decode_batch(
                    preds, config.IDX2CHAR, mode=mode, layout=layout, beam_width=args.beam_width
                )
                timings[mode] += time.time() - start

            for i, label in enumerate(labels_text):
                greedy_text = decoded["greedy"][i][0]
                constrained_text, confidence = decoded["constrained"][i]
                counts["greedy"] += greedy_text == label
                counts["constrained"] += constrained_text == label
                if len(greedy_text) != layout.length:
                    bad_format += 1
                if greedy_text != label and constrained_text == label:
                    recovered.append((label, greedy_text, constrained_text))
                submissions.append(f"{track_ids[i]},{constrained_text};{confidence:.4f}")
            total += len(labels_text)

    print(f"\n{'='*64}\nValidation: {total} track"
          + (f" | ensemble {len(models)} checkpoint" if len(models) > 1 else ""))
    print(f"{'='*64}")
    for mode in ("greedy", "constrained"):
        acc = counts[mode] / total * 100
        print(f"  {mode:12s} {acc:6.2f}%   ({counts[mode]}/{total} track)"
              f"   {timings[mode]/total*1000:6.2f} ms/track")
    delta = counts["constrained"] - counts["greedy"]
    print(f"\n  Chênh lệch : {delta:+d} track ({delta/total*100:+.2f} điểm)")
    print(f"  Greedy sai độ dài (≠{layout.length}): {bad_format} track")
    if recovered:
        print(f"\n  Ví dụ được constrained decode cứu ({len(recovered)} track):")
        for label, greedy_text, constrained_text in recovered[:10]:
            print(f"    {label}  greedy={greedy_text:<10s} -> constrained={constrained_text}")

    if args.save_submission:
        os.makedirs(os.path.dirname(args.save_submission) or ".", exist_ok=True)
        with open(args.save_submission, "w") as handle:
            handle.write("\n".join(submissions))
        print(f"\n📝 Đã lưu {len(submissions)} dòng → {args.save_submission}")


if __name__ == "__main__":
    main()
