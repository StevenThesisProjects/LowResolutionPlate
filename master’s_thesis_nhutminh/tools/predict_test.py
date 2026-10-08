#!/usr/bin/env python3
"""Chạy inference test set trên một checkpoint ĐÃ CÓ — không train lại gì cả.

Đây là "Phương án A" trong report/paper/paper_revision_plan.md §3: dùng lại đúng
model đã được validate (biết chắc nó đạt bao nhiêu trên val) thay vì
`train.py --submission-mode` vốn train lại từ đầu trên toàn bộ 20.000 track và
tắt early stopping — nghĩa là ship ra một model không có cách nào kiểm chứng.

Kiến trúc suy ngược từ `state_dict` nên không cần nhớ lại tổ hợp flag lúc train,
trừ `--width-downsample` (không nằm trong state_dict).

Ví dụ:
    # Test public (1.000 track) trên checkpoint S1 seed 42
    python tools/predict_test.py \
      --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
      --output results/submission_s1_seed42_public.txt

    # Test blind (3.000 track)
    python tools/predict_test.py \
      --checkpoint results/multi-seed/s1_mf_sr_ocr/s1_seed42_best.pth \
      --data-root dataset/TKzFBtn7-test-blind/TKzFBtn7-test-blind \
      --output results/submission_s1_seed42_blind.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.utils.postprocess import PlateLayout, decode_batch
from tools.eval_decode import infer_architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference test set trên checkpoint có sẵn")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True,
                        help="File submission đầu ra (mỗi dòng: track_id,TEXT;conf)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Mặc định = TEST_DATA_ROOT (test public). Đổi sang test blind nếu cần")
    parser.add_argument("--decode", choices=["greedy", "constrained"], default="constrained",
                        help="Phải khớp lúc train — S1/J1/S4 đều dùng constrained")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--width-downsample", type=int, choices=[4, 8], default=8,
                        help="BẮT BUỘC = 4 với S4 (sr_scale=1); S1/J1 dùng mặc định 8")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    config.WIDTH_DOWNSAMPLE = args.width_downsample
    data_root = args.data_root or config.TEST_DATA_ROOT

    if not os.path.exists(data_root):
        raise SystemExit(f"❌ Không tìm thấy test data: {data_root}")

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise SystemExit(f"❌ Không tìm thấy checkpoint: {args.checkpoint}")
    checkpoint_path = matches[0]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    state = torch.load(checkpoint_path, map_location="cpu")
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
    model = model.to(device).eval()
    print(f"📦 {os.path.basename(checkpoint_path)}")
    print(f"   {', '.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}")
    print(f"   Decode: {args.decode} | width_downsample={config.WIDTH_DOWNSAMPLE} | {device}")

    # `is_test=True` — không có nhãn, dataset chỉ trả về ảnh + track_id.
    dataset = MultiFrameDataset(
        data_root, mode="val", is_test=True,
        img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX, num_frames=config.NUM_FRAMES,
    )
    if len(dataset) == 0:
        raise SystemExit(f"❌ Test dataset rỗng: {data_root}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    layout = PlateLayout.from_spec(config.PLATE_LAYOUTS)
    rows = []
    with torch.no_grad():
        for batch in loader:
            images, _, _, _, track_ids = batch[:5]
            log_probs = model(images.to(device))
            decoded = decode_batch(
                log_probs, config.IDX2CHAR, mode=args.decode,
                layout=layout, beam_width=args.beam_width,
            )
            rows.extend(
                f"{tid},{text};{conf:.4f}" for tid, (text, conf) in zip(track_ids, decoded)
            )

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output, "w") as handle:
        handle.write("\n".join(rows))
    print(f"✅ {len(rows)} dòng → {args.output}")

    # Kiểm tra nhanh: số dòng phải khớp số track của tập test đang chấm.
    print(f"   (test public = 1.000 track · test blind = 3.000 track)")


if __name__ == "__main__":
    main()
