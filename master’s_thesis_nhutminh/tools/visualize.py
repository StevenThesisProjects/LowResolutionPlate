#!/usr/bin/env python3
"""Xuất ảnh minh hoạ định tính: LR input -> SR output -> attention weights -> prediction.

Phục vụ mục "Qualitative Visualizations" trong issue #9. Dùng cv2 để ghép ảnh
(đã là dependency sẵn), không cần matplotlib.

Ví dụ:
    python tools/visualize.py \
      --checkpoint results/crnn_resblock_sr_supervised_best.pth \
      --use-sr --backbone-norm group \
      --num-samples 8 --output-dir results/viz
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.utils.postprocess import PlateLayout, decode_batch


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] chuẩn hoá về [-1,1] -> ảnh BGR uint8 cho cv2."""
    array = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    array = np.clip((array * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def label_strip(width: int, text: str, height: int = 22, color=(30, 30, 30)) -> np.ndarray:
    strip = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(strip, text, (4, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return strip


def stack_labeled(image: np.ndarray, text: str) -> np.ndarray:
    return np.vstack([label_strip(image.shape[1], text), image])


def build_panel(frames, sr_frames, weights, pred, conf, truth, cell_w: int = 220) -> np.ndarray:
    """Ghép 1 panel: hàng trên = LR từng frame (kèm attention), hàng dưới = SR."""
    def resize(img):
        scale = cell_w / img.shape[1]
        return cv2.resize(img, (cell_w, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)

    top_cells, bottom_cells = [], []
    for idx, frame in enumerate(frames):
        img = resize(denormalize(frame))
        weight = weights[idx] if weights is not None else None
        # Viền dày/xanh hơn = attention cao hơn, thấy ngay frame nào được tin cậy.
        if weight is not None:
            thickness = 1 + int(round(weight * len(frames) * 3))
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 160, 0), thickness)
            caption = f"LR#{idx + 1}  attn={weight:.3f}"
        else:
            caption = f"LR#{idx + 1}"
        top_cells.append(stack_labeled(img, caption))

        if sr_frames is not None:
            bottom_cells.append(stack_labeled(resize(denormalize(sr_frames[idx])), f"SR#{idx + 1}"))

    def pad_to_max(cells):
        height = max(c.shape[0] for c in cells)
        return [np.vstack([c, np.full((height - c.shape[0], c.shape[1], 3), 245, dtype=np.uint8)])
                if c.shape[0] < height else c for c in cells]

    rows = [np.hstack(pad_to_max(top_cells))]
    if bottom_cells:
        rows.append(np.hstack(pad_to_max(bottom_cells)))

    panel = np.vstack(rows)
    ok = pred == truth
    verdict = "ĐÚNG" if ok else "SAI"
    header = label_strip(
        panel.shape[1],
        f"GT: {truth}   |   Pred: {pred}  (conf={conf:.3f})   |   {verdict}",
        height=28,
        color=(0, 130, 0) if ok else (0, 0, 200),
    )
    return np.vstack([header, panel])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/viz")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--only-errors", action="store_true", help="Chỉ xuất case dự đoán SAI")
    parser.add_argument("--preset", choices=["stable", "strong", "debug"], default="stable")
    parser.add_argument("--use-sr", action="store_true")
    parser.add_argument("--use-dcn", action="store_true")
    parser.add_argument("--backbone-norm", choices=["none", "group"], default=None)
    parser.add_argument("--fusion-mode", choices=["attention", "avg", "max"], default=None)
    parser.add_argument("--sr-scale", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--decode", choices=["greedy", "constrained"], default=None)
    parser.add_argument("--width-downsample", type=int, choices=[4, 8], default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import train as train_module

    config = Config()
    train_module._apply_preset(config, args.preset)
    if args.use_sr:
        config.USE_SR = True
    if args.use_dcn:
        config.USE_DCN = True
    for attr, value in [
        ("BACKBONE_NORM", args.backbone_norm), ("FUSION_MODE", args.fusion_mode),
        ("SR_SCALE", args.sr_scale), ("NUM_FRAMES", args.num_frames), ("DATA_ROOT", args.data_root),
        ("DECODE_MODE", args.decode), ("WIDTH_DOWNSAMPLE", args.width_downsample),
    ]:
        if value is not None:
            setattr(config, attr, value)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = MultiFrameCRNN(
        num_classes=config.NUM_CLASSES, hidden_size=config.HIDDEN_SIZE,
        rnn_dropout=config.RNN_DROPOUT, use_stn=config.USE_STN,
        backbone_channels=config.BACKBONE_CHANNELS,
        backbone_base_channels=config.BACKBONE_BASE_CHANNELS,
        backbone_blocks=config.BACKBONE_STAGE_BLOCKS,
        backbone_stage_channels=config.BACKBONE_STAGE_CHANNELS,
        use_se=config.BACKBONE_USE_SE, residual_scale=config.BACKBONE_RES_SCALE,
        frame_dropout=0.0, use_sr=config.USE_SR, sr_scale=config.SR_SCALE,
        sr_hidden_channels=config.SR_HIDDEN_CHANNELS, sr_num_blocks=config.SR_NUM_BLOCKS,
        sr_res_scale=config.SR_RES_SCALE, backbone_norm=config.BACKBONE_NORM,
        use_dcn=config.USE_DCN, dcn_hidden_channels=config.DCN_HIDDEN_CHANNELS,
        fusion_mode=config.FUSION_MODE, sr_multi_frame=config.SR_MULTI_FRAME,
        stn_pool=config.STN_POOL, width_downsample=config.WIDTH_DOWNSAMPLE,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    dataset = MultiFrameDataset(
        config.DATA_ROOT, mode="val", split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX, val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED, num_frames=config.NUM_FRAMES,
    )
    if len(dataset) == 0:
        print("❌ Validation dataset rỗng.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0
    for idx in range(len(dataset)):
        if saved >= args.num_samples:
            break
        images, _, _, truth, track_id = dataset[idx][:5]
        batch = images.unsqueeze(0).to(device)

        with torch.no_grad():
            num_frames = images.size(0)
            if config.USE_SR:
                log_probs, sr_output, _, _ = model(batch, return_sr=True)
                sr_frames = sr_output.view(1, num_frames, *sr_output.shape[1:])[0]
            else:
                log_probs, sr_frames = model(batch), None

            # Lấy lại attention weight ở mức feature để biết frame nào được tin cậy.
            weights = None
            if model.fusion.mode == "attention":
                flat = batch.view(-1, *batch.shape[2:])
                if model.use_stn:
                    flat, _ = model._apply_stn(flat)
                if model.use_dcn:
                    flat = model.dcn(flat.view(1, num_frames, *flat.shape[1:])).reshape(flat.shape)
                if model.use_sr:
                    flat = model.sr(flat, num_frames=num_frames)
                feats = model.backbone(flat)
                feats = feats.view(1, num_frames, model.cnn_channels, feats.size(2), feats.size(3))
                weights = model.fusion.last_weights(feats)[0].cpu().tolist()

        pred, conf = decode_batch(
            log_probs, config.IDX2CHAR, mode=config.DECODE_MODE,
            layout=PlateLayout.from_spec(config.PLATE_LAYOUTS),
        )[0]
        if args.only_errors and pred == truth:
            continue

        panel = build_panel(images, sr_frames, weights, pred, conf, truth)
        status = "ok" if pred == truth else "err"
        path = os.path.join(args.output_dir, f"{saved:02d}_{status}_{track_id}.png")
        cv2.imwrite(path, panel)
        saved += 1

    print(f"✅ Đã lưu {saved} ảnh minh hoạ → {args.output_dir}/")


if __name__ == "__main__":
    main()
