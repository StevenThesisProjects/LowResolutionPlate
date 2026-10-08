#!/usr/bin/env python3
"""Xuất figure định tính 4 cột cho paper (review Bước 3).

Mỗi track = 1 hàng, 4 cột:  I_LR -> I_SR -> Attention heatmap -> Prediction.
Chọn 5 case đúng + 5 case sai, ghép thành 1 ảnh grid dùng thẳng cho Figure 4,
đồng thời lưu từng track riêng để chọn lại thủ công nếu cần.

Khác `tools/visualize.py`: file đó xuất mỗi track một panel dạng 2 hàng (LR/SR)
để soi lỗi lúc debug. File này bố cục theo cột và gộp nhiều track vào một ảnh —
đúng dạng hình thường thấy trong paper.

"Tiêu biểu" ở đây = case đúng có confidence CAO nhất và case sai có confidence
THẤP nhất; chọn kiểu này thì hình phản ánh đúng hai đầu của phân phối thay vì
mấy track đầu danh sách. Đổi bằng `--pick first` nếu muốn lấy tuần tự.

Ví dụ:
    python tools/visualize_paper_figures.py \
      --checkpoint results/s4_sr_scale1_best.pth \
      --width-downsample 4 --output-dir results/paper_figures
"""
from __future__ import annotations

import argparse
import glob
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
from tools.eval_decode import infer_architecture
from tools.visualize import denormalize

FRAME_H = 46
GAP = 4
COL_LR = 210
COL_SR = 210
COL_ATTN = 170
COL_PRED = 300
# 2 dòng: track_id ở trên, nhãn cột ở dưới — 1 dòng thì track_id đè lên nhãn "I_LR".
HEADER_H = 44
PAD = 8
BG = 250


def blank(height: int, width: int) -> np.ndarray:
    return np.full((height, width, 3), BG, dtype=np.uint8)


def put_text(canvas, text, org, scale=0.42, color=(35, 35, 35), thickness=1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_frame(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize giữ tỷ lệ rồi canh giữa trong ô cố định, để các hàng thẳng cột."""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    cell = blank(height, width)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    cell[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return cell


def frame_column(frames, width: int, num_frames: int) -> np.ndarray:
    """Xếp `num_frames` ảnh theo chiều dọc trong một cột."""
    column = blank(num_frames * (FRAME_H + GAP), width)
    for idx in range(num_frames):
        y0 = idx * (FRAME_H + GAP)
        if frames is None:
            put_text(column, "(khong co SR)", (6, y0 + FRAME_H // 2), scale=0.4, color=(120, 120, 120))
            continue
        column[y0:y0 + FRAME_H] = fit_frame(denormalize(frames[idx]), width, FRAME_H)
    return column


def attention_column(weights, width: int, num_frames: int) -> np.ndarray:
    """Thanh ngang dài/đậm theo trọng số attention của từng frame."""
    column = blank(num_frames * (FRAME_H + GAP), width)
    if weights is None:
        put_text(column, "(fusion khong", (6, 20), scale=0.4, color=(120, 120, 120))
        put_text(column, " dung attention)", (6, 36), scale=0.4, color=(120, 120, 120))
        return column

    peak = max(weights) if weights else 1.0
    bar_max = width - 66
    for idx, weight in enumerate(weights):
        y0 = idx * (FRAME_H + GAP)
        centre = y0 + FRAME_H // 2
        length = max(2, int(bar_max * (weight / peak if peak > 0 else 0)))
        # Đậm dần theo trọng số -> nhìn phát biết frame nào được tin cậy nhất.
        intensity = int(60 + 150 * (weight / peak if peak > 0 else 0))
        cv2.rectangle(column, (4, centre - 8), (4 + length, centre + 8),
                      (40, intensity, 40), thickness=-1)
        put_text(column, f"{weight:.3f}", (width - 58, centre + 5), scale=0.4)
    return column


def prediction_column(width: int, height: int, truth: str, pred: str, conf: float) -> np.ndarray:
    column = blank(height, width)
    ok = pred == truth
    colour = (0, 130, 0) if ok else (0, 0, 200)
    put_text(column, f"GT   : {truth}", (8, 26), scale=0.52)
    put_text(column, f"Pred : {pred}", (8, 52), scale=0.52, color=colour)

    # Đánh dấu ký tự sai — thứ người đọc paper muốn thấy ngay. Đặt ngay dưới dòng
    # Pred, và đẩy conf/verdict xuống để không chèn lên nhau.
    if not ok and len(pred) == len(truth):
        diff = "".join("^" if a != b else " " for a, b in zip(truth, pred))
        put_text(column, f"       {diff}", (8, 70), scale=0.52, color=colour)

    put_text(column, f"conf = {conf:.3f}", (8, 100), scale=0.44)
    put_text(column, "ĐÚNG" if ok else "SAI", (8, 126), scale=0.56, color=colour, thickness=2)
    if not ok and len(pred) != len(truth):
        put_text(column, f"(dai {len(pred)} vs {len(truth)})", (8, 150), scale=0.42, color=colour)
    return column


def build_row(frames, sr_frames, weights, truth, pred, conf, track_id, num_frames) -> np.ndarray:
    lr_col = frame_column(frames, COL_LR, num_frames)
    sr_col = frame_column(sr_frames, COL_SR, num_frames)
    attn_col = attention_column(weights, COL_ATTN, num_frames)
    body_h = lr_col.shape[0]
    pred_col = prediction_column(COL_PRED, body_h, truth, pred, conf)

    body = np.hstack([lr_col, sr_col, attn_col, pred_col])
    header = blank(HEADER_H, body.shape[1])
    put_text(header, f"{track_id}", (8, 17), scale=0.46, color=(90, 90, 90))
    for label, offset, span in [("I_LR", 0, COL_LR), ("I_SR", COL_LR, COL_SR),
                                ("Attention", COL_LR + COL_SR, COL_ATTN),
                                ("Prediction", COL_LR + COL_SR + COL_ATTN, COL_PRED)]:
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        put_text(header, label, (offset + (span - text_w) // 2, 37), scale=0.46, color=(90, 90, 90))
    row = np.vstack([header, body])
    return cv2.copyMakeBorder(row, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(210, 210, 210))


def forward_track(model, config, images, device):
    """Chạy 1 track, trả về (log_probs, sr_frames, attention weights)."""
    batch = images.unsqueeze(0).to(device)
    num_frames = images.size(0)
    if config.USE_SR:
        log_probs, sr_output, _, _ = model(batch, return_sr=True)
        sr_frames = sr_output.view(1, num_frames, *sr_output.shape[1:])[0]
    else:
        log_probs, sr_frames = model(batch), None

    weights = None
    if model.fusion.mode == "attention":
        # Chạy lại phần đầu pipeline để lấy feature đúng chỗ fusion chấm điểm.
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
    return log_probs, sr_frames, weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure định tính 4 cột cho paper")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/paper_figures")
    parser.add_argument("--num-success", type=int, default=5)
    parser.add_argument("--num-failure", type=int, default=5)
    parser.add_argument("--pick", choices=["extreme", "first"], default="extreme",
                        help="extreme = đúng-tự-tin-nhất + sai-mơ-hồ-nhất; first = lấy tuần tự")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--decode", choices=["greedy", "constrained"], default="constrained")
    parser.add_argument("--width-downsample", type=int, choices=[4, 8], default=8,
                        help="Phải khớp lúc train (không nằm trong state_dict)")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    config.WIDTH_DOWNSAMPLE = args.width_downsample
    config.DECODE_MODE = args.decode
    if args.data_root:
        config.DATA_ROOT = args.data_root

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise SystemExit(f"❌ Không tìm thấy checkpoint: {args.checkpoint}")
    checkpoint_path = matches[0]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    state = torch.load(checkpoint_path, map_location="cpu")
    kwargs = infer_architecture(state, config)
    config.USE_SR = bool(kwargs.get("use_sr"))

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

    dataset = MultiFrameDataset(
        config.DATA_ROOT, mode="val", split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX, val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED, num_frames=config.NUM_FRAMES,
    )
    if len(dataset) == 0:
        raise SystemExit("❌ Validation dataset rỗng.")

    layout = PlateLayout.from_spec(config.PLATE_LAYOUTS)
    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            images, _, _, truth, track_id = dataset[idx][:5]
            log_probs, _, _ = forward_track(model, config, images, device)
            pred, conf = decode_batch(
                log_probs, config.IDX2CHAR, mode=config.DECODE_MODE, layout=layout
            )[0]
            results.append({"idx": idx, "track_id": track_id, "truth": truth,
                            "pred": pred, "conf": conf, "ok": pred == truth})

    correct = sum(r["ok"] for r in results)
    print(f"   Val: {correct}/{len(results)} = {correct / len(results) * 100:.2f}% "
          f"({config.DECODE_MODE} decode)")

    hits = [r for r in results if r["ok"]]
    misses = [r for r in results if not r["ok"]]
    if args.pick == "extreme":
        hits.sort(key=lambda r: -r["conf"])    # đúng & chắc chắn nhất
        misses.sort(key=lambda r: r["conf"])   # sai & mơ hồ nhất
    chosen = hits[:args.num_success] + misses[:args.num_failure]
    if not chosen:
        raise SystemExit("❌ Không chọn được track nào.")

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    with torch.no_grad():
        for order, record in enumerate(chosen):
            images = dataset[record["idx"]][0]
            _, sr_frames, weights = forward_track(model, config, images, device)
            row = build_row(images, sr_frames, weights, record["truth"], record["pred"],
                            record["conf"], record["track_id"], images.size(0))
            rows.append(row)
            status = "ok" if record["ok"] else "err"
            cv2.imwrite(
                os.path.join(args.output_dir, f"{order:02d}_{status}_{record['track_id']}.png"),
                row,
            )

    width = max(r.shape[1] for r in rows)
    padded = [
        r if r.shape[1] == width else
        cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(210, 210, 210))
        for r in rows
    ]
    grid_path = os.path.join(args.output_dir, "figure4_qualitative_grid.png")
    cv2.imwrite(grid_path, np.vstack(padded))

    print(f"\n✅ {len(rows)} hàng ({len(hits[:args.num_success])} đúng + "
          f"{len(misses[:args.num_failure])} sai)")
    print(f"   Grid  → {grid_path}")
    print(f"   Từng track → {args.output_dir}/NN_[ok|err]_track_*.png")


if __name__ == "__main__":
    main()
