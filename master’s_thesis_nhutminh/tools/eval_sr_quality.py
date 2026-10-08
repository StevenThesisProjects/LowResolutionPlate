#!/usr/bin/env python3
"""Đo PSNR/SSIM của nhánh SR trên validation (review Bước 2).

Chạy HẬU KỲ trên checkpoint đã train — không train lại, không đụng vào training
loop, nên có thể chạy song song lúc multi-seed đang chạy.

Ba điểm khiến phép đo này không tầm thường như "gọi thêm skimage":

1. Validation vốn KHÔNG có cặp (ảnh vào, HR target). Sample synthetic — thứ duy
   nhất mang HR khớp pixel — chỉ được tạo ở `mode="train"`. Tool này bật
   `sr_eval_mode=True` để val cũng sinh cặp đó, với augment ngẫu nhiên bị tắt
   để số đo tái lập được.
2. `I_SR` nằm trong khung ĐÃ được STN nắn, `I_HR` thì không. Phải warp HR theo
   đúng `theta` giống hệt `Trainer._sr_loss`, nếu không là so hai ảnh khác hệ
   toạ độ và con số vô nghĩa.
3. Luôn báo cáo kèm mốc `sr_base` (nội suy, không học) — PSNR tuyệt đối của SR
   không nói lên điều gì; thứ cần chứng minh là SR **hơn nội suy** bao nhiêu.

⚠️ Không so PSNR giữa các cấu hình khác `sr_scale`: S1/S3 xuất ảnh 64x256, S4
xuất 32x128 — hai thang khác nhau. Chỉ so trong cùng một `sr_scale`.

Ví dụ:
    python tools/eval_sr_quality.py --checkpoint results/s1_proposed_best.pth
    python tools/eval_sr_quality.py --checkpoint results/s4_sr_scale1_best.pth \
      --width-downsample 4 --lr-domain-match
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from tools.eval_decode import infer_architecture

try:
    from skimage.metrics import structural_similarity

    HAS_SKIMAGE = True
except ImportError:  # pragma: no cover - phụ thuộc môi trường
    HAS_SKIMAGE = False


def to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    """Ảnh chuẩn hoá về [-1,1] (mean=std=0.5) -> [0,1] để tính PSNR/SSIM."""
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR từng ảnh, data_range=1.0. Trả tensor [N]."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))


def ssim_per_image(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    """SSIM từng ảnh bằng skimage (chuẩn tham chiếu hay được trích dẫn)."""
    if not HAS_SKIMAGE:
        return []
    values = []
    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
    target_np = target.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(pred_np.shape[0]):
        values.append(
            float(structural_similarity(
                target_np[i], pred_np[i], channel_axis=2, data_range=1.0
            ))
        )
    return values


def warp_like_loss(hr_flat: torch.Tensor, theta_sel: torch.Tensor | None) -> torch.Tensor:
    """Warp HR theo `theta` — sao chép đúng logic của `Trainer._sr_loss`."""
    if theta_sel is None:
        return hr_flat
    grid = F.affine_grid(
        theta_sel.to(hr_flat.dtype), hr_flat.size(), align_corners=False
    )
    return F.grid_sample(hr_flat, grid, align_corners=False, padding_mode="border")


def summarize(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name:<26s} (không có dữ liệu)"
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return f"  {name:<26s} {mean:8.4f}  ± {variance ** 0.5:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSNR/SSIM cho nhánh SR")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width-downsample", type=int, choices=[4, 8], default=8,
                        help="Phải khớp lúc train (không nằm trong state_dict)")
    parser.add_argument("--lr-domain-match", action="store_true",
                        help="Phải khớp lúc train — đổi cỡ ảnh trước khi degrade")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chỉ chạy N track đầu (debug nhanh)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed cho degradation, để lần chạy sau ra đúng số cũ")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    config.WIDTH_DOWNSAMPLE = args.width_downsample
    if args.data_root:
        config.DATA_ROOT = args.data_root

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise SystemExit(f"❌ Không tìm thấy checkpoint: {args.checkpoint}")
    checkpoint_path = matches[0]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    state = torch.load(checkpoint_path, map_location="cpu")
    kwargs = infer_architecture(state, config)
    if not kwargs.get("use_sr"):
        raise SystemExit("❌ Checkpoint này không có nhánh SR — không đo được PSNR/SSIM.")

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

    sr_scale = kwargs.get("sr_scale", 1)
    print(f"📦 {os.path.basename(checkpoint_path)}")
    print(f"   {', '.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}")
    print(f"   Ảnh SR: {config.IMG_HEIGHT * sr_scale}x{config.IMG_WIDTH * sr_scale}"
          f"  (chỉ so được với checkpoint cùng sr_scale={sr_scale})")
    if sr_scale == 1:
        print("   ⚠️ sr_scale=1: mốc `base` là ảnh gốc giữ nguyên (không nội suy),"
              " nên đọc là 'SR có hơn việc không làm gì'.")
    if not HAS_SKIMAGE:
        print("   ⚠️ Thiếu scikit-image -> chỉ có PSNR. Cài: pip install scikit-image")

    # Seed để degradation ngẫu nhiên ra cùng kết quả giữa các lần chạy.
    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = MultiFrameDataset(
        config.DATA_ROOT,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        num_frames=config.NUM_FRAMES,
        provide_sr_target=True,
        sr_scale=sr_scale,
        lr_domain_match=args.lr_domain_match,
        sr_eval_mode=True,
    )
    if len(dataset) == 0:
        raise SystemExit("❌ Không sinh được cặp (input, HR) nào trên validation.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=args.num_workers,
    )

    psnr_sr: list[float] = []
    psnr_base: list[float] = []
    ssim_sr: list[float] = []
    ssim_base: list[float] = []
    rows: list[str] = []
    processed = 0

    with torch.no_grad():
        for images, _, _, _, track_ids, hr_targets, hr_index in loader:
            if hr_targets is None or hr_index.numel() == 0:
                continue
            images = images.to(device)
            batch_size, num_frames = images.shape[:2]

            _, sr_output, sr_base, theta = model(images, return_sr=True)
            channels, height, width = sr_output.shape[1:]

            index = hr_index.to(device)
            selected = sr_output.view(batch_size, num_frames, channels, height, width)[index]
            selected = selected.reshape(-1, channels, height, width)
            base_sel = sr_base.view(batch_size, num_frames, channels, height, width)[index]
            base_sel = base_sel.reshape(-1, channels, height, width)

            hr_flat = hr_targets.to(device).reshape(-1, *hr_targets.shape[2:])
            theta_sel = None
            if theta is not None:
                theta_sel = theta.view(batch_size, num_frames, 2, 3)[index].reshape(-1, 2, 3)
            hr_flat = warp_like_loss(hr_flat, theta_sel)

            pred_u = to_unit_range(selected.float())
            base_u = to_unit_range(base_sel.float())
            target_u = to_unit_range(hr_flat.float())

            batch_psnr_sr = psnr_per_image(pred_u, target_u).cpu().tolist()
            batch_psnr_base = psnr_per_image(base_u, target_u).cpu().tolist()
            psnr_sr += batch_psnr_sr
            psnr_base += batch_psnr_base
            batch_ssim_sr = ssim_per_image(pred_u, target_u)
            batch_ssim_base = ssim_per_image(base_u, target_u)
            ssim_sr += batch_ssim_sr
            ssim_base += batch_ssim_base

            # Gộp 5 frame của cùng track thành 1 dòng CSV.
            for local_idx, sample_idx in enumerate(index.cpu().tolist()):
                start = local_idx * num_frames
                stop = start + num_frames
                track_psnr = sum(batch_psnr_sr[start:stop]) / num_frames
                track_base = sum(batch_psnr_base[start:stop]) / num_frames
                track_ssim = (
                    sum(batch_ssim_sr[start:stop]) / num_frames if batch_ssim_sr else float("nan")
                )
                track_ssim_base = (
                    sum(batch_ssim_base[start:stop]) / num_frames if batch_ssim_base else float("nan")
                )
                rows.append(
                    f"{track_ids[sample_idx]},{track_psnr:.4f},{track_base:.4f},"
                    f"{track_ssim:.4f},{track_ssim_base:.4f}"
                )
            processed += index.numel()
            if args.limit and processed >= args.limit:
                break

    print(f"\n{'=' * 64}")
    print(f"Chất lượng SR — {processed} track ({len(psnr_sr)} ảnh)")
    print(f"{'=' * 64}")
    print(summarize("PSNR (SR học được)", psnr_sr))
    print(summarize("PSNR (base, không học)", psnr_base))
    if psnr_sr and psnr_base:
        gain = sum(psnr_sr) / len(psnr_sr) - sum(psnr_base) / len(psnr_base)
        print(f"  {'=> SR hơn base':<26s} {gain:+8.4f} dB")
    if ssim_sr:
        print()
        print(summarize("SSIM (SR học được)", ssim_sr))
        print(summarize("SSIM (base, không học)", ssim_base))
        gain = sum(ssim_sr) / len(ssim_sr) - sum(ssim_base) / len(ssim_base)
        print(f"  {'=> SR hơn base':<26s} {gain:+8.4f}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w") as handle:
            handle.write("track_id,psnr_sr,psnr_base,ssim_sr,ssim_base\n")
            handle.write("\n".join(rows) + "\n")
        print(f"\n📝 Đã lưu {len(rows)} dòng → {args.output_csv}")


if __name__ == "__main__":
    main()
