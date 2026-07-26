"""
Training loop cho Baseline 1: CRNN + STN.

Sử dụng:
  - CTC Loss (alignment-free, không cần segment ký tự)
  - AdamW optimizer + warmup/cosine decay scheduler
  - Mixed precision (AMP) cho tốc độ
  - Gradient accumulation để train batch lớn ổn định hơn
  - Metric: Exact Match accuracy (report Trang 48, 50)

Nguồn gốc: MultiFrame-LPR-main/src/training/trainer.py
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import SRPixelLoss
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class WarmupCosineScheduler:
    """Simple warmup + cosine decay scheduler with a configurable min LR ratio."""

    def __init__(self, optimizer: optim.Optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(self.total_steps * max(0.0, warmup_ratio)))
        self.min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_lr = [lr * self.min_lr_ratio for lr in self.base_lrs]
        self.step_count = 0
        self._apply_lr(self.last_lr)

    def _apply_lr(self, lrs: List[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def _compute_factor(self, step: int) -> float:
        if step <= self.warmup_steps:
            warmup_progress = step / max(1, self.warmup_steps)
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * warmup_progress

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        decay_progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self) -> None:
        self.step_count += 1
        factor = self._compute_factor(self.step_count)
        self.last_lr = [base_lr * factor for base_lr in self.base_lrs]
        self._apply_lr(self.last_lr)

    def get_last_lr(self) -> List[float]:
        return self.last_lr


class Trainer:
    """Vòng lặp train / validate / inference cho CRNN+STN."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        self.grad_accum_steps = max(1, int(getattr(config, "GRAD_ACCUM_STEPS", 1)))
        self.patience = int(getattr(config, "EARLY_STOPPING_PATIENCE", 0))
        self.use_amp = bool(getattr(config, "USE_AMP", True))
        self.warmup_ratio = float(getattr(config, "WARMUP_RATIO", 0.0))
        self.min_lr_ratio = float(getattr(config, "MIN_LR_RATIO", 0.05))
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        # CTC Loss: blank=0, zero_infinity=True tránh NaN khi input quá ngắn
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")

        # Multi-task SR supervision (fix cho lỗi PR #7: SR trước đây chỉ học
        # qua gradient CTC, không có ràng buộc pixel-level nào).
        self.use_sr = bool(getattr(config, "USE_SR", False))
        self.lambda_sr = float(getattr(config, "LAMBDA_SR", 0.1))
        self.sr_loss_fn = SRPixelLoss(
            edge_weight=float(getattr(config, "SR_EDGE_WEIGHT", 0.5))
        ).to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        total_steps = math.ceil(len(train_loader) / self.grad_accum_steps) * max(1, config.EPOCHS)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
        )
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        self.best_acc = 0.0
        self.best_train_loss = float("inf")
        self.current_epoch = 0
        self.no_improve_epochs = 0
        self.global_step = 0
        self.nan_batches = 0

    def _output_path(self, filename: str) -> str:
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        return os.path.join(self.config.OUTPUT_DIR, filename)

    def _exp_name(self) -> str:
        return getattr(self.config, "EXPERIMENT_NAME", "crnn_stn_baseline")

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scaler.get_scale() >= scale_before:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _sr_loss(self, sr_output: torch.Tensor, hr_targets: torch.Tensor, has_hr: torch.Tensor) -> torch.Tensor:
        """hr_targets: [B, F, C, H', W'], has_hr: [B] -> mask flattened to [B*F]."""
        batch_size, num_frames = hr_targets.shape[:2]
        hr_flat = hr_targets.reshape(batch_size * num_frames, *hr_targets.shape[2:]).to(self.device)
        mask_flat = has_hr.to(self.device).repeat_interleave(num_frames)
        return self.sr_loss_fn(sr_output, hr_flat, mask_flat)

    def train_one_epoch(self) -> float:
        """Train 1 epoch — forward → CTC loss (+ SR loss nếu use_sr) → backward."""
        self.model.train()
        epoch_loss = 0.0
        num_valid_steps = 0
        self.nan_batches = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.EPOCHS}")
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(pbar, start=1):
            if self.use_sr:
                images, targets, target_lengths, _, _, hr_targets, has_hr = batch
            else:
                images, targets, target_lengths, _, _ = batch
            images = images.to(self.device)
            targets = targets.to(self.device)

            with autocast("cuda", enabled=self.use_amp and self.device.type == "cuda"):
                if self.use_sr:
                    preds, sr_output = self.model(images, return_sr=True)
                else:
                    preds = self.model(images)                 # [B, T, C]
                # CTC yêu cầu input shape [T, B, C]
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    (images.size(0),), preds.size(1), dtype=torch.long, device=self.device
                )
                # CTC is numerically fragile in fp16: the forward-backward pass
                # accumulates over T timesteps and underflows, which shows up as
                # NaN once T doubles (SR upscales the input). Always run it in
                # fp32 regardless of the surrounding autocast region.
                with autocast("cuda", enabled=False):
                    ctc_loss = self.criterion(
                        preds_permuted.float(), targets, input_lengths, target_lengths
                    )
                loss = ctc_loss
                if self.use_sr:
                    sr_loss = self._sr_loss(sr_output, hr_targets, has_hr)
                    loss = loss + self.lambda_sr * sr_loss
                loss = loss / self.grad_accum_steps

            # A single bad batch would otherwise poison the epoch average and
            # hide the fact that everything else trained fine.
            if not torch.isfinite(loss):
                self.nan_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="skipped(nan)", nan=self.nan_batches)
                continue

            self.scaler.scale(loss).backward()
            should_step = (step_idx % self.grad_accum_steps == 0) or (step_idx == len(self.train_loader))
            if should_step:
                self._optimizer_step()

            epoch_loss += loss.item() * self.grad_accum_steps
            num_valid_steps += 1
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.LEARNING_RATE
            postfix = {"loss": f'{loss.item() * self.grad_accum_steps:.4f}', "lr": f"{current_lr:.2e}"}
            if self.use_sr:
                postfix["sr"] = f"{sr_loss.item():.4f}"
            if self.nan_batches:
                postfix["nan"] = self.nan_batches
            pbar.set_postfix(**postfix)

        if num_valid_steps == 0:
            print("❌ Toàn bộ batch trong epoch đều NaN — training không tiến triển được.")
            return float("nan")
        if self.nan_batches:
            print(f"⚠️ Bỏ qua {self.nan_batches} batch NaN trong epoch này.")
        return epoch_loss / num_valid_steps

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """
        Đánh giá trên validation set.

        Metric: Exact Match — chuỗi dự đoán phải khớp hoàn toàn ground truth.
        """
        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0}, []

        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []

        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                preds = self.model(images)

                input_lengths = torch.full((images.size(0),), preds.size(1), dtype=torch.long, device=self.device)
                loss = self.criterion(
                    preds.permute(1, 0, 2).float(), targets, input_lengths, target_lengths
                )
                val_loss += loss.item()

                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    if pred_text == labels_text[i]:
                        total_correct += 1
                    submission_data.append(f"{track_ids[i]},{pred_text};{conf:.4f}")
                total_samples += len(labels_text)

        val_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
        return {"loss": val_loss / len(self.val_loader), "acc": val_acc}, submission_data

    def save_model(self, path: str = None) -> None:
        if path is None:
            path = self._output_path(f"{self.config.EXPERIMENT_NAME}_best.pth")
        torch.save(self.model.state_dict(), path)

    def save_submission(self, data: List[str]) -> None:
        path = self._output_path(f"submission_{self.config.EXPERIMENT_NAME}.txt")
        with open(path, "w") as f:
            f.write("\n".join(data))
        print(f"📝 Saved {len(data)} dòng → {path}")

    def fit(self) -> None:
        """Chạy toàn bộ training loop."""
        print(f"🚀 TRAIN | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        if self.grad_accum_steps > 1:
            print(f"🔁 Gradient accumulation: {self.grad_accum_steps} steps")
        print(f"📈 Warmup ratio: {self.warmup_ratio:.2f} | Min LR ratio: {self.min_lr_ratio:.2f}")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            train_loss = self.train_one_epoch()
            val_metrics, submission_data = self.validate()

            current_lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            improved = False
            if self.val_loader is not None and val_metrics["acc"] > self.best_acc:
                improved = True
                self.best_acc = val_metrics["acc"]
                self.no_improve_epochs = 0
                self.save_model()
                model_path = self._output_path(f"{self._exp_name()}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_metrics['acc']:.2f}%)")
                if submission_data:
                    self.save_submission(submission_data)
            elif self.val_loader is None and train_loss < self.best_train_loss:
                improved = True
                if self.best_train_loss == float("inf"):
                    improve_pct = 0.0
                else:
                    improve_pct = (self.best_train_loss - train_loss) / max(self.best_train_loss, 1e-12) * 100
                self.best_train_loss = train_loss
                self.save_model()
                model_path = self._output_path(f"{self._exp_name()}_best.pth")
                print(
                    f"  ⭐ Saved Best Model (train loss): {model_path} | "
                    f"Loss: {train_loss:.4f} | Improve: {improve_pct:.2f}%"
                )

            if self.val_loader is not None:
                if improved:
                    self.no_improve_epochs = 0
                else:
                    self.no_improve_epochs += 1
                    if self.patience > 0 and self.no_improve_epochs >= self.patience:
                        print(f"🛑 Early stopping: no improvement for {self.no_improve_epochs} epochs")
                        break

        if self.val_loader is None:
            self.save_model()
            model_path = self._output_path(f"{self._exp_name()}_best.pth")
            print(f"  💾 Saved final model: {model_path}")

        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """
        Inference trên bất kỳ DataLoader nào.

        Returns:
            List of (track_id, predicted_text, confidence)
        """
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        with torch.no_grad():
            for images, _, _, _, track_ids in tqdm(loader, desc="Inference"):
                images = images.to(self.device)
                preds = self.model(images)
                for i, (pred_text, conf) in enumerate(decode_with_confidence(preds, self.idx2char)):
                    results.append((track_ids[i], pred_text, conf))
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        """Inference test set và lưu file submission."""
        print("🔮 Running inference on test data...")
        results = self.predict(test_loader)
        submission_data = [f"{tid},{text};{conf:.4f}" for tid, text, conf in results]
        path = self._output_path(output_filename)
        with open(path, "w") as f:
            f.write("\n".join(submission_data))
        print(f"✅ Saved {len(submission_data)} predictions → {path}")
