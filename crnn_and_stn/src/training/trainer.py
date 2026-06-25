"""
Training loop cho Baseline 1: CRNN + STN.

Sử dụng:
  - CTC Loss (alignment-free, không cần segment ký tự)
  - AdamW optimizer + OneCycleLR scheduler
  - Mixed precision (AMP) cho tốc độ
  - Metric: Exact Match accuracy (report Trang 48, 50)

Nguồn gốc: MultiFrame-LPR-main/src/training/trainer.py
"""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


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
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        # CTC Loss: blank=0, zero_infinity=True tránh NaN khi input quá ngắn
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS,
        )
        self.scaler = GradScaler()
        self.best_acc = 0.0
        self.best_train_loss = float("inf")
        self.current_epoch = 0

    def _output_path(self, filename: str) -> str:
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        return os.path.join(self.config.OUTPUT_DIR, filename)

    def _exp_name(self) -> str:
        return getattr(self.config, "EXPERIMENT_NAME", "crnn_stn_baseline")

    def train_one_epoch(self) -> float:
        """Train 1 epoch — forward → CTC loss → backward."""
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.EPOCHS}")

        for images, targets, target_lengths, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.device.type == "cuda"):
                preds = self.model(images)                    # [B, T, C]
                # CTC yêu cầu input shape [T, B, C]
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    (images.size(0),), preds.size(1), dtype=torch.long
                )
                loss = self.criterion(preds_permuted, targets, input_lengths, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

        return epoch_loss / len(self.train_loader)

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
                preds = self.model(images)

                input_lengths = torch.full((images.size(0),), preds.size(1), dtype=torch.long)
                loss = self.criterion(
                    preds.permute(1, 0, 2), targets, input_lengths, target_lengths
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

            if val_metrics["acc"] > self.best_acc:
                self.best_acc = val_metrics["acc"]
                self.save_model()
                model_path = self._output_path(f"{self._exp_name()}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_metrics['acc']:.2f}%)")
                if submission_data:
                    self.save_submission(submission_data)

            # Submission/full-train mode: không có val -> chọn best theo train loss.
            if self.val_loader is None and train_loss < self.best_train_loss:
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
