"""Trainer class encapsulating the training and validation loop."""
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.ema import ModelEMA
from src.utils.common import seed_everything
from src.utils.postprocess import decode_batch


class Trainer:
    """Encapsulates training, validation, and inference logic."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)
        
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self.optimizer = self._build_optimizer(model)
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=config.USE_AMP and config.DEVICE.type == "cuda")
        self.use_amp = config.USE_AMP and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if getattr(config, "AMP_DTYPE", "float16") == "float16" else torch.bfloat16
        self.label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
        self.ema = ModelEMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None

        # Supervised SR settings
        self.use_learnable_sr = getattr(config, 'USE_LEARNABLE_SR', False)
        self.sr_loss_weight = getattr(config, 'SR_LOSS_WEIGHT', 0.3)
        self.sr_loss_weight_min = getattr(config, 'SR_LOSS_WEIGHT_MIN', 0.05)
        
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_use_ema = True
        self.current_epoch = 0
        self.epochs_without_improvement = 0
    
    def _build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {
                "params": backbone_params,
                "lr": self.config.LEARNING_RATE * self.config.BACKBONE_LR_RATIO,
            },
            {
                "params": head_params,
                "lr": self.config.LEARNING_RATE,
            },
        ]
        return optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)

    def _build_scheduler(self):
        sched_type = getattr(self.config, "SCHEDULER_TYPE", "cosine")

        if sched_type == "onecycle":
            max_lrs = [
                self.config.LEARNING_RATE * self.config.BACKBONE_LR_RATIO,
                self.config.LEARNING_RATE,
            ]
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                steps_per_epoch=len(self.train_loader),
                epochs=self.config.EPOCHS,
                pct_start=self.config.ONECYCLE_PCT_START,
                div_factor=25.0,
                final_div_factor=1000.0,
                anneal_strategy="cos",
            )

        warmup_epochs = min(self.config.WARMUP_EPOCHS, max(1, self.config.EPOCHS - 1))

        warmup = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )

        if sched_type == "cosine_restarts":
            # Warm restarts: LR spikes back up periodically for better exploration
            t0 = getattr(self.config, "COSINE_T0", 15)
            t_mult = getattr(self.config, "COSINE_T_MULT", 2)
            cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0,
                T_mult=t_mult,
                eta_min=self.config.MIN_LR,
            )
        else:
            # Plain cosine decay
            cosine_epochs = max(1, self.config.EPOCHS - warmup_epochs)
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=self.config.MIN_LR,
            )

        return optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    @property
    def _step_scheduler_per_batch(self) -> bool:
        return getattr(self.config, "SCHEDULER_TYPE", "cosine") == "onecycle"

    def _set_dataset_epoch(self, epoch: int) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch, self.config.EPOCHS)

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[-1]["lr"]

    def _decode_kwargs(self, for_inference: bool = False) -> Dict:
        plate_len = getattr(self.config, "PLATE_LENGTH", 0) or None
        if for_inference:
            beam_width = self.config.BEAM_WIDTH if self.config.USE_INFERENCE_BEAM else 1
            return {
                "use_tta": self.config.USE_INFERENCE_TTA,
                "beam_width": beam_width,
                "expected_length": plate_len,
            }
        beam_width = self.config.BEAM_WIDTH if self.config.USE_BEAM_DECODE else 1
        return {
            "use_tta": self.config.USE_VAL_TTA,
            "beam_width": beam_width,
            "expected_length": plate_len,
        }

    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    def _get_exp_name(self) -> str:
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def _is_bad_loss(self, loss_value: float) -> bool:
        return (
            not math.isfinite(loss_value)
            or loss_value > self.config.MAX_TRAIN_LOSS
        )

    def _sr_weight(self) -> float:
        """Cosine-decayed SR loss weight: high early → low late."""
        if not self.use_learnable_sr:
            return 0.0
        progress = self.current_epoch / max(self.config.EPOCHS - 1, 1)
        w_max = self.sr_loss_weight
        w_min = self.sr_loss_weight_min
        return w_min + 0.5 * (w_max - w_min) * (1 + math.cos(math.pi * progress))

    def _compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """CTC loss with optional label smoothing (uniform prior regularizer)."""
        loss = self.criterion(
            preds.permute(1, 0, 2), targets, input_lengths, target_lengths
        )
        if self.label_smoothing > 0:
            # Penalize confident predictions with KL(uniform || pred)
            log_probs = preds  # already log_softmax from model
            smooth_loss = -log_probs.mean()
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss
        return loss

    def _compute_sr_loss(
        self,
        sr_output: torch.Tensor,
        hr_targets: torch.Tensor,
    ) -> torch.Tensor:
        """L1 pixel loss for Supervised SR.

        Computes pixel loss between the SR-enhanced images and the HR targets.
        """
        # hr_targets shape: [B, F, 3, H, W] → flatten to [B*F, 3, H, W]
        b, f = hr_targets.shape[:2]
        hr_flat = hr_targets.view(b * f, *hr_targets.shape[2:])

        # Mask: only include samples where HR is not all zeros
        mask = hr_flat.abs().sum(dim=(1, 2, 3)) > 0  # [B*F]
        if mask.sum() == 0:
            return torch.tensor(0.0, device=sr_output.device)
            
        sr_valid = sr_output[mask]
        hr_valid = hr_flat[mask]

        # 1. Pixel L1 Loss
        return nn.functional.l1_loss(sr_valid, hr_valid)

    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        epoch_sr_loss = 0.0
        valid_steps = 0
        skipped_batches = 0
        sr_weight = self._sr_weight()
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        
        for images, targets, target_lengths, _, _, hr_images in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            target_lengths = target_lengths.to(self.device, non_blocking=True)
            hr_images = hr_images.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            if self.use_amp:
                with autocast(device_type, dtype=self.amp_dtype):
                    if self.use_learnable_sr:
                        preds, sr_output = self.model(images, return_sr=True)
                    else:
                        preds = self.model(images)
                        sr_output = None
                    input_lengths = torch.full(
                        size=(images.size(0),),
                        fill_value=preds.size(1),
                        dtype=torch.long,
                        device=self.device,
                    )
                    loss = self._compute_loss(preds, targets, input_lengths, target_lengths)
                    # Add supervised SR loss
                    sr_loss_val = 0.0
                    if sr_output is not None and sr_weight > 0:
                        sr_loss = self._compute_sr_loss(sr_output, hr_images)
                        sr_loss_val = sr_loss.item()
                        loss = loss + sr_weight * sr_loss
            else:
                if self.use_learnable_sr:
                    preds, sr_output = self.model(images, return_sr=True)
                else:
                    preds = self.model(images)
                    sr_output = None
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long,
                    device=self.device,
                )
                loss = self._compute_loss(preds, targets, input_lengths, target_lengths)
                sr_loss_val = 0.0
                if sr_output is not None and sr_weight > 0:
                    sr_loss = self._compute_sr_loss(sr_output, hr_images)
                    sr_loss_val = sr_loss.item()
                    loss = loss + sr_weight * sr_loss

            loss_value = loss.item()
            if self._is_bad_loss(loss_value):
                skipped_batches += 1
                pbar.set_postfix({
                    'loss': 'skip',
                    'lr': f"{self._current_lr():.2e}",
                    'skip': skipped_batches,
                })
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                stepped = self.scaler.get_scale() >= scale_before
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                self.optimizer.step()
                stepped = True

            if stepped and self.ema is not None:
                self.ema.update(self.model)

            if stepped and self._step_scheduler_per_batch:
                self.scheduler.step()

            epoch_loss += loss_value
            epoch_sr_loss += sr_loss_val
            valid_steps += 1
            postfix = {'loss': loss_value, 'lr': f"{self._current_lr():.2e}"}
            if self.use_learnable_sr:
                postfix['sr'] = f"{sr_loss_val:.4f}"
            pbar.set_postfix(postfix)
        
        if valid_steps == 0:
            return float("inf")
        if skipped_batches > 0:
            print(f"  ⚠️ Skipped {skipped_batches} unstable batches this epoch.")
        if self.use_learnable_sr and valid_steps > 0:
            avg_sr = epoch_sr_loss / valid_steps
            print(f"  🔍 SR Loss: {avg_sr:.4f} | λ_sr: {sr_weight:.4f}")
        return epoch_loss / valid_steps

    def _run_validation_pass(
        self,
        use_ema: bool,
        decode_kwargs: Dict,
    ) -> Tuple[Dict[str, float], List[str]]:
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}, []

        backup_state = None
        if use_ema and self.ema is not None:
            backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.load_into(self.model)

        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        with torch.no_grad():
            val_iter = tqdm(
                self.val_loader,
                desc="Val",
                leave=False,
            )
            for images, targets, target_lengths, labels_text, track_ids, _hr in val_iter:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                target_lengths = target_lengths.to(self.device, non_blocking=True)

                if decode_kwargs.get("use_tta"):
                    from src.utils.postprocess import forward_with_tta
                    preds = forward_with_tta(self.model, images)
                elif self.use_amp:
                    with autocast(device_type, dtype=self.amp_dtype):
                        preds = self.model(images)
                else:
                    preds = self.model(images)

                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long,
                    device=self.device,
                )
                loss = self.criterion(
                    preds.float().permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths,
                )  # No label smoothing in val — clean metric
                loss_value = loss.item()
                if math.isfinite(loss_value):
                    val_loss += loss_value

                from src.utils.postprocess import decode_with_confidence
                decoded_list = decode_with_confidence(
                    preds,
                    self.idx2char,
                    beam_width=decode_kwargs.get("beam_width", 1),
                    expected_length=decode_kwargs.get("expected_length"),
                )

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")

                total_samples += len(labels_text)

        if backup_state is not None:
            self.model.load_state_dict(backup_state)

        avg_val_loss = val_loss / max(len(self.val_loader), 1)
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        return {'loss': avg_val_loss, 'acc': val_acc}, submission_data

    def validate(
        self,
        use_ema: bool = True,
        for_inference: bool = False,
    ) -> Tuple[Dict[str, float], List[str]]:
        decode_kwargs = self._decode_kwargs(for_inference=for_inference)
        metrics, submission = self._run_validation_pass(
            use_ema=use_ema,
            decode_kwargs=decode_kwargs,
        )
        return metrics, submission

    def validate_best_variant(self) -> Tuple[Dict[str, float], List[str], bool]:
        """Fast greedy validation on raw vs EMA (no TTA/beam during training)."""
        decode_kwargs = self._decode_kwargs(for_inference=False)

        raw_metrics, raw_submission = self._run_validation_pass(
            use_ema=False,
            decode_kwargs=decode_kwargs,
        )

        if self.ema is None:
            return raw_metrics, raw_submission, False

        ema_metrics, ema_submission = self._run_validation_pass(
            use_ema=True,
            decode_kwargs=decode_kwargs,
        )

        if ema_metrics['acc'] >= raw_metrics['acc']:
            print(
                f"  📊 Val raw: {raw_metrics['acc']:.2f}% | "
                f"Val EMA: {ema_metrics['acc']:.2f}% → using EMA"
            )
            return ema_metrics, ema_submission, True

        print(
            f"  📊 Val raw: {raw_metrics['acc']:.2f}% | "
            f"Val EMA: {ema_metrics['acc']:.2f}% → using raw"
        )
        return raw_metrics, raw_submission, False

    def save_submission(self, submission_data: List[str]) -> None:
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def save_model(self, path: str = None, use_ema: bool = True) -> None:
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")

        if use_ema and self.ema is not None:
            torch.save(self.ema.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        print(
            f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS} | "
            f"LR: {self.config.LEARNING_RATE:.2e} | Warmup: {self.config.WARMUP_EPOCHS} | "
            f"EMA: {self.config.USE_EMA} | Val TTA/Beam: {self.config.USE_VAL_TTA}/{self.config.USE_BEAM_DECODE}"
        )
        
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            self._set_dataset_epoch(epoch)
            
            avg_train_loss = self.train_one_epoch()
            val_metrics, submission_data, use_ema_for_best = self.validate_best_variant()
            val_loss = val_metrics['loss']
            val_acc = val_metrics['acc']

            if not self._step_scheduler_per_batch:
                self.scheduler.step()
            current_lr = self._current_lr()
            
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch + 1
                self.best_use_ema = use_ema_for_best
                self.epochs_without_improvement = 0
                self.save_model(use_ema=use_ema_for_best)
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_acc:.2f}%)")
                if submission_data:
                    self.save_submission(submission_data)
            else:
                self.epochs_without_improvement += 1

            if (
                self.val_loader is not None
                and self.config.EARLY_STOP_PATIENCE > 0
                and self.epochs_without_improvement >= self.config.EARLY_STOP_PATIENCE
            ):
                print(
                    f"⏹️ Early stopping: no val improvement for "
                    f"{self.config.EARLY_STOP_PATIENCE} epochs."
                )
                break

            if not math.isfinite(avg_train_loss):
                print("⏹️ Stopping early due to unstable training loss.")
                break
        
        if self.val_loader is None:
            self.save_model(use_ema=True)
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  💾 Saved final model: {model_path}")
        
        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}% (epoch {self.best_epoch})")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        decode_kwargs = self._decode_kwargs(for_inference=True)
        
        with torch.no_grad():
            for images, _, _, _, track_ids, _ in loader:
                images = images.to(self.device)
                decoded_list = decode_batch(model=self.model, images=images, idx2char=self.idx2char, **decode_kwargs)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        print("🔮 Running inference on test data...")
        
        results = []
        self.model.eval()
        decode_kwargs = self._decode_kwargs(for_inference=True)
        with torch.no_grad():
            for images, _, _, _, track_ids, _ in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                decoded_list = decode_batch(
                    model=self.model,
                    images=images,
                    idx2char=self.idx2char,
                    **decode_kwargs,
                )
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        submission_data = [f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
        output_path = self._get_output_path(output_filename)
        with open(output_path, 'w') as f:
            f.write("\n".join(submission_data))
        
        print(f"✅ Saved {len(submission_data)} predictions to {output_path}")
