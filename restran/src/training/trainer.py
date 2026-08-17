"""Trainer class encapsulating the training and validation loop."""
import csv
import math
import os
import statistics
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class Trainer:
    """Encapsulates training, validation, and inference logic."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
        teacher: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: The neural network model.
            train_loader: Training data loader.
            val_loader: Validation data loader (can be None).
            config: Configuration object with training parameters.
            idx2char: Index to character mapping for decoding.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)
        
        # Mixed precision: bf16 has no overflow so it needs no loss scaling
        amp_dtype = getattr(config, 'AMP_DTYPE', 'fp16').lower()
        if amp_dtype not in ('fp16', 'bf16'):
            raise ValueError(f"AMP_DTYPE must be 'fp16' or 'bf16', got {amp_dtype!r}")
        self.amp_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
        self.skip_grad_norm = float(getattr(config, 'SKIP_GRAD_NORM', 0.0))
        self.skip_abort_ratio = float(getattr(config, 'SKIP_ABORT_RATIO', 0.0))
        self.aborted = False

        # Feature distillation from a teacher trained on undegraded HR frames.
        # Matching in feature space (not pixels) sidesteps the fact that lr-00i and
        # hr-00i are different captures and cannot be registered to each other.
        self.teacher = teacher
        self.distill_weight = float(getattr(config, 'DISTILL_WEIGHT', 0.0))
        # Pixel-wise SR supervision against ECC-registered HR frames. Only pairs whose
        # registration actually converged are used — a badly registered target teaches
        # the wrong thing, which is exactly the failure mode this whole step removes.
        self.sr_pixel_weight = float(getattr(config, 'SR_PIXEL_WEIGHT', 0.0))
        self.sr_ncc_min = float(getattr(config, 'SR_NCC_MIN', 0.8))
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        # Loss and optimizer
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS
        )
        # GradScaler is only meaningful for fp16; disabled it passes tensors through
        self.scaler = GradScaler(enabled=(self.amp_dtype == torch.float16))
        
        # Tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.history: List[Dict[str, float]] = []
        self.log_batch_loss = getattr(config, 'LOG_BATCH_LOSS', False)
        self._batch_csv_ready = False

    def _get_output_path(self, filename: str) -> str:
        """Get full path for output file in configured directory."""
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    def _get_exp_name(self) -> str:
        """Get experiment name from config."""
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def _grad_norm_by_block(self, total_norm: float) -> str:
        """Per-submodule gradient norms, to localise where a spike originates.

        Reading this tells you whether the blow-up comes from the STN warp, the
        backbone or the transformer head — which is what decides the actual fix.

        Called after `clip_grad_norm_`, so the stored grads are already scaled down.
        Clipping scales every gradient by the same factor, so multiplying back by
        `total_norm / clip` recovers the true pre-clip magnitudes.
        """
        clip = self.config.GRAD_CLIP
        rescale = total_norm / clip if math.isfinite(total_norm) and total_norm > clip else 1.0
        parts = []
        for name in ('stn', 'backbone', 'fusion', 'transformer', 'head'):
            module = getattr(self.model, name, None)
            if module is None:
                continue
            total = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.detach().float().norm(2).item() ** 2
            parts.append(f"{name}={total ** 0.5 * rescale:.1f}")
        return " ".join(parts)

    def train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict of loss statistics over all batches of the epoch (mean/min/max/
            std/median), plus the mean gradient norm before clipping.
        """
        self.model.train()
        batch_losses: List[float] = []
        grad_norms: List[float] = []
        batch_rows: List[Tuple] = []
        skipped = 0
        skipped_nonfinite = 0
        distill_losses: List[float] = []
        sr_pixel_losses: List[float] = []
        spike_reported = False
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")

        for images, targets, target_lengths, _, track_ids, hr_images, reg_hr, reg_q in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            use_distill = self.teacher is not None and hr_images is not None
            use_sr_pixel = self.sr_pixel_weight > 0 and reg_hr is not None and reg_q is not None
            with autocast('cuda', dtype=self.amp_dtype):
                sr_out = None
                if use_distill and use_sr_pixel:
                    preds, feat_student, sr_out = self.model(
                        images, return_features=True, return_sr=True)
                elif use_distill:
                    preds, feat_student = self.model(images, return_features=True)
                elif use_sr_pixel:
                    preds, sr_out = self.model(images, return_sr=True)
                else:
                    preds = self.model(images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long
                )
                loss = self.criterion(preds_permuted, targets, input_lengths, target_lengths)
                ctc_only = loss.detach()

                if use_distill:
                    with torch.no_grad():
                        _, feat_teacher = self.teacher(
                            hr_images.to(self.device), return_features=True
                        )
                    distill = torch.nn.functional.mse_loss(
                        feat_student.float(), feat_teacher.float()
                    )
                    loss = loss + self.distill_weight * distill
                    distill_losses.append(distill.item())

                if use_sr_pixel and sr_out is not None and reg_q is not None:
                    # [B, F] mask: only frames whose registration converged
                    mask = (reg_q.to(self.device) >= self.sr_ncc_min).float()
                    if mask.sum() > 0:
                        target = reg_hr.to(self.device)
                        per_frame = (sr_out.float() - target.float()).abs().mean(
                            dim=(2, 3, 4))
                        pixel = (per_frame * mask).sum() / mask.sum()
                        loss = loss + self.sr_pixel_weight * pixel
                        sr_pixel_losses.append(pixel.item())

            # Scale loss & backward
            self.scaler.scale(loss).backward()
            
            # Unscale (required before gradient clipping)
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping (returns the pre-clip norm, kept for diagnostics)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP
            ).item()

            # Guard: a single pathological batch with an enormous gradient can wreck a
            # healthy model, because clipping preserves the (bad) direction and still
            # takes a full-size step. Drop the batch instead of stepping on it.
            nonfinite = not math.isfinite(grad_norm)
            over_limit = self.skip_grad_norm > 0 and grad_norm > self.skip_grad_norm
            skip = nonfinite or over_limit

            # Diagnose the first spike of each epoch: an inf/nan norm means fp16
            # overflow, a finite-but-huge one means the optimisation itself blew up.
            # The per-block breakdown says which part of the model is responsible.
            if skip and not spike_reported:
                spike_reported = True
                print(f"\n  🔍 Spike epoch {self.current_epoch + 1}, batch {len(batch_losses) + 1}: "
                      f"grad_norm={grad_norm:.1f} ({'inf/nan - fp16 overflow' if nonfinite else 'huu han'}), "
                      f"loss={loss.item():.4f}")
                print(f"     grad theo khoi (truoc clip): {self._grad_norm_by_block(grad_norm)}")
                print(f"     track dau batch: {list(track_ids[:4])}")

            scale_before = self.scaler.get_scale()
            if skip:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                skipped += 1
                skipped_nonfinite += int(nonfinite)
            else:
                # Step optimizer & update scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Advance the schedule on batches processed, not only on successful steps,
            # so a run of skipped batches cannot freeze the LR. Still held back when
            # the AMP scaler reduced its scale, matching the original behaviour.
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            batch_loss = ctc_only.item()
            current_lr = self.scheduler.get_last_lr()[0]
            batch_losses.append(batch_loss)
            # AMP can produce inf/nan norms on skipped steps; exclude them from stats
            if math.isfinite(grad_norm):
                grad_norms.append(grad_norm)
            self.global_step += 1
            if self.log_batch_loss:
                batch_rows.append((
                    self.current_epoch + 1, len(batch_losses), self.global_step,
                    f"{batch_loss:.6f}", f"{current_lr:.6e}", f"{grad_norm:.6f}",
                ))

            pbar.set_postfix({'loss': batch_loss, 'lr': current_lr})

        if batch_rows:
            self._append_batch_rows(batch_rows)

        return {
            'loss': statistics.fmean(batch_losses),
            'loss_min': min(batch_losses),
            'loss_max': max(batch_losses),
            'loss_std': statistics.pstdev(batch_losses) if len(batch_losses) > 1 else 0.0,
            'loss_median': statistics.median(batch_losses),
            'grad_norm': statistics.fmean(grad_norms) if grad_norms else 0.0,
            'grad_norm_max': max(grad_norms) if grad_norms else 0.0,
            'num_batches': len(batch_losses),
            'skipped_batches': skipped,
            'skipped_nonfinite': skipped_nonfinite,
            'distill_loss': statistics.fmean(distill_losses) if distill_losses else 0.0,
            'sr_pixel_loss': statistics.fmean(sr_pixel_losses) if sr_pixel_losses else 0.0,
        }

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Run validation and generate submission data.
        
        Returns:
            Tuple of (metrics_dict, submission_data).
            metrics_dict contains at least 'loss' and 'acc'.
        """
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0, 'cer': 0.0}, []

        # track_id -> layout, so accuracy can be split by plate type. The overall
        # number hides a large Mercosur/Brazilian gap, and any rebalancing
        # experiment has to watch both sides at once.
        layouts = getattr(self.val_loader.dataset, 'layouts', {})
        per_layout: Dict[str, List[int]] = {}
        
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds: List[str] = []
        all_targets: List[str] = []
        submission_data: List[str] = []
        
        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids, *_ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)
                
                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                val_loss += loss.item()

                # Decode predictions
                decoded_list = decode_with_confidence(preds, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    
                    all_preds.append(pred_text)
                    all_targets.append(gt_text)
                    
                    hit = int(pred_text == gt_text)
                    total_correct += hit
                    stat = per_layout.setdefault(layouts.get(track_id, 'Unknown'), [0, 0])
                    stat[0] += hit
                    stat[1] += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                    
                total_samples += len(labels_text)

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_val_loss,
            'acc': val_acc,
        }
        for name, (hits, n) in per_layout.items():
            metrics[f'acc_{name.lower()}'] = (hits / n * 100) if n else 0.0
            metrics[f'n_{name.lower()}'] = n

        return metrics, submission_data

    def save_submission(self, submission_data: List[str]) -> None:
        """Save submission file with experiment name."""
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def save_model(self, path: str = None) -> None:
        """Save model checkpoint with experiment name."""
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    METRIC_FIELDS = [
        'epoch', 'train_loss', 'train_loss_min', 'train_loss_max', 'train_loss_std',
        'train_loss_median', 'grad_norm', 'grad_norm_max',
        'skipped_batches', 'skipped_nonfinite', 'distill_loss', 'sr_pixel_loss',
        'val_loss', 'val_acc', 'val_acc_mercosur', 'val_acc_brazilian', 'best_acc', 'lr',
        'train_time_s', 'val_time_s', 'epoch_time_s', 'elapsed_s', 'samples_per_s',
        'gpu_mem_peak_mb', 'timestamp',
    ]

    def _append_metrics_row(self, row: Dict[str, float]) -> None:
        """Append one epoch of metrics to the CSV, writing the header on first call.

        Flushed every epoch so the file stays usable if training is interrupted.
        """
        path = self._get_output_path(f"{self._get_exp_name()}_metrics.csv")
        write_header = not os.path.exists(path) or self.current_epoch == 0
        with open(path, 'w' if write_header else 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_batch_rows(self, rows: List[Tuple]) -> None:
        """Append per-batch loss rows to the batch-level CSV (opt-in)."""
        path = self._get_output_path(f"{self._get_exp_name()}_batches.csv")
        write_header = not self._batch_csv_ready
        with open(path, 'w' if write_header else 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'step', 'global_step', 'loss', 'lr', 'grad_norm'])
                self._batch_csv_ready = True
            writer.writerows(rows)

    def _write_summary(self, total_time: float) -> None:
        """Write a human-readable run summary next to the metrics CSV."""
        exp_name = self._get_exp_name()
        path = self._get_output_path(f"{exp_name}_summary.txt")
        epoch_times = [h['epoch_time_s'] for h in self.history]
        train_losses = [h['train_loss'] for h in self.history]

        lines = [
            f"Experiment      : {exp_name}",
            f"Finished at     : {datetime.now().isoformat(timespec='seconds')}",
            f"Device          : {self.device}",
            f"Epochs          : {len(self.history)}/{self.config.EPOCHS}",
            f"Batch size      : {self.config.BATCH_SIZE}",
            f"Learning rate   : {self.config.LEARNING_RATE}",
            f"Use STN         : {getattr(self.config, 'USE_STN', True)}",
            f"Train samples   : {len(self.train_loader.dataset)}",
            f"Val samples     : {len(self.val_loader.dataset) if self.val_loader else 0}",
            "",
            f"Total time      : {total_time / 3600:.2f} h ({total_time:.1f} s)",
            f"Epoch time      : mean {statistics.fmean(epoch_times):.1f} s | "
            f"min {min(epoch_times):.1f} s | max {max(epoch_times):.1f} s",
            f"Train loss      : first {train_losses[0]:.4f} -> last {train_losses[-1]:.4f} | "
            f"min {min(train_losses):.4f}",
            f"Best val acc    : {self.best_acc:.2f}% at epoch {self.best_epoch}",
            "",
            f"Metrics CSV     : {self._get_output_path(f'{exp_name}_metrics.csv')}",
            f"Best checkpoint : {self._get_output_path(f'{exp_name}_best.pth')}",
        ]
        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"🧾 Saved run summary: {path}")

    def fit(self) -> None:
        """Run the full training loop for specified number of epochs."""
        print(f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        run_start = time.perf_counter()

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            # Training
            train_start = time.perf_counter()
            train_stats = self.train_one_epoch()
            train_time = time.perf_counter() - train_start
            avg_train_loss = train_stats['loss']

            # Validation
            val_start = time.perf_counter()
            val_metrics, submission_data = self.validate()
            val_time = time.perf_counter() - val_start
            val_loss = val_metrics['loss']
            val_acc = val_metrics['acc']
            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time = train_time + val_time

            # Save best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.best_epoch = epoch + 1

            # Record metrics before printing so the CSV survives a crash mid-epoch
            row = {
                'epoch': epoch + 1,
                'train_loss': round(avg_train_loss, 6),
                'train_loss_min': round(train_stats['loss_min'], 6),
                'train_loss_max': round(train_stats['loss_max'], 6),
                'train_loss_std': round(train_stats['loss_std'], 6),
                'train_loss_median': round(train_stats['loss_median'], 6),
                'grad_norm': round(train_stats['grad_norm'], 6),
                'grad_norm_max': round(train_stats['grad_norm_max'], 6),
                'skipped_batches': train_stats['skipped_batches'],
                'skipped_nonfinite': train_stats['skipped_nonfinite'],
                'distill_loss': round(train_stats['distill_loss'], 6),
                'sr_pixel_loss': round(train_stats['sr_pixel_loss'], 6),
                'val_loss': round(val_loss, 6),
                'val_acc': round(val_acc, 4),
                'val_acc_mercosur': round(val_metrics.get('acc_mercosur', 0.0), 4),
                'val_acc_brazilian': round(val_metrics.get('acc_brazilian', 0.0), 4),
                'best_acc': round(self.best_acc, 4),
                'lr': f"{current_lr:.6e}",
                'train_time_s': round(train_time, 2),
                'val_time_s': round(val_time, 2),
                'epoch_time_s': round(epoch_time, 2),
                'elapsed_s': round(time.perf_counter() - run_start, 2),
                'samples_per_s': round(len(self.train_loader.dataset) / train_time, 2),
                'gpu_mem_peak_mb': round(
                    torch.cuda.max_memory_allocated() / 1024**2, 1
                ) if self.device.type == 'cuda' else 0.0,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            self.history.append(row)
            self._append_metrics_row(row)

            # Log results
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f} "
                  f"(min {train_stats['loss_min']:.4f} / max {train_stats['loss_max']:.4f} "
                  f"/ std {train_stats['loss_std']:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% "
                  f"(M {val_metrics.get('acc_mercosur', 0.0):.2f}% / "
                  f"B {val_metrics.get('acc_brazilian', 0.0):.2f}%) | "
                  f"LR: {current_lr:.2e} | "
                  f"GradNorm: {train_stats['grad_norm']:.2f} (max {train_stats['grad_norm_max']:.1f}) | "
                  f"Time: {epoch_time:.1f}s (train {train_time:.1f}s / val {val_time:.1f}s)")
            skipped = train_stats['skipped_batches']
            num_batches = train_stats['num_batches']
            if skipped:
                print(f"  ⚠️ Bỏ qua {skipped}/{num_batches} batch "
                      f"({train_stats['skipped_nonfinite']} do inf/nan, "
                      f"{skipped - train_stats['skipped_nonfinite']} do vượt ngưỡng "
                      f"{self.skip_grad_norm:g})")

            if is_best:
                self.save_model()
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_acc:.2f}%)")

                if submission_data:
                    self.save_submission(submission_data)

            # Circuit breaker: past this point every batch is being skipped, so the
            # remaining epochs would update nothing. Stop instead of burning GPU —
            # the cause needs fixing in the config, not working around at runtime.
            if self.skip_abort_ratio > 0 and num_batches > 0 and \
                    skipped / num_batches > self.skip_abort_ratio:
                self.aborted = True
                print(f"\n🛑 DỪNG SỚM ở epoch {epoch + 1}: "
                      f"{skipped}/{num_batches} batch bị bỏ "
                      f"({skipped / num_batches:.0%} > ngưỡng {self.skip_abort_ratio:.0%}).")
                print("   Model đã hỏng chứ không phải một batch xấu — các epoch sau sẽ không "
                      "cập nhật được gì.")
                print(f"   Checkpoint tốt nhất vẫn nguyên: {self.best_acc:.2f}% "
                      f"(epoch {self.best_epoch}).")
                print("   Xem dòng '🔍 Spike' phía trên để biết khối nào sinh gradient lớn, "
                      "rồi chỉnh config (hạ --lr, hoặc rút ngắn --epochs) và chạy lại.")
                break

        # Save final model if no validation was performed (submission mode)
        if self.val_loader is None:
            self.save_model()
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  💾 Saved final model: {model_path}")

        total_time = time.perf_counter() - run_start
        if self.history:
            self._write_summary(total_time)
        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}% "
              f"(epoch {self.best_epoch}) | Total time: {total_time / 3600:.2f} h")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on a data loader.
        
        Returns:
            List of (track_id, predicted_text, confidence) tuples.
        """
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        
        with torch.no_grad():
            for images, _, _, _, track_ids, *_ in loader:
                images = images.to(self.device)
                preds = self.model(images)
                
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        """Run inference on test data and save submission file.
        
        Args:
            test_loader: DataLoader for test data.
            output_filename: Name of the submission file to save.
        """
        print(f"🔮 Running inference on test data...")
        
        # Use existing predict method
        results = []
        self.model.eval()
        with torch.no_grad():
            for images, _, _, _, track_ids, *_ in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                preds = self.model(images)
                decoded_list = decode_with_confidence(preds, self.idx2char)
                
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        # Format and save submission file
        submission_data = [f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
        output_path = self._get_output_path(output_filename)
        with open(output_path, 'w') as f:
            f.write("\n".join(submission_data))
        
        print(f"✅ Saved {len(submission_data)} predictions to {output_path}")
