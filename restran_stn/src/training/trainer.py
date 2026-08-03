"""Trainer class encapsulating the training and validation loop."""
import csv
import math
import os
import statistics
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.use_amp = config.USE_AMP and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if getattr(config, "AMP_DTYPE", "float16") == "float16" else torch.bfloat16
        # GradScaler is only needed for float16; bfloat16 has fp32's range and
        # must NOT be loss-scaled (scaler stays a no-op pass-through).
        self.scaler = GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self.label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
        self.ema = ModelEMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None

        # Supervised SR settings
        self.use_learnable_sr = getattr(config, 'USE_LEARNABLE_SR', False)
        self.sr_loss_weight = getattr(config, 'SR_LOSS_WEIGHT', 0.3)
        self.sr_loss_weight_min = getattr(config, 'SR_LOSS_WEIGHT_MIN', 0.05)

        # HR privileged-information distillation
        self.use_hr_distill = getattr(config, 'USE_HR_DISTILL', False)
        self.hr_distill_weight = getattr(config, 'HR_DISTILL_WEIGHT', 0.5)

        # Positional head (multi-task fixed-length recognition) settings
        self.use_positional_head = getattr(config, 'USE_POSITIONAL_HEAD', False)
        self.pos_loss_weight = getattr(config, 'POS_LOSS_WEIGHT', 0.5)
        self.num_positions = getattr(config, 'NUM_POSITIONS', 7)
        self.pos_fuse = getattr(config, 'POS_FUSE_AT_INFERENCE', True)
        chars = getattr(config, 'CHARS', '')
        # class-index (0-based, matches positional head's num_chars ordering,
        # i.e. CTC index - 1 after dropping blank) <-> character
        self.char2posidx = {c: i for i, c in enumerate(chars)}
        self.pos_idx2char = {i: c for i, c in enumerate(chars)}
        self.pos_criterion = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
        )
        
        # Stochastic Weight Averaging
        self.use_swa = getattr(config, 'USE_SWA', False)
        self.swa_start = int(getattr(config, 'SWA_START_FRAC', 0.5) * config.EPOCHS)
        self.swa_bn_batches = getattr(config, 'SWA_BN_BATCHES', 200)
        self.swa_lr = getattr(config, 'SWA_LR', 5e-5)
        self.swa_state = None
        self.swa_n = 0

        # Lookahead(AdamW): maintain slow weights synced every k optimizer steps.
        self.use_lookahead = getattr(config, 'USE_LOOKAHEAD', False)
        self.lookahead_k = getattr(config, 'LOOKAHEAD_K', 5)
        self.lookahead_alpha = getattr(config, 'LOOKAHEAD_ALPHA', 0.5)
        self._la_step = 0
        self._la_params = None
        self._la_slow = None
        if self.use_lookahead:
            self._la_params = [p for p in model.parameters() if p.requires_grad]
            self._la_slow = [p.detach().clone() for p in self._la_params]

        # Rolling window of recent finite train losses; used to detect genuine
        # divergence spikes relative to the current loss scale (see _is_bad_loss).
        self._recent_losses = deque(maxlen=200)
        self.spike_mult = getattr(config, 'LOSS_SPIKE_MULT', 10.0)

        # Raw-weight collapse recovery. Divergence here shows up as the raw
        # weights decaying while EMA still looks fine, so a loss-spike guard
        # never fires. When the gap gets large the raw weights are restored
        # from EMA and the LR is cut, which rescues the run instead of losing it.
        self.collapse_margin = getattr(config, 'COLLAPSE_ROLLBACK_MARGIN', 15.0)
        self.collapse_lr_decay = getattr(config, 'COLLAPSE_LR_DECAY', 0.5)
        self._last_raw_acc = None
        self._last_ema_acc = None
        self._epoch_stats = {}
        self._csv_path = None
        self._last_sr_bilinear = 0.0

        # Gradient accumulation: run N micro-batches before one optimizer step,
        # giving an effective batch of BATCH_SIZE x N. Small batches were the
        # main source of noisy raw weights (raw-vs-EMA gap of 5-12 points even
        # in healthy epochs) and of the repeated late-training collapses.
        self.grad_accum_steps = max(1, int(getattr(config, 'GRAD_ACCUM_STEPS', 1)))
        self._micro_step = 0

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
        pos_classes = None
        if getattr(self.config, "USE_POS_CLASS_DECODE", False):
            template = getattr(self.config, "PLATE_POS_CLASSES", None)
            pos_classes = list(template) if template else None
        fuse = self.use_positional_head and self.pos_fuse
        pos_i2c = self.pos_idx2char if fuse else None
        if for_inference:
            beam_width = self.config.BEAM_WIDTH if self.config.USE_INFERENCE_BEAM else 1
            return {
                "use_tta": self.config.USE_INFERENCE_TTA,
                "beam_width": beam_width,
                "expected_length": plate_len,
                "pos_classes": pos_classes,
                "fuse_positional": fuse,
                "pos_idx2char": pos_i2c,
            }
        beam_width = self.config.BEAM_WIDTH if self.config.USE_BEAM_DECODE else 1
        return {
            "use_tta": self.config.USE_VAL_TTA,
            "beam_width": beam_width,
            "expected_length": plate_len,
            "pos_classes": pos_classes,
            "fuse_positional": fuse,
            "pos_idx2char": pos_i2c,
        }

    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    def _get_exp_name(self) -> str:
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    @staticmethod
    def _first_nonfinite(named_tensors) -> Optional[str]:
        """Name of the first tensor holding NaN/Inf, or None if all are clean."""
        for name, t in named_tensors:
            if t is None or not torch.is_tensor(t) or not t.is_floating_point():
                continue
            if not torch.isfinite(t).all():
                n_nan = int(torch.isnan(t).sum())
                n_inf = int(torch.isinf(t).sum())
                return f"{name} (nan={n_nan}, inf={n_inf}, numel={t.numel()})"
        return None

    def _check_weights_finite(self) -> Optional[str]:
        """Name of the first model parameter/buffer that has gone non-finite."""
        for name, p in self.model.named_parameters():
            if p.is_floating_point() and not torch.isfinite(p).all():
                return f"param:{name}"
        for name, b in self.model.named_buffers():
            if torch.is_tensor(b) and b.is_floating_point() and not torch.isfinite(b).all():
                return f"buffer:{name}"
        return None

    def _is_bad_loss(self, loss_value: float) -> bool:
        """Whether to skip this batch as a divergence spike.

        The threshold is RELATIVE to recent history, because the absolute CTC
        loss scales with sequence length: a wider input means more timesteps and
        therefore a larger loss at initialization. A fixed cap is not portable
        across input sizes — at width 160 (40 timesteps) the initial loss
        exceeded a fixed cap of 15 and *every* batch was skipped, so the model
        never trained at all.

        Rules: always skip non-finite; before enough history exists, trust any
        finite loss; afterwards skip only what is far above the running median.
        """
        if not math.isfinite(loss_value):
            return True
        if len(self._recent_losses) < 50:
            return False
        median = statistics.median(self._recent_losses)
        threshold = max(self.config.MAX_TRAIN_LOSS, self.spike_mult * median)
        return loss_value > threshold

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
        images: torch.Tensor = None,
    ) -> torch.Tensor:
        """L1 pixel loss for Supervised SR, plus a no-learning reference.

        The reference is exactly what the SR block adds its residual to: the
        interpolated input (or the input itself at scale 1). If the learned
        ``sr_loss`` does not stay below it, the SR module is not recovering
        anything an interpolation could not, and its extra compute buys nothing.
        The reference is stored on ``self._last_sr_bilinear`` for CSV logging.
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

        # No-learning reference: interpolate the raw input up to the target
        # size and score it with the same L1.
        if images is not None:
            with torch.no_grad():
                lr_flat = images.view(-1, *images.shape[2:])[mask]
                if lr_flat.shape[-2:] != hr_valid.shape[-2:]:
                    lr_flat = F.interpolate(
                        lr_flat.float(), size=hr_valid.shape[-2:],
                        mode="bicubic", align_corners=False,
                    )
                self._last_sr_bilinear = nn.functional.l1_loss(
                    lr_flat.float(), hr_valid.float()
                ).item()

        # 1. Pixel L1 Loss
        return nn.functional.l1_loss(sr_valid, hr_valid)

    def _forward_train(self, images: torch.Tensor):
        """Forward for training; returns (preds, sr_output, pos_logits).

        Missing outputs are None depending on which heads are enabled.
        """
        rs = self.use_learnable_sr
        rp = self.use_positional_head
        out = self.model(images, return_sr=rs, return_pos=rp)
        if rs and rp:
            preds, sr_output, pos_logits = out
        elif rs:
            preds, sr_output = out
            pos_logits = None
        elif rp:
            preds, pos_logits = out
            sr_output = None
        else:
            preds = out
            sr_output = None
            pos_logits = None
        return preds, sr_output, pos_logits

    def _build_pos_targets(self, labels_text):
        """Build [B, P] positional class targets and a [B] validity mask."""
        b = len(labels_text)
        p = self.num_positions
        targets = torch.zeros(b, p, dtype=torch.long, device=self.device)
        valid = torch.zeros(b, dtype=torch.bool, device=self.device)
        for i, s in enumerate(labels_text):
            if len(s) == p and all(c in self.char2posidx for c in s):
                for j, c in enumerate(s):
                    targets[i, j] = self.char2posidx[c]
                valid[i] = True
        return targets, valid

    def _compute_pos_loss(
        self,
        pos_logits: torch.Tensor,
        pos_targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy over valid samples' 7 positions."""
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pos_logits.device)
        pl = pos_logits[valid]          # [n, P, num_chars]
        tl = pos_targets[valid]         # [n, P]
        return self.pos_criterion(
            pl.reshape(-1, pl.size(-1)), tl.reshape(-1)
        )

    def _compute_hr_distill_loss(self, student_logits, hr_images):
        """KL distillation from a detached HR-teacher pass to the LR student.

        The clean HR frames (available for 100% of tracks) are downscaled to the
        model input size and run through the SAME model in eval mode with no
        grad — an easy, accurate "teacher". The LR student is pulled toward it.
        """
        if hr_images.abs().sum() == 0:
            return torch.tensor(0.0, device=student_logits.device)
        b, f = hr_images.shape[:2]
        hh, ww = hr_images.shape[3], hr_images.shape[4]
        hr_flat = hr_images.view(b * f, 3, hh, ww)
        hr_small = F.interpolate(
            hr_flat, size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            mode="bilinear", align_corners=False,
        )
        hr_input = hr_small.view(b, f, 3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH)

        was_training = self.model.training
        self.model.eval()  # no BN-stat update / dropout for the teacher pass
        with torch.no_grad():
            teacher = self.model(hr_input)  # [B, T, C] log_softmax
        if was_training:
            self.model.train()

        c = student_logits.size(-1)
        return F.kl_div(
            student_logits.reshape(-1, c),
            teacher.exp().reshape(-1, c),
            reduction="batchmean",
        )

    def _compute_total_loss(self, images, targets, target_lengths, hr_images,
                            labels_text, sr_weight):
        """Forward + combined CTC (+SR) (+positional) (+HR-distill) loss."""
        preds, sr_output, pos_logits = self._forward_train(images)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=preds.size(1),
            dtype=torch.long,
            device=self.device,
        )
        loss = self._compute_loss(preds, targets, input_lengths, target_lengths)

        sr_loss_val = 0.0
        if sr_output is not None and sr_weight > 0:
            sr_loss = self._compute_sr_loss(sr_output, hr_images, images)
            sr_loss_val = sr_loss.item()
            loss = loss + sr_weight * sr_loss

        pos_loss_val = 0.0
        if pos_logits is not None and self.pos_loss_weight > 0:
            pos_targets, valid = self._build_pos_targets(labels_text)
            pos_loss = self._compute_pos_loss(pos_logits, pos_targets, valid)
            pos_loss_val = pos_loss.item()
            loss = loss + self.pos_loss_weight * pos_loss

        distill_val = 0.0
        if self.use_hr_distill and self.hr_distill_weight > 0:
            distill = self._compute_hr_distill_loss(preds, hr_images)
            distill_val = distill.item()
            loss = loss + self.hr_distill_weight * distill

        return loss, sr_loss_val, pos_loss_val, distill_val

    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        epoch_sr_loss = 0.0
        epoch_pos_loss = 0.0
        epoch_distill_loss = 0.0
        valid_steps = 0
        skipped_batches = 0
        nan_batches = 0
        grad_norm_sum = 0.0
        grad_norm_n = 0
        sr_weight = self._sr_weight()
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")

        for images, targets, target_lengths, labels_text, _, hr_images in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            target_lengths = target_lengths.to(self.device, non_blocking=True)
            hr_images = hr_images.to(self.device, non_blocking=True)

            # With accumulation, gradients are only cleared at the start of a
            # new accumulation window.
            if self._micro_step % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            if self.use_amp:
                with autocast(device_type, dtype=self.amp_dtype):
                    loss, sr_loss_val, pos_loss_val, distill_val = self._compute_total_loss(
                        images, targets, target_lengths, hr_images,
                        labels_text, sr_weight,
                    )
            else:
                loss, sr_loss_val, pos_loss_val, distill_val = self._compute_total_loss(
                    images, targets, target_lengths, hr_images,
                    labels_text, sr_weight,
                )

            loss_value = loss.item()
            if math.isfinite(loss_value):
                self._recent_losses.append(loss_value)
            else:
                # A non-finite loss is silent about its cause; report the first
                # offending tensor so a data problem is not mistaken for a
                # divergence (and vice versa). Capped so logs stay readable.
                nan_batches += 1
                if nan_batches <= 3:
                    src = self._first_nonfinite([
                        ("input images", images),
                        ("hr targets", hr_images),
                    ]) or self._check_weights_finite() or "loss computation (inputs+weights finite)"
                    print(f"  🚨 NaN/Inf loss at epoch {self.current_epoch + 1} -> source: {src}")
            if self._is_bad_loss(loss_value):
                skipped_batches += 1
                pbar.set_postfix({
                    'loss': 'skip',
                    'lr': f"{self._current_lr():.2e}",
                    'skip': skipped_batches,
                })
                # Drop this micro-batch's contribution but keep the window
                # aligned; a partially-filled window still steps at its end.
                self._micro_step += 1
                if self._micro_step % self.grad_accum_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            # Scale so accumulated gradients average (not sum) over the window.
            accum = self.grad_accum_steps
            if self.use_amp:
                self.scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()

            self._micro_step += 1
            at_window_end = (self._micro_step % accum == 0)
            if not at_window_end:
                # Keep accumulating; stats below still count this micro-batch.
                epoch_loss += loss_value
                epoch_sr_loss += sr_loss_val
                epoch_pos_loss += pos_loss_val
                epoch_distill_loss += distill_val
                valid_steps += 1
                continue

            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRAD_CLIP
                )
                if not torch.isfinite(grad_norm):
                    # Non-finite grads (fp16 overflow): drop this step so a
                    # spike can't corrupt the weights, and let AMP lower scale.
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                    skipped_batches += 1
                    continue
                grad_norm_sum += float(grad_norm)
                grad_norm_n += 1
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                stepped = self.scaler.get_scale() >= scale_before
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRAD_CLIP
                )
                if not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    continue
                grad_norm_sum += float(grad_norm)
                grad_norm_n += 1
                self.optimizer.step()
                stepped = True

            if stepped and self.use_lookahead:
                self._lookahead_sync()

            if stepped and self.ema is not None:
                self.ema.update(self.model)

            if stepped and self._step_scheduler_per_batch:
                self.scheduler.step()

            epoch_loss += loss_value
            epoch_sr_loss += sr_loss_val
            epoch_pos_loss += pos_loss_val
            epoch_distill_loss += distill_val
            valid_steps += 1
            postfix = {'loss': loss_value, 'lr': f"{self._current_lr():.2e}"}
            if self.use_learnable_sr:
                postfix['sr'] = f"{sr_loss_val:.4f}"
            if self.use_positional_head:
                postfix['pos'] = f"{pos_loss_val:.4f}"
            if self.use_hr_distill:
                postfix['kd'] = f"{distill_val:.4f}"
            pbar.set_postfix(postfix)

        if valid_steps == 0:
            self._epoch_stats = {
                "sr_loss": 0.0, "pos_loss": 0.0, "distill_loss": 0.0,
                "sr_weight": sr_weight, "skipped_batches": skipped_batches,
                "nan_batches": nan_batches, "weights_nonfinite": "",
                "sr_loss_bilinear": self._last_sr_bilinear, "grad_norm": 0.0,
            }
            return float("inf")
        if skipped_batches > 0:
            print(f"  ⚠️ Skipped {skipped_batches} unstable batches this epoch.")
        if self.use_learnable_sr and valid_steps > 0:
            avg_sr = epoch_sr_loss / valid_steps
            print(f"  🔍 SR Loss: {avg_sr:.4f} | λ_sr: {sr_weight:.4f}")
        if self.use_positional_head and valid_steps > 0:
            avg_pos = epoch_pos_loss / valid_steps
            print(f"  🎯 Pos Loss: {avg_pos:.4f} | λ_pos: {self.pos_loss_weight:.2f}")
        if self.use_hr_distill and valid_steps > 0:
            avg_kd = epoch_distill_loss / valid_steps
            print(f"  🎓 HR-Distill KL: {avg_kd:.4f} | λ_kd: {self.hr_distill_weight:.2f}")
        if nan_batches > 0:
            print(f"  🚨 {nan_batches} batches had a non-finite loss this epoch.")
        bad_w = self._check_weights_finite()
        if bad_w:
            print(f"  🚨 Model weights are non-finite after this epoch: {bad_w}")

        self._epoch_stats = {
            "sr_loss": epoch_sr_loss / valid_steps if valid_steps else 0.0,
            "pos_loss": epoch_pos_loss / valid_steps if valid_steps else 0.0,
            "distill_loss": epoch_distill_loss / valid_steps if valid_steps else 0.0,
            "sr_weight": sr_weight,
            "skipped_batches": skipped_batches,
            "nan_batches": nan_batches,
            "weights_nonfinite": bad_w or "",
            "sr_loss_bilinear": self._last_sr_bilinear,
            "grad_norm": grad_norm_sum / grad_norm_n if grad_norm_n else 0.0,
        }
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
        # Brazilian (LLLDDDD) has consistently scored ~25 points below Mercosur
        # (LLLDLDD); tracking both per epoch shows whether a change helps the
        # weak layout or just the majority one. Greedy is scored alongside the
        # configured decode so the decode's own contribution stays visible.
        greedy_correct = 0
        layout_ok = {"Brazilian": 0, "Mercosur": 0}
        layout_tot = {"Brazilian": 0, "Mercosur": 0}
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

                fuse = self.use_positional_head and self.pos_fuse
                pos_logits = None
                if decode_kwargs.get("use_tta"):
                    from src.utils.postprocess import forward_with_tta
                    preds = forward_with_tta(self.model, images)
                    if fuse:  # positional head needs a direct (non-TTA) pass
                        _p, pos_logits = self.model(images, return_pos=True)
                elif self.use_amp:
                    with autocast(device_type, dtype=self.amp_dtype):
                        if fuse:
                            preds, pos_logits = self.model(images, return_pos=True)
                        else:
                            preds = self.model(images)
                else:
                    if fuse:
                        preds, pos_logits = self.model(images, return_pos=True)
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
                    pos_classes=decode_kwargs.get("pos_classes"),
                )
                if fuse and pos_logits is not None:
                    from src.utils.postprocess import (
                        decode_positional, fuse_ctc_positional,
                    )
                    pos_list = decode_positional(
                        pos_logits,
                        self.pos_idx2char,
                        pos_classes=decode_kwargs.get("pos_classes"),
                    )
                    decoded_list = fuse_ctc_positional(decoded_list, pos_list)

                greedy_list = (
                    decode_with_confidence(preds, self.idx2char, beam_width=1)
                    if decode_kwargs.get("beam_width", 1) > 1 else decoded_list
                )

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    if pred_text == gt_text:
                        total_correct += 1
                    if greedy_list[i][0] == gt_text:
                        greedy_correct += 1
                    if len(gt_text) == 7:
                        lay = "Mercosur" if gt_text[4].isalpha() else "Brazilian"
                        layout_tot[lay] += 1
                        if pred_text == gt_text:
                            layout_ok[lay] += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")

                total_samples += len(labels_text)

        if backup_state is not None:
            self.model.load_state_dict(backup_state)

        avg_val_loss = val_loss / max(len(self.val_loader), 1)

        def _pct(a, b):
            return (a / b * 100) if b else 0.0

        val_acc = _pct(total_correct, total_samples)
        return {
            'loss': avg_val_loss,
            'acc': val_acc,
            'acc_greedy': _pct(greedy_correct, total_samples),
            'acc_brazilian': _pct(layout_ok["Brazilian"], layout_tot["Brazilian"]),
            'acc_mercosur': _pct(layout_ok["Mercosur"], layout_tot["Mercosur"]),
        }, submission_data

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

        self._last_raw_acc = raw_metrics['acc']
        self._last_ema_acc = ema_metrics['acc']

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

    def _scale_lr(self, factor: float) -> None:
        """Permanently scale the LR, including the scheduler's base values.

        Scaling only ``param_groups`` is not enough: schedulers recompute the LR
        from ``base_lrs`` on every step, which would immediately undo it.
        """
        for g in self.optimizer.param_groups:
            g['lr'] *= factor
        scheds = getattr(self.scheduler, '_schedulers', None) or [self.scheduler]
        for sch in scheds:
            if hasattr(sch, 'base_lrs'):
                sch.base_lrs = [lr * factor for lr in sch.base_lrs]

    def _lookahead_sync(self) -> None:
        """Every k steps, pull fast weights toward slow weights and copy back."""
        self._la_step += 1
        if self._la_step % self.lookahead_k != 0:
            return
        with torch.no_grad():
            for p, slow in zip(self._la_params, self._la_slow):
                slow.add_(p.data - slow, alpha=self.lookahead_alpha)
                p.data.copy_(slow)

    def _set_swa_constant_lr(self) -> None:
        """Hold a constant LR during the SWA collection phase (per param group)."""
        ratio = self.config.BACKBONE_LR_RATIO
        lrs = [self.swa_lr * ratio, self.swa_lr]  # [backbone, head]
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g['lr'] = lr

    def _update_swa(self) -> None:
        """Accumulate an equal-weight running average of the raw model weights."""
        src = self.model.state_dict()
        if self.swa_state is None:
            self.swa_state = {k: v.detach().clone().float() for k, v in src.items()}
            self.swa_n = 1
            return
        self.swa_n += 1
        inv = 1.0 / self.swa_n
        for k, v in src.items():
            if self.swa_state[k].is_floating_point():
                self.swa_state[k].mul_(1.0 - inv).add_(v.detach().float(), alpha=inv)
            else:
                self.swa_state[k] = v.detach().clone()

    def _bn_update(self) -> None:
        """Recompute BatchNorm running stats for the current (SWA) weights."""
        bn_mods = [
            m for m in self.model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ]
        if not bn_mods:
            return
        momenta = {}
        for m in bn_mods:
            m.reset_running_stats()
            momenta[m] = m.momentum
            m.momentum = None  # cumulative moving average
        self.model.train()
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        seen = 0
        with torch.no_grad():
            for batch in self.train_loader:
                if seen >= self.swa_bn_batches:
                    break
                images = batch[0].to(self.device, non_blocking=True)
                if self.use_amp:
                    with autocast(device_type, dtype=self.amp_dtype):
                        self.model(images)
                else:
                    self.model(images)
                seen += 1
        for m in bn_mods:
            m.momentum = momenta[m]

    def _finalize_swa(self) -> None:
        """Load the SWA average, refresh BN, evaluate, and keep it if better."""
        if not self.use_swa or self.swa_state is None:
            return
        target = self.model.state_dict()
        new_state = {}
        for k, v in self.swa_state.items():
            if target[k].is_floating_point():
                new_state[k] = v.to(dtype=target[k].dtype, device=target[k].device)
            else:
                new_state[k] = v
        self.model.load_state_dict(new_state)
        self._bn_update()

        exp_name = self._get_exp_name()
        swa_path = self._get_output_path(f"{exp_name}_swa.pth")
        torch.save(self.model.state_dict(), swa_path)

        if self.val_loader is None:
            # No val to compare against — SWA becomes the final model.
            self.save_model(path=self._get_output_path(f"{exp_name}_best.pth"), use_ema=False)
            print(f"  🧪 SWA (n={self.swa_n}) saved as final model: {swa_path}")
            return

        decode_kwargs = self._decode_kwargs(for_inference=False)
        metrics, submission = self._run_validation_pass(use_ema=False, decode_kwargs=decode_kwargs)
        print(
            f"  🧪 SWA (n={self.swa_n}) Val Acc: {metrics['acc']:.2f}%  "
            f"(best single-epoch: {self.best_acc:.2f}%)"
        )
        if metrics['acc'] > self.best_acc:
            self.best_acc = metrics['acc']
            self.best_epoch = -1  # -1 marks the SWA model
            best_path = self._get_output_path(f"{exp_name}_best.pth")
            torch.save(self.model.state_dict(), best_path)
            print(f"  ⭐ SWA is the new best: {best_path} ({metrics['acc']:.2f}%)")
            if submission:
                self.save_submission(submission)

    def _log_epoch_csv(self, row: Dict) -> None:
        """Append one epoch's metrics to results/<experiment>_history.csv.

        The header is written from the first row's keys, so the file is created
        fresh per run (an existing file from an earlier run is overwritten).
        """
        if self._csv_path is None:
            exp = self._get_exp_name()
            self._csv_path = self._get_output_path(f"{exp}_history.csv")
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writeheader()
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

    def fit(self) -> None:
        print(
            f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS} | "
            f"LR: {self.config.LEARNING_RATE:.2e} | Warmup: {self.config.WARMUP_EPOCHS} | "
            f"EMA: {self.config.USE_EMA} | Val TTA/Beam: {self.config.USE_VAL_TTA}/{self.config.USE_BEAM_DECODE}"
        )
        
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            self._set_dataset_epoch(epoch)
            epoch_t0 = time.time()
            
            avg_train_loss = self.train_one_epoch()
            val_metrics, submission_data, use_ema_for_best = self.validate_best_variant()
            val_loss = val_metrics['loss']
            val_acc = val_metrics['acc']

            # Rescue a collapsing raw model: restore EMA weights and cool the LR.
            if (
                self.ema is not None
                and self._last_raw_acc is not None
                and self._last_ema_acc - self._last_raw_acc > self.collapse_margin
            ):
                self.ema.load_into(self.model)
                self._scale_lr(self.collapse_lr_decay)
                print(
                    f"  🛟 Raw weights collapsing (raw {self._last_raw_acc:.2f}% vs "
                    f"EMA {self._last_ema_acc:.2f}%) -> restored EMA, "
                    f"LR x{self.collapse_lr_decay}"
                )

            # SWA: accumulate raw weights over the final epochs (model currently
            # holds raw weights after validate_best_variant). Skip epochs whose
            # val acc dipped well below the best (transient divergence) so they
            # don't poison the average. best_acc here is still the pre-update
            # best from earlier epochs.
            if self.use_swa and epoch >= self.swa_start and math.isfinite(avg_train_loss):
                margin = getattr(self.config, 'SWA_ACC_MARGIN', 3.0)
                # Include if it's the best so far or close to it; skip dips.
                if val_acc >= max(self.best_acc - margin, 0.0):
                    self._update_swa()
                else:
                    print(
                        f"  ⏭️ SWA skip epoch {epoch + 1}: val {val_acc:.2f}% "
                        f"< best {self.best_acc:.2f}% − {margin:.1f}"
                    )

            # During the SWA collection phase, hold a constant LR (proper SWA)
            # instead of letting cosine decay toward zero.
            next_in_swa = self.use_swa and (epoch + 1) >= self.swa_start
            if not self._step_scheduler_per_batch:
                if next_in_swa:
                    self._set_swa_constant_lr()
                else:
                    self.scheduler.step()
            current_lr = self._current_lr()
            
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            st = self._epoch_stats
            self._log_epoch_csv({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 4),
                "val_raw_acc": (round(self._last_raw_acc, 4)
                                if self._last_raw_acc is not None else ""),
                "val_ema_acc": (round(self._last_ema_acc, 4)
                                if self._last_ema_acc is not None else ""),
                "used_ema": int(bool(use_ema_for_best)),
                "lr": f"{current_lr:.6e}",
                "acc_greedy": round(val_metrics.get("acc_greedy", 0.0), 4),
                "acc_brazilian": round(val_metrics.get("acc_brazilian", 0.0), 4),
                "acc_mercosur": round(val_metrics.get("acc_mercosur", 0.0), 4),
                "sr_loss": round(st.get("sr_loss", 0.0), 6),
                "sr_loss_bilinear": round(st.get("sr_loss_bilinear", 0.0), 6),
                "sr_beats_interp": int(
                    st.get("sr_loss", 0.0) < st.get("sr_loss_bilinear", 0.0)
                ),
                "sr_weight": round(st.get("sr_weight", 0.0), 6),
                "grad_norm": round(st.get("grad_norm", 0.0), 4),
                "pos_loss": round(st.get("pos_loss", 0.0), 6),
                "distill_loss": round(st.get("distill_loss", 0.0), 6),
                "skipped_batches": st.get("skipped_batches", 0),
                "nan_batches": st.get("nan_batches", 0),
                "weights_nonfinite": st.get("weights_nonfinite", ""),
                "best_acc": round(max(self.best_acc, val_acc), 4),
                "epoch_seconds": round(time.time() - epoch_t0, 1),
            })
            
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

            # Don't early-stop once SWA collection has begun — let it average
            # the full final window before finalizing.
            in_swa_phase = self.use_swa and epoch >= self.swa_start
            if (
                self.val_loader is not None
                and self.config.EARLY_STOP_PATIENCE > 0
                and self.epochs_without_improvement >= self.config.EARLY_STOP_PATIENCE
                and not in_swa_phase
            ):
                print(
                    f"⏹️ Early stopping: no val improvement for "
                    f"{self.config.EARLY_STOP_PATIENCE} epochs."
                )
                break

            if not math.isfinite(avg_train_loss):
                print("⏹️ Stopping early due to unstable training loss.")
                break

        if self.val_loader is None and not self.use_swa:
            self.save_model(use_ema=True)
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  💾 Saved final model: {model_path}")

        # Finalize SWA: average weights, refresh BN, keep if it beats the best.
        if self.use_swa:
            print("🧪 Finalizing SWA (averaging weights + BN recalibration)...")
            self._finalize_swa()

        best_tag = "SWA" if self.best_epoch == -1 else f"epoch {self.best_epoch}"
        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}% ({best_tag})")

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
