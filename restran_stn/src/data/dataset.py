"""MultiFrameDataset for license plate recognition with multi-frame input."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_jpeg_domain_transform,
    get_normalize_transform,
    get_paired_train_augment,
    get_train_transforms,
    get_val_transforms,
    get_degradation_transforms,
    get_light_transforms,
    apply_lightweight_sr,
    resize_plate_image,
)


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame license plate recognition.
    
    Handles both real LR images and synthetic LR (degraded HR) images.
    Implements Scenario-B specific validation splitting logic.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
        use_sr: bool = True,
        sr_scale: int = 2,
        sr_upsample: int = 1,
        jpeg_domain_aug: bool = False,
        layout_balance: bool = False,
        use_synthetic_lr: bool = True,
        synthetic_from_lr: bool = True,
        synthetic_sample_prob: float = 0.35,
        degradation_curriculum: bool = True,
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            img_height: Target image height.
            img_width: Target image width.
            char2idx: Character to index mapping.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
            augmentation_level: 'full' or 'light' augmentation for training.
            is_test: If True, load test data without labels (for submission).
            full_train: If True, use all tracks for training (no val split).
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        self.use_sr = use_sr
        self.sr_scale = sr_scale
        self.sr_upsample = max(1, sr_upsample)
        self.jpeg_domain_aug = jpeg_domain_aug
        self.layout_balance = layout_balance
        self._jpeg_transform = get_jpeg_domain_transform() if jpeg_domain_aug else None
        # Paired LR/HR augmentation (training only, full augmentation level):
        # identical geometric params for input and SR target.
        self._paired_augment = (
            get_paired_train_augment()
            if mode == 'train' and augmentation_level != 'light' else None
        )
        self._normalize = get_normalize_transform()
        self.use_synthetic_lr = use_synthetic_lr
        self.synthetic_from_lr = synthetic_from_lr
        self.synthetic_sample_prob = synthetic_sample_prob
        self.degradation_curriculum = degradation_curriculum
        self.current_epoch = 0
        self.total_epochs = 1
        
        if mode == 'train':
            # Training: apply augmentation on the fly
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = None
        else:
            # Validation or test: only resize and normalize
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        # Handle test mode differently
        if is_test:
            print(f"[TEST] Loaded {len(all_tracks)} tracks.")
            self._index_test_samples(all_tracks)
            print(f"-> Total: {len(self.samples)} test samples.")
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            
            selected_tracks = train_tracks if mode == 'train' else val_tracks
            print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
            
            self._index_samples(selected_tracks)
            print(f"-> Total: {len(self.samples)} samples.")

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Update curriculum strength for synthetic degradation."""
        self.current_epoch = max(0, epoch)
        self.total_epochs = max(1, total_epochs)

    def _degradation_strength(self) -> float:
        if not self.degradation_curriculum or self.mode != 'train':
            return 1.0
        # The author always degrades at full strength. Starting from 0.25 over a
        # 40% warmup left the synthetic LR far too clean for most of training,
        # so ramp from a much higher floor and reach full strength sooner.
        warmup_epochs = max(1, int(self.total_epochs * 0.25))
        progress = min(1.0, (self.current_epoch + 1) / warmup_epochs)
        return 0.6 + 0.4 * progress

    def _get_degrade_transform(self, from_lr: bool = False):
        return get_degradation_transforms(
            strength=self._degradation_strength(),
            from_lr=from_lr,
        )

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one with Scenario-B priority."""
        # If full_train mode, return all tracks as training
        if self.full_train:
            print("📌 FULL TRAIN MODE: Using all tracks for training (no validation split).")
            return all_tracks, []
        
        train_tracks, val_tracks = [], []
        
        # 1. Load split file if exists
        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            # Check consistency: If val empty or no Scenario-B, recreate
            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split file invalid or missing Scenario-B. Recreating...")
                val_tracks = []  # Reset to trigger new split logic

        # 2. Create new split if needed
        if not val_tracks:
            print("⚠️ Creating new split (Taking Val only from Scenario-B)...")
            
            # Filter Scenario-B tracks
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
            
            if not scenario_b_tracks:
                print("⚠️ Warning: No 'Scenario-B' folder found. Using random from all.")
                scenario_b_tracks = all_tracks
            
            # Val size = (1 - split_ratio) * total_scenario_b
            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            
            # Shuffle and take from beginning as val
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            
            # Train = (All) - (Val)
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]
            
            # Save track IDs (folder names)
            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Index all samples from selected tracks."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                
                track_id = os.path.basename(track_path)
                
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                # Scenario-A stores PNG (lossless); Scenario-B stores JPEG. The
                # val/test domain is Scenario-B, so PNG-domain tracks need JPEG
                # artifacts injected to close that domain gap.
                is_png_domain = bool(lr_files) and lr_files[0].lower().endswith('.png')
                layout = (
                    'Brazilian' if 'Brazilian' in track_path
                    else ('Mercosur' if 'Mercosur' in track_path else 'unknown')
                )

                # Real LR samples (include HR paths for supervised SR)
                if lr_files:
                    self.samples.append({
                        'paths': lr_files,
                        'hr_paths': hr_files if len(hr_files) == len(lr_files) else [],
                        'label': label,
                        'is_synthetic': False,
                        'synth_from_lr': False,
                        'track_id': track_id,
                        'is_png_domain': is_png_domain,
                        'layout': layout,
                    })
                
                # Synthetic LR samples (training only, probabilistic to reduce overfitting)
                if self.mode == 'train' and self.use_synthetic_lr:
                    track_rng = random.Random(f"{self.seed}:{track_id}")
                    if track_rng.random() < self.synthetic_sample_prob:
                        if hr_files:
                            self.samples.append({
                                'paths': hr_files,
                                'hr_paths': hr_files,  # HR targets = original HR
                                'label': label,
                                'is_synthetic': True,
                                'synth_from_lr': False,
                                'track_id': track_id,
                                'is_png_domain': is_png_domain,
                                'layout': layout,
                            })
                        elif self.synthetic_from_lr and lr_files:
                            self.samples.append({
                                'paths': lr_files,
                                'hr_paths': [],  # No HR available
                                'label': label,
                                'is_synthetic': True,
                                'synth_from_lr': True,
                                'track_id': track_id,
                                'is_png_domain': is_png_domain,
                                'layout': layout,
                            })
            except Exception:
                pass

        if self.mode == 'train' and self.layout_balance:
            self._balance_layouts()

    def _balance_layouts(self) -> None:
        """Oversample the minority plate layout to match the majority count.

        Brazilian plates are 35% of training but score 28 points worse on val
        (55.2% vs 83.5%), so the minority layout is duplicated until both
        layouts are equally represented.
        """
        by_layout: Dict[str, List[int]] = {}
        for i, s in enumerate(self.samples):
            by_layout.setdefault(s.get('layout', 'unknown'), []).append(i)
        known = {k: v for k, v in by_layout.items() if k != 'unknown'}
        if len(known) < 2:
            return
        target = max(len(v) for v in known.values())
        rng = random.Random(self.seed)
        added = 0
        for lay, idxs in known.items():
            deficit = target - len(idxs)
            if deficit <= 0:
                continue
            for _ in range(deficit):
                self.samples.append(dict(self.samples[rng.choice(idxs)]))
            added += deficit
            print(f"  ⚖️ Layout balance: +{deficit} '{lay}' samples "
                  f"({len(idxs)} -> {target})")
        if added:
            print(f"-> Total after balancing: {len(self.samples)} samples.")

    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test samples without labels."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            track_id = os.path.basename(track_path)
            
            # Load all LR images (sorted by frame number)
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png")) +
                glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            
            if lr_files:
                self.samples.append({
                    'paths': lr_files,
                    'label': '',  # No label for test data
                    'is_synthetic': False,
                    'track_id': track_id
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str, torch.Tensor]:
        """Load exactly 5 frames (guaranteed by dataset structure).
        
        For training: applies degradation (if synthetic) then augmentation.
        For validation: applies degradation (if synthetic) then clean transform.
        For test: only applies clean transform, returns dummy targets.

        Returns:
            images_tensor: [Frames, 3, H, W]
            target: encoded label
            target_len: label length
            label: raw label string
            track_id: track identifier
            hr_tensor: [Frames, 3, H, W] HR targets for SR loss (zeros if unavailable)
        """
        item = self.samples[idx]
        img_paths = item['paths']
        hr_paths = item.get('hr_paths', [])
        label = item['label']
        is_synthetic = item['is_synthetic']
        synth_from_lr = item.get('synth_from_lr', False)
        track_id = item['track_id']
        
        # Validation-only normalize for HR targets
        hr_transform = get_val_transforms(self.img_height, self.img_width)
        # HR supervised at SR output resolution (sr_upsample× the LR grid)
        hr_h = self.img_height * self.sr_upsample
        hr_w = self.img_width * self.sr_upsample

        images_list = []
        hr_list = []
        has_hr = len(hr_paths) == len(img_paths)
        
        for i, p in enumerate(img_paths):
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply degradation first (if synthetic), before resize/enhancement
            if is_synthetic:
                degrade = self._get_degrade_transform(from_lr=synth_from_lr)
                if len(degrade.transforms) > 0:
                    image = degrade(image=image)['image']

            # Close the PNG->JPEG domain gap: Scenario-A is lossless PNG but the
            # val/test domain (Scenario-B) is JPEG, so inject JPEG artifacts into
            # PNG-domain training samples at their native resolution.
            if (
                self.mode == 'train'
                and self.jpeg_domain_aug
                and item.get('is_png_domain', False)
            ):
                image = self._jpeg_transform(image=image)['image']

            # Load the HR target up-front: when it exists we must augment it
            # together with the LR input so the SR objective stays valid.
            hr_img = None
            if has_hr:
                hr_raw = cv2.imread(hr_paths[i], cv2.IMREAD_COLOR)
                if hr_raw is None:
                    has_hr = False
                else:
                    hr_img = resize_plate_image(
                        cv2.cvtColor(hr_raw, cv2.COLOR_BGR2RGB), hr_h, hr_w
                    )

            paired = (
                self.mode == 'train'
                and hr_img is not None
                and self._paired_augment is not None
            )

            if paired:
                # Augment ONCE at the target resolution so the LR input and the
                # HR target undergo exactly the same geometric warp, then bring
                # the LR back down to the model's input size.
                lr_big = resize_plate_image(image, hr_h, hr_w)
                out = self._paired_augment(image=lr_big, hr=hr_img)
                lr_aug, hr_img = out['image'], out['hr']
                if (hr_h, hr_w) != (self.img_height, self.img_width):
                    lr_aug = cv2.resize(
                        lr_aug, (self.img_width, self.img_height),
                        interpolation=cv2.INTER_AREA,
                    )
                if self.use_sr:
                    lr_aug = apply_lightweight_sr(lr_aug, scale=self.sr_scale)
                images_list.append(self._normalize(image=lr_aug)['image'])
                hr_list.append(self._normalize(image=hr_img)['image'])
            else:
                image = resize_plate_image(image, self.img_height, self.img_width)
                # SR at model resolution: internal upscale enhances detail,
                # then downscales back
                if self.use_sr:
                    image = apply_lightweight_sr(image, scale=self.sr_scale)
                image = self.transform(image=image)['image']
                images_list.append(image)
                if hr_img is not None:
                    hr_list.append(hr_transform(image=hr_img)['image'])

        if not images_list:
            raise RuntimeError(f"No valid frames loaded for track {track_id}")

        images_tensor = torch.stack(images_list, dim=0)

        # HR tensor: real HR targets or zeros sentinel (at SR output resolution)
        if has_hr and len(hr_list) == len(images_list):
            hr_tensor = torch.stack(hr_list, dim=0)
        else:
            hr_tensor = torch.zeros(
                (images_tensor.size(0), 3, hr_h, hr_w),
                dtype=images_tensor.dtype,
            )
        
        # Handle test mode (no labels)
        if self.is_test:
            target = [0]  # Dummy target
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if len(target) == 0:
                target = [0]
            target_len = len(target)
            
        return images_tensor, torch.tensor(target, dtype=torch.long), target_len, label, track_id, hr_tensor

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...], torch.Tensor]:
        """Custom collate function for DataLoader."""
        images, targets, target_lengths, labels_text, track_ids, hr_images = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        hr_images = torch.stack(hr_images, 0)
        return images, targets, target_lengths, labels_text, track_ids, hr_images
