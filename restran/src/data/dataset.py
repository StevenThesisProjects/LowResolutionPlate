"""MultiFrameDataset for license plate recognition with multi-frame input."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_degradation_transforms,
    get_light_transforms,
    get_native_degradation_transforms,
    get_native_geometric_transforms,
    get_native_photometric_transforms,
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
        channel_shuffle: bool = True,
        consistent_aug: bool = False,
        use_hr: bool = False,
        with_hr_pair: bool = False,
        single_frame: bool = False,
        no_synthetic: bool = False,
        pseudo_labels: str = None,
        pseudo_root: str = None,
        pseudo_min_conf: float = 0.99,
        degrade_strength: str = 'normal',
        aug_degrade: str = 'none',
        registration_file: str = None,
        letterbox: bool = False,
        syn_pipeline: str = 'legacy',
        canvas_size: Tuple[int, int] = None,
        return_syn_target: bool = False,
        relative_dropout: bool = False,
        grayscale: bool = False,
        clean_geo: bool = False,
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
        self.layouts: Dict[str, str] = {}  # track_id -> "Mercosur" / "Brazilian"
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        # Only meaningful for the training pipeline; val/test are never augmented.
        self.consistent_aug = consistent_aug and mode == 'train'
        self.use_hr = use_hr
        # Distillation: also return the track's HR frames so a teacher can encode them.
        self.with_hr_pair = with_hr_pair and mode == 'train'
        # Ablation: feed one frame replicated 5x. Keeps the [B,5,...] contract that
        # AttentionFusion hard-codes, and softmax over 5 identical features is a
        # no-op — so any gap versus the real 5 frames is what fusion actually buys.
        self.single_frame = single_frame
        # Half of every training epoch is HR degraded into "synthetic LR". Slide 45
        # shows that synthetic copy is visibly easier than the real thing, so dropping
        # it tests whether the domain gap it introduces costs more than the data adds.
        self.no_synthetic = no_synthetic
        self.pseudo_labels = pseudo_labels
        self.pseudo_root = pseudo_root
        self.pseudo_min_conf = pseudo_min_conf
        self.letterbox = letterbox
        # Luma only, replicated to 3 channels. Every Scenario-B (and test) LR frame is
        # a 4:2:0 JPEG: chroma is stored at half resolution in both axes, ~9 px tall on
        # a 17 px plate, so it carries compression noise and colour bleeding, not glyphs.
        # Applied to the final tensors in __getitem__, so train, val and test match.
        self.grayscale = grayscale
        # ECC-registered HR targets: makes a pixel-wise SR loss valid despite
        # lr-00i and hr-00i being different captures (see tools/register_pairs.py).
        self.registration = None
        if registration_file and os.path.exists(registration_file) and mode == 'train':
            import numpy as _np
            d = _np.load(registration_file, allow_pickle=True)
            self.registration = {
                str(k): (w, float(q))
                for k, w, q in zip(d['keys'], d['warps'], d['ncc_after'])
            }
            print(f"📐 Nạp {len(self.registration)} cặp đã đăng ký từ '{registration_file}'")
        
        # Synthetic branch recipe. 'legacy' degrades the HR at its own resolution and
        # then augments like a real frame; 'native' augments the HR once on the target
        # canvas, then shrinks it to the paired real LR frame's pixel size and degrades
        # there (see transforms.py for the measurements behind it). Only 'native'
        # yields a pixel-aligned HR target, which supervised SR needs.
        if syn_pipeline not in ('legacy', 'native'):
            raise ValueError(f"syn_pipeline must be 'legacy' or 'native', got {syn_pipeline!r}")
        if syn_pipeline == 'native' and letterbox:
            raise ValueError("syn_pipeline='native' does not support letterbox")
        self.syn_pipeline = syn_pipeline if mode == 'train' else 'legacy'
        self.canvas_size = tuple(canvas_size) if canvas_size else (img_height, img_width)
        self.return_syn_target = return_syn_target and self.syn_pipeline == 'native'
        self._lr_size_cache: Dict[str, Tuple[int, int]] = {}

        if mode == 'train':
            # Training: apply augmentation on the fly
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width, letterbox=letterbox)
            else:
                self.transform = get_train_transforms(
                    img_height, img_width,
                    channel_shuffle=channel_shuffle,
                    consistent=self.consistent_aug,
                    letterbox=letterbox,
                    aug_degrade=aug_degrade,
                    relative_dropout=relative_dropout,
                    clean_geo=clean_geo,
                )
            self.degrade = get_degradation_transforms(degrade_strength)
            self.hr_transform = get_val_transforms(img_height, img_width, letterbox=letterbox)
            if self.syn_pipeline == 'native':
                self.native_geo = get_native_geometric_transforms(clean_geo)
                self.native_pho = get_native_photometric_transforms()
                self.native_degrade = get_native_degradation_transforms()
                # Input: resize the degraded native-size frame exactly the way a real
                # LR frame is resized at val time. Target: already on the canvas.
                self.native_input = get_val_transforms(img_height, img_width)
                self.native_target = get_val_transforms(*self.canvas_size)
        else:
            # Validation or test: only resize and normalize
            self.transform = get_val_transforms(img_height, img_width, letterbox=letterbox)
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
            if mode == 'train' and self.pseudo_labels and self.pseudo_root:
                self._index_pseudo_samples()
            print(f"-> Total: {len(self.samples)} samples.")

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
            else:
                # An existing file wins over split_ratio, which silently makes the
                # flag look broken. Say so rather than training on the wrong split.
                n_b = sum(1 for t in all_tracks if "Scenario-B" in t)
                expected = max(1, int(n_b * (1 - split_ratio)))
                if abs(len(val_tracks) - expected) > 1:
                    print(f"⚠️ CẢNH BÁO: '{self.val_split_file}' cho val {len(val_tracks)} track, "
                          f"nhưng split_ratio={split_ratio} ứng với {expected} track.")
                    print("   File split có sẵn được ưu tiên -> split_ratio bị BỎ QUA.")
                    print("   Muốn dùng tỉ lệ mới: xoá file đó, hoặc trỏ --val-split-file "
                          "sang một đường dẫn mới.")

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
                # Plate layout (Mercosur / Brazilian) drives per-layout metrics and
                # class-balanced sampling; the two differ a lot in accuracy.
                layout = data.get('plate_layout', 'Unknown')
                self.layouts[track_id] = layout

                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                if self.use_hr:
                    # Ceiling measurement: feed the undegraded HR frames on both
                    # sides. Says how well this architecture could ever read these
                    # plates if the input were sharp, which bounds what any
                    # super-resolution front-end can buy.
                    self.samples.append({
                        'paths': hr_files,
                        'label': label,
                        'is_synthetic': False,
                        'track_id': track_id,
                        'layout': layout
                    })
                    continue

                # Real LR samples
                self.samples.append({
                    'paths': lr_files,
                    'label': label,
                    'is_synthetic': False,
                    'track_id': track_id,
                    'layout': layout,
                    'hr_paths': hr_files
                })

                # Synthetic LR samples (only in training mode)
                if self.mode == 'train' and not self.no_synthetic:
                    self.samples.append({
                        'paths': hr_files,
                        'label': label,
                        'is_synthetic': True,
                        'track_id': track_id,
                        'layout': layout,
                        'hr_paths': hr_files,
                        # The native pipeline shrinks each HR frame to the size of the
                        # real LR frame with the same index.
                        'lr_paths': lr_files
                    })
            except Exception:
                pass

    def layout_weights(self, brazilian_weight: float = 1.0) -> List[float]:
        """Per-sample sampling weights for `WeightedRandomSampler`.

        Brazilian plates are both rarer in training (36.6% of tracks) and harder to
        read (~13% lower sharpness/contrast), which shows up as a large accuracy
        gap. Up-weighting them lets the sampler even out the training stream without
        changing the number of batches per epoch.
        """
        return [
            brazilian_weight if s.get('layout') == 'Brazilian' else 1.0
            for s in self.samples
        ]

    def layout_counts(self) -> Dict[str, int]:
        """Number of indexed samples per layout, for logging."""
        counts: Dict[str, int] = {}
        for s in self.samples:
            counts[s.get('layout', 'Unknown')] = counts.get(s.get('layout', 'Unknown'), 0) + 1
        return counts

    def _index_pseudo_samples(self) -> None:
        """Add confidence-filtered test tracks to training, labelled by the model itself.

        The test set shares Scenario-B's distribution, so these tracks are exactly the
        domain the model is scored on. Measured on val, a 0.99 confidence cut keeps 63%
        of tracks at 97.5% label precision — the ~2.5% wrong labels are noise the CTC
        loss tolerates far better than the missing 4000 tracks of in-domain data.
        """
        preds = {}
        with open(self.pseudo_labels) as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line or ';' not in line:
                    continue
                track_id, rest = line.split(',', 1)
                text, conf = rest.rsplit(';', 1)
                preds[track_id] = (text, float(conf))

        added = skipped = 0
        for track_path in sorted(glob.glob(os.path.join(self.pseudo_root, "**", "track_*"),
                                           recursive=True)):
            track_id = os.path.basename(track_path)
            entry = preds.get(track_id)
            if entry is None:
                continue
            text, conf = entry
            if conf < self.pseudo_min_conf or not text:
                skipped += 1
                continue
            lr_files = sorted(glob.glob(os.path.join(track_path, "lr-*.png")) +
                              glob.glob(os.path.join(track_path, "lr-*.jpg")))
            if not lr_files:
                continue
            # Test tracks ship LR only, so there is no synthetic twin and no HR pair.
            self.samples.append({
                'paths': lr_files, 'label': text, 'is_synthetic': False,
                'track_id': track_id, 'layout': 'Unknown', 'hr_paths': []
            })
            self.layouts[track_id] = 'Unknown'
            added += 1
        print(f"🏷️  Pseudo-label: thêm {added} track (bỏ {skipped} do conf < "
              f"{self.pseudo_min_conf})")

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """Load exactly 5 frames (guaranteed by dataset structure).
        
        For training: applies degradation (if synthetic) then augmentation.
        For validation: applies degradation (if synthetic) then clean transform.
        For test: only applies clean transform, returns dummy targets.
        """
        item = self.samples[idx]
        img_paths = item['paths']
        label = item['label']
        is_synthetic = item['is_synthetic']
        track_id = item['track_id']
        
        frame_ids = list(range(len(img_paths)))
        if self.single_frame and img_paths:
            # Train: a random frame each time, so across epochs the model still sees
            # every frame and the comparison is not confounded by seeing less data.
            # Val/test: the middle frame, for a deterministic measurement.
            pick = random.randrange(len(img_paths)) if self.mode == 'train' else len(img_paths) // 2
            img_paths = [img_paths[pick]]
            frame_ids = [pick]

        images_list = []
        syn_targets = None
        replay = None
        if is_synthetic and self.syn_pipeline == 'native':
            images_list, syn_targets = self._native_synthetic(item, frame_ids)
        else:
            for p in img_paths:
                image = self._read_rgb(p)

                # Apply degradation first (if synthetic), before any transform
                if is_synthetic and self.degrade:
                    image = self.degrade(image=image)['image']

                # Apply transform (augmented for training, clean for validation/test).
                # In consistent mode the first frame draws the parameters and the
                # remaining four replay them, so all 5 frames of a track stay in the
                # same geometric frame of reference.
                if self.consistent_aug:
                    if replay is None:
                        out = self.transform(image=image)
                        replay = out['replay']
                    else:
                        out = A.ReplayCompose.replay(replay, image=image)
                    image = out['image']
                else:
                    image = self.transform(image=image)['image']
                images_list.append(image)

        if self.single_frame:
            images_tensor = images_list[0].unsqueeze(0).repeat(5, 1, 1, 1)
        else:
            images_tensor = torch.stack(images_list, dim=0)

        # HR target for supervised SR, pixel-aligned with images_tensor (native only)
        syn_tensor = None
        if self.return_syn_target and syn_targets:
            syn_tensor = (syn_targets[0].unsqueeze(0).repeat(5, 1, 1, 1) if self.single_frame
                          else torch.stack(syn_targets, dim=0))

        # Distillation pair: the same track's HR frames, run through the clean
        # val transform (no augmentation) since they only feed a frozen teacher.
        # Registered HR target for the pixel-wise SR loss. The stored affine maps the
        # LR canvas onto the HR one; applying it forward brings HR into the LR frame,
        # so the target lines up with the SR output without a differentiable warp.
        reg_target, reg_quality = None, None
        if self.registration is not None and not is_synthetic:
            frames, quals = [], []
            for i, p in enumerate(item.get('hr_paths', [])):
                entry = self.registration.get(f"{track_id}/{i}")
                im = self._read_rgb(p)
                im = cv2.resize(im, (self.img_width, self.img_height))
                if entry is not None:
                    im = cv2.warpAffine(im, entry[0], (self.img_width, self.img_height))
                # Quality is per frame: registration succeeds on some frames of a track
                # and fails on others, and a badly registered target is pure noise to
                # learn from. Masking per sample would let 4 bad frames ride along with
                # 1 good one, which is what made the first attempt look like a failure.
                quals.append(float(entry[1]) if entry is not None else 0.0)
                frames.append(self.hr_transform(image=im)['image'])
            if frames:
                reg_target = torch.stack(frames, dim=0)
                reg_quality = torch.tensor(quals, dtype=torch.float32)

        hr_tensor = None
        if self.with_hr_pair:
            hr_list = []
            for p in item.get('hr_paths', []):
                im = self._read_rgb(p)
                hr_list.append(self.hr_transform(image=im)['image'])
            hr_tensor = torch.stack(hr_list, dim=0)
        
        # Luma conversion last, on the normalised tensors: colour augmentation (HSV
        # shifts) and per-channel noise would otherwise put chroma back after a
        # read-time conversion. The luma weights sum to 1 and every channel shares the
        # same Normalize(0.5, 0.5), so luma commutes with the normalisation.
        if self.grayscale:
            images_tensor = self._to_luma(images_tensor)
            syn_tensor = self._to_luma(syn_tensor)
            reg_target = self._to_luma(reg_target)
            hr_tensor = self._to_luma(hr_tensor)

        # Handle test mode (no labels)
        if self.is_test:
            target = [0]  # Dummy target
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if len(target) == 0:
                target = [0]
            target_len = len(target)
            
        return (images_tensor, torch.tensor(target, dtype=torch.long), target_len,
                label, track_id, hr_tensor, reg_target, reg_quality, syn_tensor)

    @staticmethod
    def _read_rgb(path: str):
        """Read a frame as RGB uint8."""
        return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _to_luma(t: torch.Tensor):
        """[..., 3, H, W] -> BT.601 luma replicated to 3 channels (None passes through)."""
        if t is None:
            return None
        w = torch.tensor([0.299, 0.587, 0.114], dtype=t.dtype).view(3, 1, 1)
        return (t * w).sum(dim=-3, keepdim=True).expand_as(t).contiguous()

    def _lr_size(self, path: str) -> Tuple[int, int]:
        """(width, height) of a real LR frame, cached per worker."""
        size = self._lr_size_cache.get(path)
        if size is None:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            size = (im.shape[1], im.shape[0])
            self._lr_size_cache[path] = size
        return size

    def _native_synthetic(self, item: Dict[str, Any], frame_ids: List[int]):
        """Synthetic LR frames (and their HR targets) from the 'native' recipe.

        Per frame: HR -> canvas -> geometric aug -> shrink to the paired real LR's
        size -> camera degradation -> colour jitter (shared with the target) ->
        resize to the model input. With consistent_aug the first frame draws the
        geometric and colour parameters and the other four replay them.
        """
        canvas_h, canvas_w = self.canvas_size
        hr_paths = item['hr_paths']
        lr_paths = item.get('lr_paths') or []
        inputs, targets = [], []
        geo_replay = pho_replay = None
        for k in frame_ids:
            hr = self._read_rgb(hr_paths[k])
            shrink = hr.shape[0] * hr.shape[1] > canvas_h * canvas_w
            hr = cv2.resize(hr, (canvas_w, canvas_h),
                            interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)

            if self.consistent_aug and geo_replay is not None:
                hr = A.ReplayCompose.replay(geo_replay, image=hr)['image']
            else:
                out = self.native_geo(image=hr)
                geo_replay, hr = out['replay'], out['image']

            # Fall back to the median Scenario-B LR size if the track lacks LR frames
            lr_w, lr_h = (self._lr_size(lr_paths[min(k, len(lr_paths) - 1)])
                          if lr_paths else (47, 17))
            lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            lr = self.native_degrade(image=lr)['image']

            if self.consistent_aug and pho_replay is not None:
                out = A.ReplayCompose.replay(pho_replay, image=lr, hr=hr)
            else:
                out = self.native_pho(image=lr, hr=hr)
                pho_replay = out['replay']
            inputs.append(self.native_input(image=out['image'])['image'])
            if self.return_syn_target:
                targets.append(self.native_target(image=out['hr'])['image'])
        return inputs, (targets if self.return_syn_target else None)

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...]]:
        """Custom collate function for DataLoader."""
        (images, targets, target_lengths, labels_text, track_ids,
         hr_images, reg_targets, reg_quality, syn_targets) = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        # None unless the dataset was built with with_hr_pair (distillation).
        hr = torch.stack(hr_images, 0) if hr_images[0] is not None else None
        # A batch mixes real-LR samples (which have a registered target) with the
        # synthetic HR-degraded ones (which do not). Pad the missing entries with
        # zeros; their quality score is 0 so the loss mask drops them anyway.
        if any(t is not None for t in reg_targets):
            ref = next(t for t in reg_targets if t is not None)
            reg = torch.stack(
                [t if t is not None else torch.zeros_like(ref) for t in reg_targets], 0
            )
        else:
            reg = None
        if any(q is not None for q in reg_quality):
            ref_q = next(q for q in reg_quality if q is not None)
            qual = torch.stack(
                [q if q is not None else torch.zeros_like(ref_q) for q in reg_quality], 0
            )
        else:
            qual = None
        # Supervised-SR targets exist only for synthetic samples of the native
        # pipeline. Zero-pad the rest and say which rows are real via syn_mask.
        if any(t is not None for t in syn_targets):
            ref_s = next(t for t in syn_targets if t is not None)
            syn_hr = torch.stack(
                [t if t is not None else torch.zeros_like(ref_s) for t in syn_targets], 0
            )
            syn_mask = torch.tensor([t is not None for t in syn_targets], dtype=torch.float32)
        else:
            syn_hr, syn_mask = None, None
        return (images, targets, target_lengths, labels_text, track_ids, hr, reg, qual,
                syn_hr, syn_mask)
