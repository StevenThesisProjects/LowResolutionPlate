"""
Dataset loader cho Multi-Frame LPR.

Mỗi sample = 1 track gồm 5 frame ảnh biển số + nhãn plate_text.

Đặc điểm quan trọng:
  - Train mode: mỗi track tạo 2 sample (real LR + synthetic LR từ HR)
  - Validation: chỉ lấy track từ Scenario-B (tập khó hơn)
  - Test mode: không có label, chỉ load LR frames

Nguồn gốc: MultiFrame-LPR-main/src/data/dataset.py
"""
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
    get_degradation_transforms,
    get_light_transforms,
    get_train_transforms,
    get_val_transforms,
)


class MultiFrameDataset(Dataset):
    """Dataset đa frame cho CRNN+STN baseline."""

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "dataset/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
    ):
        """
        Args:
            root_dir           : Thư mục chứa track_* (vd: dataset/data/train)
            mode               : 'train' hoặc 'val'
            split_ratio        : Tỷ lệ train trong Scenario-B (0.9 = 90% train, 10% val)
            img_height/width   : Kích thước resize trước khi vào model
            char2idx           : Mapping ký tự → index CTC
            val_split_file     : File JSON lưu track_id validation
            seed               : Random seed cho split reproducible
            augmentation_level : 'full' hoặc 'light'
            is_test            : True khi load test set (không có label)
            full_train         : True khi train toàn bộ data (submission mode)
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

        # Chọn pipeline transform theo mode
        if mode == "train":
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        print(f"[{mode.upper()}] Quét dữ liệu: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        all_tracks = sorted(glob.glob(os.path.join(abs_root, "**", "track_*"), recursive=True))

        if not all_tracks:
            print("❌ Không tìm thấy track nào.")
            return

        if is_test:
            print(f"[TEST] {len(all_tracks)} tracks.")
            self._index_test_samples(all_tracks)
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            selected = train_tracks if mode == "train" else val_tracks
            print(f"[{mode.upper()}] {len(selected)} tracks.")
            self._index_samples(selected)

        print(f"→ Tổng samples: {len(self.samples)}")

    def _load_or_create_split(
        self, all_tracks: List[str], split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """
        Chia train/val — validation CHỈ lấy từ Scenario-B.

        Lý do: Scenario-A (PNG) dễ hơn Scenario-B (JPG nén).
        Validate trên B phản ánh đúng performance thực tế hơn.
        """
        if self.full_train:
            print("📌 FULL TRAIN: dùng toàn bộ tracks, không chia val.")
            return all_tracks, []

        train_tracks, val_tracks = [], []

        # Thử load split đã lưu
        if os.path.exists(self.val_split_file):
            print(f"📂 Load split từ '{self.val_split_file}'")
            try:
                with open(self.val_split_file, "r") as f:
                    val_ids = set(json.load(f))
                for t in all_tracks:
                    (val_tracks if os.path.basename(t) in val_ids else train_tracks).append(t)
            except Exception:
                val_ids = set()

            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split không hợp lệ, tạo lại...")
                val_tracks = []

        # Tạo split mới nếu cần
        if not val_tracks:
            print("⚠️ Tạo split mới (val chỉ từ Scenario-B)...")
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
            if not scenario_b_tracks:
                print("⚠️ Không có Scenario-B, dùng random từ toàn bộ.")
                scenario_b_tracks = all_tracks

            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]

            os.makedirs(os.path.dirname(self.val_split_file) or ".", exist_ok=True)
            with open(self.val_split_file, "w") as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)
            print(f"   Val: {len(val_tracks)} tracks | Train: {len(train_tracks)} tracks")

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Tạo danh sách samples từ các track có label."""
        for track_path in tqdm(tracks, desc=f"Index {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]

                # Lấy nhãn và loại bỏ khoảng trắng thừa (1 outlier trong dataset)
                label = data.get("plate_text", data.get("license_plate", data.get("text", "")))
                label = label.strip()
                if not label:
                    continue

                track_id = os.path.basename(track_path)
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png"))
                    + glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png"))
                    + glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )

                # Sample 1: ảnh LR thật từ camera
                self.samples.append({
                    "paths": lr_files,
                    "label": label,
                    "is_synthetic": False,
                    "track_id": track_id,
                })

                # Sample 2: synthetic LR (degrade từ HR) — chỉ khi training
                if self.mode == "train" and hr_files:
                    self.samples.append({
                        "paths": hr_files,
                        "label": label,
                        "is_synthetic": True,
                        "track_id": track_id,
                    })
            except Exception:
                pass

    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test tracks — không có annotations.json."""
        for track_path in tqdm(tracks, desc="Index test"):
            track_id = os.path.basename(track_path)
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png"))
                + glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            if lr_files:
                self.samples.append({
                    "paths": lr_files,
                    "label": "",
                    "is_synthetic": False,
                    "track_id": track_id,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """
        Load 5 frame → tensor [5, 3, H, W].

        Returns:
            images_tensor : [5, 3, 32, 128]
            target        : indices CTC của plate_text
            target_len    : độ dài chuỗi nhãn
            label         : plate_text gốc (string)
            track_id      : tên folder track
        """
        item = self.samples[idx]
        images_list = []

        for path in item["paths"]:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Synthetic LR: degrade HR trước, rồi mới augment
            if item["is_synthetic"] and self.degrade:
                image = self.degrade(image=image)["image"]

            image = self.transform(image=image)["image"]
            images_list.append(image)

        images_tensor = torch.stack(images_list, dim=0)

        if self.is_test:
            return images_tensor, torch.tensor([0]), 1, "", item["track_id"]

        target = [self.char2idx[c] for c in item["label"] if c in self.char2idx]
        if not target:
            target = [0]
        return (
            images_tensor,
            torch.tensor(target, dtype=torch.long),
            len(target),
            item["label"],
            item["track_id"],
        )

    @staticmethod
    def collate_fn(batch):
        """Gom batch cho DataLoader — CTC cần targets nối liền."""
        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        return (
            torch.stack(images, 0),
            torch.cat(targets),
            torch.tensor(target_lengths, dtype=torch.long),
            labels_text,
            track_ids,
        )
