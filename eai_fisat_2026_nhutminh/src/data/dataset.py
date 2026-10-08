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
    get_normalize_transforms,
    get_sr_eval_geometric_transforms,
    get_sr_geometric_transforms,
    get_sr_photometric_transforms,
    get_train_transforms,
    get_val_transforms,
)

# Kích thước ảnh LR gốc trong dataset (đo trên 300 track: ~46x19).
NATIVE_LR_HEIGHT = 19
NATIVE_LR_WIDTH = 46


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
        provide_sr_target: bool = False,
        sr_scale: int = 2,
        lr_domain_match: bool = False,
        num_frames: int = 5,
        sr_eval_mode: bool = False,
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
            provide_sr_target  : True để trả thêm ảnh HR sạch làm target giám sát SR
                                  (chỉ dùng khi train với --use-sr; val/test giữ nguyên
                                  5-tuple cũ, không đổi hành vi)
            sr_scale            : Hệ số upscale của SR target (phải khớp model.sr_scale)
            lr_domain_match     : True để degrade HR ở đúng cỡ LR gốc trước khi resize
                                  (sửa Nguyên nhân #4 — domain gap synthetic/real LR)
            num_frames          : Số frame dùng mỗi track (Ablation 3, mặc định 5 = toàn bộ)
            sr_eval_mode        : CHỈ dùng để đo PSNR/SSIM trên val, không dùng khi train.
                                  Bật lên thì val cũng sinh cặp (input, HR target) khớp
                                  pixel — mặc định val không có cặp nào vì sample synthetic
                                  chỉ được tạo ở mode train. Augment hình học/quang học bị
                                  tắt để số đo tái lập được; degradation vẫn giữ vì đó là
                                  thứ mô phỏng ảnh LR thật. Mặc định False => đường train
                                  và val hiện tại không đổi chút nào.
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        # Đo SR mà quên bật provide_sr_target thì `has_hr_target` luôn False và
        # tool sẽ báo "không có cặp nào" một cách khó hiểu — bật luôn cho chắc.
        if sr_eval_mode:
            provide_sr_target = True
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        self.provide_sr_target = provide_sr_target
        self.sr_scale = sr_scale
        self.lr_domain_match = lr_domain_match
        self.num_frames = num_frames
        self.sr_eval_mode = sr_eval_mode

        # Chọn pipeline transform theo mode
        if mode == "train":
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms(domain_match=lr_domain_match)
        else:
            self.transform = get_val_transforms(img_height, img_width)
            # Đo SR cần degradation để ảnh vào giống phân phối lúc train; val
            # thường thì không degrade gì cả.
            self.degrade = (
                get_degradation_transforms(domain_match=lr_domain_match)
                if sr_eval_mode
                else None
            )

        # Nhánh SR có giám sát dùng pipeline riêng để cặp (input, target) khớp
        # pixel: hình học chạy một lần ở cỡ target, input suy ra bằng downscale;
        # quang học chạy đồng thời trên cả hai. Xem chú thích trong transforms.py.
        self.sr_target_height = img_height * sr_scale
        self.sr_target_width = img_width * sr_scale
        if self.provide_sr_target:
            if sr_eval_mode:
                # Tất định — xem chú thích ở transforms.py. Nhánh quang học không
                # được dùng ở chế độ này (xem `_build_sr_pair`).
                self.sr_geometric = get_sr_eval_geometric_transforms(
                    self.sr_target_height, self.sr_target_width
                )
                self.sr_photometric = None
            else:
                self.sr_geometric = get_sr_geometric_transforms(
                    self.sr_target_height, self.sr_target_width
                )
                self.sr_photometric = get_sr_photometric_transforms()
            self.sr_input_norm = get_normalize_transforms(img_height, img_width)
            self.sr_target_norm = get_normalize_transforms(
                self.sr_target_height, self.sr_target_width
            )
        else:
            self.sr_geometric = None
            self.sr_photometric = None
            self.sr_input_norm = None
            self.sr_target_norm = None

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

                # Sample 1: ảnh LR thật từ camera.
                # Bỏ ở chế độ đo SR: ảnh LR thật KHÔNG phải bản downscale chính xác
                # của HR thật nên không có cặp khớp pixel để tính PSNR/SSIM.
                if not self.sr_eval_mode:
                    self.samples.append({
                        "paths": lr_files,
                        "label": label,
                        "is_synthetic": False,
                        "track_id": track_id,
                    })

                # Sample 2: synthetic LR (degrade từ HR) — khi training, hoặc khi đo SR
                if (self.mode == "train" or self.sr_eval_mode) and hr_files:
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

    def _select_frames(self, paths: List[str]) -> List[str]:
        """Chọn `num_frames` frame trải đều trong track (Ablation 3).

        Trải đều thay vì lấy N frame đầu: với N=2 ta muốn frame đầu và cuối
        (đa dạng nhất về góc/blur) chứ không phải 2 frame liên tiếp gần giống nhau.
        """
        if self.num_frames >= len(paths) or len(paths) == 0:
            return paths
        if self.num_frames == 1:
            return [paths[len(paths) // 2]]  # frame giữa thường ổn định nhất
        step = (len(paths) - 1) / (self.num_frames - 1)
        return [paths[round(i * step)] for i in range(self.num_frames)]

    def __getitem__(self, idx: int):
        """
        Load 5 frame → tensor [5, 3, H, W].

        Returns (5-tuple mặc định, không đổi hành vi cũ):
            images_tensor : [5, 3, 32, 128]
            target        : indices CTC của plate_text
            target_len    : độ dài chuỗi nhãn
            label         : plate_text gốc (string)
            track_id      : tên folder track

        Nếu `provide_sr_target=True` (chỉ bật khi train với --use-sr), trả thêm:
            hr_target     : [5, 3, H*scale, W*scale] khi sample là synthetic —
                             khớp pixel với input vì cùng một ảnh đã augment;
                             `None` với sample LR thật (không có HR tương ứng).
        """
        item = self.samples[idx]
        images_list = []
        hr_list = []
        has_hr_target = bool(self.provide_sr_target and item["is_synthetic"])
        paths = self._select_frames(item["paths"])

        for path in paths:
            raw = cv2.imread(path, cv2.IMREAD_COLOR)
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

            if has_hr_target:
                image, hr_image = self._build_sr_pair(raw)
                images_list.append(image)
                hr_list.append(hr_image)
                continue

            # Synthetic LR: degrade HR trước, rồi mới augment
            if item["is_synthetic"] and self.degrade:
                image = self.degrade(image=raw)["image"]
            else:
                image = raw
            images_list.append(self.transform(image=image)["image"])

        images_tensor = torch.stack(images_list, dim=0)
        # Sample LR thật không có ảnh HR khớp pixel nên không đóng góp SR loss.
        # Trả None thay vì một tensor zeros [5,3,64,256] (~3.9 MB/sample) chỉ để
        # bị mask đi ngay sau đó ở phía trainer.
        hr_target = torch.stack(hr_list, dim=0) if has_hr_target else None

        if self.is_test:
            if self.provide_sr_target:
                return images_tensor, torch.tensor([0]), 1, "", item["track_id"], hr_target
            return images_tensor, torch.tensor([0]), 1, "", item["track_id"]

        target = [self.char2idx[c] for c in item["label"] if c in self.char2idx]
        if not target:
            target = [0]
        if self.provide_sr_target:
            return (
                images_tensor,
                torch.tensor(target, dtype=torch.long),
                len(target),
                item["label"],
                item["track_id"],
                hr_target,
            )
        return (
            images_tensor,
            torch.tensor(target, dtype=torch.long),
            len(target),
            item["label"],
            item["track_id"],
        )

    def _build_sr_pair(self, raw) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sinh cặp (input LR, target HR) khớp pixel từ một ảnh HR gốc.

        Thứ tự quan trọng: augment hình học MỘT lần ở cỡ target, rồi mới hạ ảnh
        đã augment xuống đúng cỡ LR gốc (~46x19) để degrade. Nhờ vậy input là
        một phép downscale thuần tuý của target — hai ảnh khớp pixel tuyệt đối —
        đồng thời blur/noise/JPEG sinh ra ở cùng thang không gian với ảnh LR
        thật, thu hẹp luôn domain gap của Nguyên nhân #4.
        """
        hr_aug = self.sr_geometric(image=raw)["image"]

        if self.lr_domain_match:
            lr_source = cv2.resize(
                hr_aug, (NATIVE_LR_WIDTH, NATIVE_LR_HEIGHT), interpolation=cv2.INTER_AREA
            )
        else:
            lr_source = cv2.resize(
                hr_aug, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA
            )
        if self.degrade:
            lr_source = self.degrade(image=lr_source)["image"]

        # Cùng một phép jitter màu cho cả hai ảnh, nếu không SR phải học cách
        # hoàn tác phép chỉnh sáng — việc không liên quan gì tới khôi phục nét.
        # Ở chế độ đánh giá thì bỏ hẳn jitter (phải tất định), bỏ qua thẳng thay vì
        # gọi một Compose rỗng — không phụ thuộc vào hành vi của albumentations.
        if self.sr_eval_mode:
            jittered = {"image": lr_source, "hr": hr_aug}
        else:
            jittered = self.sr_photometric(image=lr_source, hr=hr_aug)
        return (
            self.sr_input_norm(image=jittered["image"])["image"],
            self.sr_target_norm(image=jittered["hr"])["image"],
        )

    @staticmethod
    def collate_fn(batch):
        """Gom batch cho DataLoader — CTC cần targets nối liền.

        Tự nhận diện 5-tuple (mặc định) hay 6-tuple (kèm SR target) qua
        zip(*batch), không cần biết trước dataset có bật provide_sr_target hay không.

        Với nhánh SR, chỉ những sample thật sự có ảnh HR mới được stack lại, kèm
        `hr_index` chỉ ra vị trí của chúng trong batch. Cách này thay cho việc
        nhồi tensor zeros cho mọi sample LR thật rồi mask đi — batch 64 từng phải
        chuyển thừa ~126 MB zeros mỗi vòng qua DataLoader.
        """
        elements = list(zip(*batch))
        images = torch.stack(elements[0], 0)
        targets = torch.cat(elements[1])
        target_lengths = torch.tensor(elements[2], dtype=torch.long)
        labels_text = elements[3]
        track_ids = elements[4]

        if len(elements) == 5:
            return images, targets, target_lengths, labels_text, track_ids

        positions = [i for i, hr in enumerate(elements[5]) if hr is not None]
        hr_targets = torch.stack([elements[5][i] for i in positions], 0) if positions else None
        hr_index = torch.tensor(positions, dtype=torch.long)
        return images, targets, target_lengths, labels_text, track_ids, hr_targets, hr_index
