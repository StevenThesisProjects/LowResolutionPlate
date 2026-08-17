"""Augmentation pipelines for training, validation, and degradation."""
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class LetterboxResize(A.ImageOnlyTransform):
    """Resize preserving aspect ratio, then pad to the target canvas.

    `A.Resize` stretches to a fixed canvas, which applies a *different* geometric
    distortion per scenario: measured on this dataset, Scenario-A crops (aspect 2.14)
    get stretched 1.87x horizontally while Scenario-B and the test set (aspect 2.74)
    get 1.46x. Half the training tracks therefore reach the model under a distortion
    the test set never shows. Letterboxing makes the geometry identical everywhere.

    Pad value 128 matches the fill already used by `A.Affine`, so padded borders look
    like the borders augmentation already produces.
    """

    def __init__(self, height: int, width: int, pad_value: int = 128, p: float = 1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.pad_value = pad_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(self.height / h, self.width / w)
        new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (self.height - new_h) // 2
        left = (self.width - new_w) // 2
        return cv2.copyMakeBorder(
            img, top, self.height - new_h - top, left, self.width - new_w - left,
            cv2.BORDER_CONSTANT, value=(self.pad_value,) * 3,
        )

    def get_transform_init_args_names(self):
        return ("height", "width", "pad_value")


def _resize(img_height: int, img_width: int, letterbox: bool):
    return (LetterboxResize(img_height, img_width) if letterbox
            else A.Resize(height=img_height, width=img_width))


def get_train_transforms(
    img_height: int = 32,
    img_width: int = 128,
    channel_shuffle: bool = True,
    consistent: bool = False,
    letterbox: bool = False,
):
    """Training augmentation pipeline with geometric and color transforms.

    Args:
        channel_shuffle: Keep `ChannelShuffle`. Plate background colour separates
            Mercosur from Brazilian, so shuffling channels destroys a real signal.
        consistent: Build a `ReplayCompose` so the caller can apply *identical*
            parameters to all 5 frames of a track. Independent per-frame geometry
            breaks the spatial correspondence that STN + AttentionFusion exploit,
            and val/test see no augmentation at all.
    """
    steps = [
        _resize(img_height, img_width, letterbox),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.05, 0.05),
            rotate=(-5, 5),
            fill=128,
            p=0.5
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.Rotate(limit=10, p=0.3),
    ]
    if channel_shuffle:
        steps.append(A.ChannelShuffle(p=0.3))
    steps += [
        A.CoarseDropout(
            num_holes_range=(2, 5),
            hole_height_range=(4, 8),
            hole_width_range=(4, 8),
            p=0.3
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]
    return A.ReplayCompose(steps) if consistent else A.Compose(steps)


def get_light_transforms(img_height: int = 32, img_width: int = 128,
                         letterbox: bool = False) -> A.Compose:
    """Light training pipeline: resize + normalize only."""
    return A.Compose([
        _resize(img_height, img_width, letterbox),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_degradation_transforms() -> A.Compose:
    """Pipeline to convert HR images to synthetic LR."""
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0)
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
        ], p=0.7),
        A.ImageCompression(quality_range=(20, 50), p=0.5),
        A.Downscale(scale_range=(0.3, 0.5), p=0.5),
    ])


def get_val_transforms(img_height: int = 32, img_width: int = 128,
                       letterbox: bool = False) -> A.Compose:
    """Validation transform pipeline (resize + normalize only)."""
    return A.Compose([
        _resize(img_height, img_width, letterbox),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
