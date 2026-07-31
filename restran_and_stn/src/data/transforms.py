"""Augmentation pipelines for training, validation, degradation, and SR preprocessing."""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def enhance_plate_image(image: np.ndarray) -> np.ndarray:
    """Enhance plate contrast and edges at the current resolution (post-resize)."""
    denoised = cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)

    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge([l_channel, a_channel, b_channel]),
        cv2.COLOR_LAB2RGB,
    )

    sharpen_kernel = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_lightweight_sr(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """Gentle contrast enhancement for blurry plates.

    Previous version upscaled then sharpened then downscaled, which amplified
    noise/artifacts on real LR images. Now only applies mild CLAHE to boost
    character-background contrast without introducing ringing.
    """
    if scale <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 4))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge([l_channel, a_channel, b_channel]),
        cv2.COLOR_LAB2RGB,
    )
    return enhanced


def resize_plate_image(
    image: np.ndarray,
    img_height: int = 32,
    img_width: int = 128,
) -> np.ndarray:
    """Resize plate crop to model input size."""
    return cv2.resize(
        image,
        (img_width, img_height),
        interpolation=cv2.INTER_CUBIC,
    )


def get_train_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Training augmentation pipeline with geometric, color, and occlusion transforms."""
    return A.Compose([
        A.Affine(
            scale=(0.93, 1.07),
            translate_percent=(0.06, 0.06),
            rotate=(-6, 6),
            fill=128,
            p=0.5
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        # Thêm GridDistortion để làm biến dạng nhẹ ảnh, giúp mô hình bớt overfit
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
        A.ChannelShuffle(p=0.15),
        # Tăng mạnh CoarseDropout (Cutout)
        A.CoarseDropout(
            num_holes_range=(2, 6),       # Tăng số lỗ
            hole_height_range=(4, 12),    # Tăng kích thước lỗ
            hole_width_range=(4, 16),
            fill=0,
            p=0.5,                        # Tăng xác suất xảy ra
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def get_light_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Light training pipeline: normalize only (resize handled in dataset)."""
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_degradation_transforms(
    strength: float = 1.0,
    from_lr: bool = False,
) -> A.Compose:
    """Pipeline to synthesize LR-looking plates from HR or sharpened LR crops.

    Args:
        strength: 0..1 curriculum factor; lower = milder blur/noise early in training.
        from_lr: When True, apply a stronger stack because the source is already LR.
    """
    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return A.Compose([])

    blur_low, blur_high = (3, 5) if not from_lr else (3, 5)
    blur_p = 0.25 + 0.30 * strength
    noise_p = 0.25 + 0.30 * strength
    jpeg_p = 0.15 + 0.25 * strength
    down_p = 0.10 + 0.25 * strength

    quality_low = max(35, int(60 - 25 * strength))
    quality_high = max(quality_low + 5, int(80 - 15 * strength))
    down_low = max(0.55, 0.80 - 0.25 * strength)
    down_high = max(down_low + 0.05, 0.95 - 0.05 * strength)

    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(blur_low, blur_high), p=1.0),
            A.MotionBlur(blur_limit=(blur_low, blur_high), p=1.0),
        ], p=blur_p),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05 * strength + 0.01), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.92, 1.08), p=1.0),
        ], p=noise_p),
        A.ImageCompression(quality_range=(quality_low, quality_high), p=jpeg_p),
        A.Downscale(scale_range=(down_low, down_high), p=down_p),
        A.RandomBrightnessContrast(brightness_limit=0.15 * strength, contrast_limit=0.15 * strength, p=0.4 * strength),
    ])


def get_val_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Validation transform pipeline (normalize only; resize handled in dataset)."""
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
