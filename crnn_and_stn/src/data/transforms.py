"""Pipeline augmentation và degradation cho dataset LRLPR."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """
    Augmentation khi training (report Trang 46).

    Bao gồm: Affine, Perspective, Brightness/Contrast, HSV, Rotate, CoarseDropout.
    """
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.05, 0.05),
            rotate=(-5, 5),
            fill=128,
            p=0.5,
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3,
        ),
        A.Rotate(limit=10, p=0.3),
        A.ChannelShuffle(p=0.3),
        A.CoarseDropout(
            num_holes_range=(2, 5),
            hole_height_range=(4, 8),
            hole_width_range=(4, 8),
            p=0.3,
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_light_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Augmentation nhẹ: chỉ resize + normalize (dùng khi debug nhanh)."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_degradation_transforms(
    domain_match: bool = False,
    native_lr_height: int = 19,
    native_lr_width: int = 46,
) -> A.Compose:
    """
    Tạo Synthetic LR từ ảnh HR (report Trang 44).

    Mô phỏng: blur, noise, nén JPEG, downscale — giống điều kiện camera thật.
    Mỗi track train tạo thêm 1 sample synthetic → tăng gấp đôi dữ liệu.

    Args:
        domain_match: sửa Nguyên nhân #4 của issue #9. Mặc định (False) degrade
            ở ngay độ phân giải HR (~96x240) rồi mới resize xuống 32x128 — blur
            bị co lại, noise bị trung bình hoá, nên đặc tính nhiễu KHÁC hẳn ảnh
            LR thật (~19x46 rồi phóng LÊN 32x128). Bật True sẽ hạ HR về đúng cỡ
            LR gốc TRƯỚC khi degrade, để nhiễu/blur sinh ra ở cùng thang không
            gian với ảnh LR thật, thu hẹp domain gap train/validation.
        native_lr_height/width: cỡ ảnh LR gốc trong dataset (theo config ~46x19).
    """
    degrade = [
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.7),
        A.ImageCompression(quality_range=(20, 50), p=0.5),
        A.Downscale(scale_range=(0.3, 0.5), p=0.5),
    ]
    if domain_match:
        degrade.insert(0, A.Resize(height=native_lr_height, width=native_lr_width))
    return A.Compose(degrade)


def get_val_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Transform cho validation/test: không augmentation, chỉ resize + normalize."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
