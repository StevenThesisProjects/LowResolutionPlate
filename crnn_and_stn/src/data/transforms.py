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
    if not domain_match:
        # Chuỗi gốc, chạy ở ngay độ phân giải HR (~115x42).
        return A.Compose([
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
        ])

    # Ở thang LR gốc, mọi phép trên đều mạnh hơn hẳn vì kích thước ảnh nhỏ đi
    # ~2.5 lần trong khi kernel/khối JPEG vẫn tính bằng pixel:
    #   - Downscale(0.3) biến 46x19 thành 13x5 px, tức 2 px cho mỗi ký tự. Ảnh
    #     thành nhiễu thuần nhưng vẫn mang nhãn 7 ký tự, nên CTC bị kéo về blank.
    #     Phép này cũng thừa: A.Resize xuống cỡ LR gốc CHÍNH LÀ phần mất phân
    #     giải mà nó định mô phỏng, giữ lại là đếm hai lần.
    #   - blur kernel 5 px trên ảnh cao 19 px xoá hơn 25% chiều cao.
    #   - JPEG quality 20 trên khối 8x8 của ảnh 46 px rộng phá luôn nét chữ.
    # Tham số dưới đây chọn bằng cách đo phương sai Laplacian trên 200 track
    # Scenario-B: mục tiêu là ảnh synthetic có độ nét gần ảnh LR thật nhất có thể
    # mà KHÔNG sample nào rơi xuống dưới phân vị 1% của LR thật. Cấu hình này cho
    # median 1.96x so với LR thật và 0/200 sample bị phá; siết mạnh hơn nữa kéo
    # được về 1.47x nhưng bắt đầu làm nát sample, tức đánh đổi sai chiều.
    return A.Compose([
        A.Resize(height=native_lr_height, width=native_lr_width),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.MotionBlur(blur_limit=(3, 3), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.6),
        A.ImageCompression(quality_range=(30, 65), p=0.8),
    ])


def get_val_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Transform cho validation/test: không augmentation, chỉ resize + normalize."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Pipeline giữ căn chỉnh cho cặp (input LR, target HR) khi train SR có giám sát
# ---------------------------------------------------------------------------
#
# Pipeline train mặc định KHÔNG dùng được cho SR: `get_train_transforms` bốc
# Affine/Rotate/Perspective ngẫu nhiên lên ảnh input, trong khi target HR đi qua
# `get_val_transforms` (chỉ resize). SR do đó bị bắt map một ảnh xoay tới ±15°,
# dịch, méo phối cảnh sang một ảnh KHÔNG bị biến đổi — bài toán vô nghiệm, và
# nghiệm tối ưu duy nhất của L1 là "đoán trung bình có điều kiện", tức làm mờ.
#
# 3 hàm dưới tách phép biến đổi thành 2 nhóm để cặp ảnh luôn khớp pixel:
#   - hình học : chạy MỘT lần ở độ phân giải target, input suy ra bằng downscale
#   - quang học: chạy đồng thời trên cả 2 ảnh qua `additional_targets`
# ChannelShuffle và CoarseDropout bị loại khỏi nhánh SR: hoán vị kênh buộc SR
# học lại một phép hoán vị vô nghĩa, còn CoarseDropout đục lỗ ở toạ độ tuyệt đối
# nên rơi vào vị trí khác nhau trên 2 ảnh khác kích thước.


def get_sr_geometric_transforms(img_height: int, img_width: int) -> A.Compose:
    """Biến đổi hình học, chạy ở độ phân giải TARGET rồi mới hạ xuống input."""
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
        A.Rotate(limit=10, p=0.3),
    ])


def get_sr_photometric_transforms() -> A.Compose:
    """Biến đổi quang học, áp dụng cùng tham số lên cả input và target."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3,
            ),
        ],
        additional_targets={"hr": "image"},
        # Input và target cố ý khác kích thước (LR vs HR); mọi phép ở đây đều
        # thuần quang học nên không phụ thuộc kích thước.
        is_check_shapes=False,
    )


def get_normalize_transforms(img_height: int, img_width: int) -> A.Compose:
    """Resize + normalize + to-tensor, không augmentation."""
    return get_val_transforms(img_height, img_width)
