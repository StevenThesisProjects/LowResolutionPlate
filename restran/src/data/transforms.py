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


def _geo_kwargs(clean: bool) -> dict:
    """Interpolation/border settings for the geometric augmentations.

    Default (clean=False) keeps albumentations' behaviour: bilinear resampling and a
    constant fill — grey (128) for Affine but black (0) for Rotate and Perspective.
    Measured on real LR frames already upsampled to 32x128, one bilinear warp cuts
    the interior Laplacian variance to 0.76x (Affine), 0.61x (Perspective) and 0.88x
    (Rotate), while 7-15% of the pixels become flat fill with a hard artificial edge
    — val/test frames have neither. clean=True resamples bicubically and replicates
    the border instead of painting it.
    """
    if not clean:
        return {}
    return {'interpolation': cv2.INTER_CUBIC, 'border_mode': cv2.BORDER_REPLICATE}


def get_train_transforms(
    img_height: int = 32,
    img_width: int = 128,
    channel_shuffle: bool = True,
    consistent: bool = False,
    letterbox: bool = False,
    aug_degrade: str = "none",
    relative_dropout: bool = False,
    clean_geo: bool = False,
):
    """Training augmentation pipeline with geometric and color transforms.

    Args:
        channel_shuffle: Keep `ChannelShuffle`. Plate background colour separates
            Mercosur from Brazilian, so shuffling channels destroys a real signal.
        consistent: Build a `ReplayCompose` so the caller can apply *identical*
            parameters to all 5 frames of a track. Independent per-frame geometry
            breaks the spatial correspondence that STN + AttentionFusion exploit,
            and val/test see no augmentation at all.
        relative_dropout: Size CoarseDropout holes as 1/8-1/4 of the input height
            instead of a fixed 4-8 px. Identical at 32x128; at the 16x64 input of the
            SR runs the fixed holes cover ~4x more of the plate (~12% vs ~3% of the
            area when applied), which handicaps every SR run against the baseline.
    """
    hole = ((max(1, img_height // 8), max(2, img_height // 4)) if relative_dropout
            else (4, 8))
    geo = _geo_kwargs(clean_geo)
    steps = [
        _resize(img_height, img_width, letterbox),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.05, 0.05),
            rotate=(-5, 5),
            fill=128,
            p=0.5,
            **geo
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3, **geo),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.Rotate(limit=10, p=0.3, **geo),
    ]
    if channel_shuffle:
        steps.append(A.ChannelShuffle(p=0.3))
    # Capture-degradation augmentation. The real-LR half of training otherwise only
    # ever sees geometric and colour jitter, while the test set spans Laplacian
    # variance 65 to 15880 and its blurriest quartile produces 50% of all errors.
    # Kept mild and probabilistic: the aggressive variant (JPEG q=10-40 at p=1.0 on
    # the synthetic branch) cost 4.25 points and destabilised training.
    if aug_degrade != "none":
        strong = aug_degrade == "medium"
        steps += [
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.4 if strong else 0.25),
            # std_range must be set explicitly: the albumentations default perturbs
            # pixels by ~52/255, which on a 32x128 plate with ~18px glyphs destroys
            # the characters rather than simulating sensor noise.
            A.GaussNoise(std_range=(0.02, 0.06), p=0.3 if strong else 0.2),
            A.ImageCompression(
                quality_range=(25, 70) if strong else (40, 80),
                p=0.4 if strong else 0.25),
        ]
    steps += [
        A.CoarseDropout(
            num_holes_range=(2, 5),
            hole_height_range=hole,
            hole_width_range=hole,
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


def get_degradation_transforms(strength: str = "normal") -> A.Compose:
    """Pipeline to convert HR images to synthetic LR.

    Slide 45 shows the synthetic copy keeps sharp edges the real LR has lost, and the
    synthetic half is 50% of every training epoch — so the model spends half its time
    on an easier distribution than the one it is scored on. `strength="strong"` pushes
    JPEG quality down to Scenario-B's territory and always applies compression, since
    every Scenario-B and test image is a JPEG.
    """
    if strength == "calibrated":
        # Same shape as "normal" but with a sane noise level. The default GaussNoise
        # used below shifts pixels by ~52/255 on average (measured), i.e. it is closer
        # to corruption than to degradation.
        return A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0)
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
            ], p=0.7),
            A.ImageCompression(quality_range=(20, 50), p=0.5),
            A.Downscale(scale_range=(0.3, 0.5), p=0.5),
        ])
    if strength == "strong":
        return A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0)
            ], p=0.9),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=1.0)
            ], p=0.8),
            A.ImageCompression(quality_range=(10, 40), p=1.0),
            A.Downscale(scale_range=(0.25, 0.45), p=0.7),
        ])
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


# ---------------------------------------------------------------------------
# "native" synthetic pipeline (--syn-pipeline native)
# ---------------------------------------------------------------------------
# Measured with a trained model on 600 val tracks: real LR reads at 78.3%, but the
# same tracks' HR frames pushed through the legacy synthetic recipe read at 98.3%
# (and still 98.0% after merely downscaling HR to the real LR's pixel size). Half of
# every epoch is therefore spent on plates that are far easier than the ones scored.
# The native pipeline degrades at the real LR's own size (Scenario-B ~47x17), where
# a 3px kernel and an 8x8 JPEG block are as destructive as they are in the camera.
#
# The order also makes an exact (input, HR target) pair for supervised SR: geometry
# is applied once, to the HR at the target canvas, and the input is *derived* from
# that augmented image by downscaling — so the two are pixel-aligned by construction.
# Photometric jitter is replayed on both so SR never has to undo a colour shift.
# CoarseDropout and ChannelShuffle are left out: holes at absolute coordinates land
# in different places at two resolutions, and a channel permutation is not something
# an SR target should contain.


def get_native_geometric_transforms(clean_geo: bool = False) -> A.ReplayCompose:
    """Geometric jitter (same ranges as the train pipeline), run on the HR canvas."""
    geo = _geo_kwargs(clean_geo)
    return A.ReplayCompose([
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.05, 0.05),
            rotate=(-5, 5),
            fill=128,
            p=0.5,
            **geo
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3, **geo),
        A.Rotate(limit=10, p=0.3, **geo),
    ])


def get_native_photometric_transforms() -> A.ReplayCompose:
    """Colour jitter applied with identical parameters to the input and its HR target."""
    return A.ReplayCompose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
        ],
        additional_targets={'hr': 'image'},
        # Input and target are deliberately different sizes; every op is per-pixel.
        is_check_shapes=False,
    )


def get_native_degradation_transforms() -> A.Compose:
    """Camera-like degradation applied at the real LR frame's own pixel size.

    On the same 600 val tracks this recipe brings the model's accuracy on synthetic
    input from 98.3% (legacy) to 93.2% and the median Laplacian variance from 2392
    to 266 (real LR: 340), i.e. the synthetic half finally overlaps the real one.
    Noise keeps an explicit std_range: albumentations' default is ~52/255.
    """
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.MotionBlur(blur_limit=(3, 3), p=1.0),
        ], p=0.7),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.5),
        A.ImageCompression(quality_range=(30, 65), p=0.8),
    ])


def get_val_transforms(img_height: int = 32, img_width: int = 128,
                       letterbox: bool = False) -> A.Compose:
    """Validation transform pipeline (resize + normalize only)."""
    return A.Compose([
        _resize(img_height, img_width, letterbox),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
