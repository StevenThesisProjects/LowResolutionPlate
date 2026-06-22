"""
Các module dùng chung cho Baseline 1: CRNN + STN.

Chỉ chứa 3 thành phần cần thiết (không có ResNet/Transformer):
  - STNBlock        : căn chỉnh hình học từng frame (report Trang 36-38)
  - CNNBackbone     : trích xuất feature map (report Trang 28)
  - AttentionFusion : gộp 5 frame thành 1 feature map (report Trang 29)

Nguồn gốc: MultiFrame-LPR-main/src/models/components.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STNBlock(nn.Module):
    """
    Spatial Transformer Network — học biến đổi affine 6 tham số.

    Mỗi frame LR được warp (xoay/căn/méo) trước khi vào CNN.
    Khởi tạo identity matrix → ban đầu không thay đổi ảnh, học dần qua training.

    Paper: https://arxiv.org/pdf/1506.02025
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Localization network: CNN nhỏ → dự đoán 6 tham số affine
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)),  # ép về kích thước cố định cho FC
        )

        # Regressor: feature → ma trận affine 2x3 (6 số)
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6),
        )

        # Khởi tạo = identity transform [1,0,0; 0,1,0] → không warp lúc đầu
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, 3, H, W] — một frame đã resize
        Returns:
            theta: [Batch, 2, 3] — ma trận biến đổi affine
        """
        xs = self.localization(x)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """
    Gộp feature từ 5 frame bằng attention có trọng số.

    Score Net chấm điểm "chất lượng" từng frame → Softmax → weighted sum.
    Frame rõ hơn được trọng số cao hơn.

    Report Trang 29: Score Net → Softmax → ∑(weight × feature)
    """

    def __init__(self, channels: int):
        super().__init__()
        # Mạng 1x1 conv nhỏ để tính điểm attention cho mỗi pixel/frame
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch*5, C, H, W] — feature maps của tất cả frame (đã qua CNN)
        Returns:
            fused: [Batch, C, H, W] — một feature map đại diện
        """
        total_frames, c, h, w = x.size()
        num_frames = 5  # cố định theo dataset ICPR LRLPR
        batch_size = total_frames // num_frames

        # [B*F, C, H, W] → [B, F, C, H, W]
        x_view = x.view(batch_size, num_frames, c, h, w)

        # Tính attention score cho từng frame: [B, F, 1, H, W]
        scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        weights = F.softmax(scores, dim=1)  # chuẩn hóa theo chiều frame

        # Weighted sum: gộp 5 frame → 1
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class CNNBackbone(nn.Module):
    """
    CNN backbone cho CRNN — trích xuất feature map dạng chuỗi.

    Sau 5 block conv + pool, chiều cao feature map = 1, chiều rộng = độ dài chuỗi.
    Output: [Batch, 512, 1, W'] — W' time steps cho BiLSTM.

    Kiến trúc gốc từ paper CRNN: https://arxiv.org/pdf/1507.05717
    """

    def __init__(self, out_channels: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3→64, H/2, W/2
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 2: 64→128, H/4, W/4
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 3: 128→256, giảm H nhiều hơn W (giữ chiều ngang cho sequence)
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4: 256→512
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5: ép chiều cao về 1 (sẵn sàng cho RNN)
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
