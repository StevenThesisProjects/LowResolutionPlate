"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    ResNetFeatureExtractor,
    AttentionFusion,
    PositionalEncoding,
    STNBlock,
    SuperResolutionBlock,
)

class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using optional supervised SR, STN, ResNet34
    and Transformer.

    Pipeline: Input (5 frames) -> [Optional SR] -> [Optional STN] -> ResNet34
              -> Attention Fusion -> Transformer -> CTC Head

    When ``use_learnable_sr=True`` and ``return_sr=True``, ``forward``
    returns ``(logits, sr_output)`` so the trainer can compute an auxiliary
    L1 loss between ``sr_output`` and HR targets.
    """
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_stn: bool = True,
        use_learnable_sr: bool = False,
        sr_hidden: int = 64,
        sr_num_blocks: int = 4,
        pretrained_backbone: bool = False,
        num_frames: int = 5,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.use_learnable_sr = use_learnable_sr
        self.num_frames = num_frames

        if self.use_learnable_sr:
            self.sr_block = SuperResolutionBlock(
                in_channels=3,
                hidden=sr_hidden,
                num_blocks=sr_num_blocks,
            )

        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        self.backbone = ResNetFeatureExtractor(pretrained=pretrained_backbone)
        self.fusion = AttentionFusion(channels=self.cnn_channels, num_frames=num_frames)
        
        # 4. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_sr: bool = False,
    ):
        """
        Args:
            x: [Batch, Frames, 3, H, W]
            return_sr: If True and use_learnable_sr, also return the SR-
                       enhanced frames for auxiliary loss computation.
        Returns:
            logits: [Batch, Seq_Len, Num_Classes]  (log_softmax)
            sr_output: [Batch*Frames, 3, H, W] or None
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)

        sr_output = None
        if self.use_learnable_sr:
            x_flat = self.sr_block(x_flat)
            if return_sr:
                sr_output = x_flat  # For L1 loss against HR targets

        if self.use_stn:
            theta = self.stn(x_flat)  # [B*F, 2, 3]
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        
        features = self.backbone(x_aligned)  # [B*F, 512, 1, W']
        fused = self.fusion(features)       # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        logits = self.head(self.head_dropout(seq_out))
        logits = logits.log_softmax(2)

        if return_sr:
            return logits, sr_output
        return logits