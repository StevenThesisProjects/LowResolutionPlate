"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import ResNetFeatureExtractor, AttentionFusion, PositionalEncoding, STNBlock
from src.models.sr import StackedSRNet
from src.models.edvr import EDVRLite

class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using optional STN, ResNet34 and Transformer.
    Pipeline: Input (5 frames) -> [Optional STN] -> ResNet34 -> Attention Fusion -> Transformer -> CTC Head
    """
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        pretrained: bool = False,
        use_sr: bool = False,
        sr_features: int = 32,
        sr_blocks: int = 8,
        sr_scale: int = 2,
        sr_arch: str = 'stacked',
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.use_sr = use_sr

        # Optional multi-frame SR front-end: [B,5,3,H,W] -> [B,5,3,2H,2W].
        # Replaces the fixed bilinear upscale in the data pipeline with a learned,
        # cross-frame one — the only place the model can combine frames at pixel level.
        if not use_sr:
            self.sr = None
        elif sr_arch == 'edvr':
            # Aligns frames with pyramidal deformable convs before fusing them
            self.sr = EDVRLite(
                num_features=sr_features, num_blocks=sr_blocks, scale=sr_scale
            )
        elif sr_arch == 'stacked':
            self.sr = StackedSRNet(
                num_features=sr_features, num_blocks=sr_blocks, scale=sr_scale
            )
        else:
            raise ValueError(f"sr_arch must be 'stacked' or 'edvr', got {sr_arch!r}")
        
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone: ResNet34
        self.backbone = ResNetFeatureExtractor(pretrained=pretrained)
        
        # 3. Attention Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
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
        
        # 5. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False,
                return_sr: bool = False):
        """
        Args:
            x: [Batch, Frames, 3, H, W]
            return_features: also return the post-transformer sequence
                [Batch, Seq_Len, 512], used as the distillation target.
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes], optionally with features.
        """
        sr_out = None
        if self.sr is not None:
            x = self.sr(x)
            sr_out = x

        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)  # [B*F, C, H, W]
        
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
        
        out = self.head(seq_out)              # [B, W', Num_Classes]
        logits = out.log_softmax(2)
        if return_features and return_sr:
            return logits, seq_out, sr_out
        if return_sr:
            return logits, sr_out
        return (logits, seq_out) if return_features else logits