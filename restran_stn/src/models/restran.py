"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    ResNetFeatureExtractor,
    AttentionFusion,
    CrossFrameFusion,
    MultiFrameSuperResolutionBlock,
    PositionalEncoding,
    PositionalHead,
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
        sr_scale: int = 1,
        sr_channel_attention: bool = False,
        sr_multi_frame: bool = False,
        pretrained_backbone: bool = False,
        num_frames: int = 5,
        use_positional_head: bool = False,
        num_positions: int = 7,
        use_cross_frame_fusion: bool = False,
        backbone_norm: str = "batch",
        stn_shared: bool = False,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.stn_shared = stn_shared
        self.use_learnable_sr = use_learnable_sr
        self.use_positional_head = use_positional_head
        self.num_frames = num_frames

        if self.use_learnable_sr:
            if sr_multi_frame:
                self.sr_block = MultiFrameSuperResolutionBlock(
                    in_channels=3,
                    hidden=sr_hidden,
                    num_blocks=sr_num_blocks,
                    scale=sr_scale,
                    num_frames=num_frames,
                    channel_attention=sr_channel_attention,
                )
            else:
                self.sr_block = SuperResolutionBlock(
                    in_channels=3,
                    hidden=sr_hidden,
                    num_blocks=sr_num_blocks,
                    scale=sr_scale,
                    channel_attention=sr_channel_attention,
                )

        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        self.backbone = ResNetFeatureExtractor(
            pretrained=pretrained_backbone, norm=backbone_norm
        )
        if use_cross_frame_fusion:
            self.fusion = CrossFrameFusion(
                channels=self.cnn_channels,
                num_frames=num_frames,
                num_heads=transformer_heads,
                dropout=dropout,
            )
        else:
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

        # Optional fixed-length positional recognition head (multi-task w/ CTC).
        # num_chars = num_classes - 1 (drop the CTC blank at index 0).
        if self.use_positional_head:
            self.positional_head = PositionalHead(
                d_model=self.cnn_channels,
                num_positions=num_positions,
                num_chars=num_classes - 1,
                num_heads=transformer_heads,
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        return_sr: bool = False,
        return_pos: bool = False,
    ):
        """
        Args:
            x: [Batch, Frames, 3, H, W]
            return_sr: If True and use_learnable_sr, also return the SR-
                       enhanced frames for auxiliary loss computation.
            return_pos: If True and use_positional_head, also return the
                       fixed-length positional logits [Batch, num_positions,
                       num_chars] (raw, no softmax).
        Returns:
            logits: [Batch, Seq_Len, Num_Classes]  (log_softmax)
            sr_output: [Batch*Frames, 3, H, W] or None  (only if return_sr)
            pos_logits: [Batch, num_positions, num_chars] or None (if return_pos)

            Return arity: logits, then sr_output (if return_sr), then pos_logits
            (if return_pos), in that order. With neither flag, just logits.
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)

        sr_output = None
        if self.use_learnable_sr:
            x_flat = self.sr_block(x_flat)
            if return_sr:
                sr_output = x_flat  # For L1 loss against HR targets

        if self.use_stn:
            # Shared mode predicts one affine per track so all 5 frames stay
            # mutually aligned before fusion.
            theta = self.stn(x_flat, num_frames=f if self.stn_shared else None)
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

        pos_logits = None
        if return_pos and self.use_positional_head:
            pos_logits = self.positional_head(seq_out)  # [B, P, num_chars]

        outputs = [logits]
        if return_sr:
            outputs.append(sr_output)
        if return_pos:
            outputs.append(pos_logits)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)