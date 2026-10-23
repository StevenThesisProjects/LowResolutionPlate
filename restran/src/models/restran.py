"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (ResNetFeatureExtractor, AttentionFusion, PositionalEncoding,
                                   STNBlock, TemporalFusion)
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
        stn_shared: bool = False,
        fusion_2d: bool = False,
        sharpness_prior: bool = False,
        fusion: str = 'attention',
        fusion_layers: int = 2,
        frame_dropout: float = 0.0,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.use_sr = use_sr
        # One affine per track instead of one per frame: the 5 frames arrive already
        # aligned to NCC 0.842 by the per-frame crop, so 5 independent warps can pull
        # them apart again right before the fusion that relies on that alignment.
        self.stn_shared = stn_shared
        self.fusion_2d = fusion_2d
        self.sharpness_prior = sharpness_prior
        # Training-only: replace a frame's features by the mean of the track's other
        # frames, so fusion cannot lean on one frame (CRNN+STN repo uses 0.05)
        self.frame_dropout = frame_dropout

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
        self.backbone = ResNetFeatureExtractor(pretrained=pretrained,
                                              collapse_height=not fusion_2d)
        
        # 3. Multi-frame fusion: per-frame quality attention (original), or a
        # cross-frame Transformer where frames can compare notes before pooling
        if fusion not in ('attention', 'temporal'):
            raise ValueError(f"fusion must be 'attention' or 'temporal', got {fusion!r}")
        if fusion == 'temporal' and sharpness_prior:
            raise ValueError("sharpness_prior only applies to fusion='attention'")
        self.temporal_fusion = fusion == 'temporal'
        if self.temporal_fusion:
            self.fusion = TemporalFusion(channels=self.cnn_channels, num_layers=fusion_layers,
                                         heads=transformer_heads, dropout=dropout)
        else:
            self.fusion = AttentionFusion(channels=self.cnn_channels,
                                          sharpness_prior=sharpness_prior)
        
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

        self.register_buffer('_lap_kernel', torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3), persistent=False)

    def _drop_frames(self, features: torch.Tensor, b: int, f: int) -> torch.Tensor:
        """Swap each frame's features, with prob frame_dropout, for the mean of the kept ones.

        At least one frame per track is always kept, so the substitute is never empty.
        """
        feats = features.view(b, f, *features.shape[1:])
        drop = torch.rand(b, f, device=feats.device) < self.frame_dropout
        drop = drop & ~drop.all(dim=1, keepdim=True)
        keep = (~drop).to(feats.dtype).view(b, f, *([1] * (feats.dim() - 2)))
        mean_kept = (feats * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True)
        return (feats * keep + mean_kept * (1 - keep)).reshape_as(features)

    def _sequence_head(self, feats: torch.Tensor):
        """[N, 512, H', W'] feature map -> (log-probs [N, W', C], sequence [N, W', 512])."""
        # Collapse whatever height survived, after fusion rather than before it.
        seq = F.adaptive_avg_pool2d(feats, (1, None)).squeeze(2).permute(0, 2, 1)
        seq = self.transformer(self.pos_encoder(seq))
        return self.head(seq).log_softmax(2), seq

    def forward(self, x: torch.Tensor, return_features: bool = False,
                return_sr: bool = False, return_frame_logits: bool = False):
        """
        Args:
            x: [Batch, Frames, 3, H, W]
            return_features: also return the post-transformer sequence
                [Batch, Seq_Len, 512], used as the distillation target.
            return_sr: also return the SR front-end output [Batch, Frames, 3, H', W'].
            return_frame_logits: also return per-frame log-probs
                [Batch * Frames, Seq_Len, Num_Classes] — each frame read on its own by
                the same Transformer + head, for the auxiliary per-frame CTC loss.
        Returns:
            Logits [Batch, Seq_Len, Num_Classes] alone, or a tuple
            (logits, [features], [sr], [frame_logits]) holding the requested extras
            in that order.
        """
        sr_out = None
        if self.sr is not None:
            x = self.sr(x)
            sr_out = x

        b, f, c, h, w = x.size()

        # Per-frame Laplacian variance, measured on the input before any warping so it
        # reflects the capture itself rather than the STN's resampling.
        log_sharp = None
        if self.sharpness_prior:
            gray = x.mean(2, keepdim=True).reshape(b * f, 1, h, w)
            lap = F.conv2d(gray, self._lap_kernel.to(gray.dtype), padding=1)
            log_sharp = torch.log(lap.reshape(b, f, -1).var(-1) + 1e-6)

        x_flat = x.view(b * f, c, h, w)  # [B*F, C, H, W]
        
        if self.use_stn:
            theta = self.stn(x_flat)  # [B*F, 2, 3]
            if self.stn_shared:
                theta = theta.view(b, f, 2, 3).mean(1, keepdim=True)
                theta = theta.expand(b, f, 2, 3).reshape(b * f, 2, 3)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        
        features = self.backbone(x_aligned)  # [B*F, 512, 1, W']
        # Frame dropout touches only what fusion sees; the per-frame aux path below
        # still reads every frame's own features.
        fusion_in = (self._drop_frames(features, b, f)
                     if self.training and self.frame_dropout > 0 else features)
        if self.temporal_fusion:
            fused = self.fusion(fusion_in, num_frames=f)   # [B, 512, 1, W']
        else:
            fused = self.fusion(fusion_in, log_sharp)      # [B, 512, H', W']

        # [B, C, H', W'] -> pool height -> [B, W', C] -> PE -> Transformer -> head
        logits, seq_out = self._sequence_head(fused)

        outputs = [logits]
        if return_features:
            outputs.append(seq_out)
        if return_sr:
            outputs.append(sr_out)
        if return_frame_logits:
            outputs.append(self._sequence_head(features)[0])
        return tuple(outputs) if len(outputs) > 1 else logits