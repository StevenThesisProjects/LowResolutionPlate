"""Alignment-aware multi-frame SR front-end (EDVR-style: PCD align + TSA fusion).

Why this instead of the channel-stacking design in `sr.py`:

The slide's SR builds on EDSR / RCAN / ESPCN, all of which are *single-image* SR.
Its only multi-frame element is concatenating the 5 frames on the channel axis, which
implicitly assumes they are spatially aligned. Measured on this dataset they are not —
plate width grows +2.3 to +3.8 px across the 5 frames in 84-98% of tracks — and a 3x3
convolution cannot align features several pixels apart. Runs #19-#21 measured that
design as neutral-to-harmful.

This module aligns the neighbouring frames to the middle one with pyramidal deformable
convolutions before fusing them, which is the mechanism the earlier design lacked.

Output contract: the fused frame is replicated across the 5-frame axis, so
`AttentionFusion` downstream becomes a no-op (softmax over identical features). That is
deliberate — it makes the comparison "explicit pixel-level aligned fusion" versus
"implicit feature-level fusion" clean, with run #18 (single raw frame, 56.53%) and
run #9 (feature fusion, 77.19%) as the two reference points.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from src.models.sr import ResBlock, Upsampler


class DCNv2Pack(nn.Module):
    """Deformable conv whose offsets and modulation mask are predicted from a feature."""

    def __init__(self, num_features: int):
        super().__init__()
        # 3x3 kernel -> 18 offset channels (dx, dy per tap) + 9 modulation channels
        self.conv_offset_mask = nn.Conv2d(num_features, 27, 3, padding=1)
        self.dcn = DeformConv2d(num_features, num_features, 3, padding=1)
        # Start from a plain convolution: zero offsets, mask sigmoid(0) = 0.5
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x: torch.Tensor, offset_feat: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset_mask(offset_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        return self.dcn(x, offset, torch.sigmoid(mask))


class PCDAlign(nn.Module):
    """Pyramid, Cascading and Deformable alignment of one frame onto the reference.

    Offsets are estimated coarse-to-fine: a large displacement is resolved at 1/4
    resolution and refined upward, which is what lets it handle motion of several
    pixels — the regime this dataset actually has.
    """

    def __init__(self, nf: int = 32):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Level 3 (coarsest, 1/4)
        self.off_l3_1 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.off_l3_2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.dcn_l3 = DCNv2Pack(nf)
        # Level 2 (1/2)
        self.off_l2_1 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.off_l2_2 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.dcn_l2 = DCNv2Pack(nf)
        self.feat_l2 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        # Level 1 (full)
        self.off_l1_1 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.off_l1_2 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.dcn_l1 = DCNv2Pack(nf)
        self.feat_l1 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        # Cascading refinement
        self.off_casc_1 = nn.Conv2d(nf * 2, nf, 3, padding=1)
        self.off_casc_2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.dcn_casc = DCNv2Pack(nf)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, nbr, ref):
        """nbr/ref: lists [L1, L2, L3] of [B, nf, H, W] features, L3 coarsest."""
        n1, n2, n3 = nbr
        r1, r2, r3 = ref

        off3 = self.lrelu(self.off_l3_1(torch.cat([n3, r3], 1)))
        off3 = self.lrelu(self.off_l3_2(off3))
        feat3 = self.lrelu(self.dcn_l3(n3, off3))

        off2 = self.lrelu(self.off_l2_1(torch.cat([n2, r2], 1)))
        off2 = self.lrelu(self.off_l2_2(torch.cat([off2, self._up(off3, off2)], 1)))
        feat2 = self.dcn_l2(n2, off2)
        feat2 = self.lrelu(self.feat_l2(torch.cat([feat2, self._up(feat3, feat2)], 1)))

        off1 = self.lrelu(self.off_l1_1(torch.cat([n1, r1], 1)))
        off1 = self.lrelu(self.off_l1_2(torch.cat([off1, self._up(off2, off1)], 1)))
        feat1 = self.dcn_l1(n1, off1)
        feat1 = self.feat_l1(torch.cat([feat1, self._up(feat2, feat1)], 1))

        off = self.lrelu(self.off_casc_1(torch.cat([feat1, r1], 1)))
        off = self.lrelu(self.off_casc_2(off))
        return self.lrelu(self.dcn_casc(feat1, off))


class TSAFusion(nn.Module):
    """Temporal-Spatial Attention fusion of the aligned frames.

    Temporal weights come from per-pixel similarity to the reference frame, so a frame
    that aligned badly in some region is down-weighted only there — the same intuition
    as the existing AttentionFusion, but applied after explicit alignment.
    """

    def __init__(self, nf: int = 32, num_frames: int = 5, center: int = 2):
        super().__init__()
        self.center = center
        self.attn_nbr = nn.Conv2d(nf, nf, 3, padding=1)
        self.attn_ref = nn.Conv2d(nf, nf, 3, padding=1)
        self.fusion = nn.Conv2d(num_frames * nf, nf, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, aligned: torch.Tensor) -> torch.Tensor:
        """aligned: [B, T, nf, H, W] -> [B, nf, H, W]"""
        b, t, c, h, w = aligned.size()
        emb = self.attn_nbr(aligned.view(-1, c, h, w)).view(b, t, c, h, w)
        emb_ref = self.attn_ref(aligned[:, self.center])
        corr = (emb * emb_ref.unsqueeze(1)).sum(2)          # [B, T, H, W]
        weights = torch.sigmoid(corr).unsqueeze(2)          # [B, T, 1, H, W]
        feat = (aligned * weights).view(b, t * c, h, w)
        feat = self.fusion(feat)
        return feat * self.spatial(feat)


class EDVRLite(nn.Module):
    """5 frames -> aligned, fused, upscaled -> the result replicated back to 5 frames."""

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 5,
        num_features: int = 32,
        num_blocks: int = 5,
        scale: int = 2,
    ):
        super().__init__()
        if scale not in (1, 2):
            raise ValueError(f"scale must be 1 or 2, got {scale}")
        self.num_frames = num_frames
        self.center = num_frames // 2
        self.scale = scale
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.feat_extract = nn.Sequential(*[ResBlock(num_features) for _ in range(2)])
        # Pyramid: stride-2 convs down to 1/2 and 1/4
        self.down_l2 = nn.Conv2d(num_features, num_features, 3, stride=2, padding=1)
        self.down_l2b = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.down_l3 = nn.Conv2d(num_features, num_features, 3, stride=2, padding=1)
        self.down_l3b = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.pcd = PCDAlign(num_features)
        self.tsa = TSAFusion(num_features, num_frames, self.center)

        self.recon = nn.Sequential(*[ResBlock(num_features) for _ in range(num_blocks)])
        self.upsampler = Upsampler(num_features) if scale == 2 else nn.Identity()
        self.tail = nn.Conv2d(num_features, in_channels, 3, padding=1)
        # Identity start: output = interpolate(reference frame) + 0, so the network
        # begins exactly at the plain-baseline behaviour and only learns the delta.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C, H, W] -> [B, T, C, H*scale, W*scale] (all T frames identical)"""
        b, t, c, h, w = x.size()

        feat = self.lrelu(self.conv_first(x.reshape(-1, c, h, w)))
        l1 = self.feat_extract(feat)
        l2 = self.lrelu(self.down_l2b(self.lrelu(self.down_l2(l1))))
        l3 = self.lrelu(self.down_l3b(self.lrelu(self.down_l3(l2))))

        nf = l1.size(1)
        l1 = l1.view(b, t, nf, *l1.shape[-2:])
        l2 = l2.view(b, t, nf, *l2.shape[-2:])
        l3 = l3.view(b, t, nf, *l3.shape[-2:])
        ref = [l1[:, self.center], l2[:, self.center], l3[:, self.center]]

        aligned = torch.stack(
            [self.pcd([l1[:, i], l2[:, i], l3[:, i]], ref) for i in range(t)], dim=1
        )

        fused = self.tsa(aligned)
        out = self.tail(self.upsampler(self.recon(fused)))

        base = x[:, self.center]
        if self.scale == 2:
            base = F.interpolate(base, scale_factor=2, mode='bilinear', align_corners=False)
        out = base + out
        return out.unsqueeze(1).expand(-1, t, -1, -1, -1).contiguous()
