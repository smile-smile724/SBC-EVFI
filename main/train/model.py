from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .event_mask import BlurryMaskPredictor, DeblurHead, OffsetHead


class StructureEncoder(nn.Module):
    def __init__(self, out_ch: int = 32):
        super().__init__()
        mid_ch = out_ch // 2

        self.semantic = nn.Sequential(
            nn.Conv2d(3, mid_ch, 3, 2, 1),
            nn.GroupNorm(4, mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.GroupNorm(4, mid_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
        )

        self.edge = nn.Sequential(
            nn.Conv2d(3, mid_ch, 3, 1, 1),
            nn.GroupNorm(4, mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        return torch.cat([self.semantic(x), self.edge(x)], dim=1)


class LIFNeuron(nn.Module):
    def __init__(self, thresh: float = 0.5, decay: float = 0.5):
        super().__init__()
        self.thresh = thresh
        self.decay = decay

    def forward(self, x: torch.Tensor, membrane: torch.Tensor | None = None):
        if membrane is None:
            membrane = torch.zeros_like(x)
        membrane = self.decay * membrane + x
        spike = (membrane >= self.thresh).to(x.dtype)
        membrane = membrane * (1 - spike)
        spike = spike + (membrane >= self.thresh).to(x.dtype) - spike.detach() + spike
        return spike, membrane


class SNNReconstruct(nn.Module):
    def __init__(self, T: int = 8, hidden: int = 128, phase: int = 1):
        super().__init__()
        self.T = T
        self._phase = phase

        self.encoder_body = nn.Sequential(
            nn.Conv2d(2, hidden, 3, 1, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
        )
        self.encoder_spike = nn.Sigmoid()
        self.encoder_offset = nn.Identity()

        self.lif = LIFNeuron(thresh=1.0, decay=0.3)

        self.structure_encoder = StructureEncoder(out_ch=32)
        self.deblur_head_bwd = DeblurHead()
        self.deblur_head_fwd = DeblurHead()
        self.blurry_predictor_fwd = BlurryMaskPredictor(hidden=hidden)
        self.blurry_predictor_bwd = BlurryMaskPredictor(hidden=hidden)

        self.polarity_proj = nn.Sequential(
            nn.Conv2d(2, 8, 3, 1, 1),
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=False),
        )

        offset_in_ch = 2 * hidden + 32 + 9

        self.offset_head_fwd = OffsetHead(offset_in_ch, hid=64)
        self.offset_head_bwd = OffsetHead(offset_in_ch, hid=64)

        if phase == 2:
            self.deblur_head_fusion = DeblurHead()

            fusion_in_ch = 15  # cascade_fwd(3)+cascade_bwd(3)+prev_kf(3)+next_kf(3)+t_ratio(1)+blurry_fwd(1)+blurry_bwd(1)

            self.fusion_head = nn.Sequential(
                nn.Conv2d(fusion_in_ch, hidden, 1, 1, 0),
                nn.GroupNorm(min(8, hidden), hidden),
                nn.ReLU(inplace=False),
                nn.Conv2d(hidden, hidden, 5, 1, 2),
                nn.GroupNorm(min(8, hidden), hidden),
                nn.ReLU(inplace=False),
                nn.Conv2d(hidden, hidden // 2, 5, 1, 2),
                nn.GroupNorm(min(8, hidden // 2), hidden // 2),
                nn.ReLU(inplace=False),
                nn.Conv2d(hidden // 2, 7, 5, 1, 2),
            )

    def forward(self, event_slices: torch.Tensor):
        if event_slices.dim() == 4:
            B, T2, H, W = event_slices.shape
            T = T2 // 2
            event_slices = event_slices.view(B, T, 2, H, W)
        else:
            B, T, _, H, W = event_slices.shape
        T_use = min(T, self.T)

        encoder_input = event_slices[:, :T_use].reshape(-1, 2, H, W)
        if self.training:
            body_out = torch.utils.checkpoint.checkpoint(
                self.encoder_body, encoder_input, use_reentrant=False)
        else:
            body_out = self.encoder_body(encoder_input)

        spike_feats = self.encoder_spike(body_out)
        offset_feats = self.encoder_offset(body_out)

        spike_feats = spike_feats.view(event_slices.shape[0], T_use, -1, H, W)
        offset_feats = offset_feats.view(event_slices.shape[0], T_use, -1, H, W)

        all_feats = [spike_feats[:, t] for t in range(T_use)]
        all_offset_feats = [offset_feats[:, t] for t in range(T_use)]
        all_membranes = []
        membrane = None

        for t in range(T_use):
            spike, membrane = self.lif(spike_feats[:, t], membrane)
            all_membranes.append(membrane.clone())

        return all_feats, all_membranes, all_offset_feats