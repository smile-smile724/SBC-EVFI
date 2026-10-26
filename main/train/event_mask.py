from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device, dtype):
    kernel_size = int(kernel_size)
    ax = torch.arange(kernel_size, device=device, dtype=dtype)
    ax = ax - kernel_size // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel


def make_event_mask(
    event_pos: torch.Tensor,
    event_neg: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 0.5,
):
    kernel_size = int(kernel_size)
    density = event_pos + event_neg
    mask = (density > 1e-6).to(density.dtype)
    kernel = _gaussian_kernel_2d(kernel_size, sigma, density.device, density.dtype)
    blurred = F.conv2d(mask, kernel, padding=kernel_size // 2)
    mask = blurred.clamp(0.0, 1.0)
    return mask


def make_event_delta_smooth(
    event_pos: torch.Tensor,
    event_neg: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 0.5,
):
    kernel_size = int(kernel_size)
    density = event_pos + event_neg
    kernel = _gaussian_kernel_2d(kernel_size, sigma, density.device, density.dtype)
    smooth = F.conv2d(density, kernel, padding=kernel_size // 2)
    return smooth


def make_event_delta_conf(
    event_delta_smooth: torch.Tensor,
    scale: float = 10.0,
):
    return torch.clamp(event_delta_smooth / scale, 0.0, 1.0)


def make_membrane_conf(membrane: torch.Tensor, thresh: float = 0.5, min_conf: float = 0.1):
    conf = membrane.abs().mean(dim=1, keepdim=True) / thresh
    conf = torch.clamp(conf, min_conf, 1.0)
    return conf


def _make_grid_and_scale(B, H, W, device):
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device), indexing='ij')
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    scale = torch.tensor([W / 2, H / 2], device=device).view(1, 1, 1, 2)
    return grid, scale


def make_event_features_fwd(events_up, t, model, mask_kernel_size, mask_sigma):
    event_pos = events_up[:, (t - 1) * 2:(t - 1) * 2 + 1]
    event_neg = events_up[:, (t - 1) * 2 + 1:(t - 1) * 2 + 2]
    pol_feat = model.polarity_proj(torch.cat([event_pos, event_neg], dim=1))
    mask = make_event_mask(event_pos, event_neg, mask_kernel_size, mask_sigma)
    mask = mask.to(pol_feat.dtype)
    return pol_feat, mask, event_pos, event_neg


def make_event_features_bwd(events_up, t, model, mask_kernel_size, mask_sigma):
    event_pos = events_up[:, t * 2:t * 2 + 1]
    event_neg = events_up[:, t * 2 + 1:t * 2 + 2]
    pol_feat = model.polarity_proj(torch.cat([event_pos, event_neg], dim=1))
    mask = make_event_mask(event_pos, event_neg, mask_kernel_size, mask_sigma)
    mask = mask.to(pol_feat.dtype)
    return pol_feat, mask, event_pos, event_neg


def make_event_features(events_up, t, model, mask_kernel_size, mask_sigma):
    event_pos_fwd = events_up[:, (t - 1) * 2:(t - 1) * 2 + 1]
    event_neg_fwd = events_up[:, (t - 1) * 2 + 1:(t - 1) * 2 + 2]
    event_pos_bwd = events_up[:, t * 2:t * 2 + 1]
    event_neg_bwd = events_up[:, t * 2 + 1:t * 2 + 2]

    combined = torch.cat([
        torch.cat([event_pos_fwd, event_neg_fwd], dim=1),
        torch.cat([event_pos_bwd, event_neg_bwd], dim=1)
    ], dim=0)
    combined_pol = model.polarity_proj(combined)
    pol_feat_fwd, pol_feat_bwd = combined_pol.chunk(2, dim=0)

    mask_fwd = make_event_mask(event_pos_fwd, event_neg_fwd, mask_kernel_size, mask_sigma).to(pol_feat_fwd.dtype)
    mask_bwd = make_event_mask(event_pos_bwd, event_neg_bwd, mask_kernel_size, mask_sigma).to(pol_feat_bwd.dtype)

    return pol_feat_fwd, pol_feat_bwd, mask_fwd, mask_bwd, event_pos_fwd, event_neg_fwd, event_pos_bwd, event_neg_bwd


class OffsetHead(nn.Module):
    def __init__(self, in_ch: int, hid: int = 64):
        super().__init__()
        feat_ch = (in_ch - 41) // 2
        self.feat_merge = nn.Conv2d(2 * feat_ch, feat_ch, 1, 1, 0)
        self.conv1 = nn.Conv2d(feat_ch + 41, hid, 1, 1, 0)
        self.norm1 = nn.GroupNorm(min(8, hid), hid)
        self.conv2 = nn.Conv2d(hid, hid, 5, 1, 2)
        self.norm2 = nn.GroupNorm(min(8, hid), hid)
        self.conv3 = nn.Conv2d(hid, hid // 2, 5, 1, 2)
        self.norm3 = nn.GroupNorm(min(8, hid // 2), hid // 2)
        self.conv4 = nn.Conv2d(hid // 2, 2, 5, 1, 2)

        nn.init.normal_(self.conv4.weight, std=2.5e-4)
        nn.init.zeros_(self.conv4.bias)

        self.offset_scale = nn.Parameter(torch.tensor(1.0))

        self.t_scale = nn.Parameter(torch.tensor(0.1))
        self.t_bias_proj = nn.Linear(1, hid)
        nn.init.normal_(self.t_bias_proj.weight, std=0.01)
        nn.init.zeros_(self.t_bias_proj.bias)

    def forward(self, x, mask, t_ratio=None):
        feat_ch = (x.shape[1] - 41) // 2
        feat_earlier = x[:, :feat_ch]
        feat_later = x[:, feat_ch:2*feat_ch]
        rest = x[:, 2*feat_ch:]
        feat_merged = self.feat_merge(torch.cat([feat_earlier, feat_later], dim=1))
        x = torch.cat([feat_merged, rest], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=False)

        t_val = None
        if t_ratio is not None:
            B = x.shape[0]
            t_val = torch.full((B,), float(t_ratio), device=x.device, dtype=x.dtype)
            x = x * (1.0 + mask)
            t_bias = self.t_bias_proj(t_val.view(B, 1))
            x = x + t_bias.view(B, -1, 1, 1)
        else:
            x = x * (1.0 + mask)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        offset = torch.tanh(x) * self.offset_scale

        if t_val is not None:
            offset = offset * (1.0 + t_val.view(B, 1, 1, 1) * self.t_scale)

        return offset


class DeblurHead(nn.Module):
    def __init__(self, hid: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 3, 3, 1, 1),
        )

        self.color_net = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, 1, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.color_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.last_color_bias = None

    def forward(self, warped, blurry_mask):
        x = torch.cat([warped, blurry_mask], dim=1)
        residual = self.net(x)
        residual = residual - residual.mean(dim=[-2, -1], keepdim=True)
        color_bias = self.color_net(x)
        color_bias = torch.tanh(color_bias) * 0.08
        self.last_color_bias = color_bias.detach()
        return warped + blurry_mask * residual + color_bias


class BlurryMaskPredictor(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.feat_proj = nn.Conv2d(hidden, 16, 1, 1, 0)
        in_ch = 16 + 1 + 1 + 1 + 3
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

        self.t_conv = nn.Conv2d(1, 1, 1, 1, 0)
        nn.init.normal_(self.t_conv.weight, std=0.01)
        nn.init.zeros_(self.t_conv.bias)

    def forward(self, feat_t, event_pos, event_neg, coarse_mask, warp_raw, t_ratio=None):
        feat_t_in = feat_t
        feat_t = self.feat_proj(feat_t)
        x = torch.cat([feat_t, event_pos, event_neg, coarse_mask, warp_raw], dim=1)
        result = self.net(x)
        if t_ratio is not None:
            B, H, W = feat_t.shape[0], feat_t.shape[2], feat_t.shape[3]
            t_ch = torch.full((B, 1, H, W), float(t_ratio), device=feat_t.device, dtype=feat_t.dtype)
            result = result + self.t_conv(t_ch)

        out = torch.sigmoid(result)

        if not torch.isfinite(out).all():
            if not getattr(self, '_nan_diag_done', False):
                self._nan_diag_done = True
                print(f"[BP NaN DIAG t_ratio={t_ratio}] --- INPUTS ---")
                for name, tensor in [('feat_t(in)', feat_t_in), ('event_pos', event_pos),
                                     ('event_neg', event_neg), ('coarse_mask', coarse_mask),
                                     ('warp_raw', warp_raw)]:
                    print(f"  {name}: finite={torch.isfinite(tensor).all().item()} "
                          f"mean={tensor.float().mean().item():.4f} "
                          f"std={tensor.float().std().item():.4f} "
                          f"min={tensor.float().min().item():.4f} "
                          f"max={tensor.float().max().item():.4f}")
                print(f"[BP NaN DIAG t_ratio={t_ratio}] feat_t(after proj): "
                      f"finite={torch.isfinite(feat_t).all().item()} "
                      f"mean={feat_t.float().mean().item():.4f} "
                      f"std={feat_t.float().std().item():.4f} "
                      f"min={feat_t.float().min().item():.4f} "
                      f"max={feat_t.float().max().item():.4f}")
                print(f"[BP NaN DIAG t_ratio={t_ratio}] x(cat): "
                      f"finite={torch.isfinite(x).all().item()} "
                      f"mean={x.float().mean().item():.4f} "
                      f"std={x.float().std().item():.4f} "
                      f"min={x.float().min().item():.4f} "
                      f"max={x.float().max().item():.4f}")
                y = x
                for idx, layer in enumerate(self.net):
                    y = layer(y)
                    _dtype = str(y.dtype).replace('torch.', '')
                    print(f"[BP NaN DIAG t_ratio={t_ratio}] net[{idx}]({type(layer).__name__}): "
                          f"finite={torch.isfinite(y).all().item()} "
                          f"mean={y.float().mean().item():.4f} "
                          f"std={y.float().std().item():.4f} "
                          f"min={y.float().min().item():.4f} "
                          f"max={y.float().max().item():.4f} "
                          f"dtype={_dtype}")
                    if not torch.isfinite(y).all():
                        break
                if torch.isfinite(result).all():
                    print(f"[BP NaN DIAG t_ratio={t_ratio}] result(after net+t_conv): finite=True, sigmoid is fine")
                else:
                    print(f"[BP NaN DIAG t_ratio={t_ratio}] result(after net+t_conv): finite=False")
                print(f"[BP NaN DIAG t_ratio={t_ratio}] out(sigmoid): "
                      f"mean={out.float().mean().item():.6f} "
                      f"nan_count={torch.isnan(out).sum().item()}")

        return out