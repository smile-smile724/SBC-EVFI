"""Standard image quality metrics: PSNR, SSIM.

SSIM follows Wang et al. 2004: Gaussian kernel (sigma=1.5, ksize=11) with padding.
PSNR uses standard MSE-based formula.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    return g


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0):
    """Compute PSNR between two images (any shape, will be flattened)."""
    mse = F.mse_loss(pred, target)
    if mse.item() < 1e-10:
        return float("inf")
    return 10.0 * math.log10(data_range ** 2 / mse.item())


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
):
    """Compute SSIM between two images (B, C, H, W) or (C, H, W).

    Returns (ssim_map, mean_ssim) where ssim_map has shape (B, C, H, W).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    kernel_1d = _gaussian_kernel_1d(kernel_size, sigma, pred.device, pred.dtype)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.expand(pred.shape[1], 1, kernel_size, kernel_size)

    mu1 = F.conv2d(pred, kernel_2d, groups=pred.shape[1], padding=kernel_size // 2)
    mu2 = F.conv2d(target, kernel_2d, groups=pred.shape[1], padding=kernel_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel_2d, groups=pred.shape[1], padding=kernel_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel_2d, groups=pred.shape[1], padding=kernel_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel_2d, groups=pred.shape[1], padding=kernel_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map, ssim_map.mean()