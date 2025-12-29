from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from models.common.model.layers import ssim


@dataclass
class LossConfig:
    ssim_lambda: float = 0.2  # weight on SSIM term (matches common practice)
    sem_lambda: float = 1.0
    ignore_index: int = 255
    # Stabilize semantic optimization when learning per-Gaussian logits.
    # Large-magnitude logits can make softmax output exactly 0/1 in float32,
    # killing gradients (observed in practice).
    sem_temperature: float = 1.0
    sem_logit_clip: float = 20.0


def photo_l1_ssim(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, *, ssim_lambda: float) -> torch.Tensor:
    """Photometric loss: (1-λ)*L1 + λ*SSIM.

    Args:
        pred_rgb, gt_rgb: (B,3,H,W) in [0,1]
    """
    l1 = torch.mean(torch.abs(pred_rgb - gt_rgb))
    # ssim() returns per-pixel 1-SSIM like residual (see S4C implementation)
    ssim_res = torch.mean(ssim(pred_rgb, gt_rgb, pad_reflection=False, gaussian_average=True, comp_mode=True))
    return (1.0 - ssim_lambda) * l1 + ssim_lambda * ssim_res


def semantic_ce(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    """Cross entropy for semantic completion supervision.

    Args:
        pred_logits: (B,C,H,W)
        target: (B,H,W) long
    """
    return F.cross_entropy(pred_logits, target, ignore_index=ignore_index)


def semantic_nll_from_probs(
    pred_probs: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Negative log-likelihood using probabilities instead of logits.

    This is useful when the rendering step is a linear blend (e.g., splatting):
    rasterize per-Gaussian class probabilities, renormalize per-pixel, then apply NLL.

    Args:
        pred_probs: (B,C,H,W) probabilities (should sum to 1 along C).
        target: (B,H,W) long.
    """
    if pred_probs.dim() != 4:
        raise ValueError(f"Expected pred_probs (B,C,H,W), got {tuple(pred_probs.shape)}")
    if target.dim() != 3:
        raise ValueError(f"Expected target (B,H,W), got {tuple(target.shape)}")
    if pred_probs.shape[0] != target.shape[0] or pred_probs.shape[2:] != target.shape[1:]:
        raise ValueError("Shape mismatch between pred_probs and target")

    C = int(pred_probs.shape[1])
    valid = (target != ignore_index) & (target >= 0) & (target < C)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_probs.device, dtype=pred_probs.dtype)

    # Gather p(target) per pixel
    tgt = target.clone()
    tgt[~valid] = 0
    tgt = tgt.unsqueeze(1)  # (B,1,H,W)
    p = pred_probs.gather(1, tgt).squeeze(1)  # (B,H,W)
    # Use (p + eps) instead of clamp to keep gradients when p==0.
    nll = -torch.log((p.clamp(min=0.0)) + eps)
    return nll[valid].mean()


def depth_smoothness(
    pred_depth: torch.Tensor,
    img_01: torch.Tensor,
    *,
    downsample_to_pred: bool = True,
) -> torch.Tensor:
    """Edge-aware smoothness for predicted depth.

    This matches the intent of the report's "smoothness" term: encourage piecewise-smooth
    geometry while preserving depth discontinuities aligned with image edges.

    Args:
      pred_depth: (B,1,h,w) depth in meters (or any positive scale)
      img_01: (B,3,H,W) RGB in [0,1]
    """
    if pred_depth.dim() != 4 or pred_depth.shape[1] != 1:
        raise ValueError(f"Expected pred_depth (B,1,h,w), got {tuple(pred_depth.shape)}")
    if img_01.dim() != 4 or img_01.shape[1] != 3:
        raise ValueError(f"Expected img_01 (B,3,H,W), got {tuple(img_01.shape)}")

    if downsample_to_pred and img_01.shape[-2:] != pred_depth.shape[-2:]:
        img = F.interpolate(img_01, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)
    else:
        img = img_01

    # Normalize depth scale for stable gradients.
    depth = pred_depth / (pred_depth.mean(dim=(2, 3), keepdim=True).clamp(min=1e-3))

    def grad_x(t: torch.Tensor) -> torch.Tensor:
        return t[:, :, :, 1:] - t[:, :, :, :-1]

    def grad_y(t: torch.Tensor) -> torch.Tensor:
        return t[:, :, 1:, :] - t[:, :, :-1, :]

    d_dx = grad_x(depth).abs()
    d_dy = grad_y(depth).abs()

    i_dx = grad_x(img).abs().mean(dim=1, keepdim=True)
    i_dy = grad_y(img).abs().mean(dim=1, keepdim=True)

    w_dx = torch.exp(-i_dx)
    w_dy = torch.exp(-i_dy)

    return (d_dx * w_dx).mean() + (d_dy * w_dy).mean()
