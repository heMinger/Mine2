from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from ssc_gs.models.dinov2 import DinoV2Frozen, DinoV2Config


@dataclass
class FFGaussianConfig:
    num_classes: int = 20
    num_depth_bins: int = 64
    depth_min: float = 1.0
    depth_max: float = 72.0
    max_scale: float = 1.2
    min_scale: float = 0.05

    dino_out_dim: int = 256


class FeedForwardGaussianPredictor(nn.Module):
    """Feed-forward Gaussian generator.

    Takes a single keyframe image and predicts P Gaussians associated with the DINO
    token grid (P = h*w).

    Outputs Gaussians in the *camera* coordinate frame of the keyframe.
    """

    def __init__(self, cfg: FFGaussianConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.dino = DinoV2Frozen(DinoV2Config(out_dim=cfg.dino_out_dim))

        # Token-wise heads (applied to (B, F, h, w))
        Fdim = int(cfg.dino_out_dim)
        self.depth_head = nn.Conv2d(Fdim, cfg.num_depth_bins + 1, kernel_size=1)
        self.opacity_head = nn.Conv2d(Fdim, 1, kernel_size=1)
        self.scale_head = nn.Conv2d(Fdim, 3, kernel_size=1)
        self.sem_head = nn.Conv2d(Fdim, cfg.num_classes, kernel_size=1)

        # We keep rotation fixed as identity for now (anisotropic scales already exist).

        depth_bins = torch.linspace(cfg.depth_min, cfg.depth_max, cfg.num_depth_bins)
        self.register_buffer("depth_bins", depth_bins, persistent=False)

    @staticmethod
    def _make_uv_grid_norm(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """(h,w,3) homogeneous normalized coords in [-1,1] matching S4C normalized K."""
        u = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        v = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        uu = u[None, :].expand(h, w)
        vv = v[:, None].expand(h, w)
        uv = torch.stack([uu * 2.0 - 1.0, vv * 2.0 - 1.0], dim=-1)  # (h,w,2)
        ones = torch.ones((h, w, 1), device=device, dtype=dtype)
        return torch.cat([uv, ones], dim=-1)  # (h,w,3)

    def forward(self, *, img_m11_chw: torch.Tensor, K_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
        img_m11_chw: (3,H,W) in [-1,1]
        K_norm: (3,3) normalized intrinsics as in S4C

        Returns dict with:
          means_cam: (P,3)
          opacities: (P,1)
          scales: (P,3)
          sem_probs: (P,C)
          token_hw: (2,) tensor [h,w]
        """
        device = img_m11_chw.device
        dtype = img_m11_chw.dtype

        img = ((img_m11_chw[None] + 1.0) * 0.5).clamp(0, 1)  # (1,3,H,W)
        feats = self.dino(img, return_pyramid=False)  # (1,F,h,w)
        _, _, h, w = feats.shape

        depth_logits = self.depth_head(feats)  # (1,D+1,h,w)
        opacity_logits = self.opacity_head(feats)  # (1,1,h,w)
        scale_logits = self.scale_head(feats)  # (1,3,h,w)
        sem_logits = self.sem_head(feats)  # (1,C,h,w)

        depth_p = F.softmax(depth_logits, dim=1)
        p_valid = depth_p[:, : self.cfg.num_depth_bins]  # (1,D,h,w)
        p_empty = depth_p[:, self.cfg.num_depth_bins :]  # (1,1,h,w)

        # Differentiable depth expectation; empty bin contributes 0 depth.
        depth = torch.sum(p_valid * self.depth_bins.view(1, -1, 1, 1), dim=1, keepdim=True)  # (1,1,h,w)

        # Ray directions from normalized intrinsics
        K_inv = torch.inverse(K_norm.to(device=device, dtype=dtype))
        uvd = self._make_uv_grid_norm(h, w, device=device, dtype=dtype).view(-1, 3)  # (P,3)
        dirs = (K_inv @ uvd.t()).t()  # (P,3)
        means_cam = dirs * depth.view(-1, 1)  # (P,3)

        # Opacity: also gate by non-empty probability
        opacities = torch.sigmoid(opacity_logits).view(-1, 1) * (1.0 - p_empty.view(-1, 1))

        # Scales in meters-ish (positive)
        scales01 = torch.sigmoid(scale_logits).permute(0, 2, 3, 1).reshape(-1, 3)
        scales = self.cfg.min_scale + (self.cfg.max_scale - self.cfg.min_scale) * scales01

        sem_probs = F.softmax(sem_logits, dim=1).permute(0, 2, 3, 1).reshape(-1, self.cfg.num_classes)

        return {
            "means_cam": means_cam,
            "opacities": opacities,
            "scales": scales,
            "sem_probs": sem_probs,
            "token_hw": torch.tensor([h, w], device=device, dtype=torch.int64),
        }
