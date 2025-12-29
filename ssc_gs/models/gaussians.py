from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GaussianSceneTensors:
    means: torch.Tensor  # (N,3)
    quats: torch.Tensor  # (N,4)
    scales_log: torch.Tensor  # (N,3) stored in log-space
    opacities_logit: torch.Tensor  # (N,)
    rgb_logits: torch.Tensor  # (N,3)
    sem_logits: torch.Tensor  # (N,C)
    anchor_feat: torch.Tensor  # (N,F)


class GaussianScene(nn.Module):
    """Trainable Gaussian parameters (geometry + appearance + semantics + anchor features)."""

    def __init__(
        self,
        init: GaussianSceneTensors,
    ) -> None:
        super().__init__()

        self.means = nn.Parameter(init.means)
        self.quats = nn.Parameter(init.quats)
        self.scales_log = nn.Parameter(init.scales_log)
        self.opacities_logit = nn.Parameter(init.opacities_logit)

        self.rgb_logits = nn.Parameter(init.rgb_logits)
        self.sem_logits = nn.Parameter(init.sem_logits)
        self.anchor_feat = nn.Parameter(init.anchor_feat)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.scales_log)

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities_logit)

    @property
    def rgb(self) -> torch.Tensor:
        return torch.sigmoid(self.rgb_logits)


def init_gaussians_from_points(
    points_world: torch.Tensor,
    *,
    num_classes: int,
    feat_dim: int = 256,
    init_scale: float = 0.03,
    init_opacity: float = 0.1,
    device: Optional[torch.device] = None,
    rgb_init: Optional[torch.Tensor] = None,
) -> GaussianSceneTensors:
    """Create an initial Gaussian set from a point cloud.

    Args:
        points_world: (N,3) float.
        num_classes: semantic class count.
        feat_dim: anchor feature dim.
        init_scale: isotropic initial scale in world units.
        init_opacity: initial opacity (sigmoid(opacity_logit)).
        rgb_init: optional per-point RGB in [0,1], shape (N,3).

    Returns:
        GaussianSceneTensors for `GaussianScene`.
    """
    if device is None:
        device = points_world.device

    N = points_world.shape[0]
    if N == 0:
        raise ValueError("points_world is empty")

    means = points_world.to(device=device, dtype=torch.float32)

    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0  # identity rotation in wxyz convention

    scales_log = torch.full((N, 3), float(torch.log(torch.tensor(init_scale))), device=device)
    opacities_logit = torch.full((N,), float(torch.log(torch.tensor(init_opacity / (1 - init_opacity)))), device=device)

    if rgb_init is None:
        rgb_logits = torch.zeros((N, 3), device=device)
    else:
        rgb = rgb_init.clamp(0, 1).to(device=device, dtype=torch.float32)
        eps = 1e-4
        rgb_logits = torch.log(rgb.clamp(eps, 1 - eps) / (1 - rgb.clamp(eps, 1 - eps)))

    sem_logits = torch.zeros((N, num_classes), device=device)
    anchor_feat = torch.zeros((N, feat_dim), device=device)

    return GaussianSceneTensors(
        means=means,
        quats=quats,
        scales_log=scales_log,
        opacities_logit=opacities_logit,
        rgb_logits=rgb_logits,
        sem_logits=sem_logits,
        anchor_feat=anchor_feat,
    )


def sample_points_uniform(points_world_hw3: torch.Tensor, mask_hw: torch.Tensor, max_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample up to `max_points` points from an (H,W,3) pointmap with a boolean mask."""
    if points_world_hw3.dim() != 3 or points_world_hw3.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3), got {tuple(points_world_hw3.shape)}")
    if mask_hw.shape != points_world_hw3.shape[:2]:
        raise ValueError("mask shape mismatch")

    idx = torch.nonzero(mask_hw, as_tuple=False)
    if idx.numel() == 0:
        raise ValueError("mask has no valid points")

    if idx.shape[0] > max_points:
        perm = torch.randperm(idx.shape[0], device=idx.device)[:max_points]
        idx = idx[perm]

    pts = points_world_hw3[idx[:, 0], idx[:, 1]]
    return pts, idx
