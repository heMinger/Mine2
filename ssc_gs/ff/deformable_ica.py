from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeformableICAConfig:
    feat_dim: int
    num_views_max: int
    num_levels: int
    num_points: int = 7
    hidden: int = 256


class KeypointsGenerator(nn.Module):
    """Generate a small set of 3D keypoints around each Gaussian center.

    This is a lightweight PyTorch approximation of GaussianFormer keypoint generation.
    We use axis-aligned offsets derived from per-axis scales (no rotation support).
    """

    def __init__(self, num_points: int = 7, scale_mult: float = 1.0):
        super().__init__()
        if num_points not in (1, 7):
            raise ValueError("Only num_points in {1,7} supported in this minimal implementation")
        self.num_points = int(num_points)
        self.scale_mult = float(scale_mult)

    def forward(self, means_world: torch.Tensor, scales_xyz: torch.Tensor) -> torch.Tensor:
        # means_world: (P,3), scales_xyz: (P,3)
        if means_world.dim() != 2 or means_world.shape[1] != 3:
            raise ValueError(f"means_world must be (P,3), got {tuple(means_world.shape)}")
        if scales_xyz.shape != means_world.shape:
            raise ValueError("scales_xyz must have shape (P,3)")

        P = int(means_world.shape[0])
        if self.num_points == 1:
            return means_world[:, None, :]  # (P,1,3)

        s = scales_xyz * self.scale_mult
        zeros = torch.zeros((P, 1), device=means_world.device, dtype=means_world.dtype)
        sx = s[:, 0:1]
        sy = s[:, 1:2]
        sz = s[:, 2:3]

        offsets = torch.stack(
            [
                torch.cat([zeros, zeros, zeros], dim=1),
                torch.cat([sx, zeros, zeros], dim=1),
                torch.cat([-sx, zeros, zeros], dim=1),
                torch.cat([zeros, sy, zeros], dim=1),
                torch.cat([zeros, -sy, zeros], dim=1),
                torch.cat([zeros, zeros, sz], dim=1),
                torch.cat([zeros, zeros, -sz], dim=1),
            ],
            dim=1,
        )  # (P,7,3)
        return means_world[:, None, :] + offsets


def _project_world_to_view_grid(
    *,
    points_world: torch.Tensor,
    camtoworld: torch.Tensor,
    K_norm: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world points to normalized image grid coords for grid_sample.

    Args:
      points_world: (N,3)
      camtoworld: (4,4)
      K_norm: (3,3) with normalized intrinsics (mapping x/z,y/z to [-1,1] space)

    Returns:
      grid: (1,1,N,2) float in [-1,1] (approximately)
      valid: (N,) bool
    """
    worldtocam = torch.linalg.inv(camtoworld)
    ones = torch.ones((points_world.shape[0], 1), device=points_world.device, dtype=points_world.dtype)
    pw = torch.cat([points_world, ones], dim=1)  # (N,4)
    pc = (worldtocam @ pw.t()).t()[:, :3]

    z = pc[:, 2].clamp(min=eps)
    x = pc[:, 0] / z
    y = pc[:, 1] / z

    u = K_norm[0, 0] * x + K_norm[0, 2]
    v = K_norm[1, 1] * y + K_norm[1, 2]

    valid = (pc[:, 2] > eps) & (u >= -1.5) & (u <= 1.5) & (v >= -1.5) & (v <= 1.5)
    grid = torch.stack([u, v], dim=-1).view(1, 1, -1, 2)
    return grid, valid


class DeformableICA(nn.Module):
    """Deformable ICA (PyTorch) via keypoints projection + grid_sample.

    This mirrors GaussianFormer-style deformable feature aggregation conceptually:
      - generate a few 3D keypoints per Gaussian
      - project to each view/level feature map
      - sample features at those locations
      - fuse with learned softmax weights

    Notes:
      - This is B=1 and uses axis-aligned keypoints (no rotation).
      - Multi-level is supported by passing multiple feature maps per view.
    """

    def __init__(self, cfg: DeformableICAConfig):
        super().__init__()
        self.cfg = cfg
        self.kps = KeypointsGenerator(num_points=int(cfg.num_points), scale_mult=1.0)

        in_dim = int(cfg.feat_dim) + 6  # feat + (mean xyz, log scale xyz)
        self.weights_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(cfg.hidden)),
            nn.GELU(),
            nn.Linear(int(cfg.hidden), int(cfg.num_views_max) * int(cfg.num_levels) * int(cfg.num_points)),
        )
        self.out_proj = nn.Linear(int(cfg.feat_dim), int(cfg.feat_dim))
        self.ff = nn.Sequential(
            nn.LayerNorm(int(cfg.feat_dim)),
            nn.Linear(int(cfg.feat_dim), int(cfg.feat_dim) * 4),
            nn.GELU(),
            nn.Linear(int(cfg.feat_dim) * 4, int(cfg.feat_dim)),
        )

    @staticmethod
    def _as_list(x: Iterable[str] | Sequence[str]) -> List[str]:
        return list(x)

    def forward(
        self,
        *,
        instance_feat: torch.Tensor,  # (P,F)
        means_world: torch.Tensor,  # (P,3)
        scales_xyz: torch.Tensor,  # (P,3)
        feats_by_view: List[List[torch.Tensor]],  # views -> levels -> (1,F,h,w)
        Ks_norm: List[torch.Tensor],
        camtoworlds: List[torch.Tensor],
    ) -> torch.Tensor:
        P, Fdim = instance_feat.shape
        V = len(feats_by_view)
        L = len(feats_by_view[0]) if V > 0 else 0
        if V == 0 or L == 0:
            return instance_feat
        if L != int(self.cfg.num_levels):
            raise ValueError(f"Expected num_levels={self.cfg.num_levels}, got {L}")
        if V > int(self.cfg.num_views_max):
            raise ValueError(f"Too many views ({V}) for num_views_max={self.cfg.num_views_max}")

        # Generate keypoints (P,K,3) and flatten to (P*K,3)
        kps = self.kps(means_world, scales_xyz)  # (P,K,3)
        Kp = int(kps.shape[1])
        pts = kps.reshape(P * Kp, 3)

        # Sample features for each view/level at keypoint projections.
        sampled = []  # list of (P,K,F) for each (view,level)
        valid_all = []  # list of (P,K) bool for each (view,level)
        for v in range(V):
            K_norm = Ks_norm[v].to(device=instance_feat.device, dtype=instance_feat.dtype)
            c2w = camtoworlds[v].to(device=instance_feat.device, dtype=instance_feat.dtype)
            grid, valid = _project_world_to_view_grid(points_world=pts, camtoworld=c2w, K_norm=K_norm)
            valid = valid.view(P, Kp)

            for l in range(L):
                fmap = feats_by_view[v][l]
                if fmap.dim() != 4 or fmap.shape[0] != 1 or fmap.shape[1] != Fdim:
                    raise ValueError("Expected fmap (1,F,h,w)")
                # grid: (1,1,P*K,2) => output (1,F,1,P*K)
                samp = F.grid_sample(fmap, grid, mode="bilinear", align_corners=False)
                samp = samp.view(1, Fdim, 1, P, Kp).permute(3, 4, 1, 0, 2).reshape(P, Kp, Fdim)
                sampled.append(samp)
                valid_all.append(valid)

            # Stack view/level first, then flatten keypoints into token dimension.
            # sampled_vl: (P, V*L, K, F) -> sampled_t: (P, V*L*K, F)
            sampled_vl = torch.stack(sampled, dim=1)  # (P, V*L, K, F)
            valid_vl = torch.stack(valid_all, dim=1)  # (P, V*L, K)
            T = int(sampled_vl.shape[1] * sampled_vl.shape[2])
            sampled_t = sampled_vl.reshape(P, T, Fdim)
            valid_tok = valid_vl.reshape(P, T)

        # Predict weights over tokens (P, Vmax*L*K), then slice to active V.
        xyz = means_world.to(dtype=instance_feat.dtype)
        log_s = torch.log(scales_xyz.clamp(min=1e-3)).to(dtype=instance_feat.dtype)
        w_in = torch.cat([instance_feat, xyz, log_s], dim=-1)
        w_logits_full = self.weights_mlp(w_in)  # (P, Vmax*L*K)

        # Keep only active views
        total_full = int(self.cfg.num_views_max) * int(self.cfg.num_levels) * int(self.cfg.num_points)
        if w_logits_full.shape[1] != total_full:
            raise RuntimeError("Unexpected weight head output shape")

        # Reshape to (P, Vmax, L, K) and slice V
        w_logits = w_logits_full.view(P, int(self.cfg.num_views_max), int(self.cfg.num_levels), int(self.cfg.num_points))
        w_logits = w_logits[:, :V, :, :Kp].reshape(P, T)

        # Mask invalid projections
        w_logits = w_logits.masked_fill(~valid_tok, float("-inf"))
        # Avoid all-masked rows (if a gaussian is totally out of view): set to zeros.
        all_miss = ~valid_tok.any(dim=1)
        if all_miss.any():
            w_logits = w_logits.clone()
            w_logits[all_miss] = 0.0

        w = torch.softmax(w_logits, dim=1).to(sampled_t.dtype)  # (P,T)
        fused = torch.sum(sampled_t * w[:, :, None], dim=1)  # (P,F)

        out = instance_feat + self.out_proj(fused)
        out = out + self.ff(out)
        return out
