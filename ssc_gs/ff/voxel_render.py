from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from ssc_gs.ff.gaussian_to_voxel import VoxelGridSpec


@dataclass
class RayMarchConfig:
    num_steps: int = 64
    depth_min: float = 0.5
    depth_max: float = 72.0


def _grid_sample_3d(volume: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """volume: (1,C,D,H,W), grid: (1,N,1,1,3) in [-1,1]. Returns (1,C,N,1,1)."""
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def render_semantic_from_voxels(
    *,
    sigma: torch.Tensor,
    sem_probs: torch.Tensor,
    K_norm: torch.Tensor,
    camtoworld: torch.Tensor,
    world_to_velo_key: torch.Tensor,
    T_velo_to_cam: torch.Tensor,
    image_hw: Tuple[int, int],
    grid: VoxelGridSpec,
    cfg: RayMarchConfig,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render 2D semantic probabilities by ray-marching a voxel volume.

    The voxel volume is defined in *keyframe velodyne coordinates*.
    We render from an arbitrary view (camtoworld) by transforming sampled 3D points:
      cam_j -> world -> velo_key -> voxel grid.

    Args:
      sigma: (Z,Y,X) in [0,1]
      sem_probs: (C,Z,Y,X) per-voxel probs
      K_norm: (3,3) normalized intrinsics for this view
      camtoworld: (4,4)
      world_to_velo_key: (4,4)
      T_velo_to_cam: (4,4) (velo->cam for this camera type)
      image_hw: (H,W)

    Returns:
      pred_probs: (1,C,H,W)
      valid_mask: (1,1,H,W) where any density was accumulated
    """
    device = sigma.device
    dtype = sigma.dtype
    H, W = map(int, image_hw)

    C = int(sem_probs.shape[0])

    # volumes for grid_sample are (N,C,D,H,W)
    vol_sigma = sigma[None, None].to(torch.float32)  # (1,1,Z,Y,X)
    vol_sem = sem_probs[None].to(torch.float32)  # (1,C,Z,Y,X)

    # Pixel grid in normalized [-1,1]
    u = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
    v = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
    uu = u[None, :].expand(H, W) * 2.0 - 1.0
    vv = v[:, None].expand(H, W) * 2.0 - 1.0

    ones = torch.ones_like(uu)
    pix = torch.stack([uu, vv, ones], dim=-1).view(-1, 3)  # (Npix,3)

    K_inv = torch.inverse(K_norm.to(device=device, dtype=dtype))
    dirs_cam = (K_inv @ pix.t()).t()  # (Npix,3)

    # Depth samples
    t = torch.linspace(cfg.depth_min, cfg.depth_max, cfg.num_steps, device=device, dtype=dtype)  # (S,)

    # (Npix,S,3) points in camera
    pts_cam = dirs_cam[:, None, :] * t[None, :, None]

    # cam -> world
    Npix = pts_cam.shape[0]
    S = pts_cam.shape[1]
    pts_h = torch.cat([pts_cam, torch.ones((Npix, S, 1), device=device, dtype=dtype)], dim=-1)  # (Npix,S,4)
    camtoworld = camtoworld.to(device=device, dtype=dtype)
    pts_world = (pts_h @ camtoworld.t())  # (Npix,S,4)

    # world -> key velo
    world_to_velo_key = world_to_velo_key.to(device=device, dtype=dtype)
    pts_velo_key = (pts_world @ world_to_velo_key.t())[..., :3]  # (Npix,S,3)

    # Convert to voxel grid normalized coords for grid_sample.
    origin = torch.tensor(grid.origin, device=device, dtype=dtype)
    voxel = float(grid.voxel_size)
    X, Y, Z = map(int, grid.dims_xyz)

    xyz = (pts_velo_key - origin[None, None, :]) / voxel  # (Npix,S,3) in voxel units
    # grid_sample expects (x,y,z) normalized to [-1,1] in order W,H,D
    gx = (xyz[..., 0] / max(X - 1, 1)) * 2.0 - 1.0
    gy = (xyz[..., 1] / max(Y - 1, 1)) * 2.0 - 1.0
    gz = (xyz[..., 2] / max(Z - 1, 1)) * 2.0 - 1.0

    grid_ = torch.stack([gx, gy, gz], dim=-1)  # (Npix,S,3)
    # reshape for grid_sample: (1, Npix*S, 1, 1, 3)
    grid_ = grid_.view(1, Npix * S, 1, 1, 3)

    # Sample sigma and sem
    sig_samp = _grid_sample_3d(vol_sigma, grid_).view(1, 1, Npix, S)  # (1,1,Npix,S)
    sem_samp = _grid_sample_3d(vol_sem, grid_).view(1, C, Npix, S)  # (1,C,Npix,S)

    sig = sig_samp.clamp(0, 1)

    # Convert sigma to alpha per step (simple mapping).
    alpha = sig  # (1,1,Npix,S)

    # Transmittance
    one_minus = (1.0 - alpha).clamp(min=0.0, max=1.0)
    # cumulative product along S (exclusive)
    T = torch.cumprod(torch.cat([torch.ones((1, 1, Npix, 1), device=device), one_minus[:, :, :, :-1]], dim=3), dim=3)
    w = (T * alpha).clamp(min=0.0)  # (1,1,Npix,S)

    # Weighted sum of sem probs along ray
    wC = w.expand(-1, C, -1, -1)
    sem_agg = torch.sum(wC * sem_samp, dim=3)  # (1,C,Npix)
    occ = torch.sum(w, dim=3).clamp(min=0.0)  # (1,1,Npix)

    pred = sem_agg / (occ + eps)
    pred = pred.view(1, C, H, W)

    valid = (occ.view(1, 1, H, W) > 1e-4)
    return pred, valid.to(pred.dtype)
