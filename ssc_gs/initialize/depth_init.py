from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from ssc_gs.models.gaussians import sample_points_uniform


@dataclass
class DepthInitConfig:
    cache_dir: str = "mapanything_cache"
    max_points: int = 200_000
    stride: int = 1


def depth_to_world_pointmap(
    *,
    depth_hw: torch.Tensor,
    K_px: torch.Tensor,
    camtoworld: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a depth map (z depth in camera frame) to a world-frame pointmap.

    Args:
        depth_hw: (H,W) float, depth is z in camera frame.
        K_px: (3,3) intrinsics in pixels.
        camtoworld: (4,4) camera-to-world.

    Returns:
        points_world_hw3: (H,W,3)
        valid_mask_hw: (H,W) bool (depth>0)
    """
    if depth_hw.dim() != 2:
        raise ValueError(f"depth_hw must be (H,W), got {tuple(depth_hw.shape)}")
    if K_px.shape != (3, 3):
        raise ValueError(f"K_px must be (3,3), got {tuple(K_px.shape)}")
    if camtoworld.shape != (4, 4):
        raise ValueError(f"camtoworld must be (4,4), got {tuple(camtoworld.shape)}")

    H, W = depth_hw.shape
    device = depth_hw.device

    valid = depth_hw > 0

    fx = K_px[0, 0]
    fy = K_px[1, 1]
    cx = K_px[0, 2]
    cy = K_px[1, 2]

    xs = torch.arange(W, device=device, dtype=depth_hw.dtype)
    ys = torch.arange(H, device=device, dtype=depth_hw.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    z = depth_hw
    x = (grid_x - cx) / fx * z
    y = (grid_y - cy) / fy * z

    ones = torch.ones_like(z)
    pts_cam_h = torch.stack([x, y, z, ones], dim=-1)  # (H,W,4)

    # (H,W,4) @ (4,4)^T -> (H,W,4)
    pts_world_h = pts_cam_h @ camtoworld.transpose(0, 1)
    pts_world = pts_world_h[..., :3]

    return pts_world, valid


def sample_world_points_from_depth(
    *,
    depth_hw: torch.Tensor,
    K_px: torch.Tensor,
    camtoworld: torch.Tensor,
    max_points: int,
    stride: int = 1,
) -> torch.Tensor:
    points_world_hw3, valid_mask_hw = depth_to_world_pointmap(depth_hw=depth_hw, K_px=K_px, camtoworld=camtoworld)

    if stride > 1:
        points_world_hw3 = points_world_hw3[::stride, ::stride]
        valid_mask_hw = valid_mask_hw[::stride, ::stride]

    pts, _ = sample_points_uniform(points_world_hw3, valid_mask_hw, max_points=max_points)
    return pts


def cache_path(cache_dir: str, sample_index: int) -> Path:
    return Path(cache_dir) / f"{sample_index:08d}.npz"


def load_cached_points(cache_dir: str, sample_index: int) -> Optional[torch.Tensor]:
    p = cache_path(cache_dir, sample_index)
    if not p.exists():
        return None
    data = np.load(p)
    pts = torch.from_numpy(data["points_world"]).float()
    return pts


def save_cached_points(cache_dir: str, sample_index: int, points_world: torch.Tensor) -> None:
    p = cache_path(cache_dir, sample_index)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, points_world=points_world.detach().cpu().numpy().astype(np.float32))
