from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()


@dataclass
class VoxelGridSpec:
    origin: Tuple[float, float, float] = (0.0, -25.6, -2.0)
    voxel_size: float = 0.2
    dims_xyz: Tuple[int, int, int] = (256, 256, 32)  # (X,Y,Z)


def _flat_index(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, *, Y: int, Z: int) -> torch.Tensor:
    return (ix * (Y * Z) + iy * Z + iz).to(torch.int64)


def splat_gaussians_to_voxels_trilinear(
    *,
    means_xyz: torch.Tensor,
    opacities: torch.Tensor,
    sem_probs: torch.Tensor,
    grid: VoxelGridSpec,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable splatting from Gaussians to a dense voxel grid.

    This is a pure-PyTorch approximation of Gaussian-to-Voxel splatting:
    - each Gaussian contributes to the 8 nearest voxel centers via trilinear weights.
    - weights are multiplied by opacity.

    Args:
      means_xyz: (P,3) points in the same coordinate frame as the voxel grid.
      opacities: (P,1) in [0,1]
      sem_probs: (P,C) probs sum=1

    Returns:
      sigma_sum: (Z,Y,X) float32 summed opacity per voxel (pre-squash)
      sigma: (Z,Y,X) float32 in [0,1] using 1-exp(-sigma_sum)
      sem_probs_vox: (C,Z,Y,X) float32 normalized per-voxel semantic probs
    """
    device = means_xyz.device
    dtype = means_xyz.dtype

    origin = torch.tensor(grid.origin, device=device, dtype=dtype)
    voxel = float(grid.voxel_size)
    X, Y, Z = map(int, grid.dims_xyz)

    P = int(means_xyz.shape[0])
    C = int(sem_probs.shape[1])

    # Continuous voxel coordinates
    xyz = (means_xyz - origin[None]) / voxel  # (P,3)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    z0 = torch.floor(z)

    fx = (x - x0).clamp(0, 1)
    fy = (y - y0).clamp(0, 1)
    fz = (z - z0).clamp(0, 1)

    x0 = x0.to(torch.int64)
    y0 = y0.to(torch.int64)
    z0 = z0.to(torch.int64)

    # 8 corners
    corners = []
    weights = []
    for dx in (0, 1):
        wx = (1 - fx) if dx == 0 else fx
        ix = x0 + dx
        for dy in (0, 1):
            wy = (1 - fy) if dy == 0 else fy
            iy = y0 + dy
            for dz in (0, 1):
                wz = (1 - fz) if dz == 0 else fz
                iz = z0 + dz
                w = (wx * wy * wz)  # (P,)
                corners.append((ix, iy, iz))
                weights.append(w)

    sigma_sum = torch.zeros((Z * Y * X,), device=device, dtype=torch.float32)
    sem_acc = torch.zeros((Z * Y * X, C), device=device, dtype=torch.float32)

    opa = opacities.view(-1).to(torch.float32).clamp(0, 1)
    sem = sem_probs.to(torch.float32)

    for (ix, iy, iz), w in zip(corners, weights):
        inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        if not inside.any():
            continue
        flat = _flat_index(ix[inside], iy[inside], iz[inside], Y=Y, Z=Z)
        ww = (opa[inside] * w[inside].to(torch.float32))
        sigma_sum.index_add_(0, flat, ww)
        sem_acc.index_add_(0, flat, ww[:, None] * sem[inside])

    sigma = 1.0 - torch.exp(-sigma_sum)
    sigma = sigma.clamp(0, 1)

    # Normalize semantics by accumulated opacity (not squashed sigma)
    denom = sigma_sum.clamp(min=eps)
    sem_probs_vox = (sem_acc / denom[:, None]).t()  # (C, N)

    sigma_sum = sigma_sum.view(Z, Y, X)
    sigma = sigma.view(Z, Y, X)
    sem_probs_vox = sem_probs_vox.view(C, Z, Y, X)

    return sigma_sum, sigma, sem_probs_vox


def splat_gaussians_to_voxels_gaussian_kernel(
    *,
    means_xyz: torch.Tensor,
    opacities: torch.Tensor,
    scales_xyz: torch.Tensor,
    sem_probs: torch.Tensor,
    grid: VoxelGridSpec,
    max_radius: int = 3,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-aware Gaussian->Voxel splatting (pure PyTorch).

    This is a practical approximation inspired by GaussianFormer-style local aggregation:
    - Each Gaussian contributes to a small (2r+1)^3 neighborhood in voxel space.
    - We use an axis-aligned Gaussian kernel with per-axis scales.
    - Accumulated per-voxel opacity is `sigma_sum`; squashed occupancy is `sigma = 1-exp(-sigma_sum)`.

    Notes:
      - We intentionally cap the neighborhood radius (`max_radius`) for performance.
      - This is designed for P ~ token_count (few hundred to few thousand).

    Args:
      means_xyz: (P,3) points in the same coordinate frame as the voxel grid.
      opacities: (P,1) in [0,1].
      scales_xyz: (P,3) positive scales (meters) for x/y/z.
      sem_probs: (P,C) probabilities.

    Returns:
      sigma_sum: (Z,Y,X)
      sigma: (Z,Y,X)
      sem_probs_vox: (C,Z,Y,X)
    """
    if means_xyz.dim() != 2 or means_xyz.shape[1] != 3:
        raise ValueError(f"Expected means_xyz (P,3), got {tuple(means_xyz.shape)}")
    if scales_xyz.shape != means_xyz.shape:
        raise ValueError("Expected scales_xyz shape == means_xyz shape")

    device = means_xyz.device
    dtype = means_xyz.dtype

    origin = torch.tensor(grid.origin, device=device, dtype=dtype)
    voxel = float(grid.voxel_size)
    X, Y, Z = map(int, grid.dims_xyz)
    C = int(sem_probs.shape[1])

    # Continuous voxel coordinates
    xyz = (means_xyz - origin[None]) / voxel  # (P,3)

    # Center voxel indices
    ix0 = torch.round(xyz[:, 0]).to(torch.int64)
    iy0 = torch.round(xyz[:, 1]).to(torch.int64)
    iz0 = torch.round(xyz[:, 2]).to(torch.int64)

    # Convert scales (meters) -> scales in voxel units, and clamp to avoid div-by-zero
    s_vox = (scales_xyz / voxel).to(torch.float32).clamp(min=0.5)  # (P,3)

    # Neighborhood radius per Gaussian (in voxels)
    r = torch.ceil(s_vox.max(dim=1).values * 2.0).to(torch.int64)  # ~2 sigma
    r = torch.clamp(r, min=1, max=int(max_radius))

    sigma_sum = torch.zeros((Z * Y * X,), device=device, dtype=torch.float32)
    sem_acc = torch.zeros((Z * Y * X, C), device=device, dtype=torch.float32)

    opa = opacities.view(-1).to(torch.float32).clamp(0, 1)
    sem = sem_probs.to(torch.float32)

    # Iterate offsets (small cube); gate per-Gaussian using its radius.
    offsets = []
    for dx in range(-int(max_radius), int(max_radius) + 1):
        for dy in range(-int(max_radius), int(max_radius) + 1):
            for dz in range(-int(max_radius), int(max_radius) + 1):
                offsets.append((dx, dy, dz))

    # Precompute fractional center positions (in voxel coordinates)
    cx = xyz[:, 0].to(torch.float32)
    cy = xyz[:, 1].to(torch.float32)
    cz = xyz[:, 2].to(torch.float32)

    for dx, dy, dz in offsets:
        # Keep only gaussians whose radius covers this offset
        off_ok = (r >= max(abs(dx), abs(dy), abs(dz)))
        if not off_ok.any():
            continue

        ix = (ix0 + dx)
        iy = (iy0 + dy)
        iz = (iz0 + dz)

        inside = off_ok & (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        if not inside.any():
            continue

        # Voxel center coordinates for this neighbor
        vx = (ix[inside].to(torch.float32) + 0.0)
        vy = (iy[inside].to(torch.float32) + 0.0)
        vz = (iz[inside].to(torch.float32) + 0.0)

        # Delta in voxel units
        dxv = (vx - cx[inside]) / s_vox[inside, 0]
        dyv = (vy - cy[inside]) / s_vox[inside, 1]
        dzv = (vz - cz[inside]) / s_vox[inside, 2]
        d2 = dxv * dxv + dyv * dyv + dzv * dzv

        # Gaussian kernel weight
        w = torch.exp(-0.5 * d2).to(torch.float32)
        ww = (opa[inside] * w).to(torch.float32)

        flat = _flat_index(ix[inside], iy[inside], iz[inside], Y=Y, Z=Z)
        sigma_sum.index_add_(0, flat, ww)
        sem_acc.index_add_(0, flat, ww[:, None] * sem[inside])

    sigma = 1.0 - torch.exp(-sigma_sum)
    sigma = sigma.clamp(0, 1)

    denom = sigma_sum.clamp(min=eps)
    sem_probs_vox = (sem_acc / denom[:, None]).t()  # (C, N)

    sigma_sum = sigma_sum.view(Z, Y, X)
    sigma = sigma.view(Z, Y, X)
    sem_probs_vox = sem_probs_vox.view(C, Z, Y, X)
    return sigma_sum, sigma, sem_probs_vox


def splat_features_to_voxels_gaussian_kernel(
    *,
    means_xyz: torch.Tensor,
    opacities: torch.Tensor,
    scales_xyz: torch.Tensor,
    feats: torch.Tensor,
    grid: VoxelGridSpec,
    max_radius: int = 3,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-aware Gaussian->Voxel splatting for arbitrary per-Gaussian features.

    This is used to approximate GaussianFormer Phase3 voxelization when sparse conv
    libraries are not available: we splat into a (potentially coarse) dense voxel grid
    and then run a small 3D CNN on the voxel features.

    Args:
      means_xyz: (P,3)
      opacities: (P,1)
      scales_xyz: (P,3)
      feats: (P,F)

    Returns:
      weight_sum: (Z,Y,X) accumulated weights per voxel (pre-squash)
      sigma: (Z,Y,X) = 1-exp(-weight_sum)
      feat_vox: (F,Z,Y,X) normalized by weight_sum
    """
    if means_xyz.dim() != 2 or means_xyz.shape[1] != 3:
        raise ValueError(f"Expected means_xyz (P,3), got {tuple(means_xyz.shape)}")
    if scales_xyz.shape != means_xyz.shape:
        raise ValueError("Expected scales_xyz shape == means_xyz shape")
    if opacities.dim() != 2 or opacities.shape[1] != 1:
        raise ValueError("Expected opacities (P,1)")
    if feats.dim() != 2 or feats.shape[0] != means_xyz.shape[0]:
        raise ValueError("Expected feats (P,F) with same P as means_xyz")

    device = means_xyz.device
    dtype = means_xyz.dtype

    origin = torch.tensor(grid.origin, device=device, dtype=dtype)
    voxel = float(grid.voxel_size)
    X, Y, Z = map(int, grid.dims_xyz)
    Fdim = int(feats.shape[1])

    xyz = (means_xyz - origin[None]) / voxel  # (P,3)
    ix0 = torch.round(xyz[:, 0]).to(torch.int64)
    iy0 = torch.round(xyz[:, 1]).to(torch.int64)
    iz0 = torch.round(xyz[:, 2]).to(torch.int64)

    s_vox = (scales_xyz / voxel).to(torch.float32).clamp(min=0.5)
    r = torch.ceil(s_vox.max(dim=1).values * 2.0).to(torch.int64)
    r = torch.clamp(r, min=1, max=int(max_radius))

    weight_sum = torch.zeros((Z * Y * X,), device=device, dtype=torch.float32)
    feat_sum = torch.zeros((Z * Y * X, Fdim), device=device, dtype=torch.float32)

    opa = opacities.view(-1).to(torch.float32).clamp(0, 1)
    feats_f = feats.to(torch.float32)

    offsets = []
    for dx in range(-int(max_radius), int(max_radius) + 1):
        for dy in range(-int(max_radius), int(max_radius) + 1):
            for dz in range(-int(max_radius), int(max_radius) + 1):
                offsets.append((dx, dy, dz))

    cx = xyz[:, 0].to(torch.float32)
    cy = xyz[:, 1].to(torch.float32)
    cz = xyz[:, 2].to(torch.float32)

    for dx, dy, dz in offsets:
        off_ok = (r >= max(abs(dx), abs(dy), abs(dz)))
        if not off_ok.any():
            continue

        ix = (ix0 + dx)
        iy = (iy0 + dy)
        iz = (iz0 + dz)

        inside = off_ok & (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        if not inside.any():
            continue

        vx = (ix[inside].to(torch.float32) + 0.0)
        vy = (iy[inside].to(torch.float32) + 0.0)
        vz = (iz[inside].to(torch.float32) + 0.0)

        dxv = (vx - cx[inside]) / s_vox[inside, 0]
        dyv = (vy - cy[inside]) / s_vox[inside, 1]
        dzv = (vz - cz[inside]) / s_vox[inside, 2]
        d2 = dxv * dxv + dyv * dyv + dzv * dzv

        w = torch.exp(-0.5 * d2).to(torch.float32)
        ww = (opa[inside] * w).to(torch.float32)

        flat = _flat_index(ix[inside], iy[inside], iz[inside], Y=Y, Z=Z)
        weight_sum.index_add_(0, flat, ww)
        feat_sum.index_add_(0, flat, ww[:, None] * feats_f[inside])

    sigma = 1.0 - torch.exp(-weight_sum)
    sigma = sigma.clamp(0, 1)

    denom = weight_sum.clamp(min=eps)
    feat_vox = (feat_sum / denom[:, None]).t().contiguous()  # (F,N)

    weight_sum = weight_sum.view(Z, Y, X)
    sigma = sigma.view(Z, Y, X)
    feat_vox = feat_vox.view(Fdim, Z, Y, X)
    return weight_sum, sigma, feat_vox
