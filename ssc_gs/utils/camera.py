from __future__ import annotations

import torch


def kitti360_normK_to_pixelK(K_norm: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert S4C/KITTI-360 normalized intrinsics ([-1,1] NDC) to pixel intrinsics.

    S4C normalizes intrinsics in `Kitti360Dataset._load_calibs`:
      fx_norm = (fx / W) * 2
      cx_norm = (cx / W) * 2 - 1
    and similarly for y with H.

    Args:
        K_norm: (..., 3, 3) intrinsics in normalized coordinates.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        (..., 3, 3) intrinsics in pixel coordinates.
    """
    if K_norm.shape[-2:] != (3, 3):
        raise ValueError(f"Expected K_norm[...,3,3], got {tuple(K_norm.shape)}")

    K = K_norm.clone()
    K[..., 0, 0] = K_norm[..., 0, 0] * (width / 2.0)
    K[..., 1, 1] = K_norm[..., 1, 1] * (height / 2.0)
    K[..., 0, 2] = (K_norm[..., 0, 2] + 1.0) * (width / 2.0)
    K[..., 1, 2] = (K_norm[..., 1, 2] + 1.0) * (height / 2.0)
    return K


def camtoworld_to_viewmat(camtoworld: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world (c2w) to world-to-camera (view matrix)."""
    return torch.linalg.inv(camtoworld)
