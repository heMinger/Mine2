from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from gsplat.rendering import rasterization

from ssc_gs.utils.camera import camtoworld_to_viewmat


@dataclass
class RenderOutputs:
    rgb: torch.Tensor  # (H,W,3) in [0,1]
    alpha: torch.Tensor  # (H,W,1) in [0,1]
    sem_logits: Optional[torch.Tensor] = None  # (H,W,C)
    meta_rgb: Optional[Dict] = None
    meta_sem: Optional[Dict] = None


class GSplatRenderer:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        packed: bool = True,
        antialiased: bool = False,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        channel_chunk: int = 32,
        camera_model: str = "pinhole",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.packed = packed
        self.antialiased = antialiased
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.channel_chunk = int(channel_chunk)
        self.camera_model = camera_model

    def render(
        self,
        *,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        rgb: torch.Tensor,
        camtoworld: torch.Tensor,
        K_px: torch.Tensor,
        sem_logits: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> RenderOutputs:
        """Render RGB (+ optional semantic logits) for a single camera."""
        if camtoworld.shape != (4, 4):
            raise ValueError(f"Expected camtoworld (4,4), got {tuple(camtoworld.shape)}")
        if K_px.shape != (3, 3):
            raise ValueError(f"Expected K_px (3,3), got {tuple(K_px.shape)}")

        W = int(width) if width is not None else self.width
        H = int(height) if height is not None else self.height

        viewmats = camtoworld_to_viewmat(camtoworld)[None]  # (1,4,4)
        Ks = K_px[None]  # (1,3,3)

        rasterize_mode = "antialiased" if self.antialiased else "classic"

        rgb_img, alpha, meta_rgb = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=rgb,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            packed=self.packed,
            rasterize_mode=rasterize_mode,
            camera_model=self.camera_model,
            render_mode="RGB",
            channel_chunk=self.channel_chunk,
        )

        # Output is (C,H,W,D)
        rgb_img = rgb_img[0].clamp(0, 1)
        alpha = alpha[0].clamp(0, 1)

        sem_out = None
        meta_sem = None
        if sem_logits is not None:
            sem_img, alpha2, meta_sem = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=sem_logits,
                viewmats=viewmats,
                Ks=Ks,
                width=W,
                height=H,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                packed=self.packed,
                rasterize_mode=rasterize_mode,
                camera_model=self.camera_model,
                render_mode="RGB",
                channel_chunk=self.channel_chunk,
            )
            # sem_img: (1,H,W,C)
            sem_out = sem_img[0]
            # Use alpha from first pass.
            _ = alpha2

        return RenderOutputs(rgb=rgb_img, alpha=alpha, sem_logits=sem_out, meta_rgb=meta_rgb, meta_sem=meta_sem)
