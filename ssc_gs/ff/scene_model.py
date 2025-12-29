from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spconv.pytorch as spconv  # type: ignore
except Exception:  # pragma: no cover
    spconv = None

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from ssc_gs.models.dinov2 import DinoV2Config, DinoV2Frozen
from ssc_gs.ff.gaussian_to_voxel import (
    VoxelGridSpec,
    splat_features_to_voxels_gaussian_kernel,
    splat_gaussians_to_voxels_gaussian_kernel,
)

from ssc_gs.ff.deformable_ica import DeformableICA, DeformableICAConfig


@dataclass
class FFSceneConfig:
    num_classes: int = 20
    # dino_out_dim: int = 256
    dino_out_dim: int = 64

    # Self-encoding: per-token Gaussians
    num_depth_bins: int = 64
    depth_min: float = 1.0
    depth_max: float = 72.0

    min_scale: float = 0.05
    max_scale: float = 1.2

    # Cross-view fusion / refinement
    refine_hidden: int = 256
    fuse_hidden: int = 256

    # Phase3 self-encoding voxel CNN (dense fallback when sparse conv libs are unavailable)
    voxel_feat_dim: int = 32
    voxel_cnn_dim: int = 32
    voxel_kernel_radius: int = 2
    voxel_grid_voxel_size: float = 0.4
    voxel_grid_dims_xyz: Tuple[int, int, int] = (128, 128, 32)

    # Phase3 self-encoding backend
    use_sparse_conv: bool = True

    # ICA (Image Cross-Attention)
    ica_heads: int = 8  # kept for backward-compat; not used by DeformableICA

    # Phase3 refinement loop
    num_refine_layers: int = 3

    # Deformable ICA (GaussianFormer-like)
    ica_num_points: int = 7
    ica_levels: Tuple[str, ...] = ("f", "f2", "f4")


class Voxel3DUNetLite(nn.Module):
    def __init__(self, in_ch: int, hid: int):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(in_ch, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv3d(hid, hid, 3, stride=2, padding=1), nn.GroupNorm(8, hid), nn.GELU())
        self.mid = nn.Sequential(nn.Conv3d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU())
        self.dec = nn.Sequential(nn.Conv3d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1,C,Z,Y,X)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        xm = self.mid(x2)
        xu = F.interpolate(xm, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        return self.dec(xu + x1)


class SparseConv3DUNetLite(nn.Module):
    def __init__(self, in_ch: int, hid: int, out_ch: int):
        super().__init__()
        if spconv is None:
            raise RuntimeError("spconv is not available")

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, hid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hid),
            nn.ReLU(True),
            spconv.SubMConv3d(hid, hid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hid),
            nn.ReLU(True),
            spconv.SubMConv3d(hid, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ImageCrossAttention(nn.Module):
    """Deprecated.

    Kept only to avoid breaking older checkpoints that reference this symbol.
    The current Phase3 uses DeformableICA (projection + grid_sample) instead.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.q_ln = nn.LayerNorm(dim)
        self.kv_ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=int(heads), batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q_feat: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_ln(q_feat)[None]
        kv = self.kv_ln(kv_tokens)[None]
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = out[0]
        out = q_feat + out
        out = out + self.ff(out)
        return out


class CrossViewFusion(nn.Module):
    """Cross-view fusion module ("cross-attention" in the spec).

    Practical implementation:
      - For each Gaussian we sample a feature from each supervising view (geometric projection).
      - We average sampled features across views.
      - We fuse anchor and multiview features with an MLP to produce an updated feature.

    This behaves like a lightweight cross-attention without building a full HW token KV bank.
    """

    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, anchor_feat: torch.Tensor, mv_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([anchor_feat, mv_feat], dim=-1)
        return anchor_feat + self.mlp(x)


class RefinementHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
            nn.GELU(),
        )
        self.d_rgb = nn.Linear(feat_dim, 3)
        self.d_sem = nn.Linear(feat_dim, num_classes)
        self.d_opacity = nn.Linear(feat_dim, 1)
        self.d_scale = nn.Linear(feat_dim, 3)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.body(feat)
        return {
            "d_feat": x,
            "d_rgb": self.d_rgb(x),
            "d_sem": self.d_sem(x),
            "d_opacity": self.d_opacity(x).squeeze(-1),
            "d_scale": self.d_scale(x),
        }


class FFSceneModel(nn.Module):
    """Feed-forward scene Gaussian predictor with 3 stages:

    1) Self-encoding: predict Gaussians from keyframe DINO token grid.
    2) Cross-attention (fusion): inject multi-view features by projecting Gaussians.
    3) Refinement: predict residuals for RGB/semantic/opacity/scale.

    Output Gaussians are in WORLD coordinates (so rendering can use per-view camtoworld).
    """

    def __init__(self, cfg: FFSceneConfig):
        super().__init__()
        self.cfg = cfg

        self.dino = DinoV2Frozen(DinoV2Config(out_dim=cfg.dino_out_dim))
        Fdim = int(cfg.dino_out_dim)

        self.depth_head = nn.Conv2d(Fdim, cfg.num_depth_bins + 1, kernel_size=1)
        self.opacity_head = nn.Conv2d(Fdim, 1, kernel_size=1)
        self.scale_head = nn.Conv2d(Fdim, 3, kernel_size=1)
        self.rgb_head = nn.Conv2d(Fdim, 3, kernel_size=1)
        self.sem_head = nn.Conv2d(Fdim, cfg.num_classes, kernel_size=1)

        # Phase3 ICA (GaussianFormer-like): deformable projection sampling from multi-view, multi-level feature maps.
        self.ica = DeformableICA(
            DeformableICAConfig(
                feat_dim=Fdim,
                num_views_max=8,
                num_levels=len(tuple(getattr(cfg, "ica_levels", ("f", "f2", "f4")))),
                num_points=int(getattr(cfg, "ica_num_points", 7)),
                hidden=int(cfg.fuse_hidden),
            )
        )
        self.refine = RefinementHead(feat_dim=Fdim, hidden=int(cfg.refine_hidden), num_classes=cfg.num_classes)

        # Phase3 self-encoding (voxelize -> 3D CNN -> devoxelize).
        self.voxel_feat_in = nn.Sequential(nn.LayerNorm(Fdim), nn.Linear(Fdim, int(cfg.voxel_feat_dim)))
        self.voxel_cnn = Voxel3DUNetLite(in_ch=1 + int(cfg.voxel_feat_dim), hid=int(cfg.voxel_cnn_dim))
        self.sparse_cnn = None
        if spconv is not None:
            # sparse input: [opacity, voxel_feat_in(anchor_feat)] at Gaussian center voxels
            self.sparse_cnn = SparseConv3DUNetLite(
                in_ch=1 + int(cfg.voxel_feat_dim),
                hid=int(cfg.voxel_cnn_dim),
                out_ch=int(cfg.voxel_cnn_dim),
            )
        self.voxel_feat_out = nn.Sequential(nn.LayerNorm(int(cfg.voxel_cnn_dim)), nn.Linear(int(cfg.voxel_cnn_dim), Fdim))

        depth_bins = torch.linspace(cfg.depth_min, cfg.depth_max, cfg.num_depth_bins)
        self.register_buffer("depth_bins", depth_bins, persistent=False)

    @staticmethod
    def _make_uv_grid_norm(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        v = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        uu = u[None, :].expand(h, w)
        vv = v[:, None].expand(h, w)
        uv = torch.stack([uu * 2.0 - 1.0, vv * 2.0 - 1.0], dim=-1)  # (h,w,2)
        ones = torch.ones((h, w, 1), device=device, dtype=dtype)
        return torch.cat([uv, ones], dim=-1)  # (h,w,3)

    def _self_encode(
        self,
        *,
        img_m11_chw: torch.Tensor,
        K_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = img_m11_chw.device
        dtype = img_m11_chw.dtype

        img_01 = ((img_m11_chw[None] + 1.0) * 0.5).clamp(0, 1)  # (1,3,H,W)
        feat_pyr = self.dino(img_01, return_pyramid=True)
        feat = feat_pyr["f"]  # (1,F,h,w)
        _, Fdim, h, w = feat.shape

        depth_logits = self.depth_head(feat)  # (1,D+1,h,w)
        opacity_logits = self.opacity_head(feat)  # (1,1,h,w)
        scale_logits = self.scale_head(feat)  # (1,3,h,w)
        rgb_logits = self.rgb_head(feat)  # (1,3,h,w)
        sem_logits = self.sem_head(feat)  # (1,C,h,w)

        depth_p = F.softmax(depth_logits, dim=1)
        p_valid = depth_p[:, : self.cfg.num_depth_bins]
        p_empty = depth_p[:, self.cfg.num_depth_bins :]
        depth = torch.sum(p_valid * self.depth_bins.view(1, -1, 1, 1), dim=1, keepdim=True)  # (1,1,h,w)

        # Unprojection in normalized coordinates
        K_inv = torch.inverse(K_norm.to(device=device, dtype=dtype))
        uvd = self._make_uv_grid_norm(h, w, device=device, dtype=dtype).view(-1, 3)  # (P,3)
        dirs = (K_inv @ uvd.t()).t()  # (P,3)

        means_cam = dirs * depth.view(-1, 1)  # (P,3)

        opacities = torch.sigmoid(opacity_logits).view(-1, 1) * (1.0 - p_empty.view(-1, 1))

        scales01 = torch.sigmoid(scale_logits).permute(0, 2, 3, 1).reshape(-1, 3)
        scales = self.cfg.min_scale + (self.cfg.max_scale - self.cfg.min_scale) * scales01

        rgb = torch.sigmoid(rgb_logits).permute(0, 2, 3, 1).reshape(-1, 3)
        sem = sem_logits.permute(0, 2, 3, 1).reshape(-1, self.cfg.num_classes)

        # Anchor features per Gaussian (token feature vector)
        anchor_feat = feat.permute(0, 2, 3, 1).reshape(-1, Fdim)

        token_hw = torch.tensor([h, w], device=device, dtype=torch.int64)
        return {
            "feat_pyr": feat_pyr,
            "pred_depth": depth,  # (1,1,h,w)
            "anchor_feat": anchor_feat,  # (P,F)
            "means_cam": means_cam,
            "opacities": opacities,
            "scales": scales,
            "rgb": rgb,
            "sem_logits": sem,
            "token_hw": token_hw,
        }

    @staticmethod
    def _sample_feat_at_grid(
        feat: torch.Tensor,
        grid: torch.Tensor,
        *,
        align_corners: bool = False,
    ) -> torch.Tensor:
        """Sample a feature map at point locations.

        Args:
            feat: (1,C,h,w)
            grid: (1,1,P,2) in [-1,1]

        Returns:
            (P,C)
        """
        if feat.dim() != 4 or feat.shape[0] != 1:
            raise ValueError(f"Expected feat (1,C,h,w), got {tuple(feat.shape)}")
        if grid.dim() != 4 or grid.shape[0] != 1 or grid.shape[1] != 1 or grid.shape[-1] != 2:
            raise ValueError(f"Expected grid (1,1,P,2), got {tuple(grid.shape)}")

        # grid_sample expects (N,Hout,Wout,2). We treat points as Wout=P.
        grid_gs = grid.permute(0, 2, 1, 3).contiguous()  # (1,P,1,2)
        samp = F.grid_sample(feat, grid_gs, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
        # samp: (1,C,P,1)
        return samp[0, :, :, 0].t().contiguous()  # (P,C)

    @staticmethod
    def _project_world_to_view_grid(
        *,
        means_world: torch.Tensor,
        camtoworld: torch.Tensor,
        K_norm: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project world points to normalized image grid coords for grid_sample.

        Returns:
          grid: (1,1,P,2) in [-1,1]
          valid: (P,) bool
        """
        worldtocam = torch.linalg.inv(camtoworld)
        ones = torch.ones((means_world.shape[0], 1), device=means_world.device, dtype=means_world.dtype)
        pw = torch.cat([means_world, ones], dim=1)  # (P,4)
        pc = (worldtocam @ pw.t()).t()[:, :3]

        z = pc[:, 2].clamp(min=eps)
        x = pc[:, 0] / z
        y = pc[:, 1] / z

        # normalized pixel coords using normalized intrinsics
        u = K_norm[0, 0] * x + K_norm[0, 2]
        v = K_norm[1, 1] * y + K_norm[1, 2]

        valid = (pc[:, 2] > eps) & (u >= -1.5) & (u <= 1.5) & (v >= -1.5) & (v <= 1.5)
        grid = torch.stack([u, v], dim=-1).view(1, 1, -1, 2)
        return grid, valid

    # Note: multiview feature fusion was a practical earlier approximation.
    # The report's Phase3 uses ICA (image cross-attention) with the keyframe's DINO features,
    # which we implement via ImageCrossAttention.

    @staticmethod
    def _sample_voxel_context_trilinear(
        *,
        means_velo: torch.Tensor,
        sigma: torch.Tensor,
        sem_probs_vox: torch.Tensor,
        grid: VoxelGridSpec,
    ) -> torch.Tensor:
        """Sample voxel context at point locations (pure PyTorch trilinear).

        Args:
          means_velo: (P,3) in velodyne coords
          sigma: (Z,Y,X)
          sem_probs_vox: (C,Z,Y,X)

        Returns:
          ctx: (P, C+1) = [sigma, sem_probs]
        """
        device = means_velo.device
        dtype = means_velo.dtype
        origin = torch.tensor(grid.origin, device=device, dtype=dtype)
        voxel = float(grid.voxel_size)
        X, Y, Z = map(int, grid.dims_xyz)

        # Continuous coords
        xyz = (means_velo - origin[None]) / voxel
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

        C = int(sem_probs_vox.shape[0])
        ctx = torch.zeros((means_velo.shape[0], C + 1), device=device, dtype=torch.float32)

        # helper to gather voxel values safely
        def gather(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor):
            inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
            if not inside.any():
                return inside, None, None
            sig_v = sigma[iz[inside], iy[inside], ix[inside]].to(torch.float32)  # (n,)
            sem_v = sem_probs_vox[:, iz[inside], iy[inside], ix[inside]].to(torch.float32).t()  # (n,C)
            return inside, sig_v, sem_v

        for dx in (0, 1):
            wx = (1 - fx) if dx == 0 else fx
            ix = x0 + dx
            for dy in (0, 1):
                wy = (1 - fy) if dy == 0 else fy
                iy = y0 + dy
                for dz in (0, 1):
                    wz = (1 - fz) if dz == 0 else fz
                    iz = z0 + dz
                    w = (wx * wy * wz).to(torch.float32)
                    inside, sig_v, sem_v = gather(ix, iy, iz)
                    if sig_v is None:
                        continue
                    ctx[inside, 0] += w[inside] * sig_v
                    ctx[inside, 1:] += w[inside, None] * sem_v

        return ctx

    @staticmethod
    def _sparse_unique_voxels(
        *,
        ix: torch.Tensor,
        iy: torch.Tensor,
        iz: torch.Tensor,
        feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge duplicate voxel indices by averaging features.

        Returns:
          uniq_xyz: (M,3) int64 (x,y,z)
          uniq_feat: (M,C) float32
          inv: (P,) int64 mapping each original point -> unique row
          counts: (M,) float32
        """
        xyz = torch.stack([ix, iy, iz], dim=1).to(torch.int64)
        uniq_xyz, inv = torch.unique(xyz, dim=0, return_inverse=True)
        M = int(uniq_xyz.shape[0])
        C = int(feats.shape[1])

        uniq_feat = torch.zeros((M, C), device=feats.device, dtype=torch.float32)
        counts = torch.zeros((M,), device=feats.device, dtype=torch.float32)

        feats_f = feats.to(torch.float32)
        uniq_feat.index_add_(0, inv, feats_f)
        counts.index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
        uniq_feat = uniq_feat / counts.clamp(min=1.0)[:, None]
        return uniq_xyz, uniq_feat, inv, counts

    def forward(
        self,
        *,
        imgs_m11_chw: List[torch.Tensor],
        Ks_norm: List[torch.Tensor],
        camtoworlds: List[torch.Tensor],
        encoder_index: int = 0,
        supervise_indices: Optional[List[int]] = None,
        # Optional external initialization (e.g., MapAnything point cloud in WORLD frame).
        init_points_world: Optional[torch.Tensor] = None,
        # Self-encoding Gaussian-to-Voxel (GaussianFormer-like) to build voxel context.
        T_velo_to_cam: Optional[torch.Tensor] = None,
        voxel_grid: Optional[VoxelGridSpec] = None,
        voxel_max_radius: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """Predict a scene Gaussian set in WORLD coordinates.

        During training, you should pass multiple views and set supervise_indices.
        During evaluation/inference, you can pass a single image (len==1).
        """
        if supervise_indices is None:
            supervise_indices = [j for j in range(len(imgs_m11_chw)) if j != encoder_index]

        img0 = imgs_m11_chw[encoder_index]
        K0 = Ks_norm[encoder_index]
        c2w0 = camtoworlds[encoder_index]

        # Always compute keyframe DINO pyramid (used by ICA and for initializing per-point features).
        enc = self._self_encode(img_m11_chw=img0, K_norm=K0)

        ones = None
        pred_depth = enc["pred_depth"]

        if init_points_world is None:
            means_cam = enc["means_cam"]
            ones = torch.ones((means_cam.shape[0], 1), device=means_cam.device, dtype=means_cam.dtype)
            means_world = (c2w0 @ torch.cat([means_cam, ones], dim=1).t()).t()[:, :3]
            anchor_feat = enc["anchor_feat"]
            rgb = enc["rgb"]
            sem_logits = enc["sem_logits"]
            opacities = enc["opacities"].view(-1)
            scales = enc["scales"]
        else:
            # External init: points are already in WORLD coords.
            means_world = init_points_world.to(device=img0.device, dtype=img0.dtype)
            if means_world.dim() != 2 or means_world.shape[1] != 3:
                raise ValueError(f"Expected init_points_world (P,3), got {tuple(means_world.shape)}")

            P = int(means_world.shape[0])
            if P == 0:
                raise ValueError("init_points_world is empty")

            # Initialize per-point anchor features by projecting into the keyframe and sampling DINO features.
            feat0 = enc["feat_pyr"]["f"]  # (1,F,h,w)
            grid0, valid0 = self._project_world_to_view_grid(means_world=means_world, camtoworld=c2w0, K_norm=K0)
            anchor_feat = self._sample_feat_at_grid(feat0, grid0, align_corners=False)
            if valid0 is not None:
                anchor_feat = anchor_feat * valid0.to(anchor_feat.dtype)[:, None]

            # Initialize RGB by sampling the keyframe image (in [0,1]).
            img0_01 = ((img0[None] + 1.0) * 0.5).clamp(0, 1)
            rgb = self._sample_feat_at_grid(img0_01, grid0, align_corners=False).clamp(0, 1)

            # Conservative defaults for other attributes; refinement will learn to adjust.
            sem_logits = torch.zeros((P, self.cfg.num_classes), device=means_world.device, dtype=means_world.dtype)
            opacities = torch.full((P,), 0.1, device=means_world.device, dtype=means_world.dtype)
            scales = torch.full((P, 3), float(self.cfg.min_scale), device=means_world.device, dtype=means_world.dtype)

            # Compute camera-frame centers for optional voxel interaction.
            worldtocam0 = torch.linalg.inv(c2w0)
            ones = torch.ones((P, 1), device=means_world.device, dtype=means_world.dtype)
            means_cam = (worldtocam0 @ torch.cat([means_world, ones], dim=1).t()).t()[:, :3]

        # Build feature pyramids for all provided views (multi-view, multi-level deformable ICA).
        # DINO is frozen; this is the main added compute for GaussianFormer-like ICA.
        feats_by_view: List[List[torch.Tensor]] = []
        level_names = tuple(getattr(self.cfg, "ica_levels", ("f", "f2", "f4")))
        for j in range(len(imgs_m11_chw)):
            if j == encoder_index:
                pyr = enc["feat_pyr"]
            else:
                img_01 = ((imgs_m11_chw[j][None] + 1.0) * 0.5).clamp(0, 1)
                pyr = self.dino(img_01, return_pyramid=True)
            feats_by_view.append([pyr[name] for name in level_names])

        # Multi-layer refinement loop (GaussianFormer-style stacking).
        voxel_sigma_mean = torch.tensor(0.0, device=means_cam.device)

        num_layers = int(getattr(self.cfg, "num_refine_layers", 1))
        if num_layers < 1:
            num_layers = 1

        if T_velo_to_cam is not None and voxel_grid is None:
            voxel_grid = VoxelGridSpec(voxel_size=float(self.cfg.voxel_grid_voxel_size), dims_xyz=tuple(self.cfg.voxel_grid_dims_xyz))

        means_velo = None
        if T_velo_to_cam is not None:
            T_velo_to_cam = T_velo_to_cam.to(device=means_cam.device, dtype=means_cam.dtype)
            cam_to_velo = torch.linalg.inv(T_velo_to_cam)
            means_velo = (cam_to_velo @ torch.cat([means_cam, ones], dim=1).t()).t()[:, :3]

        for _layer in range(num_layers):
            # (A) Self-encoding 3D interaction (voxelize -> 3D CNN -> devoxelize) if calibration is available.
            if means_velo is not None and voxel_grid is not None:
                use_sparse = bool(getattr(self.cfg, "use_sparse_conv", True)) and (self.sparse_cnn is not None) and (spconv is not None)

                v_in = self.voxel_feat_in(anchor_feat)  # (P,Fv)
                in_feat = torch.cat([opacities.view(-1, 1).to(v_in.dtype), v_in], dim=1)  # (P,1+Fv)

                if use_sparse:
                    # Convert velodyne coords to voxel indices.
                    origin = torch.tensor(voxel_grid.origin, device=means_velo.device, dtype=means_velo.dtype)
                    voxel = float(voxel_grid.voxel_size)
                    X, Y, Z = map(int, voxel_grid.dims_xyz)
                    xyz = (means_velo - origin[None]) / voxel
                    ix = torch.round(xyz[:, 0]).to(torch.int64)
                    iy = torch.round(xyz[:, 1]).to(torch.int64)
                    iz = torch.round(xyz[:, 2]).to(torch.int64)
                    inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)

                    if inside.any():
                        uniq_xyz, uniq_feat, inv, _counts = self._sparse_unique_voxels(
                            ix=ix[inside], iy=iy[inside], iz=iz[inside], feats=in_feat[inside]
                        )
                        # spconv indices are (batch, z, y, x)
                        batch = torch.zeros((uniq_xyz.shape[0], 1), device=uniq_xyz.device, dtype=torch.int32)
                        idx_zyx = torch.stack(
                            [uniq_xyz[:, 2].to(torch.int32), uniq_xyz[:, 1].to(torch.int32), uniq_xyz[:, 0].to(torch.int32)], dim=1
                        )
                        indices = torch.cat([batch, idx_zyx], dim=1).contiguous()
                        spatial_shape = (Z, Y, X)
                        st = spconv.SparseConvTensor(
                            features=uniq_feat.to(torch.float32),
                            indices=indices,
                            spatial_shape=spatial_shape,
                            batch_size=1,
                        )
                        out = self.sparse_cnn(st)
                        out_feat_uniq = out.features  # (M,Cv)

                        ctx = torch.zeros((means_velo.shape[0], out_feat_uniq.shape[1]), device=means_velo.device, dtype=out_feat_uniq.dtype)
                        ctx_inside = out_feat_uniq[inv]
                        ctx[inside] = ctx_inside
                        anchor_feat = anchor_feat + self.voxel_feat_out(ctx)
                        voxel_sigma_mean = opacities.mean()
                    else:
                        voxel_sigma_mean = opacities.mean()
                else:
                    weight_sum, sigma, feat_vox = splat_features_to_voxels_gaussian_kernel(
                        means_xyz=means_velo,
                        opacities=opacities.view(-1, 1),
                        scales_xyz=scales,
                        feats=v_in,
                        grid=voxel_grid,
                        max_radius=int(getattr(self.cfg, "voxel_kernel_radius", voxel_max_radius)),
                    )
                    voxel_sigma_mean = weight_sum.mean()
                    vol_in = torch.cat([sigma[None], feat_vox], dim=0)[None]  # (1,1+Fv,Z,Y,X)
                    vol_out = self.voxel_cnn(vol_in)[0]  # (Cv,Z,Y,X)
                    ctx_c = self._sample_voxel_context_trilinear(
                        means_velo=means_velo,
                        sigma=torch.zeros_like(sigma),
                        sem_probs_vox=vol_out.contiguous(),
                        grid=voxel_grid,
                    )[:, 1:]  # drop dummy sigma => (P,Cv)
                    anchor_feat = anchor_feat + self.voxel_feat_out(ctx_c)

            # (B) ICA: deformable aggregation from multi-view, multi-level image feature maps.
            anchor_feat = self.ica(
                instance_feat=anchor_feat,
                means_world=means_world,
                scales_xyz=scales,
                feats_by_view=feats_by_view,
                Ks_norm=Ks_norm,
                camtoworlds=camtoworlds,
            )

            # (C) Refinement head: update Gaussian attributes + feature.
            ref = self.refine(anchor_feat)
            anchor_feat = anchor_feat + ref["d_feat"]
            rgb = (rgb + ref["d_rgb"]).clamp(0, 1)
            sem_logits = sem_logits + ref["d_sem"]
            opacities = (opacities + ref["d_opacity"]).sigmoid().view(-1)
            scales = scales * torch.exp(ref["d_scale"].clamp(min=-2.0, max=2.0))
            scales = scales.clamp(min=self.cfg.min_scale, max=self.cfg.max_scale)

        mv_feat = torch.zeros_like(anchor_feat)

        # Identity quaternion in gsplat convention (wxyz)
        quats = torch.zeros((means_world.shape[0], 4), device=means_world.device, dtype=means_world.dtype)
        quats[:, 0] = 1.0

        return {
            "means_world": means_world,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "rgb": rgb,
            "sem_logits": sem_logits,
            "anchor_feat": anchor_feat,
            "mv_feat": mv_feat,
            "voxel_sigma_mean": voxel_sigma_mean,
            "pred_depth": pred_depth,
        }
