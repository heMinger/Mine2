from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RefinerConfig:
    feat_dim: int = 256
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_classes: int = 20


class ImageCrossAttention(nn.Module):
    """Inject 2D features into per-Gaussian features via cross-attention.

    Minimal implementation:
      - Queries: Gaussian anchor features (N,F)
      - Keys/Values: flattened feature map (H*W,F)

    NOTE: This ignores geometric projection of Gaussians to pixels.
    For a full implementation, you typically sample per-Gaussian 2D features
    by projecting mean to the image plane and attending locally.
    """

    def __init__(self, feat_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)

    def forward(self, gauss_feat: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        """Args:
            gauss_feat: (N,F)
            feat_map: (B,F,h,w) or (F,h,w)
        Returns:
            updated_feat: (N,F)
        """
        if feat_map.dim() == 4:
            if feat_map.shape[0] != 1:
                raise ValueError("This minimal refiner expects B=1")
            feat_map = feat_map[0]
        if feat_map.dim() != 3:
            raise ValueError(f"Expected (F,h,w), got {tuple(feat_map.shape)}")

        Fdim, h, w = feat_map.shape
        kv = feat_map.reshape(Fdim, h * w).transpose(0, 1).unsqueeze(0)  # (1,HW,F)

        q = gauss_feat.unsqueeze(0)  # (1,N,F)
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        out = out.squeeze(0)
        return out


class RefinementHead(nn.Module):
    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        hidden = int(cfg.feat_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.feat_dim),
            nn.Linear(cfg.feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.feat_dim),
        )

        self.rgb_delta = nn.Linear(cfg.feat_dim, 3)
        self.sem_delta = nn.Linear(cfg.feat_dim, cfg.num_classes)
        self.opacity_delta = nn.Linear(cfg.feat_dim, 1)

    def forward(self, feat: torch.Tensor):
        x = feat + self.mlp(feat)
        return {
            "feat": x,
            "d_rgb": self.rgb_delta(x),
            "d_sem": self.sem_delta(x),
            "d_opacity": self.opacity_delta(x).squeeze(-1),
        }


class GaussianRefiner(nn.Module):
    """Minimal refiner: ICA -> head -> update RGB/sem/opacity (+ anchor features).

    This is the place to plug in GaussianFormer-style sparse 3D interaction.
    """

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        self.cfg = cfg
        self.ica = ImageCrossAttention(cfg.feat_dim, cfg.num_heads)
        self.head = RefinementHead(cfg)

    def forward(
        self,
        *,
        anchor_feat: torch.Tensor,
        dino_feat: torch.Tensor,
    ):
        injected = self.ica(anchor_feat, dino_feat)
        out = self.head(injected)
        return out
