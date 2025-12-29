from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DinoV2Config:
    model_name: str = "dinov2_vitb14_reg"  # also: dinov2_vitl14_reg
    dino_dim: Optional[int] = None
    out_dim: int = 256
    patch_size: int = 14
    use_amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "bf16"


class DinoV2Frozen(nn.Module):
    """Frozen DINOv2 feature extractor + lightweight adapter.

    Notes:
      - DINOv2 via torch.hub returns token embeddings; we reshape patch tokens to a feature map.
      - Adapter maps to `out_dim` and (optionally) builds pseudo multi-scale via pooling.

    This module is designed to be *frozen* by default.
    """

    def __init__(self, cfg: DinoV2Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = torch.hub.load("facebookresearch/dinov2", cfg.model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Infer DINO embedding dim without running a forward pass.
        # A CPU forward here can fail when xFormers kernels are enabled (they require CUDA/half types).
        dino_dim = cfg.dino_dim
        if dino_dim is None:
            dino_dim = getattr(self.backbone, "embed_dim", None)
        if dino_dim is None:
            dino_dim = getattr(self.backbone, "dim", None)
        if dino_dim is None:
            norm = getattr(self.backbone, "norm", None)
            if norm is not None and hasattr(norm, "normalized_shape") and norm.normalized_shape:
                dino_dim = int(norm.normalized_shape[0])
        if dino_dim is None:
            raise RuntimeError(
                "Could not infer DINOv2 embedding dim without a forward pass. "
                "Set DinoV2Config(dino_dim=...) explicitly for your model."
            )

        self.adapter = nn.Sequential(
            nn.Conv2d(dino_dim, cfg.out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=cfg.out_dim),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _extract_tokens(dino_out: torch.Tensor | dict) -> torch.Tensor:
        """Normalize DINOv2 outputs to a token tensor (B, T, D)."""
        if torch.is_tensor(dino_out):
            return dino_out
        if isinstance(dino_out, dict):
            # DINOv2 models commonly return a dict from forward_features.
            for key in ("x_norm_patchtokens", "x_norm_patch_tokens", "patchtokens", "x_norm_tokens"):
                v = dino_out.get(key)
                if torch.is_tensor(v) and v.dim() == 3:
                    return v
            # Fallback: pick the first 3D tensor value.
            for v in dino_out.values():
                if torch.is_tensor(v) and v.dim() == 3:
                    return v
        raise RuntimeError(f"Unexpected DINOv2 output type/structure: {type(dino_out)}")

    @torch.no_grad()
    def forward_tokens(self, x_01: torch.Tensor) -> torch.Tensor:
        """Return token features (B, T, D). Input expected in [0,1]."""
        return self._extract_tokens(self.backbone(x_01, is_training=True))

    def forward(self, x_01: torch.Tensor, *, return_pyramid: bool = True):
        """Extract DINOv2 features.

        Args:
            x_01: (B,3,H,W) float in [0,1].
            return_pyramid: if True, returns dict with f{1,2,4} scales.

        Returns:
            If return_pyramid:
                {"f": (B,C,h,w), "f2": (B,C,h/2,w/2), "f4": (B,C,h/4,w/4)}
            else:
                (B,C,h,w)
        """
        B, _, H, W = x_01.shape

        # DINOv2 ViT requires spatial dims divisible by patch size.
        # Our training images can be arbitrary; pad (replicate) to the next multiple.
        p = self.cfg.patch_size
        Hp = int(math.ceil(H / p) * p)
        Wp = int(math.ceil(W / p) * p)
        if Hp != H or Wp != W:
            x_01 = F.pad(x_01, (0, Wp - W, 0, Hp - H), mode="replicate")

        # DINO expects ImageNet-style normalization; the hub model typically handles it internally
        # when using its provided preprocessing. Here we rely on the model's internal behavior.
        # If you want explicit normalization, add it here.
        amp_enabled = self.cfg.use_amp and x_01.is_cuda
        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                tokens = self._extract_tokens(self.backbone(x_01, is_training=True))  # (B, T, D)

        # Heuristic: treat tokens excluding possible class/reg tokens as patch tokens.
        # Most dinov2 models output [cls, reg, ...patches]. We'll drop first 1-2 tokens if needed.
        T = tokens.shape[1]
        # Compute patch grid size (using padded H/W)
        ph = x_01.shape[-2] // self.cfg.patch_size
        pw = x_01.shape[-1] // self.cfg.patch_size
        expected_patches = ph * pw

        if T == expected_patches + 1:
            patch_tokens = tokens[:, 1:, :]
        elif T == expected_patches + 2:
            patch_tokens = tokens[:, 2:, :]
        elif T == expected_patches:
            patch_tokens = tokens
        else:
            # Fall back: take last expected_patches tokens
            patch_tokens = tokens[:, -expected_patches:, :]

        feat = patch_tokens.transpose(1, 2).reshape(B, -1, ph, pw)  # (B,D,ph,pw)
        feat = self.adapter(feat)  # (B,C,ph,pw)

        if not return_pyramid:
            return feat

        return {
            "f": feat,
            "f2": F.avg_pool2d(feat, kernel_size=2, stride=2, ceil_mode=True),
            "f4": F.avg_pool2d(feat, kernel_size=4, stride=4, ceil_mode=True),
        }
