from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from ssc_gs.models.gaussians import sample_points_uniform
from ssc_gs.utils.camera import kitti360_normK_to_pixelK


@dataclass
class MapAnythingInitConfig:
    hf_model_name: str = "facebook/map-anything-apache"
    cache_dir: str = "mapanything_cache"
    max_points: int = 20_000
    use_amp: bool = True
    amp_dtype: str = "bf16"
    local_files_only: bool = False


class MapAnythingInitializer:
    def __init__(self, cfg: MapAnythingInitConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._model: Optional[Any] = None

    def _lazy_load_model(self) -> Any:
        if self._model is None:
            try:
                from mapanything.models import MapAnything  # pyright: ignore[reportMissingImports]
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "MapAnything dependencies are not available (import failed). "
                    "Either install MapAnything's requirements (including its 'uniception' dependency), "
                    "or run this script in offline mode with precomputed caches (no --use_online_mapanything)."
                ) from e

            # Avoid surprising network access when running on offline machines.
            self._model = MapAnything.from_pretrained(self.cfg.hf_model_name, local_files_only=bool(self.cfg.local_files_only)).to(self.device)
        return self._model

    def cache_path(self, sample_index: int) -> Path:
        return Path(self.cfg.cache_dir) / f"{sample_index:08d}.npz"

    def load_cached_points(self, sample_index: int) -> Optional[torch.Tensor]:
        p = self.cache_path(sample_index)
        if not p.exists():
            return None
        data = np.load(p)
        pts = torch.from_numpy(data["points_world"]).float()
        return pts

    def save_cached_points(self, sample_index: int, points_world: torch.Tensor) -> None:
        p = self.cache_path(sample_index)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, points_world=points_world.detach().cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def infer_points_from_window(
        self,
        *,
        images_m11_chw: List[torch.Tensor],
        images_raw_01_chw: Optional[List[torch.Tensor]] = None,
        K_norm_list: List[torch.Tensor],
        camtoworld_list: List[torch.Tensor],
        image_paths: Optional[List[str]] = None,
        sample_index: int,
        force_recompute: bool = False,
        cached_only: bool = False,
    ) -> torch.Tensor:
        """Run MapAnything inference and return a merged point set in world frame.

        Args:
            images_m11_chw: list of (3,H,W) in [-1,1] (resized)
            images_raw_01_chw: list of (3,H_raw,W_raw) in [0,1] (no resize)
            K_norm_list: list of (3,3) normalized intrinsics from S4C
            camtoworld_list: list of (4,4)
            sample_index: used for cache naming

        Returns:
            points_world: (N,3)
        """
        if not force_recompute:
            cached = self.load_cached_points(sample_index)
            if cached is not None:
                return cached.to(self.device)

        if bool(cached_only):
            raise RuntimeError(
                f"MapAnything cached-only mode: missing cache for sample_index={sample_index}. "
                f"Expected file: {self.cache_path(sample_index)}. "
                "Either precompute caches, disable MapAnything init, or allow online model loading."
            )

        try:
            from mapanything.utils.image import load_images, preprocess_inputs  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "MapAnything utilities could not be imported. "
                "Install MapAnything dependencies or use cached initialization points."
            ) from e

        model = self._lazy_load_model()

        # Always keep external poses for cam->world mapping of predicted camera-frame points.
        c2w_list: list[torch.Tensor] = list(camtoworld_list)

        # Prefer raw image files when available.
        # SSCBench/S4C frequently resizes to (192,640) with aspect distortion.
        # MapAnything is sensitive to that and can collapse to near-field-only depth/points.
        if image_paths is not None and len(image_paths) > 0:
            processed = load_images(list(image_paths))
        elif images_raw_01_chw is not None:
            # Use raw tensors (no resize, [0, 1]) provided by dataset
            input_views = []
            for img_01, K_norm, c2w in zip(images_raw_01_chw, K_norm_list, camtoworld_list):
                # img_01 is (3, H, W) in [0, 1]
                # Convert to HWC [0, 255] uint8
                img_hwc = (img_01.permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8)

                H, W = img_01.shape[1], img_01.shape[2]
                K_px = kitti360_normK_to_pixelK(K_norm.float(), width=W, height=H)

                input_views.append(
                    {
                        "img": img_hwc.cpu().numpy(),
                        "intrinsics": K_px.cpu().numpy(),
                        "camera_poses": c2w.cpu().numpy(),
                        "is_metric_scale": True,
                    }
                )
            
            # Use fixed_size to avoid resizing to 518x...
            H_raw, W_raw = images_raw_01_chw[0].shape[1], images_raw_01_chw[0].shape[2]
            processed = preprocess_inputs(
                input_views, 
                verbose=False,
                resize_mode="fixed_size",
                size=(W_raw, H_raw)
            )
        else:
            input_views = []
            for img_m11, K_norm, c2w in zip(images_m11_chw, K_norm_list, camtoworld_list):
                # convert to HWC in [0,255] uint8
                img_01 = (img_m11.float() + 1.0) * 0.5
                img_hwc = (img_01.permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8)

                H, W = img_m11.shape[1], img_m11.shape[2]
                K_px = kitti360_normK_to_pixelK(K_norm.float(), width=W, height=H)

                input_views.append(
                    {
                        "img": img_hwc.cpu().numpy(),
                        "intrinsics": K_px.cpu().numpy(),
                        "camera_poses": c2w.cpu().numpy(),
                        "is_metric_scale": True,
                    }
                )

            processed = preprocess_inputs(input_views, verbose=False)
        outputs = model.infer(
            processed,
            memory_efficient_inference=True,
            ignore_calibration_inputs=False,
            ignore_pose_inputs=True,
            ignore_depth_inputs=False,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=False,
            use_amp=self.cfg.use_amp,
            amp_dtype=self.cfg.amp_dtype,
            apply_mask=True,
            mask_edges=True,
        )

        all_pts = []
        for view_idx, pred in enumerate(outputs):
            if "metric_scaling_factor" in pred:
                print(f"[DEBUG] View {view_idx} metric_scaling_factor: {pred['metric_scaling_factor']}")
            
            # IMPORTANT: MapAnything runs its own preprocessing (resize/crop) and returns depths/points
            # in that internal pixel space along with matched intrinsics.
            # To avoid any resolution/intrinsics mismatch, we use MapAnything's pts3d_cam directly
            # and transform it to the *provided* world frame using the input camtoworld.
            pts_cam_hw3 = pred["pts3d_cam"][0].to(self.device)  # (H',W',3) in camera frame
            mask_hw = pred["mask"][0].squeeze(-1).to(self.device).bool()  # (H',W')

            c2w = c2w_list[view_idx].to(self.device, dtype=pts_cam_hw3.dtype)
            ones = torch.ones((*pts_cam_hw3.shape[:2], 1), device=self.device, dtype=pts_cam_hw3.dtype)
            pts_cam_h = torch.cat([pts_cam_hw3, ones], dim=-1)  # (H',W',4)
            pts_world_hw4 = pts_cam_h @ c2w.t()
            pts_world_hw3 = pts_world_hw4[..., :3]

            pts, _ = sample_points_uniform(pts_world_hw3, mask_hw, self.cfg.max_points)
            all_pts.append(pts)

        points_world = torch.cat(all_pts, dim=0)
        self.save_cached_points(sample_index, points_world)
        return points_world
