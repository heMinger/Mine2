from __future__ import annotations

import argparse

import torch

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from ssc_gs.data.kitti360_split import ensure_kitti360_split_file
from ssc_gs.data.s4c_kitti360 import S4CKitti360KeyframeDataset
from ssc_gs.initialize.mapanything_init import MapAnythingInitConfig, MapAnythingInitializer
from ssc_gs.losses import LossConfig
from ssc_gs.models.gaussians import GaussianScene, init_gaussians_from_points
from ssc_gs.models.refiner import GaussianRefiner, RefinerConfig
from ssc_gs.render.gsplat_renderer import GSplatRenderer
from ssc_gs.train import TrainConfig, train_step


class DummyDino:
    def __init__(self, feat_dim: int, patch_size: int = 14):
        self.feat_dim = feat_dim
        self.patch_size = patch_size

    def to(self, device):
        return self

    def __call__(self, x_01: torch.Tensor, *, return_pyramid: bool = False) -> torch.Tensor:
        b, _, h, w = x_01.shape
        ph = max(1, h // self.patch_size)
        pw = max(1, w // self.patch_size)
        return torch.zeros((b, self.feat_dim, ph, pw), device=x_01.device, dtype=x_01.dtype)


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test: load cache + run 1 train step (dummy DINO).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cache_dir", type=str, default="/data/lmh/ssc_gs_pointcloud_cache")
    p.add_argument("--max_points", type=int, default=50_000)
    args = p.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    split_file = ensure_kitti360_split_file(
        pseudo_seg_root="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr",
        pose_root="/data/lmh_data/KITTI360/poses",
        out_path="/home/lmh/VGM-GS-S4C/ssc_gs/splits/kitti360_panoptic_deeplab_train_files.txt",
    )

    ds = Kitti360Dataset(
        data_path="/data/lmh_data/KITTI360",
        pose_path="/data/lmh_data/KITTI360/poses",
        split_path=split_file,
        target_image_size=(192, 640),
        return_stereo=False,
        return_fisheye=False,
        return_depth=False,
        return_segmentation=True,
        segmentation_mode="panoptic_deeplab",
        data_segmentation_path="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr",
        frame_count=1,
        is_preprocessed=False,
    )
    key_ds = S4CKitti360KeyframeDataset(ds, keyframe_idx=0, has_pseudo_sem=True)

    sample = key_ds[0]

    init_cfg = MapAnythingInitConfig(cache_dir=args.cache_dir, max_points=args.max_points)
    initializer = MapAnythingInitializer(init_cfg, device=device)

    cached = initializer.load_cached_points(sample.index)
    if cached is None:
        raise RuntimeError(f"Missing cache for index={sample.index} under {args.cache_dir}")

    pts = cached.to(device)

    cfg = TrainConfig(
        kitti360_data_path="/data/lmh_data/KITTI360",
        kitti360_pose_path="/data/lmh_data/KITTI360/poses",
        pseudo_seg_path="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr",
        split_file=split_file,
        init_max_points=args.max_points,
        device=str(device),
        steps=1,
    )

    init_tensors = init_gaussians_from_points(
        pts,
        num_classes=cfg.num_classes,
        feat_dim=cfg.feat_dim,
        init_scale=0.03,
        init_opacity=0.1,
        device=device,
    )
    scene = GaussianScene(init_tensors).to(device)

    refiner = GaussianRefiner(RefinerConfig(feat_dim=cfg.feat_dim, num_classes=cfg.num_classes)).to(device)
    dino = DummyDino(cfg.feat_dim)

    h, w = sample.image.shape[1:]
    renderer = GSplatRenderer(width=w, height=h, packed=cfg.packed)

    opt = torch.optim.Adam(list(scene.parameters()) + list(refiner.parameters()), lr=cfg.lr)
    loss_cfg = LossConfig(ssim_lambda=0.2, sem_lambda=1.0)

    stats = train_step(
        scene=scene,
        refiner=refiner,
        dino=dino,
        renderer=renderer,
        loss_cfg=loss_cfg,
        sample=sample,
        optimizer=opt,
    )

    print("smoke_train: PASS", stats)


if __name__ == "__main__":
    main()
