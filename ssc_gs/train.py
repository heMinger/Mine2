from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from ssc_gs.data.s4c_kitti360 import S4CKitti360KeyframeDataset
from ssc_gs.data.kitti360_split import ensure_kitti360_split_file
from ssc_gs.initialize.mapanything_init import MapAnythingInitConfig, MapAnythingInitializer
from ssc_gs.losses import LossConfig, photo_l1_ssim, semantic_ce, semantic_nll_from_probs
from ssc_gs.models.dinov2 import DinoV2Config, DinoV2Frozen
from ssc_gs.models.gaussians import GaussianScene, GaussianSceneTensors, init_gaussians_from_points
from ssc_gs.models.refiner import GaussianRefiner, RefinerConfig
from ssc_gs.render.gsplat_renderer import GSplatRenderer
from ssc_gs.utils.camera import kitti360_normK_to_pixelK


class _DummyDino:
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


@dataclass
class TrainConfig:
    # Data
    kitti360_data_path: str
    kitti360_pose_path: str
    pseudo_seg_path: str
    split_file: str = ""
    image_size_h: int = 192
    image_size_w: int = 640
    is_preprocessed: bool = False

    # Model
    num_classes: int = 20
    feat_dim: int = 256

    # Render
    packed: bool = True

    # Init
    use_online_mapanything: bool = False
    mapanything_cache_dir: str = "mapanything_cache"
    init_max_points: int = 200_000

    # Train
    lr: float = 1e-3
    steps: int = 100
    device: str = "cuda"


def build_s4c_dataset(cfg: TrainConfig):
    repo_root = Path(__file__).resolve().parents[1]
    split_file = cfg.split_file
    if split_file == "":
        split_file = ensure_kitti360_split_file(
            pseudo_seg_root=cfg.pseudo_seg_path,
            pose_root=cfg.kitti360_pose_path,
            out_path=str(repo_root / "ssc_gs" / "splits" / "kitti360_panoptic_deeplab_train_files.txt"),
        )

    ds = Kitti360Dataset(
        data_path=cfg.kitti360_data_path,
        pose_path=cfg.kitti360_pose_path,
        split_path=split_file,
        target_image_size=(cfg.image_size_h, cfg.image_size_w),
        return_stereo=False,
        return_fisheye=False,
        return_segmentation=True,
        segmentation_mode="panoptic_deeplab",
        data_segmentation_path=cfg.pseudo_seg_path,
        frame_count=1,
        is_preprocessed=cfg.is_preprocessed,
    )
    return S4CKitti360KeyframeDataset(ds, keyframe_idx=0, has_pseudo_sem=True)


def train_step(
    *,
    scene: GaussianScene,
    refiner: GaussianRefiner,
    dino: DinoV2Frozen,
    renderer: GSplatRenderer,
    loss_cfg: LossConfig,
    sample,
    optimizer,
):
    device = next(scene.parameters()).device

    img_m11 = sample.image.to(device)  # (3,H,W) in [-1,1]
    H, W = img_m11.shape[1:]
    gt_rgb = ((img_m11 + 1.0) * 0.5).clamp(0, 1)[None]  # (1,3,H,W)

    K_px = kitti360_normK_to_pixelK(sample.K_norm.to(device), width=W, height=H)
    camtoworld = sample.camtoworld.to(device)

    # DINOv2 features
    dino_feats = dino(gt_rgb, return_pyramid=False)  # (1,F,h,w)

    # Refinement deltas (apply as residuals for rendering; do NOT in-place overwrite Parameters)
    out = refiner(anchor_feat=scene.anchor_feat, dino_feat=dino_feats)

    rgb_logits_eff = scene.rgb_logits + out["d_rgb"]
    sem_logits_eff = scene.sem_logits + out["d_sem"]
    opacities_logit_eff = scene.opacities_logit + out["d_opacity"]

    rgb_eff = torch.sigmoid(rgb_logits_eff)
    opacities_eff = torch.sigmoid(opacities_logit_eff)
    # Stabilize semantics: avoid softmax saturation (0/1 probs -> zero gradients)
    # by optionally applying temperature and clipping.
    sem_t = float(getattr(loss_cfg, "sem_temperature", 1.0))
    if sem_t <= 0:
        sem_t = 1.0
    sem_logits_eff = sem_logits_eff / sem_t

    sem_clip = float(getattr(loss_cfg, "sem_logit_clip", 0.0))
    if sem_clip and sem_clip > 0:
        # Differentiable "soft clip" to keep logits bounded while still allowing
        # gradients to flow when raw logits are extremely large.
        c = float(loss_cfg.sem_logit_clip)
        sem_logits_eff = (c * sem_logits_eff) / (c + sem_logits_eff.abs())

    sem_probs_eff = torch.softmax(sem_logits_eff, dim=1)

    # Render
    renders = renderer.render(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=opacities_eff,
        rgb=rgb_eff,
        camtoworld=camtoworld,
        K_px=K_px,
        sem_logits=sem_probs_eff,
    )

    pred_rgb = renders.rgb.permute(2, 0, 1)[None]  # (1,3,H,W)
    loss_photo = photo_l1_ssim(pred_rgb, gt_rgb, ssim_lambda=loss_cfg.ssim_lambda)

    loss_sem = torch.tensor(0.0, device=device)
    if renders.sem_logits is not None and sample.pseudo_sem is not None:
        target = sample.pseudo_sem.to(device)[None]  # (1,H,W)

        # GSplat returns linear-blended channel values. When rendering per-Gaussian class probabilities,
        # the per-pixel channel sum is proportional to coverage (≈ alpha). Pixels with ~0 coverage
        # should not be supervised; otherwise p(target) collapses to eps and NLL becomes -log(eps).
        raw = renders.sem_logits.permute(2, 0, 1)[None]  # (1,C,H,W)
        raw = raw.clamp(min=0)
        denom = raw.sum(dim=1, keepdim=True)  # (1,1,H,W)
        pred_probs = raw / (denom + 1e-6)

        # Only supervise pixels where the splat actually contributes.
        covered = denom.squeeze(1) > 1e-6  # (1,H,W)
        if covered.any():
            target_masked = target.clone()
            target_masked[~covered] = loss_cfg.ignore_index
            loss_sem = semantic_nll_from_probs(pred_probs, target_masked, ignore_index=loss_cfg.ignore_index)

    loss = loss_photo + loss_cfg.sem_lambda * loss_sem

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "loss_photo": float(loss_photo.detach().cpu()),
        "loss_sem": float(loss_sem.detach().cpu()),
        "H": H,
        "W": W,
    }


def save_checkpoint(
    ckpt_path: Path,
    *,
    global_step: int,
    scene: GaussianScene,
    refiner: GaussianRefiner,
    optimizer: torch.optim.Optimizer,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "global_step": int(global_step),
        "scene": scene.state_dict(),
        "refiner": refiner.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        payload["extra"] = extra
        if isinstance(extra, dict) and ("args" in extra) and ("args" not in payload):
            payload["args"] = extra["args"]
    torch.save(payload, str(ckpt_path))


def load_checkpoint(
    ckpt_path: Path,
    *,
    device: torch.device,
    scene: GaussianScene,
    refiner: GaussianRefiner,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = torch.load(str(ckpt_path), map_location=device)
    scene.load_state_dict(payload["scene"], strict=True)
    refiner.load_state_dict(payload["refiner"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("global_step", -1))


def build_scene_from_checkpoint_state(scene_state: Dict[str, torch.Tensor], *, device: torch.device) -> GaussianScene:
    means = scene_state["means"].to(device=device)
    quats = scene_state["quats"].to(device=device)
    scales_log = scene_state["scales_log"].to(device=device)
    opacities_logit = scene_state["opacities_logit"].to(device=device)
    rgb_logits = scene_state["rgb_logits"].to(device=device)
    sem_logits = scene_state["sem_logits"].to(device=device)
    anchor_feat = scene_state["anchor_feat"].to(device=device)

    init = GaussianSceneTensors(
        means=torch.zeros_like(means),
        quats=torch.zeros_like(quats),
        scales_log=torch.zeros_like(scales_log),
        opacities_logit=torch.zeros_like(opacities_logit),
        rgb_logits=torch.zeros_like(rgb_logits),
        sem_logits=torch.zeros_like(sem_logits),
        anchor_feat=torch.zeros_like(anchor_feat),
    )
    return GaussianScene(init).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kitti360_data_path", type=str, default="/data/lmh_data/KITTI360")
    p.add_argument("--kitti360_pose_path", type=str, default="/data/lmh_data/KITTI360/poses")
    p.add_argument(
        "--pseudo_seg_path",
        type=str,
        default="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr",
    )
    p.add_argument(
        "--split_file",
        type=str,
        default="",
        help="Optional: path to a KITTI-360 split file (<seq> <img_id> <l|r>). If empty, it is built from pseudo_seg_path.",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default="",
        help="Optional: restrict training to a single KITTI-360 sequence (e.g. 2013_05_28_drive_0000_sync).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--is_preprocessed", action="store_true", help="Use pre-resized KITTI-360 folders data_{H}x{W}")
    p.add_argument("--dino_mode", type=str, default="real", choices=["real", "dummy"], help="Use real DINOv2 (downloads weights) or dummy features (no download).")
    p.add_argument("--use_online_mapanything", action="store_true")
    p.add_argument("--mapanything_cache_dir", type=str, default="/data/lmh/ssc_gs_pointcloud_cache")
    p.add_argument(
        "--init_num_frames",
        type=int,
        default=1,
        help="Number of (cached) frames to merge for Gaussian initialization (world-frame points are concatenated).",
    )
    p.add_argument(
        "--init_stride",
        type=int,
        default=10,
        help="Stride (in dataset order) when picking frames for initialization.",
    )
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="If set, saves checkpoints into this directory (latest.pt and optionally step_*.pt).",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If >0 and --ckpt_dir is set, saves a checkpoint every N global steps (also updates latest.pt).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle training samples. Default is False to keep samples local (important when training a single scene init).",
    )
    p.add_argument(
        "--subset_size",
        type=int,
        default=0,
        help="If >0, train only on the first N samples of the dataset (keeps cache indexing consistent and avoids cross-sequence drift).",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N samples (after any sequence filtering). Useful to train near the scene init window to ensure non-zero coverage.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional: path to a checkpoint (.pt) to resume from.",
    )
    p.add_argument(
        "--sem_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for per-Gaussian semantic logits (larger -> less peaky).",
    )
    p.add_argument(
        "--sem_logit_clip",
        type=float,
        default=20.0,
        help="Differentiable soft-clip scale for semantic logits before softmax (keeps logits bounded but preserves gradient flow).",
    )
    args = p.parse_args()

    ckpt_extra = {
        "args": vars(args),
    }

    cfg = TrainConfig(
        kitti360_data_path=args.kitti360_data_path,
        kitti360_pose_path=args.kitti360_pose_path,
        pseudo_seg_path=args.pseudo_seg_path,
        split_file=args.split_file,
        device=args.device,
        steps=args.steps,
        lr=args.lr,
        is_preprocessed=args.is_preprocessed,
        use_online_mapanything=args.use_online_mapanything,
        mapanything_cache_dir=args.mapanything_cache_dir,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ds = build_s4c_dataset(cfg)

    from torch.utils.data import Subset

    if args.sequence:
        # Keep original indices so MapAnything cache naming (by dataset index) stays valid.
        base = ds.ds  # underlying S4C Kitti360Dataset
        if not hasattr(base, "_datapoints"):
            raise RuntimeError("Underlying dataset does not expose _datapoints; cannot filter by sequence")
        seq = str(args.sequence)
        seq_indices = [i for i, dp in enumerate(base._datapoints) if dp[0] == seq]
        if len(seq_indices) == 0:
            raise RuntimeError(f"No datapoints found for sequence='{seq}'.")
        ds = Subset(ds, seq_indices)

    if args.start and args.start > 0:
        start = int(args.start)
        if start >= len(ds):
            raise RuntimeError(f"--start={start} is out of range for dataset length {len(ds)}")
        ds = Subset(ds, list(range(start, len(ds))))

    if args.subset_size and args.subset_size > 0:
        ds = Subset(ds, list(range(min(int(args.subset_size), len(ds)))))
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=bool(args.shuffle),
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: batch[0],
    )

    s = next(iter(dl))

    resume_path = Path(args.resume) if args.resume else None
    resume_payload: Optional[Dict[str, Any]] = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(str(resume_path))
        resume_payload = torch.load(str(resume_path), map_location=device)
        scene = build_scene_from_checkpoint_state(resume_payload["scene"], device=device)
    else:
        # Init gaussians (from cached offline point cloud, unless online MapAnything is requested)
        init_cfg = MapAnythingInitConfig(cache_dir=cfg.mapanything_cache_dir, max_points=cfg.init_max_points)
        initializer = MapAnythingInitializer(init_cfg, device=device)

        if cfg.use_online_mapanything:
            # window size=1 for now (keyframe only); extend to temporal window by using the raw S4C dataset output.
            pts = initializer.infer_points_from_window(
                images_m11_chw=[s.image],
                K_norm_list=[s.K_norm],
                camtoworld_list=[s.camtoworld],
                sample_index=s.index,
            )
        else:
            # Merge cached points from multiple frames to cover a moving camera sequence.
            num_frames = max(1, int(args.init_num_frames))
            stride = max(1, int(args.init_stride))

            pts_list = []
            picked = 0
            for j in range(0, min(len(ds), stride * num_frames * 2), stride):
                sj = ds[j]
                cached = initializer.load_cached_points(sj.index)
                if cached is None:
                    continue
                pts_list.append(cached)
                picked += 1
                if picked >= num_frames:
                    break

            if len(pts_list) == 0:
                raise RuntimeError(
                    f"Missing MapAnything cache for initial indices (starting at index={s.index}). "
                    f"Precompute caches under {cfg.mapanything_cache_dir}/"
                )

            pts = torch.cat([p.to(device) for p in pts_list], dim=0)
            if pts.shape[0] > cfg.init_max_points:
                perm = torch.randperm(pts.shape[0], device=device)[: cfg.init_max_points]
                pts = pts[perm]

        init_tensors = init_gaussians_from_points(
            pts,
            num_classes=cfg.num_classes,
            feat_dim=cfg.feat_dim,
            init_scale=0.03,
            init_opacity=0.1,
            device=device,
        )
        scene = GaussianScene(init_tensors).to(device)

    if args.dino_mode == "dummy":
        dino = _DummyDino(cfg.feat_dim)
    else:
        dino = DinoV2Frozen(DinoV2Config(out_dim=cfg.feat_dim)).to(device)
    refiner = GaussianRefiner(RefinerConfig(feat_dim=cfg.feat_dim, num_classes=cfg.num_classes)).to(device)

    # Renderer uses image size
    H, W = s.image.shape[1:]
    renderer = GSplatRenderer(width=W, height=H, packed=cfg.packed)

    opt = torch.optim.Adam(list(scene.parameters()) + list(refiner.parameters()), lr=cfg.lr)
    loss_cfg = LossConfig(
        ssim_lambda=0.2,
        sem_lambda=1.0,
        sem_temperature=float(args.sem_temperature),
        sem_logit_clip=float(args.sem_logit_clip),
    )

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    start_global_step = 0

    if resume_payload is not None and resume_path is not None:
        # We already used resume_payload to build correctly-shaped modules; now load full state.
        last = load_checkpoint(
            resume_path,
            device=device,
            scene=scene,
            refiner=refiner,
            optimizer=opt,
        )
        start_global_step = last + 1
        print(f"Resumed from {resume_path} (last_global_step={last})")

    # Training loop
    it = iter(dl)
    for local_step in range(cfg.steps):
        global_step = start_global_step + local_step
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        stats = train_step(
            scene=scene,
            refiner=refiner,
            dino=dino,
            renderer=renderer,
            loss_cfg=loss_cfg,
            sample=batch,
            optimizer=opt,
        )

        if global_step % 10 == 0:
            print(
                f"step={global_step} loss={stats['loss']:.4f} photo={stats['loss_photo']:.4f} sem={stats['loss_sem']:.4f}"
            )

        if ckpt_dir is not None:
            latest_path = ckpt_dir / "latest.pt"
            do_periodic = args.save_every > 0 and (global_step % args.save_every == 0)
            if do_periodic:
                save_checkpoint(
                    latest_path,
                    global_step=global_step,
                    scene=scene,
                    refiner=refiner,
                    optimizer=opt,
                    extra=ckpt_extra,
                )
                step_path = ckpt_dir / f"step_{global_step:07d}.pt"
                save_checkpoint(
                    step_path,
                    global_step=global_step,
                    scene=scene,
                    refiner=refiner,
                    optimizer=opt,
                    extra=ckpt_extra,
                )

    if ckpt_dir is not None:
        save_checkpoint(
            ckpt_dir / "latest.pt",
            global_step=start_global_step + cfg.steps - 1,
            scene=scene,
            refiner=refiner,
            optimizer=opt,
            extra=ckpt_extra,
        )


if __name__ == "__main__":
    main()
