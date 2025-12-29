from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from s4c.datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from ssc_gs.losses import LossConfig, depth_smoothness, photo_l1_ssim, semantic_nll_from_probs
from ssc_gs.render.gsplat_renderer import GSplatRenderer
from ssc_gs.utils.camera import kitti360_normK_to_pixelK

from ssc_gs.ff.scene_model import FFSceneConfig, FFSceneModel
from ssc_gs.ff.gaussian_to_voxel import VoxelGridSpec
from ssc_gs.initialize.mapanything_init import MapAnythingInitConfig, MapAnythingInitializer


def _ensure_split_file(pseudo_seg_root: str, pose_root: str) -> str:
    # Reuse existing helper from ssc_gs.train
    from ssc_gs.train import ensure_kitti360_split_file

    repo_root = Path(__file__).resolve().parents[1]
    out_path = str(repo_root / "ssc_gs" / "splits" / "kitti360_panoptic_deeplab_train_files.txt")
    return ensure_kitti360_split_file(pseudo_seg_root=pseudo_seg_root, pose_root=pose_root, out_path=out_path)


def save_checkpoint(path: Path, *, step: int, model: torch.nn.Module, opt: torch.optim.Optimizer, extra: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"global_step": int(step), "model": model.state_dict(), "optimizer": opt.state_dict(), "extra": extra}, path)


def main() -> None:
    p = argparse.ArgumentParser("Feed-forward scene Gaussian training (self-encoding + cross-attn + refinement, S4C multi-view supervision)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=100, help="Total optimization steps (ignored if --epochs>0)")
    p.add_argument("--epochs", type=int, default=0, help="Train for N full epochs over the dataset (overrides --steps)")
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--kitti360_data_path", type=str, default="/data/lmh_data/KITTI360")
    p.add_argument("--kitti360_pose_path", type=str, default="/data/lmh_data/KITTI360/poses")
    p.add_argument(
        "--pseudo_seg_path",
        type=str,
        default="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr",
    )

    p.add_argument("--data_fc", type=int, default=3, help="S4C-style multiview frame_count")
    p.add_argument("--dilation", "--data_dilation", dest="dilation", type=int, default=1)
    p.add_argument("--keyframe_offset", type=int, default=0)
    p.add_argument("--additional_random_front_offset", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--subset_size", type=int, default=0)

    p.add_argument("--ckpt_dir", type=str, default="/data/lmh_data/VGM-GS-S4C-cp")
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume_strict", type=int, default=1, choices=[0, 1], help="1=strict checkpoint load (default), 0=allow missing/unexpected keys")
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--encoder_index", type=int, default=0, help="ids_encoder (use this view for inference)")
    p.add_argument("--include_keyframe_supervision", action="store_true")

    # Optional MapAnything init (feed-forward). Uses known K/poses from dataset.
    p.add_argument(
        "--disable_mapanything_init",
        action="store_true",
        help="Disable MapAnything initialization (fallback to token-grid self-encoding). MapAnything init is ON by default.",
    )
    p.add_argument(
        "--mapanything_hf_model_name",
        type=str,
        default="facebook/map-anything-apache",
        help="HuggingFace model id or local path for MapAnything.from_pretrained",
    )
    p.add_argument("--mapanything_cache_dir", type=str, default="mapanything_cache", help="Cache directory for MapAnything inferred point clouds")
    p.add_argument("--mapanything_max_points", type=int, default=20_000, help="Max points (global) returned by MapAnything initializer")
    p.add_argument("--mapanything_force_recompute", action="store_true", help="Ignore cached MapAnything results and recompute")
    p.add_argument(
        "--mapanything_local_files_only",
        action="store_true",
        help="Do not access the network when loading MapAnything weights (requires local HF cache)",
    )
    p.add_argument(
        "--mapanything_cached_only",
        action="store_true",
        help="Never load MapAnything weights; require cached points to exist",
    )

    # Losses
    p.add_argument("--supervise_rgb", action="store_true")
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--sem_lambda", type=float, default=1.0)
    p.add_argument("--ignore_index", type=int, default=255)
    p.add_argument("--smooth_lambda", type=float, default=0.01, help="Edge-aware depth smoothness weight (keyframe).")

    # Self-encoding Gaussian->Voxel (GaussianFormer-like). This happens inside the model.
    p.add_argument("--voxel_reg_weight", type=float, default=0.0)
    p.add_argument("--voxel_max_radius", type=int, default=3)
    p.add_argument(
        "--use_sparse_conv",
        type=int,
        default=1,
        choices=[0, 1],
        help="Phase3 self-encoding backend: 1=spconv sparse (default if available), 0=force dense voxel CNN.",
    )

    # Renderer
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=192)

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split_file = _ensure_split_file(args.pseudo_seg_path, args.kitti360_pose_path)

    # S4C requires random_fisheye_offset to be active when additional_random_front_offset is used.
    fisheye_offset = (0, 0) if bool(args.additional_random_front_offset) else 0

    ds = Kitti360Dataset(
        data_path=args.kitti360_data_path,
        pose_path=args.kitti360_pose_path,
        split_path=split_file,
        target_image_size=None,
        return_stereo=False,
        return_depth=False,
        return_fisheye=False,
        fisheye_offset=fisheye_offset,
        return_segmentation=True,
        segmentation_mode="panoptic_deeplab",
        data_segmentation_path=args.pseudo_seg_path,
        frame_count=int(args.data_fc),
        dilation=int(args.dilation),
        keyframe_offset=int(args.keyframe_offset),
        additional_random_front_offset=bool(args.additional_random_front_offset),
        is_preprocessed=False,
    )

    if args.subset_size and args.subset_size > 0:
        from torch.utils.data import Subset

        ds = Subset(ds, list(range(min(int(args.subset_size), len(ds)))))

    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=bool(args.shuffle), num_workers=int(args.num_workers), collate_fn=lambda b: b)

    model = FFSceneModel(FFSceneConfig(use_sparse_conv=bool(int(args.use_sparse_conv) != 0))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    map_init = None
    if not bool(getattr(args, "disable_mapanything_init", False)):
        map_cfg = MapAnythingInitConfig(
            hf_model_name=str(args.mapanything_hf_model_name),
            cache_dir=str(args.mapanything_cache_dir),
            max_points=int(args.mapanything_max_points),
            local_files_only=bool(args.mapanything_local_files_only),
        )
        map_init = MapAnythingInitializer(map_cfg, device=device)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        strict = bool(int(getattr(args, "resume_strict", 1)) != 0)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
        if (missing or unexpected) and (not strict):
            print(f"[resume] non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("global_step", -1)) + 1
        print(f"Resumed from {args.resume} (start_step={start_step})")

    # Phase3 self-encoding voxel grid can be coarser than evaluation grid to reduce compute.
    grid = VoxelGridSpec(voxel_size=0.4, dims_xyz=(128, 128, 32))
    loss_cfg = LossConfig(ssim_lambda=float(args.ssim_lambda), sem_lambda=float(args.sem_lambda), ignore_index=int(args.ignore_index))
    renderer = GSplatRenderer(width=int(args.width), height=int(args.height), packed=True, antialiased=False)

    # Access calibration (for self-encoding voxel reg in keyframe velo coords)
    base_ds = ds.dataset if hasattr(ds, "dataset") else ds
    T_velo_to_cam = torch.tensor(base_ds._calibs["T_velo_to_cam"]["00"], dtype=torch.float32, device=device)
    cam_to_velo = torch.inverse(T_velo_to_cam)

    total_steps = int(args.steps)
    use_epochs = int(args.epochs) > 0
    if use_epochs:
        total_steps = int(args.epochs) * len(dl)

    # When resuming, skip forward within the epoch approximately.
    start_epoch = 0
    skip_in_epoch = 0
    if len(dl) > 0:
        start_epoch = int(start_step // len(dl))
        skip_in_epoch = int(start_step % len(dl))

    global_step = int(start_step)

    def _run_one_batch(batch_item: Dict[str, Any], *, step: int, epoch: int) -> None:
        nonlocal global_step

        # batch is list of dicts
        data = batch_item
        imgs = data["imgs"]
        imgs_raw = data.get("imgs_raw", None)
        projs = data["projs"]
        poses = data["poses"]
        segs = data["segs_gt"]

        # Build tensors
        num_views = len(imgs)
        imgs_t: list[torch.Tensor] = [im.to(device) for im in imgs]
        imgs_raw_t: Optional[list[torch.Tensor]] = [im.to(device) for im in imgs_raw] if imgs_raw is not None else None
        Ks_t: list[torch.Tensor] = [torch.tensor(K, dtype=torch.float32, device=device) for K in projs]
        c2w_t: list[torch.Tensor] = [torch.tensor(P, dtype=torch.float32, device=device) for P in poses]

        enc_idx = int(args.encoder_index)
        if enc_idx < 0 or enc_idx >= num_views:
            raise ValueError(f"encoder_index {enc_idx} out of range for num_views={num_views}")

        supervise_indices = [j for j in range(num_views) if j != enc_idx]
        if bool(args.include_keyframe_supervision):
            supervise_indices = list(range(num_views))

        init_points_world = None
        if map_init is not None:
            # dataset provides a stable integer index for caching
            sample_index = int(data.get("index", [0])[0]) if hasattr(data.get("index", None), "__len__") else int(data.get("index", 0))
            
            # Only use perspective views for MapAnything init (first data_fc frames)
            # Kitti360Dataset returns [p_left, (p_right), f_left, (f_right)]
            # We assume return_stereo=False (default), so [p_left, f_left]
            # p_left has data_fc frames.
            n_persp = int(args.data_fc)
            
            init_points_world = map_init.infer_points_from_window(
                images_m11_chw=imgs_t[:n_persp],
                images_raw_01_chw=imgs_raw_t[:n_persp] if imgs_raw_t is not None else None,
                K_norm_list=Ks_t[:n_persp],
                camtoworld_list=c2w_t[:n_persp],
                sample_index=sample_index,
                force_recompute=bool(args.mapanything_force_recompute),
                cached_only=bool(args.mapanything_cached_only),
            )

        pred = model(
            imgs_m11_chw=imgs_t,
            Ks_norm=Ks_t,
            camtoworlds=c2w_t,
            encoder_index=enc_idx,
            supervise_indices=supervise_indices,
            init_points_world=init_points_world,
            T_velo_to_cam=T_velo_to_cam,
            voxel_grid=grid,
            voxel_max_radius=int(args.voxel_max_radius),
        )

        voxel_reg = pred.get("voxel_sigma_mean", torch.tensor(0.0, device=device))
        pred_depth = pred.get("pred_depth", None)

        # Multi-view 2D supervision (S4C-aligned sampling): encode one view, supervise other views.
        loss_sem = torch.tensor(0.0, device=device)
        loss_photo = torch.tensor(0.0, device=device)
        loss_smooth = torch.tensor(0.0, device=device)
        n_sup = 0

        for j in supervise_indices:
            imgj = imgs_t[j]
            Hj, Wj = imgj.shape[-2:]
            gt_rgb = ((imgj + 1.0) * 0.5).clamp(0, 1)[None]  # (1,3,H,W)

            K_px = kitti360_normK_to_pixelK(Ks_t[j], width=Wj, height=Hj)
            renders = renderer.render(
                means=pred["means_world"],
                quats=pred["quats"],
                scales=pred["scales"],
                opacities=pred["opacities"].view(-1),
                rgb=pred["rgb"],
                camtoworld=c2w_t[j],
                K_px=K_px,
                sem_logits=torch.softmax(pred["sem_logits"], dim=-1),
                width=Wj,
                height=Hj,
            )

            if bool(args.supervise_rgb):
                pred_rgb = renders.rgb.permute(2, 0, 1)[None]
                loss_photo = loss_photo + photo_l1_ssim(pred_rgb, gt_rgb, ssim_lambda=loss_cfg.ssim_lambda)

            # Pseudo semantic supervision
            target_np = segs[j]
            if torch.is_tensor(target_np):
                target = target_np.to(device=device)
            else:
                target = torch.tensor(target_np, dtype=torch.int64, device=device)

            if target.shape[-2:] != (Hj, Wj):
                target = torch.nn.functional.interpolate(target[None, None].float(), size=(Hj, Wj), mode="nearest").long()[0, 0]

            raw = renders.sem_logits
            if raw is not None:
                raw = raw.permute(2, 0, 1)[None].clamp(min=0)  # (1,C,H,W)
                denom = raw.sum(dim=1, keepdim=True)
                pred_probs = raw / (denom + 1e-6)
                covered = denom.squeeze(1) > 1e-6
                if covered.any():
                    target_b = target[None]
                    target_masked = target_b.clone()
                    target_masked[~covered] = int(loss_cfg.ignore_index)
                    loss_sem = loss_sem + semantic_nll_from_probs(pred_probs, target_masked, ignore_index=int(loss_cfg.ignore_index))

            n_sup += 1

        # Smoothness is applied on the keyframe's predicted depth at token resolution.
        if pred_depth is not None and float(args.smooth_lambda) > 0:
            img0_01 = ((imgs_t[enc_idx][None] + 1.0) * 0.5).clamp(0, 1)
            loss_smooth = depth_smoothness(pred_depth, img0_01)

        if n_sup > 0:
            loss_photo = loss_photo / float(n_sup)
            loss_sem = loss_sem / float(n_sup)

        loss = (
            loss_photo
            + float(loss_cfg.sem_lambda) * loss_sem
            + float(args.voxel_reg_weight) * voxel_reg
            + float(args.smooth_lambda) * loss_smooth
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if int(args.log_every) > 0 and (step % int(args.log_every) == 0):
            print(
                f"epoch={epoch} step={step} loss={float(loss.detach().cpu()):.4f} "
                f"photo={float(loss_photo.detach().cpu()):.4f} sem={float(loss_sem.detach().cpu()):.4f} "
                f"smooth={float(loss_smooth.detach().cpu()):.4f} voxel_reg={float(voxel_reg.detach().cpu()):.4f} sup_views={n_sup}"
            )

        if args.ckpt_dir and args.save_every > 0 and (step % int(args.save_every) == 0):
            ckpt_dir = Path(args.ckpt_dir)
            extra = {"args": vars(args)}
            save_checkpoint(ckpt_dir / "latest.pt", step=step, model=model, opt=opt, extra=extra)
            save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", step=step, model=model, opt=opt, extra=extra)


    if use_epochs:
        # Epoch-based loop over DataLoader.
        for epoch in range(start_epoch, int(args.epochs)):
            for i, batch in enumerate(dl):
                if global_step >= total_steps:
                    break

                # collate_fn=lambda b: b => batch is list; we use first element
                if not batch:
                    continue
                if epoch == start_epoch and i < skip_in_epoch:
                    continue

                _run_one_batch(batch[0], step=global_step, epoch=epoch)
                global_step += 1

            if global_step >= total_steps:
                break
    else:
        # Step-based loop cycling the DataLoader.
        it = iter(dl)
        for local in range(int(total_steps)):
            step = int(start_step + local)
            epoch = int(step // max(1, len(dl)))
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            if not batch:
                continue
            _run_one_batch(batch[0], step=step, epoch=epoch)


if __name__ == "__main__":
    main()
