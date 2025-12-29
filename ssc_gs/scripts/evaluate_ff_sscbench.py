from __future__ import annotations

import argparse
import logging
import sys
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

# Workaround: some Python environments install a top-level package named `scripts` (site-packages),
# which can shadow S4C's `s4c/scripts` folder used by SSCBench utilities.
# Ensure we resolve `scripts.*` from the vendored S4C tree.
_s4c_root = _REPO_ROOT / "s4c"
_mod_scripts = sys.modules.get("scripts")
if _mod_scripts is not None:
    mod_file = getattr(_mod_scripts, "__file__", "") or ""
    if "site-packages" in mod_file and (str(_s4c_root) in sys.path):
        sys.modules.pop("scripts", None)
        import importlib

        importlib.invalidate_caches()

# Reuse SSCBench utilities from S4C
from s4c.scripts.benchmarks.sscbench.point_utils import get_fov_mask
from s4c.scripts.benchmarks.sscbench.sscbench_dataset import SSCBenchDataset

from ssc_gs.ff.scene_model import FFSceneModel, FFSceneConfig
from ssc_gs.ff.gaussian_to_voxel import VoxelGridSpec, splat_gaussians_to_voxels_gaussian_kernel

from ssc_gs.initialize.mapanything_init import MapAnythingInitConfig, MapAnythingInitializer

# Reuse evaluation logic (metrics + voxelization) from our S4C-compatible evaluator
from ssc_gs.scripts.evaluate_model_sscbench import (
    SIZES,
    _load_label_maps,
    _load_kitti360_poses,
    _nearest_pose,
    compute_occupancy_numbers,
    compute_occupancy_numbers_segmentation,
    convert_voxels,
    identify_additional_invalids,
    voxelize_gaussians_to_sscbench,
)


def _unique_sorted_thresholds(vals: Optional[list[float]]) -> list[float]:
    if not vals:
        return []
    out: list[float] = []
    for v in vals:
        fv = float(v)
        if fv <= 0:
            continue
        out.append(fv)
    # unique while preserving order, then sort
    out = list(dict.fromkeys(out))
    out.sort()
    return out


def _accumulate_occ_sweep(
    *,
    sigmas_xyz: np.ndarray,
    target_xyz: np.ndarray,
    fov_xyz: np.ndarray,
    thresholds: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast per-sample occupancy counts for multiple sigma thresholds.

    Returns arrays (tp, fp, tn, fn) with shape (T,).
    """
    if len(thresholds) == 0:
        raise ValueError("thresholds must be non-empty")

    mask = (target_xyz != 255) & (fov_xyz.astype(bool))
    if not np.any(mask):
        T = len(thresholds)
        z = np.zeros((T,), dtype=np.int64)
        return z, z, z, z

    sig = np.asarray(sigmas_xyz, dtype=np.float32)[mask]
    occ_true = (target_xyz[mask] > 0)

    pos_true = int(np.count_nonzero(occ_true))
    total = int(sig.shape[0])
    neg_true = total - pos_true

    Tn = len(thresholds)
    tp = np.zeros((Tn,), dtype=np.int64)
    fp = np.zeros((Tn,), dtype=np.int64)
    tn = np.zeros((Tn,), dtype=np.int64)
    fn = np.zeros((Tn,), dtype=np.int64)

    # NOTE: thresholds is small (typically <= 10), so looping is fine.
    # We avoid recomputing mask/flattening repeatedly.
    for k, thr in enumerate(thresholds):
        occ_pred = sig > float(thr)
        _tp = int(np.count_nonzero(occ_pred & occ_true))
        _pred_pos = int(np.count_nonzero(occ_pred))
        _fp = _pred_pos - _tp
        _fn = pos_true - _tp
        _tn = neg_true - _fp

        tp[k] = _tp
        fp[k] = _fp
        tn[k] = _tn
        fn[k] = _fn

    return tp, fp, tn, fn


def _resolve_checkpoint_path(cp_arg: str) -> Path:
    p = Path(cp_arg)
    if p.is_file():
        return p
    if p.is_dir():
        latest = p / "latest.pt"
        if latest.exists():
            return latest
        pts = sorted(p.glob("*.pt"))
        if pts:
            return pts[-1]
    raise FileNotFoundError(cp_arg)


def _try_load_processed_img(
    dataset: SSCBenchDataset,
    *,
    seq: str,
    frame_id: int,
) -> Optional[torch.Tensor]:
    """Load+process a single SSCBench image into (3,H,W) [-1,1]."""
    try:
        imgs = dataset.load_images(seq, [int(frame_id)])
        if not imgs:
            return None
        img = dataset.process_img(imgs[0], color_aug_fn=None)
        if not torch.is_tensor(img):
            img = torch.tensor(img)
        return img
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser("SSCBenchmark evaluation (feed-forward GS->voxel)")
    parser.add_argument("--sscbench_data_root", "-ssc", type=str, required=True)
    parser.add_argument("--voxel_gt_path", "-vgt", type=str, required=True)
    parser.add_argument("--checkpoint", "-cp", type=str, required=True, help="FF model checkpoint (dir or .pt)")

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--full", "-f", action="store_true")
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=[9],
        help="KITTI-360 drive ids to evaluate (e.g. 5, 9).",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=[192, 640])
    parser.add_argument("--native_resolution", action="store_true", help="Use native image resolution (disable resizing). Overrides --resolution.")

    # Optional MapAnything init (recommended; matches ff_train default).
    parser.add_argument(
        "--disable_mapanything_init",
        action="store_true",
        help="Disable MapAnything initialization (fallback to token-grid self-encoding). MapAnything init is ON by default.",
    )
    parser.add_argument(
        "--mapanything_hf_model_name",
        type=str,
        default="facebook/map-anything-apache",
        help="HuggingFace model id or local path for MapAnything.from_pretrained",
    )
    parser.add_argument("--mapanything_cache_dir", type=str, default="mapanything_cache_sscbench", help="Cache directory for MapAnything inferred point clouds")
    parser.add_argument("--mapanything_max_points", type=int, default=20_000, help="Max points (global) returned by MapAnything initializer")
    parser.add_argument(
        "--mapanything_n_views",
        type=int,
        default=3,
        help="How many temporal views to provide to MapAnything for initialization (default: 3).",
    )
    parser.add_argument(
        "--mapanything_dilation",
        type=int,
        default=1,
        help="Frame step between MapAnything init views (default: 1).",
    )
    parser.add_argument("--mapanything_force_recompute", action="store_true", help="Ignore cached MapAnything results and recompute")
    parser.add_argument(
        "--mapanything_local_files_only",
        action="store_true",
        help="Do not access the network when loading MapAnything weights (requires local HF cache)",
    )
    parser.add_argument(
        "--mapanything_cached_only",
        action="store_true",
        help="Never load MapAnything weights; require cached points to exist",
    )

    # Keep defaults aligned with S4C-compatible evaluator.
    parser.add_argument("--sigma_cutoff", type=float, default=0.25)
    parser.add_argument(
        "--sigma_sweep",
        type=float,
        nargs="*",
        default=None,
        help=(
            "If provided, evaluates occupancy IoU for multiple sigma thresholds in one run. "
            "This reuses the same voxelization output, so it does NOT rerun the model per threshold. "
            "Example: --sigma_sweep 0.25 0.1 0.05 0.02 0.01 0.005"
        ),
    )
    parser.add_argument("--pred_sem_channels", type=int, default=20)
    parser.add_argument("--use_additional_invalids", action="store_true", default=True)

    parser.add_argument(
        "--voxel_mode",
        type=str,
        default="gaussian_kernel",
        choices=("gaussian_kernel", "opacity_sum"),
        help=(
            "How to convert predicted Gaussians into SSCBench voxels. "
            "'gaussian_kernel' (default) spreads each Gaussian to a local voxel neighborhood using its scale. "
            "'opacity_sum' assigns each Gaussian only to its containing voxel (S4C-compatible fallback), "
            "which can be much sparser."
        ),
    )

    parser.add_argument("--voxel_max_radius", type=int, default=3)

    parser.add_argument(
        "--sanity_hard_voxelize",
        action="store_true",
        default=False,
        help=(
            "Sanity check: ignore Gaussian kernel splatting and sigma_cutoff. "
            "Hard-voxelize Gaussian centers into the SSCBench velodyne grid and compute occupancy IoU. "
            "This helps detect coordinate/transform issues (off-grid, wrong frame)."
        ),
    )

    parser.add_argument(
        "--debug_stats",
        action="store_true",
        default=False,
        help="Print per-sample diagnostic stats (ranges, inside-grid ratios, sigma quantiles).",
    )
    parser.add_argument(
        "--debug_first_n",
        type=int,
        default=0,
        help="If >0, only run first N samples and print debug stats for them.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict", type=int, default=1, choices=[0, 1], help="1=strict checkpoint load (default), 0=allow missing/unexpected keys")

    parser.add_argument(
        "--pose_source",
        type=str,
        default="kitti360",
        choices=("kitti360", "none"),
        help="How to get per-frame cam-to-world poses. 'kitti360' loads poses.txt; 'none' uses identity poses.",
    )
    parser.add_argument(
        "--pose_root",
        type=str,
        default=None,
        help="Root containing KITTI-360 pose folders (default: <voxel_gt_path>/../poses).",
    )

    args = parser.parse_args()

    sweep_thresholds = _unique_sorted_thresholds(list(args.sigma_sweep) if args.sigma_sweep is not None else None)
    if sweep_thresholds:
        print(f"[sigma_sweep] thresholds={sweep_thresholds}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    map_init: Optional[MapAnythingInitializer] = None
    if not bool(getattr(args, "disable_mapanything_init", False)):
        map_cfg = MapAnythingInitConfig(
            hf_model_name=str(args.mapanything_hf_model_name),
            cache_dir=str(args.mapanything_cache_dir),
            max_points=int(args.mapanything_max_points),
            local_files_only=bool(args.mapanything_local_files_only),
        )
        map_init = MapAnythingInitializer(map_cfg, device=device)

    pose_root = Path(args.pose_root) if args.pose_root else (Path(args.voxel_gt_path).resolve().parent / "poses")
    poses_cache: Dict[str, Dict[int, np.ndarray]] = {}

    label_maps = _load_label_maps()

    target_res = tuple(args.resolution)
    if args.native_resolution:
        target_res = None

    dataset = SSCBenchDataset(
        data_path=args.sscbench_data_root,
        voxel_gt_path=args.voxel_gt_path,
        sequences=tuple(int(s) for s in args.sequences),
        target_image_size=target_res,
        return_stereo=False,
        frame_count=1,
        color_aug=False,
    )

    fov_mask = get_fov_mask().reshape(256, 256, 32)

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    logging.info(f"Loading FF checkpoint: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location=device)

    model = FFSceneModel(FFSceneConfig()).to(device)
    strict = bool(int(getattr(args, "strict", 1)) != 0)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if (missing or unexpected) and (not strict):
        print(f"[load] non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    grid = VoxelGridSpec()

    # Calibration for world<->velo
    calibs = dataset._calibs
    T_cam_to_pose = torch.as_tensor(calibs["T_cam_to_pose"]["00"], dtype=torch.float32)
    T_velo_to_pose = torch.as_tensor(calibs["T_velo_to_pose"], dtype=torch.float32)
    T_velo_to_cam = torch.as_tensor(calibs["T_velo_to_cam"]["00"], dtype=torch.float32)
    cam_to_velo = torch.linalg.inv(T_velo_to_cam)

    thresholds = sweep_thresholds if sweep_thresholds else [float(args.sigma_cutoff)]

    results = {}
    for size in SIZES:
        results[size] = {
            # occupancy counts per threshold
            "tp": np.zeros((len(thresholds),), dtype=np.int64),
            "fp": np.zeros((len(thresholds),), dtype=np.int64),
            "tn": np.zeros((len(thresholds),), dtype=np.int64),
            "fn": np.zeros((len(thresholds),), dtype=np.int64),
            # segmentation counts (computed only for args.sigma_cutoff)
            "tp_seg": np.zeros(15),
            "fp_seg": np.zeros(15),
            "tn_seg": np.zeros(15),
            "fn_seg": np.zeros(15),
        }

    results_hard = None
    if bool(args.sanity_hard_voxelize):
        results_hard = {size: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for size in SIZES}

    eval_len = len(dataset) if args.full else min(len(dataset), int(args.max_samples) if args.max_samples else len(dataset))
    if int(args.debug_first_n) > 0:
        eval_len = min(eval_len, int(args.debug_first_n))
    pbar = tqdm(range(eval_len))

    for i in pbar:
        seq, frame_id, _is_right = dataset._datapoints[i]
        data = dataset[i]

        # GT: sscbench ids -> label ids
        target = convert_voxels(data["voxel_gt"][0].astype(int), label_maps["sscbench_to_label"])
        if args.use_additional_invalids:
            invalids = identify_additional_invalids(target)
            target[invalids == 1] = 255

        # Pose: pose->world (KITTI-360 poses.txt)
        if args.pose_source == "none":
            pose_to_world_np = np.eye(4, dtype=np.float32)
        else:
            if seq not in poses_cache:
                poses_cache[seq] = _load_kitti360_poses(pose_root, seq)
            pose_to_world_np = _nearest_pose(poses_cache[seq], int(frame_id))
            if pose_to_world_np is None:
                pose_to_world_np = np.eye(4, dtype=np.float32)

        pose_to_world = torch.as_tensor(pose_to_world_np, dtype=torch.float32)
        velo_to_world = pose_to_world @ T_velo_to_pose
        world_to_velo_pose = torch.linalg.inv(velo_to_world).to(device)

        cam_to_world = (pose_to_world @ T_cam_to_pose).to(device)

        # Predict Gaussians from the single image (feed-forward)
        img0 = data["imgs"][0]
        if not torch.is_tensor(img0):
            img0 = torch.tensor(img0)
        img0 = img0.to(device)

        K0 = torch.as_tensor(data["projs"][0], dtype=torch.float32, device=device)

        init_points_world = None
        if map_init is not None:
            # Build a small temporal window to help MapAnything recover metric scale from known motion.
            n_views = max(int(getattr(args, "mapanything_n_views", 1)), 1)
            dilation = max(int(getattr(args, "mapanything_dilation", 1)), 1)

            imgs_win: list[torch.Tensor] = []
            Ks_win: list[torch.Tensor] = []
            c2w_win: list[torch.Tensor] = []
            img_paths_win: list[str] = []

            for k in range(n_views):
                fid = int(frame_id) + k * dilation
                # IMPORTANT: run MapAnything on the raw rectified image file (data_rect).
                # The SSCBenchDataset "imgs" are resized to (192,640) with aspect distortion, which can
                # severely harm MapAnything's metric depth/scale. Using raw files matches MapAnything's
                # intended preprocessing path (load_images).
                seq_name = str(seq)
                if "drive_" not in seq_name:
                    seq_name = f"2013_05_28_drive_{int(seq):04d}_sync"
                img_path = Path(args.sscbench_data_root) / "data_2d_raw" / seq_name / "image_00" / "data_rect" / f"{int(fid):06d}.png"
                if not img_path.exists():
                    continue

                # pose -> world (nearest available), then cam->world
                if args.pose_source == "none":
                    pose_to_world_k = np.eye(4, dtype=np.float32)
                else:
                    pose_to_world_k = _nearest_pose(poses_cache.get(str(seq), {}), int(fid))
                    if pose_to_world_k is None:
                        if str(seq) not in poses_cache:
                            poses_cache[str(seq)] = _load_kitti360_poses(pose_root, str(seq))
                        pose_to_world_k = _nearest_pose(poses_cache[str(seq)], int(fid))
                    if pose_to_world_k is None:
                        pose_to_world_k = np.eye(4, dtype=np.float32)

                c2w_k = (torch.as_tensor(pose_to_world_k, dtype=torch.float32) @ T_cam_to_pose).to(device)

                # Keep tensor window only as a fallback; MapAnything init uses img_paths_win.
                imgk = _try_load_processed_img(dataset, seq=str(seq), frame_id=fid)
                if imgk is not None:
                    imgs_win.append(imgk.to(device))
                Ks_win.append(K0)
                c2w_win.append(c2w_k)
                img_paths_win.append(str(img_path))

            if len(imgs_win) == 0:
                imgs_win = [img0]
                Ks_win = [K0]
                c2w_win = [cam_to_world]
            if len(img_paths_win) == 0:
                # As a last resort, fall back to processed tensors (may degrade MapAnything scale).
                img_paths_win = []

            # Stable cache key per (sequence, frame_id).
            if isinstance(seq, (int, np.integer)):
                seq_id_int = int(seq)
            else:
                seq_id_int = 0
                m = re.search(r"drive_(\d{4})_sync", str(seq))
                if m is not None:
                    seq_id_int = int(m.group(1))
                else:
                    # fallback: extract last group of digits if present
                    m2 = re.search(r"(\d+)$", str(seq))
                    if m2 is not None:
                        try:
                            seq_id_int = int(m2.group(1))
                        except Exception:
                            seq_id_int = 0

            sample_index = int(seq_id_int) * 1_000_000 + int(frame_id)

            # Use raw images from dataset if available (avoids resizing issues)
            images_raw_01_chw = []
            if "imgs_raw" in data:
                # data["imgs_raw"] is a list of tensors
                for img in data["imgs_raw"]:
                    if torch.is_tensor(img):
                        images_raw_01_chw.append(img.to(device))
            
            # If we have raw images, prefer them over paths (which might trigger default resizing in load_images)
            if len(images_raw_01_chw) > 0:
                img_paths_win = []

            init_points_world = map_init.infer_points_from_window(
                images_m11_chw=imgs_win,
                images_raw_01_chw=images_raw_01_chw if len(images_raw_01_chw) > 0 else None,
                K_norm_list=Ks_win,
                camtoworld_list=c2w_win,
                image_paths=img_paths_win,
                sample_index=sample_index,
                force_recompute=bool(args.mapanything_force_recompute),
                cached_only=bool(args.mapanything_cached_only),
            )

        with torch.no_grad():
            out = model(
                imgs_m11_chw=[img0],
                Ks_norm=[K0],
                camtoworlds=[cam_to_world],
                encoder_index=0,
                supervise_indices=[],
                init_points_world=init_points_world,
            )

            means_world = out["means_world"]
            opacities = out["opacities"].clamp(0, 1)
            sem_logits = out["sem_logits"]

            # SSCBench grid is defined in velodyne coordinates.
            # Most of the stack assumes Gaussians are in WORLD coords, so we use world->velo from poses.
            # However, if initialization/model outputs end up in a local (camera/relative) frame,
            # transforming with world->velo will push everything far outside the voxel grid.
            # We detect this by comparing inside-grid ratios for two candidate transforms:
            #  - pose-derived world->velo
            #  - camera-local cam->velo (treat predicted points as camera coords)
            ones = torch.ones((means_world.shape[0], 1), device=device, dtype=means_world.dtype)
            means_h = torch.cat([means_world, ones], dim=1)

            means_velo_pose = (world_to_velo_pose @ means_h.t()).t()[:, :3]
            world_to_velo_cam = cam_to_velo.to(device=device, dtype=means_world.dtype)
            means_velo_cam = (world_to_velo_cam @ means_h.t()).t()[:, :3]

            # Compute inside-grid ratios for both candidates.
            g = grid
            origin = torch.tensor(g.origin, device=device, dtype=means_world.dtype)
            voxel = float(g.voxel_size)
            X, Y, Z = map(int, g.dims_xyz)
            maxs = origin + torch.tensor([X * voxel, Y * voxel, Z * voxel], device=device, dtype=means_world.dtype)

            inside_pose = (
                (means_velo_pose[:, 0] >= origin[0])
                & (means_velo_pose[:, 0] < maxs[0])
                & (means_velo_pose[:, 1] >= origin[1])
                & (means_velo_pose[:, 1] < maxs[1])
                & (means_velo_pose[:, 2] >= origin[2])
                & (means_velo_pose[:, 2] < maxs[2])
            )
            inside_cam = (
                (means_velo_cam[:, 0] >= origin[0])
                & (means_velo_cam[:, 0] < maxs[0])
                & (means_velo_cam[:, 1] >= origin[1])
                & (means_velo_cam[:, 1] < maxs[1])
                & (means_velo_cam[:, 2] >= origin[2])
                & (means_velo_cam[:, 2] < maxs[2])
            )

            inside_ratio_pose = float(inside_pose.float().mean().item())
            inside_ratio_cam = float(inside_cam.float().mean().item())

            # Heuristic: if pose-derived mapping puts essentially everything off-grid but cam->velo doesn't,
            # assume the predicted points are in camera-local coordinates.
            use_cam_frame = (inside_ratio_pose < 0.01) and (inside_ratio_cam > max(0.05, inside_ratio_pose + 0.05))

            world_to_velo = (world_to_velo_cam if use_cam_frame else world_to_velo_pose)
            means_velo_common = (means_velo_cam if use_cam_frame else means_velo_pose)

            pred_occ_hard = None
            if results_hard is not None:
                origin_t = torch.tensor(g.origin, device=device, dtype=means_velo_common.dtype)
                voxel = float(g.voxel_size)
                X, Y, Z = map(int, g.dims_xyz)

                ijk = torch.floor((means_velo_common - origin_t[None]) / voxel).to(torch.int64)
                ix, iy, iz = ijk[:, 0], ijk[:, 1], ijk[:, 2]
                inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
                occ_flat = torch.zeros((X * Y * Z,), device=device, dtype=torch.bool)
                if inside.any():
                    flat = (ix[inside] * (Y * Z) + iy[inside] * Z + iz[inside]).to(torch.int64)
                    occ_flat[flat] = True
                pred_occ_hard = occ_flat.view(X, Y, Z).detach().cpu().numpy()

            if str(args.voxel_mode) == "opacity_sum":
                # NOTE: This is intentionally sparse (hard assignment to the containing voxel).
                sigmas, segs_city = voxelize_gaussians_to_sscbench(
                    means_world=means_world,
                    opacities=opacities,
                    sem_logits=sem_logits,
                    world_to_velo=world_to_velo,
                    sigma_cutoff=float(args.sigma_cutoff),
                    pred_sem_channels=int(args.pred_sem_channels),
                    device=device,
                )
            else:
                # Default: scale-aware Gaussian kernel splat (denser, closer to continuous sigma fields).
                scales = out["scales"]
                sem_probs = torch.softmax(sem_logits, dim=-1)

                means_velo = means_velo_common

                if bool(args.debug_stats):
                    inside_ratio = inside_ratio_cam if use_cam_frame else inside_ratio_pose
                    mv = means_velo.detach().cpu().numpy()
                    mw = means_world.detach().cpu().numpy()
                    op = opacities.detach().cpu().numpy().reshape(-1)
                    sc = out["scales"].detach().cpu().numpy()
                    pd = out.get("pred_depth")
                    if pd is not None:
                        pd_np = pd.detach().cpu().numpy().reshape(-1)
                        pd_s = f"pred_depth[min/med/max]={pd_np.min():.2f}/{np.median(pd_np):.2f}/{pd_np.max():.2f}"
                    else:
                        pd_s = "pred_depth[n/a]"

                    c2w_t = cam_to_world.detach().cpu().numpy()

                    print(
                        f"[debug] seq={seq} frame={frame_id} P={mv.shape[0]} inside_grid={inside_ratio:.3f} "
                        f"frame_assumption={'cam' if use_cam_frame else 'world'} "
                        f"inside_pose={inside_ratio_pose:.3f} inside_cam={inside_ratio_cam:.3f}\n"
                        f"        means_world[min]={mw.min(axis=0)} max={mw.max(axis=0)}\n"
                        f"        means_velo[x,y,z min]={mv.min(axis=0)} max={mv.max(axis=0)}\n"
                        f"        cam_to_world[t]={c2w_t[:3, 3]}\n"
                        f"        opacity[min/med/mean/p90/max]={op.min():.4g}/{np.median(op):.4g}/{op.mean():.4g}/{np.quantile(op,0.9):.4g}/{op.max():.4g} "
                        f"scales[m med max]={np.min(sc):.3g}/{np.median(sc):.3g}/{np.max(sc):.3g} {pd_s}"
                    )

                _sigma_sum, sigma, sem_vox = splat_gaussians_to_voxels_gaussian_kernel(
                    means_xyz=means_velo,
                    opacities=opacities.view(-1, 1),
                    scales_xyz=scales,
                    sem_probs=sem_probs,
                    grid=grid,
                    max_radius=int(args.voxel_max_radius),
                )

                # Our splatter returns (Z,Y,X). SSCBench evaluator expects (X,Y,Z).
                sigma_xyz = sigma.permute(2, 1, 0).contiguous()  # (X,Y,Z)
                sigmas = sigma_xyz.detach().cpu().numpy().astype(np.float32)
                occ = sigmas > float(args.sigma_cutoff)

                if bool(args.debug_stats):
                    s = sigmas
                    qs = np.quantile(s, [0.0, 0.5, 0.9, 0.99, 0.999]).astype(np.float64)
                    occ_rate = float(np.mean(s > float(args.sigma_cutoff)))
                    # GT occupancy rate under full mask (51.2m crop computed later; here is full grid).
                    gt_occ = float(np.mean((target > 0) & (target != 255) & (fov_mask.astype(bool))))
                    print(
                        f"[debug] sigma[min/med/p90/p99/p99.9]={qs[0]:.3g}/{qs[1]:.3g}/{qs[2]:.3g}/{qs[3]:.3g}/{qs[4]:.3g} "
                        f"occ_rate@cut({args.sigma_cutoff})={occ_rate:.4f} gt_occ_rate(full_mask)={gt_occ:.4f}"
                    )

                sem_xyz = sem_vox.permute(0, 3, 2, 1).contiguous()  # (C,X,Y,Z)
                sem_vox_np = sem_xyz.detach().cpu().numpy()
                segs_city = np.argmax(sem_vox_np, axis=0).astype(np.int32)  # (X,Y,Z)
                # IMPORTANT: do not mark empty voxels as 255 (ignore), because SSCBench occupancy
                # metrics treat y_pred>0 as occupied. Use sky trainId=10 as the "empty" label.
                segs_city[~occ] = 10

        segs_city_int = segs_city.astype(int)

        # convert_voxels expects every value to exist in the mapping dict.
        # Our network can predict up to `pred_sem_channels` indices which may include
        # trainIds not present in the mapping (e.g., 19). Map unknowns to sky=10.
        mapping = label_maps["cityscapes_to_label"]
        keys = np.fromiter(mapping.keys(), dtype=np.int32)
        known = np.isin(segs_city_int, keys)

        # Track ignore voxels separately (should remain ignore=255).
        ignore_mask = segs_city_int == 255

        if (not known.all()) or ignore_mask.any():
            segs_city_int = segs_city_int.copy()
            segs_city_int[~known] = 10
            segs_city_int[ignore_mask] = 10

        segs = convert_voxels(segs_city_int, mapping)
        if ignore_mask.any():
            segs = segs.astype(np.int32, copy=False)
            segs[ignore_mask] = 255

        for size in SIZES:
            num_voxels = int(size // 0.2)

            _segs = segs[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]
            _target = target[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]
            _fov = fov_mask[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]
            _sigmas = sigmas[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]

            if results_hard is not None and pred_occ_hard is not None:
                _pred_occ = pred_occ_hard[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]
                _tp_h, _fp_h, _tn_h, _fn_h = compute_occupancy_numbers(
                    _pred_occ.astype(np.int32),
                    _target,
                    _fov,
                    occ_from="segs",
                )
                results_hard[size]["tp"] += int(_tp_h)
                results_hard[size]["fp"] += int(_fp_h)
                results_hard[size]["tn"] += int(_tn_h)
                results_hard[size]["fn"] += int(_fn_h)

            # Occupancy sweep: compute counts for all thresholds with a single mask extraction.
            _tp_v, _fp_v, _tn_v, _fn_v = _accumulate_occ_sweep(
                sigmas_xyz=_sigmas,
                target_xyz=_target,
                fov_xyz=_fov,
                thresholds=thresholds,
            )
            _tp_seg, _fp_seg, _tn_seg, _fn_seg = compute_occupancy_numbers_segmentation(
                _segs, _target, _fov, labels=label_maps["labels"]
            )

            results[size]["tp"] += _tp_v
            results[size]["fp"] += _fp_v
            results[size]["tn"] += _tn_v
            results[size]["fn"] += _fn_v

            results[size]["tp_seg"] += _tp_seg
            results[size]["fp_seg"] += _fp_seg
            results[size]["tn_seg"] += _tn_seg
            results[size]["fn_seg"] += _fn_seg

    # Print summary like S4C
    for size in SIZES:
        tp_v = results[size]["tp"].astype(np.float64)
        fp_v = results[size]["fp"].astype(np.float64)
        fn_v = results[size]["fn"].astype(np.float64)

        iou_occ_v = tp_v / (tp_v + fp_v + fn_v + 1e-6)
        if len(thresholds) == 1:
            print(
                f"size={size}m occ_iou={float(iou_occ_v[0]):.4f} "
                f"tp={int(tp_v[0])} fp={int(fp_v[0])} fn={int(fn_v[0])} "
                f"(sigma_cutoff={thresholds[0]})"
            )
        else:
            best_idx = int(np.argmax(iou_occ_v))
            print(
                f"size={size}m best_occ_iou={float(iou_occ_v[best_idx]):.4f} "
                f"best_sigma={thresholds[best_idx]} "
                f"tp={int(tp_v[best_idx])} fp={int(fp_v[best_idx])} fn={int(fn_v[best_idx])}"
            )
            for thr, iou, tp1, fp1, fn1 in zip(thresholds, iou_occ_v, tp_v, fp_v, fn_v):
                print(f"  sigma={thr:<10g} occ_iou={float(iou):.4f} tp={int(tp1)} fp={int(fp1)} fn={int(fn1)}")

    if results_hard is not None:
        print("[sanity_hard_voxelize] Gaussian centers hard-voxelized occupancy")
        for size in SIZES:
            tp = float(results_hard[size]["tp"])
            fp = float(results_hard[size]["fp"])
            fn = float(results_hard[size]["fn"])
            iou = tp / (tp + fp + fn + 1e-6)
            print(f"size={size}m hard_occ_iou={iou:.4f} tp={int(tp)} fp={int(fp)} fn={int(fn)}")


if __name__ == "__main__":
    main()
