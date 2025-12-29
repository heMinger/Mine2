"""One-step diagnostic: MapAnything depth/points vs SSCBench GT occupancy (no LiDAR).

This script:
- runs MapAnything on one SSCBench sample
- builds a pseudo GT depth map by projecting occupied voxels into the camera
- reports depth error statistics and range coverage

It uses SSCBench voxel labels only (required for evaluation), not raw LiDAR.

Example:
  conda run -n vgm-gs-s4c --no-capture-output \
    python -m ssc_gs.scripts.debug_mapanything_vs_ssc_gt_depth \
      --ssc_root /data/lmh_data/sscbench-kitti \
      --voxel_gt_path /data/lmh_data/sscbench-kitti/preprocess/labels \
      --pose_root /data/lmh_data/KITTI360/poses \
      --seq_id 5 --frame_id 0 \
      --hf_model_name /home/lmh/VGM-GS-S4C/map-anything/ckpt/map-anything-apache \
      --local_files_only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

import torch.nn.functional as F

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from s4c.scripts.benchmarks.sscbench.sscbench_dataset import SSCBenchDataset
from ssc_gs.scripts.evaluate_model_sscbench import _load_kitti360_poses, _nearest_pose
from ssc_gs.utils.camera import kitti360_normK_to_pixelK


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ssc_root", type=str, required=True)
    p.add_argument("--voxel_gt_path", type=str, required=True)
    p.add_argument("--pose_root", type=str, required=True)
    p.add_argument("--seq_id", type=int, required=True)
    p.add_argument("--frame_id", type=int, required=True)
    p.add_argument("--pose_source", type=str, default="poses_txt", choices=("poses_txt", "cam0_to_world"))

    p.add_argument("--hf_model_name", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bf16")

    # For ablations: do not use dataset camera params as MapAnything inputs.
    # Note: pseudo-GT projection still uses real KITTI360 pose + dataset K, otherwise the comparison is undefined.
    p.add_argument("--ma_ignore_calibration_inputs", action="store_true")
    p.add_argument("--ma_ignore_pose_inputs", action="store_true")
    p.add_argument("--ma_use_identity_pose_input", action="store_true")
    p.add_argument("--ma_use_identity_intrinsics_input", action="store_true")

    p.add_argument("--target_h", type=int, default=192)
    p.add_argument("--target_w", type=int, default=640)

    p.add_argument("--max_vox_points", type=int, default=200_000)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _lazy_load_mapanything(hf_model_name: str, local_files_only: bool, device: torch.device) -> Any:
    from mapanything.models import MapAnything  # pyright: ignore[reportMissingImports]

    model = MapAnything.from_pretrained(hf_model_name, local_files_only=bool(local_files_only)).to(device)
    model.eval()
    return model


def _load_kitti360_cam0_to_world(pose_root: Path, sequence: str) -> dict[int, np.ndarray]:
    p = pose_root / sequence / "cam0_to_world.txt"
    arr = np.loadtxt(p)
    if arr.ndim == 1:
        arr = arr[None, :]
    out: dict[int, np.ndarray] = {}
    for row in arr:
        fid = int(row[0])
        T = row[1:].astype(np.float32).reshape(4, 4)
        out[fid] = T
    return out


def _nearest_key(d: dict[int, np.ndarray], frame_id: int) -> int | None:
    if not d:
        return None
    if frame_id in d:
        return frame_id
    keys = sorted(d.keys())
    import bisect

    pos = bisect.bisect_left(keys, frame_id)
    cand: list[int] = []
    if pos < len(keys):
        cand.append(keys[pos])
    if pos > 0:
        cand.append(keys[pos - 1])
    if not cand:
        return None
    return min(cand, key=lambda k: abs(k - frame_id))


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _zbuffer_depth_from_world_points(
    pts_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into camera and compute per-pixel min Z (camera-forward).

    Returns:
      depth_gt: (H,W) float32 with NaN for empty pixels
      valid_mask: (H,W) bool
    """
    if pts_world.shape[0] == 0:
        depth = np.full((H, W), np.nan, dtype=np.float32)
        return depth, np.zeros((H, W), dtype=bool)

    if pts_world.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts_world.shape[0], size=max_points, replace=False)
        pts_world = pts_world[idx]

    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([pts_world.astype(np.float32), np.ones((pts_world.shape[0], 1), np.float32)], axis=1)
    cam = (w2c @ pts_h.T).T[:, :3]

    z = cam[:, 2]
    in_front = z > 1e-3
    cam = cam[in_front]
    z = z[in_front]
    if cam.shape[0] == 0:
        depth = np.full((H, W), np.nan, dtype=np.float32)
        return depth, np.zeros((H, W), dtype=bool)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[inside]
    vi = vi[inside]
    z = z[inside]

    depth = np.full((H, W), np.nan, dtype=np.float32)
    if z.shape[0] == 0:
        return depth, np.zeros((H, W), dtype=bool)

    # z-buffer: take minimum z per pixel
    # Use sorting for deterministic min reduce.
    key = vi.astype(np.int64) * np.int64(W) + ui.astype(np.int64)
    order = np.argsort(key)
    key = key[order]
    z = z[order]

    # group by key
    unique_keys, first_idx = np.unique(key, return_index=True)
    # min z within each group; since sorted by key but not by z, compute per group
    # Do it with a loop over unique keys; number of unique pixels is manageable.
    for k, start in zip(unique_keys, first_idx):
        end = start + 1
        while end < key.shape[0] and key[end] == k:
            end += 1
        zmin = float(np.min(z[start:end]))
        y = int(k // W)
        x = int(k % W)
        depth[y, x] = zmin

    valid = np.isfinite(depth)
    return depth, valid


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    seq_name = f"2013_05_28_drive_{args.seq_id:04d}_sync"

    ds = SSCBenchDataset(
        data_path=args.ssc_root,
        voxel_gt_path=args.voxel_gt_path,
        sequences=(args.seq_id,),
        target_image_size=(args.target_h, args.target_w),
        return_stereo=False,
        frame_count=1,
        color_aug=False,
    )

    # Robustly locate the datapoint by (sequence, frame_id).
    # SSCBenchDataset loads both image and voxel GT based on this internal (seq,id) tuple.
    idx = None
    if hasattr(ds, "_datapoints"):
        for i, (seq, fid, _is_right) in enumerate(ds._datapoints):
            if str(seq) == str(seq_name) and int(fid) == int(args.frame_id):
                idx = int(i)
                break
    if idx is None:
        raise RuntimeError(
            f"Could not find (seq={seq_name}, frame_id={args.frame_id}) in SSCBenchDataset._datapoints. "
            "This usually means the GT label file is missing for that frame."
        )

    img_path = Path(args.ssc_root) / "data_2d_raw" / seq_name / "image_00" / "data_rect" / f"{args.frame_id:06d}.png"
    gt_path = Path(args.voxel_gt_path) / seq_name / f"{args.frame_id:06d}_1_1.npy"
    print(f"[Align] dataset_index={idx} seq={seq_name} frame_id={args.frame_id}")
    print(f"[Align] expected image={img_path}")
    print(f"[Align] expected voxel_gt={gt_path}")

    sample = ds[idx]
    img_m11 = sample["imgs"][0]
    K_norm = torch.as_tensor(sample["projs"][0], dtype=torch.float32)
    H, W = int(img_m11.shape[1]), int(img_m11.shape[2])

    cal = ds._calibs
    T_cam_to_pose = torch.as_tensor(cal["T_cam_to_pose"]["00"], dtype=torch.float32)
    T_velo_to_pose = torch.as_tensor(cal["T_velo_to_pose"], dtype=torch.float32)

    # Pose handling: prefer exact per-frame cam0_to_world when requested.
    pose_root = Path(args.pose_root)
    if args.pose_source == "cam0_to_world":
        cam0 = _load_kitti360_cam0_to_world(pose_root, seq_name)
        best_id = _nearest_key(cam0, int(args.frame_id))
        if best_id is None:
            raise RuntimeError(f"No cam0_to_world pose found for {seq_name}")
        cam_to_world = cam0[best_id].astype(np.float32)
        # Recover pose_to_world (IMU/GPS frame) from cam_to_world and calibration.
        pose_to_world = (torch.as_tensor(cam_to_world) @ torch.linalg.inv(T_cam_to_pose)).to(torch.float32)
        velo_to_world = (pose_to_world @ T_velo_to_pose).cpu().numpy()
        if int(best_id) != int(args.frame_id):
            print(f"[Pose] cam0_to_world missing frame {args.frame_id}, using nearest {best_id}")
    else:
        poses = _load_kitti360_poses(pose_root, seq_name)
        best_np = _nearest_pose(poses, int(args.frame_id))
        if best_np is None:
            raise RuntimeError(f"No poses.txt pose found for {seq_name} frame {args.frame_id}")
        # Report if we had to use nearest.
        best_id = _nearest_key(poses, int(args.frame_id))
        if best_id is not None and int(best_id) != int(args.frame_id):
            print(f"[Pose] poses.txt missing frame {args.frame_id}, using nearest {best_id}")
        pose_to_world = torch.as_tensor(best_np, dtype=torch.float32)
        cam_to_world = (pose_to_world @ T_cam_to_pose).cpu().numpy()
        velo_to_world = (pose_to_world @ T_velo_to_pose).cpu().numpy()

    # Build inputs for MapAnything
    from mapanything.utils.image import preprocess_inputs  # pyright: ignore[reportMissingImports]

    model = _lazy_load_mapanything(args.hf_model_name, args.local_files_only, device)

    img_01 = (img_m11.float() + 1.0) * 0.5
    img_hwc_255 = (img_01.permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.float32)

    K_px = kitti360_normK_to_pixelK(K_norm.float(), width=W, height=H)

    # Inputs to MapAnything (can be optionally ignored by model via flags).
    ma_pose_in = cam_to_world
    ma_K_in = K_px.cpu().numpy()
    if args.ma_use_identity_pose_input:
        ma_pose_in = np.eye(4, dtype=np.float32)
    if args.ma_use_identity_intrinsics_input:
        ma_K_in = np.eye(3, dtype=np.float32)

    views = [
        {
            "img": img_hwc_255.cpu().numpy(),
            "intrinsics": ma_K_in,
            "camera_poses": ma_pose_in,
            "is_metric_scale": True,
        }
    ]

    processed = preprocess_inputs(views, verbose=False)
    outputs = model.infer(
        processed,
        memory_efficient_inference=True,
        ignore_calibration_inputs=bool(args.ma_ignore_calibration_inputs),
        ignore_pose_inputs=bool(args.ma_ignore_pose_inputs),
        ignore_depth_inputs=False,
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
        use_amp=True,
        amp_dtype=args.amp_dtype,
        apply_mask=True,
        mask_edges=True,
    )

    pred = outputs[0]
    print("[MapAnything] pred keys:", sorted(list(pred.keys())))

    # Print pose/intrinsics consistency between what we fed and what MapAnything outputs.
    c2w_in = cam_to_world.astype(np.float32)
    K_in = K_px.cpu().numpy().astype(np.float32)

    # Gather depth-like fields for reporting.
    depth_fields: dict[str, np.ndarray] = {}
    if "depth_z" in pred:
        depth_fields["depth_z"] = _as_numpy(pred["depth_z"])[0].squeeze(-1).astype(np.float32)
    if "depth_along_ray" in pred:
        depth_fields["depth_along_ray"] = _as_numpy(pred["depth_along_ray"])[0].squeeze(-1).astype(np.float32)
    if "pts3d_cam" in pred:
        depth_fields.setdefault("pts3d_cam_z", _as_numpy(pred["pts3d_cam"])[0][..., 2].astype(np.float32))
    if not depth_fields:
        raise RuntimeError("MapAnything output contains no depth-like field.")

    # Pick a primary depth field for the error comparison.
    primary_name = "depth_z" if "depth_z" in depth_fields else sorted(depth_fields.keys())[0]
    depth_pred_np = depth_fields[primary_name]
    Hp, Wp = depth_pred_np.shape

    K_pred = _as_numpy(pred.get("intrinsics", np.asarray(K_px.cpu().numpy(), np.float32)))[0].astype(np.float32)
    c2w_pred = _as_numpy(pred.get("camera_poses", np.asarray(cam_to_world, np.float32)))[0].astype(np.float32)

    print("[Pose] cam_to_world input t=", c2w_in[:3, 3], "| pred t=", c2w_pred[:3, 3])
    rel = np.linalg.inv(c2w_in) @ c2w_pred
    rel_t = rel[:3, 3]
    # rotation difference angle (rough)
    tr = float(np.trace(rel[:3, :3]))
    tr = max(-1.0, min(3.0, tr))
    angle = float(np.arccos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))
    print(f"[Pose] rel(in->pred) |t|={float(np.linalg.norm(rel_t)):.3f}m rot_angle={angle:.4f}rad")

    print(
        "[K] input fx,fy,cx,cy=",
        (float(K_in[0, 0]), float(K_in[1, 1]), float(K_in[0, 2]), float(K_in[1, 2])),
        "| pred fx,fy,cx,cy=",
        (float(K_pred[0, 0]), float(K_pred[1, 1]), float(K_pred[0, 2]), float(K_pred[1, 2])),
        f"| pred image HxW={Hp}x{Wp}",
    )

    if "metric_scaling_factor" in pred:
        msf = _as_numpy(pred["metric_scaling_factor"])[0]
        try:
            msf_val = float(msf.reshape(-1)[0])
        except Exception:
            msf_val = float(np.asarray(msf).reshape(-1)[0])
        print(f"[MapAnything] metric_scaling_factor={msf_val}")
    else:
        msf_val = 1.0

    def qstats(x: np.ndarray) -> str:
        if x.size == 0:
            return "(empty)"
        qs = np.quantile(x, [0.01, 0.1, 0.5, 0.9, 0.99])
        return "q01/q10/q50/q90/q99=" + ", ".join([f"{float(v):.3f}" for v in qs])

    mask = pred.get("mask", None)
    if mask is not None:
        mask_np = _as_numpy(mask)[0].squeeze(-1).astype(bool)
    else:
        mask_np = np.ones((Hp, Wp), dtype=bool)

    # Compare in INPUT image space: upsample MapAnything depth/mask to (H,W).
    depth_pred_t = torch.from_numpy(depth_pred_np)[None, None]
    depth_up = (
        F.interpolate(depth_pred_t, size=(H, W), mode="bilinear", align_corners=False)
        .squeeze(0)
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )
    mask_t = torch.from_numpy(mask_np.astype(np.float32))[None, None]
    mask_up = (
        F.interpolate(mask_t, size=(H, W), mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )
    mask_up = mask_up >= 0.5

    finite = np.isfinite(depth_up)
    valid_pred = finite & mask_up

    # Load GT occupancy voxels and create world points
    vox_path = Path(args.voxel_gt_path) / seq_name / f"{args.frame_id:06d}_1_1.npy"
    vox = np.load(str(vox_path))
    valid = vox != 255
    occ = (vox > 0) & valid

    idxs = np.stack(np.nonzero(occ), axis=1)  # (N,3) x,y,z
    origin = np.array([0.0, -25.6, -2.0], np.float32)
    voxel = np.float32(0.2)
    centers_velo = origin[None] + (idxs.astype(np.float32) + 0.5) * voxel

    # velo -> world
    centers_h = np.concatenate([centers_velo, np.ones((centers_velo.shape[0], 1), np.float32)], axis=1)
    centers_world = (velo_to_world @ centers_h.T).T[:, :3]

    # Sanity: project GT voxels with INPUT pose/K into INPUT image space.
    depth_gt_in, valid_gt_in = _zbuffer_depth_from_world_points(
        centers_world,
        c2w=c2w_in,
        K=K_in,
        H=H,
        W=W,
        max_points=int(args.max_vox_points),
    )
    print(
        f"[Sanity GT->input image] pseudo depth pixels: {int(valid_gt_in.sum())} / {H*W} ({float(valid_gt_in.mean()):.6f})"
    )
    if int(valid_gt_in.sum()) > 0:
        gt_in_vals = depth_gt_in[valid_gt_in]
        print(f"[Sanity pseudoGT input depth] n={gt_in_vals.size} {qstats(gt_in_vals)}")

    joint = valid_gt_in & valid_pred
    n_gt_pix = int(valid_gt_in.sum())
    n_joint = int(joint.sum())
    print(f"[Joint] pixels with both pred(depth upsampled) & pseudoGT: {n_joint} / {H*W} ({n_joint/(H*W):.4f})")

    pred_vals = depth_up[valid_pred]
    gt_vals = depth_gt_in[valid_gt_in]

    print(f"[Pred depth] field={primary_name} (upsampled) n={pred_vals.size} {qstats(pred_vals)}")
    print(f"[PseudoGT depth] (input projection) n={gt_vals.size} {qstats(gt_vals)}")

    # Additional depth fields + scaling-factor variants.
    for name, d in depth_fields.items():
        d = d.astype(np.float32)
        vals = d[np.isfinite(d) & (d != 0)]
        print(f"[Pred depth raw] {name} n={vals.size} {qstats(vals)}")
        if float(msf_val) != 1.0:
            vals_s = (d * float(msf_val))[np.isfinite(d) & (d != 0)]
            print(f"[Pred depth scaled] {name}*msf n={vals_s.size} {qstats(vals_s)}")

    for t in (5, 10, 20, 30, 40):
        frac = float((pred_vals > t).mean()) if pred_vals.size else 0.0
        frac_s = float(((pred_vals * float(msf_val)) > t).mean()) if pred_vals.size else 0.0
        print(f"[Pred depth] fraction > {t}m: {frac:.4f} | after *msf: {frac_s:.4f}")

    if n_joint > 0:
        err_abs = np.abs(depth_up[joint] - depth_gt_in[joint])
        err_rel = err_abs / np.clip(np.abs(depth_gt_in[joint]), 1e-3, None)
        print(f"[Error abs] mean={float(err_abs.mean()):.3f} med={float(np.median(err_abs)):.3f} {qstats(err_abs)}")
        print(f"[Error rel] mean={float(err_rel.mean()):.3f} med={float(np.median(err_rel)):.3f} {qstats(err_rel)}")
    else:
        print("[Error] No overlapping pixels between pred depth and pseudoGT depth.")

    # Also: compare pointcloud range in velodyne space using INPUT pose (avoid pred pose normalization).
    if "pts3d_cam" in pred:
        pts_cam = _as_numpy(pred["pts3d_cam"])[0].reshape(-1, 3).astype(np.float32)
        m = mask_np.reshape(-1)
        pts_cam = pts_cam[m]

        # cam -> world (input) -> velo
        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), np.float32)], axis=1)
        pts_world = (c2w_in @ pts_cam_h.T).T[:, :3]

        world_to_velo = np.linalg.inv(velo_to_world)
        pts_world_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), np.float32)], axis=1)
        pts_velo = (world_to_velo @ pts_world_h.T).T[:, :3]

        if pts_velo.shape[0] > 0:
            print("[Pred pts velo bbox] min", pts_velo.min(axis=0), "max", pts_velo.max(axis=0), "mean", pts_velo.mean(axis=0))
            print("[GT occ velo bbox]  min", centers_velo.min(axis=0), "max", centers_velo.max(axis=0), "mean", centers_velo.mean(axis=0))

            if float(msf_val) != 1.0:
                pts_cam_s = pts_cam * float(msf_val)
                pts_cam_hs = np.concatenate([pts_cam_s, np.ones((pts_cam_s.shape[0], 1), np.float32)], axis=1)
                pts_world_s = (c2w_in @ pts_cam_hs.T).T[:, :3]
                pts_world_hs = np.concatenate([pts_world_s, np.ones((pts_world_s.shape[0], 1), np.float32)], axis=1)
                pts_velo_s = (world_to_velo @ pts_world_hs.T).T[:, :3]
                print("[Pred pts*msf velo bbox] min", pts_velo_s.min(axis=0), "max", pts_velo_s.max(axis=0), "mean", pts_velo_s.mean(axis=0))

            for t in (5, 10, 20, 30, 40):
                fracx = float((pts_velo[:, 0] > t).mean())
                print(f"[Pred pts] fraction x>{t}m: {fracx:.4f}")
        else:
            print("[Pred pts] empty after mask")


if __name__ == "__main__":
    main()
