from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import bisect
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from ssc_gs.train import build_scene_from_checkpoint_state

# Reuse SSCBench utilities from S4C (no edits to that repo)
from s4c.scripts.benchmarks.sscbench.point_utils import get_fov_mask
from s4c.scripts.benchmarks.sscbench.sscbench_dataset import SSCBenchDataset


logging.basicConfig(level=logging.INFO)

# Match S4C evaluator defaults (evaluation scene sizes at 0.2m)
SIZES = (12.8, 25.6, 51.2)
VOXEL_SIZE = 0.2


def _load_kitti360_poses(pose_root: Path, sequence: str) -> Dict[int, np.ndarray]:
    """Load KITTI-360 poses for a sequence.

    KITTI-360 `poses.txt` stores pose-to-world (NOT cam-to-world).
    Returns a dict: frame_id (int) -> 4x4 pose-to-world matrix.
    """
    pose_file = pose_root / sequence / "poses.txt"
    pose_data = np.loadtxt(pose_file)
    ids = pose_data[:, 0].astype(int)
    mats = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
    mats = np.concatenate((mats, np.zeros_like(mats[:, :1, :])), axis=1)
    mats[:, 3, 3] = 1.0
    return {int(i): mats[k] for k, i in enumerate(ids)}


def _nearest_pose(poses: Dict[int, np.ndarray], frame_id: int) -> np.ndarray | None:
    if not poses:
        return None
    if frame_id in poses:
        return poses[frame_id]
    keys = sorted(poses.keys())
    pos = bisect.bisect_left(keys, frame_id)
    cand = []
    if pos < len(keys):
        cand.append(keys[pos])
    if pos > 0:
        cand.append(keys[pos - 1])
    if not cand:
        return None
    best = min(cand, key=lambda k: abs(k - frame_id))
    return poses[best]


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Elementwise numer/denom with 0 where denom==0 (avoids NaNs)."""
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    out = np.zeros_like(numer, dtype=np.float64)
    np.divide(numer, denom, out=out, where=denom != 0)
    return out


def _load_label_maps() -> Dict:
    label_maps_path = Path(__file__).resolve().parents[2] / "s4c" / "scripts" / "benchmarks" / "sscbench" / "label_maps.yaml"
    with open(label_maps_path, "r") as f:
        return yaml.safe_load(f)


def convert_voxels(arr: np.ndarray, map_dict: Dict[int, int]) -> np.ndarray:
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)


def identify_additional_invalids(target: np.ndarray) -> np.ndarray:
    # Same logic as S4C evaluate_model_sscbench.py
    _t = np.concatenate([np.zeros([256, 256, 1]), target], axis=2)
    invalids = np.cumsum(np.logical_and(_t != 255, _t != 0), axis=2)[:, :, :32] == 0
    invalids[:, :, 7:] = 0
    invalids[target != 0] = 0
    return invalids


def compute_occupancy_numbers(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    fov_mask: np.ndarray,
    *,
    sigmas: np.ndarray | None = None,
    occ_from: str = "segs",
    occ_thresh: float = 0.5,
) -> Tuple[int, int, int, int]:
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    occ_true = y_true[mask] > 0

    if occ_from == "segs":
        occ_pred = y_pred[mask] > 0
    elif occ_from == "sigmas":
        if sigmas is None:
            raise ValueError("sigmas must be provided when occ_from='sigmas'")
        sig_f = np.asarray(sigmas, dtype=np.float32).flatten()
        occ_pred = sig_f[mask] > float(occ_thresh)
    else:
        raise ValueError(f"Unknown occ_from={occ_from!r}; expected 'segs' or 'sigmas'")

    tp = int(np.sum(np.logical_and(occ_true == 1, occ_pred == 1)))
    fp = int(np.sum(np.logical_and(occ_true == 0, occ_pred == 1)))
    fn = int(np.sum(np.logical_and(occ_true == 1, occ_pred == 0)))
    tn = int(np.sum(np.logical_and(occ_true == 0, occ_pred == 0)))

    return tp, fp, tn, fn


def compute_occupancy_numbers_segmentation(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    fov_mask: np.ndarray,
    labels: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_ids = list(labels.keys())[1:]
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()[mask]
    y_true = y_true.flatten()[mask]
    tp = np.zeros(len(label_ids))
    fp = np.zeros(len(label_ids))
    fn = np.zeros(len(label_ids))
    tn = np.zeros(len(label_ids))

    for label_id in label_ids:
        tp[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred == label_id))
        fp[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred == label_id))
        fn[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred != label_id))
        tn[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred != label_id))

    return tp, fp, tn, fn


@torch.no_grad()
def voxelize_gaussians_to_sscbench(
    *,
    means_world: torch.Tensor,
    opacities: torch.Tensor,
    sem_logits: torch.Tensor,
    world_to_velo: torch.Tensor,
    sigma_cutoff: float,
    pred_sem_channels: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Produces (sigmas,segs) in the same shape/semantics as S4C's evaluator.

    - sigmas: (256,256,32) float32 (proxy occupancy confidence)
    - segs: (256,256,32) int32 in Cityscapes trainId space [0..18]
    """
    # SSCBench grid in velodyne coordinates
    vox_origin = torch.tensor([0.0, -25.6, -2.0], device=device)
    voxel_size = VOXEL_SIZE
    X, Y, Z = (256, 256, 32)
    num_vox = X * Y * Z

    means_world = means_world.to(device)
    opacities = opacities.to(device).clamp(0, 1)
    sem_logits = sem_logits.to(device)

    C = min(int(pred_sem_channels), int(sem_logits.shape[1]))
    sem_logits = sem_logits[:, :C]

    # world -> velo
    N = means_world.shape[0]
    ones = torch.ones((N, 1), device=device, dtype=means_world.dtype)
    pts_h = torch.cat([means_world, ones], dim=1)
    pts_velo = (world_to_velo.to(device) @ pts_h.t()).t()[:, :3]

    ijk = torch.floor((pts_velo - vox_origin) / voxel_size).to(torch.int64)
    ix, iy, iz = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)

    sigmas = torch.zeros((num_vox,), device=device, dtype=torch.float32)
    seg_acc = torch.zeros((num_vox, C), device=device, dtype=torch.float16)

    if inside.any():
        ix = ix[inside]
        iy = iy[inside]
        iz = iz[inside]
        flat = (ix * (Y * Z) + iy * Z + iz).to(torch.int64)

        w = opacities[inside].to(torch.float32)
        sigmas.index_add_(0, flat, w)
        seg_acc.index_add_(0, flat, (w[:, None] * sem_logits[inside].to(torch.float32)).to(torch.float16))

    # Convert summed opacities into a smooth occupancy proxy in [0,1].
    # Using 1-exp(-sum_opacity) avoids the hard saturation from clamp(sum,0,1)
    # (which can make all non-zero voxels have sigma==1 and renders occ_thresh useless).
    sigmas = 1.0 - torch.exp(-sigmas)
    sigmas = sigmas.clamp(0, 1)

    # IMPORTANT: Cityscapes trainId=0 is 'road' (occupied). For empty space we want
    # a trainId that maps to unlabeled(0) under label_maps['cityscapes_to_label'].
    # In S4C's mapping, trainId=10 ('sky') -> unlabeled(0).
    SKY_TRAINID = 10
    segs = torch.full((num_vox,), SKY_TRAINID, device=device, dtype=torch.int64)
    has = sigmas > 0
    if has.any():
        segs[has] = seg_acc[has].to(torch.float32).argmax(dim=1).to(torch.int64)

    # Apply sigma cutoff like S4C
    segs[sigmas < float(sigma_cutoff)] = SKY_TRAINID

    sigmas_np = sigmas.reshape(256, 256, 32).detach().cpu().numpy().astype(np.float32)
    segs_np = segs.reshape(256, 256, 32).detach().cpu().numpy().astype(np.int32)
    return sigmas_np, segs_np


def _resolve_checkpoint_path(cp_arg: str) -> Path:
    p = Path(cp_arg)
    if p.is_file():
        return p
    if p.is_dir():
        # Prefer our convention
        latest = p / "latest.pt"
        if latest.exists():
            return latest
        # Fallback: any pt
        pts = sorted(p.glob("*.pt"))
        if pts:
            return pts[-1]
    raise FileNotFoundError(cp_arg)


def main() -> None:
    parser = argparse.ArgumentParser("SSCBenchmark evaluation (ssc_gs, S4C-compatible metrics)")
    parser.add_argument("--sscbench_data_root", "-ssc", type=str, required=True)
    parser.add_argument("--voxel_gt_path", "-vgt", type=str, required=True)
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=[9],
        help="KITTI-360 sequence ids to evaluate (default: 9, matches SSCBench setup).",
    )
    parser.add_argument("--resolution", "-r", default=(192, 640), nargs=2, type=int)
    parser.add_argument("--checkpoint", "-cp", type=str, required=True)
    parser.add_argument("--full", "-f", action="store_true")

    # extra knobs (default to S4C-like behavior)
    parser.add_argument("--sigma_cutoff", type=float, default=0.25)
    # Occupancy thresholding: default keeps S4C-compatible behavior.
    # GaussianFormer-style occupancy uses an explicit threshold (bin_logits > 0.5).
    parser.add_argument(
        "--occ_from",
        type=str,
        default="segs",
        choices=("segs", "sigmas"),
        help="How to derive occupancy for TP/FP/FN/TN: from seg labels (>0) or from sigmas (>occ_thresh).",
    )
    parser.add_argument(
        "--occ_thresh",
        type=float,
        default=0.5,
        help="Occupancy threshold when --occ_from=sigmas (analogous to GaussianFormer sigmoid_thresh).",
    )
    parser.add_argument("--use_additional_invalids", action="store_true", default=True)
    parser.add_argument("--dataset_length", type=int, default=10, help="Used when not --full")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Hard cap on evaluated samples (overrides -f and --dataset_length)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index into the dataset (skips the first N samples).",
    )
    parser.add_argument("--pred_sem_channels", type=int, default=19, help="Interpret first N channels as Cityscapes trainIds")
    parser.add_argument("--device", type=str, default="cuda")

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
        help="Root containing KITTI-360 pose folders (default: <voxel_gt_path>/../data_poses).",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pose_root = Path(args.pose_root) if args.pose_root else (Path(args.voxel_gt_path).resolve().parent / "poses")
    poses_cache: Dict[str, Dict[int, np.ndarray]] = {}

    label_maps = _load_label_maps()

    dataset = SSCBenchDataset(
        data_path=args.sscbench_data_root,
        voxel_gt_path=args.voxel_gt_path,
        sequences=tuple(int(s) for s in args.sequences),
        target_image_size=tuple(args.resolution),
        return_stereo=False,
        frame_count=1,
        color_aug=False,
    )

    if (args.dataset_length is not None) and (not args.full):
        dataset.length = int(args.dataset_length)

    fov_mask = get_fov_mask().reshape(256, 256, 32)

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    logging.info(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location=device)

    scene = build_scene_from_checkpoint_state(payload["scene"], device=device)
    scene.load_state_dict(payload["scene"], strict=True)
    scene.eval()

    # Calibration for world<->velo
    calibs = dataset._calibs
    T_cam_to_pose = torch.as_tensor(calibs["T_cam_to_pose"]["00"], dtype=torch.float32)
    T_velo_to_pose = torch.as_tensor(calibs["T_velo_to_pose"], dtype=torch.float32)

    results = {}
    for size in SIZES:
        results[size] = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tp_seg": np.zeros(15),
            "fp_seg": np.zeros(15),
            "tn_seg": np.zeros(15),
            "fn_seg": np.zeros(15),
        }

    start_idx = max(0, int(args.start))
    if start_idx >= len(dataset):
        raise ValueError(f"--start {start_idx} is >= dataset length {len(dataset)}")

    eval_len = len(dataset) - start_idx
    if args.max_samples is not None:
        eval_len = min(eval_len, int(args.max_samples))

    pbar = tqdm(range(start_idx, start_idx + eval_len))

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
                # Fallback: identity (keeps evaluation running, but metrics for this sample will be meaningless)
                pose_to_world_np = np.eye(4, dtype=np.float32)

        pose_to_world = torch.as_tensor(pose_to_world_np, dtype=torch.float32)
        velo_to_world = pose_to_world @ T_velo_to_pose
        world_to_velo = torch.linalg.inv(velo_to_world).to(device)

        # Predict in Cityscapes trainId space
        sigmas, segs_city = voxelize_gaussians_to_sscbench(
            means_world=scene.means,
            opacities=scene.opacities,
            sem_logits=scene.sem_logits,
            world_to_velo=world_to_velo,
            sigma_cutoff=args.sigma_cutoff,
            pred_sem_channels=args.pred_sem_channels,
            device=device,
        )

        # Map Cityscapes -> label ids (1..15, 0 for unlabeled)
        segs = convert_voxels(segs_city.astype(int), label_maps["cityscapes_to_label"])

        for size in SIZES:
            num_voxels = int(size // 0.2)

            _segs = segs[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]
            _target = target[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]
            _fov = fov_mask[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]
            _sigmas = sigmas[:num_voxels, (128 - num_voxels // 2) : (128 + num_voxels // 2), :]

            _tp, _fp, _tn, _fn = compute_occupancy_numbers(
                _segs,
                _target,
                _fov,
                sigmas=_sigmas,
                occ_from=args.occ_from,
                occ_thresh=args.occ_thresh,
            )
            _tp_seg, _fp_seg, _tn_seg, _fn_seg = compute_occupancy_numbers_segmentation(
                _segs, _target, _fov, labels=label_maps["labels"]
            )

            results[size]["tp"] += _tp
            results[size]["fp"] += _fp
            results[size]["tn"] += _tn
            results[size]["fn"] += _fn

            results[size]["tp_seg"] += _tp_seg
            results[size]["fp_seg"] += _fp_seg
            results[size]["tn_seg"] += _tn_seg
            results[size]["fn_seg"] += _fn_seg

        # show running metric for 51.2
        size = 51.2
        tp = results[size]["tp"]
        fp = results[size]["fp"]
        fn = results[size]["fn"]
        denom_r = (tp + fn)
        denom_p = (tp + fp)
        denom_i = (tp + fp + fn)
        recall = (tp / denom_r) if denom_r > 0 else 0.0
        precision = (tp / denom_p) if denom_p > 0 else 0.0
        iou = (tp / denom_i) if denom_i > 0 else 0.0
        pbar.set_postfix_str(f"IoU: {iou*100:.2f} Prec: {precision*100:.2f} Rec: {recall*100:.2f}")

    results_table = np.zeros((19, 3), dtype=np.float32)

    for size_i, size in enumerate(SIZES):
        tp = results[size]["tp"]
        fp = results[size]["fp"]
        fn = results[size]["fn"]

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        results_table[0, size_i] = iou
        results_table[1, size_i] = precision
        results_table[2, size_i] = recall

        with np.errstate(divide="ignore", invalid="ignore"):
            recall_seg = _safe_divide(results[size]["tp_seg"], (results[size]["tp_seg"] + results[size]["fn_seg"]))
            precision_seg = _safe_divide(results[size]["tp_seg"], (results[size]["tp_seg"] + results[size]["fp_seg"]))
            iou_seg = _safe_divide(
                results[size]["tp_seg"],
                (results[size]["tp_seg"] + results[size]["fp_seg"] + results[size]["fn_seg"]),
            )

        weights = label_maps["weights"]
        weights_val = np.array(list(weights.values()))
        weighted_mean_iou = float(np.sum(weights_val * iou_seg) / np.sum(weights_val))
        mean_iou = float(np.mean(iou_seg))

        results_table[3, size_i] = mean_iou
        results_table[4:, size_i] = iou_seg

        logging.info("#" * 50)
        logging.info(f"Results for size {size}.")
        logging.info("Occupancy metrics")
        logging.info(f"Recall: {recall*100:.2f}%")
        logging.info(f"Precision: {precision*100:.2f}%")
        logging.info(f"IoU: {iou*100:.2f}")

        logging.info("Occupancy metrics segmentation")
        for li in range(15):
            logging.info(
                f"{label_maps['labels'][li+1]}; IoU: {iou_seg[li]*100:.2f}; "
                f"Precision: {precision_seg[li]*100:.2f}%; Recall: {recall_seg[li]*100:.2f}%"
            )

        logging.info(f"Mean IoU: {mean_iou*100:.2f}")
        logging.info(f"Weighted Mean IoU: {weighted_mean_iou*100:.2f}")

    # Print results table like S4C
    results_table_str = ""
    for r in range(19):
        results_table_str += f"{results_table[r, 0]*100:.2f}\t{results_table[r, 1]*100:.2f}\t{results_table[r, 2]*100:.2f}\n"
    print(results_table_str)


if __name__ == "__main__":
    with torch.no_grad():
        main()
