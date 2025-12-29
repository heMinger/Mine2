from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from ssc_gs.data.kitti360_split import ensure_kitti360_split_file
from ssc_gs.data.s4c_kitti360 import S4CKitti360KeyframeDataset
from ssc_gs.train import build_scene_from_checkpoint_state


def _voxel_params() -> Tuple[torch.Tensor, float, Tuple[int, int, int]]:
    # SSCBench/KITTI360 voxel grid: (51.2,51.2,6.4) at 0.2m => (256,256,32)
    vox_origin = torch.tensor([0.0, -25.6, -2.0], dtype=torch.float32)
    voxel_size = 0.2
    dims = (256, 256, 32)
    return vox_origin, voxel_size, dims


@torch.no_grad()
def voxelize_gaussians_semantic(
    means_world: torch.Tensor,
    sem_logits: torch.Tensor,
    *,
    world_to_velo: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Voxelize per-Gaussian semantic logits into a dense voxel grid.

    Returns:
        pred_labels: (X,Y,Z) long in [0,num_classes-1], with 255 for empty voxels.
    """
    vox_origin, voxel_size, dims = _voxel_params()
    X, Y, Z = dims

    # transform means to velodyne frame of this sample
    N = means_world.shape[0]
    ones = torch.ones((N, 1), device=means_world.device, dtype=means_world.dtype)
    pts_h = torch.cat([means_world, ones], dim=1)  # (N,4)
    pts_velo = (world_to_velo @ pts_h.t()).t()[:, :3]

    # compute voxel indices
    vox_origin = vox_origin.to(device=device)
    pts = pts_velo.to(device=device)

    ijk = torch.floor((pts - vox_origin) / voxel_size).to(torch.int64)
    ix, iy, iz = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
    if inside.sum() == 0:
        return torch.full((X, Y, Z), 255, dtype=torch.long, device="cpu")

    ix = ix[inside]
    iy = iy[inside]
    iz = iz[inside]
    logits = sem_logits[inside].to(device=device)

    flat = (ix * (Y * Z) + iy * Z + iz).to(torch.int64)  # (M,)
    num_vox = X * Y * Z

    # accumulate logits per voxel (float16 to reduce memory)
    acc = torch.zeros((num_vox, num_classes), device=device, dtype=torch.float16)
    acc.index_add_(0, flat, logits.to(torch.float16))

    # voxels with no points => empty
    has = (acc.abs().sum(dim=1) > 0)
    pred = torch.full((num_vox,), 255, device=device, dtype=torch.long)
    pred[has] = acc[has].to(torch.float32).argmax(dim=1).to(torch.long)

    return pred.reshape(X, Y, Z).to("cpu")


@torch.no_grad()
def update_confusion_3d(conf: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, *, num_classes: int, ignore_index: int) -> None:
    # pred/target: (X,Y,Z) long
    valid = target != ignore_index
    target = target[valid]
    pred = pred[valid]
    if target.numel() == 0:
        return

    # ignore empty/unlabeled predictions
    valid_pred = (pred != ignore_index) & (pred >= 0) & (pred < num_classes)
    target = target[valid_pred]
    pred = pred[valid_pred]
    if target.numel() == 0:
        return

    # clamp unknown labels to ignore
    target = torch.where(target < num_classes, target, torch.full_like(target, ignore_index))
    valid2 = target != ignore_index
    target = target[valid2]
    pred = pred[valid2]
    if target.numel() == 0:
        return

    k = target * num_classes + pred
    bins = torch.bincount(k, minlength=num_classes * num_classes)
    conf += bins.reshape(num_classes, num_classes)


def compute_iou(conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    valid = denom > 0
    return iou, valid


def load_gt_voxel(voxel_root: Path, *, sequence: str, frame_id: int) -> Optional[torch.Tensor]:
    p = voxel_root / sequence / f"{frame_id:06d}_1_1.npy"
    if not p.exists():
        return None
    arr = np.load(str(p))
    # stored float32 but are integer-ish labels
    gt = torch.from_numpy(arr).to(torch.int64)
    return gt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--voxel_gt_root", type=str, default="/data/lmh_data/KITTI360/s4c_labels")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--ignore_index", type=int, default=255)

    # data
    p.add_argument("--kitti360_data_path", type=str, default="/data/lmh_data/KITTI360")
    p.add_argument("--kitti360_pose_path", type=str, default="/data/lmh_data/KITTI360/poses")
    p.add_argument(
        "--pseudo_seg_path",
        type=str,
        default="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr",
    )
    p.add_argument("--split_file", type=str, default="")
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    voxel_root = Path(args.voxel_gt_root)
    if not voxel_root.exists():
        raise FileNotFoundError(str(voxel_root))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split_file = args.split_file
    if split_file == "":
        repo_root = Path(__file__).resolve().parents[2]
        split_file = ensure_kitti360_split_file(
            pseudo_seg_root=args.pseudo_seg_path,
            pose_root=args.kitti360_pose_path,
            out_path=str(repo_root / "ssc_gs" / "splits" / "kitti360_panoptic_deeplab_eval_files.txt"),
        )

    base = Kitti360Dataset(
        data_path=args.kitti360_data_path,
        pose_path=args.kitti360_pose_path,
        split_path=split_file,
        target_image_size=(192, 640),
        return_stereo=False,
        return_fisheye=False,
        return_segmentation=True,
        segmentation_mode="panoptic_deeplab",
        data_segmentation_path=args.pseudo_seg_path,
        frame_count=1,
        is_preprocessed=False,
    )
    ds = S4CKitti360KeyframeDataset(base, keyframe_idx=0, has_pseudo_sem=False)

    gen = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=lambda b: b[0], generator=gen)

    payload = torch.load(str(ckpt_path), map_location=device)
    scene = build_scene_from_checkpoint_state(payload["scene"], device=device)
    scene.load_state_dict(payload["scene"], strict=True)
    scene.eval()

    # pull calibs from the underlying dataset
    calibs = base._calibs
    T_cam_to_pose = torch.as_tensor(calibs["T_cam_to_pose"]["00"], dtype=torch.float32)
    T_velo_to_pose = torch.as_tensor(calibs["T_velo_to_pose"], dtype=torch.float32)
    pose_to_velo = torch.linalg.inv(T_velo_to_pose)  # pose -> velo

    conf = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)

    seen = 0
    tried = 0
    for sample in dl:
        if args.max_samples > 0 and seen >= args.max_samples:
            break
        tried += 1

        # map dataset index -> (sequence, id)
        seq, frame_id, _is_right = base._datapoints[sample.index]
        gt = load_gt_voxel(voxel_root, sequence=seq, frame_id=int(frame_id))
        if gt is None:
            continue

        # compute pose_to_world from camtoworld and known cam->pose
        camtoworld = sample.camtoworld.to(torch.float32)
        pose_to_world = camtoworld @ torch.linalg.inv(T_cam_to_pose)

        # world->velo for this frame
        velo_to_world = pose_to_world @ T_velo_to_pose
        world_to_velo = torch.linalg.inv(velo_to_world).to(device)

        pred = voxelize_gaussians_semantic(
            scene.means.to(device),
            scene.sem_logits.to(device),
            world_to_velo=world_to_velo,
            num_classes=args.num_classes,
            device=device,
        )

        update_confusion_3d(conf, pred, gt, num_classes=args.num_classes, ignore_index=args.ignore_index)
        seen += 1

        if seen % 1 == 0:
            iou, valid = compute_iou(conf.float())
            miou = float(iou[valid].mean().item()) if valid.any() else 0.0
            print(f"seen={seen} mIoU={miou*100:.2f}")

    iou, valid = compute_iou(conf.float())
    miou = float(iou[valid].mean().item()) if valid.any() else 0.0

    print("=== Voxel Eval Done ===")
    print(f"samples={seen}")
    print(f"tried={tried}")
    print(f"mIoU={miou*100:.2f}")


if __name__ == "__main__":
    main()
