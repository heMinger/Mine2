from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from ssc_gs.data.kitti360_split import ensure_kitti360_split_file
from ssc_gs.initialize.depth_init import save_cached_points, sample_world_points_from_depth
from ssc_gs.utils.camera import kitti360_normK_to_pixelK


def build_dataset(args: argparse.Namespace) -> Kitti360Dataset:
    repo_root = Path(__file__).resolve().parents[2]
    split_file = args.split_file
    if split_file == "":
        split_file = ensure_kitti360_split_file(
            pseudo_seg_root=args.pseudo_seg_path,
            pose_root=args.kitti360_pose_path,
            out_path=str(repo_root / "ssc_gs" / "splits" / "kitti360_panoptic_deeplab_train_files.txt"),
        )

    return Kitti360Dataset(
        data_path=args.kitti360_data_path,
        pose_path=args.kitti360_pose_path,
        split_path=split_file,
        target_image_size=(args.image_size_h, args.image_size_w),
        return_stereo=False,
        return_fisheye=False,
        return_depth=True,
        return_segmentation=False,
        segmentation_mode=None,
        data_segmentation_path=None,
        frame_count=1,
        is_preprocessed=args.is_preprocessed,
    )


@torch.no_grad()
def process_one(ds: Kitti360Dataset, index: int, *, device: torch.device, args: argparse.Namespace) -> int:
    data = ds[index]

    # S4C returns lists even for frame_count=1
    depth = torch.as_tensor(data["depths"][0]).float()[0]  # (H,W)
    K_norm = torch.as_tensor(data["projs"][0]).float()
    camtoworld = torch.as_tensor(data["poses"][0]).float()

    H, W = depth.shape
    K_px = kitti360_normK_to_pixelK(K_norm, width=W, height=H)

    pts = sample_world_points_from_depth(
        depth_hw=depth.to(device),
        K_px=K_px.to(device),
        camtoworld=camtoworld.to(device),
        max_points=args.max_points,
        stride=args.stride,
    )

    # Match the training-side cache key: `S4CKitti360KeyframeDataset` uses data["index"]
    sample_index = int(torch.as_tensor(data.get("index")).reshape(-1)[0].item())

    save_cached_points(args.cache_dir, sample_index, pts)
    return pts.shape[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute offline init point clouds from KITTI-360 LiDAR-projected depth.")
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
    p.add_argument("--cache_dir", type=str, default="/data/lmh/ssc_gs_pointcloud_cache")
    p.add_argument("--image_size_h", type=int, default=192)
    p.add_argument("--image_size_w", type=int, default=640)
    p.add_argument("--is_preprocessed", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1, help="Exclusive end index; -1 means full dataset")
    p.add_argument("--max_items", type=int, default=-1, help="Optional cap on number of items processed")
    p.add_argument("--max_points", type=int, default=200_000)
    p.add_argument("--stride", type=int, default=1, help="Subsample the depth grid by this stride before sampling")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    ds = build_dataset(args)
    total = len(ds)

    start = max(args.start, 0)
    end = total if args.end < 0 else min(args.end, total)

    processed = 0
    for i in range(start, end):
        # Determine cache key without doing expensive work? We need __getitem__ to know the `index`.
        # So we conservatively check overwrite using the loop counter as a fallback.
        if not args.overwrite:
            approx_p = Path(args.cache_dir) / f"{i:08d}.npz"
            if approx_p.exists():
                continue

        try:
            npts = process_one(ds, i, device=device, args=args)
        except FileNotFoundError as e:
            print(f"skip i={i} (missing file): {e}")
            continue

        processed += 1
        if processed % 10 == 0:
            print(f"processed={processed} last_i={i} points={npts}")

        if args.max_items > 0 and processed >= args.max_items:
            break

    print(f"done: processed={processed} cache_dir={args.cache_dir}")


if __name__ == "__main__":
    main()
