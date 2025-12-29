from __future__ import annotations

import argparse0
from pathlib import Path
from typing import Any, Dict

import torch

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from s4c.datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from ssc_gs.ff.scene_model import FFSceneConfig, FFSceneModel


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    # Matches gsplat/examples/utils.py
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser("Export feed-forward predicted Gaussians to a gsplat checkpoint (.pt)")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path or directory (from ssc_gs/ff_train.py)")
    ap.add_argument("--device", type=str, default="cuda")

    # Data (same as ff_train.py)
    ap.add_argument("--kitti360_data_path", type=str, default="/data/lmh_data/KITTI360")
    ap.add_argument("--kitti360_pose_path", type=str, default="/data/lmh_data/KITTI360/poses")
    ap.add_argument(
        "--pseudo_seg_path",
        type=str,
        default="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr",
    )
    ap.add_argument("--split_file", type=str, default="", help="Optional explicit split file. If empty, dataset default is used.")
    ap.add_argument("--data_fc", type=int, default=3, help="Multi-view frame_count (S4C style)")
    ap.add_argument("--dilation", type=int, default=1)
    ap.add_argument("--keyframe_offset", type=int, default=0)
    ap.add_argument("--additional_random_front_offset", action="store_true")
    ap.add_argument("--target_h", type=int, default=192)
    ap.add_argument("--target_w", type=int, default=640)
    ap.add_argument("--sample_index", type=int, default=0)

    # Model knobs (must match training if you changed them)
    ap.add_argument("--use_sparse_conv", type=int, default=1, choices=[0, 1])
    ap.add_argument("--voxel_max_radius", type=int, default=3)

    # Export
    ap.add_argument("--output_ckpt", type=str, required=True)
    ap.add_argument("--sh_degree", type=int, default=3, help="SH degree for colors in exported gsplat ckpt")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    ds = Kitti360Dataset(
        data_path=args.kitti360_data_path,
        pose_path=args.kitti360_pose_path,
        split_path=args.split_file if args.split_file else None,
        target_image_size=(int(args.target_h), int(args.target_w)),
        return_stereo=False,
        return_depth=False,
        return_fisheye=False,
        fisheye_offset=(0, 0) if bool(args.additional_random_front_offset) else 0,
        return_segmentation=False,
        frame_count=int(args.data_fc),
        dilation=int(args.dilation),
        keyframe_offset=int(args.keyframe_offset),
        additional_random_front_offset=bool(args.additional_random_front_offset),
        is_preprocessed=False,
    )

    if int(args.sample_index) < 0 or int(args.sample_index) >= len(ds):
        raise ValueError(f"sample_index out of range: {args.sample_index} (len={len(ds)})")

    sample: Dict[str, Any] = ds[int(args.sample_index)]
    imgs = sample["imgs"]
    projs = sample["projs"]
    poses = sample["poses"]

    imgs_t = [im.to(device) for im in imgs]
    Ks_t = [torch.tensor(K, dtype=torch.float32, device=device) for K in projs]
    c2w_t = [torch.tensor(P, dtype=torch.float32, device=device) for P in poses]

    model = FFSceneModel(FFSceneConfig(use_sparse_conv=bool(int(args.use_sparse_conv) != 0))).to(device)

    ckpt_path = _resolve_checkpoint_path(args.ckpt)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Calibration for self-encoding voxel context (optional, but typically available)
    T_velo_to_cam = torch.tensor(ds._calibs["T_velo_to_cam"]["00"], dtype=torch.float32, device=device)

    pred = model(
        imgs_m11_chw=imgs_t,
        Ks_norm=Ks_t,
        camtoworlds=c2w_t,
        encoder_index=0,
        supervise_indices=[j for j in range(len(imgs_t)) if j != 0],
        T_velo_to_cam=T_velo_to_cam,
        voxel_grid=None,
        voxel_max_radius=int(args.voxel_max_radius),
    )

    means = pred["means_world"].detach().cpu()
    scales_log = torch.log(pred["scales"].clamp(min=1e-6)).detach().cpu()
    quats = pred["quats"].detach().cpu()
    opacities_logit = torch.logit(pred["opacities"].clamp(1e-6, 1.0 - 1e-6)).detach().cpu()

    rgb = pred["rgb"].detach().cpu().clamp(0, 1)
    sh_degree = int(args.sh_degree)
    colors = torch.zeros((rgb.shape[0], (sh_degree + 1) ** 2, 3), dtype=rgb.dtype)
    colors[:, 0, :] = rgb_to_sh(rgb)
    sh0 = colors[:, :1, :]
    shN = colors[:, 1:, :]

    out = {
        "step": int(payload.get("global_step", 0)),
        "splats": {
            "means": means,
            "scales": scales_log,
            "quats": quats,
            "opacities": opacities_logit,
            "sh0": sh0,
            "shN": shN,
        },
        "extra": {
            "source": "ssc_gs.ff.FFSceneModel",
            "kitti360_sample_index": int(args.sample_index),
            "image_size": [int(args.target_h), int(args.target_w)],
        },
    }

    out_path = Path(args.output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(out_path))
    print(f"Wrote gsplat checkpoint: {out_path}")


if __name__ == "__main__":
    main()
