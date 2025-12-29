from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ssc_gs.bootstrap import bootstrap_workspace

bootstrap_workspace()

from ssc_gs.data.kitti360_split import ensure_kitti360_split_file
from ssc_gs.train import TrainConfig, build_s4c_dataset, build_scene_from_checkpoint_state
from ssc_gs.utils.camera import kitti360_normK_to_pixelK
from ssc_gs.render.gsplat_renderer import GSplatRenderer


@torch.no_grad()
def update_confusion(conf: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, *, num_classes: int, ignore_index: int) -> None:
    # pred/target: (H,W) long
    valid = target != ignore_index
    target = target[valid]
    pred = pred[valid]
    if target.numel() == 0:
        return

    k = target * num_classes + pred
    bins = torch.bincount(k, minlength=num_classes * num_classes)
    conf += bins.reshape(num_classes, num_classes)


def compute_iou(conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # conf: [gt, pred]
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    valid = denom > 0
    return iou, valid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt) saved by ssc_gs.train")
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
    p.add_argument("--max_samples", type=int, default=200, help="Evaluate at most N samples (0 = all)")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(
        kitti360_data_path=args.kitti360_data_path,
        kitti360_pose_path=args.kitti360_pose_path,
        pseudo_seg_path=args.pseudo_seg_path,
        split_file=args.split_file,
        device=str(device),
    )

    # split
    if cfg.split_file == "":
        repo_root = Path(__file__).resolve().parents[2]
        cfg.split_file = ensure_kitti360_split_file(
            pseudo_seg_root=cfg.pseudo_seg_path,
            pose_root=cfg.kitti360_pose_path,
            out_path=str(repo_root / "ssc_gs" / "splits" / "kitti360_panoptic_deeplab_eval_files.txt"),
        )

    ds = build_s4c_dataset(cfg)
    gen = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=lambda b: b[0], generator=gen)

    payload = torch.load(str(ckpt_path), map_location=device)
    scene = build_scene_from_checkpoint_state(payload["scene"], device=device)
    scene.load_state_dict(payload["scene"], strict=True)
    scene.eval()

    conf = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64, device="cpu")

    # Renderer is sized per sample (we recreate if H/W changes)
    renderer: GSplatRenderer | None = None
    renderer_hw: tuple[int, int] | None = None

    seen = 0
    for sample in dl:
        if args.max_samples > 0 and seen >= args.max_samples:
            break
        if sample.pseudo_sem is None:
            continue

        img_m11 = sample.image  # (3,H,W)
        H, W = img_m11.shape[1:]

        if renderer is None or renderer_hw != (H, W):
            renderer = GSplatRenderer(width=W, height=H, packed=True)
            renderer_hw = (H, W)

        K_px = kitti360_normK_to_pixelK(sample.K_norm.to(device), width=W, height=H)
        camtoworld = sample.camtoworld.to(device)

        renders = renderer.render(
            means=scene.means.to(device),
            quats=scene.quats.to(device),
            scales=scene.scales.to(device),
            opacities=scene.opacities.to(device),
            rgb=scene.rgb.to(device),
            camtoworld=camtoworld,
            K_px=K_px,
            sem_logits=scene.sem_logits.to(device),
        )

        if renders.sem_logits is None:
            raise RuntimeError("Renderer did not return sem_logits")

        pred = renders.sem_logits.argmax(dim=-1).to("cpu").long()  # (H,W)
        target = sample.pseudo_sem.to("cpu").long()  # (H,W)

        update_confusion(conf, pred, target, num_classes=args.num_classes, ignore_index=args.ignore_index)
        seen += 1

        if seen % 50 == 0:
            iou, valid = compute_iou(conf.float())
            miou = float(iou[valid].mean().item()) if valid.any() else 0.0
            print(f"seen={seen} mIoU={miou*100:.2f}")

    iou, valid = compute_iou(conf.float())
    miou = float(iou[valid].mean().item()) if valid.any() else 0.0

    print("=== Eval Done ===")
    print(f"samples={seen}")
    print(f"mIoU={miou*100:.2f}")

    # print per-class iou (compact)
    iou_np = iou.cpu().numpy()
    valid_np = valid.cpu().numpy()
    for c in range(args.num_classes):
        if not valid_np[c]:
            continue
        print(f"class={c:02d} iou={iou_np[c]*100:.2f}")


if __name__ == "__main__":
    main()
