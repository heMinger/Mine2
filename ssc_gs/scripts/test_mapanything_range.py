import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ssc_gs.bootstrap import bootstrap_workspace
bootstrap_workspace()

from s4c.datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from ssc_gs.initialize.mapanything_init import MapAnythingInitializer, MapAnythingInitConfig
from ssc_gs.train import ensure_kitti360_split_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti360_data_path", type=str, default="/data/lmh_data/KITTI360")
    parser.add_argument("--kitti360_pose_path", type=str, default="/data/lmh_data/KITTI360/poses")
    parser.add_argument("--pseudo_seg_path", type=str, default="/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr")
    parser.add_argument("--hf_model_name", type=str, default="/home/lmh/VGM-GS-S4C/map-anything/ckpt/map-anything")
    args = parser.parse_args()

    # Setup dataset
    out_path = str(repo_root / "ssc_gs" / "splits" / "kitti360_panoptic_deeplab_train_files.txt")
    split_path = ensure_kitti360_split_file(pseudo_seg_root=args.pseudo_seg_path, pose_root=args.kitti360_pose_path, out_path=out_path)

    print("Initializing dataset...")
    ds = Kitti360Dataset(
        data_path=args.kitti360_data_path,
        pose_path=args.kitti360_pose_path,
        split_path=split_path,
        target_image_size=(192, 640),
        return_stereo=False,
        frame_count=1,
        data_segmentation_path=args.pseudo_seg_path,
        segmentation_mode="panoptic_deeplab", # To match training setup
        is_preprocessed=False
    )
    
    print(f"Dataset length: {len(ds)}")
    
    # Get one sample
    idx = 0
    print(f"Loading sample {idx}...")
    data = ds[idx]
    imgs_raw = data.get("imgs_raw")
    if imgs_raw is None:
        print("Error: imgs_raw not found in dataset output!")
        return

    print(f"Got imgs_raw: {len(imgs_raw)} images")
    print(f"Raw image shape: {imgs_raw[0].shape}")
    print(f"Raw image range: [{imgs_raw[0].min()}, {imgs_raw[0].max()}]")
    print(f"Raw image mean: {imgs_raw[0].mean()}")
    from torchvision.utils import save_image
    save_image(imgs_raw[0], "debug_input.png")
    print("Saved debug_input.png")

    # Setup MapAnything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MapAnythingInitConfig(
        hf_model_name=args.hf_model_name,
        local_files_only=True,
        max_points=20000
    )
    initializer = MapAnythingInitializer(cfg, device)

    # Prepare inputs
    imgs_raw_t = [im.to(device) for im in imgs_raw]
    # We need K_norm and c2w as well, even if using raw images, for the interface
    projs = data["projs"]
    poses = data["poses"]
    
    Ks_t = [torch.tensor(K, dtype=torch.float32, device=device) for K in projs]
    c2w_t = [torch.tensor(P, dtype=torch.float32, device=device) for P in poses]
    
    from ssc_gs.utils.camera import kitti360_normK_to_pixelK
    
    # Crop center 512x376
    W_crop = 512
    H_crop = 376
    start_x = (1408 - W_crop) // 2
    
    img_crop = imgs_raw_t[0][:, :, start_x:start_x+W_crop]
    
    # Adjust K_norm
    K_norm = Ks_t[0].cpu()
    K_px = kitti360_normK_to_pixelK(K_norm, 1408, 376)
    
    K_px_new = K_px.clone()
    K_px_new[0, 2] -= start_x
    
    # Convert back to K_norm
    K_norm_new = K_px_new.clone()
    K_norm_new[0, 0] /= (W_crop / 2.0)
    K_norm_new[1, 1] /= (H_crop / 2.0)
    K_norm_new[0, 2] = (K_norm_new[0, 2] / (W_crop / 2.0)) - 1.0
    K_norm_new[1, 2] = (K_norm_new[1, 2] / (H_crop / 2.0)) - 1.0
    
    K_norm_new = K_norm_new.to(device)
    
    print("Running MapAnything inference (Center Crop 512x376)...")
    points = initializer.infer_points_from_window(
        images_m11_chw=[], 
        images_raw_01_chw=[img_crop],
        K_norm_list=[K_norm_new],
        camtoworld_list=[c2w_t[0]],
        sample_index=0,
        force_recompute=True
    )
    
    print(f"Predicted points: {points.shape}")
    
    # Analyze range in Camera frame (approx)
    # Since points are in World frame, we project back to camera 0 to see depth
    c2w0 = c2w_t[0]
    w2c0 = torch.inverse(c2w0)
    ones = torch.ones((points.shape[0], 1), device=device)
    pts_h = torch.cat([points, ones], dim=1)
    pts_cam = (w2c0 @ pts_h.T).T
    
    z = pts_cam[:, 2].cpu().numpy()
    print(f"Depth (Z) stats in Camera 0 frame:")
    print(f"  Min: {z.min():.2f} m")
    print(f"  Max: {z.max():.2f} m")
    print(f"  Mean: {z.mean():.2f} m")
    print(f"  Median: {np.median(z):.2f} m")
    print(f"  90th %: {np.percentile(z, 90):.2f} m")
    print(f"  99th %: {np.percentile(z, 99):.2f} m")

if __name__ == "__main__":
    main()
