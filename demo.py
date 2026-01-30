""" Demo to show prediction results.
    Author: chenxi-wang
"""

from utils.pose_rerank import rerank_grasps_by_pose_confidence
from utils.pose_gating import pose_confidence_from_diff9d
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument("--diff9d_pkl",type=str,default=None,help="Path to Diff9D pose result pkl (optional)")
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg
def pose_confidence_from_diff9d(diff9d_data):
    """
    Supports Wild6D/NOCS-style result dict:
      keys:
        - pred_scores: float OR array-like
        - pred_RTs: (N,4,4) OR (4,4)  (optional, not used here)
    Returns:
      float or None
    """
    import numpy as np

    if not isinstance(diff9d_data, dict):
        return None

    if "pred_scores" not in diff9d_data:
        return None

    ps = diff9d_data["pred_scores"]
    if isinstance(ps, (float, int)):
        return float(ps)

    # array-like
    ps = np.asarray(ps).reshape(-1)
    if ps.size == 0:
        return None
    return float(ps[0])

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]

    # ===== [NEW] export cloud & grippers for third-party inspection =====
    try:
        import os

        out_cloud_dir = os.path.join("outputs", "ei_demo", "cloud")
        out_gripper_dir = os.path.join("outputs", "ei_demo", "grippers")
        os.makedirs(out_cloud_dir, exist_ok=True)
        os.makedirs(out_gripper_dir, exist_ok=True)

        # export point cloud
        o3d.io.write_point_cloud(
            os.path.join(out_cloud_dir, "cloud.ply"),
            cloud
        )

        # export top-1 gripper only (clean & clear)
        if len(gg) > 0:
            top1 = gg[0]
            mesh = top1.to_open3d_geometry()
            o3d.io.write_triangle_mesh(
                os.path.join(out_gripper_dir, "gripper_top1.ply"),
                mesh
            )
            print("[EI] Exported cloud.ply and gripper_top1.ply")
        else:
            print("[EI] No grasps to export")

    except Exception as e:
        print("[EI] Export skipped:", e)
    # ===== [NEW] export cloud & grippers =====

    # original visualization (may fail on server)
    grippers = gg.to_open3d_geometry_list()
    try:
        o3d.visualization.draw_geometries([cloud, *grippers])
    except Exception as e:
        print("[EI] Visualization skipped (headless?):", e)


def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    # ---- EI s2a: pose confidence (Diff9D side) ----
    pose_conf = None
    if cfgs.diff9d_pkl is not None:
        try:
            import pickle
            with open(cfgs.diff9d_pkl, "rb") as f:
                diff9d_data = pickle.load(f)
                # ===== [NEW] export k pose samples (pred_RTs) to txt =====
                try:
                    import os
                    out_dir = os.path.join("outputs", "ei_demo", "poses")
                    os.makedirs(out_dir, exist_ok=True)

                    pred_RTs = diff9d_data.get("pred_RTs", None)
                    if pred_RTs is None:
                        print("[EI] pred_RTs not found in pkl, skip pose export")
                    else:
                        pred_RTs = np.asarray(pred_RTs)
                        print(f"[EI] pred_RTs shape = {pred_RTs.shape}")
                        for k in range(min(5, pred_RTs.shape[0])):
                            np.savetxt(os.path.join(out_dir, f"pose_{k}.txt"), pred_RTs[k], fmt="%.6f")
                        print(f"[EI] Saved pose_0~pose_4.txt to {out_dir}")
                except Exception as e:
                    print("[EI] Pose export skipped:", e)
                # ===== [NEW] export k pose samples (pred_RTs) to txt =====

            pose_conf = pose_confidence_from_diff9d(diff9d_data)
            print(f"[EI] Loaded Diff9D pkl: {cfgs.diff9d_pkl}")

            if pose_conf is None:
                print("[EI] Pose confidence unavailable (no pred_scores or empty)")
            else:
                print(f"[EI] Pose confidence = {pose_conf:.4f}")

        except Exception as e:
            print(f"[EI] Failed to load Diff9D pkl: {e}")
            pose_conf = None
    else:
        print("[EI] No Diff9D pkl provided, using grasp-only baseline")
    # ---- EI s2b: pose-aware grasp re-ranking (lightweight, safe) ----
    try:
        if pose_conf is None:
            print("[EI] Re-ranking skipped (pose_conf is None)")
        else:
            # (S4.2 evidence) log before/after to verify effect
            print("[EI] top5 scores BEFORE:", gg.scores[:5])
            gg.scores = gg.scores * float(pose_conf)
            print("[EI] top5 scores AFTER :", gg.scores[:5])
            print("[EI] Grasp scores re-ranked (scaled) by pose confidence")
    except Exception as e:
        print("[EI] Re-ranking skipped:", e)
    # ---- EI s2b: pose-aware grasp re-ranking (lightweight, safe) ----

    vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
