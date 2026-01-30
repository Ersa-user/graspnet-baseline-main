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

def vis_grasps(gg, cloud, pred_RTs=None):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    # ===== [NEW] verify pred_RTs passed into vis_grasps =====
    try:
        import os
        if pred_RTs is None:
            print("[EI] vis_grasps: pred_RTs is None")
        else:
            pred_RTs_np = np.asarray(pred_RTs)
            print(f"[EI] vis_grasps: pred_RTs shape = {pred_RTs_np.shape}")

            out_dir = os.path.join("outputs", "ei_demo", "poses_from_vis")
            os.makedirs(out_dir, exist_ok=True)
            for k in range(min(5, pred_RTs_np.shape[0])):
                np.savetxt(os.path.join(out_dir, f"pose_{k}.txt"), pred_RTs_np[k], fmt="%.6f")
            print("[EI] Saved pose_0~pose_4.txt to outputs/ei_demo/poses_from_vis/")
    except Exception as e:
        print("[EI] pred_RTs verify/export skipped:", e)
    # ===== [NEW] verify pred_RTs passed into vis_grasps =====

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

        # export top-1 gripper + k=5 paired grippers (analysis)
        if len(gg) > 0:
            top1 = gg[0]

            # 原来的 top1 文件保留
            mesh0 = top1.to_open3d_geometry()
            o3d.io.write_triangle_mesh(
                os.path.join(out_gripper_dir, "gripper_top1.ply"),
                mesh0
            )

            # 新增：按 pose_k 输出 5 份（用于配对分析）
            out_k_dir = os.path.join("outputs", "ei_demo", "grippers_by_pose")
            os.makedirs(out_k_dir, exist_ok=True)

            # write grasp results into k0~k4 subfolders (placeholder)
            base_k_dir = os.path.join("outputs", "ei_demo", "grippers_by_pose")

            for k in range(5):
                k_dir = os.path.join(base_k_dir, f"k{k}")
                os.makedirs(k_dir, exist_ok=True)

                # 1) 写占位用的 grasp
                meshk = top1.to_open3d_geometry()
                o3d.io.write_triangle_mesh(
                    os.path.join(k_dir, "top1.ply"),
                    meshk
                )

                # ===== [NEW] pose-guided crop cloud per k =====
                try:
                    if pred_RTs is not None:
                        T = np.asarray(pred_RTs[k], dtype=np.float32)  # (4,4)
                        T_inv = np.linalg.inv(T)

                        pts_c = np.asarray(cloud.points, dtype=np.float32)
                        ones = np.ones((pts_c.shape[0], 1), dtype=np.float32)
                        pts_c_h = np.concatenate([pts_c, ones], axis=1)

                        # camera -> object
                        pts_o_h = (T_inv @ pts_c_h.T).T
                        pts_o = pts_o_h[:, :3]

                        # cube crop around object origin
                        s = 0.15
                        m = (
                                (np.abs(pts_o[:, 0]) < s) &
                                (np.abs(pts_o[:, 1]) < s) &
                                (np.abs(pts_o[:, 2]) < s)
                        )
                        pts_o_crop = pts_o[m]

                        # object -> camera (for visualization)
                        pts_o_crop_h = np.concatenate(
                            [pts_o_crop, np.ones((pts_o_crop.shape[0], 1), dtype=np.float32)],
                            axis=1
                        )
                        pts_c_crop_h = (T @ pts_o_crop_h.T).T
                        pts_c_crop = pts_c_crop_h[:, :3]

                        crop_pc = o3d.geometry.PointCloud()
                        crop_pc.points = o3d.utility.Vector3dVector(pts_c_crop)

                        o3d.io.write_point_cloud(
                            os.path.join(k_dir, "crop_cloud.ply"),
                            crop_pc
                        )

                        print(f"[EI] k{k}: crop_cloud.ply saved, n={len(pts_c_crop)}")
                except Exception as e:
                    print(f"[EI] k{k}: crop export skipped:", e)
                # ===== [NEW] pose-guided crop cloud per k =====
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

    # ===== [NEW] per-k pose-conditioned grasp inference (using crop_cloud) =====
    if pred_RTs is not None:
        import os, copy
        import torch

        base_k_dir = os.path.join("outputs", "ei_demo", "grippers_by_pose")
        os.makedirs(base_k_dir, exist_ok=True)

        # 原始点云（camera frame）
        pts_c_all = np.asarray(cloud.points, dtype=np.float32)

        # 取 num_point（和你 demo 里一致；如果 cfgs 里叫 num_point 就用它，否则用 end_points 的长度）
        try:
            num_point = int(cfgs.num_point)
        except Exception:
            # end_points['point_clouds'] shape: (1, N, 3)
            num_point = int(end_points["point_clouds"].shape[1])

        best_k = None
        best_score = -1e9
        best_mesh = None
        for k in range(min(5, np.asarray(pred_RTs).shape[0])):
            k_dir = os.path.join(base_k_dir, f"k{k}")
            os.makedirs(k_dir, exist_ok=True)

            # --- 1) 用 pred_RTs[k] 做 pose-guided crop（与你在 vis_grasps 里一致）---
            T = np.asarray(pred_RTs[k], dtype=np.float32)
            T_inv = np.linalg.inv(T)

            ones = np.ones((pts_c_all.shape[0], 1), dtype=np.float32)
            pts_c_h = np.concatenate([pts_c_all, ones], axis=1)  # (N,4)

            pts_o_h = (T_inv @ pts_c_h.T).T  # (N,4)
            pts_o = pts_o_h[:, :3]

            s = 0.15
            m = (
                    (np.abs(pts_o[:, 0]) < s) &
                    (np.abs(pts_o[:, 1]) < s) &
                    (np.abs(pts_o[:, 2]) < s)
            )
            pts_o_crop = pts_o[m]

            # object -> camera
            pts_o_crop_h = np.concatenate([pts_o_crop, np.ones((pts_o_crop.shape[0], 1), dtype=np.float32)], axis=1)
            pts_c_crop = (T @ pts_o_crop_h.T).T[:, :3]  # (Nc,3)

            if pts_c_crop.shape[0] < 50:
                print(f"[EI] k{k}: too few crop points ({pts_c_crop.shape[0]}), skip")
                continue

            # --- 2) 采样到 num_point（GraspNet 输入需要固定点数）---
            if pts_c_crop.shape[0] >= num_point:
                idx = np.random.choice(pts_c_crop.shape[0], num_point, replace=False)
            else:
                idx = np.random.choice(pts_c_crop.shape[0], num_point, replace=True)
            pts_in = pts_c_crop[idx]  # (num_point,3)

            # --- 3) 构造 end_points_k，替换点云后再跑 get_grasps ---
            end_points_k = copy.deepcopy(end_points)
            pc_tensor = torch.from_numpy(pts_in).float().unsqueeze(0)  # (1,num_point,3)
            if torch.cuda.is_available():
                pc_tensor = pc_tensor.cuda()
            end_points_k["point_clouds"] = pc_tensor

            gg_k = get_grasps(net, end_points_k)
            gg_k.nms()
            gg_k.sort_by_score()

            # --- 4) 导出 top1 grasp 到对应目录，并记录 best-k ---
            if len(gg_k) > 0:
                top1_k = gg_k[0]
                score_k = float(gg_k.scores[0])

                mesh_k = top1_k.to_open3d_geometry()
                o3d.io.write_triangle_mesh(os.path.join(k_dir, "top1_posecond.ply"), mesh_k)
                print(f"[EI] k{k}: saved top1_posecond.ply, score={score_k:.4f}")

                if score_k > best_score:
                    best_score = score_k
                    best_k = k
                    best_mesh = mesh_k
            else:
                print(f"[EI] k{k}: no grasps")
        if best_mesh is not None:
            robust_dir = os.path.join("outputs", "ei_demo", "robust")
            os.makedirs(robust_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(robust_dir, "robust_top1.ply"), best_mesh)
            print(f"[EI] Robust grasp saved: best_k={best_k}, score={best_score:.4f}")
    # 仍保留原来的可视化/导出（方便对照）
    vis_grasps(gg, cloud, pred_RTs)
    # ===== [NEW] per-k pose-conditioned grasp inference =====

if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
