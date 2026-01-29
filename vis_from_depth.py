import os
import numpy as np
import open3d as o3d
import cv2

# ==================================================
# 基础路径配置（只改这里就能换 scene / frame）
# ==================================================
SCENE_ROOT = "/home/dingxin/data/GraspNet/dataset/scenes/scene_0100/realsense"
FRAME_ID   = "0003"   # 当前帧编号

DEPTH_PATH = os.path.join(SCENE_ROOT, "depth", f"{FRAME_ID}.png")
RGB_PATH   = os.path.join(SCENE_ROOT, "rgb",   f"{FRAME_ID}.png")
CAMK_PATH  = os.path.join(SCENE_ROOT, "camK.npy")

# 输出目录（统一工程规范）
OUT_DIR = "outputs/pointclouds"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PLY = os.path.join(OUT_DIR, f"scene0100_rs_{FRAME_ID}_points.ply")

# ==================================================
# 读取数据
# ==================================================
depth = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED).astype(np.float32)
rgb   = cv2.imread(RGB_PATH)[:, :, ::-1]  # BGR -> RGB
K     = np.load(CAMK_PATH)

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

h, w = depth.shape
us, vs = np.meshgrid(np.arange(w), np.arange(h))

# GraspNet: depth unit = mm
Z = depth / 1000.0   # mm -> m
X = (us - cx) * Z / fx
Y = (vs - cy) * Z / fy

points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
colors = rgb.reshape(-1, 3) / 255.0

# ==================================================
# Open3D 点云
# ==================================================
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

o3d.io.write_point_cloud(OUT_PLY, pcd)
print("Saved point cloud to:", OUT_PLY)
