import os
import numpy as np
import open3d as o3d

# ==================================================
# 基础路径配置（只改这里就能换 scene / frame）
# ==================================================
SCENE_ID = "scene_0100"
CAMERA   = "realsense"
FRAME_ID = "0003"

# 你的 grasp 预测 .npy 根目录（按你现在日志结构）
# 例：logs/eval/rs/epoch15/fast_no_collision_gpu1
PRED_ROOT = "logs/eval/rs/epoch15/fast_no_collision_gpu1"

GRASP_NPY = os.path.join(PRED_ROOT, SCENE_ID, CAMERA, f"{FRAME_ID}.npy")

# 输出目录（统一工程规范）
OUT_DIR = "outputs/grasp_lines"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PLY = os.path.join(OUT_DIR, f"{SCENE_ID}_rs_{FRAME_ID}_grasps_lines.ply")

# ==================================================
# 可视化参数（论文风格）
# ==================================================
SHOW_TOP_K = 30          # 论文常用：10~50
LINE_RADIUS = 0.002      # “线”的粗细（m）1~3mm
COLOR = (0.1, 0.7, 1.0)  # 青蓝色，接近论文里那种蓝

# 线段长度（m）
APPROACH_LEN = 0.06      # 代表 approach/插入方向
BACK_LEN = 0.02          # 代表夹爪后方连线（让形状更像夹爪）


def cylinder_between(p0, p1, radius, color):
    """生成连接 p0->p1 的细圆柱 mesh（用来当线段）"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-9:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)

    # 把 cylinder 默认的 z 轴对齐到 v
    z = np.array([0.0, 0.0, 1.0])
    v_unit = v / length
    axis = np.cross(z, v_unit)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-9:
        # 平行/反平行
        if np.dot(z, v_unit) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        else:
            R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(z, v_unit), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cyl.rotate(R, center=(0, 0, 0))

    # cylinder 的中心在原点，沿 z 轴从 [-h/2, +h/2]
    mid = (p0 + p1) / 2.0
    cyl.translate(mid)
    return cyl


def grasp17_to_lines_mesh(g):
    """
    g: (17,)  [score, width, height, depth, R(9), t(3), obj_id]
    论文风格线夹爪：画一个“U”形（两根平行线 + 后方一根连接线）
    """
    score = float(g[0])
    width = float(g[1])

    R = g[4:13].reshape(3, 3)
    t = g[13:16].astype(np.float64)

    # 约定：用 R 的列向量作为坐标轴（不同实现可能会交换列）
    a = R[:, 0]  # approach 方向
    b = R[:, 1]  # closing（左右开合）方向

    # 夹爪两指中心线的起点（后方）与终点（前方）
    p_back = t - a * BACK_LEN
    p_front = t + a * APPROACH_LEN

    # 两根手指的左右偏移
    left_shift  = + b * (width / 2.0)
    right_shift = - b * (width / 2.0)

    # 两根平行线（两指）
    L0 = p_back + left_shift
    L1 = p_front + left_shift
    R0 = p_back + right_shift
    R1 = p_front + right_shift

    # 后方连接线（形成 U 形）
    B0 = p_back + left_shift
    B1 = p_back + right_shift

    meshes = []
    for s, e in [(L0, L1), (R0, R1), (B0, B1)]:
        m = cylinder_between(s, e, radius=LINE_RADIUS, color=COLOR)
        if m is not None:
            meshes.append(m)

    merged = o3d.geometry.TriangleMesh()
    for m in meshes:
        merged += m

    return merged, score


def main():
    if not os.path.exists(GRASP_NPY):
        raise FileNotFoundError(f"GRASP_NPY not found: {GRASP_NPY}")

    G = np.load(GRASP_NPY)  # (N, 17)
    assert G.ndim == 2 and G.shape[1] == 17, f"Unexpected grasp array shape: {G.shape}"

    # topK by score
    idx = np.argsort(-G[:, 0])[:SHOW_TOP_K]
    G = G[idx]

    merged = o3d.geometry.TriangleMesh()
    for g in G:
        m, _ = grasp17_to_lines_mesh(g)
        merged += m

    merged.compute_vertex_normals()
    o3d.io.write_triangle_mesh(OUT_PLY, merged, write_ascii=False)
    print("Saved:", OUT_PLY, " topK=", SHOW_TOP_K)


if __name__ == "__main__":
    main()
