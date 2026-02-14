import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class Real275GraspNetDataset(Dataset):
    """
    GraspNet-friendly Real275 dataset loader.
    Reads npz produced by convert_real_processed_to_graspnet.py

    Expected structure:
      root/test/scene_x/0000.npz

    npz fields:
      - points: (N,3) float32 (meters)
      - colors: (N,3) float32 (0~1)  [optional]
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        num_points: int = 20000,
        seed_points: int = 1024,
        random_sample: bool = True,
    ):
        self.root = root
        self.split = split
        self.num_points = int(num_points)
        self.seed_points = int(seed_points)
        self.random_sample = bool(random_sample)

        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"[Real275] Split dir not found: {split_root}")

        pattern = os.path.join(split_root, "scene_*", "*.npz")
        self.files = sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise RuntimeError(f"[Real275] No npz files found with: {pattern}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _ensure_colors(points: np.ndarray, colors: np.ndarray | None):
        if colors is None:
            colors = np.zeros_like(points, dtype=np.float32)
        else:
            colors = colors.astype(np.float32)
            if colors.shape != points.shape:
                # if colors are missing/invalid, fallback to zeros
                colors = np.zeros_like(points, dtype=np.float32)
        return colors

    def _sample_or_pad(self, points: np.ndarray, colors: np.ndarray):
        """
        Make exactly self.num_points.
        - if N >= num_points: sample without replacement (or take first if random_sample=False)
        - if N < num_points: sample with replacement (repeat)
        """
        N = points.shape[0]
        M = self.num_points

        if N == M:
            return points, colors

        if N > M:
            if self.random_sample:
                idx = np.random.choice(N, M, replace=False)
            else:
                idx = np.arange(M)
        else:
            # pad by repeating
            idx = np.random.choice(N, M, replace=True) if self.random_sample else np.arange(M) % N

        return points[idx], colors[idx]

    def __getitem__(self, idx: int):
        path = self.files[idx]
        d = np.load(path)

        points = d["points"].astype(np.float32)  # (N,3)
        colors = d["colors"].astype(np.float32) if "colors" in d else None
        colors = self._ensure_colors(points, colors)

        # fix point count
        points, colors = self._sample_or_pad(points, colors)  # (num_points,3)

        # parse scene/frame from path: .../test/scene_1/0000.npz
        scene = os.path.basename(os.path.dirname(path))
        frame = os.path.splitext(os.path.basename(path))[0]

        # convert to torch
        pc = torch.from_numpy(points)  # (N,3)
        cc = torch.from_numpy(colors)  # (N,3)

        # IMPORTANT: GraspNet forward expects batch dimension: (B,N,3)
        pc_b = pc.unsqueeze(0)  # (1,N,3)
        cc_b = cc.unsqueeze(0)  # (1,N,3)

        # Provide batch_grasp_point to satisfy some forks' grasp_generator forward
        # Use first seed_points points as "seeds" (simple, stable)
        Ns = min(self.seed_points, pc_b.shape[1])
        batch_grasp_point = pc_b[:, :Ns, :].contiguous()  # (1,Ns,3)

        return {
            # GraspNet-friendly keys
            "point_clouds": pc_b,                 # (1,N,3)
            "cloud_colors": cc_b,                 # (1,N,3)
            "batch_grasp_point": batch_grasp_point,  # (1,Ns,3)  <- avoids KeyError

            # meta
            "scene": scene,
            "frame": frame,
            "path": path,
        }
