import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

# IMPORTANT: this is your NEW dataset (GraspNet-friendly)
from dataset_real275 import Real275GraspNetDataset

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Make sure repo root & common subdirs are importable
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "models"), os.path.join(REPO_ROOT, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module_from_path(py_path: str, module_name: str):
    if py_path is None or not os.path.exists(py_path):
        return None
    try:
        return SourceFileLoader(module_name, py_path).load_module()
    except Exception as e:
        print(f"[LOAD-FAIL] {py_path} | {repr(e)}")
        return None


def locate_graspnet_and_decode():
    # For your repo, it is clearly in models/graspnet.py
    graspnet_file = os.path.join(REPO_ROOT, "models", "graspnet.py")
    if not os.path.exists(graspnet_file):
        raise FileNotFoundError(f"Cannot find: {graspnet_file}")

    m = _load_module_from_path(graspnet_file, "_graspnet_mod")
    if m is None:
        raise ImportError(f"Failed to import module from: {graspnet_file}")

    if not hasattr(m, "GraspNet"):
        raise ImportError("models/graspnet.py has no symbol: GraspNet")
    if not hasattr(m, "pred_decode"):
        raise ImportError("models/graspnet.py has no symbol: pred_decode")

    print("[LOCATE-OK]")
    print("  GraspNet from :", graspnet_file)
    print("  pred_decode from:", graspnet_file)

    return m, getattr(m, "GraspNet"), getattr(m, "pred_decode")


def load_checkpoint(net, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    net.load_state_dict(state, strict=False)


def _ensure_batch_grasp_point(end_points: dict, fallback_ns: int = 1024):
    """
    Ensure end_points has 'batch_grasp_point': (B, Ns, 3)
    Prefer dataset-provided field; else fallback to seed/graspable keys; else slice point_clouds.
    """
    if "batch_grasp_point" in end_points:
        return end_points

    candidates = [
        "seed_xyz", "seed_points",
        "graspable_seed_xyz", "xyz_graspable",
        "graspable_xyz", "graspable_points",
    ]
    for k in candidates:
        if k in end_points:
            end_points["batch_grasp_point"] = end_points[k]
            return end_points

    pc = end_points["point_clouds"]  # (B,N,3)
    Ns = min(fallback_ns, pc.shape[1])
    end_points["batch_grasp_point"] = pc[:, :Ns, :].contiguous()
    return end_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/home/dingxin/data/Real275/Real_graspnet_input")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="/home/dingxin/data/Real275/Real_eval/graspnet_baseline")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--num_points", type=int, default=20000)
    ap.add_argument("--seed_points", type=int, default=1024)
    ap.add_argument("--fallback_ns", type=int, default=1024)
    ap.add_argument("--random_sample", action="store_true", help="random sample points each epoch")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # locate net/decoder
    m_net, GraspNet, pred_decode = locate_graspnet_and_decode()

    # ---------------- PATCH training-only deps ----------------
    # 1) disable process_grasp_labels (needs object_poses_list etc.)
    if hasattr(m_net, "process_grasp_labels"):
        m_net.process_grasp_labels = lambda ep: ep

    # 2) patch match_grasp_view_and_label in BOTH places
    import utils.label_generation as LG

    def _infer_grasp_top_views_rot(ep):
        # If a fork stores predicted rotations, use them
        for k in ["grasp_top_views_rot", "grasp_top_view_rot", "top_view_rot", "top_views_rot"]:
            if k in ep:
                return ep[k]

        pc = ep["point_clouds"]
        B = pc.shape[0]
        Ns = None
        if "batch_grasp_point" in ep:
            Ns = int(ep["batch_grasp_point"].shape[1])
        if Ns is None:
            Ns = args.fallback_ns

        eye = torch.eye(3, device=pc.device, dtype=pc.dtype).view(1, 1, 3, 3)
        return eye.repeat(B, Ns, 1, 1)

    def _patched_match_grasp_view_and_label(ep):
        rot = _infer_grasp_top_views_rot(ep)
        # must return 5 items
        return rot, None, None, None, ep

    LG.match_grasp_view_and_label = _patched_match_grasp_view_and_label
    if hasattr(m_net, "match_grasp_view_and_label"):
        m_net.match_grasp_view_and_label = _patched_match_grasp_view_and_label
    # ---------------------------------------------------------

    # build net
    try:
        net = GraspNet().to(device)
    except TypeError:
        try:
            net = GraspNet(input_feature_dim=0).to(device)
        except TypeError:
            net = GraspNet(input_feature_dim=3).to(device)

    if hasattr(net, "is_training"):
        net.is_training = False

    net.eval()
    load_checkpoint(net, args.ckpt, device)

    # sanity: must expose submodules for split-inference
    if not (hasattr(net, "view_estimator") and hasattr(net, "grasp_generator")):
        raise RuntimeError("This GraspNet fork does not expose view_estimator/grasp_generator; cannot run split-inference path.")

    # dataset
    ds = Real275GraspNetDataset(
        args.data_root,
        split=args.split,
        num_points=args.num_points,
        seed_points=args.seed_points,
        random_sample=args.random_sample,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    for batch in tqdm(dl, desc="GraspNet baseline inference"):
        with torch.no_grad():
            # dataset already returns batched tensors: (1,N,3)
            pc = batch["point_clouds"].to(device)
            cc = batch["cloud_colors"].to(device)
            bgp = batch["batch_grasp_point"].to(device)

            # fix double-batch: (B,1,N,3) -> (B,N,3)
            if pc.dim() == 4 and pc.size(1) == 1:
                pc = pc.squeeze(1)
            if cc.dim() == 4 and cc.size(1) == 1:
                cc = cc.squeeze(1)

            # batch_grasp_point: (B,1,Ns,3) -> (B,Ns,3)
            if bgp.dim() == 4 and bgp.size(1) == 1:
                bgp = bgp.squeeze(1)

            end_points = {
                "point_clouds": pc,  # (B,N,3)
                "cloud_colors": cc,  # (B,N,3)
                "batch_grasp_point": bgp,  # (B,Ns,3)
            }
            # 1) view estimator
            end_points = net.view_estimator(end_points)

            # 2) ensure batch_grasp_point exists (use dataset, else fallback)
            end_points = _ensure_batch_grasp_point(end_points, fallback_ns=args.fallback_ns)

            # 3) grasp generator (match fn patched -> no batch_grasp_view_rot)
            end_points = net.grasp_generator(end_points)

            # 4) decode
            grasps = pred_decode(end_points)

        B = end_points["point_clouds"].shape[0]
        for i in range(B):
            scene = batch["scene"][i]
            frame = batch["frame"][i]
            out_scene = os.path.join(args.out_dir, args.split, scene)
            os.makedirs(out_scene, exist_ok=True)

            g = grasps[i]
            if isinstance(g, torch.Tensor):
                g = g.detach().cpu().numpy()
            else:
                g = np.asarray(g)

            if args.topk > 0 and g.ndim >= 2 and g.shape[0] > args.topk:
                g = g[: args.topk]

            out_path = os.path.join(out_scene, f"{frame}_grasps.npz")
            np.savez_compressed(out_path, grasps=g)

    print("\n[DONE]")
    print("saved to:", os.path.join(args.out_dir, args.split))


if __name__ == "__main__":
    main()
