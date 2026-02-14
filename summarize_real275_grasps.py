import os
import glob
import json
import argparse
import numpy as np
from collections import defaultdict


def safe_quantile(x, q):
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root dir that contains split/scene_x/*_grasps.npz, e.g. /.../Real_eval/graspnet_baseline")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--score_col", type=int, default=0,
                    help="Which column is used as grasp score (default 0).")
    ap.add_argument("--assume_sorted", action="store_true",
                    help="If set, use grasps[0,score_col] as top1; otherwise use max over topk.")
    ap.add_argument("--out", type=str, default="summary_real275_graspnet_baseline",
                    help="Output prefix (will write .json and .txt)")
    args = ap.parse_args()

    split_dir = os.path.join(args.root, args.split)
    pattern = os.path.join(split_dir, "scene_*", "*_grasps.npz")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError(f"No grasp npz found: {pattern}")

    # global accumulators
    top1_scores = []
    topk_max_scores = []
    topk_mean_scores = []
    topk_p95_scores = []
    empty_count = 0

    # per-scene
    per_scene = defaultdict(lambda: {
        "n": 0,
        "empty": 0,
        "top1": [],
        "topk_max": [],
        "topk_mean": [],
        "topk_p95": [],
    })

    for fp in files:
        scene = os.path.basename(os.path.dirname(fp))
        data = np.load(fp)
        g = data["grasps"]
        per_scene[scene]["n"] += 1

        if g.ndim != 2 or g.shape[0] == 0:
            empty_count += 1
            per_scene[scene]["empty"] += 1
            continue

        scores = g[:, args.score_col].astype(np.float32)

        if args.assume_sorted:
            top1 = float(scores[0])
        else:
            top1 = float(np.max(scores))

        topk_max = float(np.max(scores))
        topk_mean = float(np.mean(scores))
        topk_p95 = safe_quantile(scores, 0.95)

        top1_scores.append(top1)
        topk_max_scores.append(topk_max)
        topk_mean_scores.append(topk_mean)
        topk_p95_scores.append(topk_p95)

        per_scene[scene]["top1"].append(top1)
        per_scene[scene]["topk_max"].append(topk_max)
        per_scene[scene]["topk_mean"].append(topk_mean)
        per_scene[scene]["topk_p95"].append(topk_p95)

    def pack_stats(arr):
        arr = np.asarray(arr, dtype=np.float32)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "std": float(np.std(arr)) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "p05": safe_quantile(arr, 0.05),
            "p25": safe_quantile(arr, 0.25),
            "p75": safe_quantile(arr, 0.75),
            "p95": safe_quantile(arr, 0.95),
            "min": float(np.min(arr)) if arr.size else float("nan"),
            "max": float(np.max(arr)) if arr.size else float("nan"),
        }

    summary = {
        "root": args.root,
        "split": args.split,
        "num_files": len(files),
        "empty_frames": int(empty_count),
        "empty_ratio": float(empty_count) / float(len(files)),
        "score_col": args.score_col,
        "assume_sorted": bool(args.assume_sorted),
        "global": {
            "top1": pack_stats(top1_scores),
            "topk_max": pack_stats(topk_max_scores),
            "topk_mean": pack_stats(topk_mean_scores),
            "topk_p95": pack_stats(topk_p95_scores),
        },
        "per_scene": {},
    }

    # per-scene stats
    for scene, d in sorted(per_scene.items(), key=lambda x: x[0]):
        summary["per_scene"][scene] = {
            "n": int(d["n"]),
            "empty": int(d["empty"]),
            "empty_ratio": float(d["empty"]) / float(d["n"]) if d["n"] else float("nan"),
            "top1": pack_stats(d["top1"]),
            "topk_max": pack_stats(d["topk_max"]),
            "topk_mean": pack_stats(d["topk_mean"]),
            "topk_p95": pack_stats(d["topk_p95"]),
        }

    out_json = args.out + ".json"
    out_txt = args.out + ".txt"

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # pretty text
    g = summary["global"]
    lines = []
    lines.append(f"Real275 GraspNet baseline summary")
    lines.append(f"root: {args.root}")
    lines.append(f"split: {args.split}")
    lines.append(f"files: {len(files)}")
    lines.append(f"empty: {empty_count} ({summary['empty_ratio']*100:.2f}%)")
    lines.append("")
    for k in ["top1", "topk_max", "topk_mean", "topk_p95"]:
        st = g[k]
        lines.append(f"[{k}] count={st['count']} mean={st['mean']:.6f} std={st['std']:.6f} "
                     f"median={st['median']:.6f} p95={st['p95']:.6f} min={st['min']:.6f} max={st['max']:.6f}")
    lines.append("")
    lines.append("Per-scene (mean top1 / empty_ratio):")
    for scene, d in summary["per_scene"].items():
        lines.append(f"  {scene}: mean_top1={d['top1']['mean']:.6f} empty_ratio={d['empty_ratio']*100:.2f}% (n={d['n']})")

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

    print("[DONE]")
    print(" json:", os.path.abspath(out_json))
    print(" txt :", os.path.abspath(out_txt))


if __name__ == "__main__":
    main()
