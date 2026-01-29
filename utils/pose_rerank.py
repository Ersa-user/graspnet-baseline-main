"""
Pose-consistency re-ranking utilities for EI low-cost fusion (s2b).
"""

import numpy as np


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) / max(float(temperature), 1e-6)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def rerank_grasps_by_pose_confidence(grasp_scores: np.ndarray,
                                    pose_scores: np.ndarray,
                                    temperature: float = 1.0) -> np.ndarray:
    """
    Minimal placeholder reranker:
    - Convert pose_scores -> weights via softmax
    - Use max weight as a global multiplier on grasp scores (no geometry yet)

    Args:
        grasp_scores: (N,) raw grasp scores
        pose_scores:  (K,) Diff9D pred_scores
    Returns:
        new_scores: (N,) re-ranked scores
    """
    grasp_scores = np.asarray(grasp_scores, dtype=np.float32)
    pose_scores = np.asarray(pose_scores, dtype=np.float32)
    if pose_scores.size == 0:
        return grasp_scores

    w = softmax(pose_scores, temperature=temperature)
    return grasp_scores * float(np.max(w))
