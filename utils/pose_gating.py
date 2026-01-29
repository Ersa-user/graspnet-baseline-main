"""
Pose confidence gating utilities for EI low-cost fusion.
"""

import numpy as np


def pose_confidence_from_diff9d(pkl_data):
    """
    Extract a scalar pose confidence from Diff9D / Wild6D results.

    This function is designed to be:
    - backward-compatible with original Diff9D/Wild6D pkl format
    - robust to scalar / array pred_scores
    - safe for EI low-cost fusion and future end-to-end usage

    Args:
        pkl_data (dict): loaded results_xxx.pkl from Diff9D

    Returns:
        float or None: pose confidence (higher means more confident),
                       None if confidence is unavailable
    """
    if not isinstance(pkl_data, dict):
        return None

    if "pred_scores" not in pkl_data:
        return None

    scores = pkl_data["pred_scores"]

    # Case 1: scalar score (common in Wild6D results)
    if isinstance(scores, (float, int)):
        return float(scores)

    # Case 2: array-like scores (future multi-hypothesis / posterior)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return None

    # For now: use max score as confidence
    # (later can be replaced by entropy / gap / variance, etc.)
    return float(scores.max())
