"""
Pose confidence gating utilities for EI low-cost fusion.
"""

import numpy as np


def pose_confidence_from_diff9d(pkl_data):
    """
    Extract a scalar pose confidence from Diff9D results.

    Args:
        pkl_data (dict): loaded results_xxx.pkl from Diff9D

    Returns:
        float: pose confidence in [0, 1]
    """
    scores = pkl_data.get("pred_scores", None)
    if scores is None or len(scores) == 0:
        return 0.0

    scores = np.asarray(scores, dtype=np.float32)
    return float(scores.max())
