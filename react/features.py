from __future__ import annotations

import numpy as np


def hand_size_metric(points: np.ndarray) -> float:
    """Use wrist-to-middle-MCP distance as a scale reference."""
    wrist = points[0]
    middle_mcp = points[9]
    distance = np.linalg.norm(middle_mcp - wrist)
    return max(float(distance), 1e-6)


def to_feature_vec(landmarks, width: int, height: int) -> np.ndarray:
    """
    Convert 21 MediaPipe hand landmarks into a 42-D normalized feature vector.

    The representation is centered at the wrist and divided by hand size, making
    it less sensitive to absolute image position and camera distance.
    """
    points = np.asarray(
        [(lm.x * width, lm.y * height) for lm in landmarks.landmark],
        dtype=np.float32,
    )

    wrist = points[0].copy()
    scale = hand_size_metric(points)

    points -= wrist
    points /= scale

    return points.reshape(-1)
