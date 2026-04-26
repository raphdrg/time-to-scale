"""
map.py — Landmark map and data association.
"""

import numpy as np


class LandmarkMap:
    """
    Tracks known landmarks and performs nearest-neighbour data association
    using Mahalanobis distance (or raw range-bearing distance as fallback).
    """

    def __init__(self, assoc_threshold: float = 1.5):
        self.threshold = assoc_threshold
        self.n_landmarks = 0

    def associate(
        self,
        obs: np.ndarray,        # (2,) [range, bearing]
        state: np.ndarray,
        covariance: np.ndarray,
        gt_id: int | None = None,
    ) -> int | None:
        """
        Returns 0-based landmark index if association found, else None.
        If gt_id is provided (debug mode), bypass distance check.
        """
        if gt_id is not None and gt_id < self.n_landmarks:
            return gt_id

        if self.n_landmarks == 0:
            return None

        rx, ry, rth = state[:3]
        r, bearing = obs
        th_w = (rth + bearing + np.pi) % (2 * np.pi) - np.pi
        obs_world = np.array([rx + r * np.cos(th_w), ry + r * np.sin(th_w)])

        best_dist = float("inf")
        best_idx = None

        for i in range(self.n_landmarks):
            lm_state_idx = 3 + 2 * i
            lm_pos = state[lm_state_idx: lm_state_idx + 2]
            diff = obs_world - lm_pos

            # Simple Euclidean association (swap for Mahalanobis if desired)
            dist = np.linalg.norm(diff)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx if best_dist < self.threshold else None

    def add_landmark(self, obs: np.ndarray, state: np.ndarray) -> int:
        """Register a new landmark; returns its 0-based index."""
        idx = self.n_landmarks
        self.n_landmarks += 1
        return idx

    def get_positions(self, state: np.ndarray) -> list[tuple[float, float]]:
        """Return list of (x, y) world-frame landmark positions from state vector."""
        return [
            (state[3 + 2 * i], state[3 + 2 * i + 1])
            for i in range(self.n_landmarks)
        ]