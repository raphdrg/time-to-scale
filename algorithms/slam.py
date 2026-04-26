"""
slam.py — Main SLAM pipeline.
Called each simulation step with sensor observations.
"""

import numpy as np
from .ekf import EKF
from .map import LandmarkMap


class SLAM:
    """
    EKF-SLAM pipeline for MuJoCo simulation.

    Expected coordinate frame: robot-centric x-forward, y-left, z-up.
    State vector: [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                - motion_noise (3,)   : [std_x, std_y, std_theta]
                - obs_noise    (2,)   : [std_range, std_bearing]
                - assoc_threshold     : Mahalanobis distance for data association
                - init_pose    (3,)   : [x, y, theta] initial robot pose (world frame)
        """
        self.cfg = config
        self.ekf = EKF(
            init_pose=np.array(config["init_pose"]),
            motion_noise=np.array(config["motion_noise"]),
            obs_noise=np.array(config["obs_noise"]),
        )
        self.map = LandmarkMap(assoc_threshold=config.get("assoc_threshold", 1.5))
        self.step_count = 0

    # ------------------------------------------------------------------
    # Main interface — call this once per simulation step
    # ------------------------------------------------------------------
    def update(
        self,
        odometry: np.ndarray,           # shape (3,): [dx, dy, dtheta] in robot frame
        observations: list[np.ndarray], # list of shape-(2,) arrays: [range, bearing]
        obs_ids: list[int] | None = None,  # optional ground-truth landmark IDs (for debugging)
    ) -> dict:
        """
        Run one SLAM cycle.

        Returns:
            {
              "pose":      np.ndarray (3,)  — estimated [x, y, theta] in world frame,
              "landmarks": list of (x, y)   — estimated landmark positions,
              "covariance": np.ndarray      — full state covariance matrix,
            }
        """
        # 1. Prediction step (motion model)
        self.ekf.predict(odometry)

        # 2. Update step (sensor observations)
        for i, obs in enumerate(observations):
            gt_id = obs_ids[i] if obs_ids else None
            lm_idx = self.map.associate(obs, self.ekf.state, self.ekf.covariance, gt_id)

            if lm_idx is None:
                # New landmark — initialise
                lm_idx = self.map.add_landmark(obs, self.ekf.state)
                self.ekf.augment_state(obs, self.ekf.state[:3])

            self.ekf.update(obs, lm_idx)

        self.step_count += 1

        return {
        "pose": self.ekf.state[:3].copy(),
        "landmarks": self.map.get_positions(self.ekf.state),
        "covariance": self.ekf.covariance.copy(),
        "n_landmarks": self.map.n_landmarks,
        "time": self.step_count * 0.1,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reset(self, init_pose: np.ndarray | None = None):
        """Reset SLAM state (e.g. between episodes)."""
        pose = init_pose if init_pose is not None else np.array(self.cfg["init_pose"])
        self.ekf = EKF(
            init_pose=pose,
            motion_noise=np.array(self.cfg["motion_noise"]),
            obs_noise=np.array(self.cfg["obs_noise"]),
        )
        self.map = LandmarkMap(assoc_threshold=self.cfg.get("assoc_threshold", 1.5))
        self.step_count = 0