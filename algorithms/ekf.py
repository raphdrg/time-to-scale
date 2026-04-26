"""
ekf.py — Extended Kalman Filter for EKF-SLAM.
"""

import numpy as np


class EKF:
    """
    Full-state EKF.
    State: mu = [x, y, theta, lm1_x, lm1_y, ...]
    """

    def __init__(
        self,
        init_pose: np.ndarray,   # (3,)
        motion_noise: np.ndarray, # (3,) std devs
        obs_noise: np.ndarray,    # (2,) std devs [range, bearing]
    ):
        self.state = init_pose.copy().astype(float)  # grows as landmarks are added
        n = len(self.state)
        self.covariance = np.zeros((n, n))
        # High uncertainty in robot pose initially (tune as needed)
        self.covariance[:3, :3] = np.diag([1e-4, 1e-4, 1e-4])

        self.Q = np.diag(motion_noise ** 2)  # process noise
        self.R = np.diag(obs_noise ** 2)     # observation noise

    # ------------------------------------------------------------------
    # Prediction (motion model: differential drive / odometry)
    # ------------------------------------------------------------------
    def predict(self, u: np.ndarray):
        """
        u: (3,) odometry delta [dx, dy, dtheta] in robot frame.
        """
        x, y, th = self.state[:3]
        dx, dy, dth = u

        # Rotate displacement into world frame
        cos_th, sin_th = np.cos(th), np.sin(th)
        self.state[0] += cos_th * dx - sin_th * dy
        self.state[1] += sin_th * dx + cos_th * dy
        self.state[2] = self._wrap(th + dth)

        # Jacobian of motion w.r.t. state (only robot block is non-trivial)
        n = len(self.state)
        F = np.eye(n)
        F[0, 2] = -sin_th * dx - cos_th * dy
        F[1, 2] =  cos_th * dx - sin_th * dy

        # Noise Jacobian (robot DOF only)
        G = np.zeros((n, 3))
        G[:3, :3] = np.array([
            [cos_th, -sin_th, 0],
            [sin_th,  cos_th, 0],
            [0,       0,      1],
        ])

        self.covariance = F @ self.covariance @ F.T + G @ self.Q @ G.T

    # ------------------------------------------------------------------
    # Update (range-bearing observation)
    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray, lm_idx: int):
        """
        obs:    (2,) [range, bearing]
        lm_idx: index of landmark in self.map (0-based, not counting robot state)
        """
        n = len(self.state)
        lm_state_idx = 3 + 2 * lm_idx  # position in state vector

        lm_x = self.state[lm_state_idx]
        lm_y = self.state[lm_state_idx + 1]
        rx, ry, rth = self.state[:3]

        dx = lm_x - rx
        dy = lm_y - ry
        q = dx ** 2 + dy ** 2

        # Predicted observation
        z_hat = np.array([np.sqrt(q), self._wrap(np.arctan2(dy, dx) - rth)])

        # Innovation
        innov = obs - z_hat
        innov[1] = self._wrap(innov[1])

        # Jacobian H (2 x n)
        H = np.zeros((2, n))
        sq = np.sqrt(q)
        H[0, 0] = -dx / sq;        H[0, 1] = -dy / sq
        H[1, 0] =  dy / q;         H[1, 1] = -dx / q;   H[1, 2] = -1
        H[0, lm_state_idx] =  dx / sq;  H[0, lm_state_idx + 1] =  dy / sq
        H[1, lm_state_idx] = -dy / q;   H[1, lm_state_idx + 1] =  dx / q

        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innov
        self.state[2] = self._wrap(self.state[2])
        self.covariance = (np.eye(n) - K @ H) @ self.covariance

    # ------------------------------------------------------------------
    # State augmentation — add new landmark
    # ------------------------------------------------------------------
    def augment_state(self, obs: np.ndarray, robot_pose: np.ndarray):
        """
        Initialise a new landmark from a range-bearing observation.
        Extends state vector and covariance in-place.
        """
        rx, ry, rth = robot_pose
        r, bearing = obs
        th_w = self._wrap(rth + bearing)

        lm_x = rx + r * np.cos(th_w)
        lm_y = ry + r * np.sin(th_w)

        self.state = np.append(self.state, [lm_x, lm_y])

        n_old = len(self.state) - 2
        n_new = len(self.state)
        cov_new = np.zeros((n_new, n_new))
        cov_new[:n_old, :n_old] = self.covariance

        # Initialise landmark covariance with large uncertainty
        cov_new[n_old:, n_old:] = np.eye(2) * 1.0

        self.covariance = cov_new

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi