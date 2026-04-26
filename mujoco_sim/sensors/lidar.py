"""
LiDAR sensor simulation.

Uses MuJoCo's mj_ray() for ray casting — this fires a ray from a world-frame
origin in a world-frame direction and returns the distance to the first geom
hit.  It is physically correct: rays respect the geometry of the scene.

2D mode  (RPLIDAR-style):   elevation_angles=[0.0]
                             One horizontal ring of rays.
3D mode  (Velodyne-style):  elevation_angles=[-15,-10,-5,0,5,10,15]
                             Multiple stacked rings at different elevations.

Ray directions are precomputed in the sensor's local frame at init time.
Each sweep, we only rotate them into world frame via the site's rotation
matrix — much cheaper than recomputing sin/cos every step.

Noise model:
    range_measured = range_true + N(0, noise_std²)
    Hits outside [min_range, max_range] are discarded.
"""

import math
import numpy as np
import mujoco
from dataclasses import dataclass


@dataclass
class LiDARScan:
    t: float                     # simulation time (s)
    points: np.ndarray           # (N, 3) world-frame XYZ hit positions
    ranges: np.ndarray           # (N,)   measured ranges (noisy, metres)
    azimuths: np.ndarray         # (N,)   azimuth angles (radians, 0 = forward)
    elevations: np.ndarray       # (N,)   elevation angles (radians)
    sensor_pos: np.ndarray       # (3,)   world position of sensor at sweep time

    @property
    def xy(self) -> np.ndarray:
        """Convenience: 2D hit points (N, 2), ignores Z."""
        return self.points[:, :2]

    @property
    def num_hits(self) -> int:
        return len(self.ranges)


class LiDARSensor:
    """
    Simulated LiDAR via MuJoCo ray casting.

    Parameters
    ----------
    model : mujoco.MjModel
    num_rays : int
        Number of azimuth rays per elevation ring.  360 = 1° resolution.
    max_range : float
        Maximum valid range [m].  Rays beyond this are dropped.
    min_range : float
        Minimum valid range [m].  Sensor blind zone — close hits are dropped.
    noise_std : float
        1-sigma range noise [m].  Typical solid-state LiDAR: 0.01–0.03 m.
    elevation_angles : list of float
        Elevation angles in degrees.  [0.0] → 2D scan.  Multiple values → 3D.
    lidar_site_name : str
        Name of the MuJoCo <site> that defines the sensor's pose.
    base_body_name : str
        Name of the robot body to exclude from ray hits (prevents self-hits).
    """

    def __init__(
        self,
        model,
        num_rays: int = 360,
        max_range: float = 8.0,
        min_range: float = 0.05,
        noise_std: float = 0.01,
        elevation_angles: list = None,
        lidar_site_name: str = "lidar_site",
        base_body_name: str = "base",
    ):
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, lidar_site_name
        )
        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
        )
        if self.site_id < 0:
            raise ValueError(f"Site '{lidar_site_name}' not found in model.")
        if self.base_body_id < 0:
            raise ValueError(f"Body '{base_body_name}' not found in model.")

        self.num_rays         = num_rays
        self.max_range        = max_range
        self.min_range        = min_range
        self.noise_std        = noise_std
        self.elevation_angles = elevation_angles if elevation_angles is not None else [0.0]

        # Precompute ray bundles once — (elevation_rad, (num_rays, 3) local dirs)
        self._azimuths   = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
        self._ray_bundles = self._build_directions()

    def _build_directions(self):
        """
        Build local-frame ray direction vectors.

        Sensor frame convention: X=forward, Y=left, Z=up.
        Azimuth 0 points along +X (forward), increases counter-clockwise.
        Elevation 0 is horizontal, positive = upward.
        """
        bundles = []
        for elev_deg in self.elevation_angles:
            elev  = math.radians(elev_deg)
            cos_e = math.cos(elev)
            sin_e = math.sin(elev)
            dirs = np.stack([
                cos_e * np.cos(self._azimuths),
                cos_e * np.sin(self._azimuths),
                sin_e * np.ones(self.num_rays),
            ], axis=1)  # (num_rays, 3)
            bundles.append((math.radians(elev_deg), dirs))
        return bundles

    # ------------------------------------------------------------------

    def sweep(self, model, data) -> LiDARScan:
        """
        Fire all rays for one complete sweep.

        Iterates over each elevation ring, then each azimuth ray within that
        ring.  Rotates local directions into world frame using the site's
        rotation matrix (data.site_xmat).  Calls mj_ray() for each ray.
        Adds gaussian range noise to valid hits.

        Returns a LiDARScan containing only valid hits (within range gates).
        """
        lidar_pos = data.site_xpos[self.site_id].copy()
        lidar_rot = data.site_xmat[self.site_id].reshape(3, 3)

        # Rotate all direction bundles into world frame at once (matrix multiply)
        # dirs_local: (num_rays, 3)  →  dirs_world: (num_rays, 3)
        # Using @ operator: (num_rays, 3) @ (3, 3).T = (num_rays, 3)
        hit_points     = []
        hit_ranges     = []
        hit_azimuths   = []
        hit_elevations = []

        geom_id = np.array([-1], dtype=np.int32)  # reused buffer

        for elev_rad, dirs_local in self._ray_bundles:
            dirs_world = dirs_local @ lidar_rot.T  # (num_rays, 3)

            for i, d_world in enumerate(dirs_world):
                geom_id[0] = -1
                dist = mujoco.mj_ray(
                    model, data,
                    lidar_pos,          # ray origin (world frame)
                    d_world,            # ray direction (world frame)
                    None,               # geomgroup filter: None = all groups
                    1,                  # flg_static: include static geoms
                    self.base_body_id,  # exclude robot body — avoids self-hit
                    geom_id,
                )

                if dist < 0 or dist > self.max_range:
                    continue
                if dist < self.min_range:
                    continue

                dist_noisy = dist + np.random.randn() * self.noise_std
                dist_noisy = max(self.min_range, dist_noisy)

                hit_world = lidar_pos + d_world * dist_noisy

                hit_points.append(hit_world)
                hit_ranges.append(dist_noisy)
                hit_azimuths.append(self._azimuths[i])
                hit_elevations.append(elev_rad)

        if hit_points:
            return LiDARScan(
                t=data.time,
                points=np.array(hit_points, dtype=np.float32),
                ranges=np.array(hit_ranges, dtype=np.float32),
                azimuths=np.array(hit_azimuths, dtype=np.float32),
                elevations=np.array(hit_elevations, dtype=np.float32),
                sensor_pos=lidar_pos,
            )

        return LiDARScan(
            t=data.time,
            points=np.empty((0, 3), dtype=np.float32),
            ranges=np.empty(0, dtype=np.float32),
            azimuths=np.empty(0, dtype=np.float32),
            elevations=np.empty(0, dtype=np.float32),
            sensor_pos=lidar_pos,
        )
