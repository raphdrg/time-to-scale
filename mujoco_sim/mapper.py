"""
2D occupancy grid mapper.

Consumes LiDARScan objects (world-frame hit points + sensor position) and
builds a probabilistic 2D map using log-odds Bayesian updates.

How it works
------------
Each grid cell stores a log-odds value L:
    L  >>  0  →  cell is probably occupied   (wall / obstacle)
    L  <<  0  →  cell is probably free       (open space)
    L  ≈   0  →  unknown (never seen)

On each LiDAR sweep:
  For every ray that returned a hit:
    1.  Walk from sensor cell to the cell just before the hit  →  mark FREE
        (Bresenham line — the ray passed through these cells unobstructed)
    2.  Mark the hit cell  →  OCCUPIED

This free-space carving is what makes occupancy grids useful: after a few
sweeps the driveable area becomes clearly free, not just "unvisited".

Log-odds update:
    L[cell] += LOG_ODDS_HIT   (occupied update)
    L[cell] += LOG_ODDS_FREE  (free update, applied to ray cells)
Values are clamped to [-MAX_CLAMP, MAX_CLAMP] to prevent over-confidence.

Coordinate convention
---------------------
World X → grid column (right = +col)
World Y → grid row    (up    = -row, because image origin is top-left)
Robot starts at the centre of the grid.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sensors.lidar import LiDARScan


# Log-odds update values
LOG_ODDS_HIT  =  0.85   # strong evidence of obstacle
LOG_ODDS_FREE = -0.40   # weak evidence of free space (rays pass through fast)
MAX_CLAMP     = 10.0    # prevent over-confidence in either direction


class OccupancyGrid:
    """
    2D log-odds occupancy grid.

    Parameters
    ----------
    width_m, height_m : float
        Physical size of the grid in metres.  Robot spawns at centre.
    resolution : float
        Metres per cell.  Smaller = more detail, more memory.
        0.05 m → 2 cm resolution at 20×20 m = 400×400 cells.
    """

    def __init__(
        self,
        width_m: float = 20.0,
        height_m: float = 20.0,
        resolution: float = 0.05,
    ):
        self.res = resolution
        self.cols = int(width_m  / resolution)
        self.rows = int(height_m / resolution)
        # Make dims odd so the physical centre falls exactly on a cell centre
        if self.cols % 2 == 0: self.cols += 1
        if self.rows % 2 == 0: self.rows += 1

        self.cx = self.cols // 2   # column index of world origin
        self.cy = self.rows // 2   # row    index of world origin

        # Log-odds grid, initialised to 0 (unknown)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        self._scan_count = 0

    # ── Coordinate helpers ────────────────────────────────────────────────

    def world_to_cell(self, x: float, y: float):
        """Convert world (x, y) metres → (col, row) grid indices."""
        col = int(self.cx + x / self.res)
        row = int(self.cy - y / self.res)   # flip Y: world +Y = image up
        return col, row

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    # ── Core update ───────────────────────────────────────────────────────

    def update(self, scan: LiDARScan):
        """
        Integrate one LiDAR sweep into the map.

        For each hit point:
          - Bresenham from sensor position → hit cell-1: mark FREE
          - Hit cell: mark OCCUPIED
        """
        if scan.num_hits == 0:
            return

        sx, sy = scan.sensor_pos[0], scan.sensor_pos[1]
        src_col, src_row = self.world_to_cell(sx, sy)

        for i in range(scan.num_hits):
            hx, hy = scan.points[i, 0], scan.points[i, 1]
            dst_col, dst_row = self.world_to_cell(hx, hy)

            if not self.in_bounds(dst_col, dst_row):
                continue

            # Walk the ray: mark free cells along it
            ray_cells = _bresenham(src_col, src_row, dst_col, dst_row)
            for col, row in ray_cells[:-1]:   # exclude the hit cell itself
                if self.in_bounds(col, row):
                    self.grid[row, col] = np.clip(
                        self.grid[row, col] + LOG_ODDS_FREE,
                        -MAX_CLAMP, MAX_CLAMP
                    )

            # Mark hit cell as occupied
            self.grid[dst_row, dst_col] = np.clip(
                self.grid[dst_row, dst_col] + LOG_ODDS_HIT,
                -MAX_CLAMP, MAX_CLAMP
            )

        self._scan_count += 1

    # ── Output ────────────────────────────────────────────────────────────

    def probability_map(self) -> np.ndarray:
        """Convert log-odds grid to probability [0, 1]. 0.5 = unknown."""
        return 1.0 / (1.0 + np.exp(-self.grid))

    def save(self, path: str, trajectory: np.ndarray = None):
        """
        Render the occupancy grid and save to PNG.

        Colour convention:
          black  = occupied  (high probability)
          white  = free      (low probability)
          gray   = unknown   (probability ≈ 0.5, never updated)

        Parameters
        ----------
        trajectory : (N, 2) array of world XY positions, optional.
            If provided, draws the robot's path on the map.
        """
        prob = self.probability_map()

        # Map probability to greyscale: free=1 (white), occupied=0 (black),
        # unknown=0.5 (mid-grey)
        grey = 1.0 - prob   # flip: high prob → dark

        # Render as RGB so we can overlay a coloured trajectory
        rgb = np.stack([grey, grey, grey], axis=-1)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor("#888888")
        fig.patch.set_facecolor("#111111")

        extent = [
            -self.cx * self.res,
             (self.cols - self.cx) * self.res,
            -(self.rows - self.cy) * self.res,
             self.cy * self.res,
        ]
        ax.imshow(rgb, origin="upper", extent=extent, vmin=0, vmax=1,
                  interpolation="nearest")

        if trajectory is not None and len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                    "-", color="dodgerblue", linewidth=1.5, label="robot path")
            ax.plot(*trajectory[0],  "go", markersize=7, label="start")
            ax.plot(*trajectory[-1], "ro", markersize=7, label="end")

        occ_patch  = mpatches.Patch(color="black",   label="occupied")
        free_patch = mpatches.Patch(color="white",   label="free")
        unk_patch  = mpatches.Patch(color="#888888", label="unknown")
        handles = [occ_patch, free_patch, unk_patch]
        if trajectory is not None:
            handles += [
                plt.Line2D([0],[0], color="dodgerblue", lw=1.5, label="robot path"),
                plt.Line2D([0],[0], marker="o", color="g", lw=0, markersize=7, label="start"),
                plt.Line2D([0],[0], marker="o", color="r", lw=0, markersize=7, label="end"),
            ]

        ax.legend(handles=handles, loc="upper right",
                  facecolor="#222222", labelcolor="white", fontsize=9)
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.set_title(
            f"Occupancy grid — {self._scan_count} sweeps  |  "
            f"res={self.res}m  |  {self.cols}×{self.rows} cells",
            color="white"
        )
        ax.tick_params(colors="gray")
        ax.grid(False)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Map saved → {path}")


# ── Bresenham line ────────────────────────────────────────────────────────────

def _bresenham(c0: int, r0: int, c1: int, r1: int):
    """
    Return list of (col, row) cells on the line from (c0,r0) to (c1,r1).
    Classic integer Bresenham — no floating point, very fast.
    Includes both endpoints.
    """
    cells = []
    dc = abs(c1 - c0);  sc = 1 if c1 > c0 else -1
    dr = abs(r1 - r0);  sr = 1 if r1 > r0 else -1
    err = dc - dr

    c, r = c0, r0
    while True:
        cells.append((c, r))
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr;  c += sc
        if e2 <  dc:
            err += dc;  r += sr

    return cells
