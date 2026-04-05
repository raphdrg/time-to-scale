"""
RPLIDAR C1 - Live scan + global map visualizer
  Left panel:  current 360° sweep (refreshes every scan)
  Right panel: accumulating occupancy grid (global map)

Run: python main.py
"""

import sys
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # works best on macOS; change to "Qt5Agg" if needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from lidar_reader import LidarReader


# ── Occupancy grid ──────────────────────────────────────────────────────────

class GlobalMap:
    def __init__(self, cfg):
        self.res = cfg["resolution_m"]
        self.weight = cfg["accumulate_weight"]
        w = int(cfg["width_m"] / self.res)
        h = int(cfg["height_m"] / self.res)
        # make dimensions odd so the sensor sits exactly at center
        self.w = w + (w % 2 == 0)
        self.h = h + (h % 2 == 0)
        self.grid = np.zeros((self.h, self.w), dtype=np.float32)
        self.cx = self.w // 2   # sensor column
        self.cy = self.h // 2   # sensor row

    def add_scan(self, points):
        for (x, y) in points:
            col = int(self.cx + x / self.res)
            row = int(self.cy - y / self.res)   # flip y so up = positive y
            if 0 <= col < self.w and 0 <= row < self.h:
                self.grid[row, col] = min(self.grid[row, col] + self.weight, 1.0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    disp = config["display"]
    refresh_interval_ms = int(1000 / disp["refresh_hz"])
    max_range = config["lidar"]["max_range_m"]

    # start lidar
    reader = LidarReader(config)
    try:
        reader.start()
    except Exception as e:
        print(f"Failed to connect to LIDAR: {e}")
        print("Check: port in config.yaml, driver installed, cable connected.")
        sys.exit(1)

    gmap = GlobalMap(config["global_map"])

    # ── figure setup ──────────────────────────────────────────────────────
    fig, (ax_scan, ax_map) = plt.subplots(1, 2, figsize=(14, 7),
                                           facecolor="#111111")
    fig.suptitle("RPLIDAR C1", color="white", fontsize=13)

    def _style_ax(ax, title):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_aspect("equal")

    # current scan axes
    _style_ax(ax_scan, "Current Scan")
    ax_scan.set_xlim(-max_range, max_range)
    ax_scan.set_ylim(-max_range, max_range)
    ax_scan.set_xlabel("x (m)", color="gray")
    ax_scan.set_ylabel("y (m)", color="gray")
    ax_scan.axhline(0, color="#333333", lw=0.5)
    ax_scan.axvline(0, color="#333333", lw=0.5)
    # sensor marker
    ax_scan.plot(0, 0, "o", color="yellow", ms=5, zorder=5)

    scat_scan = ax_scan.scatter([], [], s=disp["current_scan_size"],
                                 c=[], cmap="plasma",
                                 vmin=0, vmax=max_range, lw=0)

    # global map axes
    _style_ax(ax_map, "Global Map (accumulated)")
    occ_cmap = LinearSegmentedColormap.from_list(
        "occ", ["#111111", "#00FFAA"], N=256)
    img_map = ax_map.imshow(
        gmap.grid, cmap=occ_cmap, vmin=0, vmax=1,
        origin="upper",
        extent=[-config["global_map"]["width_m"] / 2,
                 config["global_map"]["width_m"] / 2,
                -config["global_map"]["height_m"] / 2,
                 config["global_map"]["height_m"] / 2],
    )
    ax_map.set_xlabel("x (m)", color="gray")
    ax_map.set_ylabel("y (m)", color="gray")
    ax_map.plot(0, 0, "o", color="yellow", ms=5, zorder=5)

    # status text
    status_txt = fig.text(0.5, 0.01, "", ha="center", color="gray", fontsize=9)

    last_scan_count = [0]

    # ── animation update ──────────────────────────────────────────────────
    def update(_frame):
        scan = reader.get_scan()

        if scan and reader.scan_count != last_scan_count[0]:
            last_scan_count[0] = reader.scan_count

            xs = np.array([p[0] for p in scan])
            ys = np.array([p[1] for p in scan])
            dists = np.hypot(xs, ys)

            # current scan
            scat_scan.set_offsets(np.c_[xs, ys])
            scat_scan.set_array(dists)

            # global map
            gmap.add_scan(scan)
            img_map.set_data(gmap.grid)

            status_txt.set_text(
                f"scan #{reader.scan_count}  |  {len(scan)} points  |  "
                f"map cells: {int((gmap.grid > 0).sum())}"
            )

        return scat_scan, img_map, status_txt

    ani = animation.FuncAnimation(
        fig, update,
        interval=refresh_interval_ms,
        blit=True,
        cache_frame_data=False,
    )

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
