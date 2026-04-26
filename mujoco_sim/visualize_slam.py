"""
visualize_slam.py — Run with: python3 mujoco_sim/visualize_slam.py
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

with open("mujoco_sim/slam_log.pkl", "rb") as f:
    log = pickle.load(f)

print(f"Loaded {len(log)} frames")
print(f"Keys in each frame: {log[0].keys()}")
print(f"First pose: {log[0]['pose']}")
print(f"Last pose:  {log[-1]['pose']}")
print(f"Landmarks at end: {len(log[-1]['landmarks'])}")

# ── Pull out all poses ──
poses = np.array([f["pose"] for f in log])

# ── Pull out all landmark positions (use final frame) ──
final_lms = log[-1]["landmarks"]
lm_arr = np.array(final_lms) if final_lms else np.empty((0, 2))

# ── Compute axis limits from poses only (ignore outlier landmarks) ──
pad = 1.0
x_min, x_max = poses[:, 0].min() - pad, poses[:, 0].max() + pad
y_min, y_max = poses[:, 1].min() - pad, poses[:, 1].max() + pad

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#111")

ax_map   = axes[0]
ax_stats = axes[1]

for ax in axes:
    ax.set_facecolor("#111")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

def draw_ellipse(ax, x, y, cov2x2, color):
    try:
        vals, vecs = np.linalg.eigh(cov2x2)
        vals = np.maximum(vals, 1e-9)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * 2.0 * np.sqrt(vals)
        e = Ellipse((x, y), w, h, angle=angle, color=color, alpha=0.25, zorder=4)
        ax.add_patch(e)
    except Exception:
        pass

# ── Uncertainty over time ──
uncertainty = []
n_lms_over_time = []
times = []
for i, f in enumerate(log):
    cov = f["covariance"]
    uncertainty.append(float(np.trace(cov[:2, :2])))
    n_lms_over_time.append(len(f["landmarks"]))
    times.append(i * 0.1)

def animate(frame):
    ax_map.cla()
    ax_stats.cla()

    for ax in [ax_map, ax_stats]:
        ax.set_facecolor("#111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Map panel ──
    ax_map.set_title(f"SLAM Map   t={frame * 0.1:.1f}s", color="white", fontsize=12)
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.1, color="white")

    # Path so far
    ax_map.plot(poses[:frame+1, 0], poses[:frame+1, 1],
                color="dodgerblue", linewidth=2, label="estimated path", zorder=3)
    ax_map.plot(poses[0, 0], poses[0, 1], "go", markersize=10, label="start", zorder=5)

    # Current pose
    px, py, pth = log[frame]["pose"]
    ax_map.plot(px, py, "wo", markersize=7, zorder=6)
    ax_map.annotate("", xy=(px + 0.25*np.cos(pth), py + 0.25*np.sin(pth)),
                    xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", color="cyan", lw=2))

    # Covariance ellipse
    cov = log[frame]["covariance"]
    if cov.shape[0] >= 2:
        draw_ellipse(ax_map, px, py, cov[:2, :2], "cyan")

    # Landmarks visible so far
    lms = log[frame]["landmarks"]
    if lms:
        la = np.array(lms)
        # only show landmarks within axis bounds
        mask = ((la[:, 0] > x_min) & (la[:, 0] < x_max) &
                (la[:, 1] > y_min) & (la[:, 1] < y_max))
        if mask.any():
            ax_map.scatter(la[mask, 0], la[mask, 1],
                           c="yellow", s=30, zorder=5, label=f"landmarks ({mask.sum()})")

    ax_map.legend(loc="upper right", facecolor="#222", labelcolor="white", fontsize=8)

    # ── Stats panel ──
    ax_stats.set_title("Pose Uncertainty Over Time", color="white", fontsize=12)
    ax_stats.set_xlabel("time (s)")
    ax_stats.set_ylabel("trace(cov_xy)  [pose uncertainty]", color="cyan")
    ax_stats.tick_params(axis="y", labelcolor="cyan")
    ax_stats.plot(times[:frame+1], uncertainty[:frame+1], color="cyan", linewidth=1.5)
    ax_stats.grid(True, alpha=0.1, color="white")

    ax2 = ax_stats.twinx()
    ax2.set_facecolor("#111")
    ax2.set_ylabel("# landmarks", color="yellow")
    ax2.tick_params(axis="y", labelcolor="yellow")
    ax2.plot(times[:frame+1], n_lms_over_time[:frame+1],
             color="yellow", linewidth=1.5, linestyle="--")

ani = FuncAnimation(fig, animate, frames=len(log), interval=80, repeat=False)
plt.tight_layout()
plt.show()