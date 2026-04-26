"""
MuJoCo simulation: 4-wheeled mobile base with simulated lidar + IMU.

Usage:
    python mujoco_sim/sim.py              # auto-drive, shows matplotlib map at end
    mjpython mujoco_sim/sim.py --viewer   # interactive viewer with WASD driving (macOS needs mjpython)
"""

import argparse
import math
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt


# ---------- config ----------
MODEL_PATH = "mujoco_sim/model.xml"
SIM_DURATION = 10.0
LIDAR_RANGE = 8.0
LIDAR_NUM_RAYS = 180
LIDAR_RATE_HZ = 10
MAP_SAVE_PATH = "mujoco_sim/map_output.png"

# Drive params (ctrl is torque via motor, range [-1, 1])
DRIVE_CTRL = 0.4        # forward/back throttle
TURN_CTRL = 0.3         # turn throttle


def build_lidar_directions(num_rays):
    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles), np.zeros(num_rays)], axis=1)
    return dirs


def do_lidar_sweep(model, data, site_id, directions, base_body_id):
    lidar_pos = data.site_xpos[site_id].copy()
    lidar_rot = data.site_xmat[site_id].reshape(3, 3)

    hits = []
    for d_local in directions:
        d_world = lidar_rot @ d_local
        geom_id = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            model, data, lidar_pos, d_world,
            None, 1, base_body_id, geom_id,
        )
        if 0 < dist < LIDAR_RANGE:
            hit = lidar_pos + d_world * dist
            hits.append(hit[:2])
    return np.array(hits) if hits else np.empty((0, 2))


def read_imu(data, accel_adr, gyro_adr):
    accel = data.sensordata[accel_adr:accel_adr + 3].copy()
    gyro = data.sensordata[gyro_adr:gyro_adr + 3].copy()
    return accel, gyro


def set_drive(data, forward, turn):
    """Set motor controls. forward/turn in [-1, 1].
    Left pair = forward + turn, right pair = forward - turn."""
    left = np.clip(forward + turn, -1, 1)
    right = np.clip(forward - turn, -1, 1)
    data.ctrl[0] = left    # FL
    data.ctrl[1] = right   # FR
    data.ctrl[2] = left    # RL
    data.ctrl[3] = right   # RR


# ── Auto-drive ──────────────────────────────────

def auto_drive_policy(t):
    """Returns (forward, turn) each in [-1, 1]."""
    if t < 2.0:
        return DRIVE_CTRL, 0.0
    elif t < 3.5:
        return 0.0, TURN_CTRL
    elif t < 5.5:
        return DRIVE_CTRL, 0.0
    elif t < 7.0:
        return 0.0, TURN_CTRL
    elif t < 9.0:
        return DRIVE_CTRL, 0.0
    else:
        return 0.0, 0.0


def run_auto(model, data):
    accel_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")]
    gyro_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
    lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    lidar_dirs = build_lidar_directions(LIDAR_NUM_RAYS)
    lidar_period = 1.0 / LIDAR_RATE_HZ
    next_lidar_time = 0.0

    all_hits = []
    trajectory = []
    imu_log = []

    print(f"Running auto-drive for {SIM_DURATION}s ...")

    while data.time < SIM_DURATION:
        fwd, turn = auto_drive_policy(data.time)
        set_drive(data, fwd, turn)
        mujoco.mj_step(model, data)

        accel, gyro = read_imu(data, accel_adr, gyro_adr)
        imu_log.append({"t": data.time, "accel": accel, "gyro": gyro})

        pos = data.site_xpos[lidar_site_id][:2].copy()
        trajectory.append(pos)

        if data.time >= next_lidar_time:
            hits = do_lidar_sweep(model, data, lidar_site_id, lidar_dirs, base_body_id)
            if len(hits) > 0:
                all_hits.append(hits)
            next_lidar_time += lidar_period

    all_hits = np.concatenate(all_hits, axis=0) if all_hits else np.empty((0, 2))

    print(f"Done. {len(all_hits)} lidar points, {len(imu_log)} IMU samples.")
    accels = np.array([s["accel"] for s in imu_log])
    gyros = np.array([s["gyro"] for s in imu_log])
    print(f"Accel mean: {accels.mean(axis=0).round(3)}")
    print(f"Gyro  mean: {gyros.mean(axis=0).round(3)}")

    plot_map(all_hits, trajectory, MAP_SAVE_PATH)


# ── Viewer mode ─────────────────────────────────

VIEWER_FPS = 60


def run_viewer(model, data):
    # Toggle state: press to start, press again to stop
    active = {"fwd": False, "back": False, "left": False, "right": False}

    def key_callback(keycode):
        # GLFW arrow keys: UP=265 DOWN=264 LEFT=263 RIGHT=262
        if keycode == 265:      # UP — toggle forward
            active["fwd"] = not active["fwd"]
            active["back"] = False   # cancel opposite
        elif keycode == 264:    # DOWN — toggle backward
            active["back"] = not active["back"]
            active["fwd"] = False
        elif keycode == 263:    # LEFT — toggle turn left
            active["left"] = not active["left"]
            active["right"] = False
        elif keycode == 262:    # RIGHT — toggle turn right
            active["right"] = not active["right"]
            active["left"] = False
        elif keycode == 32:     # SPACE — stop everything
            active["fwd"] = active["back"] = active["left"] = active["right"] = False

    print("Arrow keys to drive (toggle on/off). SPACE = stop all.")
    print("  UP/DOWN = forward/back    LEFT/RIGHT = turn")

    steps_per_frame = int(1.0 / (model.opt.timestep * VIEWER_FPS))
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            # Set up camera to track the robot
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 180
            viewer.cam.lookat[2] = 0.2  # look slightly above ground

            while viewer.is_running():
                t0 = time.time()

                fwd = 0.0
                turn = 0.0
                if active["fwd"]:   fwd += DRIVE_CTRL
                if active["back"]:  fwd -= DRIVE_CTRL
                if active["left"]:  turn -= TURN_CTRL
                if active["right"]: turn += TURN_CTRL

                set_drive(data, fwd, turn)

                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)

                viewer.sync()

                dt = time.time() - t0
                remaining = 1.0 / VIEWER_FPS - dt
                if remaining > 0:
                    time.sleep(remaining)

    except RuntimeError as e:
        if "mjpython" in str(e):
            print("\nOn macOS run:  mjpython mujoco_sim/sim.py --viewer")
            sys.exit(1)
        raise


# ── Plot ────────────────────────────────────────

def plot_map(all_hits, trajectory, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    if len(all_hits) > 0:
        ax.scatter(all_hits[:, 0], all_hits[:, 1], s=0.3, c="lime", alpha=0.6, label="lidar hits")
    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5, label="robot path")
    ax.plot(traj[0, 0], traj[0, 1], "go", markersize=8, label="start")
    ax.plot(traj[-1, 0], traj[-1, 1], "ro", markersize=8, label="end")
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("2D lidar map from MuJoCo simulation")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Map saved -> {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if args.viewer:
        run_viewer(model, data)
    else:
        run_auto(model, data)


if __name__ == "__main__":
    main()
