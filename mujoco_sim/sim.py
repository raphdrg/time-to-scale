
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

#DSki Algorithms
from algorithms import SLAM 


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
    Left wheels = forward + turn, right wheels = forward - turn."""
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


def hits_to_observations(hits, robot_pos, robot_theta):
    obs = []
    for hit in hits:
        dx = hit[0] - robot_pos[0]
        dy = hit[1] - robot_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) - robot_theta
        obs.append(np.array([r, bearing]))
    return obs

def run_auto(model, data):

    slam = SLAM({
        "init_pose": [0.0, 0.0, 0.0],
        "motion_noise": [0.05, 0.05, 0.01],
        "obs_noise": [0.1, 0.05],
        "assoc_threshold": 1.0,
    })

    prev_pos = np.array([0.0, 0.0])
    prev_theta = 0.0

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
    slam_log = []  # <-- initialised here, before the loop

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

                robot_pos = data.site_xpos[lidar_site_id][:2].copy()
                quat = data.xquat[base_body_id]
                theta = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                                   1 - 2*(quat[2]**2 + quat[3]**2))

                d = robot_pos - prev_pos
                dx =  np.cos(prev_theta) * d[0] + np.sin(prev_theta) * d[1]
                dy = -np.sin(prev_theta) * d[0] + np.cos(prev_theta) * d[1]
                dtheta = theta - prev_theta

                obs = hits_to_observations(hits[::10], robot_pos, theta)

                result = slam.update(np.array([dx, dy, dtheta]), obs)
                slam_log.append(result)  # <-- collected inside the loop

                print(f"t={data.time:.1f}s | estimated pose: {result['pose'].round(3)}")

                prev_pos = robot_pos
                prev_theta = theta

            next_lidar_time += lidar_period

    all_hits = np.concatenate(all_hits, axis=0) if all_hits else np.empty((0, 2))

    print(f"Done. {len(all_hits)} lidar points, {len(imu_log)} IMU samples.")
    accels = np.array([s["accel"] for s in imu_log])
    gyros = np.array([s["gyro"] for s in imu_log])
    print(f"Accel mean: {accels.mean(axis=0).round(3)}")
    print(f"Gyro  mean: {gyros.mean(axis=0).round(3)}")

    import pickle
    with open("mujoco_sim/slam_log.pkl", "wb") as f:
        pickle.dump(slam_log, f)
    print(f"SLAM log saved. {len(slam_log)} frames.")

    plot_map(all_hits, trajectory, MAP_SAVE_PATH)
# ── Viewer mode ─────────────────────────────────

VIEWER_FPS = 60
IMPULSE_FRAMES = 15  # how many frames each key press drives for (~0.25s)


def run_viewer(model, data):
    # Each key gets a countdown: when > 0 the command is active
    impulse = {"fwd": 0, "turn": 0}

    def key_callback(keycode):
        if keycode == 87:    # W
            impulse["fwd"] = IMPULSE_FRAMES
        elif keycode == 83:  # S
            impulse["fwd"] = -IMPULSE_FRAMES
        elif keycode == 65:  # A — turn left
            impulse["turn"] = -IMPULSE_FRAMES
        elif keycode == 68:  # D — turn right
            impulse["turn"] = IMPULSE_FRAMES

    print("WASD to drive (tap). Close window to exit.")
    print("  W/S = nudge forward/back    A/D = nudge left/right")

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

                # Force all debug visualization off every frame
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False

                fwd = 0.0
                turn = 0.0
                if impulse["fwd"] > 0:
                    fwd = DRIVE_CTRL
                    impulse["fwd"] -= 1
                elif impulse["fwd"] < 0:
                    fwd = -DRIVE_CTRL
                    impulse["fwd"] += 1

                if impulse["turn"] > 0:
                    turn = TURN_CTRL
                    impulse["turn"] -= 1
                elif impulse["turn"] < 0:
                    turn = -TURN_CTRL
                    impulse["turn"] += 1

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
    import matplotlib
    matplotlib.use("Agg")  # must be set before importing pyplot — no GUI, safe on all threads
    import matplotlib.pyplot as plt

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
    # plt.show() removed — not safe inside mjpython; open the saved file instead


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