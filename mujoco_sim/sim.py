"""
MuJoCo simulation: 4-wheeled mobile base with simulated lidar + IMU.

Usage
-----
Auto-drive (headless):
    python mujoco_sim/sim.py
    python mujoco_sim/sim.py --random
    python mujoco_sim/sim.py --random --seed 42

Interactive viewer + live mapping (macOS needs mjpython):
    mjpython mujoco_sim/sim.py --viewer
    mjpython mujoco_sim/sim.py --viewer --random
    mjpython mujoco_sim/sim.py --viewer --random --seed 42

Controls: arrow keys to drive (toggle on/off), SPACE = stop all.
Close the window → map_output.png is saved.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from sensors   import IMUSensor, LiDARSensor
from mapper    import OccupancyGrid
from world_gen import generate_world_xml


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = "mujoco_sim/model.xml"
MAP_SAVE_PATH = "mujoco_sim/map_output.png"

SIM_DURATION   = 10.0
LIDAR_RANGE    = 8.0
LIDAR_NUM_RAYS = 180
LIDAR_RATE_HZ  = 10

DRIVE_CTRL = 0.4
TURN_CTRL  = 0.3
VIEWER_FPS = 60


# ── Raw lidar helpers (used by run_auto) ──────────────────────────────────────

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
        dist = mujoco.mj_ray(model, data, lidar_pos, d_world,
                             None, 1, base_body_id, geom_id)
        if 0 < dist < LIDAR_RANGE:
            hit = lidar_pos + d_world * dist
            hits.append(hit[:2])
    return np.array(hits) if hits else np.empty((0, 2))


def read_imu(data, accel_adr, gyro_adr):
    accel = data.sensordata[accel_adr:accel_adr + 3].copy()
    gyro  = data.sensordata[gyro_adr:gyro_adr + 3].copy()
    return accel, gyro


def set_drive(data, forward, turn):
    left  = np.clip(forward + turn, -1, 1)
    right = np.clip(forward - turn, -1, 1)
    data.ctrl[0] = left;  data.ctrl[1] = right
    data.ctrl[2] = left;  data.ctrl[3] = right


# ── Auto-drive policy ─────────────────────────────────────────────────────────

def auto_drive_policy(t):
    if   t < 2.0: return DRIVE_CTRL, 0.0
    elif t < 3.5: return 0.0,        TURN_CTRL
    elif t < 5.5: return DRIVE_CTRL, 0.0
    elif t < 7.0: return 0.0,        TURN_CTRL
    elif t < 9.0: return DRIVE_CTRL, 0.0
    else:         return 0.0,        0.0


# ── Auto run ──────────────────────────────────────────────────────────────────

def run_auto(model, data):
    accel_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")]
    gyro_adr  = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
    lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    lidar_dirs    = build_lidar_directions(LIDAR_NUM_RAYS)
    lidar_period  = 1.0 / LIDAR_RATE_HZ
    next_lidar_t  = 0.0

    occ_map    = OccupancyGrid(width_m=22.0, height_m=22.0, resolution=0.05)
    lidar_sens = LiDARSensor(model, num_rays=360, max_range=LIDAR_RANGE,
                             min_range=0.05, noise_std=0.01,
                             elevation_angles=[0.0])

    all_hits   = []
    trajectory = []
    imu_log    = []

    print(f"Running auto-drive for {SIM_DURATION}s ...")

    while data.time < SIM_DURATION:
        fwd, turn = auto_drive_policy(data.time)
        set_drive(data, fwd, turn)
        mujoco.mj_step(model, data)

        accel, gyro = read_imu(data, accel_adr, gyro_adr)
        imu_log.append({"t": data.time, "accel": accel, "gyro": gyro})

        trajectory.append(data.site_xpos[lidar_site_id][:2].copy())

        if data.time >= next_lidar_t:
            hits = do_lidar_sweep(model, data, lidar_site_id, lidar_dirs, base_body_id)
            if len(hits) > 0:
                all_hits.append(hits)

            scan = lidar_sens.sweep(model, data)
            occ_map.update(scan)

            next_lidar_t += lidar_period

    all_hits = np.concatenate(all_hits, axis=0) if all_hits else np.empty((0, 2))

    print(f"Done. {len(all_hits)} lidar points, {len(imu_log)} IMU samples.")
    accels = np.array([s["accel"] for s in imu_log])
    gyros  = np.array([s["gyro"]  for s in imu_log])
    print(f"Accel mean: {accels.mean(axis=0).round(3)}")
    print(f"Gyro  mean: {gyros.mean(axis=0).round(3)}")

    occ_map.save(MAP_SAVE_PATH, trajectory=np.array(trajectory))


# ── Viewer run (live mapping) ─────────────────────────────────────────────────

def run_viewer(model, data):
    """
    Interactive viewer with live occupancy grid mapping.
    Arrow keys toggle drive on/off. SPACE stops everything.
    Close the window → map_output.png saved automatically.
    """
    lidar_sens    = LiDARSensor(model, num_rays=360, max_range=LIDAR_RANGE,
                                min_range=0.05, noise_std=0.01,
                                elevation_angles=[0.0])
    occ_map       = OccupancyGrid(width_m=22.0, height_m=22.0, resolution=0.05)
    lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
    lidar_period  = 1.0 / LIDAR_RATE_HZ
    next_lidar_t  = 0.0
    sweep_count   = [0]
    trajectory    = []

    # Speed level per direction: 0=off, 1=normal, 2=fast
    speed = {"fwd": 0, "back": 0, "left": 0, "right": 0}

    def key_callback(keycode):
        if keycode == 265:      # UP
            speed["back"] = 0
            speed["fwd"] = (speed["fwd"] + 1) % 3   # 0→1→2→0
        elif keycode == 264:    # DOWN
            speed["fwd"] = 0
            speed["back"] = (speed["back"] + 1) % 3
        elif keycode == 263:    # LEFT
            speed["right"] = 0
            speed["left"] = (speed["left"] + 1) % 3
        elif keycode == 262:    # RIGHT
            speed["left"] = 0
            speed["right"] = (speed["right"] + 1) % 3
        elif keycode == 32:     # SPACE — stop everything
            for k in speed: speed[k] = 0

    print("Arrow keys to drive. Press once = normal, twice = fast, third = stop.")
    print("SPACE = stop all.  Close window to save map.")
    print("  ↑/↓ = forward/back    ←/→ = turn left/right")

    steps_per_frame = int(1.0 / (model.opt.timestep * VIEWER_FPS))
    base_body_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance    = 4.0
            viewer.cam.elevation   = -30
            viewer.cam.azimuth     = 180
            viewer.cam.lookat[2]   = 0.2

            while viewer.is_running():
                t0 = time.time()

                fwd = turn = 0.0
                fwd  += DRIVE_CTRL * speed["fwd"]
                fwd  -= DRIVE_CTRL * speed["back"]
                turn -= TURN_CTRL  * speed["left"]
                turn += TURN_CTRL  * speed["right"]

                set_drive(data, fwd, turn)

                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)

                # Live mapping
                if data.time >= next_lidar_t:
                    scan = lidar_sens.sweep(model, data)
                    occ_map.update(scan)
                    trajectory.append(data.site_xpos[lidar_site_id][:2].copy())
                    sweep_count[0] += 1
                    next_lidar_t += lidar_period

                viewer.sync()

                remaining = 1.0 / VIEWER_FPS - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

    except RuntimeError as e:
        if "mjpython" in str(e):
            print("\nOn macOS run:  mjpython mujoco_sim/sim.py --viewer [--random]")
            sys.exit(1)
        raise

    # Save map when viewer closes
    print(f"\nViewer closed — {sweep_count[0]} LiDAR sweeps recorded.")
    if sweep_count[0] > 0:
        occ_map.save(MAP_SAVE_PATH, trajectory=np.array(trajectory))
    else:
        print("No sweeps recorded — map not saved.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer",    action="store_true",
                        help="Interactive viewer with live mapping (needs mjpython on macOS)")
    parser.add_argument("--random",    action="store_true",
                        help="Generate a random world instead of model.xml")
    parser.add_argument("--seed",      type=int, default=None,
                        help="RNG seed for the random world")
    parser.add_argument("--obstacles", type=int, default=20,
                        help="Number of random obstacles (default: 20)")
    args = parser.parse_args()

    if args.random:
        print(f"Generating random world (obstacles={args.obstacles}, seed={args.seed})")
        xml   = generate_world_xml(num_obstacles=args.obstacles, seed=args.seed)
        model = mujoco.MjModel.from_xml_string(xml)
    else:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    data = mujoco.MjData(model)

    if args.viewer:
        run_viewer(model, data)
    else:
        run_auto(model, data)


if __name__ == "__main__":
    main()
