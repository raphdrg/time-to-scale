"""
MuJoCo simulation: 4-wheeled mobile base with simulated LiDAR + IMU.

Usage
-----
Auto-drive (headless, scripted path):
    python mujoco_sim/sim.py
    python mujoco_sim/sim.py --random          # random world
    python mujoco_sim/sim.py --random --seed 7 # reproducible random world

Interactive viewer + live mapping (requires mjpython on macOS):
    mjpython mujoco_sim/sim.py --viewer
    mjpython mujoco_sim/sim.py --viewer --random
    mjpython mujoco_sim/sim.py --viewer --random --seed 42

In viewer mode: drive with arrow keys, close the window → map_output.png is saved.
"""

import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(__file__))
from sensors   import IMUSensor, LiDARSensor
from mapper    import OccupancyGrid
from world_gen import generate_world_xml


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "model.xml")
MAP_SAVE_PATH = os.path.join(os.path.dirname(__file__), "map_output.png")

SIM_DURATION  = 10.0   # seconds (auto mode only)
LIDAR_RATE_HZ = 10     # LiDAR sweeps per second

DRIVE_CTRL     = 0.4   # forward/back throttle  [-1, 1]
TURN_CTRL      = 0.3   # turn throttle
VIEWER_FPS     = 60
IMPULSE_FRAMES = 15    # frames a key press stays active  (~0.25 s at 60 fps)


# ── Shared sensor config ──────────────────────────────────────────────────────

def _make_sensors(model):
    """Instantiate IMU and LiDAR with consistent settings."""
    imu = IMUSensor(
        model,
        noise_accel=0.02,
        noise_gyro=0.002,
        bias_accel_drift=1e-4,
        bias_gyro_drift=1e-5,
    )
    lidar = LiDARSensor(
        model,
        num_rays=360,
        max_range=8.0,
        min_range=0.05,
        noise_std=0.01,
        elevation_angles=[0.0],   # 2D horizontal scan
    )
    return imu, lidar


def _make_map():
    """Create a fresh occupancy grid sized for the default world."""
    return OccupancyGrid(width_m=22.0, height_m=22.0, resolution=0.05)


# ── Drive helpers ─────────────────────────────────────────────────────────────

def auto_drive_policy(t: float):
    """Fixed scripted drive pattern.  Returns (forward, turn) in [-1, 1]."""
    if   t < 2.0: return DRIVE_CTRL, 0.0
    elif t < 3.5: return 0.0,        TURN_CTRL
    elif t < 5.5: return DRIVE_CTRL, 0.0
    elif t < 7.0: return 0.0,        TURN_CTRL
    elif t < 9.0: return DRIVE_CTRL, 0.0
    else:         return 0.0,        0.0


def set_drive(data, forward: float, turn: float):
    """Differential drive — ctrl order: FL, FR, RL, RR."""
    left  = np.clip(forward + turn, -1, 1)
    right = np.clip(forward - turn, -1, 1)
    data.ctrl[0] = left;  data.ctrl[1] = right
    data.ctrl[2] = left;  data.ctrl[3] = right


# ── Auto run ──────────────────────────────────────────────────────────────────

def run_auto(model, data):
    """
    Headless run: scripted drive, full sensor logging, saves map at the end.
    """
    dt = model.opt.timestep
    imu, lidar = _make_sensors(model)
    occ_map     = _make_map()

    lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
    lidar_period  = 1.0 / LIDAR_RATE_HZ
    next_lidar_t  = 0.0

    imu_log    = []
    lidar_log  = []
    trajectory = []

    print(f"Auto-drive for {SIM_DURATION}s  |  "
          f"physics {1/dt:.0f} Hz  |  LiDAR {LIDAR_RATE_HZ} Hz  |  "
          f"map {occ_map.cols}×{occ_map.rows} @ {occ_map.res}m/cell")

    while data.time < SIM_DURATION:
        set_drive(data, *auto_drive_policy(data.time))
        mujoco.mj_step(model, data)

        imu_log.append(imu.read(data, dt))
        trajectory.append(data.site_xpos[lidar_site_id][:2].copy())

        if data.time >= next_lidar_t:
            scan = lidar.sweep(model, data)
            lidar_log.append(scan)
            occ_map.update(scan)
            next_lidar_t += lidar_period

    total_hits = sum(s.num_hits for s in lidar_log)
    print(f"Done — {len(imu_log)} IMU samples  |  "
          f"{len(lidar_log)} LiDAR sweeps  |  {total_hits} hits")

    occ_map.save(MAP_SAVE_PATH, trajectory=np.array(trajectory))
    return imu_log, lidar_log, occ_map


# ── Viewer run ────────────────────────────────────────────────────────────────

def run_viewer(model, data):
    """
    Interactive viewer with live mapping.

    - Drive the robot with WASD
    - LiDAR sweeps and occupancy grid update run in the background every frame
    - Close the window → map_output.png is saved automatically

    On macOS this must be launched with mjpython, not python.
    """
    imu, lidar = _make_sensors(model)
    occ_map     = _make_map()

    lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    dt            = model.opt.timestep
    lidar_period  = 1.0 / LIDAR_RATE_HZ
    next_lidar_t  = 0.0

    trajectory  = []
    imu_log     = []
    sweep_count = [0]

    impulse = {"fwd": 0, "turn": 0}

    def key_callback(keycode):
        if   keycode == 265: impulse["fwd"]  =  IMPULSE_FRAMES  # ↑ forward
        elif keycode == 264: impulse["fwd"]  = -IMPULSE_FRAMES  # ↓ back
        elif keycode == 263: impulse["turn"] =  IMPULSE_FRAMES  # ← turn left
        elif keycode == 262: impulse["turn"] = -IMPULSE_FRAMES  # → turn right

    print("Viewer ready — arrow keys to drive (tap).  Close window to save map.")
    print("  ↑/↓ = forward/back    ←/→ = turn left/right")

    steps_per_frame = max(1, int(1.0 / (dt * VIEWER_FPS)))

    _SUPPRESS = [
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_INERTIA,
        mujoco.mjtVisFlag.mjVIS_COM,
        mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
        mujoco.mjtVisFlag.mjVIS_PERTFORCE,
        mujoco.mjtVisFlag.mjVIS_PERTOBJ,
    ]

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            # Camera tracks the robot
            viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance    = 5.0
            viewer.cam.elevation   = -35
            viewer.cam.azimuth     = 180
            viewer.cam.lookat[2]   = 0.2

            while viewer.is_running():
                t0 = time.time()

                for flag in _SUPPRESS:
                    viewer.opt.flags[flag] = False

                # --- Drive ---
                fwd = turn = 0.0
                if   impulse["fwd"]  > 0: fwd  =  DRIVE_CTRL; impulse["fwd"]  -= 1
                elif impulse["fwd"]  < 0: fwd  = -DRIVE_CTRL; impulse["fwd"]  += 1
                if   impulse["turn"] > 0: turn =  TURN_CTRL;  impulse["turn"] -= 1
                elif impulse["turn"] < 0: turn = -TURN_CTRL;  impulse["turn"] += 1

                set_drive(data, fwd, turn)

                # --- Step physics ---
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)

                # --- IMU (every frame) ---
                imu_log.append(imu.read(data, dt * steps_per_frame))

                # --- LiDAR + mapping (at LIDAR_RATE_HZ) ---
                if data.time >= next_lidar_t:
                    scan = lidar.sweep(model, data)
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
            print("\nOn macOS run with mjpython:  mjpython mujoco_sim/sim.py --viewer [--random]")
            sys.exit(1)
        raise

    # --- Viewer closed: save map ---
    print(f"\nViewer closed — {sweep_count[0]} LiDAR sweeps recorded.")
    if sweep_count[0] > 0:
        occ_map.save(MAP_SAVE_PATH, trajectory=np.array(trajectory))
    else:
        print("No sweeps recorded — map not saved.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR+IMU simulation")
    parser.add_argument(
        "--viewer", action="store_true",
        help="Interactive WASD viewer with live mapping (requires mjpython on macOS)"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Generate a random world instead of using the fixed model.xml"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for the random world (omit for a different world each run)"
    )
    parser.add_argument(
        "--obstacles", type=int, default=20,
        help="Number of random obstacles (default: 20)"
    )
    args = parser.parse_args()

    # --- Build model ---
    if args.random:
        print(f"Generating random world  "
              f"(obstacles={args.obstacles}, seed={args.seed})")
        xml   = generate_world_xml(
            num_obstacles=args.obstacles,
            seed=args.seed,
        )
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
