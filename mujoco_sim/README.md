# MuJoCo Mobile Base Simulation

Simulated 4-wheeled differential-drive robot with a 2D lidar and IMU. No hardware needed — everything runs in MuJoCo.

## Quick Start

```bash
# Auto-drive mode: robot drives a preset path, builds a 2D lidar map
python mujoco_sim/sim.py

# Interactive viewer: drive the robot manually with WASD
mjpython mujoco_sim/sim.py --viewer
```

> **macOS note:** The interactive viewer requires `mjpython` (ships with the `mujoco` pip package) instead of `python`. This is a macOS graphics threading requirement.

## Controls (Viewer Mode)

| Key | Action |
|-----|--------|
| UP    | Toggle forward |
| DOWN  | Toggle backward |
| LEFT  | Toggle turn left |
| RIGHT | Toggle turn right |
| SPACE | Stop all movement |

Controls are **toggles**: press a key once to start, press the **same key again** to stop. For example, press LEFT to start turning left — the robot keeps turning until you press LEFT again. Pressing the opposite direction (e.g. RIGHT while turning left) automatically cancels the current direction. SPACE stops all movement immediately.

The camera automatically tracks the robot. You can still orbit and zoom with the mouse.

## What's in the Simulation

### Robot (`model.xml`)

- **Chassis**: box body with a free joint (6-DOF)
- **4 cylinder wheels**: hinge joints around the Y axis, driven by motor actuators with `gear=50`
- **Lidar**: a site on top of the chassis; rays are cast in `sim.py` using `mj_ray` (not a MuJoCo sensor)
- **IMU**: MuJoCo `accelerometer` + `gyro` sensors attached to the chassis

### Environment

- 20m x 20m ground plane
- Barrier wall at x=3 (directly in front of spawn)
- Side walls at y=+4 and y=-4

### Sensors

**Lidar** (simulated in Python):
- 180 rays, 360-degree sweep in the XY plane
- 8m max range
- 10 Hz sweep rate
- Self-hits excluded (rays ignore the robot's own body)

**IMU** (MuJoCo sensors):
- 3-axis accelerometer (body frame)
- 3-axis gyroscope (body frame)
- Sampled every physics step (500 Hz at dt=0.002)

## Auto-Drive Output

Running `python mujoco_sim/sim.py` drives the robot on a preset path and saves a 2D lidar map to `mujoco_sim/map_output.png`. The map shows:

- **Green dots**: lidar hit points (walls, barriers)
- **Blue line**: robot trajectory
- **Green/red circles**: start/end positions

## Files

| File | Description |
|------|-------------|
| `model.xml` | MuJoCo MJCF model (robot + environment) |
| `sim.py` | Simulation script (lidar raycasting, IMU readout, driving, mapping) |
| `map_output.png` | Generated 2D lidar map (auto-drive mode) |

## Dependencies

- `mujoco` (includes `mjpython`)
- `numpy`
- `matplotlib`
