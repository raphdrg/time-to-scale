# mujoco_sim

MuJoCo-based robot simulation with simulated LiDAR and IMU, and occupancy grid mapping.

## Setup

```bash
uv venv --python /opt/homebrew/bin/python3.13 --clear
source .venv/bin/activate
uv pip install mujoco matplotlib numpy
```

## Running

All commands are run from the project root (`time-to-scale/`).

### Headless auto-drive

The robot drives a scripted path and saves a map at the end.

```bash
python mujoco_sim/main.py
python mujoco_sim/main.py --random
python mujoco_sim/main.py --random --seed 42
python mujoco_sim/main.py --random --obstacles 30
```

### Interactive viewer + live mapping

Drive the robot yourself. The map is saved when you close the window.
**macOS requires `mjpython`** (bundled with the mujoco package).

```bash
mjpython mujoco_sim/main.py --viewer
mjpython mujoco_sim/main.py --viewer --random
mjpython mujoco_sim/main.py --viewer --random --seed 42
```

### Controls (viewer mode)

| Key | 1st press | 2nd press | 3rd press |
|-----|-----------|-----------|-----------|
| `↑` | Forward (normal) | Forward (fast) | Stop |
| `↓` | Reverse (normal) | Reverse (fast) | Stop |
| `←` | Turn left (normal) | Turn left (fast) | Stop |
| `→` | Turn right (normal) | Turn right (fast) | Stop |
| `Space` | Stop all | — | — |

Each arrow key cycles through three states: **normal speed → fast → stop**. To stop turning or moving, just press the same key again until it stops. Pressing the opposite direction (e.g. `↑` while reversing) automatically cancels the current one. `Space` is an emergency stop that kills all movement. Close the window to save the map.

### View the map

```bash
open mujoco_sim/map_output.png
```

**Map legend:** white = free space, black = walls/obstacles, gray = unexplored.

## Project structure

```
mujoco_sim/
├── main.py           Entry point — handles all commands
├── vehicle/
│   └── model.xml     Robot definition (chassis, 4 wheels, LiDAR site, IMU)
├── worlds/
│   ├── __init__.py   load_fixed() and load_random()
│   └── random_gen.py Procedural world generator (reads vehicle/model.xml)
├── sensors/
│   ├── imu.py        IMU: accel + gyro with gaussian noise and bias drift
│   └── lidar.py      LiDAR: 360-ray casting with range noise
└── mapper.py         Occupancy grid: log-odds Bayesian + Bresenham ray carving
```

`vehicle/model.xml` is the single source of truth for the robot.
Both the fixed world and random worlds read from it — change the robot once, all worlds get it.

## Sim → hardware path

`IMUSensor` and `LiDARSensor` return typed dataclasses (`IMUReading`, `LiDARScan`).
`mapper.py` only consumes those dataclasses — no sim-specific code.
To run on hardware: replace the sensor classes with hardware readers that return the same dataclasses. The mapper and any algorithms on top stay unchanged.
