"""
Procedural MuJoCo world generator.

Generates a random arena with box obstacles and returns a MuJoCo XML string.
The robot spawns at the origin (0, 0) — a circular clear zone around it is
always kept free so it doesn't start inside a wall.

Usage
-----
    from world_gen import generate_world_xml
    import mujoco

    xml   = generate_world_xml(num_obstacles=20, seed=42)
    model = mujoco.MjModel.from_xml_string(xml)
"""

import numpy as np


def generate_world_xml(
    num_obstacles: int = 20,
    world_size: float = 10.0,     # half-extent: world spans [-size, +size]
    spawn_clear_r: float = 1.5,   # radius around origin kept obstacle-free
    seed: int = None,
) -> str:
    """
    Generate a randomised MuJoCo world XML string.

    Parameters
    ----------
    num_obstacles : int
        How many random box obstacles to place.
    world_size : float
        Half-width of the arena in metres.  Robot can see up to 8 m, so 10 m
        gives walls just outside max LiDAR range.
    spawn_clear_r : float
        Radius around (0,0) kept free of obstacles — robot spawn zone.
    seed : int or None
        RNG seed for reproducibility.  None = different world every run.
    """
    rng = np.random.default_rng(seed)

    obstacles_xml = _random_obstacles(rng, num_obstacles, world_size, spawn_clear_r)

    wall_t = 0.2          # wall thickness
    wall_h = 0.8          # wall height (taller than LiDAR site at 0.18 m)
    s = world_size

    xml = f"""
<mujoco model="random_world">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <map znear="0.05" zfar="100"/>
    <rgba contactpoint="0 0 0 0" contactforce="0 0 0 0"/>
  </visual>

  <default>
    <joint damping="1" armature="0.01"/>
    <geom friction="1.0 0.1 0.1" density="1000"/>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512"
             rgb1="0.25 0.25 0.25" rgb2="0.35 0.35 0.35"/>
    <material name="grid_mat" texture="grid" texrepeat="20 20" texuniform="true"/>
    <material name="wall_mat"     rgba="0.55 0.55 0.60 1"/>
    <material name="obstacle_mat" rgba="0.75 0.40 0.15 1"/>
  </asset>

  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" size="{s+2} {s+2} 0.1" material="grid_mat"/>

    <!-- Boundary walls (4 sides) -->
    <body name="wall_north" pos="0  {s} {wall_h}">
      <geom type="box" size="{s} {wall_t} {wall_h}" material="wall_mat"/>
    </body>
    <body name="wall_south" pos="0 -{s} {wall_h}">
      <geom type="box" size="{s} {wall_t} {wall_h}" material="wall_mat"/>
    </body>
    <body name="wall_east"  pos=" {s} 0 {wall_h}">
      <geom type="box" size="{wall_t} {s} {wall_h}" material="wall_mat"/>
    </body>
    <body name="wall_west"  pos="-{s} 0 {wall_h}">
      <geom type="box" size="{wall_t} {s} {wall_h}" material="wall_mat"/>
    </body>

    <!-- Random obstacles -->
{obstacles_xml}

    <!-- ===== Mobile base ===== -->
    <body name="base" pos="0 0 0.05">
      <joint name="root" type="free"/>
      <geom name="chassis" type="box" size="0.2 0.15 0.04" pos="0 0 0.05"
            rgba="0.2 0.2 0.8 1"/>

      <site name="imu_site"   pos="0 0 0.05" size="0.02"/>

      <geom type="cylinder" size="0.02 0.04" pos="0 0 0.13"
            rgba="0.1 0.9 0.1 1" density="500"/>
      <site name="lidar_site" pos="0 0 0.18" size="0.03"/>

      <body name="wheel_fl" pos=" 0.15  0.17 0">
        <joint name="joint_fl" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_fr" pos=" 0.15 -0.17 0">
        <joint name="joint_fr" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_rl" pos="-0.15  0.17 0">
        <joint name="joint_rl" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_rr" pos="-0.15 -0.17 0">
        <joint name="joint_rr" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>
  </worldbody>

  <sensor>
    <accelerometer name="imu_accel" site="imu_site"/>
    <gyro          name="imu_gyro"  site="imu_site"/>
  </sensor>

  <actuator>
    <motor name="motor_fl" joint="joint_fl" gear="50" ctrlrange="-1 1"/>
    <motor name="motor_fr" joint="joint_fr" gear="50" ctrlrange="-1 1"/>
    <motor name="motor_rl" joint="joint_rl" gear="50" ctrlrange="-1 1"/>
    <motor name="motor_rr" joint="joint_rr" gear="50" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
    return xml


def _random_obstacles(rng, n, world_size, clear_r):
    """Generate XML for n random box obstacles, avoiding the spawn zone."""
    lines = []
    placed = 0
    attempts = 0
    max_attempts = n * 20

    while placed < n and attempts < max_attempts:
        attempts += 1

        x    = rng.uniform(-(world_size - 1.0),  world_size - 1.0)
        y    = rng.uniform(-(world_size - 1.0),  world_size - 1.0)
        yaw  = rng.uniform(0, 90)          # degrees, random orientation

        # Skip if inside the spawn clear zone
        if np.hypot(x, y) < clear_r:
            continue

        sx = rng.uniform(0.15, 0.8)        # half-size x
        sy = rng.uniform(0.15, 0.8)        # half-size y
        sz = rng.uniform(0.3,  1.2)        # half-size z (height)

        lines.append(
            f'    <body name="obs_{placed}" pos="{x:.3f} {y:.3f} {sz:.3f}" '
            f'euler="0 0 {yaw:.1f}">\n'
            f'      <geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" '
            f'material="obstacle_mat"/>\n'
            f'    </body>'
        )
        placed += 1

    return "\n".join(lines)
