"""
Random world XML generator.

Parses vehicle/model.xml, removes fixed walls, injects random box obstacles
and boundary walls.  The robot definition is never touched.
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

_FIXED_WALL_NAMES = {"barrier", "wall_left", "wall_right"}

_VEHICLE_MODEL = os.path.join(os.path.dirname(__file__), "..", "vehicle", "model.xml")


def generate_world_xml(
    num_obstacles: int = 20,
    world_size: float = 10.0,
    spawn_clear_r: float = 0.5,
    seed: int = None,
) -> str:
    rng  = np.random.default_rng(seed)
    tree = ET.parse(os.path.abspath(_VEHICLE_MODEL))
    root = tree.getroot()

    # Add obstacle material if missing
    asset = root.find("asset")
    if asset is not None:
        existing = {m.get("name") for m in asset.findall("material")}
        if "obstacle_mat" not in existing:
            ET.SubElement(asset, "material", name="obstacle_mat",
                          rgba="0.75 0.40 0.15 1")

    # Strip fixed walls from worldbody, keep ground + robot
    worldbody = root.find("worldbody")
    for body in worldbody.findall("body"):
        if body.get("name") in _FIXED_WALL_NAMES:
            worldbody.remove(body)

    # Boundary walls
    s, wt, wh = world_size, 0.2, 0.8
    _add_box(worldbody, "wall_north", ( 0,  s, wh), (s,  wt, wh), "wall_mat")
    _add_box(worldbody, "wall_south", ( 0, -s, wh), (s,  wt, wh), "wall_mat")
    _add_box(worldbody, "wall_east",  ( s,  0, wh), (wt, s,  wh), "wall_mat")
    _add_box(worldbody, "wall_west",  (-s,  0, wh), (wt, s,  wh), "wall_mat")

    # Random obstacles
    placed = attempts = 0
    while placed < num_obstacles and attempts < num_obstacles * 20:
        attempts += 1
        x, y = rng.uniform(-(s - 1.0), s - 1.0), rng.uniform(-(s - 1.0), s - 1.0)
        if np.hypot(x, y) < spawn_clear_r:
            continue
        sx, sy, sz = rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6), rng.uniform(0.15, 0.8)
        yaw = rng.uniform(0, 90)
        body = ET.SubElement(worldbody, "body",
                             name=f"obs_{placed}",
                             pos=f"{x:.3f} {y:.3f} {sz:.3f}",
                             euler=f"0 0 {yaw:.1f}")
        ET.SubElement(body, "geom", type="box",
                      size=f"{sx:.3f} {sy:.3f} {sz:.3f}",
                      material="obstacle_mat")
        placed += 1

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _add_box(parent, name, pos, size, material):
    body = ET.SubElement(parent, "body", name=name,
                         pos=f"{pos[0]} {pos[1]} {pos[2]}")
    ET.SubElement(body, "geom", type="box",
                  size=f"{size[0]} {size[1]} {size[2]}",
                  material=material)
