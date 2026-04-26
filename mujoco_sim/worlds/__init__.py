"""
World loaders — returns (MjModel, MjData) ready to simulate.

    from worlds import load_fixed, load_random

    model, data = load_fixed()
    model, data = load_random(num_obstacles=20, seed=42)
"""

import os
import mujoco
from .random_gen import generate_world_xml

VEHICLE_MODEL = os.path.join(os.path.dirname(__file__), "..", "vehicle", "model.xml")


def load_fixed():
    """Load the fixed world from vehicle/model.xml."""
    model = mujoco.MjModel.from_xml_path(os.path.abspath(VEHICLE_MODEL))
    return model, mujoco.MjData(model)


def load_random(num_obstacles: int = 20, seed: int = None):
    """Generate a random world built on top of vehicle/model.xml."""
    xml   = generate_world_xml(num_obstacles=num_obstacles, seed=seed)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)
