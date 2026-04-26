"""
IMU sensor simulation.

MuJoCo's <accelerometer> and <gyro> sensors already produce physically correct
readings — specific force (linear accel minus gravity) and angular velocity,
both expressed in the sensor body frame.  This class adds the noise model that
a real MEMS sensor exhibits:

  1. White noise  — independent gaussian noise on every sample.  Scales with
                    the sensor's "noise density" spec (units: m/s²/√Hz or
                    rad/s/√Hz).
  2. Bias         — a DC offset that is non-zero and varies slowly over time.
  3. Bias random walk — the bias itself drifts as a Wiener process.  This is
                    the dominant error source in dead-reckoning over minutes.

Model per axis:
    bias[t]   = bias[t-1] + N(0, bias_drift² · dt)
    output[t] = truth[t]  + bias[t] + N(0, noise²)
"""

import numpy as np
import mujoco
from dataclasses import dataclass


@dataclass
class IMUReading:
    t: float                          # simulation time (s)
    accel: np.ndarray                 # noisy accel, body frame (m/s²)
    gyro: np.ndarray                  # noisy gyro,  body frame (rad/s)
    accel_true: np.ndarray            # ground truth from MuJoCo physics
    gyro_true: np.ndarray             # ground truth from MuJoCo physics
    accel_bias: np.ndarray            # current bias state (for analysis)
    gyro_bias: np.ndarray             # current bias state (for analysis)


class IMUSensor:
    """
    Wraps MuJoCo's built-in accelerometer + gyro sensors and adds a realistic
    noise model.

    Parameters
    ----------
    model : mujoco.MjModel
    noise_accel : float
        1-sigma white noise per sample [m/s²].  Typical MEMS: 0.01–0.1.
    noise_gyro : float
        1-sigma white noise per sample [rad/s].  Typical MEMS: 0.001–0.01.
    bias_accel_drift : float
        Bias random-walk rate [m/s² per √s].  Controls how fast accel bias
        wanders.  Typical: 1e-4 to 1e-3.
    bias_gyro_drift : float
        Bias random-walk rate [rad/s per √s].  Typical: 1e-5 to 1e-4.
    accel_sensor_name : str
        Name of the MuJoCo <accelerometer> sensor in model.xml.
    gyro_sensor_name : str
        Name of the MuJoCo <gyro> sensor in model.xml.
    """

    def __init__(
        self,
        model,
        noise_accel: float = 0.02,
        noise_gyro: float = 0.002,
        bias_accel_drift: float = 1e-4,
        bias_gyro_drift: float = 1e-5,
        accel_sensor_name: str = "imu_accel",
        gyro_sensor_name: str = "imu_gyro",
    ):
        def _adr(name):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise ValueError(f"Sensor '{name}' not found in model.")
            return model.sensor_adr[sid]

        self._accel_adr = _adr(accel_sensor_name)
        self._gyro_adr  = _adr(gyro_sensor_name)

        self.noise_accel      = noise_accel
        self.noise_gyro       = noise_gyro
        self.bias_accel_drift = bias_accel_drift
        self.bias_gyro_drift  = bias_gyro_drift

        # Persistent bias state — evolves every call to read()
        self.bias_accel = np.zeros(3)
        self.bias_gyro  = np.zeros(3)

    # ------------------------------------------------------------------

    def read(self, data, dt: float) -> IMUReading:
        """
        Read one IMU sample.  Call once per physics step.

        Parameters
        ----------
        data : mujoco.MjData
            Current simulation state.
        dt : float
            Physics timestep (model.opt.timestep).  Used to scale bias drift
            correctly — the Wiener process increment is N(0, σ²·dt).
        """
        accel_true = data.sensordata[self._accel_adr: self._accel_adr + 3].copy()
        gyro_true  = data.sensordata[self._gyro_adr:  self._gyro_adr  + 3].copy()

        # Bias random walk — sqrt(dt) gives correct Wiener process scaling
        self.bias_accel += np.random.randn(3) * self.bias_accel_drift * np.sqrt(dt)
        self.bias_gyro  += np.random.randn(3) * self.bias_gyro_drift  * np.sqrt(dt)

        accel_out = accel_true + self.bias_accel + np.random.randn(3) * self.noise_accel
        gyro_out  = gyro_true  + self.bias_gyro  + np.random.randn(3) * self.noise_gyro

        return IMUReading(
            t=data.time,
            accel=accel_out,
            gyro=gyro_out,
            accel_true=accel_true,
            gyro_true=gyro_true,
            accel_bias=self.bias_accel.copy(),
            gyro_bias=self.bias_gyro.copy(),
        )

    def reset(self):
        """Reset bias to zero.  Call when resetting the simulation."""
        self.bias_accel = np.zeros(3)
        self.bias_gyro  = np.zeros(3)
