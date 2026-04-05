import math
import time
import threading
from pyrplidar import PyRPlidar


class LidarReader:
    """
    Reads scans from RPLIDAR C1 in a background thread.
    Each complete 360° sweep is stored as current_scan.
    """

    def __init__(self, config):
        cfg = config["lidar"]
        self.port = cfg["port"]
        self.baudrate = cfg["baudrate"]
        self.timeout = cfg["timeout"]
        self.scan_mode = cfg["scan_mode"]
        self.min_quality = cfg["min_quality"]
        self.min_range = cfg["min_range_m"]
        self.max_range = cfg["max_range_m"]

        self._lidar = None
        self._lock = threading.Lock()
        self._current_scan = []   # list of (x, y) meters
        self._running = False
        self._thread = None
        self.scan_count = 0       # total completed sweeps

    def start(self):
        self._lidar = PyRPlidar()
        self._lidar.connect(port=self.port, baudrate=self.baudrate, timeout=self.timeout)

        info = self._lidar.get_info()
        health = self._lidar.get_health()
        print(f"LIDAR info:   {info}")
        print(f"LIDAR health: {health}")

        self._lidar.set_motor_pwm(660)
        time.sleep(2)  # let motor spin up

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("LIDAR reader started.")

    def _read_loop(self):
        # Try express scan first; fall back to standard scan if not supported
        try:
            scan_generator = self._lidar.start_scan_express(self.scan_mode)
            # peek to confirm it works
            gen = scan_generator()
            first = next(gen)
            print(f"Express scan mode {self.scan_mode} active.")
        except Exception as e:
            print(f"Express scan mode {self.scan_mode} failed ({e}), falling back to standard scan.")
            self._lidar.stop()
            scan_generator = self._lidar.start_scan()
            gen = scan_generator()
            first = next(gen)

        import itertools
        pending = []
        prev_angle = None

        for measurement in itertools.chain([first], gen):
            if not self._running:
                break

            angle = measurement.angle       # degrees [0, 360)
            distance_mm = measurement.distance
            quality = measurement.quality

            # filter bad measurements
            if quality >= self.min_quality and distance_mm > 0:
                dist_m = distance_mm / 1000.0
                if self.min_range <= dist_m <= self.max_range:
                    rad = math.radians(angle)
                    x = dist_m * math.cos(rad)
                    y = dist_m * math.sin(rad)
                    pending.append((x, y))

            # detect full sweep: angle wraps from ~360 back to ~0
            if prev_angle is not None and angle < prev_angle - 180:
                with self._lock:
                    self._current_scan = pending.copy()
                    self.scan_count += 1
                pending = []

            prev_angle = angle

    def get_scan(self):
        """Return the latest complete scan as list of (x, y) in meters."""
        with self._lock:
            return list(self._current_scan)

    def stop(self):
        self._running = False
        if self._lidar:
            self._lidar.stop()
            self._lidar.set_motor_pwm(0)
            self._lidar.disconnect()
        print("LIDAR reader stopped.")
