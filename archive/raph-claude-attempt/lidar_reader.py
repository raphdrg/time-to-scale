import math
import time
import threading
import serial

class LidarReader:
    """
    Reads scans from RPLIDAR C1 in a background thread.
    Each complete 360° sweep is stored as current_scan.
    Uses raw pyserial instead of pyrplidar (broken on Python 3.14).
    """

    SYNC_BYTE1 = 0xA5
    CMD_SCAN   = 0x20
    CMD_STOP   = 0x25
    CMD_RESET  = 0x40

    def __init__(self, config):
        cfg = config["lidar"]
        self.port        = cfg["port"]
        self.baudrate    = cfg["baudrate"]
        self.timeout     = cfg["timeout"]
        self.min_quality = cfg["min_quality"]
        self.min_range   = cfg["min_range_m"]
        self.max_range   = cfg["max_range_m"]

        self._serial      = None
        self._lock        = threading.Lock()
        self._current_scan = []
        self._running     = False
        self._thread      = None
        self.scan_count   = 0

    def _send_command(self, cmd):
        self._serial.write(bytes([self.SYNC_BYTE1, cmd]))

    def start(self):
        self._serial = serial.Serial(
            self.port,
            baudrate=self.baudrate,
            timeout=self.timeout
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

        # Reset and wait for motor spin-up
        self._send_command(self.CMD_RESET)
        time.sleep(2)
        self._serial.reset_input_buffer()

        # Start scan and read descriptor
        self._send_command(self.CMD_SCAN)
        descriptor = self._serial.read(7)
        print(f"LIDAR descriptor: {descriptor.hex()}")
        time.sleep(1)

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("LIDAR reader started.")

    def _read_point(self):
        raw = self._serial.read(5)
        if len(raw) < 5:
            return None

        quality  = (raw[0] >> 2) & 0x3F
        angle    = ((raw[1] | (raw[2] << 8)) >> 1) / 64.0
        distance = (raw[3] | (raw[4] << 8)) / 4.0  # mm

        if quality < self.min_quality or distance <= 0:
            return None

        dist_m = distance / 1000.0
        if not (self.min_range <= dist_m <= self.max_range):
            return None

        return angle, dist_m

    def _read_loop(self):
        pending    = []
        prev_angle = None

        while self._running:
            result = self._read_point()
            if result is None:
                continue

            angle, dist_m = result
            rad = math.radians(angle)
            x   = dist_m * math.cos(rad)
            y   = dist_m * math.sin(rad)
            pending.append((x, y))

            # Detect full 360° sweep by angle wraparound
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
        if self._serial and self._serial.is_open:
            self._send_command(self.CMD_STOP)
            time.sleep(0.1)
            self._serial.close()
        print("LIDAR reader stopped.")