"""Tello drone camera source implementation.

Provides video stream access from DJI Tello drone using djitellopy library.
"""

import numpy as np
from typing import Optional, Tuple
import time

from .camera_interface import CameraInterface
from .config import Config

try:
    from djitellopy import Tello

    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False
    print("Warning: djitellopy not installed. Tello functionality will be unavailable.")
    print("Install with: pip install djitellopy")


class TelloSource(CameraInterface):
    """Tello drone camera source.

    Provides video stream from DJI Tello drone and basic flight controls.
    """

    def __init__(self, config: Config):
        """Initialize Tello source.

        Args:
            config: Configuration object with Tello settings
        """
        super().__init__()

        if not TELLO_AVAILABLE:
            raise ImportError(
                "djitellopy is required for Tello support. Install with: pip install djitellopy"
            )

        self.config = config
        self.drone: Optional[Tello] = None
        self.frame_width = 960
        self.frame_height = 720
        self._stream_on = False

    def open(self) -> bool:
        """Connect to Tello drone and start video stream.

        Returns:
            True if successful, False otherwise
        """
        try:
            print("Connecting to Tello...")
            self.drone = Tello()
            self.drone.connect()

            # Get battery level
            battery = self.drone.get_battery()
            print(f"Tello connected. Battery: {battery}%")

            if battery < 10:
                print("Warning: Battery level is very low!")

            # Start video stream
            print("Starting video stream...")
            self.drone.streamon()
            self._stream_on = True

            # Wait for stream to initialize
            time.sleep(2)

            self._is_opened = True
            self._frame_count = 0

            print("Tello video stream ready")
            return True

        except Exception as e:
            print(f"Error connecting to Tello: {e}")
            print("Make sure the Tello is powered on and you're connected to its WiFi network.")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the Tello video stream.

        Returns:
            Tuple of (success, frame) where frame is RGB numpy array
        """
        if not self.is_opened():
            return False, None

        try:
            frame = self.drone.get_frame_read().frame

            if frame is None:
                return False, None

            # Frame is already in RGB format from djitellopy
            self._frame_count += 1
            return True, frame

        except Exception as e:
            print(f"Error reading Tello frame: {e}")
            return False, None

    def release(self):
        """Disconnect from Tello and stop video stream."""
        if self.drone is not None:
            try:
                if self._stream_on:
                    self.drone.streamoff()
                    self._stream_on = False

                # Land if flying
                if self.is_flying():
                    print("Landing drone...")
                    self.land()

                self.drone.end()
                print("Tello disconnected")

            except Exception as e:
                print(f"Error during Tello cleanup: {e}")

            finally:
                self._is_opened = False

    def is_opened(self) -> bool:
        """Check if Tello is connected.

        Returns:
            True if connected, False otherwise
        """
        return self._is_opened and self.drone is not None

    def get_frame_size(self) -> Tuple[int, int]:
        """Get the frame size.

        Returns:
            Tuple of (height, width)
        """
        return (self.frame_height, self.frame_width)

    # Flight control methods

    def takeoff(self):
        """Take off."""
        if self.is_opened():
            print("Taking off...")
            self.drone.takeoff()

    def land(self):
        """Land."""
        if self.is_opened():
            print("Landing...")
            self.drone.land()

    def is_flying(self) -> bool:
        """Check if drone is flying.

        Returns:
            True if flying, False otherwise
        """
        if not self.is_opened():
            return False

        try:
            return self.drone.is_flying
        except:
            return False

    def emergency(self):
        """Emergency stop (cuts motors immediately)."""
        if self.is_opened():
            print("EMERGENCY STOP!")
            self.drone.emergency()

    def move_forward(self, distance: int):
        """Move forward by distance in cm.

        Args:
            distance: Distance in cm (20-500)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.move_forward(distance)

    def move_back(self, distance: int):
        """Move backward by distance in cm.

        Args:
            distance: Distance in cm (20-500)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.move_back(distance)

    def move_left(self, distance: int):
        """Move left by distance in cm.

        Args:
            distance: Distance in cm (20-500)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.move_left(distance)

    def move_right(self, distance: int):
        """Move right by distance in cm.

        Args:
            distance: Distance in cm (20-500)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.move_right(distance)

    def move_up(self, distance: int):
        """Move up by distance in cm.

        Args:
            distance: Distance in cm (20-500)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.move_up(distance)

    def move_down(self, distance: int):
        """Move down by distance in cm.

        Args:
            distance: Distance in cm (20-500)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.move_down(distance)

    def rotate_clockwise(self, degrees: int):
        """Rotate clockwise by degrees.

        Args:
            degrees: Rotation in degrees (1-360)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.rotate_clockwise(degrees)

    def rotate_counter_clockwise(self, degrees: int):
        """Rotate counter-clockwise by degrees.

        Args:
            degrees: Rotation in degrees (1-360)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.rotate_counter_clockwise(degrees)

    def send_rc_control(self, left_right: int, forward_backward: int, up_down: int, yaw: int):
        """Send RC control commands.

        Args:
            left_right: -100 to 100 (left/right velocity)
            forward_backward: -100 to 100 (forward/backward velocity)
            up_down: -100 to 100 (up/down velocity)
            yaw: -100 to 100 (yaw velocity)
        """
        if self.is_opened() and self.config.tello_enable_commands:
            self.drone.send_rc_control(left_right, forward_backward, up_down, yaw)

    def get_battery(self) -> int:
        """Get battery percentage.

        Returns:
            Battery percentage (0-100)
        """
        if self.is_opened():
            try:
                return self.drone.get_battery()
            except:
                return 0
        return 0

    def get_flight_time(self) -> int:
        """Get flight time in seconds.

        Returns:
            Flight time in seconds
        """
        if self.is_opened():
            try:
                return self.drone.get_flight_time()
            except:
                return 0
        return 0

    def get_height(self) -> int:
        """Get current height in cm.

        Returns:
            Height in cm
        """
        if self.is_opened():
            try:
                return self.drone.get_height()
            except:
                return 0
        return 0

    def get_temperature(self) -> int:
        """Get drone temperature in Celsius.

        Returns:
            Temperature in Celsius
        """
        if self.is_opened():
            try:
                return self.drone.get_temperature()
            except:
                return 0
        return 0

    def get_telemetry(self) -> dict:
        """Get all telemetry data.

        Returns:
            Dictionary with telemetry information
        """
        if not self.is_opened():
            return {}

        telemetry = {
            "battery": self.get_battery(),
            "flight_time": self.get_flight_time(),
            "height": self.get_height(),
            "temperature": self.get_temperature(),
            "is_flying": self.is_flying(),
            "frame_count": self._frame_count,
        }

        return telemetry
