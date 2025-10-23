"""Webcam camera source implementation.

Provides webcam access through OpenCV for testing depth estimation without a drone.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from .camera_interface import CameraInterface
from .config import Config


class WebcamSource(CameraInterface):
    """Webcam camera source using OpenCV.

    Provides a simple interface to read frames from a USB/built-in webcam.
    """

    def __init__(self, config: Config):
        """Initialize webcam source.

        Args:
            config: Configuration object with camera settings
        """
        super().__init__()
        self.config = config
        self.camera_id = config.camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width = 0
        self.frame_height = 0

    def open(self) -> bool:
        """Open the webcam.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.display_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)

            # Get actual frame size
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._is_opened = True
            self._frame_count = 0

            print(
                f"Webcam opened: {self.frame_width}x{self.frame_height}"
                f" @ {self.config.camera_fps}fps"
            )
            return True

        except Exception as e:
            print(f"Error opening webcam: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the webcam.

        Returns:
            Tuple of (success, frame) where frame is RGB numpy array
        """
        if not self.is_opened():
            return False, None

        try:
            ret, frame = self.cap.read()

            if not ret:
                return False, None

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._frame_count += 1
            return True, frame_rgb

        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None

    def release(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            print("Webcam released")

    def is_opened(self) -> bool:
        """Check if webcam is opened.

        Returns:
            True if opened, False otherwise
        """
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def get_frame_size(self) -> Tuple[int, int]:
        """Get the frame size.

        Returns:
            Tuple of (height, width)
        """
        return (self.frame_height, self.frame_width)

    def set_camera_id(self, camera_id: int) -> bool:
        """Change the camera device.

        Args:
            camera_id: New camera ID

        Returns:
            True if successful, False otherwise
        """
        was_opened = self.is_opened()

        if was_opened:
            self.release()

        self.camera_id = camera_id

        if was_opened:
            return self.open()

        return True

    def get_camera_info(self) -> dict:
        """Get camera information.

        Returns:
            Dictionary with camera properties
        """
        if not self.is_opened():
            return {}

        info = {
            "camera_id": self.camera_id,
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "backend": self.cap.getBackendName(),
            "frame_count": self._frame_count,
        }

        return info

    def adjust_brightness(self, value: int):
        """Adjust camera brightness.

        Args:
            value: Brightness value (camera-dependent range)
        """
        if self.is_opened():
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def adjust_contrast(self, value: int):
        """Adjust camera contrast.

        Args:
            value: Contrast value (camera-dependent range)
        """
        if self.is_opened():
            self.cap.set(cv2.CAP_PROP_CONTRAST, value)

    def adjust_exposure(self, value: int):
        """Adjust camera exposure.

        Args:
            value: Exposure value (camera-dependent range)
        """
        if self.is_opened():
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
