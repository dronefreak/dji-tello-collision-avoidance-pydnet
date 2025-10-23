"""Abstract camera interface for different video sources.

Provides a common interface for webcams, Tello drones, and other video sources.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple


class CameraInterface(ABC):
    """Abstract base class for camera interfaces.

    All camera sources (webcam, Tello, etc.) should inherit from this class and
    implement the required methods.
    """

    def __init__(self):
        """Initialize the camera interface."""
        self._is_opened = False
        self._frame_count = 0

    @abstractmethod
    def open(self) -> bool:
        """Open/connect to the camera.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.

        Returns:
            Tuple of (success, frame) where:
                - success: True if frame was read successfully
                - frame: RGB image as numpy array (H, W, 3) or None if failed
        """
        pass

    @abstractmethod
    def release(self):
        """Close/disconnect from the camera."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is opened/connected.

        Returns:
            True if camera is ready, False otherwise
        """
        pass

    @abstractmethod
    def get_frame_size(self) -> Tuple[int, int]:
        """Get the frame size from the camera.

        Returns:
            Tuple of (height, width)
        """
        pass

    def get_frame_count(self) -> int:
        """Get the number of frames read since opening.

        Returns:
            Frame count
        """
        return self._frame_count

    def reset_frame_count(self):
        """Reset the frame counter."""
        self._frame_count = 0

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
