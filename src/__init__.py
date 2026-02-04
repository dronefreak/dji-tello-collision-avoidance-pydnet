"""DJI Tello Collision Avoidance with PyDNet.

A collision avoidance system for DJI Tello drones using monocular depth estimation.
Also supports webcam-based demos for testing without a drone.

Modules:
    - depth_estimator: Depth estimation using PyDNet (requires TensorFlow)
    - camera_interface: Abstract camera interface
    - webcam_source: Webcam camera implementation
    - tello_source: Tello drone camera implementation (requires djitellopy)
    - collision_avoidance: Navigation logic based on depth maps
    - utils: Utility functions for visualization and preprocessing (requires opencv-python)
    - config: Configuration management
"""

__version__ = "2.0.0"
__author__ = "dronefreak"

from .config import Config
from .camera_interface import CameraInterface

# Core imports - always available
__all__ = ["Config", "CameraInterface"]

# OpenCV-dependent modules (optional)
try:
    from .utils import apply_colormap, preprocess_image, visualize_depth

    __all__.extend(["apply_colormap", "preprocess_image", "visualize_depth"])
except ImportError:
    apply_colormap = None  # type: ignore[misc, assignment]
    preprocess_image = None  # type: ignore[misc, assignment]
    visualize_depth = None  # type: ignore[misc, assignment]

# TensorFlow-dependent modules (optional)
try:
    from .depth_estimator import DepthEstimator

    __all__.append("DepthEstimator")
except ImportError:
    DepthEstimator = None  # type: ignore[misc, assignment]
