"""DJI Tello Collision Avoidance with PyDNet.

A collision avoidance system for DJI Tello drones using monocular depth estimation.
Also supports webcam-based demos for testing without a drone.

Modules:
    - depth_estimator: Depth estimation using PyDNet
    - camera_interface: Abstract camera interface
    - webcam_source: Webcam camera implementation
    - tello_source: Tello drone camera implementation
    - collision_avoidance: Navigation logic based on depth maps
    - utils: Utility functions for visualization and preprocessing
    - config: Configuration management
"""

__version__ = "2.0.0"
__author__ = "dronefreak"

from .config import Config
from .depth_estimator import DepthEstimator
from .camera_interface import CameraInterface
from .utils import apply_colormap, preprocess_image, visualize_depth

__all__ = [
    "Config",
    "DepthEstimator",
    "CameraInterface",
    "apply_colormap",
    "preprocess_image",
    "visualize_depth",
]
