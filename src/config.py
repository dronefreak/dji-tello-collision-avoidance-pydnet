"""Configuration management for the collision avoidance system.

This module provides a centralized configuration class for managing model parameters,
camera settings, and navigation options.
"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Configuration class for depth estimation and collision avoidance."""

    # Model settings
    model_name: str = "pydnet"
    checkpoint_dir: str = "checkpoint/IROS18/pydnet"
    resolution: int = 1  # 1:High, 2:Quarter, 3:Eighth

    # Image dimensions
    input_width: int = 512
    input_height: int = 256

    # Display settings
    display_width: int = 640
    display_height: int = 480
    show_fps: bool = True
    colormap: str = "plasma"  # Options: plasma, viridis, magma, inferno, turbo

    # Depth visualization
    depth_scale: float = 20.0  # Scaling factor for depth visualization

    # Camera settings
    camera_id: int = 0  # Webcam device ID
    camera_fps: int = 30

    # Tello settings
    tello_speed: int = 50  # Movement speed (0-100)
    tello_enable_commands: bool = False  # Safety: disabled by default

    # Collision avoidance
    enable_collision_avoidance: bool = False
    min_safe_depth: float = 0.3  # Minimum safe depth threshold
    center_tolerance: float = 0.2  # Tolerance for center region (20% of frame)
    rotation_speed: int = 30  # Yaw rotation speed

    # Performance
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.5  # GPU memory allocation

    # Recording
    save_output: bool = False
    output_dir: str = "output"
    save_depth_maps: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.resolution not in [1, 2, 3]:
            raise ValueError("Resolution must be 1 (High), 2 (Quarter), or 3 (Eighth)")

        if not 0 <= self.tello_speed <= 100:
            raise ValueError("Tello speed must be between 0 and 100")

        if self.min_safe_depth <= 0:
            raise ValueError("Minimum safe depth must be positive")

        if not 0 <= self.center_tolerance <= 1:
            raise ValueError("Center tolerance must be between 0 and 1")

        # Create output directory if saving is enabled
        if self.save_output and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return {
            "model_name": self.model_name,
            "checkpoint_dir": self.checkpoint_dir,
            "resolution": self.resolution,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "show_fps": self.show_fps,
            "colormap": self.colormap,
            "depth_scale": self.depth_scale,
            "camera_id": self.camera_id,
            "camera_fps": self.camera_fps,
            "tello_speed": self.tello_speed,
            "tello_enable_commands": self.tello_enable_commands,
            "enable_collision_avoidance": self.enable_collision_avoidance,
            "min_safe_depth": self.min_safe_depth,
            "center_tolerance": self.center_tolerance,
            "rotation_speed": self.rotation_speed,
            "use_gpu": self.use_gpu,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "save_output": self.save_output,
            "output_dir": self.output_dir,
            "save_depth_maps": self.save_depth_maps,
        }

    def get_model_input_shape(self) -> Tuple[int, int]:
        """Get the input shape for the model."""
        return (self.input_height, self.input_width)

    def get_display_shape(self) -> Tuple[int, int]:
        """Get the display shape for visualization."""
        return (self.display_height, self.display_width)

    def validate_checkpoint(self) -> bool:
        """Check if checkpoint directory exists."""
        return os.path.exists(self.checkpoint_dir)


# Default configuration instance
DEFAULT_CONFIG = Config()
