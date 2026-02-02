"""Collision avoidance logic based on depth estimation.

Implements navigation algorithms that use depth maps to avoid obstacles and navigate
towards regions with greater depth.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time

from .config import Config
from .utils import find_max_depth_region


class CollisionAvoidance:
    """Collision avoidance system using depth maps.

    Analyzes depth maps to determine safe navigation directions and generates control
    commands for the drone.
    """

    def __init__(self, config: Config):
        """Initialize collision avoidance system.

        Args:
            config: Configuration object
        """
        self.config = config
        self.min_safe_depth = config.min_safe_depth
        self.emergency_depth_threshold = config.emergency_depth_threshold
        self.center_tolerance = config.center_tolerance
        self.rotation_speed = config.rotation_speed

        # State tracking
        self.last_command = None
        self.last_command_time = 0
        self.command_history = []
        self.max_history = 30
        self._imminent_collision_triggered = False

    def analyze_depth(self, depth_map: np.ndarray) -> Dict[str, Any]:
        """Analyze depth map for obstacle detection and navigation.

        Args:
            depth_map: Depth map as numpy array (H, W)

        Returns:
            Dictionary containing analysis results:
                - min_depth: Minimum depth value
                - max_depth: Maximum depth value
                - mean_depth: Mean depth value
                - center_depth: Depth at center of frame
                - is_safe: Whether it's safe to move forward
                - is_imminent_collision: Whether an imminent collision is detected
                - max_depth_position: Position of maximum depth region
                - suggested_action: Recommended navigation action
        """
        h, w = depth_map.shape

        # Basic statistics
        min_depth = float(np.min(depth_map))
        max_depth = float(np.max(depth_map))
        mean_depth = float(np.mean(depth_map))

        # Center region analysis
        center_y, center_x = h // 2, w // 2
        center_region = depth_map[
            center_y - h // 4 : center_y + h // 4, center_x - w // 4 : center_x + w // 4
        ]
        center_depth = float(np.mean(center_region))

        # Safety check
        is_safe = center_depth > self.min_safe_depth

        # Imminent collision detection - critical safety check
        # Check if minimum depth in center region is below emergency threshold
        center_min_depth = float(np.min(center_region))
        is_imminent_collision = center_min_depth < self.emergency_depth_threshold

        # Track imminent collision state
        if is_imminent_collision:
            self._imminent_collision_triggered = True

        # Find region with maximum depth
        max_depth_pos = find_max_depth_region(depth_map)

        # Determine suggested action (emergency_stop overrides other actions)
        if is_imminent_collision:
            suggested_action = "emergency_stop"
        else:
            suggested_action = self._determine_action(depth_map, max_depth_pos)

        analysis = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "mean_depth": mean_depth,
            "center_depth": center_depth,
            "center_min_depth": center_min_depth,
            "is_safe": is_safe,
            "is_imminent_collision": is_imminent_collision,
            "max_depth_position": max_depth_pos,
            "suggested_action": suggested_action,
        }

        return analysis

    def _determine_action(self, depth_map: np.ndarray, max_depth_pos: Tuple[int, int]) -> str:
        """Determine the best navigation action based on depth map.

        Args:
            depth_map: Depth map
            max_depth_pos: Position of maximum depth region (y, x)

        Returns:
            Suggested action: 'forward', 'rotate_left', 'rotate_right', 'stop'
        """
        h, w = depth_map.shape
        center_x = w // 2
        tolerance_pixels = int(w * self.center_tolerance)

        max_y, max_x = max_depth_pos

        # Check if maximum depth is in center region
        if abs(max_x - center_x) < tolerance_pixels:
            # Safe to move forward
            center_depth = depth_map[h // 2, center_x]
            if center_depth > self.min_safe_depth:
                return "forward"
            else:
                return "stop"

        # Need to rotate towards maximum depth region
        elif max_x < center_x - tolerance_pixels:
            return "rotate_left"
        else:
            return "rotate_right"

    def get_rc_command(self, depth_map: np.ndarray) -> Tuple[int, int, int, int, bool]:
        """Generate RC control command based on depth map.

        Args:
            depth_map: Depth map

        Returns:
            Tuple of (left_right, forward_backward, up_down, yaw, is_emergency)
            - RC values are in range [-100, 100]
            - is_emergency is True if an emergency stop was triggered
        """
        analysis = self.analyze_depth(depth_map)
        action = analysis["suggested_action"]
        is_emergency = analysis["is_imminent_collision"]

        # Initialize command (all zeros = hover)
        left_right = 0
        forward_backward = 0
        up_down = 0
        yaw = 0

        # Generate command based on action
        if action == "emergency_stop":
            # All zeros - immediate stop, caller should handle emergency landing
            pass
        elif action == "forward":
            forward_backward = self.config.tello_speed
        elif action == "rotate_left":
            yaw = -self.rotation_speed
        elif action == "rotate_right":
            yaw = self.rotation_speed
        elif action == "stop":
            # Hover in place
            pass

        # Record command
        self.last_command = (left_right, forward_backward, up_down, yaw)
        self.last_command_time = time.time()
        self._add_to_history(action)

        return (left_right, forward_backward, up_down, yaw, is_emergency)

    def get_discrete_command(self, depth_map: np.ndarray) -> Dict[str, Any]:
        """Generate discrete movement command based on depth map.

        Args:
            depth_map: Depth map

        Returns:
            Dictionary with command details:
                - action: Action string
                - distance: Movement distance (if applicable)
                - angle: Rotation angle (if applicable)
                - is_emergency: Whether emergency stop was triggered
                - analysis: Depth analysis results
        """
        analysis = self.analyze_depth(depth_map)
        action = analysis["suggested_action"]

        command = {
            "action": action,
            "distance": 0,
            "angle": 0,
            "is_emergency": analysis["is_imminent_collision"],
            "analysis": analysis,
        }

        if action == "forward":
            command["distance"] = 30  # cm
        elif action in ["rotate_left", "rotate_right"]:
            command["angle"] = 30  # degrees
        elif action == "emergency_stop":
            command["distance"] = 0
            command["angle"] = 0

        self._add_to_history(action)

        return command

    def _add_to_history(self, action: str):
        """Add action to history."""
        self.command_history.append(
            {
                "action": action,
                "timestamp": time.time(),
            }
        )

        # Keep only recent history
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

    def get_command_history(self) -> list:
        """Get command history.

        Returns:
            List of recent commands
        """
        return self.command_history.copy()

    def clear_history(self):
        """Clear command history."""
        self.command_history = []

    def get_last_command(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the last RC command sent.

        Returns:
            Last RC command or None
        """
        return self.last_command

    def get_time_since_last_command(self) -> float:
        """Get time elapsed since last command.

        Returns:
            Time in seconds
        """
        if self.last_command_time == 0:
            return float("inf")
        return time.time() - self.last_command_time

    def was_imminent_collision_triggered(self) -> bool:
        """Check if an imminent collision was ever triggered.

        Returns:
            True if imminent collision was detected at any point
        """
        return self._imminent_collision_triggered

    def reset_imminent_collision_state(self):
        """Reset the imminent collision triggered flag.

        Call this after handling an emergency situation.
        """
        self._imminent_collision_triggered = False

    def compute_obstacle_map(
        self, depth_map: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Create binary obstacle map from depth map.

        Args:
            depth_map: Depth map
            threshold: Depth threshold (uses config default if None)

        Returns:
            Binary obstacle map (1 = obstacle, 0 = free)
        """
        if threshold is None:
            threshold = self.min_safe_depth

        obstacle_map = (depth_map < threshold).astype(np.uint8)
        return obstacle_map

    def get_safe_directions(self, depth_map: np.ndarray, num_sectors: int = 8) -> Dict[str, float]:
        """Analyze depth in different directional sectors.

        Args:
            depth_map: Depth map
            num_sectors: Number of angular sectors to analyze

        Returns:
            Dictionary mapping directions to average depth
        """
        h, w = depth_map.shape
        center_y, center_x = h // 2, w // 2

        # Create polar coordinate map
        y_coords, x_coords = np.ogrid[:h, :w]
        y_coords = y_coords - center_y
        x_coords = x_coords - center_x

        angles = np.arctan2(y_coords, x_coords)

        # Divide into sectors
        sector_size = 2 * np.pi / num_sectors
        sectors = {}

        for i in range(num_sectors):
            sector_start = -np.pi + i * sector_size
            sector_end = sector_start + sector_size

            mask = (angles >= sector_start) & (angles < sector_end)
            sector_depth = np.mean(depth_map[mask]) if mask.any() else 0.0

            # Convert to direction name
            angle_deg = np.degrees(sector_start + sector_size / 2)
            direction = self._angle_to_direction(angle_deg)
            sectors[direction] = float(sector_depth)

        return sectors

    def _angle_to_direction(self, angle_deg: float) -> str:
        """Convert angle to direction name."""
        # Normalize angle to [0, 360)
        angle_deg = angle_deg % 360

        if angle_deg < 22.5 or angle_deg >= 337.5:
            return "right"
        elif angle_deg < 67.5:
            return "front-right"
        elif angle_deg < 112.5:
            return "front"
        elif angle_deg < 157.5:
            return "front-left"
        elif angle_deg < 202.5:
            return "left"
        elif angle_deg < 247.5:
            return "back-left"
        elif angle_deg < 292.5:
            return "back"
        else:
            return "back-right"
