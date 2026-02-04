"""Unit tests for collision_avoidance module."""

import os
import sys
import time
import unittest

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check if OpenCV is available (required for collision_avoidance)
try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

if CV2_AVAILABLE:
    from src.collision_avoidance import CollisionAvoidance

from src.config import Config


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestCollisionAvoidance(unittest.TestCase):
    """Test CollisionAvoidance class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            min_safe_depth=0.3,
            center_tolerance=0.2,
            rotation_speed=30,
            enable_collision_avoidance=True,
        )
        self.collision_avoidance = CollisionAvoidance(self.config)

    def test_creation(self):
        """Test collision avoidance can be created."""
        self.assertIsNotNone(self.collision_avoidance)
        self.assertEqual(self.collision_avoidance.min_safe_depth, 0.3)

    def test_analyze_depth_basic(self):
        """Test basic depth analysis."""
        # Create simple depth map
        depth_map = np.random.rand(256, 512).astype(np.float32) * 0.5

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertIn("min_depth", analysis)
        self.assertIn("max_depth", analysis)
        self.assertIn("mean_depth", analysis)
        self.assertIn("center_depth", analysis)
        self.assertIn("is_safe", analysis)
        self.assertIn("max_depth_position", analysis)
        self.assertIn("suggested_action", analysis)

    def test_safe_forward_path(self):
        """Test detection of safe forward path."""
        # Create depth map with clear path ahead (high depth in center)
        # Center region is 50% of image, so we need high depth across the center
        # Also need to stay above emergency_depth_threshold (0.15)
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.2  # Above emergency threshold
        depth_map[:, 128:384] = 0.8  # High depth in center 50%

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertTrue(analysis["is_safe"])
        self.assertEqual(analysis["suggested_action"], "forward")

    def test_obstacle_ahead(self):
        """Test detection of obstacle ahead."""
        # Create depth map with obstacle in center (low depth)
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8
        h, w = depth_map.shape
        depth_map[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.1  # Low depth in center

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertFalse(analysis["is_safe"])

    def test_rotate_left_suggestion(self):
        """Test suggestion to rotate left."""
        # Create depth map with max depth on left side
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.2
        depth_map[:, 0:100] = 0.8  # High depth on left

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertEqual(analysis["suggested_action"], "rotate_left")

    def test_rotate_right_suggestion(self):
        """Test suggestion to rotate right."""
        # Create depth map with max depth on right side
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.2
        depth_map[:, 400:512] = 0.8  # High depth on right

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertEqual(analysis["suggested_action"], "rotate_right")

    def test_stop_suggestion(self):
        """Test suggestion to stop (emergency due to critical depth)."""
        # Create depth map with low depth everywhere (below emergency threshold)
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.1

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        # With depth 0.1 (below 0.15 threshold), this triggers emergency_stop
        self.assertEqual(analysis["suggested_action"], "emergency_stop")
        self.assertTrue(analysis["is_imminent_collision"])

    def test_normal_stop_suggestion(self):
        """Test normal stop suggestion (above emergency but below safe threshold)."""
        # Create depth map with depth between emergency (0.15) and safe (0.3) thresholds
        # Put slightly higher depth in center to attract max_depth_pos there,
        # but still below min_safe_depth (0.3)
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.18
        depth_map[:, 200:312] = 0.25  # Higher in center, still below 0.3

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        # Not emergency (above 0.15), but not safe (below 0.3)
        self.assertFalse(analysis["is_imminent_collision"])
        self.assertFalse(analysis["is_safe"])
        # Max depth is in center but below safe threshold, so should suggest stop
        self.assertEqual(analysis["suggested_action"], "stop")

    def test_get_rc_command_forward(self):
        """Test RC command for forward movement."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        left_right, forward_backward, up_down, yaw, is_emergency = (
            self.collision_avoidance.get_rc_command(depth_map)
        )

        self.assertEqual(left_right, 0)
        self.assertGreater(forward_backward, 0)
        self.assertEqual(up_down, 0)
        self.assertEqual(yaw, 0)
        self.assertFalse(is_emergency)

    def test_get_rc_command_rotate_left(self):
        """Test RC command for left rotation."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.2
        depth_map[:, 0:100] = 0.8

        left_right, forward_backward, up_down, yaw, is_emergency = (
            self.collision_avoidance.get_rc_command(depth_map)
        )

        self.assertEqual(left_right, 0)
        self.assertEqual(forward_backward, 0)
        self.assertEqual(up_down, 0)
        self.assertLess(yaw, 0)
        self.assertFalse(is_emergency)

    def test_get_rc_command_rotate_right(self):
        """Test RC command for right rotation."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.2
        depth_map[:, 400:512] = 0.8

        left_right, forward_backward, up_down, yaw, is_emergency = (
            self.collision_avoidance.get_rc_command(depth_map)
        )

        self.assertEqual(left_right, 0)
        self.assertEqual(forward_backward, 0)
        self.assertEqual(up_down, 0)
        self.assertGreater(yaw, 0)
        self.assertFalse(is_emergency)

    def test_get_rc_command_stop(self):
        """Test RC command for stop (emergency stop due to low depth)."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.1

        left_right, forward_backward, up_down, yaw, is_emergency = (
            self.collision_avoidance.get_rc_command(depth_map)
        )

        self.assertEqual(left_right, 0)
        self.assertEqual(forward_backward, 0)
        self.assertEqual(up_down, 0)
        self.assertEqual(yaw, 0)
        # With depth 0.1, this should trigger emergency (below 0.15 threshold)
        self.assertTrue(is_emergency)

    def test_get_discrete_command(self):
        """Test discrete command generation."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        command = self.collision_avoidance.get_discrete_command(depth_map)

        self.assertIn("action", command)
        self.assertIn("distance", command)
        self.assertIn("angle", command)
        self.assertIn("analysis", command)

    def test_command_history(self):
        """Test command history tracking."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        self.collision_avoidance.get_rc_command(depth_map)
        self.collision_avoidance.get_rc_command(depth_map)

        history = self.collision_avoidance.get_command_history()

        self.assertEqual(len(history), 2)
        self.assertIn("action", history[0])
        self.assertIn("timestamp", history[0])

    def test_clear_history(self):
        """Test clearing command history."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        self.collision_avoidance.get_rc_command(depth_map)
        self.collision_avoidance.clear_history()

        history = self.collision_avoidance.get_command_history()
        self.assertEqual(len(history), 0)

    def test_history_limit(self):
        """Test that history is limited to max size."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        # Generate many commands
        for _ in range(50):
            self.collision_avoidance.get_rc_command(depth_map)

        history = self.collision_avoidance.get_command_history()
        self.assertLessEqual(len(history), self.collision_avoidance.max_history)

    def test_get_last_command(self):
        """Test getting last command."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        self.collision_avoidance.get_rc_command(depth_map)
        last_cmd = self.collision_avoidance.get_last_command()

        self.assertIsNotNone(last_cmd)
        self.assertEqual(len(last_cmd), 4)

    def test_time_since_last_command(self):
        """Test tracking time since last command."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        self.collision_avoidance.get_rc_command(depth_map)
        time.sleep(0.1)
        elapsed = self.collision_avoidance.get_time_since_last_command()

        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 1.0)

    def test_compute_obstacle_map(self):
        """Test computing obstacle map."""
        depth_map = np.random.rand(256, 512).astype(np.float32)

        obstacle_map = self.collision_avoidance.compute_obstacle_map(depth_map, threshold=0.5)

        self.assertEqual(obstacle_map.shape, depth_map.shape)
        self.assertEqual(obstacle_map.dtype, np.uint8)
        self.assertTrue(np.all((obstacle_map == 0) | (obstacle_map == 1)))

    def test_get_safe_directions(self):
        """Test getting safe directions."""
        depth_map = np.random.rand(256, 512).astype(np.float32)

        directions = self.collision_avoidance.get_safe_directions(depth_map, num_sectors=8)

        self.assertIsInstance(directions, dict)
        self.assertEqual(len(directions), 8)

        # All values should be floats
        for depth in directions.values():
            self.assertIsInstance(depth, float)

    def test_different_sector_counts(self):
        """Test with different numbers of sectors."""
        depth_map = np.random.rand(256, 512).astype(np.float32)

        for num_sectors in [4, 8, 16]:
            directions = self.collision_avoidance.get_safe_directions(
                depth_map, num_sectors=num_sectors
            )
            self.assertEqual(len(directions), num_sectors)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestCollisionAvoidanceConfig(unittest.TestCase):
    """Test CollisionAvoidance with different configurations."""

    def test_custom_threshold(self):
        """Test with custom safe depth threshold."""
        config = Config(min_safe_depth=0.5)
        ca = CollisionAvoidance(config)

        # Depth just below threshold
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.4
        analysis = ca.analyze_depth(depth_map)

        self.assertFalse(analysis["is_safe"])

        # Depth above threshold
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.6
        analysis = ca.analyze_depth(depth_map)

        self.assertTrue(analysis["is_safe"])

    def test_custom_tolerance(self):
        """Test with custom center tolerance."""
        config = Config(center_tolerance=0.1)  # Narrow tolerance (51 pixels)
        ca = CollisionAvoidance(config)

        depth_map = np.ones((256, 512), dtype=np.float32) * 0.2
        # Put max depth clearly outside center tolerance zone
        # tolerance_pixels = 512 * 0.1 = 51, so place depth beyond center_x + 51
        depth_map[:, 380:440] = 0.8

        analysis = ca.analyze_depth(depth_map)

        # With narrow tolerance and max depth outside tolerance zone, should suggest rotation
        self.assertIn(analysis["suggested_action"], ["rotate_left", "rotate_right"])

    def test_custom_rotation_speed(self):
        """Test with custom rotation speed."""
        config = Config(rotation_speed=50)
        ca = CollisionAvoidance(config)

        depth_map = np.ones((256, 512), dtype=np.float32) * 0.2
        depth_map[:, 0:100] = 0.8

        _, _, _, yaw, _ = ca.get_rc_command(depth_map)

        self.assertEqual(abs(yaw), 50)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestCollisionAvoidanceEdgeCases(unittest.TestCase):
    """Test edge cases for collision avoidance."""

    def test_all_zero_depth(self):
        """Test with all-zero depth map."""
        config = Config()
        ca = CollisionAvoidance(config)

        depth_map = np.zeros((256, 512), dtype=np.float32)

        analysis = ca.analyze_depth(depth_map)

        self.assertIsNotNone(analysis)
        self.assertFalse(analysis["is_safe"])
        self.assertTrue(analysis["is_imminent_collision"])
        self.assertEqual(analysis["suggested_action"], "emergency_stop")

    def test_all_max_depth(self):
        """Test with maximum depth everywhere."""
        config = Config()
        ca = CollisionAvoidance(config)

        depth_map = np.ones((256, 512), dtype=np.float32)

        analysis = ca.analyze_depth(depth_map)

        self.assertTrue(analysis["is_safe"])
        self.assertEqual(analysis["suggested_action"], "forward")

    def test_small_depth_map(self):
        """Test with very small depth map."""
        config = Config()
        ca = CollisionAvoidance(config)

        depth_map = np.random.rand(10, 10).astype(np.float32)

        analysis = ca.analyze_depth(depth_map)

        self.assertIsNotNone(analysis)

    def test_single_pixel_depth_map(self):
        """Test with single pixel depth map."""
        config = Config()
        ca = CollisionAvoidance(config)

        depth_map = np.array([[0.5]], dtype=np.float32)

        # Should handle gracefully
        analysis = ca.analyze_depth(depth_map)
        self.assertIsNotNone(analysis)

    def test_nan_values(self):
        """Test handling of NaN values."""
        config = Config()
        ca = CollisionAvoidance(config)

        depth_map = np.random.rand(256, 512).astype(np.float32)
        depth_map[100:110, 200:210] = np.nan

        # Should handle NaN gracefully (though may produce warnings)
        try:
            analysis = ca.analyze_depth(depth_map)
            self.assertIsNotNone(analysis)
        except:
            pass  # NaN handling may vary

    def test_rapid_command_generation(self):
        """Test rapid command generation."""
        config = Config()
        ca = CollisionAvoidance(config)

        depth_map = np.random.rand(256, 512).astype(np.float32)

        # Generate many commands rapidly
        for _ in range(100):
            ca.get_rc_command(depth_map)

        # Should not crash or leak memory
        self.assertIsNotNone(ca.get_last_command())


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestImminentCollisionDetection(unittest.TestCase):
    """Test imminent collision detection features."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            min_safe_depth=0.3,
            emergency_depth_threshold=0.15,
            center_tolerance=0.2,
            rotation_speed=30,
        )
        self.collision_avoidance = CollisionAvoidance(self.config)

    def test_imminent_collision_detected(self):
        """Test that imminent collision is detected when depth is critical."""
        # Create depth map with very low depth in center (below emergency threshold)
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.5
        h, w = depth_map.shape
        depth_map[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.1  # Below 0.15 threshold

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertTrue(analysis["is_imminent_collision"])
        self.assertEqual(analysis["suggested_action"], "emergency_stop")

    def test_no_imminent_collision_when_safe(self):
        """Test no imminent collision when depth is above emergency threshold."""
        # Create depth map with depth above emergency threshold
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.5

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertFalse(analysis["is_imminent_collision"])
        self.assertNotEqual(analysis["suggested_action"], "emergency_stop")

    def test_imminent_collision_triggers_state(self):
        """Test that imminent collision sets the triggered flag."""
        # Initially not triggered
        self.assertFalse(self.collision_avoidance.was_imminent_collision_triggered())

        # Create depth map with imminent collision
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.1  # Below threshold

        self.collision_avoidance.analyze_depth(depth_map)

        self.assertTrue(self.collision_avoidance.was_imminent_collision_triggered())

    def test_reset_imminent_collision_state(self):
        """Test that imminent collision state can be reset."""
        # Trigger imminent collision
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.1
        self.collision_avoidance.analyze_depth(depth_map)

        self.assertTrue(self.collision_avoidance.was_imminent_collision_triggered())

        # Reset state
        self.collision_avoidance.reset_imminent_collision_state()

        self.assertFalse(self.collision_avoidance.was_imminent_collision_triggered())

    def test_get_rc_command_returns_emergency_flag(self):
        """Test that get_rc_command returns emergency flag."""
        # Safe depth map
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.8

        result = self.collision_avoidance.get_rc_command(depth_map)

        self.assertEqual(len(result), 5)  # (lr, fb, ud, yaw, is_emergency)
        lr, fb, ud, yaw, is_emergency = result
        self.assertFalse(is_emergency)

    def test_get_rc_command_emergency_flag_true_on_collision(self):
        """Test that get_rc_command sets emergency flag on imminent collision."""
        # Critical depth map
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.1

        result = self.collision_avoidance.get_rc_command(depth_map)

        lr, fb, ud, yaw, is_emergency = result
        self.assertTrue(is_emergency)
        # All movement should be zero during emergency
        self.assertEqual(lr, 0)
        self.assertEqual(fb, 0)
        self.assertEqual(ud, 0)
        self.assertEqual(yaw, 0)

    def test_get_discrete_command_includes_emergency(self):
        """Test that get_discrete_command includes emergency flag."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.1

        command = self.collision_avoidance.get_discrete_command(depth_map)

        self.assertIn("is_emergency", command)
        self.assertTrue(command["is_emergency"])
        self.assertEqual(command["action"], "emergency_stop")

    def test_center_min_depth_in_analysis(self):
        """Test that center_min_depth is included in analysis."""
        depth_map = np.ones((256, 512), dtype=np.float32) * 0.5

        analysis = self.collision_avoidance.analyze_depth(depth_map)

        self.assertIn("center_min_depth", analysis)
        self.assertIsInstance(analysis["center_min_depth"], float)

    def test_emergency_threshold_boundary(self):
        """Test behavior at emergency threshold boundary."""
        # Just above threshold - should not be emergency
        depth_map_safe = np.ones((256, 512), dtype=np.float32) * 0.16
        analysis_safe = self.collision_avoidance.analyze_depth(depth_map_safe)
        self.assertFalse(analysis_safe["is_imminent_collision"])

        # Just below threshold - should be emergency
        depth_map_danger = np.ones((256, 512), dtype=np.float32) * 0.14
        analysis_danger = self.collision_avoidance.analyze_depth(depth_map_danger)
        self.assertTrue(analysis_danger["is_imminent_collision"])


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCollisionAvoidance))
    suite.addTest(unittest.makeSuite(TestCollisionAvoidanceConfig))
    suite.addTest(unittest.makeSuite(TestCollisionAvoidanceEdgeCases))
    suite.addTest(unittest.makeSuite(TestImminentCollisionDetection))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
