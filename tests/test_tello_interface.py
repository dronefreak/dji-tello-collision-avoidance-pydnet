"""Unit tests for tello_interface functionality.

Tests the Tello source and integration using mocks (no real drone needed).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check if OpenCV is available (required for imports through src package)
try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.config import Config


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestTelloSourceImport(unittest.TestCase):
    """Test Tello source import behavior."""

    def test_import_without_djitellopy(self):
        """Test importing when djitellopy is not available."""
        # This test verifies the import doesn't crash
        with patch.dict("sys.modules", {"djitellopy": None}):
            try:
                from src.tello_source import TELLO_AVAILABLE

                self.assertFalse(TELLO_AVAILABLE)
            except ImportError:
                pass  # Expected if djitellopy truly not available


@patch("src.tello_source.Tello")
class TestTelloSource(unittest.TestCase):
    """Test TelloSource class with mocked Tello."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(tello_speed=50, tello_enable_commands=True)

    def test_tello_creation(self, mock_tello_class):
        """Test Tello source can be created."""
        from src.tello_source import TelloSource

        tello = TelloSource(self.config)
        self.assertIsNotNone(tello)

    def test_initial_state(self, mock_tello_class):
        """Test initial state is not opened."""
        from src.tello_source import TelloSource

        tello = TelloSource(self.config)
        self.assertFalse(tello.is_opened())

    def test_open_success(self, mock_tello_class):
        """Test successful connection to Tello."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        result = tello.open()

        self.assertTrue(result)
        self.assertTrue(tello.is_opened())
        mock_drone.connect.assert_called_once()
        mock_drone.streamon.assert_called_once()

    def test_open_failure(self, mock_tello_class):
        """Test failed connection to Tello."""
        from src.tello_source import TelloSource

        mock_tello_class.return_value.connect.side_effect = Exception("Connection failed")

        tello = TelloSource(self.config)
        result = tello.open()

        self.assertFalse(result)
        self.assertFalse(tello.is_opened())

    def test_read_frame(self, mock_tello_class):
        """Test reading a frame from Tello."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85

        # Mock frame
        test_frame = np.random.randint(0, 255, (720, 960, 3), dtype=np.uint8)
        mock_frame_read = MagicMock()
        mock_frame_read.frame = test_frame
        mock_drone.get_frame_read.return_value = mock_frame_read

        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        success, frame = tello.read()

        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 960, 3))

    def test_read_failure(self, mock_tello_class):
        """Test failed frame read."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_frame_read = MagicMock()
        mock_frame_read.frame = None
        mock_drone.get_frame_read.return_value = mock_frame_read

        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        success, frame = tello.read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    def test_release(self, mock_tello_class):
        """Test releasing Tello connection."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_drone.is_flying = False
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()
        self.assertTrue(tello.is_opened())

        tello.release()

        self.assertFalse(tello.is_opened())
        mock_drone.streamoff.assert_called_once()
        mock_drone.end.assert_called_once()

    def test_get_frame_size(self, mock_tello_class):
        """Test getting frame size."""
        from src.tello_source import TelloSource

        tello = TelloSource(self.config)
        height, width = tello.get_frame_size()

        self.assertEqual(height, 720)
        self.assertEqual(width, 960)

    def test_takeoff(self, mock_tello_class):
        """Test takeoff command."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()
        tello.takeoff()

        mock_drone.takeoff.assert_called_once()

    def test_land(self, mock_tello_class):
        """Test land command."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()
        tello.land()

        mock_drone.land.assert_called_once()

    def test_emergency_stop(self, mock_tello_class):
        """Test emergency stop."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()
        tello.emergency()

        mock_drone.emergency.assert_called_once()

    def test_movement_commands(self, mock_tello_class):
        """Test various movement commands."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        # Test movement commands
        tello.move_forward(50)
        mock_drone.move_forward.assert_called_with(50)

        tello.move_back(30)
        mock_drone.move_back.assert_called_with(30)

        tello.move_left(40)
        mock_drone.move_left.assert_called_with(40)

        tello.move_right(35)
        mock_drone.move_right.assert_called_with(35)

        tello.move_up(25)
        mock_drone.move_up.assert_called_with(25)

        tello.move_down(20)
        mock_drone.move_down.assert_called_with(20)

    def test_rotation_commands(self, mock_tello_class):
        """Test rotation commands."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        tello.rotate_clockwise(90)
        mock_drone.rotate_clockwise.assert_called_with(90)

        tello.rotate_counter_clockwise(45)
        mock_drone.rotate_counter_clockwise.assert_called_with(45)

    def test_rc_control(self, mock_tello_class):
        """Test RC control command."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        tello.send_rc_control(10, 20, 30, 40)
        mock_drone.send_rc_control.assert_called_with(10, 20, 30, 40)

    def test_get_battery(self, mock_tello_class):
        """Test getting battery level."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 75
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        battery = tello.get_battery()
        self.assertEqual(battery, 75)

    def test_get_telemetry(self, mock_tello_class):
        """Test getting all telemetry."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 80
        mock_drone.get_flight_time.return_value = 120
        mock_drone.get_height.return_value = 50
        mock_drone.get_temperature.return_value = 45
        mock_drone.is_flying = True
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        telemetry = tello.get_telemetry()

        self.assertEqual(telemetry["battery"], 80)
        self.assertEqual(telemetry["flight_time"], 120)
        self.assertEqual(telemetry["height"], 50)
        self.assertEqual(telemetry["temperature"], 45)
        self.assertTrue(telemetry["is_flying"])

    def test_commands_disabled_by_config(self, mock_tello_class):
        """Test that commands are disabled when config says so."""
        from src.tello_source import TelloSource

        config_no_commands = Config(tello_enable_commands=False)

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(config_no_commands)
        tello.open()

        # Try to send commands (should be ignored)
        tello.move_forward(50)
        tello.send_rc_control(10, 20, 30, 40)

        # Commands should not be called
        mock_drone.move_forward.assert_not_called()
        mock_drone.send_rc_control.assert_not_called()

    def test_is_flying(self, mock_tello_class):
        """Test checking if drone is flying."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_drone.is_flying = True
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        self.assertTrue(tello.is_flying())

    def test_frame_count_increment(self, mock_tello_class):
        """Test frame count increments on read."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        test_frame = np.random.randint(0, 255, (720, 960, 3), dtype=np.uint8)
        mock_frame_read = MagicMock()
        mock_frame_read.frame = test_frame
        mock_drone.get_frame_read.return_value = mock_frame_read
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        self.assertEqual(tello.get_frame_count(), 0)

        tello.read()
        self.assertEqual(tello.get_frame_count(), 1)

        tello.read()
        tello.read()
        self.assertEqual(tello.get_frame_count(), 3)

    def test_low_battery_warning(self, mock_tello_class):
        """Test low battery warning on connection."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 5  # Very low
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        # Should still connect but print warning
        result = tello.open()

        self.assertTrue(result)

    def test_release_while_flying(self, mock_tello_class):
        """Test releasing while drone is flying."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_drone.is_flying = True
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()
        tello.release()

        # Should land before disconnecting
        mock_drone.land.assert_called_once()


class TestTelloEdgeCases(unittest.TestCase):
    """Test edge cases for Tello source."""

    @patch("src.tello_source.Tello")
    def test_read_without_opening(self, mock_tello_class):
        """Test reading before opening connection."""
        from src.tello_source import TelloSource

        config = Config()
        tello = TelloSource(config)

        success, frame = tello.read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    @patch("src.tello_source.Tello")
    def test_commands_without_opening(self, mock_tello_class):
        """Test sending commands before opening."""
        from src.tello_source import TelloSource

        config = Config(tello_enable_commands=True)
        tello = TelloSource(config)

        # Should not crash
        tello.takeoff()
        tello.land()

        mock_tello_class.assert_not_called()

    @patch("src.tello_source.Tello", None)
    def test_creation_without_djitellopy(self):
        """Test that creation fails gracefully without djitellopy."""
        from src.tello_source import TelloSource

        config = Config()

        with self.assertRaises(ImportError):
            TelloSource(config)


@patch("src.tello_source.Tello")
class TestTelloSafetyFeatures(unittest.TestCase):
    """Test safety features for Tello source."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(tello_speed=50, tello_enable_commands=True)

    def test_rc_control_clamps_values(self, mock_tello_class):
        """Test that RC control values are clamped to valid range."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        # Test values beyond range are clamped
        tello.send_rc_control(150, -200, 300, -400)
        mock_drone.send_rc_control.assert_called_with(100, -100, 100, -100)

    def test_rc_control_accepts_valid_values(self, mock_tello_class):
        """Test that valid RC control values pass through unchanged."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        # Test valid values pass through
        tello.send_rc_control(50, -30, 0, 75)
        mock_drone.send_rc_control.assert_called_with(50, -30, 0, 75)

    def test_rc_control_converts_floats_to_int(self, mock_tello_class):
        """Test that RC control converts float values to int."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        # Test float values are converted to int
        tello.send_rc_control(50.7, -30.2, 0.0, 75.9)
        mock_drone.send_rc_control.assert_called_with(50, -30, 0, 75)

    def test_takeoff_blocked_when_commands_disabled(self, mock_tello_class):
        """Test that takeoff is blocked when commands are disabled."""
        from src.tello_source import TelloSource

        config_no_commands = Config(tello_enable_commands=False)

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(config_no_commands)
        tello.open()

        result = tello.takeoff()

        self.assertFalse(result)
        mock_drone.takeoff.assert_not_called()

    def test_takeoff_allowed_when_commands_enabled(self, mock_tello_class):
        """Test that takeoff works when commands are enabled."""
        from src.tello_source import TelloSource

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(self.config)
        tello.open()

        result = tello.takeoff()

        self.assertTrue(result)
        mock_drone.takeoff.assert_called_once()

    def test_land_always_works(self, mock_tello_class):
        """Test that land works even when commands are disabled."""
        from src.tello_source import TelloSource

        config_no_commands = Config(tello_enable_commands=False)

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_drone.is_flying = False
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(config_no_commands)
        tello.open()

        result = tello.land()

        self.assertTrue(result)
        mock_drone.land.assert_called_once()

    def test_emergency_always_works(self, mock_tello_class):
        """Test that emergency stop works even when commands are disabled."""
        from src.tello_source import TelloSource

        config_no_commands = Config(tello_enable_commands=False)

        mock_drone = MagicMock()
        mock_drone.get_battery.return_value = 85
        mock_tello_class.return_value = mock_drone

        tello = TelloSource(config_no_commands)
        tello.open()

        result = tello.emergency()

        self.assertTrue(result)
        mock_drone.emergency.assert_called_once()

    def test_takeoff_returns_false_when_not_opened(self, mock_tello_class):
        """Test that takeoff returns False when not connected."""
        from src.tello_source import TelloSource

        tello = TelloSource(self.config)
        # Don't call open()

        result = tello.takeoff()

        self.assertFalse(result)

    def test_land_returns_false_when_not_opened(self, mock_tello_class):
        """Test that land returns False when not connected."""
        from src.tello_source import TelloSource

        tello = TelloSource(self.config)
        # Don't call open()

        result = tello.land()

        self.assertFalse(result)

    def test_emergency_returns_false_when_not_opened(self, mock_tello_class):
        """Test that emergency returns False when not connected."""
        from src.tello_source import TelloSource

        tello = TelloSource(self.config)
        # Don't call open()

        result = tello.emergency()

        self.assertFalse(result)


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTelloSourceImport))
    suite.addTest(unittest.makeSuite(TestTelloSource))
    suite.addTest(unittest.makeSuite(TestTelloEdgeCases))
    suite.addTest(unittest.makeSuite(TestTelloSafetyFeatures))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
