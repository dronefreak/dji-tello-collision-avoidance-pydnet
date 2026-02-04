"""Unit tests for webcam_demo functionality.

Tests the webcam source and integration without requiring actual hardware.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check if OpenCV is available (required for webcam_source)
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.camera_interface import CameraInterface
from src.config import Config

if CV2_AVAILABLE:
    from src.webcam_source import WebcamSource


class TestCameraInterface(unittest.TestCase):
    """Test abstract CameraInterface base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with self.assertRaises(TypeError):
            CameraInterface()

    def test_frame_count_initialization(self):
        """Test frame count is initialized to zero."""

        # Create a concrete implementation for testing
        class TestCamera(CameraInterface):
            def open(self):
                return True

            def read(self):
                return True, np.zeros((480, 640, 3))

            def release(self):
                pass

            def is_opened(self):
                return True

            def get_frame_size(self):
                return (480, 640)

        camera = TestCamera()
        self.assertEqual(camera.get_frame_count(), 0)

    def test_reset_frame_count(self):
        """Test resetting frame count."""

        class TestCamera(CameraInterface):
            def open(self):
                self._frame_count = 10
                return True

            def read(self):
                return True, np.zeros((480, 640, 3))

            def release(self):
                pass

            def is_opened(self):
                return True

            def get_frame_size(self):
                return (480, 640)

        camera = TestCamera()
        camera.open()
        self.assertEqual(camera.get_frame_count(), 10)
        camera.reset_frame_count()
        self.assertEqual(camera.get_frame_count(), 0)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestWebcamSource(unittest.TestCase):
    """Test WebcamSource class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(camera_id=0)

    def test_webcam_creation(self):
        """Test webcam source can be created."""
        webcam = WebcamSource(self.config)
        self.assertIsNotNone(webcam)
        self.assertEqual(webcam.camera_id, 0)

    def test_initial_state(self):
        """Test initial state is closed."""
        webcam = WebcamSource(self.config)
        self.assertFalse(webcam.is_opened())

    @patch("cv2.VideoCapture")
    def test_open_success(self, mock_video_capture):
        """Test successful webcam opening."""
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 640,  # Width
            4: 480,  # Height
            5: 30,  # FPS
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        result = webcam.open()

        self.assertTrue(result)
        self.assertTrue(webcam.is_opened())

    @patch("cv2.VideoCapture")
    def test_open_failure(self, mock_video_capture):
        """Test failed webcam opening."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        result = webcam.open()

        self.assertFalse(result)
        self.assertFalse(webcam.is_opened())

    @patch("cv2.VideoCapture")
    def test_read_frame(self, mock_video_capture):
        """Test reading a frame."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640

        # Mock frame read
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        webcam.open()

        success, frame = webcam.read()

        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

    @patch("cv2.VideoCapture")
    def test_read_failure(self, mock_video_capture):
        """Test failed frame read."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        webcam.open()

        success, frame = webcam.read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    @patch("cv2.VideoCapture")
    def test_release(self, mock_video_capture):
        """Test releasing webcam."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        webcam.open()
        self.assertTrue(webcam.is_opened())

        webcam.release()
        self.assertFalse(webcam.is_opened())
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_get_frame_size(self, mock_video_capture):
        """Test getting frame size."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 1920, 4: 1080}.get(prop, 0)  # Width  # Height
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        webcam.open()

        height, width = webcam.get_frame_size()

        self.assertEqual(height, 1080)
        self.assertEqual(width, 1920)

    @patch("cv2.VideoCapture")
    def test_context_manager(self, mock_video_capture):
        """Test using webcam as context manager."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_video_capture.return_value = mock_cap

        config = Config(camera_id=0)

        with WebcamSource(config) as webcam:
            self.assertTrue(webcam.is_opened())

        # Should be released after exiting context
        mock_cap.release.assert_called()

    @patch("cv2.VideoCapture")
    def test_frame_count_increment(self, mock_video_capture):
        """Test frame count increments on read."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        webcam.open()

        self.assertEqual(webcam.get_frame_count(), 0)

        webcam.read()
        self.assertEqual(webcam.get_frame_count(), 1)

        webcam.read()
        webcam.read()
        self.assertEqual(webcam.get_frame_count(), 3)

    @patch("cv2.VideoCapture")
    def test_get_camera_info(self, mock_video_capture):
        """Test getting camera information."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640, 4: 480, 5: 30}.get(prop, 0)
        mock_cap.getBackendName.return_value = "V4L2"
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        webcam.open()

        info = webcam.get_camera_info()

        self.assertIn("camera_id", info)
        self.assertIn("width", info)
        self.assertIn("height", info)
        self.assertIn("fps", info)
        self.assertIn("backend", info)
        self.assertEqual(info["camera_id"], 0)

    @patch("cv2.VideoCapture")
    def test_set_camera_id(self, mock_video_capture):
        """Test changing camera ID."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_video_capture.return_value = mock_cap

        webcam = WebcamSource(self.config)
        webcam.open()

        result = webcam.set_camera_id(1)

        self.assertTrue(result)
        self.assertEqual(webcam.camera_id, 1)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestWebcamIntegration(unittest.TestCase):
    """Integration tests for webcam functionality."""

    @patch("cv2.VideoCapture")
    @patch("cv2.cvtColor")
    def test_full_capture_cycle(self, mock_cvt_color, mock_video_capture):
        """Test a full capture cycle."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640

        # Create test frames
        test_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        mock_cap.read.side_effect = [(True, frame) for frame in test_frames]
        mock_cvt_color.side_effect = lambda img, _: img  # Pass through
        mock_video_capture.return_value = mock_cap

        config = Config(camera_id=0)
        webcam = WebcamSource(config)

        # Open
        self.assertTrue(webcam.open())

        # Read multiple frames
        frames_read = 0
        for _ in range(5):
            success, frame = webcam.read()
            if success:
                frames_read += 1

        # Release
        webcam.release()

        self.assertEqual(frames_read, 5)
        self.assertEqual(webcam.get_frame_count(), 5)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestWebcamEdgeCases(unittest.TestCase):
    """Test edge cases for webcam source."""

    def test_read_without_opening(self):
        """Test reading before opening camera."""
        config = Config()
        webcam = WebcamSource(config)

        success, frame = webcam.read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    def test_double_open(self):
        """Test opening camera twice."""
        config = Config()
        webcam = WebcamSource(config)

        with patch("cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 640
            mock_vc.return_value = mock_cap

            webcam.open()
            # Open again - should handle gracefully
            webcam.open()

    def test_double_release(self):
        """Test releasing camera twice."""
        config = Config()
        webcam = WebcamSource(config)

        with patch("cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 640
            mock_vc.return_value = mock_cap

            webcam.open()
            webcam.release()
            # Release again - should handle gracefully
            webcam.release()


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCameraInterface))
    suite.addTest(unittest.makeSuite(TestWebcamSource))
    suite.addTest(unittest.makeSuite(TestWebcamIntegration))
    suite.addTest(unittest.makeSuite(TestWebcamEdgeCases))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
