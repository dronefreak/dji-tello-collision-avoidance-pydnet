"""Unit tests for depth_estimator module."""

import os
import sys
import unittest

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config

# Check if TensorFlow is available
try:
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
    from src.depth_estimator import DepthEstimator, PyDNetModel
except ImportError:
    TF_AVAILABLE = False
    DepthEstimator = None  # type: ignore
    PyDNetModel = None  # type: ignore


def skip_if_no_tensorflow(test_func):
    """Decorator to skip tests if TensorFlow is not available."""
    return unittest.skipIf(not TF_AVAILABLE, "TensorFlow not installed")(test_func)


@unittest.skipIf(not TF_AVAILABLE, "TensorFlow not installed")
class TestPyDNetModel(unittest.TestCase):
    """Test PyDNet model architecture."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = PyDNetModel()

    def test_model_creation(self):
        """Test model can be created."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.name, "pydnet")

    def test_model_forward_pass(self):
        """Test forward pass through the model."""
        # Create dummy input
        batch_size = 1
        height, width = 256, 512
        dummy_input = np.random.rand(batch_size, height, width, 3).astype(np.float32)

        # Forward pass
        outputs = self.model(dummy_input, training=False)

        # Check outputs
        self.assertIn("high", outputs)
        self.assertIn("quarter", outputs)
        self.assertIn("eighth", outputs)

        # Check output shapes
        self.assertEqual(outputs["high"].shape[0], batch_size)
        self.assertEqual(outputs["high"].shape[-1], 1)  # Single channel

    def test_output_shapes(self):
        """Test that output shapes are correct at different resolutions."""
        batch_size = 2
        height, width = 256, 512
        dummy_input = np.random.rand(batch_size, height, width, 3).astype(np.float32)

        outputs = self.model(dummy_input, training=False)

        # High resolution is H/2, W/2 (encoder goes 6 levels down, decoder 5 up)
        self.assertEqual(outputs["high"].shape[1], height // 2)
        self.assertEqual(outputs["high"].shape[2], width // 2)

        # Quarter and eighth should be smaller
        self.assertLess(outputs["quarter"].shape[1], outputs["high"].shape[1])
        self.assertLess(outputs["eighth"].shape[1], outputs["quarter"].shape[1])


@unittest.skipIf(not TF_AVAILABLE, "TensorFlow not installed")
class TestDepthEstimator(unittest.TestCase):
    """Test DepthEstimator wrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(input_width=512, input_height=256, resolution=1, use_gpu=False)
        self.estimator = DepthEstimator(self.config)

    def test_estimator_creation(self):
        """Test estimator can be created."""
        self.assertIsNotNone(self.estimator)
        self.assertIsNotNone(self.estimator.model)

    def test_input_shape(self):
        """Test input shape is set correctly."""
        expected_shape = (256, 512, 3)
        self.assertEqual(self.estimator.input_shape, expected_shape)

    def test_resolution_map(self):
        """Test resolution mapping."""
        self.assertEqual(self.estimator.resolution_map[1], "high")
        self.assertEqual(self.estimator.resolution_map[2], "quarter")
        self.assertEqual(self.estimator.resolution_map[3], "eighth")

    def test_predict_with_correct_shape(self):
        """Test prediction with correct input shape."""
        # Create dummy image
        image = np.random.rand(256, 512, 3).astype(np.float32)

        # Predict
        depth_map = self.estimator.predict(image)

        # Check output
        self.assertIsInstance(depth_map, np.ndarray)
        self.assertEqual(len(depth_map.shape), 2)  # Should be 2D
        self.assertEqual(depth_map.dtype, np.float32)

    def test_predict_with_wrong_shape(self):
        """Test prediction fails with wrong input shape."""
        # Wrong shape
        image = np.random.rand(128, 256, 3).astype(np.float32)

        with self.assertRaises(ValueError):
            self.estimator.predict(image)

    def test_predict_updates_inference_times(self):
        """Test that prediction updates inference time tracking."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        # Initial state
        self.assertEqual(len(self.estimator.inference_times), 0)

        # Run prediction
        self.estimator.predict(image)

        # Check timing was recorded
        self.assertGreater(len(self.estimator.inference_times), 0)
        self.assertGreater(self.estimator.get_last_inference_time(), 0)

    def test_get_fps(self):
        """Test FPS calculation."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        # Run a few predictions
        for _ in range(5):
            self.estimator.predict(image)

        fps = self.estimator.get_fps()
        self.assertGreater(fps, 0)
        self.assertIsInstance(fps, float)

    def test_reset_stats(self):
        """Test resetting statistics."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        # Run predictions
        self.estimator.predict(image)
        self.assertGreater(len(self.estimator.inference_times), 0)

        # Reset
        self.estimator.reset_stats()
        self.assertEqual(len(self.estimator.inference_times), 0)

    def test_callable_interface(self):
        """Test that estimator is callable."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        # Call as function
        depth_map = self.estimator(image)

        self.assertIsInstance(depth_map, np.ndarray)
        self.assertEqual(len(depth_map.shape), 2)

    def test_different_resolutions(self):
        """Test prediction at different resolutions."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        for resolution in [1, 2, 3]:
            config = Config(input_width=512, input_height=256, resolution=resolution, use_gpu=False)
            estimator = DepthEstimator(config)

            depth_map = estimator.predict(image)
            self.assertIsInstance(depth_map, np.ndarray)

    def test_inference_time_limit(self):
        """Test that inference times list is limited."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        # Run many predictions
        for _ in range(50):
            self.estimator.predict(image)

        # Check that list is capped
        self.assertLessEqual(len(self.estimator.inference_times), 30)

    def test_depth_map_range(self):
        """Test that depth map values are in valid range."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        depth_map = self.estimator.predict(image)

        # Depth values should be between 0 and 1 (sigmoid output)
        self.assertGreaterEqual(depth_map.min(), 0.0)
        self.assertLessEqual(depth_map.max(), 1.0)


@unittest.skipIf(not TF_AVAILABLE, "TensorFlow not installed")
class TestDepthEstimatorWithConfig(unittest.TestCase):
    """Test DepthEstimator with different configurations."""

    def test_cpu_mode(self):
        """Test estimator runs on CPU."""
        config = Config(use_gpu=False)
        estimator = DepthEstimator(config)

        image = np.random.rand(256, 512, 3).astype(np.float32)
        depth_map = estimator.predict(image)

        self.assertIsNotNone(depth_map)

    def test_custom_dimensions(self):
        """Test with custom input dimensions."""
        config = Config(input_width=640, input_height=320, use_gpu=False)
        estimator = DepthEstimator(config)

        image = np.random.rand(320, 640, 3).astype(np.float32)
        depth_map = estimator.predict(image)

        self.assertIsNotNone(depth_map)


@unittest.skipIf(not TF_AVAILABLE, "TensorFlow not installed")
class TestDepthEstimatorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(use_gpu=False)
        self.estimator = DepthEstimator(self.config)

    def test_all_zeros_image(self):
        """Test with all-black image."""
        image = np.zeros((256, 512, 3), dtype=np.float32)
        depth_map = self.estimator.predict(image)

        self.assertIsNotNone(depth_map)
        # Output is half input size due to network architecture
        self.assertEqual(depth_map.shape, (128, 256))

    def test_all_ones_image(self):
        """Test with all-white image."""
        image = np.ones((256, 512, 3), dtype=np.float32)
        depth_map = self.estimator.predict(image)

        self.assertIsNotNone(depth_map)
        # Output is half input size due to network architecture
        self.assertEqual(depth_map.shape, (128, 256))

    def test_predict_consistency(self):
        """Test that same input gives consistent output."""
        image = np.random.rand(256, 512, 3).astype(np.float32)

        depth1 = self.estimator.predict(image)
        depth2 = self.estimator.predict(image)

        # Should be identical (no dropout/randomness in inference)
        np.testing.assert_array_almost_equal(depth1, depth2, decimal=5)


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPyDNetModel))
    suite.addTest(unittest.makeSuite(TestDepthEstimator))
    suite.addTest(unittest.makeSuite(TestDepthEstimatorWithConfig))
    suite.addTest(unittest.makeSuite(TestDepthEstimatorEdgeCases))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
