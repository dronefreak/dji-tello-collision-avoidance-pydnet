"""Unit tests for utils module."""

import os
import sys
import unittest

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check if OpenCV is available (required for utils module)
try:
    import cv2

    CV2_AVAILABLE = True
    from src.utils import (apply_colormap, draw_center_region, draw_crosshair, draw_depth_stats,
                           draw_fps, find_max_depth_region, preprocess_image, resize_with_aspect_ratio,
                           visualize_depth)
except ImportError:
    CV2_AVAILABLE = False


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestApplyColormap(unittest.TestCase):
    """Test apply_colormap function."""

    def test_basic_colormap(self):
        """Test basic colormap application."""
        depth_map = np.random.rand(100, 100).astype(np.float32)
        colored = apply_colormap(depth_map)

        self.assertEqual(colored.shape, (100, 100, 3))
        self.assertEqual(colored.dtype, np.uint8)
        self.assertGreaterEqual(colored.min(), 0)
        self.assertLessEqual(colored.max(), 255)

    def test_different_colormaps(self):
        """Test different colormap options."""
        depth_map = np.random.rand(50, 50).astype(np.float32)

        colormaps = ["plasma", "viridis", "magma", "inferno", "turbo"]
        for cmap in colormaps:
            colored = apply_colormap(depth_map, colormap=cmap)
            self.assertEqual(colored.shape, (50, 50, 3))

    def test_invalid_colormap(self):
        """Test with invalid colormap name (should fallback to plasma)."""
        depth_map = np.random.rand(50, 50).astype(np.float32)
        colored = apply_colormap(depth_map, colormap="invalid_colormap")

        # Should still work (fallback to plasma)
        self.assertEqual(colored.shape, (50, 50, 3))

    def test_with_vmin_vmax(self):
        """Test with custom normalization range."""
        depth_map = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        colored = apply_colormap(depth_map, vmin=0.0, vmax=1.0)

        self.assertEqual(colored.shape, (2, 2, 3))

    def test_constant_depth_map(self):
        """Test with constant depth values."""
        depth_map = np.ones((50, 50), dtype=np.float32) * 0.5
        colored = apply_colormap(depth_map)

        # Should handle constant values without error
        self.assertEqual(colored.shape, (50, 50, 3))

    def test_extreme_values(self):
        """Test with extreme depth values."""
        depth_map = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        colored = apply_colormap(depth_map)

        self.assertEqual(colored.shape, (2, 2, 3))


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestPreprocessImage(unittest.TestCase):
    """Test preprocess_image function."""

    def test_basic_preprocessing(self):
        """Test basic image preprocessing."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        target_size = (256, 512)

        preprocessed = preprocess_image(image, target_size, normalize=True)

        self.assertEqual(preprocessed.shape, (256, 512, 3))
        self.assertEqual(preprocessed.dtype, np.float32)
        self.assertGreaterEqual(preprocessed.min(), 0.0)
        self.assertLessEqual(preprocessed.max(), 1.0)

    def test_without_normalization(self):
        """Test preprocessing without normalization."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        target_size = (256, 512)

        preprocessed = preprocess_image(image, target_size, normalize=False)

        self.assertEqual(preprocessed.dtype, np.float32)
        self.assertGreater(preprocessed.max(), 1.0)  # Not normalized

    def test_grayscale_to_rgb(self):
        """Test conversion of grayscale to RGB."""
        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        target_size = (256, 512)

        preprocessed = preprocess_image(gray_image, target_size)

        self.assertEqual(preprocessed.shape, (256, 512, 3))

    def test_rgba_to_rgb(self):
        """Test conversion of RGBA to RGB."""
        rgba_image = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        target_size = (256, 512)

        preprocessed = preprocess_image(rgba_image, target_size)

        self.assertEqual(preprocessed.shape, (256, 512, 3))

    def test_different_target_sizes(self):
        """Test with different target sizes."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        sizes = [(128, 256), (256, 512), (512, 1024)]
        for target_size in sizes:
            preprocessed = preprocess_image(image, target_size)
            self.assertEqual(preprocessed.shape[:2], target_size)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestVisualizeDepth(unittest.TestCase):
    """Test visualize_depth function."""

    def test_vertical_stack(self):
        """Test vertical stacking of image and depth."""
        image = np.random.rand(256, 512, 3).astype(np.float32)
        depth_map = np.random.rand(256, 512).astype(np.float32)

        vis = visualize_depth(image, depth_map, stack_vertical=True)

        # Should be stacked vertically (height doubled)
        self.assertEqual(vis.shape, (512, 512, 3))
        self.assertEqual(vis.dtype, np.uint8)

    def test_horizontal_stack(self):
        """Test horizontal stacking of image and depth."""
        image = np.random.rand(256, 512, 3).astype(np.float32)
        depth_map = np.random.rand(256, 512).astype(np.float32)

        vis = visualize_depth(image, depth_map, stack_vertical=False)

        # Should be stacked horizontally (width doubled)
        self.assertEqual(vis.shape, (256, 1024, 3))

    def test_with_uint8_image(self):
        """Test with uint8 image input."""
        image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
        depth_map = np.random.rand(256, 512).astype(np.float32)

        vis = visualize_depth(image, depth_map)

        self.assertEqual(vis.dtype, np.uint8)

    def test_different_colormaps(self):
        """Test with different colormaps."""
        image = np.random.rand(256, 512, 3).astype(np.float32)
        depth_map = np.random.rand(256, 512).astype(np.float32)

        for cmap in ["plasma", "viridis", "magma"]:
            vis = visualize_depth(image, depth_map, colormap=cmap)
            self.assertIsNotNone(vis)

    def test_mismatched_sizes(self):
        """Test with mismatched image and depth sizes."""
        image = np.random.rand(256, 512, 3).astype(np.float32)
        depth_map = np.random.rand(128, 256).astype(np.float32)

        # Should resize depth to match image
        vis = visualize_depth(image, depth_map)
        self.assertIsNotNone(vis)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestDrawFunctions(unittest.TestCase):
    """Test drawing utility functions."""

    def test_draw_fps(self):
        """Test FPS drawing."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        fps = 30.5

        result = draw_fps(image, fps)

        self.assertEqual(result.shape, image.shape)
        self.assertIsNot(result, image)  # Should be a copy

    def test_draw_depth_stats(self):
        """Test depth statistics drawing."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = np.random.rand(256, 512).astype(np.float32)

        result = draw_depth_stats(image, depth_map)

        self.assertEqual(result.shape, image.shape)

    def test_draw_crosshair(self):
        """Test crosshair drawing."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        position = (320, 240)

        result = draw_crosshair(image, position, size=20)

        self.assertEqual(result.shape, image.shape)
        # Check that something was drawn (non-zero pixels)
        self.assertGreater(np.sum(result), 0)

    def test_draw_center_region(self):
        """Test center region drawing."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_center_region(image, tolerance=0.2)

        self.assertEqual(result.shape, image.shape)
        self.assertGreater(np.sum(result), 0)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestFindMaxDepthRegion(unittest.TestCase):
    """Test find_max_depth_region function."""

    def test_basic_functionality(self):
        """Test basic max depth region finding."""
        # Create depth map with known max region
        depth_map = np.zeros((200, 200), dtype=np.float32)
        depth_map[80:120, 80:120] = 1.0  # Max region in center

        max_pos = find_max_depth_region(depth_map, window_size=(50, 50))

        # Should find position near center
        self.assertIsInstance(max_pos, tuple)
        self.assertEqual(len(max_pos), 2)

    def test_with_different_window_sizes(self):
        """Test with different window sizes."""
        depth_map = np.random.rand(200, 200).astype(np.float32)

        window_sizes = [(30, 30), (50, 50), (100, 100)]
        for window_size in window_sizes:
            max_pos = find_max_depth_region(depth_map, window_size=window_size)
            self.assertIsInstance(max_pos, tuple)

    def test_uniform_depth(self):
        """Test with uniform depth map."""
        depth_map = np.ones((200, 200), dtype=np.float32) * 0.5

        max_pos = find_max_depth_region(depth_map)

        # Should return some valid position
        self.assertIsInstance(max_pos, tuple)
        self.assertEqual(len(max_pos), 2)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestResizeWithAspectRatio(unittest.TestCase):
    """Test resize_with_aspect_ratio function."""

    def test_resize_by_width(self):
        """Test resizing by target width."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        target_width = 320

        resized = resize_with_aspect_ratio(image, target_width=target_width)

        self.assertEqual(resized.shape[1], target_width)
        # Check aspect ratio maintained
        expected_height = int(480 * (320 / 640))
        self.assertEqual(resized.shape[0], expected_height)

    def test_resize_by_height(self):
        """Test resizing by target height."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        target_height = 240

        resized = resize_with_aspect_ratio(image, target_height=target_height)

        self.assertEqual(resized.shape[0], target_height)
        # Check aspect ratio maintained
        expected_width = int(640 * (240 / 480))
        self.assertEqual(resized.shape[1], expected_width)

    def test_no_target_returns_original(self):
        """Test that no target returns original image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        resized = resize_with_aspect_ratio(image)

        np.testing.assert_array_equal(resized, image)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not installed")
class TestUtilsEdgeCases(unittest.TestCase):
    """Test edge cases for utility functions."""

    def test_zero_depth_map(self):
        """Test with all-zero depth map."""
        depth_map = np.zeros((100, 100), dtype=np.float32)
        colored = apply_colormap(depth_map)

        self.assertEqual(colored.shape, (100, 100, 3))

    def test_single_pixel_image(self):
        """Test with minimal image size."""
        image = np.random.rand(1, 1, 3).astype(np.float32)
        depth_map = np.random.rand(1, 1).astype(np.float32)

        vis = visualize_depth(image, depth_map)
        self.assertIsNotNone(vis)

    def test_large_depth_values(self):
        """Test with very large depth values."""
        depth_map = np.random.rand(100, 100).astype(np.float32) * 1000
        colored = apply_colormap(depth_map)

        self.assertEqual(colored.shape, (100, 100, 3))


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestApplyColormap))
    suite.addTest(unittest.makeSuite(TestPreprocessImage))
    suite.addTest(unittest.makeSuite(TestVisualizeDepth))
    suite.addTest(unittest.makeSuite(TestDrawFunctions))
    suite.addTest(unittest.makeSuite(TestFindMaxDepthRegion))
    suite.addTest(unittest.makeSuite(TestResizeWithAspectRatio))
    suite.addTest(unittest.makeSuite(TestUtilsEdgeCases))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
