"""Utility functions for image processing and visualization.

Provides functions for colormap application, image preprocessing, and depth map
visualization.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from matplotlib import cm


def apply_colormap(
    depth_map: np.ndarray,
    colormap: str = "plasma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Apply a colormap to a depth map for visualization.

    Args:
        depth_map: Depth map as 2D numpy array
        colormap: Name of matplotlib colormap (plasma, viridis, magma, inferno, turbo)
        vmin: Minimum value for normalization (None for auto)
        vmax: Maximum value for normalization (None for auto)

    Returns:
        Colored depth map as RGB numpy array (H, W, 3) with values in [0, 255]
    """
    # Normalize depth map
    if vmin is None:
        vmin = depth_map.min()
    if vmax is None:
        vmax = depth_map.max()

    # Avoid division by zero
    if vmax - vmin < 1e-6:
        normalized = np.zeros_like(depth_map)
    else:
        normalized = (depth_map - vmin) / (vmax - vmin)

    # Clip to [0, 1]
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    try:
        cmap = cm.get_cmap(colormap)
    except ValueError:
        print(f"Warning: Unknown colormap '{colormap}', using 'plasma'")
        cmap = cm.get_cmap("plasma")

    colored = cmap(normalized)

    # Convert to RGB (0-255)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return colored_rgb


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int], normalize: bool = True
) -> np.ndarray:
    """Preprocess image for depth estimation.

    Args:
        image: Input image as numpy array (H, W, 3)
        target_size: Target size as (height, width)
        normalize: Whether to normalize to [0, 1]

    Returns:
        Preprocessed image as numpy array
    """
    # Resize image
    height, width = target_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Ensure RGB format
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)

    # Convert to float32
    preprocessed = resized.astype(np.float32)

    # Normalize to [0, 1]
    if normalize:
        preprocessed = preprocessed / 255.0

    return preprocessed


def visualize_depth(
    image: np.ndarray,
    depth_map: np.ndarray,
    colormap: str = "plasma",
    depth_scale: float = 20.0,
    stack_vertical: bool = True,
) -> np.ndarray:
    """Create a visualization combining original image and colored depth map.

    Args:
        image: Original image (H, W, 3) in [0, 1] or [0, 255]
        depth_map: Depth map (H, W)
        colormap: Colormap name
        depth_scale: Scaling factor for depth visualization
        stack_vertical: If True, stack vertically; otherwise horizontally

    Returns:
        Combined visualization as numpy array (uint8)
    """
    # Ensure image is uint8
    if image.max() <= 1.0:
        img_vis = (image * 255).astype(np.uint8)
    else:
        img_vis = image.astype(np.uint8)

    # Resize depth map to match image dimensions if needed
    if depth_map.shape[:2] != img_vis.shape[:2]:
        depth_map = cv2.resize(depth_map, (img_vis.shape[1], img_vis.shape[0]))

    # Apply colormap to depth
    depth_colored = apply_colormap(depth_map * depth_scale, colormap=colormap)

    # Stack images
    if stack_vertical:
        combined = np.vstack([img_vis, depth_colored])
    else:
        combined = np.hstack([img_vis, depth_colored])

    return combined


def draw_fps(image: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw FPS counter on image.

    Args:
        image: Input image
        fps: FPS value to display
        position: Text position (x, y)

    Returns:
        Image with FPS counter
    """
    img_copy = image.copy()
    text = f"FPS: {fps:.1f}"

    cv2.putText(
        img_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
    )

    return img_copy


def draw_depth_stats(
    image: np.ndarray, depth_map: np.ndarray, position: Tuple[int, int] = (10, 60)
) -> np.ndarray:
    """Draw depth statistics on image.

    Args:
        image: Input image
        depth_map: Depth map
        position: Text position (x, y)

    Returns:
        Image with depth statistics
    """
    img_copy = image.copy()

    min_depth = depth_map.min()
    max_depth = depth_map.max()
    mean_depth = depth_map.mean()

    text = f"Depth: min={min_depth:.3f} max={max_depth:.3f} mean={mean_depth:.3f}"

    cv2.putText(
        img_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    return img_copy


def find_max_depth_region(
    depth_map: np.ndarray, window_size: Tuple[int, int] = (50, 50)
) -> Tuple[int, int]:
    """Find the region with maximum depth using a sliding window.

    Uses cv2.blur() for efficient O(n) mean computation instead of naive O(n²)
    nested loop approach.

    Args:
        depth_map: Depth map (H, W)
        window_size: Size of the sliding window (height, width)

    Returns:
        Center coordinates (y, x) of the region with maximum depth
    """
    h, w = depth_map.shape
    win_h, win_w = window_size

    # Handle edge case where window is larger than image
    if win_h >= h or win_w >= w:
        return (h // 2, w // 2)

    # Use cv2.blur for efficient mean filter computation
    # cv2.blur computes the mean over the window for every pixel
    # This is O(n) due to separable filter optimization
    depth_float = depth_map.astype(np.float32)
    mean_map = cv2.blur(depth_float, (win_w, win_h))

    # Find the position of maximum mean depth
    # Exclude borders where the window would extend outside the image
    border_h, border_w = win_h // 2, win_w // 2
    valid_region = mean_map[border_h : h - border_h, border_w : w - border_w]

    if valid_region.size == 0:
        return (h // 2, w // 2)

    # Find max position in valid region
    max_idx = np.argmax(valid_region)
    local_y, local_x = np.unravel_index(max_idx, valid_region.shape)

    # Convert back to full image coordinates
    max_y = local_y + border_h
    max_x = local_x + border_w

    return (int(max_y), int(max_x))


def draw_crosshair(
    image: np.ndarray,
    position: Tuple[int, int],
    size: int = 20,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a crosshair at specified position.

    Args:
        image: Input image
        position: Crosshair center (x, y)
        size: Crosshair arm length
        color: Line color (B, G, R)
        thickness: Line thickness

    Returns:
        Image with crosshair
    """
    img_copy = image.copy()
    x, y = position

    # Horizontal line
    cv2.line(img_copy, (x - size, y), (x + size, y), color, thickness)
    # Vertical line
    cv2.line(img_copy, (x, y - size), (x, y + size), color, thickness)

    return img_copy


def draw_center_region(
    image: np.ndarray,
    tolerance: float = 0.2,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangle indicating the center tolerance region.

    Args:
        image: Input image
        tolerance: Tolerance as fraction of image width
        color: Rectangle color (B, G, R)
        thickness: Line thickness

    Returns:
        Image with center region
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    center_x = w // 2
    tolerance_pixels = int(w * tolerance)

    x1 = center_x - tolerance_pixels
    x2 = center_x + tolerance_pixels

    cv2.rectangle(img_copy, (x1, 0), (x2, h), color, thickness)

    return img_copy


def save_image(image: np.ndarray, filename: str) -> bool:
    """Save image to file.

    Args:
        image: Image to save
        filename: Output filename

    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(filename, image)
        return True
    except Exception as e:
        print(f"Error saving image to {filename}: {e}")
        return False


def resize_with_aspect_ratio(
    image: np.ndarray, target_width: Optional[int] = None, target_height: Optional[int] = None
) -> np.ndarray:
    """Resize image maintaining aspect ratio.

    Args:
        image: Input image
        target_width: Target width (height computed to maintain aspect ratio)
        target_height: Target height (width computed to maintain aspect ratio)

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if target_width is not None:
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
    elif target_height is not None:
        scale = target_height / h
        new_h = target_height
        new_w = int(w * scale)
    else:
        return image

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized
