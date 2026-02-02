"""Depth Estimation using PyDNet.

This module implements the PyDNet depth estimation network, refactored for
TensorFlow 2.x. Based on the paper "Towards real-time unsupervised monocular
depth estimation on CPU" (IROS 2018).

Original paper: https://arxiv.org/abs/1806.11430
"""

import numpy as np
import tensorflow as tf
from collections import deque
from typing import Optional
import time

from .config import Config


class PyDNetModel(tf.keras.Model):
    """PyDNet model architecture."""

    def __init__(self, name="pydnet"):
        super(PyDNetModel, self).__init__(name=name)

        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")
        self.conv4 = tf.keras.layers.Conv2D(96, 3, strides=2, padding="same", activation="relu")
        self.conv5 = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")
        self.conv6 = tf.keras.layers.Conv2D(192, 3, strides=2, padding="same", activation="relu")

        # Decoder - upsampling path
        self.upconv5 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same")
        self.iconv5 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")

        self.upconv4 = tf.keras.layers.Conv2DTranspose(96, 3, strides=2, padding="same")
        self.iconv4 = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu")

        self.upconv3 = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same")
        self.iconv3 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")

        self.upconv2 = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")
        self.iconv2 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")

        self.upconv1 = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same")
        self.iconv1 = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")

        # Disparity outputs at multiple scales
        self.disp4 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")
        self.disp3 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")
        self.disp2 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")
        self.disp1 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")

    def call(self, inputs, training=False):
        """Forward pass through the network."""
        # Encoder
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        # Decoder with skip connections
        upconv5 = self.upconv5(conv6)
        concat5 = tf.concat([upconv5, conv5], axis=-1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = tf.concat([upconv4, conv4], axis=-1)
        iconv4 = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4)
        concat3 = tf.concat([upconv3, conv3], axis=-1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3(iconv3)

        upconv2 = self.upconv2(iconv3)
        concat2 = tf.concat([upconv2, conv2], axis=-1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2(iconv2)

        upconv1 = self.upconv1(iconv2)
        concat1 = tf.concat([upconv1, conv1], axis=-1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1(iconv1)

        # Return multiple resolution outputs
        return {
            "high": disp1,  # Full resolution (H)
            "quarter": disp2,  # Quarter resolution (Q)
            "eighth": disp3,  # Eighth resolution (E)
        }


class DepthEstimator:
    """Depth estimation wrapper for PyDNet.

    Handles model loading, inference, and output processing.
    """

    def __init__(self, config: Config):
        """Initialize the depth estimator.

        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        self.model: Optional[PyDNetModel] = None
        self.input_shape = (config.input_height, config.input_width, 3)
        self.resolution_map = {1: "high", 2: "quarter", 3: "eighth"}
        # Use deque with maxlen for O(1) append and automatic size limiting
        self._max_inference_samples = 30
        self.inference_times: deque = deque(maxlen=self._max_inference_samples)

        # Configure GPU if available
        self._configure_gpu()

        # Build and load model
        self._build_model()

    def _configure_gpu(self):
        """Configure GPU settings."""
        gpus = tf.config.list_physical_devices("GPU")

        if gpus and self.config.use_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Set memory limit if specified
                if self.config.gpu_memory_fraction < 1.0:
                    memory_limit = int(
                        tf.config.experimental.get_memory_info("GPU:0")["current"]
                        * self.config.gpu_memory_fraction
                    )
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ],
                    )
                print(f"GPU configured: {len(gpus)} device(s) available")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("Running on CPU")

    def _build_model(self):
        """Build and initialize the model."""
        self.model = PyDNetModel()

        # Build the model by running a dummy forward pass
        dummy_input = tf.zeros((1, self.config.input_height, self.config.input_width, 3))
        _ = self.model(dummy_input, training=False)

        print(f"Model initialized with input shape: {self.input_shape}")

    def load_weights(self, checkpoint_path: Optional[str] = None):
        """Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file. Uses config default if None.
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_dir

        if not self.config.validate_checkpoint():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. " "Please download the model weights."
            )

        try:
            self.model.load_weights(checkpoint_path)
            print(f"Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {checkpoint_path}: {e}")
            print("Running with randomly initialized weights (for testing only)")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from an input image.

        Args:
            image: Input image as numpy array (H, W, 3), normalized to [0, 1]

        Returns:
            Depth map as numpy array (H, W)
        """
        if image.shape[:2] != (self.config.input_height, self.config.input_width):
            raise ValueError(
                f"Image shape {image.shape[:2]} does not match expected "
                f"{(self.config.input_height, self.config.input_width)}"
            )

        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)

        # Inference
        start_time = time.time()
        outputs = self.model(image_batch, training=False)
        inference_time = time.time() - start_time

        # Track inference times for FPS calculation (deque handles size limiting)
        self.inference_times.append(inference_time)

        # Get output at specified resolution
        resolution_key = self.resolution_map[self.config.resolution]
        depth_map = outputs[resolution_key].numpy()[0, :, :, 0]

        return depth_map

    def get_fps(self) -> float:
        """Get average FPS over recent inferences."""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_last_inference_time(self) -> float:
        """Get the last inference time in seconds."""
        return self.inference_times[-1] if self.inference_times else 0.0

    def reset_stats(self):
        """Reset inference time statistics."""
        self.inference_times.clear()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Make the estimator callable."""
        return self.predict(image)
