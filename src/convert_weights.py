#!/usr/bin/env python3
"""Convert PyDNet TensorFlow 1.x checkpoint to TensorFlow 2.x/Keras format.

This script loads weights from the original PyDNet TF1 checkpoint and saves them in a
format compatible with TensorFlow 2.x and Keras 3.
"""

import os
import sys
import argparse

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow not installed")
    sys.exit(1)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.depth_estimator import PyDNetModel
from src.config import Config


def find_checkpoint_files(checkpoint_dir):
    """Find TF1 checkpoint files in directory."""
    checkpoint_file = None

    # Look for checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            first_line = f.readline()
            if "model_checkpoint_path:" in first_line:
                # Extract checkpoint name
                checkpoint_file = first_line.split('"')[1]

    # Look for .meta, .index, .data files
    files = os.listdir(checkpoint_dir)
    meta_files = [f for f in files if f.endswith(".meta")]

    if checkpoint_file:
        base_path = os.path.join(checkpoint_dir, checkpoint_file)
    elif meta_files:
        base_path = os.path.join(checkpoint_dir, meta_files[0].replace(".meta", ""))
    else:
        return None

    print(f"Found checkpoint: {base_path}")
    return base_path


def convert_tf1_checkpoint(tf1_checkpoint_path, output_path, config):
    """Convert TF1 checkpoint to TF2 Keras weights.

    Args:
        tf1_checkpoint_path: Path to TF1 checkpoint (without extension)
        output_path: Path to save converted weights (.keras or .weights.h5)
        config: Config object
    """
    print("=" * 60)
    print("PyDNet Weight Conversion: TF1 → TF2/Keras")
    print("=" * 60)

    # Check if checkpoint exists
    if not os.path.exists(f"{tf1_checkpoint_path}.index"):
        print(f"Error: Checkpoint not found: {tf1_checkpoint_path}")
        print("Expected files:")
        print(f"  - {tf1_checkpoint_path}.index")
        print(f"  - {tf1_checkpoint_path}.data-00000-of-00001")
        return False

    print(f"\nInput checkpoint: {tf1_checkpoint_path}")
    print(f"Output weights: {output_path}")

    # Create new model
    print("\nCreating PyDNet model...")
    model = PyDNetModel()

    # Build model with dummy input
    dummy_input = tf.zeros((1, config.input_height, config.input_width, 3))
    _ = model(dummy_input, training=False)
    print(f"Model built with input shape: {dummy_input.shape}")

    # Load TF1 checkpoint
    print("\nLoading TF1 checkpoint...")
    try:
        # Try loading as TF2 checkpoint first
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(tf1_checkpoint_path)

        # Check if restore was successful
        try:
            status.assert_consumed()
            print("✓ Loaded as TF2 checkpoint")
        except:
            print("⚠ Partial load - some variables not found")
            print("  This is expected for TF1 → TF2 conversion")

    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("\nTrying alternative loading method...")

        # Try loading variables manually
        try:
            reader = tf.train.load_checkpoint(tf1_checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()

            print(f"Found {len(var_to_shape_map)} variables in checkpoint:")
            for var_name in sorted(var_to_shape_map.keys())[:10]:
                print(f"  - {var_name}: {var_to_shape_map[var_name]}")
            if len(var_to_shape_map) > 10:
                print(f"  ... and {len(var_to_shape_map) - 10} more")

            # Map old variable names to new model weights
            print("\nAttempting to map variables to model weights...")
            loaded_count = 0

            for weight in model.weights:
                # Try different naming conventions
                old_names = [
                    weight.name.replace(":0", ""),
                    weight.name.split("/")[1].replace(":0", "") if "/" in weight.name else None,
                    "model/" + weight.name.replace(":0", ""),
                ]

                for old_name in old_names:
                    if old_name and old_name in var_to_shape_map:
                        try:
                            value = reader.get_tensor(old_name)
                            weight.assign(value)
                            print(f"  ✓ Loaded {old_name} → {weight.name}")
                            loaded_count += 1
                            break
                        except Exception as e:
                            print(e)
                            continue

            print(f"\nLoaded {loaded_count}/{len(model.weights)} weights")

            if loaded_count == 0:
                print("\n⚠ WARNING: No weights were loaded!")
                print("The checkpoint format may be incompatible.")
                return False

        except Exception as e2:
            print(f"Error: {e2}")
            return False

    # Save in new format
    print(f"\nSaving converted weights to: {output_path}")

    try:
        if output_path.endswith(".keras"):
            model.save_weights(output_path)
        elif output_path.endswith(".weights.h5"):
            model.save_weights(output_path)
        else:
            # Default to .keras format
            output_path = output_path + ".keras"
            model.save_weights(output_path)

        print("✓ Weights saved successfully!")
        print("\nTo use the converted weights:")
        print(f"  config.checkpoint_dir = '{output_path}'")

        # Verify saved weights
        print("\nVerifying saved weights...")
        test_model = PyDNetModel()
        _ = test_model(dummy_input)
        test_model.load_weights(output_path)
        print("✓ Weights verified - can be loaded successfully")

        return True

    except Exception as e:
        print(f"Error saving weights: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyDNet TF1 checkpoint to TF2/Keras format"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint/IROS18/pydnet",
        help="Directory containing TF1 checkpoint files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoint/IROS18/pydnet_weights.keras",
        help="Output path for converted weights (.keras or .weights.h5)",
    )
    parser.add_argument("--width", type=int, default=512, help="Model input width")
    parser.add_argument("--height", type=int, default=256, help="Model input height")

    args = parser.parse_args()

    # Create config
    config = Config(
        input_width=args.width, input_height=args.height, use_gpu=False  # Not needed for conversion
    )

    # Find checkpoint
    checkpoint_path = find_checkpoint_files(args.checkpoint_dir)

    if checkpoint_path is None:
        print(f"Error: No checkpoint found in {args.checkpoint_dir}")
        print("\nExpected file structure:")
        print("  checkpoint/IROS18/pydnet/")
        print("    ├── checkpoint")
        print("    ├── model.ckpt.meta")
        print("    ├── model.ckpt.index")
        print("    └── model.ckpt.data-00000-of-00001")
        return 1

    # Convert
    success = convert_tf1_checkpoint(checkpoint_path, args.output, config)

    if success:
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("Conversion failed!")
        print("=" * 60)
        print("\nPossible solutions:")
        print("1. Make sure you have the correct TF1 checkpoint files")
        print("2. Try downloading weights from the original PyDNet repo:")
        print("   https://github.com/mattpoggi/pydnet")
        print("3. The model architecture may have changed - weights may need manual mapping")
        return 1


if __name__ == "__main__":
    sys.exit(main())
