#!/usr/bin/env python3
"""Webcam Demo for Depth Estimation.

Test the depth estimation system using a webcam without needing a drone. Press 'q' or
ESC to quit, 'p' to pause, 's' to save screenshot.
"""

import argparse
import os
import sys

import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.collision_avoidance import CollisionAvoidance
from src.config import Config
from src.depth_estimator import DepthEstimator
from src.utils import draw_fps, preprocess_image, visualize_depth
from src.webcam_source import WebcamSource


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Webcam depth estimation demo")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint/IROS18/pydnet",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Resolution: 1=High, 2=Quarter, 3=Eighth",
    )
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=512, help="Input width for model")
    parser.add_argument("--height", type=int, default=256, help="Input height for model")
    parser.add_argument(
        "--colormap",
        type=str,
        default="plasma",
        choices=["plasma", "viridis", "magma", "inferno", "turbo"],
        help="Colormap for depth visualization",
    )
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU and run on CPU only")
    parser.add_argument(
        "--show_analysis", action="store_true", help="Show collision avoidance analysis"
    )
    parser.add_argument("--save_output", action="store_true", help="Save output frames to disk")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create configuration
    config = Config(
        checkpoint_dir=args.checkpoint_dir,
        resolution=args.resolution,
        camera_id=args.camera_id,
        input_width=args.width,
        input_height=args.height,
        colormap=args.colormap,
        use_gpu=not args.no_gpu,
        save_output=args.save_output,
    )

    print("=" * 60)
    print("Webcam Depth Estimation Demo")
    print("=" * 60)
    print(f"Model: PyDNet (Resolution: {config.resolution})")
    print(f"Input size: {config.input_width}x{config.input_height}")
    print(f"Camera: {config.camera_id}")
    print(f"Device: {'GPU' if config.use_gpu else 'CPU'}")
    print("=" * 60)
    print("\nControls:")
    print("  q/ESC - Quit")
    print("  p     - Pause/Resume")
    print("  s     - Save screenshot")
    print("  a     - Toggle analysis overlay")
    print("=" * 60)

    # Initialize components
    try:
        print("\nInitializing depth estimator...")
        depth_estimator = DepthEstimator(config)

        # Try to load weights (optional for demo)
        try:
            depth_estimator.load_weights()
        except FileNotFoundError:
            print("\nWarning: No pretrained weights found.")
            print("The model will run with random weights (for testing structure only).")
            print("To use pretrained weights, download them and place in:", config.checkpoint_dir)

        print("\nOpening webcam...")
        camera = WebcamSource(config)

        if not camera.open():
            print("Error: Could not open webcam!")
            return 1

        # Optional collision avoidance
        collision_avoidance = None
        show_analysis = args.show_analysis
        if show_analysis:
            collision_avoidance = CollisionAvoidance(config)
            print("Collision avoidance analysis enabled")

        print("\nStarting capture... Press 'q' to quit\n")

        # Main loop
        paused = False
        frame_count = 0

        while True:
            if not paused:
                # Read frame
                success, frame = camera.read()

                if not success:
                    print("Error reading frame")
                    break

                # Preprocess for depth estimation
                input_img = preprocess_image(
                    frame, (config.input_height, config.input_width), normalize=True
                )

                # Estimate depth
                depth_map = depth_estimator.predict(input_img)

                # Visualize
                vis_frame = visualize_depth(
                    input_img,
                    depth_map,
                    colormap=config.colormap,
                    depth_scale=config.depth_scale,
                    stack_vertical=True,
                )

                # Add FPS
                fps = depth_estimator.get_fps()
                vis_frame = draw_fps(vis_frame, fps)

                # Add collision avoidance analysis if enabled
                if show_analysis and collision_avoidance:
                    analysis = collision_avoidance.analyze_depth(depth_map)

                    # Draw analysis text
                    y_offset = 60
                    cv2.putText(
                        vis_frame,
                        f"Action: {analysis['suggested_action']}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        vis_frame,
                        f"Center Depth: {analysis['center_depth']:.3f}",
                        (10, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        vis_frame,
                        f"Safe: {'Yes' if analysis['is_safe'] else 'No'}",
                        (10, y_offset + 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if analysis["is_safe"] else (0, 0, 255),
                        1,
                    )

                # Save if enabled
                if config.save_output:
                    filename = os.path.join(config.output_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(filename, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

                frame_count += 1

            # Display
            cv2.imshow("Depth Estimation - Webcam Demo", cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("p"):  # Pause
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord("s"):  # Save screenshot
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                print(f"Saved screenshot: {filename}")
            elif key == ord("a"):  # Toggle analysis
                show_analysis = not show_analysis
                if show_analysis and collision_avoidance is None:
                    collision_avoidance = CollisionAvoidance(config)
                print(f"Analysis: {'ON' if show_analysis else 'OFF'}")

        # Cleanup
        print("\nCleaning up...")
        camera.release()
        cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames")
        print(f"Average FPS: {depth_estimator.get_fps():.2f}")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
