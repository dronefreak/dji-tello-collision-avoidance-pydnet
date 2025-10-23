#!/usr/bin/env python3
"""Tello Drone Demo with Depth Estimation.

Run depth estimation on DJI Tello drone video stream with optional collision avoidance.
Commands are disabled by default for safety.

Press 'q' or ESC to quit, 't' to takeoff, 'l' to land, 'e' for emergency stop.
"""

import argparse
import os
import sys
import time

import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.collision_avoidance import CollisionAvoidance
from src.config import Config
from src.depth_estimator import DepthEstimator
from src.tello_source import TelloSource
from src.utils import draw_fps, preprocess_image, visualize_depth


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tello drone depth estimation demo")

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
        "--enable_commands",
        action="store_true",
        help="Enable collision avoidance commands (USE WITH CAUTION)",
    )
    parser.add_argument(
        "--enable_auto_flight",
        action="store_true",
        help="Enable autonomous flight with collision avoidance (DANGEROUS)",
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
        input_width=args.width,
        input_height=args.height,
        colormap=args.colormap,
        use_gpu=not args.no_gpu,
        tello_enable_commands=args.enable_commands,
        enable_collision_avoidance=args.enable_auto_flight,
        save_output=args.save_output,
    )

    print("=" * 60)
    print("Tello Drone Depth Estimation Demo")
    print("=" * 60)
    print(f"Model: PyDNet (Resolution: {config.resolution})")
    print(f"Input size: {config.input_width}x{config.input_height}")
    print(f"Device: {'GPU' if config.use_gpu else 'CPU'}")
    print(f"Commands: {'ENABLED' if config.tello_enable_commands else 'DISABLED'}")
    print(f"Auto-flight: {'ENABLED' if config.enable_collision_avoidance else 'DISABLED'}")
    print("=" * 60)

    if config.enable_collision_avoidance:
        print("\n⚠️  WARNING: Autonomous flight is ENABLED!")
        print("The drone will navigate automatically based on depth estimation.")
        print("Press 'e' for emergency stop at any time!")
        response = input("\nType 'YES' to confirm and continue: ")
        if response != "YES":
            print("Aborted.")
            return 1

    print("\nControls:")
    print("  q/ESC - Quit")
    print("  t     - Takeoff")
    print("  l     - Land")
    print("  e     - Emergency stop (cuts motors)")
    print("  p     - Pause/Resume")
    print("  s     - Save screenshot")
    print("  b     - Show battery info")
    if config.tello_enable_commands:
        print("\n  Movement (only if commands enabled):")
        print("  w/s   - Forward/Backward")
        print("  a/d   - Left/Right")
        print("  up/down - Ascend/Descend")
        print("  left/right - Rotate")
    print("=" * 60)

    # Initialize components
    try:
        print("\nInitializing depth estimator...")
        depth_estimator = DepthEstimator(config)

        # Try to load weights
        try:
            depth_estimator.load_weights()
        except FileNotFoundError:
            print("\nWarning: No pretrained weights found.")
            print("Download weights and place in:", config.checkpoint_dir)
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                return 1

        print("\nConnecting to Tello...")
        drone = TelloSource(config)

        if not drone.open():
            print("Error: Could not connect to Tello!")
            print("Make sure:")
            print("  1. Tello is powered on")
            print("  2. You're connected to Tello's WiFi network")
            print("  3. djitellopy is installed (pip install djitellopy)")
            return 1

        # Initialize collision avoidance
        collision_avoidance = CollisionAvoidance(config)

        print("\nStarting video stream... Press 't' to takeoff\n")

        # Main loop
        paused = False
        auto_mode = config.enable_collision_avoidance
        frame_count = 0
        last_telemetry_time = time.time()

        while True:
            if not paused:
                # Read frame
                success, frame = drone.read()

                if not success:
                    print("Error reading frame")
                    time.sleep(0.1)
                    continue

                # Preprocess for depth estimation
                input_img = preprocess_image(
                    frame, (config.input_height, config.input_width), normalize=True
                )

                # Estimate depth
                depth_map = depth_estimator.predict(input_img)

                # Collision avoidance analysis
                analysis = collision_avoidance.analyze_depth(depth_map)

                # Auto-flight control
                if auto_mode and drone.is_flying():
                    rc_command = collision_avoidance.get_rc_command(depth_map)
                    drone.send_rc_control(*rc_command)

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

                # Add telemetry
                y_offset = 60
                cv2.putText(
                    vis_frame,
                    f"Battery: {drone.get_battery()}%",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis_frame,
                    f"Height: {drone.get_height()}cm",
                    (10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Add collision avoidance info
                cv2.putText(
                    vis_frame,
                    f"Action: {analysis['suggested_action']}",
                    (10, y_offset + 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                cv2.putText(
                    vis_frame,
                    f"Safe: {'Yes' if analysis['is_safe'] else 'No'}",
                    (10, y_offset + 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0) if analysis["is_safe"] else (0, 0, 255),
                    1,
                )

                if auto_mode:
                    cv2.putText(
                        vis_frame,
                        "AUTO MODE",
                        (vis_frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # Save if enabled
                if config.save_output:
                    filename = os.path.join(config.output_dir, f"tello_{frame_count:06d}.jpg")
                    cv2.imwrite(filename, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

                # Print telemetry periodically
                if time.time() - last_telemetry_time > 5.0:
                    telemetry = drone.get_telemetry()
                    print(
                        f"Battery: {telemetry['battery']}% | Height: {telemetry['height']}cm | "
                        f"Temp: {drone.get_temperature()}°C | Frames: {frame_count}"
                    )
                    last_telemetry_time = time.time()

                frame_count += 1

            # Display
            cv2.imshow("Depth Estimation - Tello Drone", cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Quit
                break
            elif key == ord("t"):  # Takeoff
                if not drone.is_flying():
                    drone.takeoff()
                    print("Taking off...")
            elif key == ord("l"):  # Land
                if drone.is_flying():
                    auto_mode = False  # Disable auto mode before landing
                    drone.send_rc_control(0, 0, 0, 0)  # Stop all movement
                    time.sleep(0.5)
                    drone.land()
                    print("Landing...")
            elif key == ord("e"):  # Emergency
                print("EMERGENCY STOP!")
                drone.emergency()
                break
            elif key == ord("p"):  # Pause
                paused = not paused
                if paused and drone.is_flying():
                    drone.send_rc_control(0, 0, 0, 0)
                print("Paused" if paused else "Resumed")
            elif key == ord("s"):  # Screenshot
                filename = f"tello_screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                print(f"Saved: {filename}")
            elif key == ord("b"):  # Battery
                telemetry = drone.get_telemetry()
                print(f"\nTelemetry: {telemetry}")

            # Manual controls (if enabled)
            if config.tello_enable_commands and not auto_mode and drone.is_flying():
                if key == ord("w"):
                    drone.send_rc_control(0, 30, 0, 0)
                elif key == ord("s"):
                    drone.send_rc_control(0, -30, 0, 0)
                elif key == ord("a"):
                    drone.send_rc_control(-30, 0, 0, 0)
                elif key == ord("d"):
                    drone.send_rc_control(30, 0, 0, 0)
                elif key == 82:  # Up arrow
                    drone.send_rc_control(0, 0, 30, 0)
                elif key == 84:  # Down arrow
                    drone.send_rc_control(0, 0, -30, 0)
                elif key == 81:  # Left arrow
                    drone.send_rc_control(0, 0, 0, -30)
                elif key == 83:  # Right arrow
                    drone.send_rc_control(0, 0, 0, 30)

        # Cleanup
        print("\nCleaning up...")
        if drone.is_flying():
            print("Landing drone...")
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            drone.land()
            time.sleep(3)

        drone.release()
        cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames")
        print(f"Average FPS: {depth_estimator.get_fps():.2f}")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if "drone" in locals() and drone.is_flying():
            print("Emergency landing...")
            drone.land()
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        if "drone" in locals() and drone.is_flying():
            print("Emergency landing...")
            drone.land()
        return 1


if __name__ == "__main__":
    sys.exit(main())
