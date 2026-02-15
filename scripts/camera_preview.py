#!/usr/bin/env python3
"""Simple camera preview with optional teleoperation and recording."""
import argparse
import datetime
import time
import cv2
import numpy as np
from pathlib import Path


def setup_teleoperation(
    follower_port: str,
    leader_port: str,
    calibration_dir: str,
    follower_id: str = "ggando_so101_follower",
    leader_id: str = "ggando_so101_leader",
):
    """Initialize leader and follower robots for teleoperation."""
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    calibration_path = Path(calibration_dir)

    # Initialize follower
    follower_config = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        calibration_dir=calibration_path / "robots" / "so101_follower",
        use_degrees=True,
    )
    follower = make_robot_from_config(follower_config)
    follower.connect()

    # Initialize leader
    leader_config = SO101LeaderConfig(
        port=leader_port,
        id=leader_id,
        calibration_dir=calibration_path / "teleoperators" / "so101_leader",
        use_degrees=True,
    )
    leader = make_teleoperator_from_config(leader_config)
    leader.connect()

    return follower, leader


def main():
    parser = argparse.ArgumentParser(description="Camera preview with optional teleop")
    parser.add_argument("--teleop", action="store_true", help="Enable teleoperation")
    parser.add_argument("--follower_port", default="/dev/ttyACM0", help="Follower port")
    parser.add_argument("--leader_port", default="/dev/ttyACM1", help="Leader port")
    parser.add_argument("--calibration_dir", default="/home/gota/.cache/huggingface/lerobot/calibration")
    parser.add_argument("--camera", default="/dev/video0", help="Camera device (ignored with --realsense)")
    parser.add_argument("--realsense", nargs="?", const="auto", default=None, metavar="SERIAL",
                        help="Use RealSense SDK (auto-detects device, or pass serial)")
    parser.add_argument("--crop", action="store_true", help="Show training crop [0, 80, 480, 480]")
    parser.add_argument("--record", type=str, default=None, metavar="DIR",
                        help="Record video to DIR (press 'r' to start/stop, 's' for snapshot)")
    args = parser.parse_args()

    follower = None
    leader = None

    if args.teleop:
        print("Setting up teleoperation...")
        follower, leader = setup_teleoperation(
            args.follower_port,
            args.leader_port,
            args.calibration_dir,
        )
        print("Teleop ready. Move leader arm to control follower.")

    # Camera setup
    pipe = None
    cap = None
    if args.realsense:
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        rs_cfg = rs.config()
        if args.realsense != "auto":
            rs_cfg.enable_device(args.realsense)
        rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipe.start(rs_cfg)
        vs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(f"RealSense {args.realsense}: {vs.width()}x{vs.height()} @ {vs.fps()}fps")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"OpenCV {args.camera}: {w}x{h}")

    # Recording state
    record_dir = None
    if args.record:
        record_dir = Path(args.record)
        record_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None
    recording = False

    controls = "q=quit"
    if record_dir:
        controls += " | r=record | s=snapshot"
    print(f"Camera preview started. Controls: {controls}")

    try:
        while True:
            if pipe:
                frames = pipe.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # Teleop loop
            if args.teleop and follower and leader:
                leader_state = leader.get_action()
                follower.send_action(leader_state)

            # Display
            if args.crop:
                h, w = frame.shape[:2]
                crop_x, crop_y, crop_w, crop_h = 0, 80, 480, 480
                if crop_y + crop_h <= h and crop_x + crop_w <= w:
                    display = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                else:
                    display = frame
            else:
                display = frame

            # Recording indicator
            show = display.copy()
            if recording:
                cv2.circle(show, (20, 20), 8, (0, 0, 255), -1)
                cv2.putText(show, "REC", (35, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Camera (q to quit)", show)

            # Write frame if recording
            if recording and video_writer is not None:
                video_writer.write(display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and record_dir:
                if not recording:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = record_dir / f"recording_{ts}.mp4"
                    h, w = display.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                    )
                    recording = True
                    print(f"Recording started: {video_path}")
                else:
                    recording = False
                    video_writer.release()
                    video_writer = None
                    print("Recording stopped")
            elif key == ord('s') and record_dir:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = record_dir / f"snapshot_{ts}.png"
                cv2.imwrite(str(snap_path), display)
                print(f"Snapshot saved: {snap_path}")

    finally:
        if video_writer is not None:
            video_writer.release()
        if pipe:
            pipe.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if follower:
            follower.disconnect()
        if leader:
            leader.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
