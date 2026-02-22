"""
Custom recording script with joint-space reset between episodes.

Reuses lerobot's recording infrastructure (record_loop, LeRobotDataset, etc.)
but replaces the between-episodes reset with a 2-step interpolated reset:
  1. SAFE_JOINTS - all joints to 0 (arm extended forward), gripper open
  2. Interpolate to HOME position

Usage:
    uv run python scripts/record.py --config configs/record_10cm.yaml
    uv run python scripts/record.py --config configs/record_30cm.yaml --resume
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.record import record_loop
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.rl.gym_manipulator import _IK_MOTOR_NAMES, _clamp_degrees
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_wrist_cam(cfg: dict):
    """Build wrist camera config — RealSense SDK or OpenCV depending on config."""
    cam_type = cfg.get("wrist_cam_type", "opencv")
    width = cfg.get("wrist_cam_width", cfg.get("cam_width", 640))
    height = cfg.get("wrist_cam_height", cfg.get("cam_height", 480))
    fps = cfg.get("wrist_cam_fps", cfg["fps"])

    if cam_type == "realsense":
        return RealSenseCameraConfig(
            serial_number_or_name=cfg["wrist_cam_serial"],
            fps=fps,
            width=width,
            height=height,
        )
    return OpenCVCameraConfig(
        index_or_path=cfg["wrist_cam"],
        fps=fps,
        width=width,
        height=height,
    )


def make_follower(cfg: dict) -> SO101Follower:
    cameras = {"wrist": _make_wrist_cam(cfg)}
    if "overhead_cam" in cfg:
        cameras["overhead"] = OpenCVCameraConfig(
            index_or_path=cfg["overhead_cam"],
            width=cfg["cam_width"],
            height=cfg["cam_height"],
            fps=cfg["fps"],
        )
    config = SO101FollowerConfig(
        id="ggando_so101_follower",
        port=cfg["follower_port"],
        use_degrees=True,
        cameras=cameras,
    )
    return SO101Follower(config)


def make_leader(cfg: dict) -> SO101Leader:
    config = SO101LeaderConfig(
        id="ggando_so101_leader",
        port=cfg["leader_port"],
        use_degrees=True,
    )
    return SO101Leader(config)


def _interpolate_move(bus, target_joints: np.ndarray, gripper: float, steps: int = 150, dt: float = 0.025):
    """Interpolate linearly from current position to target over `steps` steps."""
    pos_dict = bus.sync_read("Present_Position", num_retry=3)
    current = np.array([pos_dict[name] for name in _IK_MOTOR_NAMES])
    current_gripper = pos_dict.get("gripper", gripper)

    joint_traj = np.linspace(current, target_joints, steps)
    gripper_traj = np.linspace(current_gripper, gripper, steps)

    for i in range(steps):
        action_dict = {name: joint_traj[i][j] for j, name in enumerate(_IK_MOTOR_NAMES)}
        action_dict["gripper"] = gripper_traj[i]
        bus.sync_write("Goal_Position", action_dict, num_retry=3)
        busy_wait(dt)


def reset_to_home(robot: SO101Follower, cfg: dict) -> None:
    """2-step interpolated joint-space reset: safe joints (all zeros) -> home position."""
    bus = robot.bus
    home_joints = np.array(cfg["home_joints_deg"])
    home_gripper = cfg["home_gripper"]

    logging.info(f"Resetting to home joints: {home_joints}")

    # STEP 1: Snap to SAFE_JOINTS (all zeros) to avoid collisions
    safe_joints = _clamp_degrees(np.zeros(5))
    action_dict = {name: safe_joints[i] for i, name in enumerate(_IK_MOTOR_NAMES)}
    action_dict["gripper"] = home_gripper
    bus.sync_write("Goal_Position", action_dict, num_retry=3)
    busy_wait(cfg["safe_joint_wait_s"])

    # STEP 2: Interpolate to home position
    home_clamped = _clamp_degrees(home_joints.copy())
    _interpolate_move(bus, home_clamped, home_gripper)


def main():
    parser = argparse.ArgumentParser(description="Record SO101 episodes with joint-space reset")
    parser.add_argument("--config", type=str, required=True, help="Path to recording config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted recording session")
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_logging()

    repo_id = cfg["repo_id"]
    fps = cfg["fps"]
    task = cfg["task"]
    num_episodes = cfg["num_episodes"]
    episode_time_s = cfg["episode_time_s"]

    # Build robot and teleoperator
    robot = make_follower(cfg)
    teleop = make_leader(cfg)

    # Dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    # Create or resume dataset
    if args.resume:
        dataset = LeRobotDataset(
            repo_id,
            batch_encoding_size=32,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, fps, dataset_features)
    else:
        sanity_check_dataset_name(repo_id, policy_cfg=None)
        dataset = LeRobotDataset.create(
            repo_id,
            fps,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
            batch_encoding_size=32,
        )

    # Re-init logging with DEBUG file log in dataset directory
    init_logging(log_file=Path(dataset.root) / "record.log", file_level="DEBUG")

    # Connect hardware
    robot.connect()
    teleop.connect()

    # Log actual camera resolutions
    for name, cam in robot.cameras.items():
        logging.info(f"Camera '{name}': {cam.width}x{cam.height} @ {cam.fps}fps")

    # Init rerun viewer for live camera preview
    _init_rerun(session_name="recording")

    # Keyboard listener (right arrow = exit early, left = rerecord, esc = stop)
    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < num_episodes and not events["stop_recording"]:
            # Reset before each episode
            log_say("Resetting arm", play_sounds=True)
            reset_to_home(robot, cfg)

            # Sync leader to follower position, then release torque for teleoperation
            follower_pos = robot.bus.sync_read("Present_Position", num_retry=3)
            leader_goal = {name: follower_pos[name] for name in follower_pos}
            teleop.bus.sync_write("Goal_Position", leader_goal, num_retry=3)
            busy_wait(0.5)
            teleop.bus.sync_write("Torque_Enable", 0, num_retry=3)

            log_say(f"Recording episode {dataset.num_episodes}", play_sounds=True)
            record_loop(
                robot=robot,
                events=events,
                fps=fps,
                teleop=teleop,
                dataset=dataset,
                control_time_s=episode_time_s,
                single_task=task,
                display_data=True,
            )

            # Handle rerecord
            if events["rerecord_episode"]:
                log_say("Re-record episode", play_sounds=True)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1
            logging.info(f"Saved episode {recorded_episodes}/{num_episodes}")

    log_say("Stop recording", play_sounds=True, blocking=True)

    robot.disconnect()
    teleop.disconnect()

    if listener is not None:
        listener.stop()

    log_say("Exiting", play_sounds=True)


if __name__ == "__main__":
    main()
