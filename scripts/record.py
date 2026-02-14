"""
Custom recording script with placo-based IK reset between episodes.

Reuses lerobot's recording infrastructure (record_loop, LeRobotDataset, etc.)
but replaces the between-episodes reset with a 4-step IK reset sequence:
  1. SAFE_JOINTS — all joints to 0° (arm extended forward), gripper open
  2. Wrist top-down — wrist_flex=90°, wrist_roll=90°
  3. Move ABOVE target — closed-loop IK to target + 7cm
  4. Lower to target — closed-loop IK to final EE position

Usage:
    uv run python scripts/record.py
    uv run python scripts/record.py --resume
"""

import argparse
import logging
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
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

# =============================================================================
# Configuration — edit these for your setup
# =============================================================================

HF_USER = "gtgando"
#REPO_ID = f"{HF_USER}/so101_pick_place_smolvla_v2"
REPO_ID = f"{HF_USER}/so101_pick_place_smolvla_v2"
TASK = "Pick up the cube and place it in the bowl"

FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
WRIST_CAM = "/dev/video0"
OVERHEAD_CAM = "/dev/video2"

NUM_EPISODES = 20
EPISODE_TIME_S = 10
FPS = 30

# Home joint positions (degrees) — reset target
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
HOME_JOINTS_DEG = np.array([0.0, -104.66, 96.09, 48.92, 90.0])
HOME_GRIPPER = 5.0

# Timings
SAFE_JOINT_WAIT_S = 1.5
HOME_JOINT_WAIT_S = 1.5


def make_follower() -> SO101Follower:
    """Create SO101 follower with use_degrees=True and front camera."""
    config = SO101FollowerConfig(
        id="ggando_so101_follower",
        port=FOLLOWER_PORT,
        use_degrees=True,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=WRIST_CAM,
                width=640,
                height=480,
                fps=FPS,
            ),
            "overhead": OpenCVCameraConfig(
                index_or_path=OVERHEAD_CAM,
                width=640,
                height=480,
                fps=FPS,
            ),
        },
    )
    return SO101Follower(config)


def make_leader() -> SO101Leader:
    """Create SO101 leader with use_degrees=True."""
    config = SO101LeaderConfig(
        id="ggando_so101_leader",
        port=LEADER_PORT,
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


def ik_reset(robot: SO101Follower) -> None:
    """
    2-step interpolated joint-space reset: safe joints (all zeros) → home position.
    """
    bus = robot.bus
    logging.info(f"Resetting to home joints: {HOME_JOINTS_DEG}")

    # STEP 1: Snap to SAFE_JOINTS (all zeros) to avoid collisions
    safe_joints = _clamp_degrees(np.zeros(5))
    action_dict = {name: safe_joints[i] for i, name in enumerate(_IK_MOTOR_NAMES)}
    action_dict["gripper"] = HOME_GRIPPER
    bus.sync_write("Goal_Position", action_dict, num_retry=3)
    busy_wait(SAFE_JOINT_WAIT_S)

    # STEP 2: Interpolate to home position
    home_joints = _clamp_degrees(HOME_JOINTS_DEG.copy())
    _interpolate_move(bus, home_joints, HOME_GRIPPER)


def main():
    parser = argparse.ArgumentParser(description="Record SO101 episodes with IK reset")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted recording session")
    args = parser.parse_args()

    init_logging()

    # Build robot and teleoperator
    robot = make_follower()
    teleop = make_leader()

    # Dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    # Create or resume dataset
    if args.resume:
        dataset = LeRobotDataset(
            REPO_ID,
            batch_encoding_size=32,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, FPS, dataset_features)
    else:
        sanity_check_dataset_name(REPO_ID, policy_cfg=None)
        dataset = LeRobotDataset.create(
            REPO_ID,
            FPS,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
            batch_encoding_size=32,
        )

    # Connect hardware
    robot.connect()
    teleop.connect()

    # Init rerun viewer for live camera preview
    _init_rerun(session_name="recording")

    # Keyboard listener (right arrow = exit early, left = rerecord, esc = stop)
    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            # IK reset before each episode
            log_say("Resetting arm", play_sounds=True)
            ik_reset(robot)

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
                fps=FPS,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_S,
                single_task=TASK,
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
            logging.info(f"Saved episode {recorded_episodes}/{NUM_EPISODES}")

    log_say("Stop recording", play_sounds=True, blocking=True)

    robot.disconnect()
    teleop.disconnect()

    if listener is not None:
        listener.stop()

    log_say("Exiting", play_sounds=True)


if __name__ == "__main__":
    main()
