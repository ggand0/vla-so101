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
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.model.kinematics import RobotKinematics
from lerobot.record import record_loop
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
REPO_ID = f"{HF_USER}/so101_pick_place_smolvla"
TASK = "Pick up the cube and place it in the bowl"

FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
FRONT_CAM = "/dev/video0"

NUM_EPISODES = 2
EPISODE_TIME_S = 20
FPS = 30

# URDF for placo IK
URDF_PATH = str(Path(__file__).resolve().parent.parent / "models" / "so101_new_calib.urdf")
TARGET_FRAME = "gripper_frame_link"

# IK reset target (meters) — EE position for start of each episode
IK_RESET_TARGET = np.array([0.25, 0.0, 0.07])
HEIGHT_OFFSET = 0.07  # meters above target for step 3

# Wrist orientation for top-down (degrees)
WRIST_FLEX_DEG = 90.0
WRIST_ROLL_DEG = 90.0

# IK control parameters
MAX_DELTA_DEG = 10.0  # max joint delta per IK step
CONVERGE_THRESHOLD_M = 0.015  # 1.5cm
STUCK_THRESHOLD_M = 0.0005  # consider stuck if error changes less than this
MAX_IK_STEPS = 50
STUCK_PATIENCE = 3

# Timings
SAFE_JOINT_WAIT_S = 1.5
WRIST_WAIT_S = 1.0
IK_STEP_WAIT_S = 0.1


def make_follower() -> SO101Follower:
    """Create SO101 follower with use_degrees=True and front camera."""
    config = SO101FollowerConfig(
        id="ggando_so101_follower",
        port=FOLLOWER_PORT,
        use_degrees=True,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=FRONT_CAM,
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


def ik_reset(robot: SO101Follower, kinematics: RobotKinematics) -> None:
    """
    4-step IK reset: safe joints → wrist top-down → above target → lower to target.

    Operates entirely in degrees (use_degrees=True on bus, placo FK/IK in degrees).
    """
    bus = robot.bus
    logging.info(f"IK reset to EE target: {IK_RESET_TARGET}")

    # STEP 1: Move to SAFE_JOINTS (all zeros), gripper open
    safe_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    safe_joints = _clamp_degrees(safe_joints)
    action_dict = {name: safe_joints[i] for i, name in enumerate(_IK_MOTOR_NAMES)}
    action_dict["gripper"] = 50.0
    bus.sync_write("Goal_Position", action_dict, num_retry=3)
    busy_wait(SAFE_JOINT_WAIT_S)

    # STEP 2: Set wrist to top-down orientation
    pos_dict = bus.sync_read("Present_Position", num_retry=3)
    current_joints = np.array([pos_dict[name] for name in _IK_MOTOR_NAMES])
    current_joints[3] = WRIST_FLEX_DEG  # wrist_flex
    current_joints[4] = WRIST_ROLL_DEG  # wrist_roll
    current_joints = _clamp_degrees(current_joints)
    action_dict = {name: current_joints[i] for i, name in enumerate(_IK_MOTOR_NAMES)}
    action_dict["gripper"] = 50.0
    bus.sync_write("Goal_Position", action_dict, num_retry=3)
    busy_wait(WRIST_WAIT_S)

    # STEP 3 & 4: Move above target, then lower to target
    above_target = IK_RESET_TARGET.copy()
    above_target[2] += HEIGHT_OFFSET

    ik_targets = [
        ("above", above_target),
        ("lower", IK_RESET_TARGET),
    ]

    for step_name, target_pos in ik_targets:
        prev_error = float("inf")
        stuck_count = 0

        for step_i in range(MAX_IK_STEPS):
            # Read current joint positions (degrees)
            current_pos_dict = bus.sync_read("Present_Position", num_retry=3)
            current_joints = np.array([current_pos_dict[name] for name in _IK_MOTOR_NAMES])

            # FK → current EE position
            current_ee_pose = kinematics.forward_kinematics(current_joints)
            current_ee = current_ee_pose[:3, 3]
            error = np.linalg.norm(target_pos - current_ee)

            if error < CONVERGE_THRESHOLD_M:
                logging.info(f"IK {step_name}: converged in {step_i} steps (error={error:.4f}m)")
                break

            # Stuck detection
            if abs(error - prev_error) < STUCK_THRESHOLD_M:
                stuck_count += 1
                if stuck_count >= STUCK_PATIENCE:
                    logging.warning(
                        f"IK {step_name}: stuck after {step_i} steps (error={error:.4f}m)"
                    )
                    break
            else:
                stuck_count = 0
            prev_error = error

            # IK → target joint positions
            desired_ee_pose = current_ee_pose.copy()
            desired_ee_pose[:3, 3] = target_pos
            target_joints = kinematics.inverse_kinematics(current_joints, desired_ee_pose)

            # Clamp delta per step
            delta = target_joints[:5] - current_joints
            delta = np.clip(delta, -MAX_DELTA_DEG, MAX_DELTA_DEG)
            target_joints_clamped = current_joints + delta
            target_joints_clamped = _clamp_degrees(target_joints_clamped)

            # Write to bus (preserve gripper)
            gripper_pos = current_pos_dict.get("gripper", 50.0)
            action_dict = {name: target_joints_clamped[i] for i, name in enumerate(_IK_MOTOR_NAMES)}
            action_dict["gripper"] = gripper_pos
            bus.sync_write("Goal_Position", action_dict, num_retry=3)
            busy_wait(IK_STEP_WAIT_S)
        else:
            logging.warning(f"IK {step_name}: did not converge after {MAX_IK_STEPS} steps (error={error:.4f}m)")


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

    # Load placo kinematics
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=list(_IK_MOTOR_NAMES),
    )
    logging.info(f"Loaded placo kinematics from {URDF_PATH}")

    # Keyboard listener (right arrow = exit early, left = rerecord, esc = stop)
    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            # IK reset before each episode
            log_say("Resetting arm", play_sounds=True)
            ik_reset(robot, kinematics)

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
                display_data=False,
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
