"""
Run a trained SmolVLA policy on the SO-101 follower arm.

Loads the checkpoint, connects to the robot (with cameras), and runs
a closed-loop control loop: observe → predict action → send to robot.

Usage:
    uv run python scripts/infer.py
    uv run python scripts/infer.py --checkpoint outputs/train/smolvla_so101_pick_place/checkpoints/010000
    uv run python scripts/infer.py --num-rollouts 5
"""

import argparse
import logging
import time

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.rl.gym_manipulator import _IK_MOTOR_NAMES, _clamp_degrees
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CHECKPOINT = "outputs/train/smolvla_so101_pick_place/checkpoints/last/pretrained_model"
TASK = "Pick up the cube and place it in the bowl"

FOLLOWER_PORT = "/dev/ttyACM0"
WRIST_CAM = "/dev/video0"
OVERHEAD_CAM = "/dev/video2"

FPS = 30
EPISODE_TIME_S = 15

# Home joint positions (degrees) — same as record.py
HOME_JOINTS_DEG = np.array([0.0, -104.66, 96.09, 48.92, 90.0])
HOME_GRIPPER = 5.0


def make_follower() -> SO101Follower:
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


def _interpolate_move(bus, target_joints: np.ndarray, gripper: float, steps: int = 150, dt: float = 0.025):
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


def reset_to_home(robot: SO101Follower) -> None:
    bus = robot.bus
    logging.info("Resetting to home position")

    # Step 1: Snap to safe joints (all zeros)
    safe_joints = _clamp_degrees(np.zeros(5))
    action_dict = {name: safe_joints[i] for i, name in enumerate(_IK_MOTOR_NAMES)}
    action_dict["gripper"] = HOME_GRIPPER
    bus.sync_write("Goal_Position", action_dict, num_retry=3)
    busy_wait(1.5)

    # Step 2: Interpolate to home position
    home_joints = _clamp_degrees(HOME_JOINTS_DEG.copy())
    _interpolate_move(bus, home_joints, HOME_GRIPPER)


def run_episode(robot: SO101Follower, policy, device, dataset_features, events, episode_time_s):
    """Run one inference episode: observe → predict → act at FPS."""
    policy.reset()

    action_keys = list(robot.action_features.keys())
    start_time = time.perf_counter()
    step = 0

    logging.info(f"Running policy for up to {episode_time_s}s at {FPS} fps")

    while True:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            logging.info("Episode ended early (keyboard)")
            break

        elapsed = time.perf_counter() - start_time
        if elapsed >= episode_time_s:
            logging.info(f"Episode done ({elapsed:.1f}s, {step} steps)")
            break

        # Get observation from robot
        obs_raw = robot.get_observation()

        # Build observation dict matching dataset feature keys
        obs_frame = build_dataset_frame(dataset_features, obs_raw, prefix="observation")

        # Predict action
        action_values = predict_action(
            obs_frame,
            policy,
            device,
            policy.config.use_amp,
            task=TASK,
            robot_type=robot.robot_type,
        )

        # Send action to robot
        action = {key: action_values[i].item() for i, key in enumerate(action_keys)}
        robot.send_action(action)

        step += 1
        busy_wait(1.0 / FPS - (time.perf_counter() - loop_start))

    return step


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLA inference on SO-101")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to pretrained_model directory",
    )
    parser.add_argument("--num-rollouts", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--episode-time", type=float, default=EPISODE_TIME_S, help="Episode duration (s)")
    args = parser.parse_args()

    episode_time_s = args.episode_time

    init_logging()

    # Load policy
    logging.info(f"Loading policy from {args.checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    device = get_safe_torch_device(policy.config.device)
    policy.eval()
    logging.info(f"Policy loaded on {device} ({sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M params)")

    # Build robot
    robot = make_follower()
    robot.connect()

    # Dataset features (needed for build_dataset_frame to map obs keys)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    dataset_features = {**obs_features, **action_features}

    # Keyboard listener
    listener, events = init_keyboard_listener()

    try:
        for ep in range(args.num_rollouts):
            if events["stop_recording"]:
                break

            logging.info(f"--- Episode {ep + 1}/{args.num_rollouts} ---")
            reset_to_home(robot)
            logging.info("Press right arrow to start (or it starts automatically in 3s)")
            busy_wait(3.0)

            steps = run_episode(robot, policy, device, dataset_features, events, episode_time_s)
            logging.info(f"Episode {ep + 1} complete: {steps} steps")

            if ep < args.num_rollouts - 1:
                logging.info("Waiting 2s before next episode...")
                busy_wait(2.0)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        reset_to_home(robot)
        robot.disconnect()
        if listener is not None:
            listener.stop()
        logging.info("Done")


if __name__ == "__main__":
    main()
