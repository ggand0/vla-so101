# vla-so101

SmolVLA fine-tuned on SO-101 for autonomous pick-and-place.

[Demo video](https://x.com/gtgando/status/2025427031102722300)

## Hardware

- **Robot**: [SO-101](https://github.com/TheRobotStudio/SO-ARM100) follower + leader arms (Feetech STS3215 servos)
- **Wrist camera**: Intel RealSense D405 (the InnoMaker UVC camera recommended by the [official SO-101 repo](https://github.com/TheRobotStudio/SO-ARM100) works too)
- **Overhead camera**: Logitech C920
- **GPU**: NVIDIA RTX 3090 (~10h for 20k training steps)

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install

```bash
git clone https://github.com/gtgando/vla-so101.git
git clone https://github.com/gtgando/lerobot.git
cd vla-so101
uv sync
```

This project uses a [custom LeRobot fork](https://github.com/gtgando/lerobot) installed as a local editable dependency from `../lerobot`. `uv sync` installs it automatically with the `feetech`, `smolvla`, `kinematics`, and `intelrealsense` extras. The two repos should sit side by side:

```
├── lerobot/
└── vla-so101/
```

### 3. Calibrate robots

Follow the [official HuggingFace SO-101 calibration guide](https://huggingface.co/docs/lerobot/so101) to calibrate both the leader and follower arms before recording.

### 4. Hardware setup

Find your device ports and camera paths:

```bash
uv run lerobot-find-port
uv run lerobot-find-cameras opencv
```

Edit the config YAML (e.g. `configs/record_10cm_v3.yaml`) with your specific:
- `follower_port` / `leader_port` (e.g. `/dev/ttyACM0`, `/dev/ttyACM1`)
- `wrist_cam_serial` (RealSense serial number)
- `overhead_cam` (video device path, e.g. `/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0`)
- `home_joints_deg` (home position in degrees)

## Usage

### 1. Record demonstrations

```bash
uv run python scripts/record.py --config configs/record_10cm_v3.yaml
```

Teleoperate the follower arm with the leader arm. Keyboard controls:
- **Right arrow** — skip to next episode
- **Left arrow** — re-record current episode
- **Esc** — stop recording

Pass `--resume` to continue a previously interrupted session.

The config specifies the HuggingFace `repo_id`, camera setup, episode count, FPS (30), and episode duration (10s). Data is saved in LeRobot v2.1 format (parquet + AV1 video).

### 2. Train

```bash
./scripts/train_smolvla_v3_dual.sh
```

Key parameters: batch size 64, 20k steps, checkpoints saved every 5k steps. Fine-tunes `lerobot/smolvla_base` with both wrist and overhead cameras.

To resume from a checkpoint:

```bash
uv run python scripts/train.py \
  --dataset.repo_id=gtgando/so101_pick_place_10cm_v3 \
  --resume-from outputs/train/smolvla_so101_10cm_v3_dual/checkpoints/020000/pretrained_model
```

### 3. Run inference

```bash
uv run python scripts/infer.py --record --direct-reset --num-rollouts 5
```

This runs the trained policy in a closed loop (observe → predict → act) and records a picture-in-picture video with audio. Videos are saved to `recordings/`.

Flags:
- `--checkpoint <path>` — model path (defaults to latest v3 dual checkpoint)
- `--num-rollouts <n>` — number of episodes
- `--episode-time <s>` — episode duration (default 10s)
- `--record` — record video
- `--direct-reset` — interpolate straight to home position
- `--no-overhead` — wrist camera only

## Results

- 75 episodes of teleoperated demonstrations
- 20k training steps (~10h on RTX 3090)
- 60-80% success rate on pick-and-place

[Demo video](https://x.com/gtgando/status/2025427031102722300)

## Project structure

```
configs/       — recording configs (YAML)
scripts/       — record, train, infer, utilities
models/        — robot URDF for IK-based reset
outputs/       — training checkpoints (gitignored)
recordings/    — inference videos (gitignored)
```
