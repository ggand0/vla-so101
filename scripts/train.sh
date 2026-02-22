#!/usr/bin/env bash
set -euo pipefail

# SmolVLA fine-tuning on SO-101 pick-and-place demos
#
# Pretrained base: lerobot/smolvla_base (HuggingFace Hub)
# Freezes vision encoder, trains action expert + state projection
#
# Usage:
#   ./scripts/train.sh

export RUST_LOG=error

DATASET="gtgando/so101_pick_place_10cm_v1"
OUTPUT_DIR="outputs/train/smolvla_so101_10cm"
JOB_NAME="smolvla_so101_10cm"

exec uv run python scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET}" \
  --dataset.image_transforms.enable=true \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --save_checkpoint=true \
  --eval_freq=0 \
  --log_freq=100 \
  --num_workers=4 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --wandb.enable=false \
  "$@"
