#!/usr/bin/env bash
set -euo pipefail

# SO-101 data collection with IK reset between episodes
#
# Before running:
#   1. Verify ports:    uv run lerobot-find-port
#   2. Find cameras:    uv run lerobot-find-cameras opencv
#   3. Edit scripts/record.py to update ports/camera if needed
#
# Usage:
#   ./scripts/record.sh              # Record 50 episodes
#   ./scripts/record.sh --resume     # Resume interrupted session

export RUST_LOG=error

uv run python scripts/record.py "$@"
