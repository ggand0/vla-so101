#!/usr/bin/env python
"""SmolVLA fine-tuning wrapper.

The pretrained smolvla_base config has input_features keyed as camera1/camera2/camera3,
but our dataset uses observation.images.wrist and observation.images.overhead.
This wrapper clears the pretrained input_features after config validation so that
make_policy() auto-detects the correct camera names from the dataset metadata.
"""

import os
import warnings

# Suppress noisy warnings from dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*video decoding and encoding.*torchvision.*")
warnings.filterwarnings("ignore", message=".*We will use 90% of the memory.*")

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset as _original_make_dataset
from lerobot.scripts.train import init_logging, train
import lerobot.datasets.factory as factory_module

_original_validate = TrainPipelineConfig.validate

# Cameras to exclude from training (set via --exclude-cameras CLI arg)
_exclude_cameras: set[str] = set()


def _patched_validate(self):
    _original_validate(self)
    if self.policy is not None:
        self.policy.input_features = {}


def _filtered_make_dataset(cfg):
    dataset = _original_make_dataset(cfg)
    if _exclude_cameras:
        for cam in _exclude_cameras:
            key = f"observation.images.{cam}"
            if key in dataset.meta.features:
                del dataset.meta.features[key]
            if key in dataset.meta.stats:
                del dataset.meta.stats[key]
    return dataset


TrainPipelineConfig.validate = _patched_validate
factory_module.make_dataset = _filtered_make_dataset

if __name__ == "__main__":
    import sys

    # Parse custom args before lerobot's parser sees them
    remaining = []
    resume_from = None
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--exclude-cameras" and i + 1 < len(sys.argv):
            _exclude_cameras.update(sys.argv[i + 1].split(","))
            i += 2
        elif sys.argv[i].startswith("--resume-from"):
            # Support --resume-from=path or --resume-from path
            if "=" in sys.argv[i]:
                resume_from = sys.argv[i].split("=", 1)[1]
                i += 1
            elif i + 1 < len(sys.argv):
                resume_from = sys.argv[i + 1]
                i += 2
            else:
                raise ValueError("--resume-from requires a checkpoint path")
        else:
            remaining.append(sys.argv[i])
            i += 1

    if resume_from:
        # Convert checkpoint path to what lerobot's resume expects:
        # --resume=true --config_path=<checkpoint>/pretrained_model/train_config.json
        # and remove --policy.path if present
        from pathlib import Path
        ckpt = Path(resume_from)
        config_json = ckpt / "pretrained_model" / "train_config.json"
        if not config_json.exists():
            # Maybe they passed the pretrained_model dir directly
            config_json = ckpt / "train_config.json"
        if not config_json.exists():
            raise FileNotFoundError(
                f"Cannot find train_config.json in {resume_from}. "
                f"Pass the checkpoint directory (e.g. checkpoints/010000)."
            )
        # Remove --policy.path from remaining args
        remaining = [a for a in remaining if not a.startswith("--policy.path")]
        remaining.extend(["--resume=true", f"--config_path={config_json}"])

    sys.argv = remaining

    init_logging()
    train()
