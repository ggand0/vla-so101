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
from lerobot.scripts.train import init_logging, train

_original_validate = TrainPipelineConfig.validate


def _patched_validate(self):
    _original_validate(self)
    if self.policy is not None:
        self.policy.input_features = {}


TrainPipelineConfig.validate = _patched_validate

if __name__ == "__main__":
    init_logging()
    train()
