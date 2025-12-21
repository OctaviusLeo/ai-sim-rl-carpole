# common.py
# This file contains common utility functions and classes for the project.
from __future__ import annotations
import os
import random
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Paths:
    outputs_dir: str = "outputs"
    videos_dir: str = "videos"

def ensure_dirs() -> None:
    os.makedirs(Paths.outputs_dir, exist_ok=True)
    os.makedirs(Paths.videos_dir, exist_ok=True)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
