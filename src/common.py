# common.py
# This file contains common utility functions and classes for the project.
from __future__ import annotations
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Paths:
    outputs_dir: str = "outputs"
    videos_dir: str = "videos"
    runs_dir: str = "outputs/runs"


@dataclass
class TrainConfig:
    env: str
    timesteps: int
    seed: int
    n_steps: int = 2048
    batch_size: int = 64
    gae_lambda: float = 0.95
    gamma: float = 0.99
    n_epochs: int = 10
    ent_coef: float = 0.0
    learning_rate: float = 3e-4
    clip_range: float = 0.2


@dataclass
class EvalConfig:
    env: str
    model_path: str
    episodes: int
    seeds: list[int]


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def create_run_dir(config: TrainConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash()
    run_name = f"{timestamp}_{config.env}_seed{config.seed}_{git_hash}"
    run_dir = Path(Paths.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)
    return run_dir


def save_config(config: TrainConfig | EvalConfig, path: Path) -> None:
    with open(path / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)


def load_config(path: Path, config_type: type) -> TrainConfig | EvalConfig:
    with open(path / "config.json", "r") as f:
        data = json.load(f)
    return config_type(**data)


def save_metrics(metrics: Dict[str, Any], path: Path, filename: str = "metrics.json") -> None:
    with open(path / filename, "w") as f:
        json.dump(metrics, f, indent=2)


def ensure_dirs() -> None:
    os.makedirs(Paths.outputs_dir, exist_ok=True)
    os.makedirs(Paths.videos_dir, exist_ok=True)
    os.makedirs(Paths.runs_dir, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
