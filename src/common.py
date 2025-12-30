# common.py
# This file contains common utility functions and classes for the project.
from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml
from scipy import stats


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


def save_config(config: Union[TrainConfig, EvalConfig, Dict[str, Any]], path: Union[Path, str]) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: Configuration object or dictionary to save
        path: Path to save the config file (can be a directory or file path)
    """
    path = Path(path)
    if path.is_dir():
        config_path = path / "config.json"
    else:
        config_path = path

    # Convert dataclass to dict if needed
    if hasattr(config, "__dataclass_fields__"):
        config_data = asdict(config)
    else:
        config_data = config

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)


def load_config(
    path: Union[Path, str], config_type: Optional[type] = None
) -> Union[TrainConfig, EvalConfig, Dict[str, Any]]:
    """
    Load configuration from a JSON or YAML file.

    Args:
        path: Path to the configuration file or directory containing config.json
        config_type: Optional type of config (TrainConfig or EvalConfig)

    Returns:
        Configuration object of the specified type or dictionary
    """
    path = Path(path)

    # If path is a directory, look for config.json
    if path.is_dir():
        path = path / "config.json"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Use .json, .yaml, or .yml")

    if config_type is not None:
        return config_type(**data)
    return data


def ensure_dir(directory: Union[str, Path]) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute mean and confidence interval for data.

    Args:
        data: Array-like of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)

    if n == 1:
        return mean, mean, mean

    stderr = stats.sem(data)
    margin = stderr * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, mean - margin, mean + margin


def merge_config_with_args(
    config: Union[TrainConfig, EvalConfig], args: dict
) -> Union[TrainConfig, EvalConfig]:
    """
    Merge command-line arguments into a configuration object.
    CLI arguments override config file values.

    Args:
        config: Base configuration from file
        args: Dictionary of command-line arguments

    Returns:
        New configuration object with overrides applied
    """
    config_dict = asdict(config)
    config_type = type(config)
    field_names = {f.name for f in fields(config_type)}

    for key, value in args.items():
        if value is not None and key in field_names:
            config_dict[key] = value

    return config_type(**config_dict)


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
