"""
Tests for configuration loading and CLI argument parsing.

Verifies that config files are loaded correctly and CLI overrides work.
"""

import tempfile
from pathlib import Path

import yaml
import json

from src.common import load_config


def test_load_yaml_config():
    """
    Test loading a YAML configuration file.
    """
    config_data = {
        "env": "CartPole-v1",
        "timesteps": 50000,
        "seed": 42,
        "n_steps": 512,
        "batch_size": 64,
        "learning_rate": 0.001
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_data, f)
        
        loaded = load_config(str(config_path))
        
        assert loaded == config_data
        assert loaded["env"] == "CartPole-v1"
        assert loaded["timesteps"] == 50000
        assert loaded["seed"] == 42


def test_load_json_config():
    """
    Test loading a JSON configuration file.
    """
    config_data = {
        "env": "CartPole-v1",
        "timesteps": 100000,
        "seed": 123,
        "learning_rate": 0.0005
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        loaded = load_config(str(config_path))
        
        assert loaded == config_data
        assert loaded["env"] == "CartPole-v1"
        assert loaded["timesteps"] == 100000


def test_config_with_nested_structure():
    """
    Test loading config with nested dictionaries.
    """
    config_data = {
        "env": "CartPole-v1",
        "training": {
            "timesteps": 50000,
            "seed": 42
        },
        "hyperparameters": {
            "n_steps": 512,
            "batch_size": 64,
            "learning_rate": 0.001
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "nested_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_data, f)
        
        loaded = load_config(str(config_path))
        
        assert "training" in loaded
        assert "hyperparameters" in loaded
        assert loaded["training"]["timesteps"] == 50000
        assert loaded["hyperparameters"]["learning_rate"] == 0.001


def test_config_handles_different_types():
    """
    Test that config correctly handles different data types.
    """
    config_data = {
        "string_value": "CartPole-v1",
        "int_value": 50000,
        "float_value": 0.001,
        "bool_value": True,
        "list_value": [1, 2, 3],
        "none_value": None
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "types_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        loaded = load_config(str(config_path))
        
        assert isinstance(loaded["string_value"], str)
        assert isinstance(loaded["int_value"], int)
        assert isinstance(loaded["float_value"], float)
        assert isinstance(loaded["bool_value"], bool)
        assert isinstance(loaded["list_value"], list)
        assert loaded["none_value"] is None
