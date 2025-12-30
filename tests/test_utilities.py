"""
Tests for utility functions in common.py.

Tests helper functions for config management, file I/O, and data processing.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.common import (
    get_git_hash,
    save_config,
    load_config,
    ensure_dir,
    compute_confidence_interval
)


def test_get_git_hash():
    """
    Test that get_git_hash returns a valid hash or 'unknown'.
    """
    git_hash = get_git_hash()
    
    assert isinstance(git_hash, str), "Git hash should be a string"
    assert len(git_hash) > 0, "Git hash should not be empty"
    
    # Should be either 'unknown' or a hex string
    if git_hash != "unknown":
        assert len(git_hash) == 7, "Git hash should be 7 characters"
        # Check it's valid hex
        try:
            int(git_hash, 16)
        except ValueError:
            pytest.fail("Git hash is not valid hexadecimal")


def test_save_and_load_config_json():
    """
    Test saving and loading config in JSON format.
    """
    config = {
        "env": "CartPole-v1",
        "timesteps": 50000,
        "seed": 42,
        "learning_rate": 0.001
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        
        # Save config
        save_config(config, str(config_path))
        assert config_path.exists(), "Config file not created"
        
        # Load config
        loaded_config = load_config(str(config_path))
        
        # Verify contents match
        assert loaded_config == config, "Loaded config doesn't match saved config"


def test_save_and_load_config_yaml():
    """
    Test saving and loading config in YAML format.
    """
    config = {
        "env": "CartPole-v1",
        "timesteps": 50000,
        "seed": 42,
        "n_steps": 512,
        "batch_size": 64
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        
        # Save as YAML
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        # Load config
        loaded_config = load_config(str(config_path))
        
        # Verify contents match
        assert loaded_config == config, "Loaded YAML config doesn't match"


def test_load_config_invalid_format():
    """
    Test that loading an unsupported format raises an error.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.txt"
        config_path.write_text("not a valid config format")
        
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(str(config_path))


def test_ensure_dir():
    """
    Test that ensure_dir creates directories correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "nested" / "directory" / "structure"
        
        # Directory shouldn't exist yet
        assert not test_dir.exists()
        
        # Create it
        ensure_dir(str(test_dir))
        
        # Now it should exist
        assert test_dir.exists()
        assert test_dir.is_dir()
        
        # Calling again should not raise an error
        ensure_dir(str(test_dir))


def test_compute_confidence_interval():
    """
    Test confidence interval calculation.
    """
    import numpy as np
    
    # Test with known data
    data = [10.0, 12.0, 11.0, 13.0, 11.5]
    mean, ci_low, ci_high = compute_confidence_interval(data, confidence=0.95)
    
    # Check mean is correct
    assert abs(mean - 11.5) < 0.01, "Mean calculation incorrect"
    
    # CI should bracket the mean
    assert ci_low < mean, "CI lower bound should be less than mean"
    assert ci_high > mean, "CI upper bound should be greater than mean"
    
    # CI should be symmetric for this data
    # (not always true but should be close for small symmetric data)
    lower_margin = mean - ci_low
    upper_margin = ci_high - mean
    assert abs(lower_margin - upper_margin) < 0.5, "CI should be roughly symmetric"


def test_compute_confidence_interval_edge_cases():
    """
    Test confidence interval with edge cases.
    """
    # Single value
    data_single = [42.0]
    mean, ci_low, ci_high = compute_confidence_interval(data_single)
    assert mean == 42.0
    # With one sample, CI should equal the mean
    assert ci_low == mean
    assert ci_high == mean
    
    # Identical values
    data_identical = [5.0, 5.0, 5.0, 5.0]
    mean, ci_low, ci_high = compute_confidence_interval(data_identical)
    assert mean == 5.0
    # No variance means tight CI
    assert abs(ci_low - mean) < 0.01
    assert abs(ci_high - mean) < 0.01
