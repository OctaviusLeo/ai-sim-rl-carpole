"""
Tests for evaluation functionality.

Verifies that model evaluation works correctly and produces valid metrics.
"""

import tempfile

import numpy as np
import pytest

from src.train import train_ppo
from src.evaluate import evaluate_model, multi_seed_evaluation


def test_evaluate_model_basic():
    """
    Test basic model evaluation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train a simple model
        model, _ = train_ppo(
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42,
            output_dir=tmpdir,
            verbose=0
        )
        
        # Evaluate it
        returns, success_rate = evaluate_model(
            model,
            env_name="CartPole-v1",
            n_episodes=10,
            seed=123
        )
        
        # Check outputs
        assert len(returns) == 10, "Should return 10 episode returns"
        assert all(isinstance(r, (int, float)) for r in returns), "Returns should be numeric"
        assert all(r >= 0 for r in returns), "Returns should be non-negative"
        assert 0.0 <= success_rate <= 1.0, "Success rate should be in [0, 1]"


def test_evaluate_model_deterministic():
    """
    Test that evaluation is deterministic with fixed seed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = train_ppo(
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42,
            output_dir=tmpdir,
            verbose=0
        )
        
        # Evaluate twice with same seed
        returns1, sr1 = evaluate_model(model, "CartPole-v1", n_episodes=5, seed=999)
        returns2, sr2 = evaluate_model(model, "CartPole-v1", n_episodes=5, seed=999)
        
        # Should be identical
        np.testing.assert_array_equal(returns1, returns2)
        assert sr1 == sr2


def test_multi_seed_evaluation():
    """
    Test multi-seed evaluation aggregates results correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = train_ppo(
            env_name="CartPole-v1",
            total_timesteps=2000,
            seed=42,
            output_dir=tmpdir,
            verbose=0
        )
        
        # Run multi-seed evaluation
        results = multi_seed_evaluation(
            model,
            env_name="CartPole-v1",
            episodes_per_seed=5,
            num_seeds=3,
            base_seed=100
        )
        
        # Check result structure
        assert "num_seeds" in results
        assert "total_episodes" in results
        assert "mean_return" in results
        assert "std_return" in results
        assert "ci_95_low" in results
        assert "ci_95_high" in results
        assert "mean_success_rate" in results
        
        # Check values
        assert results["num_seeds"] == 3
        assert results["total_episodes"] == 15  # 3 seeds * 5 episodes
        assert results["mean_return"] >= 0
        assert results["ci_95_low"] <= results["mean_return"]
        assert results["ci_95_high"] >= results["mean_return"]


def test_evaluate_model_different_episodes():
    """
    Test evaluation with different episode counts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = train_ppo(
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42,
            output_dir=tmpdir,
            verbose=0
        )
        
        for n_episodes in [1, 5, 10]:
            returns, _ = evaluate_model(
                model,
                env_name="CartPole-v1",
                n_episodes=n_episodes,
                seed=42
            )
            
            assert len(returns) == n_episodes, \
                f"Expected {n_episodes} returns, got {len(returns)}"


def test_success_rate_calculation():
    """
    Test that success rate is calculated correctly.
    
    For CartPole-v1, success is typically defined as reaching max episode length.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train a model
        model, _ = train_ppo(
            env_name="CartPole-v1",
            total_timesteps=5000,
            seed=42,
            output_dir=tmpdir,
            verbose=0
        )
        
        returns, success_rate = evaluate_model(
            model,
            env_name="CartPole-v1",
            n_episodes=20,
            seed=123,
            success_threshold=500.0
        )
        
        # Count how many episodes reached success threshold
        successful_episodes = sum(1 for r in returns if r >= 500.0)
        expected_success_rate = successful_episodes / len(returns)
        
        assert abs(success_rate - expected_success_rate) < 0.01, \
            "Success rate calculation doesn't match manual count"
