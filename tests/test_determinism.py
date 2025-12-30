"""
Tests for training determinism and reproducibility.

Verifies that identical seeds produce identical results,
which is critical for scientific reproducibility.
"""

import tempfile

import numpy as np

from src.evaluate import evaluate_model
from src.train import train_ppo


def test_seed_determinism():
    """
    Test that training with the same seed produces identical results.

    This is crucial for reproducibility in RL research.
    """
    seed = 12345
    timesteps = 2000
    env_name = "CartPole-v1"

    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:

        # Train first model
        model1, metrics1 = train_ppo(
            env_name=env_name,
            total_timesteps=timesteps,
            seed=seed,
            output_dir=tmpdir1,
            n_steps=128,
            batch_size=64,
            learning_rate=0.001,
            verbose=0,
        )

        # Train second model with same seed
        model2, metrics2 = train_ppo(
            env_name=env_name,
            total_timesteps=timesteps,
            seed=seed,
            output_dir=tmpdir2,
            n_steps=128,
            batch_size=64,
            learning_rate=0.001,
            verbose=0,
        )

        # Check training metrics match
        assert (
            metrics1["total_episodes"] == metrics2["total_episodes"]
        ), "Episode counts differ between runs with same seed"

        # Training curves should be very similar (allowing small numerical error)
        assert (
            abs(metrics1["final_mean_return"] - metrics2["final_mean_return"]) < 1.0
        ), "Final returns differ significantly between runs with same seed"


def test_different_seeds_produce_different_results():
    """
    Verify that different seeds produce different training outcomes.

    This ensures our seeding is actually working.
    """
    seed1 = 111
    seed2 = 999
    timesteps = 2000
    env_name = "CartPole-v1"

    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:

        model1, metrics1 = train_ppo(
            env_name=env_name,
            total_timesteps=timesteps,
            seed=seed1,
            output_dir=tmpdir1,
            n_steps=128,
            batch_size=64,
            learning_rate=0.001,
            verbose=0,
        )

        model2, metrics2 = train_ppo(
            env_name=env_name,
            total_timesteps=timesteps,
            seed=seed2,
            output_dir=tmpdir2,
            n_steps=128,
            batch_size=64,
            learning_rate=0.001,
            verbose=0,
        )

        # Different seeds should produce at least somewhat different results
        # (not a strict requirement but good sanity check)
        returns_differ = abs(metrics1["final_mean_return"] - metrics2["final_mean_return"]) > 0.1
        episodes_differ = metrics1["total_episodes"] != metrics2["total_episodes"]

        # At least one metric should differ
        assert returns_differ or episodes_differ, "Different seeds produced identical results"


def test_evaluation_determinism():
    """
    Test that evaluation with the same seed produces identical results.
    """
    seed = 42
    episodes = 5
    env_name = "CartPole-v1"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train a quick model
        model, _ = train_ppo(
            env_name=env_name, total_timesteps=1000, seed=100, output_dir=tmpdir, verbose=0
        )

        # Evaluate twice with same seed
        returns1, _ = evaluate_model(model, env_name, n_episodes=episodes, seed=seed)
        returns2, _ = evaluate_model(model, env_name, n_episodes=episodes, seed=seed)

        # Results should be identical
        np.testing.assert_array_equal(
            returns1, returns2, err_msg="Evaluation returns differ with same seed"
        )
