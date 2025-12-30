"""
Smoke tests for training functionality.

These tests verify that basic training operations complete successfully
without errors, even if the agent doesn't learn much in the short time.
"""

import tempfile
from pathlib import Path

from src.train import train_ppo


def test_basic_training():
    """
    Test that basic training runs without errors.

    This is a smoke test - we just verify it doesn't crash.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model, metrics = train_ppo(
            env_name="CartPole-v1", total_timesteps=1000, seed=42, output_dir=tmpdir, verbose=0
        )

        assert model is not None, "Training returned None model"
        assert "total_episodes" in metrics, "Metrics missing total_episodes"
        assert "final_mean_return" in metrics, "Metrics missing final_mean_return"
        assert metrics["total_episodes"] > 0, "No episodes were run"


def test_training_with_custom_hyperparameters():
    """
    Test training with non-default hyperparameters.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model, metrics = train_ppo(
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=123,
            output_dir=tmpdir,
            n_steps=64,
            batch_size=32,
            learning_rate=0.0005,
            verbose=0,
        )

        assert model is not None
        assert metrics["total_episodes"] > 0


def test_training_saves_artifacts():
    """
    Verify that training saves expected files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        model, metrics = train_ppo(
            env_name="CartPole-v1",
            total_timesteps=1000,
            seed=42,
            output_dir=str(output_dir),
            verbose=0,
        )

        # Check that key artifacts are saved
        assert (output_dir / "model.zip").exists(), "Model file not saved"
        assert (output_dir / "training_returns.png").exists(), "Training plot not saved"
        assert (output_dir / "training_metrics.json").exists(), "Metrics file not saved"

        # Check tensorboard directory
        tensorboard_dirs = list(output_dir.glob("tensorboard/*"))
        assert len(tensorboard_dirs) > 0, "TensorBoard logs not created"


def test_training_different_environments():
    """
    Test training on different Gym environments.
    """
    environments = ["CartPole-v1"]

    for env_name in environments:
        with tempfile.TemporaryDirectory() as tmpdir:
            model, metrics = train_ppo(
                env_name=env_name, total_timesteps=500, seed=42, output_dir=tmpdir, verbose=0
            )

            assert model is not None, f"Training failed for {env_name}"
            assert metrics["total_episodes"] > 0, f"No episodes for {env_name}"


def test_training_metrics_structure():
    """
    Verify that training returns well-structured metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model, metrics = train_ppo(
            env_name="CartPole-v1", total_timesteps=1000, seed=42, output_dir=tmpdir, verbose=0
        )

        # Check all expected keys are present
        expected_keys = ["total_episodes", "final_mean_return", "max_return", "min_return"]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

        # Check value types and ranges
        assert isinstance(metrics["total_episodes"], int)
        assert metrics["total_episodes"] > 0
        assert isinstance(metrics["final_mean_return"], (int, float))
        assert metrics["max_return"] >= metrics["final_mean_return"]
        assert metrics["min_return"] <= metrics["final_mean_return"]
