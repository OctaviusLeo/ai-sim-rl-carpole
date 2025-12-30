# train.py
# This script trains a PPO model on the CartPole-v1 environment and logs the training reward.
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

try:
    from .common import (
        TrainConfig,
        create_run_dir,
        ensure_dirs,
        load_config,
        merge_config_with_args,
        save_config,
        save_metrics,
        set_global_seed,
    )
except ImportError:
    from common import (
        TrainConfig,
        create_run_dir,
        ensure_dirs,
        load_config,
        merge_config_with_args,
        save_config,
        save_metrics,
        set_global_seed,
    )


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current = 0.0

    def _on_step(self) -> bool:
        # Vectorized env returns arrays
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        r = float(rewards[0])
        d = bool(dones[0])
        self._current += r
        if d:
            self.episode_rewards.append(self._current)
            self._current = 0.0
        return True


def train_ppo(
    env_name: str,
    total_timesteps: int,
    seed: int,
    output_dir: str,
    n_steps: int = 2048,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    verbose: int = 1,
):
    """
    Train a PPO agent on the specified environment.

    Args:
        env_name: Name of the Gym environment
        total_timesteps: Total timesteps to train for
        seed: Random seed
        output_dir: Directory to save outputs
        n_steps: Number of steps per update
        batch_size: Minibatch size
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        verbose: Verbosity level (0=none, 1=info)

    Returns:
        Tuple of (trained model, metrics dict)
    """
    set_global_seed(seed)

    env = gym.make(env_name)
    env.reset(seed=seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "tensorboard").mkdir(exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=verbose,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        gamma=gamma,
        learning_rate=learning_rate,
        tensorboard_log=str(output_path / "tensorboard"),
    )

    cb = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=cb)

    # Save model
    model_path = output_path / "model"
    model.save(str(model_path))

    # Compute metrics
    metrics = {}
    if len(cb.episode_rewards) > 0:
        # Save training plot
        plt.figure()
        plt.plot(cb.episode_rewards)
        plt.title("Training Episode Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plot_path = output_path / "training_returns.png"
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()

        metrics = {
            "total_episodes": len(cb.episode_rewards),
            "final_mean_return": float(
                sum(cb.episode_rewards[-100:]) / min(100, len(cb.episode_rewards))
            ),
            "max_return": float(max(cb.episode_rewards)),
            "min_return": float(min(cb.episode_rewards)),
        }
        save_metrics(metrics, output_path, "training_metrics.json")

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file (JSON or YAML)")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    ensure_dirs()

    if args.config:
        config = load_config(args.config, TrainConfig)
        override_dict = {
            "env": args.env if args.env != "CartPole-v1" else None,
            "timesteps": args.timesteps if args.timesteps != 200_000 else None,
            "seed": args.seed if args.seed != 42 else None,
            "n_steps": args.n_steps if args.n_steps != 2048 else None,
            "batch_size": args.batch_size if args.batch_size != 64 else None,
            "learning_rate": args.learning_rate if args.learning_rate != 3e-4 else None,
        }
        override_dict = {k: v for k, v in override_dict.items() if v is not None}
        if override_dict:
            config = merge_config_with_args(config, override_dict)
            print(f"Loaded config from {args.config} with overrides: {override_dict}")
        else:
            print(f"Loaded config from {args.config}")
    else:
        config = TrainConfig(
            env=args.env,
            timesteps=args.timesteps,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    run_dir = create_run_dir(config)
    save_config(config, run_dir)
    print(f"Run directory: {run_dir}")

    # Train the model
    model, metrics = train_ppo(
        env_name=config.env,
        total_timesteps=config.timesteps,
        seed=config.seed,
        output_dir=str(run_dir),
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        verbose=1,
    )

    print(f"Training metrics: {metrics}")
    print(f"Saved model: {run_dir / 'model.zip'}")
    print(f"All artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
