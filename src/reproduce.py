# reproduce.py
# This script reproduces a training run from a saved config.json file.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

try:
    from .common import (
        TrainConfig,
        load_config,
        save_metrics,
        set_global_seed,
    )
except ImportError:
    from common import (
        TrainConfig,
        load_config,
        save_metrics,
        set_global_seed,
    )


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current = 0.0

    def _on_step(self) -> bool:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce a training run from saved config")
    parser.add_argument(
        "run_dir", type=str, help="Path to the run directory containing config.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for reproduced run (default: same as original)",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        sys.exit(1)

    config_path = run_path / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {args.run_dir}")
        sys.exit(1)

    config = load_config(run_path, TrainConfig)
    print(f"Loaded configuration from {config_path}")
    print(f"  Environment: {config.env}")
    print(f"  Timesteps: {config.timesteps}")
    print(f"  Seed: {config.seed}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    set_global_seed(config.seed)

    output_dir = Path(args.output_dir) if args.output_dir else run_path
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving output to: {output_dir}")

    env = gym.make(config.env)
    env.reset(seed=config.seed)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gae_lambda=config.gae_lambda,
        gamma=config.gamma,
        n_epochs=config.n_epochs,
        ent_coef=config.ent_coef,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    cb = RewardLogger()
    model.learn(total_timesteps=config.timesteps, callback=cb)

    model_path = output_dir / "model"
    model.save(str(model_path))

    if len(cb.episode_rewards) > 0:
        plt.figure()
        plt.plot(cb.episode_rewards)
        plt.title("Training Episode Return (Reproduced)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plot_path = output_dir / "training_returns_reproduced.png"
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")

        metrics = {
            "total_episodes": len(cb.episode_rewards),
            "final_mean_return": float(
                sum(cb.episode_rewards[-100:]) / min(100, len(cb.episode_rewards))
            ),
            "max_return": float(max(cb.episode_rewards)),
            "min_return": float(min(cb.episode_rewards)),
        }
        save_metrics(metrics, output_dir, "training_metrics_reproduced.json")
        print(f"Training metrics: {metrics}")

    print(f"Saved model: {model_path}.zip")
    print(f"All artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
