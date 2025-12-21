# train.py
# This script trains a PPO model on the CartPole-v1 environment and logs the training reward.
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from common import ensure_dirs, set_global_seed, Paths


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    set_global_seed(args.seed)

    env = gym.make(args.env)
    env.reset(seed=args.seed)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    cb = RewardLogger()
    model.learn(total_timesteps=args.timesteps, callback=cb)

    model_path = os.path.join(Paths.outputs_dir, "cartpole_ppo")
    model.save(model_path)

    # Plot training reward (episode return)
    if len(cb.episode_rewards) > 0:
        plt.figure()
        plt.plot(cb.episode_rewards)
        plt.title("Training Episode Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plot_path = os.path.join(Paths.outputs_dir, "training_returns.png")
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {plot_path}")

    print(f"Saved model: {model_path}.zip")


if __name__ == "__main__":
    main()
