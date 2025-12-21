# evaluate.py
# Evaluate a trained PPO model on a given environment.
from __future__ import annotations

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from common import ensure_dirs, set_global_seed


def rollout(model: PPO, env: gym.Env, episodes: int, seed: int) -> dict:
    returns = []
    steps = []
    success = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        trunc = False
        ep_ret = 0.0
        ep_steps = 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_ret += float(reward)
            ep_steps += 1

        returns.append(ep_ret)
        steps.append(ep_steps)

        # For CartPole, max reward is 500; treat >=475 as "solved-ish"
        if ep_ret >= 475:
            success += 1

    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": float(success / episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--model-path", default="outputs/cartpole_ppo.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    ensure_dirs()
    set_global_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path} (train first)")

    env = gym.make(args.env)
    model = PPO.load(args.model_path, env=env)

    metrics = rollout(model, env, args.episodes, args.seed)
    print("Eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
