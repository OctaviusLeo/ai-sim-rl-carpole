# record_video.py
# This script records a video of the trained model in action.
from __future__ import annotations

import argparse
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from common import ensure_dirs, set_global_seed, Paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--model-path", default="outputs/cartpole_ppo.zip")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    ensure_dirs()
    set_global_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path} (train first)")

    env = gym.make(args.env, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=Paths.videos_dir, episode_trigger=lambda ep: True, name_prefix="cartpole_demo")
    obs, info = env.reset(seed=args.seed)

    model = PPO.load(args.model_path, env=env)

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"Video saved under: {Paths.videos_dir}/")


if __name__ == "__main__":
    main()
