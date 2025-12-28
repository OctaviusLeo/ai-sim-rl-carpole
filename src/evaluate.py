# evaluate.py
# Evaluate a trained PPO model on a given environment.
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
from scipy import stats
from stable_baselines3 import PPO

from common import ensure_dirs, save_metrics, set_global_seed


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

        if ep_ret >= 475:
            success += 1

    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": float(success / episodes),
        "returns": returns,
    }


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    n = len(data)
    if n < 2:
        return (float(np.mean(data)), float(np.mean(data)))
    mean = np.mean(data)
    stderr = stats.sem(data)
    interval = stderr * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return (float(mean - interval), float(mean + interval))


def aggregate_results(all_results: List[dict]) -> dict:
    all_returns = [r for result in all_results for r in result["returns"]]
    all_success = [result["success_rate"] for result in all_results]

    ci_low, ci_high = compute_confidence_interval(all_returns)

    return {
        "num_seeds": len(all_results),
        "total_episodes": len(all_returns),
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "mean_success_rate": float(np.mean(all_success)),
        "std_success_rate": float(np.std(all_success)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds for evaluation")
    parser.add_argument("--save-results", action="store_true", help="Save evaluation results to JSON")
    args = parser.parse_args()

    ensure_dirs()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path} (train first)")

    env = gym.make(args.env)
    model = PPO.load(args.model_path, env=env)

    if args.num_seeds == 1:
        set_global_seed(args.seed)
        metrics = rollout(model, env, args.episodes, args.seed)
        print("\nEvaluation metrics:")
        for k, v in metrics.items():
            if k != "returns":
                print(f"  {k}: {v}")

        if args.save_results:
            model_dir = Path(args.model_path).parent
            save_metrics(metrics, model_dir, "eval_metrics.json")
            print(f"\nSaved results to: {model_dir / 'eval_metrics.json'}")
    else:
        print(f"\nRunning multi-seed evaluation with {args.num_seeds} seeds...")
        all_results = []
        for i in range(args.num_seeds):
            seed = args.seed + i * 1000
            set_global_seed(seed)
            result = rollout(model, env, args.episodes, seed)
            all_results.append(result)
            print(f"  Seed {seed}: mean_return={result['mean_return']:.2f}, success_rate={result['success_rate']:.2f}")

        aggregated = aggregate_results(all_results)
        print("\nAggregated results across seeds:")
        for k, v in aggregated.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        if args.save_results:
            model_dir = Path(args.model_path).parent
            save_metrics(aggregated, model_dir, "eval_metrics_aggregated.json")
            print(f"\nSaved aggregated results to: {model_dir / 'eval_metrics_aggregated.json'}")


if __name__ == "__main__":
    main()
