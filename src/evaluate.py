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

try:
    from .common import ensure_dirs, save_metrics, set_global_seed
except ImportError:
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


def evaluate_model(
    model: PPO, env_name: str, n_episodes: int = 10, seed: int = 0, success_threshold: float = 475.0
):
    """
    Evaluate a trained model on an environment.

    Args:
        model: Trained PPO model
        env_name: Name of the environment
        n_episodes: Number of episodes to run
        seed: Random seed for evaluation
        success_threshold: Threshold for counting successes

    Returns:
        Tuple of (returns array, success_rate)
    """
    env = gym.make(env_name)
    returns = []
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        trunc = False
        ep_ret = 0.0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            ep_ret += float(reward)

        returns.append(ep_ret)
        if ep_ret >= success_threshold:
            successes += 1

    env.close()
    return np.array(returns), successes / n_episodes


def multi_seed_evaluation(
    model: PPO, env_name: str, episodes_per_seed: int = 20, num_seeds: int = 3, base_seed: int = 0
) -> dict:
    """
    Evaluate model across multiple seeds and aggregate results.

    Args:
        model: Trained PPO model
        env_name: Name of the environment
        episodes_per_seed: Number of episodes per seed
        num_seeds: Number of different seeds to use
        base_seed: Starting seed value

    Returns:
        Dictionary with aggregated evaluation metrics
    """
    all_returns = []
    all_success_rates = []

    for i in range(num_seeds):
        seed = base_seed + i * 1000
        returns, success_rate = evaluate_model(model, env_name, episodes_per_seed, seed)
        all_returns.extend(returns)
        all_success_rates.append(success_rate)

    all_returns = np.array(all_returns)
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)

    # Compute confidence interval
    n = len(all_returns)
    stderr = stats.sem(all_returns)
    ci_margin = stderr * stats.t.ppf(0.975, n - 1)

    return {
        "num_seeds": num_seeds,
        "total_episodes": len(all_returns),
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "ci_95_low": float(mean_return - ci_margin),
        "ci_95_high": float(mean_return + ci_margin),
        "mean_success_rate": float(np.mean(all_success_rates)),
        "std_success_rate": float(np.std(all_success_rates)),
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
    parser.add_argument(
        "--save-results", action="store_true", help="Save evaluation results to JSON"
    )
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
            print(
                f"  Seed {seed}: mean_return={result['mean_return']:.2f}, success_rate={result['success_rate']:.2f}"
            )

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
