# compare.py
# Compare trained models against baseline policies with statistical rigor.
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
from scipy import stats
from stable_baselines3 import PPO

from common import ensure_dirs, set_global_seed


class BaselinePolicy:
    def __init__(self, policy_type: str, action_space):
        self.policy_type = policy_type
        self.action_space = action_space

    def predict(self, obs):
        if self.policy_type == "random":
            return self.action_space.sample(), None
        elif self.policy_type == "do_nothing":
            return 0, None
        elif self.policy_type == "always_right":
            return 1, None
        else:
            raise ValueError(f"Unknown baseline policy: {self.policy_type}")


def evaluate_policy(policy, env: gym.Env, episodes: int, seed: int, deterministic: bool = True) -> dict:
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
            if isinstance(policy, BaselinePolicy):
                action, _ = policy.predict(obs)
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, info = env.step(action)
            ep_ret += float(reward)
            ep_steps += 1

        returns.append(ep_ret)
        steps.append(ep_steps)

        if ep_ret >= 475:
            success += 1

    return {
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": float(success / episodes),
    }


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    n = len(data)
    if n < 2:
        return (float(np.mean(data)), float(np.mean(data)))
    mean = np.mean(data)
    stderr = stats.sem(data)
    interval = stderr * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return (float(mean - interval), float(mean + interval))


def run_multi_seed_evaluation(policy, env: gym.Env, episodes: int, base_seed: int, num_seeds: int) -> dict:
    all_results = []
    for i in range(num_seeds):
        seed = base_seed + i * 1000
        set_global_seed(seed)
        result = evaluate_policy(policy, env, episodes, seed)
        all_results.append(result)

    all_returns = [r for result in all_results for r in result["returns"]]
    all_success = [result["success_rate"] for result in all_results]
    ci_low, ci_high = compute_confidence_interval(all_returns)

    return {
        "num_seeds": num_seeds,
        "total_episodes": len(all_returns),
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "min_return": float(np.min(all_returns)),
        "max_return": float(np.max(all_returns)),
        "mean_success_rate": float(np.mean(all_success)),
        "std_success_rate": float(np.std(all_success)),
    }


def export_to_csv(results: Dict[str, dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Policy",
            "Mean Return",
            "Std Return",
            "CI 95% Low",
            "CI 95% High",
            "Success Rate",
            "Min Return",
            "Max Return",
        ])
        for policy_name, metrics in results.items():
            writer.writerow([
                policy_name,
                f"{metrics['mean_return']:.2f}",
                f"{metrics['std_return']:.2f}",
                f"{metrics['ci_95_low']:.2f}",
                f"{metrics['ci_95_high']:.2f}",
                f"{metrics['mean_success_rate']:.2%}",
                f"{metrics['min_return']:.2f}",
                f"{metrics['max_return']:.2f}",
            ])


def export_to_markdown(results: Dict[str, dict], output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Policy | Mean Return | Std | 95% CI | Success Rate | Min | Max |\n")
        f.write("|--------|-------------|-----|--------|--------------|-----|-----|\n")

        for policy_name, metrics in results.items():
            ci_str = f"[{metrics['ci_95_low']:.1f}, {metrics['ci_95_high']:.1f}]"
            f.write(
                f"| {policy_name} "
                f"| {metrics['mean_return']:.2f} "
                f"| {metrics['std_return']:.2f} "
                f"| {ci_str} "
                f"| {metrics['mean_success_rate']:.1%} "
                f"| {metrics['min_return']:.0f} "
                f"| {metrics['max_return']:.0f} |\n"
            )

        f.write("\n## Detailed Metrics\n\n")
        for policy_name, metrics in results.items():
            f.write(f"### {policy_name}\n\n")
            f.write(f"- **Mean Return**: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}\n")
            f.write(f"- **95% Confidence Interval**: [{metrics['ci_95_low']:.2f}, {metrics['ci_95_high']:.2f}]\n")
            f.write(f"- **Success Rate**: {metrics['mean_success_rate']:.2%}\n")
            f.write(f"- **Range**: [{metrics['min_return']:.0f}, {metrics['max_return']:.0f}]\n")
            f.write(f"- **Total Episodes**: {metrics['total_episodes']} ({metrics['num_seeds']} seeds)\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained models against baseline policies")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    parser.add_argument("--output-dir", default="outputs/comparisons", help="Directory for results")
    parser.add_argument("--baselines", nargs="+", default=["random", "do_nothing"], 
                        help="Baseline policies to compare against")
    args = parser.parse_args()

    ensure_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    env = gym.make(args.env)
    results = {}

    print(f"\nRunning comprehensive evaluation on {args.env}")
    print(f"Episodes per seed: {args.episodes}, Number of seeds: {args.num_seeds}\n")

    for baseline in args.baselines:
        print(f"Evaluating baseline: {baseline}...")
        baseline_policy = BaselinePolicy(baseline, env.action_space)
        results[f"Baseline: {baseline}"] = run_multi_seed_evaluation(
            baseline_policy, env, args.episodes, args.seed, args.num_seeds
        )
        print(f"  Mean return: {results[f'Baseline: {baseline}']['mean_return']:.2f} "
              f"± {results[f'Baseline: {baseline}']['std_return']:.2f}")

    print(f"\nEvaluating trained model: {Path(args.model_path).parent.name}...")
    model = PPO.load(args.model_path, env=env)
    results["Trained PPO"] = run_multi_seed_evaluation(
        model, env, args.episodes, args.seed, args.num_seeds
    )
    print(f"  Mean return: {results['Trained PPO']['mean_return']:.2f} "
          f"± {results['Trained PPO']['std_return']:.2f}")

    csv_path = output_dir / "results.csv"
    export_to_csv(results, csv_path)
    print(f"\nExported CSV to: {csv_path}")

    md_path = output_dir / "results.md"
    export_to_markdown(results, md_path)
    print(f"Exported Markdown to: {md_path}")

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for policy_name, metrics in results.items():
        print(f"\n{policy_name}:")
        print(f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
        print(f"  95% CI: [{metrics['ci_95_low']:.2f}, {metrics['ci_95_high']:.2f}]")
        print(f"  Success Rate: {metrics['mean_success_rate']:.1%}")


if __name__ == "__main__":
    main()
