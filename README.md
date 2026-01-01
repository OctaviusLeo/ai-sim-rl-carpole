# AI Sim RL (CartPole) - Train/Eval + Video
[![CI](https://github.com/OctaviusLeo/ai-sim-rl-cartpole/workflows/CI/badge.svg)](https://github.com/OctaviusLeo/ai-sim-rl-cartpole/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A small, hiring-ready reinforcement learning project: train PPO on CartPole, evaluate with baselines + confidence intervals, and keep runs reproducible (CI included).

## Demo
![CartPole PPO demo](assets/cartpole_demo-episode-0.gif)

## Results (example)
Generated via multi-seed evaluation (20 episodes/seed × 3 seeds) and exported to CSV/Markdown.

| Policy | Mean Return | Std | 95% CI | Success Rate |
|--------|-------------|-----|--------|--------------|
| Baseline: random | 22.95 | 11.29 | [20.01, 25.89] | 0.00% |
| Baseline: do_nothing | 9.33 | 0.79 | [9.13, 9.54] | 0.00% |
| Trained PPO | 500.00 | 0.00 | [500.00, 500.00] | 100.00% |

Artifacts: [outputs/comparisons/results.md](outputs/comparisons/results.md), [outputs/comparisons/results.csv](outputs/comparisons/results.csv)

## Skills / signals
- RL training loop + evaluation discipline (baselines, multi-seed, confidence intervals)
- Reproducibility (configs saved per run, deterministic seeds)
- ML/SWE hygiene (CI, tests, formatting, artifacts)

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install (matches CI)
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 1) Train (writes outputs/runs/<timestamp>_<env>_seed<seed>_<git-hash>/)
python -m src.train --config configs/default.yaml

# 2) Compare vs baselines (writes outputs/comparisons/results.*)
python -m src.compare --model-path outputs/runs/<run-dir>/model.zip --episodes 20 --num-seeds 3 --seed 123

# 3) Record a short demo GIF (writes videos/ and/or assets/ depending on your config)
python -m src.record_video --model-path outputs/runs/<run-dir>/model.zip --gif
```

## Repo structure
- `src/` - core scripts: `train`, `evaluate`, `compare`, `reproduce`, `record_video`, plus shared utilities in `common`
- `configs/` - YAML configs (`default.yaml`, `quick_test.yaml`)
- `outputs/runs/` - per-run artifacts (config, model, metrics, TensorBoard logs)
- `outputs/comparisons/` - exported results tables (CSV/Markdown)
- `assets/`, `videos/` - demo media

<details>
<summary><strong>More details (tracking, evaluation, reproducibility)</strong></summary>

- Per-run directories under `outputs/runs/` with saved `config.json`, metrics, and TensorBoard logs.
- Multi-seed evaluation and baseline comparisons with 95% confidence intervals.
- Reproduce any run from its saved config via `python -m src.reproduce outputs/runs/<run-dir>`.

</details>

<details>
<summary><strong>Developer Guide (optional)</strong></summary>

### Install package + CLI entrypoints

```bash
pip install -e ".[dev]"

rl-train --config configs/default.yaml
rl-evaluate --model-path outputs/runs/<run-dir>/model.zip
rl-compare --model-path outputs/runs/<run-dir>/model.zip
rl-reproduce outputs/runs/<run-dir>
rl-video --model-path outputs/runs/<run-dir>/model.zip
```

### Common dev commands

```bash
# Quality
make lint
make format
make test

# Training / evaluation
make train-quick
make evaluate
make compare
```

### Notes

- CI runs lint + tests on Python 3.8–3.10 across Linux/Windows/macOS.
- Outputs are written under `outputs/` (runs, TensorBoard logs, and comparison exports).

</details>

## Future plans
- Add a second environment (`Acrobot-v1`), compare training curves
- Add hyperparameter sweep with results table
