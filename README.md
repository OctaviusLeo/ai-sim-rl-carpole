# AI Sim RL (CartPole) - Train/Eval + Video
[![CI](https://github.com/OctaviusLeo/ai-sim-rl-cartpole/workflows/CI/badge.svg)](https://github.com/OctaviusLeo/ai-sim-rl-cartpole/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Besides simulations being cool, implementing AI is used to test.

A minimal, **working** reinforcement learning demo, can extend.
- Trains a PPO agent on `CartPole-v1` (Gymnasium).
- Config file system with YAML/JSON support and CLI overrides.
- Per-run experiment tracking with config and metrics.
- TensorBoard logging for training visualization.
- Multi-seed evaluation with confidence intervals.
- Baseline comparison suite with statistical rigor.
- CSV and Markdown export for publication-ready tables.
- Reproduce exact runs from saved configurations.
- Optionally records a short rollout video.

## Demo

![CartPole PPO demo](assets/cartpole_demo-episode-0.gif)

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Train with experiment tracking (creates outputs/runs/<timestamp>_<env>_seed<seed>_<git-hash>/)
python src/train.py --timesteps 200000 --seed 42

# Train using a config file (YAML or JSON)
python src/train.py --config configs/default.yaml

# Train with config file and override specific parameters
python src/train.py --config configs/default.yaml --seed 999 --timesteps 50000

# Reproduce exact run from saved config
python src/reproduce.py outputs/runs/<run-dir>

# Evaluate with multi-seed runs and confidence intervals
python src/evaluate.py --model-path outputs/runs/<run-dir>/model.zip --episodes 20 --num-seeds 5 --save-results

# Compare against baselines with CSV/Markdown export
python src/compare.py --model-path outputs/runs/<run-dir>/model.zip --episodes 20 --num-seeds 5

# View training progress in TensorBoard
tensorboard --logdir outputs/runs/<run-dir>/tensorboard

# Record a short demo video (saved to videos/)
python src/record_video.py --model-path outputs/runs/<run-dir>/model.zip

# Record video and also export a GIF
python src/record_video.py --model-path outputs/runs/<run-dir>/model.zip --gif
```

## Repo structure
- `src/train.py` - trains PPO and saves the model with experiment tracking
- `src/evaluate.py` - evaluates the saved model with multi-seed support
- `src/compare.py` - compares models against baselines with statistical tests
- `src/reproduce.py` - reproduces a training run from saved config
- `src/record_video.py` - records a short video via Gymnasium RecordVideo wrapper
- `src/common.py` - shared utilities for experiment tracking and reproducibility
- `configs/` - example configuration files (YAML)
  - `default.yaml` - standard training configuration
  - `quick_test.yaml` - fast iteration configuration
- `outputs/runs/` - per-run experiment directories (created at runtime)
  - Each run contains: `config.json`, `model.zip`, `training_metrics.json`, `training_returns.png`, `tensorboard/`
- `outputs/comparisons/` - baseline comparison results (CSV and Markdown)
- `videos/` - recorded demos (created at runtime)

## Experiment Tracking Features

**Per-run directories**: Each training run creates a unique directory with timestamp, environment, seed, and git hash.

**Configuration management**: All hyperparameters saved to `config.json` for full reproducibility.

**Metrics tracking**: Training metrics (episode counts, returns) saved to `training_metrics.json`.

**TensorBoard logging**: Detailed training curves viewable with `tensorboard --logdir outputs/runs/<run-dir>/tensorboard`.

**Multi-seed evaluation**: Evaluate across multiple seeds with aggregated statistics and 95% confidence intervals.

**Saved evaluation results**: Use `--save-results` to export evaluation metrics to JSON.

## Configuration System

**Config file support**: Define all training hyperparameters in YAML or JSON files for clean, version-controlled experiments.

**CLI override mechanism**: Load a config file and selectively override parameters via command-line arguments.

**Automatic config saving**: Every training run saves its complete configuration to `config.json` in the run directory.

**Reproducible runs**: Use `reproduce.py` to exactly recreate any previous training run from its saved config.

**Example configs provided**: `configs/default.yaml` for standard training and `configs/quick_test.yaml` for rapid iteration.

**Flexible workflow**: Use config files for base settings and CLI args for quick experiments without editing files.

## Evaluation Suite

**Baseline policies**: Compare trained models against random and do-nothing policies for statistical validation.

**Multi-seed robustness**: Evaluate across multiple random seeds to ensure results are not cherry-picked.

**Statistical rigor**: Compute 95% confidence intervals using t-distribution for proper uncertainty quantification.

**Publication-ready exports**: Generate CSV files for spreadsheets and Markdown tables for documentation.

**Success rate tracking**: Automatically track episodes that reach near-maximum reward (configurable threshold).

**Comparison tables**: View side-by-side performance of baselines vs trained model with mean, std, CI, min, max.

### Example Results

Comparison of trained PPO agent against baseline policies on CartPole-v1 (100 episodes across 5 seeds):

| Policy | Mean Return | Std | 95% CI | Success Rate | Min | Max |
|--------|-------------|-----|--------|--------------|-----|-----|
| Baseline: random | 21.04 | 11.15 | [18.8, 23.3] | 0.0% | 9 | 66 |
| Baseline: do_nothing | 9.35 | 0.78 | [9.2, 9.5] | 0.0% | 8 | 11 |
| Trained PPO | 500.00 | 0.00 | [500.0, 500.0] | 100.0% | 500 | 500 |

The trained agent achieves perfect performance, solving the environment in all 100 evaluation episodes.

## Developer Guide

### Package Installation

Install the package in development mode to use CLI entrypoints:

```bash
# Install package
pip install -e .

# Now use CLI commands instead of python src/...
rl-train --config configs/default.yaml
rl-evaluate --model-path outputs/runs/<run-dir>/model.zip
rl-compare --model-path outputs/runs/<run-dir>/model.zip
rl-reproduce outputs/runs/<run-dir>
rl-video --model-path outputs/runs/<run-dir>/model.zip
```

### Automation Scripts

Use `make.bat` (Windows) or `make` (Unix) for common tasks:

```bash
# Windows
make install          # Install package in development mode
make train            # Train with default config
make train-quick      # Quick 10k timestep training
make evaluate         # Evaluate most recent model
make compare          # Compare most recent model against baselines
make tensorboard      # Launch TensorBoard for most recent run
make clean            # Remove temporary files

# Unix/macOS
make install
make train
make evaluate
```

### Project Structure

```
ai-sim-rl-cartpole/
 src/                      # Source code
    __init__.py          # Package initialization
    common.py            # Shared utilities
    train.py             # Training script
    evaluate.py          # Evaluation script
    compare.py           # Baseline comparison
    reproduce.py         # Reproduce runs
    record_video.py      # Video recording
 configs/                  # Configuration files
    default.yaml         # Standard training
    quick_test.yaml      # Fast iteration
 outputs/                  # Generated outputs
    runs/                # Training runs
    comparisons/         # Comparison results
 videos/                   # Recorded videos
 setup.py                  # Package configuration
 requirements.txt          # Dependencies
 make.bat                  # Windows automation
 Makefile                  # Unix automation
```

### Development Workflow

1. **Setup environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev]"
   ```

2. **Quick iteration**:
   ```bash
   make train-quick           # Fast training for testing
   make evaluate              # Check results
   ```

3. **Full training**:
   ```bash
   make train                 # Standard 200k timestep training
   make compare               # Statistical comparison
   ```

4. **Custom experiments**:
   ```bash
   rl-train --config configs/default.yaml --seed 999 --timesteps 50000
   rl-reproduce outputs/runs/<run-dir>
   ```

5. **Testing and quality**:
   ```bash
   make test                  # Run all tests
   make test-cov              # Run tests with coverage report
   make lint                  # Check code quality with flake8
   make format                # Auto-format code with black
   ```

### Testing

The project includes comprehensive test coverage:

**Test Categories:**
- **Determinism tests**: Verify identical seeds produce identical results
- **Smoke tests**: Ensure basic training runs without errors
- **Utility tests**: Test helper functions and config management
- **Evaluation tests**: Validate evaluation metrics and multi-seed aggregation

**Running tests:**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report

# Run specific test file
pytest tests/test_determinism.py -v

# Run tests with specific markers
pytest tests/ -v -m "not slow"
```

**Continuous Integration:**
- GitHub Actions CI runs on every push and PR
- Tests on Python 3.8, 3.9, 3.10 across Linux, Windows, and macOS
- Code linting with flake8
- Code formatting checks with black
- Coverage reporting to Codecov

### Tips for Interviewers

### Tips for Interviewers

This project demonstrates:
- **Scientific rigor**: Multi-seed evaluation with confidence intervals
- **Reproducibility**: Deterministic training with seed control and config tracking
- **Professional packaging**: Installable package with CLI entrypoints
- **Developer ergonomics**: Automation scripts for common workflows
- **Clean architecture**: Modular design with clear separation of concerns
- **Production practices**: Proper project structure and documentation
- **Software engineering**: Comprehensive test suite with CI/CD pipeline
- **Code quality**: Automated linting and formatting standards

## Future plans
- Add a second environment (`Acrobot-v1`), compare training curves
- Add hyperparameter sweep with results table
