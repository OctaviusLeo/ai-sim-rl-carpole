# AI Sim RL (CartPole) - Train/Eval + Video
Besides simulations being cool, implementing AI is used to test.

A minimal, **working** reinforcement learning demo you can extend.
- Trains a PPO agent on `CartPole-v1` (Gymnasium).
- Per-run experiment tracking with config and metrics.
- TensorBoard logging for training visualization.
- Multi-seed evaluation with confidence intervals.
- Optionally records a short rollout video.

Demo:

![CartPole PPO demo (mp4->gif)](videos/cartpole_demo-episode-0.gif)

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Train with experiment tracking (creates outputs/runs/<timestamp>_<env>_seed<seed>_<git-hash>/)
python src/train.py --timesteps 200000 --seed 42

# Evaluate with multi-seed runs and confidence intervals
python src/evaluate.py --model-path outputs/runs/<run-dir>/model.zip --episodes 20 --num-seeds 5 --save-results

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
- `src/record_video.py` - records a short video via Gymnasium RecordVideo wrapper
- `src/common.py` - shared utilities for experiment tracking and reproducibility
- `outputs/runs/` - per-run experiment directories (created at runtime)
  - Each run contains: `config.json`, `model.zip`, `training_metrics.json`, `training_returns.png`, `tensorboard/`
- `videos/` - recorded demos (created at runtime)

## Experiment Tracking Features

**Per-run directories**: Each training run creates a unique directory with timestamp, environment, seed, and git hash.

**Configuration management**: All hyperparameters saved to `config.json` for full reproducibility.

**Metrics tracking**: Training metrics (episode counts, returns) saved to `training_metrics.json`.

**TensorBoard logging**: Detailed training curves viewable with `tensorboard --logdir outputs/runs/<run-dir>/tensorboard`.

**Multi-seed evaluation**: Evaluate across multiple seeds with aggregated statistics and 95% confidence intervals.

**Saved evaluation results**: Use `--save-results` to export evaluation metrics to JSON.
## Future plans
- Add a second environment (`Acrobot-v1`), compare training curves
- Add hyperparameter sweep with results table
- Add tests and CI pipeline
- Package as installable module with CLI entrypoints