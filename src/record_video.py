# record_video.py
# This script records a video of the trained model in action.
from __future__ import annotations

import argparse
import os
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from moviepy.editor import VideoFileClip
from stable_baselines3 import PPO

try:
    from .common import Paths, ensure_dirs, set_global_seed
except ImportError:
    from common import Paths, ensure_dirs, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--model-path", default="outputs/cartpole_ppo.zip")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gif", action="store_true", help="Also export a GIF next to the mp4")
    args = parser.parse_args()

    ensure_dirs()
    set_global_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path} (train first)")

    env = gym.make(args.env, render_mode="rgb_array")
    # Record only the first episode to keep a single clean video asset.
    env = RecordVideo(
        env,
        video_folder=Paths.videos_dir,
        episode_trigger=lambda ep: ep == 0,
        name_prefix="cartpole_demo",
    )
    obs, info = env.reset(seed=args.seed)

    model = PPO.load(args.model_path, env=env)

    recorded_path: Optional[str] = None

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # Capture the path before resetting to avoid losing the reference.
            if hasattr(env, "video_recorder") and env.video_recorder is not None:
                recorded_path = env.video_recorder.path
            obs, info = env.reset()

    env.close()

    # Fallback: try to pick the newest mp4 in the videos directory.
    if recorded_path is None:
        if hasattr(env, "video_recorder") and env.video_recorder is not None:
            recorded_path = env.video_recorder.path
        if recorded_path is None:
            mp4s = [
                os.path.join(Paths.videos_dir, f)
                for f in os.listdir(Paths.videos_dir)
                if f.endswith(".mp4")
            ]
            recorded_path = max(mp4s, key=os.path.getmtime) if mp4s else None

    if recorded_path is None:
        raise RuntimeError(
            "No video was recorded; ensure steps > 0 and the env terminated at least once."
        )

    print(f"Video saved to: {recorded_path}")

    if args.gif:
        gif_path = os.path.splitext(recorded_path)[0] + ".gif"
        # Convert the MP4 to GIF with MoviePy; keep FPS modest to reduce file size.
        clip = VideoFileClip(recorded_path)
        clip.write_gif(gif_path, fps=min(clip.fps, 15))
        clip.close()
        print(f"GIF saved to: {gif_path}")
    else:
        print("GIF export skipped (use --gif to enable).")


if __name__ == "__main__":
    main()
