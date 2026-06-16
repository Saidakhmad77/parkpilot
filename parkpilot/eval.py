import os

os.environ.setdefault("MUJOCO_GL", "cgl")

import argparse
from collections import Counter
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v3 as iio
import numpy as np
from stable_baselines3 import SAC

from parkpilot.env import ParkingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sac_parking.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gif", default="eval.gif")
    parser.add_argument("--randomize", action="store_true", default=True, help="Use randomized starts (default: True)")
    parser.add_argument("--fixed", action="store_true", default=False, help="Use fixed start (sets randomize=False)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for first reset (reproducible sequence)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"error: model file not found: {model_path}")
        sys.exit(1)

    model = SAC.load(args.model, device="cpu")
    randomize = not args.fixed  # --fixed overrides --randomize
    env = ParkingEnv(render_mode="rgb_array", randomize=randomize)

    rewards: list[float] = []
    steps_per_episode: list[int] = []
    successes = 0
    terminal_reasons: Counter[str] = Counter()

    gif_frames = []
    gif_episode_chosen = False
    gif_episode_number: int | None = None

    try:
        for episode_idx in range(args.episodes):
            # Seed the first reset for a reproducible episode sequence; subsequent resets vary naturally
            obs, _ = env.reset(seed=args.seed if episode_idx == 0 else None)
            total_reward = 0.0
            steps = 0
            terminal_reason = ""
            success = False
            done = False
            episode_frames = [] if not gif_episode_chosen else None

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                if episode_frames is not None:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)

                total_reward += float(reward)
                steps += 1
                terminal_reason = str(info.get("terminal_reason", terminal_reason))
                success = bool(info.get("is_success", False))
                done = terminated or truncated

            rewards.append(total_reward)
            steps_per_episode.append(steps)
            successes += int(success)
            terminal_reasons[terminal_reason] += 1

            if episode_idx == 0 and episode_frames is not None:
                gif_frames = episode_frames
                gif_episode_number = 1

            if success and not gif_episode_chosen and episode_frames is not None:
                gif_frames = episode_frames
                gif_episode_chosen = True
                gif_episode_number = episode_idx + 1
    finally:
        env.close()

    success_rate = (successes / args.episodes) * 100 if args.episodes else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_steps = float(np.mean(steps_per_episode)) if steps_per_episode else 0.0

    print(f"success_rate={success_rate:.1f}% ({successes}/{args.episodes})")
    print(f"mean_reward={mean_reward:.3f}")
    print(f"mean_steps={mean_steps:.1f}")
    print(f"terminal_reasons={terminal_reasons}")

    if gif_frames:
        iio.imwrite(args.gif, np.asarray(gif_frames), duration=0.05, loop=0)
        print(f"wrote GIF from episode {gif_episode_number}: {args.gif}")
    else:
        print("no frames captured; skipped GIF")


if __name__ == "__main__":
    main()
