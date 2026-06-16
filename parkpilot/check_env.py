"""Phase 2 environment sanity checks.

Run via: uv run python parkpilot/check_env.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "cgl")

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymnasium.utils.env_checker import check_env

from parkpilot.env import MAX_EPISODE_STEPS, ParkingEnv


def run_checker() -> None:
    env = ParkingEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    print("check_env PASS")


def run_collision_checks() -> None:
    env = ParkingEnv()
    try:
        env.reset()
        env._x = 2.0
        env._y = 0.55
        env._theta = 0.0
        env._v = 0.0
        env._steer = 0.0
        env._write_sim(0.0)
        assert env._check_collision() is True
        print("collision check (near left_parked_car): PASS")

        env.reset()
        assert env._check_collision() is False
        print("collision check (start pose): PASS")
    finally:
        env.close()


def run_random_episodes() -> None:
    env = ParkingEnv()
    try:
        for episode_idx in range(3):
            env.reset()
            total_reward = 0.0
            terminal_reason = ""
            steps = 0

            for _ in range(MAX_EPISODE_STEPS):
                _, reward, terminated, truncated, info = env.step(env.action_space.sample())
                total_reward += float(reward)
                steps += 1
                terminal_reason = str(info["terminal_reason"])
                if terminated or truncated:
                    break

            print(
                f"random episode {episode_idx + 1}: "
                f"total_reward={total_reward:.3f}, "
                f"steps={steps}, "
                f"terminal_reason={terminal_reason!r}"
            )
    finally:
        env.close()

    print("random episode check PASS")


def main() -> None:
    run_checker()
    run_collision_checks()
    run_random_episodes()


if __name__ == "__main__":
    main()
