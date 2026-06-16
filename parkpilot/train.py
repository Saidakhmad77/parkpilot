import os

os.environ.setdefault("MUJOCO_GL", "cgl")

import argparse
from collections import Counter
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

from parkpilot.env import ParkingEnv


def make_env(seed: int = 0, randomize: bool = False):
    env = ParkingEnv(randomize=randomize)
    env.reset(seed=seed)
    return Monitor(env, info_keywords=("is_success",))


class TerminalReasonCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._counts: Counter[str] = Counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("terminal_reason", "") != "":
                self._counts[info["terminal_reason"]] += 1

        if self.num_timesteps > 0 and self.num_timesteps % 5000 == 0:
            for reason, count in self._counts.items():
                self.logger.record(f"terminals/{reason}", count)
            print(
                f"[step {self.num_timesteps}] "
                f"success={self._counts['success']} "
                f"collision={self._counts['collision']} "
                f"oob={self._counts['oob']} "
                f"timeout={self._counts['timeout']}"
            )

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_final_eval(model: SAC, env: Monitor, seed: int) -> None:
    successes = 0

    for episode_idx in range(10):
        obs, _ = env.reset(seed=seed + episode_idx)
        done = False
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = bool(info.get("is_success", False))

        successes += int(success)

    success_rate = successes / 10
    print(f"final_eval_success_rate={success_rate:.0%} ({successes}/10)")


def main() -> None:
    args = parse_args()

    os.makedirs("runs/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    set_random_seed(args.seed)

    env = make_env(args.seed, randomize=True)
    eval_env = make_env(args.seed + 1, randomize=True)

    try:
        env.reset(seed=args.seed)

        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=200000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            learning_starts=1000,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log="runs/",
            device="cpu",
            seed=args.seed,
            verbose=1,
        )

        eval_cb = EvalCallback(
            eval_env,
            eval_freq=5000,
            n_eval_episodes=10,
            deterministic=True,
            best_model_save_path="models/",
            log_path="runs/eval/",
        )
        terminal_cb = TerminalReasonCallback()

        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, terminal_cb],
            progress_bar=False,
        )

        model.save("models/sac_parking")
        print("saved models/sac_parking.zip")

        run_final_eval(model, eval_env, args.seed + 1000)
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
