"""Plot ParkPilot training curves for the README / blog post.

Reads the TensorBoard event file of a run (default: the 300k randomized SAC_5
run) plus the EvalCallback's ``evaluations.npz`` and writes a two-panel figure:

  * left  — episode reward over timesteps (noisy rollout signal + deterministic
            eval mean with a ±1σ band)
  * right — success rate over timesteps (rollout + deterministic eval)

Usage:
    python parkpilot/plot_results.py                 # SAC_5 -> training_curves.png
    python parkpilot/plot_results.py --run runs/SAC_3 --out phase3_curves.png
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG without a display
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(ea: EventAccumulator, tag: str):
    """Return (steps, values) arrays for a scalar tag, or empty arrays if absent."""
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    return steps, vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/SAC_5", help="TensorBoard run directory")
    p.add_argument("--evalnpz", default="runs/eval/evaluations.npz",
                   help="EvalCallback evaluations.npz (optional)")
    p.add_argument("--out", default="training_curves.png")
    p.add_argument("--title", default="ParkPilot — SAC, domain-randomized starts (300k steps)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    event_files = glob.glob(str(Path(args.run) / "events*"))
    if not event_files:
        print(f"error: no TensorBoard events under {args.run}")
        sys.exit(1)
    ea = EventAccumulator(event_files[0])
    ea.Reload()

    rew_steps, rew_vals = load_scalar(ea, "rollout/ep_rew_mean")
    suc_steps, suc_vals = load_scalar(ea, "rollout/success_rate")

    # Deterministic eval timeline (preferred for the headline curve).
    eval_ts = eval_rew_mean = eval_rew_std = eval_suc = None
    npz_path = Path(args.evalnpz)
    if npz_path.is_file():
        d = np.load(npz_path)
        eval_ts = d["timesteps"].astype(float)
        eval_rew_mean = d["results"].mean(axis=1)
        eval_rew_std = d["results"].std(axis=1)
        eval_suc = d["successes"].mean(axis=1)

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Reward panel ---
    if rew_steps.size:
        ax_r.plot(rew_steps, rew_vals, color="#9bbcff", lw=1.2,
                  label="rollout (stochastic)")
    if eval_ts is not None:
        ax_r.plot(eval_ts, eval_rew_mean, color="#1f4ed8", lw=2.0,
                  label="eval (deterministic)")
        ax_r.fill_between(eval_ts, eval_rew_mean - eval_rew_std,
                          eval_rew_mean + eval_rew_std, color="#1f4ed8", alpha=0.15)
    ax_r.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    ax_r.set_xlabel("timesteps")
    ax_r.set_ylabel("episode reward")
    ax_r.set_title("Episode reward")
    ax_r.legend(loc="lower right", fontsize=9)
    ax_r.grid(alpha=0.25)

    # --- Success-rate panel ---
    if suc_steps.size:
        ax_s.plot(suc_steps, suc_vals, color="#9bd6a3", lw=1.2,
                  label="rollout (stochastic)")
    if eval_ts is not None:
        ax_s.plot(eval_ts, eval_suc, color="#1f8a3b", lw=2.0,
                  label="eval (deterministic)")
    ax_s.axhline(0.8, color="#d62728", lw=1.0, ls="--", label="0.80 target")
    ax_s.set_ylim(-0.02, 1.02)
    ax_s.set_xlabel("timesteps")
    ax_s.set_ylabel("success rate")
    ax_s.set_title("Success rate")
    ax_s.legend(loc="lower right", fontsize=9)
    ax_s.grid(alpha=0.25)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    # Print the final numbers the README/blog will quote.
    if eval_ts is not None:
        print(f"final eval: reward={eval_rew_mean[-1]:.1f}±{eval_rew_std[-1]:.1f} "
              f"success={eval_suc[-1]:.0%} at step {int(eval_ts[-1])}")


if __name__ == "__main__":
    main()
