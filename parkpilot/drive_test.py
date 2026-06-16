"""Phase 1 scripted drive test: validate MuJoCo rendering with bicycle kinematics.

The car body has independent X, Y, and yaw joints in MJCF. Those joints do not
create car-like motion by themselves, so this script drives the shared bicycle
model (parkpilot.car), writes the resulting pose into qpos via CarBinding, calls
mj_forward, and renders the synchronized scene to a GIF.
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")

import imageio.v3 as iio
import mujoco
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parkpilot import ASSETS_DIR
from parkpilot.car import CarBinding, bicycle_step


def scripted_control(step: int) -> tuple[float, float]:
    """Return the open-loop speed and steering command for this frame."""
    if step < 100:
        return 1.0, 0.0
    if step < 250:
        return 1.0, 0.4
    if step < 400:
        return 1.0, -0.4
    if step < 500:
        return -0.8, -0.3
    return -0.8, 0.0


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "parking.xml"))
    data = mujoco.MjData(model)
    binding = CarBinding(model)

    # Bicycle model state; fine dt for smooth rendered motion.
    x = 0.0
    y = 0.0
    theta = 0.0
    dt = 0.01

    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for step in range(600):
            v, steer = scripted_control(step)
            x, y, theta, yaw_rate = bicycle_step(x, y, theta, v, steer, dt)
            binding.write(model, data, x, y, theta, v, steer, yaw_rate)

            if step % 10 == 0:
                renderer.update_scene(data, camera="overview")
                frames.append(renderer.render().copy())

    out = "drive_test.gif"
    iio.imwrite(out, np.asarray(frames), duration=0.1, loop=0)
    print(f"final pose -> x={x:.3f} m  y={y:.3f} m  theta={math.degrees(theta):.1f} deg")
    print(f"render OK  -> wrote {out} with {len(frames)} frames")


if __name__ == "__main__":
    main()
