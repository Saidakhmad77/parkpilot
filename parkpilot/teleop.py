"""Phase 1 keyboard teleop for the MuJoCo parking scene.

Keyboard input changes speed and steering commands; the shared bicycle model
(parkpilot.car) advances the pose each frame, CarBinding writes it into qpos,
and mj_forward + viewer.sync update the live view.
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parkpilot import ASSETS_DIR
from parkpilot.car import CarBinding, bicycle_step


KEY_LEFT = 263
KEY_RIGHT = 262
KEY_DOWN = 264
KEY_UP = 265


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a control value to the allowed actuator range."""
    return float(np.clip(value, low, high))


def print_banner() -> None:
    print("ParkPilot teleop")
    print("  W / Up       increase drive speed (+0.2 m/s)")
    print("  S / Down     decrease drive speed (-0.2 m/s)")
    print("  A / Left     steer left (-0.1 rad)")
    print("  D / Right    steer right (+0.1 rad)")
    print("  Space        brake")
    print("  R            reset car to origin")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "parking.xml"))
    data = mujoco.MjData(model)
    binding = CarBinding(model)
    overview_camera_id = model.camera("overview").id

    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    steer = 0.0
    dt = 1.0 / 50.0

    def reset_car() -> None:
        nonlocal x, y, theta, v, steer
        x = 0.0
        y = 0.0
        theta = 0.0
        v = 0.0
        steer = 0.0

    def key_callback(keycode: int) -> None:
        nonlocal v, steer
        if keycode in (ord("W"), KEY_UP):
            v = clamp(v + 0.2, -2.0, 2.0)
        elif keycode in (ord("S"), KEY_DOWN):
            v = clamp(v - 0.2, -2.0, 2.0)
        elif keycode in (ord("A"), KEY_LEFT):
            steer = clamp(steer - 0.1, -0.6, 0.6)
        elif keycode in (ord("D"), KEY_RIGHT):
            steer = clamp(steer + 0.1, -0.6, 0.6)
        elif keycode == ord(" "):
            v = 0.0
        elif keycode == ord("R"):
            reset_car()

    print_banner()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Show the named overview camera and joint/actuator frames on startup.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = overview_camera_id
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

        next_frame = time.time()
        while viewer.is_running():
            x, y, theta, yaw_rate = bicycle_step(x, y, theta, v, steer, dt)
            binding.write(model, data, x, y, theta, v, steer, yaw_rate)
            viewer.sync()

            next_frame += dt
            sleep_time = next_frame - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_frame = time.time()


if __name__ == "__main__":
    main()
