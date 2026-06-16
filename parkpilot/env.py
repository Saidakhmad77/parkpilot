"""ParkPilot Gymnasium environment — Phase 4.

Observation layout (float32, shape 10):
  [0]  x          car world-x position (m)
  [1]  y          car world-y position (m)
  [2]  sin θ      sine of car yaw
  [3]  cos θ      cosine of car yaw
  [4]  v          current drive speed (m/s)
  [5]  steer      current steering angle (rad)
  [6]  gdx        GOAL_X - x  (goal-relative dx)
  [7]  gdy        GOAL_Y - y  (goal-relative dy)
  [8]  sin(gdθ)   sine of heading error
  [9]  cos(gdθ)   cosine of heading error

Action space: Box(-1, 1, (2,), float32)
  a[0] -> drive velocity  = a[0] * 2.0  (clipped to [-2, 2])
  a[1] -> steer position  = a[1] * 0.6  (clipped to [-0.6, 0.6])

Reward shaping rationale:
  Dense negative rewards penalise distance to goal and heading misalignment every
  step, encouraging the agent to make progress efficiently. A per-step time penalty
  discourages dawdling. Large terminal bonuses/penalties sharply distinguish success
  from failure while keeping the dense signal dominant during learning.
"""

import os

os.environ.setdefault("MUJOCO_GL", "cgl")

import math

import gymnasium
import mujoco
import numpy as np
from gymnasium.spaces import Box

from parkpilot import ASSETS_DIR
from parkpilot.car import WHEELBASE, CarBinding, bicycle_step, wrap_angle

DT = 0.05
GOAL_X = 2.0
GOAL_Y = 0.0
GOAL_THETA = 0.0
SUCCESS_POS_TOL = 0.15
SUCCESS_HEAD_TOL = 0.17
MAX_EPISODE_STEPS = 400
OOB_X = 4.0
OOB_Y = 3.0

# ---------------------------------------------------------------------------
# Phase 4: Domain randomisation constants
# NOTE: The car is purely KINEMATIC (mj_forward only, NOT mj_step), so
# friction/mass/inertia parameters have zero effect on dynamics. We do NOT
# randomise those — they are silently ignored by MuJoCo in this setup.
# ---------------------------------------------------------------------------
START_X_RANGE = (-0.5, 2.6)
START_Y_RANGE = (-1.3, 1.3)
START_THETA_RANGE = (-math.pi, math.pi)
WHEELBASE_RANGE = (0.18, 0.28)
OBS_NOISE_STD = 0.01
START_MIN_GOAL_DIST = 0.5
START_MAX_TRIES = 100

# Reward weights
W_DIST = 1.0
W_HEADING = 0.3
TIME_PENALTY = 0.05
COLLISION_PENALTY = 100.0
SUCCESS_BONUS = 200.0
OOB_PENALTY = 100.0


class ParkingEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: str | None = None, randomize: bool = False) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "parking.xml"))
        self.data = mujoco.MjData(self.model)

        # Planar joints + drive/steer actuators (resolved by name in car.py).
        self._binding = CarBinding(self.model)

        # Collision geoms (car vs obstacle; floor excluded explicitly).
        self._car_chassis_id = int(self.model.geom("car_chassis").id)
        self._front_marker_id = int(self.model.geom("front_marker").id)
        self._left_parked_car_id = int(self.model.geom("left_parked_car").id)
        self._right_parked_car_id = int(self.model.geom("right_parked_car").id)
        self._floor_id = int(self.model.geom("floor").id)

        self._car_geom_ids = {self._car_chassis_id, self._front_marker_id}
        self._obstacle_geom_ids = {
            self._left_parked_car_id,
            self._right_parked_car_id,
        }

        self.action_space = Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = Box(
            np.full(10, -1e3, dtype=np.float32),
            np.full(10, 1e3, dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self._renderer = None

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._v = 0.0
        self._steer = 0.0
        self._step_count = 0
        self.randomize = randomize
        self._wheelbase = WHEELBASE

    def _bicycle_step(self, v: float, steer: float) -> float:
        self._x, self._y, self._theta, yaw_rate = bicycle_step(
            self._x, self._y, self._theta, v, steer, DT, self._wheelbase
        )
        return yaw_rate

    def _write_sim(self, yaw_rate: float) -> None:
        self._binding.write(
            self.model,
            self.data,
            self._x,
            self._y,
            self._theta,
            self._v,
            self._steer,
            yaw_rate,
        )

    def _get_obs(self) -> np.ndarray:
        heading_err = wrap_angle(GOAL_THETA - self._theta)
        obs = np.array(
            [
                self._x,
                self._y,
                math.sin(self._theta),
                math.cos(self._theta),
                self._v,
                self._steer,
                GOAL_X - self._x,
                GOAL_Y - self._y,
                math.sin(heading_err),
                math.cos(heading_err),
            ],
            dtype=np.float32,
        )
        # Phase 4: add observation noise during randomised training to improve robustness.
        # No noise when randomize=False -> byte-identical to previous behaviour.
        if self.randomize and OBS_NOISE_STD > 0.0:
            obs = obs + self.np_random.normal(0.0, OBS_NOISE_STD, size=10).astype(np.float32)
        return obs

    def _check_collision(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if geom1 == self._floor_id or geom2 == self._floor_id:
                continue

            car_obstacle = (
                geom1 in self._car_geom_ids and geom2 in self._obstacle_geom_ids
            ) or (geom2 in self._car_geom_ids and geom1 in self._obstacle_geom_ids)
            if car_obstacle:
                return True

        return False

    def _sample_start(self) -> None:
        """Rejection-sample a start pose that is collision-free, in-bounds, and far enough from the goal.

        Tries up to START_MAX_TRIES times, then falls back to the fixed safe start (0.3, 0, 0)
        so the loop cannot run forever.
        """
        for _ in range(START_MAX_TRIES):
            x = float(self.np_random.uniform(*START_X_RANGE))
            y = float(self.np_random.uniform(*START_Y_RANGE))
            theta = float(self.np_random.uniform(*START_THETA_RANGE))
            self._x, self._y, self._theta = x, y, theta
            self._write_sim(0.0)
            dist = math.hypot(GOAL_X - x, GOAL_Y - y)
            if (
                not self._check_collision()
                and abs(x) <= OOB_X
                and abs(y) <= OOB_Y
                and dist >= START_MIN_GOAL_DIST
            ):
                return  # accepted
        # Fallback: guaranteed-safe fixed start
        self._x, self._y, self._theta = 0.3, 0.0, 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        options = options or {}

        # --- Start pose ---
        if "start" in options:
            self._x, self._y, self._theta = options["start"]
        elif self.randomize:
            self._sample_start()
        else:
            self._x, self._y, self._theta = 0.3, 0.0, 0.0  # fixed default (backward-compat)

        # --- Wheelbase (kinematic model parameter) ---
        if "wheelbase" in options:
            self._wheelbase = float(options["wheelbase"])
        elif self.randomize:
            self._wheelbase = float(self.np_random.uniform(*WHEELBASE_RANGE))
        else:
            self._wheelbase = WHEELBASE  # fixed default (backward-compat)

        self._v = 0.0
        self._steer = 0.0
        self._step_count = 0

        self._write_sim(0.0)
        obs = self._get_obs()
        return obs, {"is_success": False, "terminal_reason": ""}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        drive_v = float(action[0]) * 2.0
        steer = float(action[1]) * 0.6

        self._v = drive_v
        self._steer = steer
        yaw_rate = self._bicycle_step(drive_v, steer)
        self._write_sim(yaw_rate)
        self._step_count += 1

        gdx = GOAL_X - self._x
        gdy = GOAL_Y - self._y
        dist_to_goal = math.hypot(gdx, gdy)
        heading_err = wrap_angle(GOAL_THETA - self._theta)
        success = (
            dist_to_goal <= SUCCESS_POS_TOL
            and abs(heading_err) <= SUCCESS_HEAD_TOL
        )
        collision = self._check_collision()
        oob = abs(self._x) > OOB_X or abs(self._y) > OOB_Y

        reward = (
            -(W_DIST * dist_to_goal)
            - (W_HEADING * abs(heading_err))
            - TIME_PENALTY
        )
        if success:
            reward += SUCCESS_BONUS
        if collision:
            reward -= COLLISION_PENALTY
        if oob:
            reward -= OOB_PENALTY

        terminated = success or collision or oob
        truncated = self._step_count >= MAX_EPISODE_STEPS

        if success:
            terminal_reason = "success"
        elif collision:
            terminal_reason = "collision"
        elif oob:
            terminal_reason = "oob"
        elif truncated:
            terminal_reason = "timeout"
        else:
            terminal_reason = ""

        obs = self._get_obs()
        info = {"is_success": success, "terminal_reason": terminal_reason}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render().copy()

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
