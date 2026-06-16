"""Shared kinematic-bicycle car model for the ParkPilot scene.

Single source of truth for the planar bicycle kinematics and the MuJoCo
pose-writing used by drive_test.py, teleop.py, and env.py. Keeping the physics
in exactly one place means the car the agent trains on (env.py) is provably the
same car you drive by hand (teleop.py) and the same one the regression test
exercises (drive_test.py). `dt` stays a per-caller argument so each context can
pick its own integration step (fine for viz, real-time for teleop, coarse for RL).
"""

import math

import mujoco

# Front-to-rear axle distance (metres). The ONE place this constant is defined.
WHEELBASE = 0.22


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def bicycle_step(
    x: float,
    y: float,
    theta: float,
    v: float,
    steer: float,
    dt: float,
    wheelbase: float = WHEELBASE,
) -> tuple[float, float, float, float]:
    """Advance the kinematic bicycle model by one step.

    Standard discrete update:
        yaw_rate = v / L * tan(steer)
        x += v*cos(theta)*dt;  y += v*sin(theta)*dt;  theta += yaw_rate*dt

    Returns the new (x, y, theta, yaw_rate). `theta` is wrapped to [-pi, pi).
    """
    yaw_rate = v / wheelbase * math.tan(steer)
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta = wrap_angle(theta + yaw_rate * dt)
    return x, y, theta, yaw_rate


class CarBinding:
    """Resolves the car's planar joints/actuators in a MuJoCo model by NAME.

    Resolving by name (not by hardcoded index) keeps the code robust if the
    joint/actuator order in parking.xml ever changes.
    """

    def __init__(self, model: "mujoco.MjModel") -> None:
        self.x_qpos = int(model.joint("car_x").qposadr[0])
        self.y_qpos = int(model.joint("car_y").qposadr[0])
        self.theta_qpos = int(model.joint("car_z_rot").qposadr[0])

        self.x_qvel = int(model.joint("car_x").dofadr[0])
        self.y_qvel = int(model.joint("car_y").dofadr[0])
        self.theta_qvel = int(model.joint("car_z_rot").dofadr[0])

        self.drive_id = int(model.actuator("drive").id)
        self.steer_id = int(model.actuator("steer").id)

    def write(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        x: float,
        y: float,
        theta: float,
        v: float,
        steer: float,
        yaw_rate: float,
    ) -> None:
        """Write the kinematic pose into qpos/qvel/ctrl and run mj_forward.

        We use mj_forward (NOT mj_step): the motion is integrated in Python by
        bicycle_step, and MuJoCo is asked only to sync geometry + collision
        detection for the directly-written pose. mj_step would let the position
        steer-actuator fight the written qpos and corrupt the motion.
        """
        data.qpos[self.x_qpos] = x
        data.qpos[self.y_qpos] = y
        data.qpos[self.theta_qpos] = theta

        # Keep qvel/ctrl consistent with the kinematic state for inspection.
        data.qvel[self.x_qvel] = v * math.cos(theta)
        data.qvel[self.y_qvel] = v * math.sin(theta)
        data.qvel[self.theta_qvel] = yaw_rate

        data.ctrl[self.drive_id] = v
        data.ctrl[self.steer_id] = steer

        mujoco.mj_forward(model, data)
