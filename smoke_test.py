"""Phase 0 smoke test: prove MuJoCo physics + offscreen rendering work on this M5 Mac.

Builds a trivial scene (a box dropped onto a ground plane), steps the physics
for a bit, and renders a single frame to PNG. If this runs and produces a
non-trivial image, the whole toolchain (physics + GL render) is good to go.
"""

import mujoco
import numpy as np
import imageio.v3 as iio

# Minimal inline scene: a ground plane + a free-floating box that falls under gravity.
XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.4 0.5 0.4 1"/>
    <body name="box" pos="0 0 1.0">
      <freejoint/>
      <geom name="box" type="box" size="0.15 0.15 0.15" rgba="0.85 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Step ~1 second of simulated time so the box visibly drops and settles.
    n_steps = int(1.0 / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    box_z = float(data.body("box").xpos[2])
    print(f"physics OK  -> stepped {n_steps} steps, box height settled at z={box_z:.3f} m")

    # Offscreen render of the final frame.
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()

    out = "phase0_frame.png"
    iio.imwrite(out, frame)
    print(f"render OK   -> wrote {out}  shape={frame.shape}  dtype={frame.dtype}  "
          f"mean_pixel={frame.mean():.1f}")


if __name__ == "__main__":
    main()
