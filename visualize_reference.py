import argparse
import time

import mujoco
import numpy as np
import pygame
from pygame.locals import QUIT

from reference_gait import CYCLE_TIME, raised_stand_targets, reference_targets


SERVO_SPEED_RAD_S = np.deg2rad(60.0) / 0.14
CONTROL_DT = 0.020


def parse_args():
    parser = argparse.ArgumentParser(description="Fast native MuJoCo reference-gait viewer")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def disable_expensive_rendering(renderer):
    # These effects are irrelevant for gait debugging and cost GPU time.
    for name in (
        "mjRND_SHADOW",
        "mjRND_REFLECTION",
        "mjRND_FOG",
        "mjRND_HAZE",
        "mjRND_SKYBOX",
    ):
        flag = getattr(mujoco.mjtRndFlag, name, None)
        if flag is not None:
            renderer.scene.flags[int(flag)] = 0


def main():
    args = parse_args()

    model = mujoco.MjModel.from_xml_path("models/arena.xml")
    data = mujoco.MjData(model)

    if model.nu != 18:
        raise RuntimeError(f"Expected 18 actuators, found {model.nu}")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    if body_id < 0:
        raise RuntimeError("Body 'body' not found in models/arena.xml")

    ctrl_low = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
    ctrl_high = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)

    stand = np.clip(raised_stand_targets().astype(np.float64), ctrl_low, ctrl_high)

    # Free-joint pose + 18 geometric joint angles.
    data.qpos[0:3] = np.array([0.0, 0.0, 0.110])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:25] = stand
    data.qvel[:] = 0.0
    data.ctrl[:] = stand
    mujoco.mj_forward(model, data)

    commanded_q = stand.copy()
    max_delta = SERVO_SPEED_RAD_S * CONTROL_DT
    physics_steps = max(1, int(round(CONTROL_DT / model.opt.timestep)))

    width = max(320, args.width)
    height = max(180, args.height)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height

    renderer = mujoco.Renderer(model, width=width, height=height)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = body_id
    camera.distance = 0.8
    camera.azimuth = 135
    camera.elevation = -25

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Hexapod reference gait — FAST VIEW")
    clock = pygame.time.Clock()

    start_time = time.perf_counter()
    frame = 0

    print("FAST VIEW: native MuJoCo physics, no MJX copy")
    print("Rendering: shadows/reflections/fog/haze/skybox disabled")
    print(f"Resolution: {width}x{height}, target FPS: {args.fps}")

    try:
        running = True
        while running:
            wall_start = time.perf_counter()

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    # Reset robot and gait phase.
                    data.qpos[0:3] = np.array([0.0, 0.0, 0.110])
                    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
                    data.qpos[7:25] = stand
                    data.qvel[:] = 0.0
                    data.ctrl[:] = stand
                    commanded_q[:] = stand
                    data.time = 0.0
                    start_time = time.perf_counter()
                    mujoco.mj_forward(model, data)

            phase = ((time.perf_counter() - start_time) % CYCLE_TIME) / CYCLE_TIME
            target_q = np.clip(reference_targets(phase), ctrl_low, ctrl_high)

            commanded_q += np.clip(target_q - commanded_q, -max_delta, max_delta)
            commanded_q = np.clip(commanded_q, ctrl_low, ctrl_high)
            data.ctrl[:] = commanded_q

            for _ in range(physics_steps):
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=camera)
            disable_expensive_rendering(renderer)
            pixels = renderer.render()

            screen.blit(
                pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1)),
                (0, 0),
            )
            pygame.display.flip()

            if frame % 50 == 0:
                pos = data.xpos[body_id]
                print(
                    f"phase={phase:.3f} x={pos[0]:+.3f} z={pos[2]:+.3f} "
                    f"fps={clock.get_fps():.1f}"
                )

            # Keep the simulator close to the real 50 Hz controller cadence.
            elapsed = time.perf_counter() - wall_start
            remaining = CONTROL_DT - elapsed
            if remaining > 0:
                time.sleep(remaining)

            clock.tick(args.fps)
            frame += 1
    finally:
        renderer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
