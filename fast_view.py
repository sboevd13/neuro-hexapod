"""Ultra-light V5 viewer.

This file deliberately does NOT import JAX, MJX, arena.py, or the neural network.
It is only for visually checking the accepted V5 gait as fast as possible.
Physics runs in native MuJoCo on CPU; rendering is low-resolution and low-effects.
"""

import argparse
import time

import mujoco
import numpy as np

from reference_gait import CYCLE_TIME, raised_stand_targets, reference_targets


DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 270
DEFAULT_RENDER_FPS = 30
CONTROL_DT = 0.020  # 50 Hz, same as the real robot
SERVO_SPEED_RAD_S = np.deg2rad(60.0) / 0.14


def parse_args():
    parser = argparse.ArgumentParser(description="Native MuJoCo V5 fast viewer")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_RENDER_FPS)
    return parser.parse_args()


def disable_expensive_rendering(model, renderer):
    try:
        model.vis.quality.shadowsize = 0
    except Exception:
        pass
    try:
        model.vis.quality.offsamples = 1
    except Exception:
        pass

    for flag_name in (
        "mjRND_SHADOW",
        "mjRND_REFLECTION",
        "mjRND_SKYBOX",
        "mjRND_HAZE",
    ):
        try:
            renderer.scene.flags[getattr(mujoco.mjtRndFlag, flag_name)] = 0
        except Exception:
            pass


def reset_robot(model, data):
    mujoco.mj_resetData(model, data)

    data.qpos[0:3] = np.array([0.0, 0.0, 0.110])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    stand = np.asarray(raised_stand_targets(), dtype=np.float64)
    data.qpos[7:25] = stand
    data.ctrl[:] = np.clip(
        stand,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    mujoco.mj_forward(model, data)
    return data.ctrl.copy(), 0.0


def run_control_step(model, data, commanded_q, phase):
    phase = (phase + CONTROL_DT / CYCLE_TIME) % 1.0
    desired_q = np.asarray(reference_targets(phase), dtype=np.float64)
    desired_q = np.clip(
        desired_q,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )

    # Same approximate MG996R command-rate limit used by the RL environment.
    max_delta = SERVO_SPEED_RAD_S * CONTROL_DT
    commanded_q = commanded_q + np.clip(
        desired_q - commanded_q,
        -max_delta,
        max_delta,
    )
    commanded_q = np.clip(
        commanded_q,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    data.ctrl[:] = commanded_q

    physics_steps = max(1, int(round(CONTROL_DT / float(model.opt.timestep))))
    for _ in range(physics_steps):
        mujoco.mj_step(model, data)

    return commanded_q, phase


def main():
    import pygame
    from pygame.locals import QUIT

    args = parse_args()
    width = max(240, int(args.width))
    height = max(160, int(args.height))
    render_fps = max(5, int(args.fps))

    model = mujoco.MjModel.from_xml_path("models/arena.xml")

    # Fast CPU settings for this visual check. The physical model is unchanged.
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    model.opt.iterations = 4
    model.opt.ls_iterations = 4

    data = mujoco.MjData(model)
    commanded_q, phase = reset_robot(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    try:
        model.vis.quality.shadowsize = 0
        model.vis.quality.offsamples = 1
    except Exception:
        pass

    renderer = mujoco.Renderer(model, width=width, height=height)
    disable_expensive_rendering(model, renderer)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = body_id
    camera.distance = 0.72
    camera.azimuth = 135
    camera.elevation = -25

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Hexapod NATIVE FAST VIEW")
    clock = pygame.time.Clock()

    # Keep controller at exactly 50 Hz independent of chosen render FPS.
    control_accumulator = 0.0
    last_wall = time.perf_counter()
    frame = 0

    print("NATIVE FAST VIEW: no JAX, no MJX, no neural actor")
    print(f"Window {width}x{height}, render target {render_fps} FPS, control 50 Hz")
    print("Shadows/reflections/skybox/haze disabled. SPACE=reset, ESC=exit")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    commanded_q, phase = reset_robot(model, data)
                    control_accumulator = 0.0
                    last_wall = time.perf_counter()

            now = time.perf_counter()
            elapsed = min(now - last_wall, 0.10)
            last_wall = now
            control_accumulator += elapsed

            # Advance simulation in 20 ms controller chunks until caught up.
            while control_accumulator >= CONTROL_DT:
                commanded_q, phase = run_control_step(
                    model, data, commanded_q, phase
                )
                control_accumulator -= CONTROL_DT

            renderer.update_scene(data, camera=camera)
            # update_scene can refresh scene flags, so force them off every frame.
            disable_expensive_rendering(model, renderer)
            pixels = renderer.render()

            surface = pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            clock.tick(render_fps)
            frame += 1

            if frame % render_fps == 0:
                pos = data.xpos[body_id]
                print(
                    f"FPS={clock.get_fps():5.1f} "
                    f"phase={phase:.3f} "
                    f"x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}"
                )
    finally:
        renderer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
