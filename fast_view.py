import argparse
import time

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from agent import ActorSimple_skip
from arena import BattleArena, _force_command, _load_actor, _load_norm, normalize_obs


WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360
RENDER_FPS = 25
CONTROL_STEPS_PER_FRAME = 2  # 2 * 20 ms at 25 FPS = real-time 50 Hz control


def parse_args():
    parser = argparse.ArgumentParser(description="Fast low-effects hexapod viewer")
    parser.add_argument("--agent", type=str, default=None, help="Actor .flax checkpoint")
    parser.add_argument("--norm", type=str, default=None, help="Observation normalization .npz")
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        default=None,
        help="Fixed command: +VX forward, +VY left, +YAW left turn",
    )
    return parser.parse_args()


def disable_expensive_rendering(env, renderer):
    # Disable expensive offscreen rendering quality first.
    try:
        env.model.vis.quality.shadowsize = 0
    except Exception:
        pass
    try:
        env.model.vis.quality.offsamples = 1
    except Exception:
        pass

    # Disable the main expensive MuJoCo render effects.
    for flag_name in (
        "mjRND_SHADOW",
        "mjRND_REFLECTION",
        "mjRND_SKYBOX",
        "mjRND_HAZE",
    ):
        try:
            flag = getattr(mujoco.mjtRndFlag, flag_name)
            renderer.scene.flags[flag] = 0
        except Exception:
            pass


def main():
    import pygame
    from pygame.locals import QUIT

    args = parse_args()
    env = BattleArena()

    actor = ActorSimple_skip(
        env.action_space_shape[0],
        env.ctrlrange_high,
        env.ctrlrange_low,
        512,
        512,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.key(42))

    dummy_obs = jp.expand_dims(state["obs"], 0)
    params = _load_actor(actor, dummy_obs, args.agent)
    obs_mean, obs_var = _load_norm(args.norm, env.observation_space_shape[0])

    if args.command is not None:
        fixed_command = np.clip(np.asarray(args.command, dtype=np.float32), -1.0, 1.0)
    elif params is None:
        # No trained actor yet: show the accepted forward V5 prior.
        fixed_command = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        fixed_command = None

    if params is None:
        print("FAST VIEW: V5 forward prior, no neural actor")
    else:
        print(f"FAST VIEW: actor={args.agent}")
        jit_action = jax.jit(actor.get_action)

    print(
        f"Window {WINDOW_WIDTH}x{WINDOW_HEIGHT}, render {RENDER_FPS} FPS, "
        f"control {RENDER_FPS * CONTROL_STEPS_PER_FRAME} Hz"
    )
    print("Shadows/reflections/skybox/haze disabled. SPACE=reset, ESC=exit")

    env.model.vis.global_.offwidth = WINDOW_WIDTH
    env.model.vis.global_.offheight = WINDOW_HEIGHT
    try:
        env.model.vis.quality.shadowsize = 0
        env.model.vis.quality.offsamples = 1
    except Exception:
        pass

    renderer = mujoco.Renderer(
        env.model,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
    disable_expensive_rendering(env, renderer)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.body_id
    camera.distance = 0.75
    camera.azimuth = 135
    camera.elevation = -25

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hexapod FAST VIEW")
    clock = pygame.time.Clock()

    inference_key = jax.random.key(7)
    frame = 0
    reward = jp.array([0.0])
    action = jp.zeros(18)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    state = jit_reset(jax.random.key(time.time_ns()))

            # Run two 20 ms controller updates for every rendered frame.
            # This keeps simulation control at 50 Hz while only rendering 25 FPS.
            for _ in range(CONTROL_STEPS_PER_FRAME):
                if fixed_command is not None:
                    state = _force_command(env, state, fixed_command)

                if params is None:
                    action = jp.zeros(18)
                else:
                    obs = jp.expand_dims(state["obs"], 0)
                    obs_norm = normalize_obs(obs, obs_mean, obs_var)
                    inference_key, action_key = jax.random.split(inference_key)
                    _, _, mean_action = jit_action(params, obs_norm, action_key)
                    action = jp.squeeze(mean_action)

                state, _, reward, _, _ = jit_step(state, action)

            # One GPU/device -> CPU transfer and one render per displayed frame.
            mjx.get_data_into(env.data, env.model, state["data"])
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=camera)
            pixels = renderer.render()
            surface = pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            if frame % RENDER_FPS == 0:
                command_np = np.asarray(jax.device_get(state["command"]))
                print(
                    f"step={int(state['step']):4d} "
                    f"cmd=({command_np[0]:+.2f},{command_np[1]:+.2f},{command_np[2]:+.0f}) "
                    f"x={float(state['last_com'][0]):+.3f} "
                    f"y={float(state['last_com'][1]):+.3f} "
                    f"reward={float(reward[0]):+.3f}"
                )

            clock.tick(RENDER_FPS)
            frame += 1
    finally:
        renderer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
