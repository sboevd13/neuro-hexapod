"""View an actual V3 policy with exactly the training controller and physics."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from locomotion_laptop import CONTROL_DT, Config, LocomotionEnv, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--reference', action='store_true')
    parser.add_argument('--command', type=float, nargs=3, default=[1, 0, 0], metavar=('VX', 'VY', 'YAW'))
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error('--speed must be positive')
    policy, config, model = None, Config(), load_model()
    if args.checkpoint:
        from laptop_policy import NumpyPolicy, load_checkpoint
        params, _, meta, xml = load_checkpoint(args.checkpoint)
        policy, config, model = NumpyPolicy(params), Config(**meta['config']), load_model(xml)
        print(f'ACTUAL CHECKPOINT: {args.checkpoint} | {meta["env_steps"]} transitions | {meta["stage"]}')
    else:
        print('REFERENCE ONLY: no neural network. This is not a trained-policy result.')
    env = LocomotionEnv(config, model)
    obs = env.reset(args.command, seed=2026)
    pending = {'command': list(args.command), 'reset': False}
    keys = {87: [1, 0, 0], 83: [-1, 0, 0], 65: [0, 1, 0], 68: [0, -1, 0],
            81: [0, 0, 1], 69: [0, 0, -1], 32: [0, 0, 0]}
    def on_key(key):
        if key in keys:
            pending['command'] = keys[key]
        elif key == 82:
            pending['reset'] = True
    print('W/S forward/back; A/D left/right WITHOUT turning; Q/E turn; SPACE idle; R reset.')
    fallen = False
    with mujoco.viewer.launch_passive(model, env.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = env.data.qpos[:3]
        viewer.cam.distance = 0.95
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        deadline = time.perf_counter()
        while viewer.is_running():
            if pending['reset']:
                obs = env.reset(pending['command'], seed=2026)
                pending['reset'], fallen = False, False
                deadline = time.perf_counter()
            env.set_command(pending['command'])
            if not fallen:
                action = np.zeros(18) if policy is None else policy.act(obs[None])[0][0]
                obs, _, terminated, _, info = env.step(action)
                # Never hide a bad policy by silently resetting or substituting
                # a zero-residual/reference action. A fall stays visible.
                fallen = terminated
                if fallen:
                    print('FALL detected. Simulation paused; press R to reset.', flush=True)
                elif env.steps % 100 == 0:
                    print(f'command={env.command.round(2)} xy={info["position"][:2].round(3)} '
                          f'body_velocity={info["velocity"].round(3)} '
                          f'max_residual={abs(info["residual_deg"]).max():.2f}deg', flush=True)
            # Explicit centering avoids tracking-camera lag/cropping during
            # translation while retaining the user's orbit/zoom adjustments.
            viewer.cam.lookat[:] = env.data.qpos[:3]
            viewer.sync()
            deadline += CONTROL_DT/args.speed
            delay = deadline-time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.2:
                deadline = time.perf_counter()


if __name__ == '__main__':
    main()
