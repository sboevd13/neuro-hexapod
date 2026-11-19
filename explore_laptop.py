"""Explore an unknown MuJoCo room with occupancy memory and V5/PPO locomotion.

Simulation prototype only: idealized planar depth rays and MuJoCo truth pose;
no physical camera, SLAM/localization or continuous neural learning yet.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from autonomy.explorer import FrontierExplorer, OccupancyMemory, range_scan
from autonomy.scene import make_scene
from locomotion_laptop import CONTROL_DT, Config, LocomotionEnv, MODEL_PATH, load_model


def contact_with_obstacle(model, data):
    """Return the first scene obstacle contact position for recovery/metrics."""
    for i in range(data.ncon):
        c = data.contact[i]
        for gid in (int(c.geom1), int(c.geom2)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            if name.startswith('nav_obstacle_'):
                return np.asarray(c.pos[:2], dtype=float).copy()
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    locomotion = p.add_mutually_exclusive_group(required=True)
    locomotion.add_argument('--reference', action='store_true', help='Use V5, no neural residual')
    locomotion.add_argument('--checkpoint', type=Path, help='Validated V3 motion checkpoint')
    p.add_argument('--seconds', type=float, default=90.)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--obstacles', type=int, default=5)
    p.add_argument('--view', action='store_true', help='Open the MuJoCo viewer')
    p.add_argument('--speed', type=float, default=1.0, help='Viewer playback speed')
    p.add_argument('--output', type=Path, default=Path('runs/autonomy'))
    p.add_argument('--continue-map', action='store_true', help='Load the saved map for this seed')
    args = p.parse_args()
    if args.seconds <= 0 or args.speed <= 0 or not 0 <= args.obstacles <= 12:
        p.error('seconds/speed must be positive; obstacles must be in 0..12')
    return args


def main():
    args = parse_args()
    policy, config, base_xml = None, Config(), MODEL_PATH.read_text()
    if args.checkpoint:
        from laptop_policy import NumpyPolicy, load_checkpoint
        params, _, metadata, base_xml = load_checkpoint(args.checkpoint)
        policy = NumpyPolicy(params)
        config = Config(**metadata['config'])
        print(f'Motion policy: {metadata["stage"]}, {metadata["env_steps"]} transitions')
    else:
        print('Motion policy: V5 REFERENCE ONLY (no learned PPO)')
    scene = make_scene(base_xml, seed=args.seed, count=args.obstacles)
    model = load_model(scene.xml)
    env = LocomotionEnv(replace(config, episode_seconds=args.seconds + 1.), model)
    obs = env.reset([0, 0, 0], seed=args.seed)
    map_path = args.output / 'occupancy.json'
    if args.continue_map:
        memory = OccupancyMemory.load(map_path, scene_seed=args.seed)
    else:
        memory = OccupancyMemory()
    explorer = FrontierExplorer(memory)
    args.output.mkdir(parents=True, exist_ok=True)
    previous_xy = env.data.qpos[:2].copy()
    collisions = 0
    last_contact = False
    traveled = 0.0
    command = np.zeros(3)
    recovery_command = np.zeros(3)
    recovery_steps = 0
    decision = 'scan'
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, env.data)
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -55
        viewer.cam.azimuth = 135
    deadline = time.perf_counter()
    ended = 'time_limit'
    total_steps = math.ceil(args.seconds / CONTROL_DT)
    try:
        for t in range(total_steps):
            if viewer is not None and not viewer.is_running():
                ended = 'viewer_closed'
                break
            # Planner runs at about 3.3 Hz; locomotion controller stays at 50 Hz.
            if t % 15 == 0 and recovery_steps == 0:
                xy = env.data.qpos[:2].copy()  # SIMULATOR TRUTH (replace with SLAM)
                yaw = math.atan2(2 * (env.data.qpos[3] * env.data.qpos[6] +
                                      env.data.qpos[4] * env.data.qpos[5]),
                                 1 - 2 * (env.data.qpos[5]**2 + env.data.qpos[6]**2))
                ranges, hits = range_scan(xy, yaw, scene.obstacles)
                memory.observe(xy, yaw, ranges, hits)
                command, decision, _ = explorer.command(xy, yaw)
                if decision == 'complete':
                    ended = 'explored_known_space'
                    break
                if t % 150 == 0:
                    print(f's={t*CONTROL_DT:.1f} position={xy.round(2)} '
                          f'coverage={memory.coverage:.1%} action={decision} '
                          f'contacts={collisions}', flush=True)
            # A contact triggers a short escape maneuver, not endless spinning
            # inside an inflated obstacle cell. This is still simulation-only.
            if recovery_steps > 0:
                env.set_command(recovery_command)
                recovery_steps -= 1
            else:
                env.set_command(command)
            action = np.zeros(18) if policy is None else policy.act(obs[None])[0][0]
            obs, _, terminated, truncated, _ = env.step(action)
            now_xy = env.data.qpos[:2].copy()
            traveled += float(np.linalg.norm(now_xy - previous_xy))
            previous_xy = now_xy
            contact_xy = contact_with_obstacle(model, env.data)
            touching = contact_xy is not None
            if touching and not last_contact:
                collisions += 1
                yaw = math.atan2(2*(env.data.qpos[3]*env.data.qpos[6] +
                                    env.data.qpos[4]*env.data.qpos[5]),
                                 1-2*(env.data.qpos[5]**2+env.data.qpos[6]**2))
                memory.mark_contact(contact_xy)
                explorer.goal = None
                away = now_xy - contact_xy
                magnitude = np.linalg.norm(away)
                if magnitude > 1e-5:
                    away = 0.7 * away / magnitude
                    recovery_command = np.array([
                        math.cos(yaw)*away[0] + math.sin(yaw)*away[1],
                        -math.sin(yaw)*away[0] + math.cos(yaw)*away[1], 0.])
                    recovery_steps = 40
                    decision = 'recover'
            last_contact = touching
            if viewer is not None:
                viewer.cam.lookat[:] = env.data.qpos[:3]
                viewer.sync()
                deadline += CONTROL_DT / args.speed
                if deadline > time.perf_counter():
                    time.sleep(deadline - time.perf_counter())
                elif deadline < time.perf_counter() - 0.2:
                    deadline = time.perf_counter()
            if terminated or truncated:
                ended = 'fall' if terminated else 'time_limit'
                break
    finally:
        if viewer is not None:
            viewer.close()
        memory.save(map_path, scene_seed=args.seed)
        metrics = {
            'scene_seed': args.seed, 'obstacles': args.obstacles,
            'locomotion': 'checkpoint' if policy is not None else 'reference',
            'termination': ended, 'elapsed_s': round(env.steps*CONTROL_DT, 3),
            'contacts': collisions, 'traveled_m': round(traveled, 3),
            'coverage': round(memory.coverage, 5),
            'mapped_cells': int(np.count_nonzero(memory.cells != memory.UNKNOWN)),
            'goal_changes': explorer.goals_reached, 'sensor_surveys': memory.survey_count,
            'localization': 'simulator_ground_truth', 'sensor': 'idealized_planar_ranges',
        }
        (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2), flush=True)
        print(f'Map saved to {map_path}; metrics saved to {args.output / "metrics.json"}')


if __name__ == '__main__':
    main()
