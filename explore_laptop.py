"""Explore an unknown MuJoCo room with occupancy memory and V5/PPO locomotion.

Simulation prototype only: idealized planar depth rays and MuJoCo truth pose;
no camera/SLAM or physical hardware. An opt-in temporal-Q experiment can
learn from sequential MuJoCo transitions; V5/frontier navigation remains default.
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
from autonomy.behavior import ACTIONS, state_features
from autonomy.temporal import (TemporalModel, safe_mask, approve_override,
                               reward_from_motion, write_replay)
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


def pose_yaw(qpos):
    return math.atan2(2*(qpos[3]*qpos[6] + qpos[4]*qpos[5]),
                      1-2*(qpos[5]**2 + qpos[6]**2))


def discrete_action(command):
    """Convert the current global planner's actually executed local action."""
    cmd = np.asarray(command)
    if cmd[0] > 0.01 and abs(cmd[2]) < 0.01:
        return 0
    if cmd[2] > 0.01 and abs(cmd[0]) < 0.01:
        return 1
    if cmd[2] < -0.01 and abs(cmd[0]) < 0.01:
        return 2
    return None


def temporal_transition(event, xy, yaw, coverage, contacts, next_state=None,
                        next_mask=None, done=False, fallen=False):
    """Finalize a 0.3-s action with its *observed* successor, never a fabricated one."""
    terminal = bool(done or fallen or contacts > event['contacts'])
    successor = (np.zeros(12, dtype=np.float32) if terminal else
                 np.asarray(next_state, np.float32))
    if successor.shape != (12,) or not np.isfinite(successor).all():
        raise ValueError('Missing or invalid successor features')
    mask = ([False, True, True] if terminal else np.asarray(next_mask, bool).tolist())
    reward = reward_from_motion(event['xy'], xy, event['yaw'], yaw,
                                event['waypoint'], coverage-event['coverage'],
                                contacts-event['contacts'], fallen)
    return {'version':2, 'state':event['state'], 'action':event['action'],
            'reward':reward, 'next_state':successor.tolist(),
            'done':terminal, 'next_mask':mask,
            'scene_seed':event['scene_seed']}


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
    p.add_argument('--behavior-model', type=Path, help='Optional trained local action-value model')
    p.add_argument('--record-experience', action='store_true', help='Store rewards from executed neural decisions')
    p.add_argument('--temporal-model', type=Path,
                   help='Opt-in Double-DQN policy, kept separate from one-step behavior model')
    p.add_argument('--record-transitions', action='store_true',
                   help='Save sequential experience from planner or temporal controller')
    p.add_argument('--temporal-epsilon', type=float, default=0.,
                   help='Safe epsilon-greedy exploration; only when temporal-model is set')
    args = p.parse_args()
    if args.behavior_model and args.temporal_model:
        p.error('Select only one of --behavior-model and --temporal-model')
    if not 0 <= args.temporal_epsilon <= 1:
        p.error('--temporal-epsilon must be in [0,1]')
    if args.temporal_epsilon and not args.temporal_model:
        p.error('--temporal-epsilon requires --temporal-model')
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
    behavior = None
    if args.behavior_model:
        from autonomy.behavior import BehaviorModel
        behavior = BehaviorModel.load(args.behavior_model)
        print(f'Behavior model: {args.behavior_model} (frozen evaluation)')
    if args.record_experience and behavior is None:
        raise ValueError('--record-experience requires --behavior-model')
    from autonomy.experience import experience_reward, save_experiences
    temporal_model = TemporalModel.load(args.temporal_model) if args.temporal_model else None
    if temporal_model is not None:
        print(f'Temporal Q controller: {args.temporal_model}; epsilon={args.temporal_epsilon}')
    temporal_rng = np.random.default_rng(args.seed + 5729)
    temporal_rows = []
    pending_temporal = None
    temporal_decisions = 0
    temporal_overrides = 0
    temporal_last = None
    temporal_repeat = 0
    experience_rows = []
    pending_experience = None
    neural_decisions = 0
    neural_overrides = 0
    last_neural_action = None
    repeated_turns = 0
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
            # Settle the last neural action's reward using what the MuJoCo body
            # actually did, not the reward predicted during 2D pretraining.
            if pending_experience is not None and (t % 15 == 0 or recovery_steps > 0):
                current_xy = env.data.qpos[:2].copy()
                current_yaw = math.atan2(2*(env.data.qpos[3]*env.data.qpos[6] +
                                            env.data.qpos[4]*env.data.qpos[5]),
                                         1-2*(env.data.qpos[5]**2 + env.data.qpos[6]**2))
                event = pending_experience
                reward = experience_reward(event['start_xy'], current_xy,
                         event['start_yaw'], current_yaw, event['waypoint'],
                         collisions-event['contacts'])
                experience_rows.append(dict(features=event['features'],
                             action=event['action'],reward=reward,scene_seed=args.seed))
                pending_experience = None
            # Planner runs at about 3.3 Hz; locomotion controller stays at 50 Hz.
            if t % 15 == 0 and recovery_steps == 0:
                xy = env.data.qpos[:2].copy()  # SIMULATOR TRUTH (replace with SLAM)
                yaw = math.atan2(2 * (env.data.qpos[3] * env.data.qpos[6] +
                                      env.data.qpos[4] * env.data.qpos[5]),
                                 1 - 2 * (env.data.qpos[5]**2 + env.data.qpos[6]**2))
                ranges, hits = range_scan(xy, yaw, scene.obstacles)
                memory.observe(xy, yaw, ranges, hits)
                command, decision, waypoint = explorer.command(xy, yaw)
                # Close the preceding sequential action after obtaining this
                # scan/map update and the *next* route waypoint.
                if pending_temporal is not None:
                    successor = (state_features(ranges, yaw, xy, waypoint)
                                 if waypoint is not None else None)
                    temporal_rows.append(temporal_transition(
                        pending_temporal, xy, yaw, memory.coverage, collisions,
                        next_state=successor,
                        next_mask=safe_mask(ranges) if waypoint is not None else None,
                        done=waypoint is None))
                    pending_temporal = None
                if temporal_model is not None and waypoint is not None:
                    err = math.atan2(waypoint[1]-xy[1], waypoint[0]-xy[0]) - yaw
                    err = math.atan2(math.sin(err), math.cos(err))
                    if abs(err) < 0.60 and min(ranges[10:13]) < 0.95:
                        state = state_features(ranges, yaw, xy, waypoint)
                        chosen = temporal_model.choose(state, ranges,
                            epsilon=args.temporal_epsilon, rng=temporal_rng)
                        temporal_decisions += 1
                        if chosen != 0:
                            temporal_repeat = (temporal_repeat+1
                                if temporal_last == chosen else 1)
                        else:
                            temporal_repeat = 0
                        temporal_last = chosen
                        if (temporal_repeat <= 6 and approve_override(
                                discrete_action(command), chosen, ranges)):
                            cmd = np.array(ACTIONS[chosen], dtype=float)
                            if not np.allclose(cmd, command):
                                temporal_overrides += 1
                            command, decision = cmd, 'temporal_q'
                    else:
                        temporal_repeat = 0
                        temporal_last = None
                if behavior is not None and waypoint is not None:
                    # Use the trained local reflex only near uncertain obstacles.
                    # Keep global frontier planning and a hard minimum clearance.
                    if min(ranges[10:13]) < 0.68:
                        candidate = behavior.choose(
                            state_features(ranges, yaw, xy, waypoint), ranges)
                        neural_decisions += 1
                        # Prevent repeated neural turning from immobilizing navigation.
                        if candidate != 0:
                            repeated_turns = repeated_turns+1 if candidate == last_neural_action else 1
                        else:
                            repeated_turns = 0
                        last_neural_action = candidate
                        # Keep planner rotations for large heading errors.
                        error = math.atan2(waypoint[1]-xy[1], waypoint[0]-xy[0]) - yaw
                        error = math.atan2(math.sin(error), math.cos(error))
                        if (abs(error) < 0.60 and repeated_turns <= 6):
                            learned_command = np.array(ACTIONS[candidate], dtype=float)
                            if not np.allclose(learned_command, command):
                                neural_overrides += 1
                            command, decision = learned_command, 'neural'
                            if args.record_experience:
                                pending_experience = dict(features=state_features(ranges, yaw, xy, waypoint).tolist(),
                                    action=int(candidate),start_xy=xy.copy(),start_yaw=yaw,
                                    waypoint=np.asarray(waypoint).copy(),contacts=collisions)
                    else:
                        repeated_turns = 0
                        last_neural_action = None
                if args.record_transitions and waypoint is not None:
                    action_id = discrete_action(command)
                    if action_id is not None:
                        pending_temporal = dict(
                            state=state_features(ranges, yaw, xy, waypoint).tolist(),
                            action=int(action_id), xy=xy.copy(), yaw=float(yaw),
                            waypoint=np.asarray(waypoint).copy(),
                            coverage=float(memory.coverage), contacts=collisions,
                            scene_seed=args.seed)
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
            if touching and pending_temporal is not None:
                # A collision terminates the macro-action: do not bootstrap
                # through a recovery maneuver controlled by a separate shield.
                temporal_rows.append(temporal_transition(
                    pending_temporal, now_xy, pose_yaw(env.data.qpos),
                    memory.coverage, collisions, done=True))
                pending_temporal = None
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
        if pending_temporal is not None:
            temporal_rows.append(temporal_transition(
                pending_temporal, env.data.qpos[:2].copy(),
                pose_yaw(env.data.qpos), memory.coverage, collisions,
                done=True, fallen=ended == 'fall'))
        if args.record_transitions:
            write_replay(args.output/'transitions.jsonl', temporal_rows)
        if pending_experience is not None:
            event = pending_experience
            xy = env.data.qpos[:2].copy()
            yaw = math.atan2(2*(env.data.qpos[3]*env.data.qpos[6] + env.data.qpos[4]*env.data.qpos[5]),
                             1-2*(env.data.qpos[5]**2 + env.data.qpos[6]**2))
            experience_rows.append(dict(features=event['features'],action=event['action'],
                reward=experience_reward(event['start_xy'],xy,event['start_yaw'],yaw,
                                         event['waypoint'],collisions-event['contacts']),
                scene_seed=args.seed))
        if args.record_experience:
            save_experiences(args.output/'experience.jsonl',experience_rows)
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
            'neural_decisions': neural_decisions, 'neural_overrides': neural_overrides,
            'recorded_experiences': len(experience_rows),
            'recorded_transitions': len(temporal_rows),
            'temporal_decisions': temporal_decisions,
            'temporal_overrides': temporal_overrides,
            'behavior': ('temporal_double_dqn' if temporal_model is not None else
                         'trained_local_value' if behavior is not None else 'frontier_baseline'),
            'localization': 'simulator_ground_truth', 'sensor': 'idealized_planar_ranges',
        }
        (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2), flush=True)
        print(f'Map saved to {map_path}; metrics saved to {args.output / "metrics.json"}')


if __name__ == '__main__':
    main()
