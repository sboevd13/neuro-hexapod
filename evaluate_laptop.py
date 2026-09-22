"""Fixed-command acceptance tests; falls and wrong direction cannot pass on reward."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from locomotion_laptop import CONTROL_DT, Config, LocomotionEnv, load_model

COMMANDS = dict(forward=[1, 0, 0], backward=[-1, 0, 0],
                left=[0, 1, 0], right=[0, -1, 0],
                forward_left=[0.7071, 0.7071, 0], forward_right=[0.7071, -0.7071, 0],
                backward_left=[-0.7071, 0.7071, 0], backward_right=[-0.7071, -0.7071, 0],
                turn_left=[0, 0, 1], turn_right=[0, 0, -1], idle=[0, 0, 0])


def evaluate(policy=None, config=None, model=None, seconds=12.0, seeds=(2026,), commands=None):
    config = replace(config or Config(), episode_seconds=seconds)
    model = model if model is not None else load_model()
    env = LocomotionEnv(config, model)
    results = []
    for name, command in (commands or COMMANDS).items():
        for seed in seeds:
            obs = env.reset(command, seed=seed)
            start = env.data.qpos[:3].copy()
            reward_sum, angle, tilt, max_residual = 0., 0., 0., 0.
            min_height = float(start[2])
            fall = False
            for step in range(round(seconds/CONTROL_DT)):
                action = np.zeros(18) if policy is None else policy.act(obs[None])[0][0]
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                angle += info['yaw_delta']
                tilt = max(tilt, float(np.max(np.abs(info['rpy'][:2]))))
                min_height = min(min_height, float(info['position'][2]))
                max_residual = max(max_residual, float(np.max(np.abs(info['residual_deg']))))
                if terminated:
                    fall = True
                    break
            elapsed = (step+1)*CONTROL_DT
            delta = env.data.qpos[:2]-start[:2]
            distance = float(np.linalg.norm(delta))
            c = np.asarray(command)
            linear = float(np.linalg.norm(c[:2]))
            yaw_rate = angle/elapsed
            progress = lateral = direction_error = 0.
            if linear > 0:
                direction = c[:2]/linear
                progress = float(np.dot(delta, direction))
                lateral = float(abs(delta[0]*direction[1]-delta[1]*direction[0]))
                direction_error = float(np.rad2deg(np.arctan2(lateral, progress)))
                motion_ok = (progress/elapsed >= 0.045*linear and
                             direction_error <= 20.0 and abs(np.rad2deg(angle)) <= 25.0)
                score = progress/elapsed - 0.5*lateral/elapsed - 0.02*abs(angle)
            elif abs(c[2]) > 0:
                motion_ok = c[2]*yaw_rate >= 0.12 and distance <= 0.25
                score = c[2]*yaw_rate*0.4-distance*0.1
            else:
                motion_ok = distance <= 0.08 and abs(angle) <= np.deg2rad(10)
                score = 0.15-distance
            passed = bool(not fall and elapsed >= seconds-1e-6 and motion_ok
                          and tilt < np.deg2rad(35) and min_height >= 0.05)
            results.append(dict(command=name, seed=int(seed), passed=passed, fallen=fall,
                seconds=round(elapsed, 3), dx_m=float(delta[0]), dy_m=float(delta[1]),
                forward_progress_m=progress, lateral_drift_m=lateral,
                direction_error_deg=direction_error, yaw_deg=float(np.rad2deg(angle)),
                yaw_rate_rad_s=float(yaw_rate), max_tilt_deg=float(np.rad2deg(tilt)),
                min_height_m=min_height, max_residual_deg=max_residual,
                reward=float(reward_sum), score=float(score)))
    passed = all(row['passed'] for row in results)
    return dict(passed=passed, passed_count=sum(row['passed'] for row in results),
                total=len(results), score=float(np.mean([r['score'] for r in results])),
                results=results)


def print_report(report):
    for row in report['results']:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['command']:16s} "
              f"dx={row['dx_m']:+.3f}m dy={row['dy_m']:+.3f}m "
              f"yaw={row['yaw_deg']:+.1f}deg tilt={row['max_tilt_deg']:.1f}deg "
              f"fall={row['fallen']}", flush=True)
    print(f"Acceptance: {report['passed_count']}/{report['total']} | score={report['score']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--reference', action='store_true', help='Explicitly test the controller WITHOUT a neural policy')
    parser.add_argument('--seconds', type=float, default=12.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027])
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.seconds < 4:
        parser.error('Acceptance evaluation must cover at least 4 seconds')
    policy, config, model = None, Config(), load_model()
    if args.checkpoint:
        from laptop_policy import NumpyPolicy, load_checkpoint
        params, _, meta, xml = load_checkpoint(args.checkpoint)
        policy, config, model = NumpyPolicy(params), Config(**meta['config']), load_model(xml)
        print(f"Checkpoint: {args.checkpoint} | {meta['env_steps']} transitions | {meta['updates']} PPO updates")
    else:
        print('REFERENCE ONLY: no learned policy is being evaluated')
    report = evaluate(policy, config, model, args.seconds, tuple(args.seeds))
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
