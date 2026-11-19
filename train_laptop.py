"""Laptop V3: native MuJoCo rollouts + conservative residual PPO in JAX.

`--steps` counts actual environment transitions, NOT optimizer iterations.
All directions are trained from the start. `best.npz` is written only after a
trained policy passes fixed forward/back/side/diagonal/turn/idle motion tests.
"""
import os
# Tiny CPU inference matrices benefit from one BLAS thread. Simulation uses its
# own optional worker pool; reserving all GPU memory is unnecessary here.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from evaluate_laptop import evaluate, print_report
from laptop_policy import (NumpyPolicy, generalized_advantage, init_params,
                           load_checkpoint, make_optimizer, make_update, save_checkpoint)
from locomotion_laptop import (Config, LocomotionEnv, MODEL_PATH, OBS_SIZE,
                               load_model, sample_command)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=100_000, help='Total transitions, cumulative when resuming')
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--workers', type=int, default=4, help='Native MuJoCo CPU worker threads')
    p.add_argument('--rollout', type=int, default=128)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=10, help='PPO updates between full acceptance evaluations')
    p.add_argument('--eval-seconds', type=float, default=12.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if min(args.steps, args.num_envs, args.workers, args.rollout, args.epochs,
           args.batch_size, args.eval_every) <= 0 or args.eval_seconds < 4:
        p.error('Counts must be positive and evaluation must cover at least 4 seconds')
    samples = args.num_envs*args.rollout
    if samples % args.batch_size:
        p.error('--num-envs * --rollout must be divisible by --batch-size')
    if args.resume:
        args.resume = args.resume.resolve()
        args.run_dir = args.run_dir.resolve() if args.run_dir else args.resume.parent
        if args.run_dir != args.resume.parent:
            p.error('Resume into the original run directory so its best checkpoint is retained')
    else:
        args.run_dir = args.run_dir or Path('checkpoints/laptop_v3')
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            p.error('Run directory already contains files. Use --resume RUN/latest.npz or a new --run-dir.')
    return args


def step_one(item):
    env, action = item
    return env.step(action)


def collect_rollout(envs, obs, policy, rng, length, executor=None):
    count = len(envs)
    data = {key: [] for key in ('obs', 'latent', 'log_prob', 'values', 'rewards',
                               'next_values', 'terminated', 'truncated')}
    falls = 0
    for _ in range(length):
        actions, latent, log_prob, values = policy.act(obs, rng)
        items = zip(envs, actions)
        results = list(executor.map(step_one, items)) if executor else [step_one(x) for x in items]
        next_obs = np.stack([r[0] for r in results])
        # Evaluate terminal observations BEFORE resetting. Timeouts bootstrap;
        # genuine falls do not. GAE traces also stop at either kind of reset.
        next_values = policy.values(next_obs)
        rewards = np.array([r[1] for r in results], np.float32)
        terminated = np.array([r[2] for r in results], np.float32)
        truncated = np.array([r[3] for r in results], np.float32)
        for key, value in dict(obs=obs, latent=latent, log_prob=log_prob, values=values,
                               rewards=rewards, next_values=next_values,
                               terminated=terminated, truncated=truncated).items():
            data[key].append(value.copy())
        falls += int(terminated.sum())
        for i in range(count):
            if terminated[i] or truncated[i]:
                next_obs[i] = envs[i].reset(sample_command(rng), seed=int(rng.integers(2**31)))
            elif envs[i].steps % 300 == 0:
                # Learn smooth mid-episode direction changes too.
                envs[i].set_command(sample_command(rng))
        obs = next_obs
    data = {key: np.stack(value) for key, value in data.items()}
    advantages, returns = generalized_advantage(data['rewards'], data['values'],
        data['next_values'], data['terminated'], data['truncated'])
    data['advantages'], data['returns'] = advantages, returns
    mean_reward = float(data['rewards'].mean())
    batch = {key: value.reshape((-1,)+value.shape[2:]).astype(np.float32)
             for key, value in data.items()
             if key in ('obs', 'latent', 'log_prob', 'values', 'advantages', 'returns')}
    return batch, obs, falls, mean_reward


def main():
    args = parse_args()
    if args.device != 'auto':
        jax.config.update('jax_platform_name', args.device)
    device = jax.devices()[0]
    print(f'Physics: native MuJoCo {mujoco.__version__}, implicitfast, 2 ms / control 50 Hz', flush=True)
    print(f'PPO: JAX {jax.__version__} on {device}; {args.num_envs} environments; '
          f'{args.workers} CPU workers', flush=True)
    rng = np.random.default_rng(args.seed)
    optimizer = make_optimizer()
    updates, env_steps, best_score = 0, 0, None
    config, xml = Config(), MODEL_PATH.read_text()
    if args.resume:
        params, opt_state, previous, xml = load_checkpoint(args.resume)
        config = Config(**previous['config'])
        updates, env_steps = previous['updates'], previous['env_steps']
        best_score = previous.get('best_score')
        rng.bit_generator.state = previous['rng_state']
        print(f'Resumed weights/optimizer/RNG at {env_steps} transitions. Physical episodes restart.', flush=True)
    else:
        params = init_params(args.seed)
        opt_state = optimizer.init(params)
    if env_steps >= args.steps:
        raise SystemExit('Checkpoint already reached --steps; choose a larger total to continue.')
    params, opt_state = jax.device_put((params, opt_state), device)
    model = load_model(xml)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    print('Preflight: testing the directional V5 reference, WITHOUT a learned policy...', flush=True)
    baseline = evaluate(config=config, model=model, seconds=args.eval_seconds)
    print_report(baseline)
    (args.run_dir/'reference_metrics.json').write_text(json.dumps(baseline, indent=2)+'\n')
    if not baseline['passed']:
        raise SystemExit('Reference failed motion checks. Training was not started. See reference_metrics.json.')

    def metadata(stage, report=None):
        return dict(config=config.to_dict(), updates=updates, env_steps=env_steps,
                    best_score=best_score, baseline_score=baseline['score'],
                    rng_state=rng.bit_generator.state, stage=stage,
                    training_config={name: getattr(args, name) for name in
                        ('num_envs', 'workers', 'rollout', 'epochs', 'batch_size',
                         'eval_every', 'eval_seconds', 'seed')},
                    evaluation=report, mujoco_version=mujoco.__version__, jax_version=jax.__version__)

    if not args.resume:
        save_checkpoint(args.run_dir/'initial_reference.npz', params, opt_state,
                        metadata('untrained_zero_residual'), xml)
    envs = [LocomotionEnv(config, model) for _ in range(args.num_envs)]
    obs = np.stack([env.reset(sample_command(rng), seed=int(rng.integers(2**31))) for env in envs])
    update = make_update(optimizer)
    executor = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print('\nFinishing the current rollout/update and saving latest.npz...', flush=True)
    old_handler = signal.signal(signal.SIGINT, request_stop)
    start = time.perf_counter()
    initial_steps = env_steps
    latest_report = None
    try:
        while env_steps < args.steps and not stop_requested:
            policy = NumpyPolicy(params)
            batch, obs, falls, reward_mean = collect_rollout(envs, obs, policy, rng, args.rollout, executor)
            sample_count = len(batch['obs'])
            device_batch = jax.device_put(batch, device)
            metrics = []
            for epoch in range(args.epochs):
                order = rng.permutation(sample_count)
                early_stop = False
                for offset in range(0, sample_count, args.batch_size):
                    idx = order[offset:offset+args.batch_size]
                    minibatch = {key: value[idx] for key, value in device_batch.items()}
                    candidate, candidate_opt, raw_metrics = update(params, opt_state, minibatch)
                    item = np.asarray(raw_metrics)
                    if not np.all(np.isfinite(item)):
                        raise FloatingPointError('Non-finite PPO update; the last valid checkpoint is retained.')
                    params, opt_state = candidate, candidate_opt
                    metrics.append(item)
                    if item[1] > 0.02:
                        early_stop = True
                        break
                if early_stop:
                    break
            updates += 1
            env_steps += sample_count
            elapsed = time.perf_counter()-start
            kl = float(np.mean(metrics, axis=0)[1])
            print(f'update={updates:5d} transitions={env_steps:8d}/{args.steps} '
                  f'samples/s={(env_steps-initial_steps)/max(elapsed,1e-6):.0f} '
                  f'reward/step={reward_mean:.4f} falls={falls} KL={kl:.5f}', flush=True)
            if updates % args.eval_every == 0 or env_steps >= args.steps:
                print('Evaluating the deterministic learned policy on all directions...', flush=True)
                latest_report = evaluate(NumpyPolicy(params), config, model, args.eval_seconds)
                latest_report['env_steps'] = env_steps
                print_report(latest_report)
                (args.run_dir/f'evaluation_{env_steps:09d}.json').write_text(
                    json.dumps(latest_report, indent=2, allow_nan=False)+'\n')
                if latest_report['passed'] and (best_score is None or latest_report['score'] > best_score):
                    best_score = latest_report['score']
                    save_checkpoint(args.run_dir/'best.npz', params, opt_state,
                                    metadata('trained_and_passed', latest_report), xml)
                    print(f'Saved passing learned policy: {args.run_dir / "best.npz"}', flush=True)
            save_checkpoint(args.run_dir/'latest.npz', params, opt_state,
                            metadata('trained_latest', latest_report), xml)
            with (args.run_dir/'progress.jsonl').open('a') as log:
                log.write(json.dumps(dict(updates=updates, env_steps=env_steps,
                    reward_mean=reward_mean, falls=falls, kl=kl, elapsed_s=elapsed))+'\n')
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if executor:
            executor.shutdown(wait=True)
    print(f'Finished at {env_steps} transitions. Latest: {args.run_dir / "latest.npz"}', flush=True)
    if (args.run_dir/'best.npz').exists():
        print(f'Use best.npz for the motion-tested learned policy: {args.run_dir / "best.npz"}', flush=True)
    else:
        print('No trained checkpoint passed every motion test yet. latest.npz is available for diagnosis.', flush=True)


if __name__ == '__main__':
    main()
