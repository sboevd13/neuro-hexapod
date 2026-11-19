"""Repeatable multi-seed headless MuJoCo navigation evaluation.

No training happens during evaluation. Each run uses a fresh output directory;
covered fraction is mapped cells (free or occupied), not area traversed.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(rows):
    if not rows:
        raise ValueError('No results')
    return {
        'n': len(rows),
        'mean_coverage': round(statistics.mean(x['coverage'] for x in rows), 5),
        'min_coverage': min(x['coverage'] for x in rows),
        'max_coverage': max(x['coverage'] for x in rows),
        'total_contacts': sum(x['contacts'] for x in rows),
        'episodes_with_contacts': sum(x['contacts'] > 0 for x in rows),
        'falls': sum(x['termination'] == 'fall' for x in rows),
        'mean_traveled_m': round(statistics.mean(x['traveled_m'] for x in rows), 4),
        'mean_goal_changes': round(statistics.mean(x['goal_changes'] for x in rows), 2),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 23, 31, 47, 59])
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--obstacles', type=int, default=5)
    p.add_argument('--output', type=Path, default=Path('runs/benchmark'))
    p.add_argument('--mode', choices=['baseline', 'learned'], default='baseline')
    p.add_argument('--behavior-model', type=Path)
    args = p.parse_args()
    if args.seconds <= 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('seconds must be positive and seeds unique')
    if args.mode == 'learned' and args.behavior_model is None:
        p.error('learned mode requires --behavior-model')
    rows = []
    for seed in args.seeds:
        dest = args.output / args.mode / f'seed_{seed}'
        command = [sys.executable, 'explore_laptop.py', '--reference',
                   '--seed', str(seed), '--obstacles', str(args.obstacles),
                   '--seconds', str(args.seconds), '--output', str(dest)]
        if args.mode == 'learned':
            command.extend(['--behavior-model', str(args.behavior_model), '--record-experience'])
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f'Failed seed {seed}: {run.stderr}\n{run.stdout[-3000:]}')
        row = json.loads((dest/'metrics.json').read_text())
        rows.append(row)
        print(f"seed={seed} coverage={row['coverage']:.1%} "
              f"contacts={row['contacts']} fallen={row['termination']=='fall'} "
              f"distance={row['traveled_m']:.2f} m", flush=True)
    report = {'mode': args.mode, 'seeds': args.seeds, 'seconds': args.seconds,
              'obstacles': args.obstacles, 'summary': summarize(rows), 'episodes': rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'{args.mode}_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
