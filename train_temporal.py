"""Offline Double-DQN stage on *sequential MuJoCo experience*, not synthetic labels.

Collect transitions first with explore_laptop.py --record-transitions; each
record has an observed next state. Save a separate versioned Q checkpoint.
"""
import argparse
import json
from pathlib import Path

from autonomy.behavior import BehaviorModel, init_weights
from autonomy.temporal import TemporalModel, read_replay, train_double_dqn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transitions', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--init-model', type=Path, help='Optional temporal or one-step warm start')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--gamma', type=float, default=0.96)
    p.add_argument('--seed', type=int, default=19)
    args = p.parse_args()
    if args.epochs <= 0 or not 0 <= args.gamma < 1:
        p.error('epochs must be positive and gamma in [0,1)')
    replay = read_replay(args.transitions)
    if args.init_model:
        try:
            initial = TemporalModel.load(args.init_model)
            mode = 'temporal_checkpoint'
        except ValueError:
            initial = TemporalModel(BehaviorModel.load(args.init_model).weights)
            mode = 'one_step_warm_start'
    else:
        initial = TemporalModel(init_weights(seed=args.seed))
        mode = 'random_init'
    trained, report = train_double_dqn(replay, initial, epochs=args.epochs,
                                        gamma=args.gamma, seed=args.seed)
    report['initialization'] = mode
    report['transition_files'] = [str(p) for p in args.transitions]
    trained.save(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    print('Checkpoint:', args.output, flush=True)


if __name__ == '__main__':
    main()
