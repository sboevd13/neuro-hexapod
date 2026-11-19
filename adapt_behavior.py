"""Adapt a neural reflex using recorded MuJoCo consequences of its own actions.

This does not automatically deploy the adapted model. Evaluate on separate
scenes before selecting it for a robot. This is one-step bandit feedback, not
multi-step reinforcement learning or sensorimotor self-supervision from video.
"""
import argparse
import json
from pathlib import Path

from autonomy.behavior import BehaviorModel
from autonomy.experience import fit_experience, observed_mse, read_experiences


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('models/behavior_v1.json'))
    p.add_argument('--experiences',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('runs/adapted_behavior.json'))
    p.add_argument('--epochs',type=int,default=25)
    args=p.parse_args()
    if args.epochs<1:
        p.error('--epochs must be positive')
    baseline=BehaviorModel.load(args.model)
    states,actions,rewards=read_experiences(args.experiences)
    adapted=fit_experience(baseline,states,actions,rewards,epochs=args.epochs)
    report={'type':'one_step_feedback_adaptation','samples':len(states),
            'training_observed_mse_before':round(observed_mse(baseline,states,actions,rewards),5),
            'training_observed_mse_after':round(observed_mse(adapted,states,actions,rewards),5),
            'note':'Training fit only. Must benchmark in held-out MuJoCo scenes.'}
    adapted.save(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
