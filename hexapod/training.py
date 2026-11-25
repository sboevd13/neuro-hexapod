"""Run PPO on the firmware-constrained XML and export the best qualified actor."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np

from .world import ROOT, robot_xml

WEIGHTS = ROOT/'hexapod/weights/gait_ppo.npz'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True,type=Path)
    parser.add_argument('--steps',type=int,default=98304)
    parser.add_argument('--output',type=Path,default=WEIGHTS)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    run.parent.mkdir(parents=True,exist_ok=True)
    xml=run.with_suffix('.xml')
    xml.write_text(ET.tostring(robot_xml(),encoding='unicode'))
    subprocess.run([sys.executable,str(ROOT/'train_laptop.py'),'--model-xml',str(xml),
                    '--steps',str(args.steps),'--num-envs','8','--workers','2','--rollout','128',
                    '--eval-every','16','--eval-seconds','12','--run-dir',str(run)],check=True)
    from laptop_policy import load_checkpoint
    params,_,meta,_=load_checkpoint(run/'best.npz')
    if meta['stage']!='trained_and_passed' or not meta['evaluation']['passed']:
        raise ValueError('Motion acceptance failed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**{k:np.asarray(params[k]) for k in ('w1','b1','w2','b2','wm','bm')})
    reports=ROOT/'validation/diploma'
    reports.mkdir(parents=True,exist_ok=True)
    stats=dict(training_steps=args.steps,selected_steps=meta['env_steps'],selected_updates=meta['updates'],
               model_sha256=meta['model_sha256'],checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
               training_config=meta['training_config'],mujoco=meta['mujoco_version'],jax=meta['jax_version'])
    (reports/'gait_training.json').write_text(json.dumps(stats,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'evaluate_laptop.py'),'--checkpoint',str(run/'best.npz'),
                    '--seconds','12','--seeds','1307','4523','8369','--json',str(reports/'gait_ppo.json')],check=True)

if __name__=='__main__':
    main()
