"""Run physically simulated two-stage pushing on RANDOMIZED unfamiliar objects.

A short contact probe supplies real sensor evidence; predict whether a further
push will move the object. Training/evaluation labels are from actual MuJoCo
body interactions. Independent simulator state is exported for auditing only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from autonomy.affordance_model import features, OutcomePredictor, evaluate
from autonomy.interaction import object_signature, target_surface, observed_motion, visible_points
from interact_laptop import Trial, interaction_scene, target_scan, touching_target
from locomotion_laptop import CONTROL_DT, MODEL_PATH, Config, LocomotionEnv, load_model
from dataclasses import replace


def randomized_trial(seed):
    """Geometry and hidden dynamics independently randomized, no width label shortcut."""
    rng=np.random.default_rng(seed)
    size_y=float(rng.uniform(.065,.22))
    size_x=float(rng.uniform(.055,.105))
    half_z=float(rng.uniform(.075,.12))
    mobile=bool(rng.random()<.70)
    mass=float(rng.uniform(.018,.30))
    friction=float(rng.uniform(.12,1.15))
    return Trial(f'object_seed_{seed}',mobile,size_x,size_y,half_z,mass,
                 x=float(rng.uniform(.52,.59)), y=0.0),friction


def observed_width(initial_scan):
    xy,yaw,d,h=initial_scan
    pts=visible_points(xy,yaw,d,h)
    if not len(pts): return None
    lateral=np.array([-math.sin(yaw),math.cos(yaw)])
    return float(np.ptp(pts@lateral)+2*float(np.min(d[h]))*math.tan(math.radians(5)))


def make_physical_scene(trial,friction):
    xml=interaction_scene(MODEL_PATH.read_text(),trial)
    root=ET.fromstring(xml)
    target=root.find('.//geom[@name="interaction_target_geom"]')
    if target is None: raise RuntimeError('Missing experimental target')
    target.set('friction',f'{friction:.5f} 0.005 0.0001')
    return ET.tostring(root,encoding='unicode')


def perform(seed, contact_short=.15, contact_long=.48, max_approach=7.0):
    """One full V5 physical encounter, returns sensor-only learning row.

    A held-out object is genuinely pushed in the scientific shadow evaluation
    to recover a label even when a deployed policy would skip the second push.
    This experiment does not claim that the high-level policy learned to walk.
    """
    trial, friction=randomized_trial(seed)
    model=load_model(make_physical_scene(trial,friction))
    budget=max_approach + contact_short + contact_long + 1.5
    env=LocomotionEnv(replace(Config(),episode_seconds=budget+1.),model)
    env.reset([0,0,0],seed=seed)
    gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'interaction_target_geom')
    before=target_scan(env,gid)
    apparent=observed_width(before)
    surface_initial=target_surface(*before)
    if apparent is None or surface_initial is None:
        return {'scene_seed':seed,'valid':False,'reason':'not_seen'}
    distance=float(np.linalg.norm(surface_initial-before[0]))
    robot_touched=False
    fell=False
    nmax=round(max_approach/CONTROL_DT)
    for t in range(nmax):
        env.set_command([.70,0,0])
        _,_,terminated,truncated,_=env.step(np.zeros(18))
        if touching_target(env,gid):
            robot_touched=True
            break
        if terminated or truncated:
            fell=bool(terminated)
            break
    if not robot_touched or fell:
        return {'scene_seed':seed,'valid':False,'reason':'fall' if fell else 'no_touch',
                'fell':fell,'touch':robot_touched}
    # Distinct physically timed contact actions; the object may move before
    # first detection, so short-probe delta is measured against initial scan.
    for _ in range(round(contact_short/CONTROL_DT)):
        env.set_command([.70,0,0]);_,_,terminated,_,_=env.step(np.zeros(18))
        if terminated: fell=True;break
    initial_probe=target_scan(env,gid)
    surf_probe=target_surface(*initial_probe)
    first_move=observed_motion(surface_initial,surf_probe)
    if first_move is None:
        return {'scene_seed':seed,'valid':False,'reason':'probe_lost','fell':fell}
    # Snapshot state for the forecast. Continue the same MuJoCo physics state.
    x=features(apparent,distance,first_move,robot_touched)
    if not fell:
        for _ in range(round(contact_long/CONTROL_DT)):
            env.set_command([.70,0,0]);_,_,terminated,_,_=env.step(np.zeros(18))
            if terminated: fell=True;break
    env.set_command([0,0,0])
    final=target_scan(env,gid)
    final_surface=target_surface(*final)
    total_move=observed_motion(surface_initial,final_surface)
    new_move=observed_motion(surf_probe,final_surface)
    if new_move is None:
        return {'scene_seed':seed,'valid':False,'reason':'final_lost','fell':fell}
    # Outcome of the *second* stage, not future observations sent as inputs.
    moved=bool(new_move>=.035)
    truth_initial=trial.x
    truth_final=float(env.data.geom_xpos[gid,0])
    return {'scene_seed':seed,'valid':True,'features':x.tolist(),
            'apparent_width_m':round(apparent,5),
            'range_m':round(distance,5),
            'short_probe_sensed_displacement_m':round(first_move,5),
            'long_push_additional_displacement_m':round(new_move,5),
            'total_sensed_displacement_m':round(float(total_move),5) if total_move is not None else None,
            'label_more_movement':int(moved),'touch':True,'fell':fell,
            'evaluation_only':{'mobile':trial.mobile,'hidden_mass_kg':round(trial.mass,5),
                               'hidden_friction':round(friction,5),'hidden_half_width_m':round(trial.half_y,5),
                               'ground_truth_displacement_x_m':round(truth_final-truth_initial,5)}}


def run_group(seeds, folder, predictor=None, update=False):
    rows=[]
    for seed in seeds:
        row=perform(seed);rows.append(row)
        if predictor is not None and row.get('valid'):
            row['predicted_movement_probability']=round(float(predictor.predict(row['features'])),5)
            row['would_push_again']=bool(row['predicted_movement_probability']>=.55)
        print(json.dumps({k:row.get(k) for k in ('scene_seed','valid','reason','label_more_movement','short_probe_sensed_displacement_m','predicted_movement_probability','would_push_again','fell')},allow_nan=False),flush=True)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'episodes.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    return rows


def validated(rows):
    good=[r for r in rows if r.get('valid') and not r.get('fell')]
    return np.asarray([r['features'] for r in good]),np.asarray([r['label_more_movement'] for r in good]),len(good)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-seeds',type=int,nargs='+',default=list(range(100,132)))
    parser.add_argument('--test-seeds',type=int,nargs='+',default=list(range(700,716)))
    parser.add_argument('--output',type=Path,default=Path('runs/random_affordance'))
    parser.add_argument('--epochs',type=int,default=120)
    args=parser.parse_args()
    if set(args.train_seeds)&set(args.test_seeds):parser.error('Train/test seeds must not overlap')
    if args.epochs<1:parser.error('epochs must be positive')
    train=run_group(args.train_seeds,args.output/'train')
    xx,yy,n=validated(train)
    if n<8 or len(set(yy))<2: raise RuntimeError('Too few valid diverse physics-training outcomes')
    naive=float(np.mean(yy));model=OutcomePredictor.fresh(71)
    model.fit(xx,yy,epochs=args.epochs)
    model.save(args.output/'model.json')
    test=run_group(args.test_seeds,args.output/'test',predictor=model)
    xt,yt,nt=validated(test)
    if nt<5:raise RuntimeError('Insufficient successful held-out physical trials')
    scores=evaluate(model,xt,yt,baseline=naive)
    report={'version':1,'physics':'MuJoCo; V5 reference contact; independent hidden mass/friction/size',
            'train_attempts':len(train),'test_attempts':len(test),
            'train_valid':n,'test_valid':nt,'train_positive_fraction':round(naive,5),
            'test_metrics':scores,
            'policy_choices_in_test':{'push_again':sum(r.get('would_push_again',False) for r in test),
                                      'observe_instead':sum(r.get('valid',False) and not r.get('would_push_again',False) for r in test)},
            'limitations':'Second push physically executed for label in ALL held-out shadow runs, regardless of policy suggestion. Model observes result of first probe, not mass/friction.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('FINAL_REPORT',json.dumps(report,allow_nan=False),flush=True)

if __name__=='__main__':main()
