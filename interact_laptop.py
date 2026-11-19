"""Simulation-only causal interaction experiment: touch a box, watch it move.

No simulator object labels or ground-truth displacements enter the decision
policy; the latter appear in metrics solely for independent verification.
Reference V5 walks the hexapod, and MuJoCo contacts move real freejoint bodies.
This is not object manipulation by a gripper, camera vision, or safe hardware code.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from dataclasses import dataclass, replace
import json
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from autonomy.explorer import Rectangle, range_scan
from autonomy.interaction import OutcomeMemory, object_signature, target_surface, observed_motion
from locomotion_laptop import CONTROL_DT, Config, LocomotionEnv, MODEL_PATH, load_model


@dataclass(frozen=True)
class Trial:
    name: str
    mobile: bool
    half_x: float
    half_y: float
    half_z: float
    mass: float
    x: float = 0.53
    y: float = 0.0


TRAINING = (
    Trial('train_light_narrow', True, .07, .07, .085, .025),
    Trial('train_fixed_wide', False, .09, .21, .11, 2.5),
)
HELD_OUT = (
    Trial('heldout_light_narrow', True, .074, .078, .09, .035, x=.56),
    Trial('heldout_fixed_wide', False, .095, .215, .11, 2.5, x=.56),
)


def interaction_scene(base_xml, trial):
    root=ET.fromstring(base_xml)
    world=root.find('worldbody')
    if world is None: raise ValueError('Missing worldbody')
    body=ET.SubElement(world,'body',{
        'name':'interaction_target',
        'pos':f'{trial.x:.4f} {trial.y:.4f} {trial.half_z:.4f}'})
    if trial.mobile:
        ET.SubElement(body,'freejoint',{'name':'interaction_target_free'})
    ET.SubElement(body,'geom',{
        'name':'interaction_target_geom','type':'box',
        'size':f'{trial.half_x:.5f} {trial.half_y:.5f} {trial.half_z:.5f}',
        'mass':f'{trial.mass:.5f}',
        'friction':'0.35 0.005 0.0001',
        'condim':'4',
        'rgba':'0.1 0.55 0.85 1' if trial.mobile else '0.75 0.25 0.25 1',
    })
    return ET.tostring(root,encoding='unicode')


def pose(env):
    q=env.data.qpos
    yaw=math.atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]**2+q[6]**2))
    return np.asarray(q[:2],float).copy(),float(yaw)


def target_scan(env, target_geom_id):
    """Simulate rays from dynamic MuJoCo geometry; return only depth measurements."""
    xy,yaw=pose(env)
    pos=env.data.geom_xpos[target_geom_id]
    dimensions=env.model.geom_size[target_geom_id]
    # Box yaw is ignored in the ray emulator; tests report this limitation.
    rectangle=Rectangle(float(pos[0]),float(pos[1]),float(dimensions[0]),float(dimensions[1]))
    depth,hit=range_scan(xy,yaw,[rectangle])
    return xy,yaw,depth,hit


def touching_target(env, geom_id):
    data=env.data
    for k in range(data.ncon):
        c=data.contact[k]
        if geom_id in (int(c.geom1),int(c.geom2)):
            other = int(c.geom2) if int(c.geom1) == geom_id else int(c.geom1)
            # The target rests on the ground at all times. Floor contact is
            # NOT a touch by the robot. Only geoms on robot bodies qualify.
            if int(env.model.geom_bodyid[other]) != 0 and int(env.model.geom_bodyid[other]) != int(env.model.geom_bodyid[geom_id]):
                return True
    return False


def run_trial(trial, memory, seed=7, seconds=8.0, contact_seconds=0.65, view=False, speed=1.0):
    model=load_model(interaction_scene(MODEL_PATH.read_text(),trial))
    env=LocomotionEnv(replace(Config(),episode_seconds=seconds+2),model)
    env.reset([0,0,0],seed=seed)
    geom_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'interaction_target_geom')
    if geom_id < 0: raise RuntimeError('Target geom not found')
    before_pos=env.data.geom_xpos[geom_id][:2].copy()
    initial=target_scan(env,geom_id)
    sign=object_signature(*initial)
    front=target_surface(*initial)
    choice=memory.choose(sign)
    if sign is None or front is None:
        raise RuntimeError('Target not seen by idealized initial range scan')
    viewer=None
    deadline=time.perf_counter()
    if view:
        from mujoco import viewer as mj_viewer
        viewer=mj_viewer.launch_passive(model,env.data)
        viewer.cam.distance=1.2
        viewer.cam.azimuth=135
        viewer.cam.elevation=-35
    def draw():
        nonlocal deadline
        if viewer is None:
            return
        if not viewer.is_running():
            raise KeyboardInterrupt('MuJoCo viewer closed')
        viewer.cam.lookat[:]=env.data.qpos[:3]
        viewer.sync()
        deadline+=CONTROL_DT/speed
        delay=deadline-time.perf_counter()
        if delay>0:
            time.sleep(delay)
    touches=0
    first_touch=None
    fallen=False
    initial_pose=initial[0].copy()
    frames=round(seconds/CONTROL_DT) if choice=='probe' else round(0.60/CONTROL_DT)
    for t in range(frames):
        env.set_command([0.70,0,0] if choice=='probe' else [0,0,0])
        _,_,terminated,_,_=env.step(np.zeros(18))
        draw()
        if touching_target(env,geom_id):
            touches+=1
            if first_touch is None: first_touch=t
        if terminated:
            fallen=True
            break
        if first_touch is not None and t-first_touch>=round(contact_seconds/CONTROL_DT):
            break
    env.set_command([0,0,0])
    # A short settling interval after the probe. The robot remains in physics.
    if not fallen and choice=='probe':
        for _ in range(round(0.4/CONTROL_DT)):
            _,_,terminated,_,_=env.step(np.zeros(18))
            draw()
            fallen=bool(terminated)
            if fallen: break
    if viewer is not None:
        viewer.close()
    final=target_scan(env,geom_id)
    surface=target_surface(*final)
    sensed=observed_motion(front,surface)
    after_pos=env.data.geom_xpos[geom_id][:2].copy()
    ground_truth=float(after_pos[0]-before_pos[0])
    outcome=memory.learn(sign,choice,sensed,first_touch is not None,fallen,trial.name)
    row={
        **outcome,
        'known_for_evaluation_only':{
            'mobile':bool(trial.mobile),
            'ground_truth_displacement_x_m':round(ground_truth,5),
            'ground_truth_displacement_total_m':round(float(np.linalg.norm(after_pos-before_pos)),5),
            'object_initial_xy':before_pos.tolist(),
            'object_final_xy':after_pos.tolist(),
        },
        'robot_distance_from_start_m':round(float(np.linalg.norm(pose(env)[0]-initial_pose)),4),
        'contact_frames':touches,
        'time_simulated_s':round(env.steps*CONTROL_DT,3),
        'sensor_initial_surface':front.tolist(),
        'sensor_final_surface':None if surface is None else surface.tolist(),
    }
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/interaction'))
    p.add_argument('--seconds',type=float,default=8)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--view',action='store_true',help='Display actual MuJoCo physics for each trial')
    p.add_argument('--speed',type=float,default=1.,help='Viewer playback speed')
    p.add_argument('--memory',type=Path,help='Load existing prior; otherwise begin new experiment')
    args=p.parse_args()
    if args.seconds<=0 or args.speed<=0: p.error('--seconds and speed must be positive')
    memory=OutcomeMemory.load(args.memory) if args.memory else OutcomeMemory()
    results=[]
    for k,trial in enumerate(TRAINING+HELD_OUT):
        if args.view:
            print('VIEWING',trial.name,'--',memory.choose('narrow' if trial.half_y < .15 else 'wide'),flush=True)
        row=run_trial(trial,memory,seed=args.seed+19*k,seconds=args.seconds,
                      view=args.view,speed=args.speed)
        results.append(row)
        print(json.dumps({key:row[key] for key in ('case','signature','action','sensed_motion_m','touch','fell','learned','posterior_pushability','known_for_evaluation_only')},allow_nan=False),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    memory.save(args.output/'affordance_memory.json')
    report={
        'version':1,'kind':'causal_contact_experiment','seed':args.seed,
        'world_observations':'idealized_range_rays_and_simulated_contact',
        'controller':'V5_reference_only','physics':'MuJoCo_freejoint_contact',
        'trials':results,
        'decisions_by_signature':{sign:memory.choose(sign) for sign in ('narrow','wide')},
        'note':'Object mobility from sensor ray endpoint changes; geom truth ONLY in evaluator.',
    }
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SAVED',args.output/'report.json',flush=True)

if __name__=='__main__': main()
