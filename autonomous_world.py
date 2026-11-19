"""Experimental single-MuJoCo-session embodied curiosity + bounded interactions.

During one continuous physics rollout, the hexapod maps a room, clusters depth
returns into object hypotheses, chooses inspection goals, approaches, probes,
uses a learned push-outcome predictor and records physical consequences. Hidden
mass, friction and freejoint flags are NOT exposed to the agent. No hardware.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from autonomy.world_scene import make_world
from autonomy.explorer import Rectangle, range_scan, FrontierExplorer, OccupancyMemory
from autonomy.curiosity import CuriosityController, ObjectMemory
from autonomy.affordance_model import features, OutcomePredictor
from autonomy.active_experiments import (OutcomeEnsemble, experiment_features,
                                         choose_action, dataset)
from autonomy.interaction import visible_points
from locomotion_laptop import Config, LocomotionEnv, MODEL_PATH, CONTROL_DT


def body_pose(env):
    q=env.data.qpos
    yaw=math.atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]**2+q[6]**2))
    return np.array(q[:2],dtype=float),float(yaw)


def sensor_view(env, walls, geom_ids):
    """World geometry is used solely to emulate range sensors, not perception."""
    xy,yaw=body_pose(env)
    boxes=list(walls)
    for gid in geom_ids:
        xyz=env.data.geom_xpos[gid]
        half=env.model.geom_size[gid]
        boxes.append(Rectangle(float(xyz[0]),float(xyz[1]),float(half[0]),float(half[1])))
    return (xy,yaw,*range_scan(xy,yaw,boxes))


def observed_target(xy,yaw,distances,hits,reference):
    """Associate laser endpoints with the most recent tracked surface.

    Association may fail if a target is heavily occluded or pushed too far;
    this is reported as missing evidence, never replaced with geom ground truth.
    """
    pts=ObjectMemory.detections(xy,yaw,distances,hits)
    if not pts: return None
    target=np.asarray(reference,dtype=float)
    best=min(pts,key=lambda x:np.linalg.norm(x-target))
    return np.asarray(best,dtype=float) if np.linalg.norm(best-target)<.40 else None


def object_contact(env, target_ids):
    """Only robot/object contacts, excluding target–floor and target–target."""
    targets=set(target_ids)
    for i in range(env.data.ncon):
        c=env.data.contact[i]
        a,b=int(c.geom1),int(c.geom2)
        if a not in targets and b not in targets:continue
        other=b if a in targets else a
        body=int(env.model.geom_bodyid[other])
        if body==0:continue
        bodyname=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,body) or ''
        if bodyname.startswith('interaction_target_'):continue
        return True
    return False


def directed_motion(xy,yaw,target,limit=.70):
    delta=np.asarray(target,dtype=float)-np.asarray(xy,dtype=float)
    heading=math.atan2(delta[1],delta[0]);err=math.atan2(math.sin(heading-yaw),math.cos(heading-yaw))
    if abs(err)>.20:
        return np.array([0.,0.,math.copysign(.78,err)])
    return np.array([limit,0.,0.])


def load_experience(path):
    if path is None:return []
    payload=json.loads(Path(path).read_text())
    if not isinstance(payload,list):raise ValueError('Expected list of prior probe outcomes')
    result=[]
    for row in payload:
        if not isinstance(row,dict):
            raise ValueError('Invalid previous probe record')
        if row.get('valid') is False:
            continue
        label=row.get('label',row.get('label_meaningful_total_movement'))
        if (len(row.get('features',[]))!=5 or label not in (0,1) or
                not np.isfinite(row['features']).all()):
            raise ValueError('Invalid prior probe record')
        # Strip audit-only MuJoCo metadata. The policy replay contains ONLY
        # range/contact features and measured physical outcome labels.
        result.append({'features':list(map(float,row['features'])),'label':int(label)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--model',type=Path,help='Optional trained prediction checkpoint')
    p.add_argument('--experience',type=Path,help='Load previous sensor-observed action outcomes')
    p.add_argument('--experiment-model',type=Path,help='Optional ensemble of learned physical displacement forecasts')
    p.add_argument('--experiment-experience',type=Path,help='Sensor-only labeled displacement experiences for online updates')
    p.add_argument('--exploration-weight',type=float,default=1.0,help='Epistemic curiosity weight for bounded experiments')
    p.add_argument('--output',type=Path,default=Path('runs/autonomous_world'))
    p.add_argument('--view',action='store_true')
    args=p.parse_args()
    if args.seconds<=0 or not 0 <= args.exploration_weight <= 5: p.error('Invalid seconds or exploration weight')
    xml,props,walls=make_world(MODEL_PATH.read_text(),seed=args.seed)
    model=mujoco.MjModel.from_xml_string(xml)
    mobile=sum(x.mobile for x in props)
    if (model.nu!=18 or model.nq!=25+7*mobile or
            model.opt.integrator!=mujoco.mjtIntegrator.mjINT_IMPLICITFAST or
            not np.isclose(model.opt.timestep,.002)):
        raise RuntimeError('Incompatible V5 model/experimental props')
    env=LocomotionEnv(replace(Config(),episode_seconds=args.seconds+2),model)
    env.reset([0,0,0],seed=args.seed)
    ids=[]
    for item in props:
        i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,item.label+'_geom')
        if i<0:raise RuntimeError('Experimental prop missing')
        ids.append(i)
    original={str(i):np.asarray(env.data.geom_xpos[i][:2]).copy() for i in ids}
    spatial=OccupancyMemory()
    frontier=FrontierExplorer(spatial)
    objects=ObjectMemory(args.seed)
    curiosity=CuriosityController(frontier,objects,minimum_standoff=.52)
    network=OutcomePredictor.load(args.model) if args.model else OutcomePredictor.fresh(71)
    prior=load_experience(args.experience)
    entries=list(prior)
    active_network = OutcomeEnsemble.load(args.experiment_model) if args.experiment_model else None
    active_entries = []
    if args.experiment_experience:
        active_entries=json.loads(args.experiment_experience.read_text())
        # This replay may contain NO privileged simulator metadata.
        _,_=dataset(active_entries)
    if active_network is None and active_entries:
        x,y=dataset(active_entries)
        active_network=OutcomeEnsemble.fit(x,y,seed=args.seed+100,members=4,steps=65)
    initial_active_samples=len(active_entries)
    active_updates=0
    active_choices=[]
    active_pushes=0
    second_duration=.48
    active_choice=None
    attempts=[]
    phase='explore'
    phase_ticks=0
    target=None
    target_id=None
    target_start=None
    initial_yaw=0.
    start_range=.0
    first_movement=None
    short_features=None
    last_seen=None
    current_command=np.zeros(3)
    viewer=None
    deadline=time.perf_counter()
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer=mj_viewer.launch_passive(model,env.data)
        viewer.cam.distance=2.6
        viewer.cam.elevation=-50
    scans=0
    touches=0
    online_updates=0
    ignored=[]
    term='time_limit'
    def finish_probe(note,final_point=None):
        nonlocal phase,phase_ticks,target,target_start,short_features,first_movement,network,touches,online_updates,active_network,active_updates,active_choice
        if target is not None:
            current=final_point
            distance=(None if current is None or target_start is None else
                      float(np.dot(current-target_start,
                          np.array([math.cos(initial_yaw),math.sin(initial_yaw)]))))
            label=(None if distance is None else int(distance>=.05))
            if short_features is not None and label is not None and note=='long_probe_complete':
                entries.append({'features':short_features.tolist(),'label':label})
                if active_network is not None:
                    active_entries.append({'features':experiment_features(short_features,second_duration).tolist(),
                                          'displacement_m':float(np.clip(distance,-.06,.30)),
                                          'action_seconds':float(second_duration)})
                    if len(active_entries)>=8:
                        xx,yy=dataset(active_entries)
                        active_network=OutcomeEnsemble.fit(xx,yy,seed=args.seed+len(active_entries),
                                                           members=len(active_network.members),steps=45)
                        active_updates+=1
                if len(entries)>=8 and len({x['label'] for x in entries})>=2:
                    x=np.array([r['features'] for r in entries])
                    y=np.array([r['label'] for r in entries])
                    network.fit(x,y,epochs=25,lr=.01,seed=args.seed+len(entries))
                    online_updates+=1
            attempts.append({'note':note,'target_surface_at_start':target_start.tolist() if target_start is not None else None,
                             'total_sensed_movement_m':None if distance is None else round(distance,5),
                             'short_probe_movement_m':first_movement,
                             'predicted_probability':last_seen,
                             'active_experiment':active_choice,
                             'label_meaningful_movement':label,
                             'physics_time_s':round(env.steps*CONTROL_DT,3),
                             'robot_xy_at_end':np.asarray(env.data.qpos[:2]).round(4).tolist(),
                             'robot_to_target_distance_m':round(float(np.linalg.norm(
                                 np.asarray(env.data.qpos[:2]) - target)),4)})
            ignored.append(np.asarray(target_start if target_start is not None else target).copy())
        phase='retreat';phase_ticks=0
        target=None;target_start=None;short_features=None;first_movement=None;active_choice=None

    try:
        for tick in range(math.ceil(args.seconds/CONTROL_DT)):
            xy,yaw,distances,hits=sensor_view(env,walls,ids)
            if tick%15==0:
                scans+=1
                spatial.observe(xy,yaw,distances,hits)
                default,kind,waypoint=frontier.command(xy,yaw)
                if phase=='explore':
                    cmd,decision,object_point=curiosity.command(xy,yaw,distances,hits,(default,kind,waypoint))
                    current_command=cmd
                    if decision=='inspect_object' and object_point is not None:
                        point=np.asarray(object_point,dtype=float)
                        if all(np.linalg.norm(point-old)>.37 for old in ignored):
                            measured=observed_target(xy,yaw,distances,hits,point)
                            if measured is not None:
                                phase='approach';phase_ticks=0
                                target=point.copy();target_start=measured.copy()
                                initial_yaw=math.atan2(point[1]-xy[1],point[0]-xy[0])
                                start_range=float(np.linalg.norm(measured-xy))
                                current_command=directed_motion(xy,yaw,point)
                elif phase=='approach' and target is not None:
                    current_command=directed_motion(xy,yaw,target,limit=.68)
                elif phase in ('short_probe','long_probe'):
                    current_command=np.array([.65,0.,0.])
                elif phase=='retreat':
                    current_command=np.array([-.70,0.,0.])
            if phase=='approach':
                phase_ticks+=1
                if object_contact(env,ids):
                    touches+=1
                    phase='short_probe';phase_ticks=0
                elif phase_ticks>round(12.0/CONTROL_DT):
                    finish_probe('object_not_reached')
            elif phase=='short_probe':
                phase_ticks+=1
                if phase_ticks>=round(.18/CONTROL_DT):
                    now=observed_target(xy,yaw,distances,hits,target)
                    if now is None:
                        finish_probe('sensor_occluded_after_probe')
                    else:
                        first_movement=float(np.dot(now-target_start,np.array([math.cos(initial_yaw),math.sin(initial_yaw)])))
                        apparent=.20  # coarse cluster width; never the hidden XML dimension
                        short_features=features(apparent,start_range,first_movement,True)
                        if active_network is not None:
                            active_choice=choose_action(active_network,short_features,
                                safety_ok=bool(env.data.qpos[2]>.075 and active_pushes<2),
                                novelty_weight=args.exploration_weight)
                            active_choices.append(dict(active_choice))
                            last_seen=None
                            should_push=active_choice['action']!='observe'
                            if should_push: active_pushes += 1
                            second_duration=float(active_choice['duration'])
                        else:
                            last_seen=float(network.predict(short_features))
                            should_push=last_seen>=.50 or len(entries)<6
                            second_duration=.48
                        if should_push:
                            phase='long_probe';phase_ticks=0
                        else:
                            finish_probe('model_chose_observe',now)
            elif phase=='long_probe':
                phase_ticks+=1
                if phase_ticks>=round(second_duration/CONTROL_DT):
                    now=observed_target(xy,yaw,distances,hits,target)
                    finish_probe('long_probe_complete',now)
            elif phase=='retreat':
                phase_ticks+=1
                if phase_ticks>=round(3.5/CONTROL_DT):
                    phase='explore';phase_ticks=0
                    curiosity.abandon()
            env.set_command(current_command)
            _,_,fallen,truncated,_=env.step(np.zeros(18))
            if viewer is not None:
                if not viewer.is_running():
                    term='viewer_closed';break
                viewer.cam.lookat[:]=env.data.qpos[:3]
                viewer.sync()
                deadline+=CONTROL_DT
                if deadline>time.perf_counter(): time.sleep(deadline-time.perf_counter())
            if fallen or truncated:
                term='fall' if fallen else 'time_limit'
                break
        else:
            term='time_limit'
    finally:
        if viewer is not None: viewer.close()
        args.output.mkdir(parents=True,exist_ok=True)
        objects.save(args.output/'objects.json')
        spatial.save(args.output/'map.json',scene_seed=args.seed)
        network.save(args.output/'predictor.json')
        if active_network is not None:
            active_network.save(args.output/'active_predictor.json')
            (args.output/'active_experiences.json').write_text(json.dumps(active_entries,indent=2,allow_nan=False)+'\n')
        (args.output/'experiences.json').write_text(json.dumps(entries,indent=2,allow_nan=False)+'\n')
        audit=[]
        for item,gid in zip(props,ids):
            now=np.asarray(env.data.geom_xpos[gid][:2])
            audit.append({'id':item.label,'hidden_mobile':item.mobile,
                          'hidden_mass_kg':item.mass,'hidden_friction':item.friction,
                          'truth_displacement_m':float(np.linalg.norm(now-original[str(gid)]))})
        report={'seed':args.seed,'termination':term,'elapsed_s':round(env.steps*CONTROL_DT,3),
                'sensor_surveys':scans,'map_coverage':round(spatial.coverage,5),
                'object_hypotheses':len(objects.tracks),'object_inspections':objects.inspections,
                'physical_touches':touches,'interactions':attempts,
                'previous_training_samples':len(prior),'new_labeled_samples':len(entries)-len(prior),
                'online_network_updates':online_updates,
                'active_network_updates':active_updates,
                'active_choices':active_choices,
                'active_pushes':active_pushes,
                'active_prior_samples':initial_active_samples,
                'active_new_labeled_samples':len(active_entries)-initial_active_samples,
                'properties_for_audit_only':audit,
                'limitations':'Idealized range, simulator truth pose, approximate AABB rays, V5 reference only; not hardware-safe or semantic object recognition'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('AUTONOMOUS_WORLD_REPORT',json.dumps(report,allow_nan=False),flush=True)

if __name__=='__main__':main()
