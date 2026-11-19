"""MuJoCo camera-assisted autonomous exploration with noisy IMU and foot odometry.

The robot's controller sees only estimated planar pose, range scans and RGB
color detections; simulator world pose is exclusively for sensor emulation and
validation. This is RGB-assisted lidar mapping, NOT learned general visual SLAM.
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
from autonomy.explorer import OccupancyMemory, FrontierExplorer
from autonomy.curiosity import CuriosityController, ObjectMemory
from autonomy.sim_perception import (add_sim_sensors, SimSensorRig, SimCamera,
                                      save_png, yaw_of_quaternion)
from autonomy.localization import PlanarEstimator, wrap
from locomotion_laptop import Config, CONTROL_DT, LocomotionEnv, MODEL_PATH


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--seconds',type=float,default=40)
    p.add_argument('--output',type=Path,default=Path('runs/visual_nav'))
    p.add_argument('--localization',choices=('estimated','odometry','oracle'),default='estimated',
                   help='Estimated scans+feet, feet/IMU only, or audit-only simulator pose')
    p.add_argument('--camera-every',type=int,default=60,
                   help='Frequency of offscreen camera frames, in 50-Hz steps')
    p.add_argument('--range-std',type=float,default=.009)
    p.add_argument('--gyro-std',type=float,default=.018)
    p.add_argument('--view',action='store_true',help='Also display the 3-D MuJoCo world')
    a=p.parse_args()
    if (a.seconds<=0 or a.camera_every<1 or a.range_std<0 or
            a.gyro_std<0):p.error('Invalid duration/IMU/camera settings')
    return a


def object_collision(model,data,ids):
    ids=set(ids)
    for j in range(data.ncon):
        c=data.contact[j]
        a,b=int(c.geom1),int(c.geom2)
        if a not in ids and b not in ids:continue
        other=b if a in ids else a
        body=int(model.geom_bodyid[other])
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or ''
        if body!=0 and not name.startswith('interaction_target_'):
            return True
    return False


def main():
    a=parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    xml,props,walls=make_world(MODEL_PATH.read_text(),seed=a.seed)
    xml=add_sim_sensors(xml)
    model=mujoco.MjModel.from_xml_string(xml)
    if model.nu!=18 or model.nq!=25+7*sum(p.mobile for p in props):
        raise RuntimeError('Sensor stack changed hexapod actuator/joint structure')
    env=LocomotionEnv(replace(Config(),episode_seconds=a.seconds+1),model)
    env.reset([0,0,0],seed=a.seed)
    geom_ids=[]
    for prop in props:
        gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,prop.label+'_geom')
        if gid<0:raise RuntimeError('Prop geometry missing')
        geom_ids.append(gid)
    sensors=SimSensorRig(model,walls,geom_ids,seed=a.seed,range_std=a.range_std,
                          gyro_std=a.gyro_std)
    estimator=PlanarEstimator(matching=a.localization=='estimated')
    memory=OccupancyMemory()
    planner=FrontierExplorer(memory)
    tracks=ObjectMemory(a.seed)
    curiosity=CuriosityController(planner,tracks,minimum_standoff=.52)
    cam=SimCamera(model)
    viewer=None
    if a.view:
        from mujoco import viewer as mj_viewer
        viewer=mj_viewer.launch_passive(model,env.data)
        viewer.cam.distance=2.7
        viewer.cam.elevation=-55
    command=np.zeros(3)
    last_vision_tick=-100000
    frames=0
    detections=0
    visible_frames=0
    snapshots=[]
    sensor_samples=0
    errors=[];heading_errors=[]
    collisions=0;last_contact=False
    goal_changes=0
    scan_count=0
    decision_counts={}
    deadline=time.perf_counter()
    stop='time_limit'
    camera_info=[]
    scan_interval=15
    try:
        for tick in range(math.ceil(a.seconds/CONTROL_DT)):
            # Sample gyro and proprioception every control tick (50 Hz).
            packet=sensors.read(env.data,scan=tick%scan_interval==0)
            odom=estimator.integrate(packet.gyro_z,packet.feet_xy,
                                     packet.contacts,CONTROL_DT,command)
            truth_xy=np.asarray(env.data.qpos[:2],float).copy()
            truth_yaw=yaw_of_quaternion(env.data.qpos[3:7])
            if tick%a.camera_every==0:
                frame,depth,found=cam.read(env.data)
                frames+=1
                detections+=len(found)
                if found:
                    visible_frames+=1
                    last_vision_tick=tick
                if frames<=2 or tick+a.camera_every>=a.seconds/CONTROL_DT:
                    filename='camera_first.png' if frames==1 else f'camera_{frames:03d}.png'
                    save_png(a.output/filename,frame)
                    snapshots.append(filename)
                if not np.all(np.isfinite(depth)):
                    raise RuntimeError('Camera returned nonfinite metric depth')
                # Add monocular bearing + RGB-D range to the observation log.
                for item in found:
                    x0,y0,x1,y1=item['bbox']
                    crop=depth[y0:y1+1,x0:x1+1]
                    measurement=dict(item)
                    measurement['depth_m']=float(np.median(crop))
                    measurement['at_s']=round(tick*CONTROL_DT,3)
                    camera_info.append(measurement)
            if tick%scan_interval==0:
                scan_count+=1
                pose=odom.copy()
                if a.localization=='oracle':
                    pose=np.array([*truth_xy,truth_yaw])
                elif a.localization=='estimated':
                    pose=estimator.correct_with_map(memory,packet.ranges,packet.hits)
                if a.localization!='oracle':
                    sensor_samples+=1
                    errors.append(float(np.linalg.norm(pose[:2]-truth_xy)))
                    heading_errors.append(abs(wrap(pose[2]-truth_yaw)))
                xy,yaw=pose[:2],float(pose[2])
                memory.observe(xy,yaw,packet.ranges,packet.hits)
                base,kind,waypoint=planner.command(xy,yaw)
                # Real camera RGB recognition gates curiosity requests. Existing
                # target can continue despite temporary occlusion in the image.
                if tick-last_vision_tick<=round(5/CONTROL_DT) or curiosity.active_id is not None:
                    cmd,kind,_=curiosity.command(xy,yaw,packet.ranges,packet.hits,
                                                  (base,kind,waypoint))
                else:
                    cmd=base
                    tracks.observe(xy,yaw,packet.ranges,packet.hits)
                command=cmd
                decision_counts[kind]=decision_counts.get(kind,0)+1
                goal_changes=planner.goals_reached
            env.set_command(command)
            _,_,fallen,truncated,_=env.step(np.zeros(18))
            touching=object_collision(model,env.data,geom_ids)
            if touching and not last_contact:
                collisions+=1
                curiosity.abandon(contact=True)
                planner.goal=None
            last_contact=touching
            if viewer is not None:
                if not viewer.is_running():stop='viewer_closed';break
                viewer.cam.lookat[:]=env.data.qpos[:3]
                viewer.sync()
                deadline+=CONTROL_DT
                if deadline>time.perf_counter():time.sleep(deadline-time.perf_counter())
            if fallen or truncated:
                stop='fall' if fallen else 'time_limit'
                break
    finally:
        if viewer is not None:viewer.close()
        cam.close()
        memory.save(a.output/'map.json',scene_seed=a.seed)
        tracks.save(a.output/'objects.json')
        if camera_info:
            (a.output/'visual_detections.json').write_text(json.dumps(camera_info,indent=2)+'\n')
        rmse=(float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
        summary={
            'scene_seed':a.seed,'localization':a.localization,
            'termination':stop,'elapsed_s':round(env.steps*CONTROL_DT,3),
            'mapped_fraction':round(memory.coverage,5),
            'tracked_object_hypotheses':len(tracks.tracks),
            'object_inspections':tracks.inspections,
            'robot_prop_contacts':collisions,
            'camera_frames':frames,'frames_with_cyan_objects':visible_frames,
            'visual_detections':detections,'camera_snapshots':snapshots,
            'scan_count':scan_count,'foot_odometry_updates':estimator.foot_updates,
            'command_fallback_updates':estimator.fallback_updates,
            'scan_matching_corrections':estimator.scan_corrections,
            'position_rmse_m':None if rmse is None else round(rmse,5),
            'final_position_error_m':None if not errors else round(errors[-1],5),
            'heading_rmse_deg':None if not heading_errors else round(math.degrees(float(
                np.sqrt(np.mean(np.square(heading_errors))))),4),
            'decisions':decision_counts,
            'controller_world_truth':a.localization=='oracle',
            'sensor_limitations':'RGB color segmentation, simulated metric depth, idealized noisy planar range; foot-local positions emulated from FK; IMU gyro; no visual odometry or full SLAM loop closure',
            'physics':'MuJoCo V5 reference; no learned 18-actuator PPO enabled',
        }
        (a.output/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print('VISUAL_NAV_REPORT',json.dumps(summary,allow_nan=False),flush=True)

if __name__=='__main__':main()
