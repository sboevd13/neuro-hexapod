"""Supervised RGB perception using rendered images and calibrated depth labels."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from .controller import Controller
from .gait import rpy
from .ports import Command,DT
from .simulation import Simulation
from .vision import RGB_WIDTH,RGB_HEIGHT,MODEL,CLASSES,features,label
from .world import CAMERA

def examples(seed,count):
    rng=np.random.default_rng(seed)
    samples=[]
    for name in ('flat','low','narrow','rough','blocked'):
        opening=None
        if seed%5==0 and name=='low':
            opening=rng.uniform(.092,.108)
        if seed%5==0 and name=='narrow':
            opening=rng.uniform(.29,.35)
        rig=Simulation(name,seed,opening=opening)
        with mujoco.Renderer(rig.model,RGB_HEIGHT,RGB_WIDTH) as renderer:
            for _ in range(count if name!='flat' else count//3):
                rig.data.qpos[:3]=[rng.uniform(-.22,1.25),rng.uniform(-.09,.09),rng.uniform(.052,.117)]
                angles=rng.uniform([-.11,-.14,-.22],[.11,.14,.22])
                q=np.empty(4)
                mujoco.mju_euler2Quat(q,angles,'xyz')
                rig.data.qpos[3:7]=q
                mujoco.mj_forward(rig.model,rig.data)
                attitude=rpy(rig.data.sensor('imu_orientation').data)
                rig.perception.floor=-.15
                scan=rig.perception.read(rig.depth(),attitude,0.)
                if not scan.valid:
                    continue
                renderer.update_scene(rig.data,camera=CAMERA)
                img=renderer.render()
                img=np.uint8(np.clip(img*rng.uniform(.83,1.15)+rng.normal(0,1.,img.shape),0,255))
                samples.append((features(img,attitude),label(scan)))
    return np.stack([x for x,y in samples]),np.stack([y for x,y in samples])

def moving_examples(seed):
    rng=np.random.default_rng(seed+950)
    samples=[]
    for name in ('flat','low','narrow','rough','blocked','mixed'):
        rig=Simulation(name,seed)
        ctrl=Controller()
        with mujoco.Renderer(rig.model,RGB_HEIGHT,RGB_WIDTH) as renderer:
            for step in range(2100 if name=='mixed' else 850):
                obs=rig.observe()
                if step%5==0 and obs.scan.valid:
                    renderer.update_scene(rig.data,camera=CAMERA)
                    img=np.uint8(np.clip(renderer.render()*rng.uniform(.94,1.06),0,255))
                    samples.append((features(img,obs.rpy),label(obs.scan)))
                target=ctrl.step(Command(forward=1.),obs,DT)
                rig.apply(target)
                if rig.data.qpos[0]>rig.course.finish or (step>150 and target.mode=='blocked'):
                    break
    return np.stack([x for x,y in samples]),np.stack([y for x,y in samples])

def main():
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import confusion_matrix
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=MODEL)
    p.add_argument('--report',type=Path)
    p.add_argument('--seeds',type=int,default=22)
    p.add_argument('--poses',type=int,default=65)
    args=p.parse_args()
    batches=[examples(800+i,args.poses) for i in range(args.seeds+4)]
    moving=[moving_examples(seed) for seed in range(930,942)]
    train=batches[:-4]+moving[:-3]
    test=batches[-4:]+moving[-3:]
    x,y=(np.concatenate([v[j] for v in train]) for j in (0,1))
    xt,yt=(np.concatenate([v[j] for v in test]) for j in (0,1))
    mean=x.mean(0)
    scale=np.maximum(x.std(0),.06)
    model=MLPRegressor(hidden_layer_sizes=(80,48),max_iter=150,early_stopping=True,
                       validation_fraction=.12,random_state=173,batch_size=128,
                       learning_rate_init=.0012,n_iter_no_change=18)
    model.fit((x-mean)/scale,y)
    predicted=model.predict((xt-mean)/scale)[:,:5].argmax(1)
    actual=yt[:,:5].argmax(1)
    report=dict(train_samples=len(x),held_out_samples=len(xt),held_out_seeds=list(range(800+args.seeds,804+args.seeds)),
                moving_held_out_seeds=[939,940,941],classes=CLASSES,
                confusion_matrix=confusion_matrix(actual,predicted,labels=range(5)).tolist(),
                accuracy=float((predicted==actual).mean()),epochs=int(model.n_iter_))
    print(json.dumps(report),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    arrays={'mean':mean,'scale':scale}
    for i,(w,b) in enumerate(zip(model.coefs_,model.intercepts_)):
        arrays[f'w{i}']=w
        arrays[f'b{i}']=b
    np.savez_compressed(args.output,**arrays)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
