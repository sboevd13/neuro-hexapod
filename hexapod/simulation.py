"""Physics, onboard IMU and camera. Only the actuator adapter advances the clock."""
import mujoco
import numpy as np
from .gait import directional_reference,rpy
from .geometry import LINKS,JOINT_LOW,JOINT_HIGH
from .perception import Pinhole,DepthPerception
from .ports import DT,Sensors
from .world import CAMERA,FOV,build

class Simulation:
    def __init__(self,course='mixed',seed=0,camera_mode='depth',opening=None):
        xml,self.course=build(course,seed,opening)
        self.model=mujoco.MjModel.from_xml_string(xml)
        self.data=mujoco.MjData(self.model)
        self.rng=np.random.default_rng(seed)
        self.camera=Pinhole(fovy=FOV)
        self.perception=DepthPerception(self.camera)
        self.camera_id=self.model.camera(CAMERA).id
        self.directions=np.ascontiguousarray(self.camera.rays/self.camera.norms[:,None])
        self.groups=np.array([1,1,0,0,0,0],np.uint8)
        self.ids=np.empty(len(self.directions),np.int32)
        self.dist=np.empty(len(self.directions))
        self.camera_mode=camera_mode
        self.rgb_renderer=None
        if camera_mode=='rgb':
            from .vision import MonoPerception,RGB_WIDTH,RGB_HEIGHT
            self.perception=MonoPerception()
            self.rgb_renderer=mujoco.Renderer(self.model,RGB_HEIGHT,RGB_WIDTH)
        elif camera_mode!='depth':
            raise ValueError('Unknown camera mode')
        self.scan=None
        self.after_substep=None
        self.steps=0
        self.frame_skip=round(DT/self.model.opt.timestep)
        self.stand=directional_reference(0.,np.zeros(3),LINKS*100)
        d=self.data
        d.qpos[1]=self.rng.uniform(-.006,.006)
        yaw=self.rng.uniform(-.012,.012)
        d.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        d.qpos[7:]=self.stand
        d.ctrl[:]=self.stand
        mujoco.mj_forward(self.model,d)
        mujoco.mj_step(self.model,d,nstep=650)
        mujoco.mj_forward(self.model,d)

    @property
    def time(self):
        return self.steps*DT

    def depth(self):
        d=self.data
        m=d.cam_xmat[self.camera_id].reshape(3,3)
        orientation=np.column_stack((-m[:,2],-m[:,0],m[:,1]))
        rays=np.ascontiguousarray(self.directions@orientation.T)
        mujoco.mj_multiRay(self.model,d,d.cam_xpos[self.camera_id],rays.ravel(),self.groups,
                           1,-1,self.ids,self.dist,len(rays),20.)
        depth=self.dist/self.camera.norms
        valid=(self.ids>=0)&(self.model.geom_group[np.maximum(self.ids,0)]==0)
        depth[~valid]=np.nan
        depth+=self.rng.normal(0,.0008,len(depth))
        return depth.reshape(self.camera.height,self.camera.width)

    def observe(self):
        attitude=rpy(self.data.sensor('imu_orientation').data)+self.rng.normal(0,.0008,3)
        gyro=self.data.sensor('imu_gyro').data+self.rng.normal(0,.002,3)
        if self.scan is None or self.time-self.scan.time>=.099:
            if self.rgb_renderer is None:
                frame=self.depth()
            else:
                self.rgb_renderer.update_scene(self.data,camera=CAMERA)
                frame=self.rgb_renderer.render()
            self.scan=self.perception.read(frame,attitude,self.time)
        return Sensors(self.time,tuple(attitude),tuple(gyro),self.scan)

    def apply(self,targets):
        a=np.asarray(targets.angles)
        if (a.shape!=(18,) or not np.isfinite(a).all() or (a<JOINT_LOW-1e-8).any() or (a>JOINT_HIGH+1e-8).any()):
            raise ValueError('Invalid geometric joint target')
        self.data.ctrl[:]=a
        if self.after_substep is None:
            mujoco.mj_step(self.model,self.data,nstep=self.frame_skip)
        else:
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model,self.data)
                self.after_substep(self.data)
        self.steps+=1

    def close(self):
        if self.rgb_renderer is not None:
            self.rgb_renderer.close()
            self.rgb_renderer=None
