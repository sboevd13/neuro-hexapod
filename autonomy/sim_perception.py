"""Simulation sensor boundary: RGB camera, gyro, foot encoders/contact and noisy range beams.

World truth is used ONLY to synthesize sensor readings and audit accuracy.
The navigation estimator receives no global position, world object coordinates,
freejoint flags or hidden mass. Range is still an idealized 2-D simulated lidar.
"""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

from .explorer import Rectangle, range_scan, SENSOR_RANGE

CAMERA_NAME = 'navigation_rgb'
GYRO_NAME = 'navigation_gyro'
FOOT_NAMES = ('FL','ML','BL','FR','MR','BR')


def add_sim_sensors(xml, paint_objects=True):
    """Mount camera and IMU; retain original dynamics, 18 actuators and qpos."""
    root = ET.fromstring(xml)
    world = root.find('worldbody')
    if world is None: raise ValueError('Missing worldbody')
    torso = next((b for b in world.findall('body') if b.get('name') == 'body'), None)
    if torso is None: raise ValueError('Missing torso')
    if any(c.get('name') == CAMERA_NAME for c in torso.findall('camera')):
        raise ValueError('Camera already mounted')
    # Camera optical -Z is robot forward +X, optical +Y roughly +Z.
    # 13-degree down pitch: sky above and low objects in the lower half.
    ET.SubElement(torso,'camera',{
        'name':CAMERA_NAME, 'pos':'0.055 0 0.090',
        'xyaxes':'0 -1 0 -0.225 0 0.97437', 'fovy':'72'})
    ET.SubElement(torso,'site', {'name':'navigation_imu_site',
                                  'pos':'0 0 0.025', 'size':'0.004'})
    sensors=root.find('sensor')
    if sensors is None: sensors=ET.SubElement(root,'sensor')
    ET.SubElement(sensors,'gyro',{'name':GYRO_NAME,'site':'navigation_imu_site'})
    if paint_objects:
        # The same cyan paint for ALL objects: absolutely no mass/mobility cue.
        for b in world.findall('body'):
            if b.get('name','').startswith('interaction_target_'):
                for geom in b.findall('geom'):
                    geom.set('rgba','0.13 0.72 0.83 1')
    return ET.tostring(root,encoding='unicode')


def detect_colored_objects(image, min_pixels=12):
    """Detect uniform cyan laboratory props in actual RGB camera pixels.

    This is low-level color segmentation, NOT learned semantic understanding.
    No simulator geom IDs, positions or labels are passed to the detector.
    """
    rgb=np.asarray(image)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
        raise ValueError('Expected an RGB uint8 camera frame')
    r=rgb[:,:,0].astype(np.int16)
    g=rgb[:,:,1].astype(np.int16)
    b=rgb[:,:,2].astype(np.int16)
    mask=(b-r>24)&(g-r>16)&(b>=g-6)&(g>55)&(b>70)
    # Contiguous x columns represent coarse independent visual targets.
    cols=np.flatnonzero(np.count_nonzero(mask,axis=0)>=2)
    if not len(cols):return []
    runs=[];start=previous=int(cols[0])
    for col in cols[1:]:
        col=int(col)
        if col>previous+2:
            runs.append((start,previous));start=col
        previous=col
    runs.append((start,previous))
    out=[];h,w=mask.shape
    for lo,hi in runs:
        ys,xs=np.where(mask[:,lo:hi+1]);xs=xs+lo
        if len(xs)<min_pixels:continue
        cx=float(np.mean(xs));cy=float(np.mean(ys))
        # Horizontal field of view derived from fovy=72 and aspect ratio.
        hfov=2*math.atan(math.tan(math.radians(72)/2)*(w/h))
        bearing=math.atan((cx-(w-1)/2)/((w/2)/math.tan(hfov/2)))
        out.append({'center_x_px':round(cx,2),'center_y_px':round(cy,2),
                    'area_px':int(len(xs)),'bearing_rad':float(bearing),
                    'bbox':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]})
    return out


class SimCamera:
    """Offscreen MuJoCo RGB+metric-depth renderer; snapshots are real frames."""
    def __init__(self, model, height=96, width=128):
        import mujoco
        self.model=model
        self.height=int(height);self.width=int(width)
        if self.width<32 or self.height<32: raise ValueError('Camera too small')
        self.renderer=mujoco.Renderer(model,height=self.height,width=self.width)

    def read(self,data):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data,camera=CAMERA_NAME)
        image=self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data,camera=CAMERA_NAME)
        depth=self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        if image.shape!=(self.height,self.width,3) or depth.shape!=(self.height,self.width):
            raise RuntimeError('Unexpected camera output')
        return image,depth,detect_colored_objects(image)

    def close(self):
        self.renderer.close()


def save_ppm(path,image):
    from pathlib import Path
    frame=np.asarray(image)
    if frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8:
        raise ValueError('Invalid RGB snapshot')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    h,w=frame.shape[:2]
    path.write_bytes(f'P6\n{w} {h}\n255\n'.encode()+frame.tobytes())


def yaw_of_quaternion(q):
    w,x,y,z=np.asarray(q,dtype=float)
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


@dataclass(frozen=True)
class Measurement:
    gyro_z: float
    feet_xy: np.ndarray
    contacts: np.ndarray
    ranges: np.ndarray
    hits: np.ndarray


class SimSensorRig:
    """Generate physically grounded sensor packets; never expose audit pose."""
    def __init__(self,model,world_walls,prop_geoms=(),seed=0,range_std=0.009,
                 gyro_std=0.018,foot_std=0.0015,dropout=0.015):
        import mujoco
        self.model=model
        self.walls=tuple(world_walls)
        self.prop_geoms=tuple(int(x) for x in prop_geoms)
        self.rng=np.random.default_rng(seed+505)
        self.range_std=float(range_std)
        self.gyro_std=float(gyro_std)
        self.foot_std=float(foot_std)
        self.dropout=float(dropout)
        if min(range_std,gyro_std,foot_std,dropout)<0 or dropout>1:
            raise ValueError('Invalid noise parameters')
        self.body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'body')
        self.gyro_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,GYRO_NAME)
        self.foot_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
                                        f'foot_collision_{s}') for s in FOOT_NAMES],dtype=int)
        if self.body_id<0 or self.gyro_id<0 or np.any(self.foot_ids<0):
            raise ValueError('Required simulated proprioceptive sensors missing')
        self.gyro_adr=int(model.sensor_adr[self.gyro_id])

    def proprioception(self,data):
        # In simulation, local foot coordinates are obtained by transforming
        # world FK output to torso frame. This equals encoder-derived FK; the
        # estimator NEVER receives the torso's world position or orientation.
        body_pos=np.asarray(data.xpos[self.body_id])
        rotation=np.asarray(data.xmat[self.body_id]).reshape(3,3)
        feet=np.array([(rotation.T@(data.geom_xpos[gid]-body_pos))[:2]
                       for gid in self.foot_ids])
        touched=set()
        for k in range(data.ncon):
            c=data.contact[k]
            for foot in self.foot_ids:
                if int(c.geom1)==int(foot) or int(c.geom2)==int(foot):
                    touched.add(int(foot))
        return (float(data.sensordata[self.gyro_adr+2]+self.rng.normal(0,self.gyro_std)),
                feet+self.rng.normal(0,self.foot_std,size=feet.shape),
                np.array([int(g) in touched for g in self.foot_ids],dtype=bool))

    def ranger(self,data):
        # This is the only location that uses ground-truth pose and geometry
        # for a *synthetic sensor* reading. Never pass those values to planner.
        q=data.qpos
        xy=np.asarray(q[:2],float)
        yaw=yaw_of_quaternion(q[3:7])
        boxes=list(self.walls)
        for gid in self.prop_geoms:
            pos=np.asarray(data.geom_xpos[gid],float)
            size=np.asarray(self.model.geom_size[gid],float)
            boxes.append(Rectangle(float(pos[0]),float(pos[1]),float(size[0]),float(size[1])))
        distances,hits=range_scan(xy,yaw,boxes)
        noise=self.rng.normal(0,self.range_std,size=len(distances))
        distances=np.clip(distances+noise,0.025,SENSOR_RANGE)
        lost=self.rng.random(len(distances))<self.dropout
        hits=hits & ~lost
        distances[lost]=SENSOR_RANGE
        return distances,hits

    def read(self,data,scan=True):
        gyro,feet,contacts=self.proprioception(data)
        ranges,hits=self.ranger(data) if scan else (np.empty(0),np.empty(0,dtype=bool))
        return Measurement(gyro,feet,contacts,ranges,hits)


def save_png(path,image):
    """Pure-stdlib PNG writer so CI artifacts need no Pillow dependency."""
    import struct
    import zlib
    from pathlib import Path
    frame=np.asarray(image)
    if frame.ndim!=3 or frame.shape[-1]!=3 or frame.dtype!=np.uint8:
        raise ValueError('Expected RGB uint8')
    h,w=frame.shape[:2]
    raw=b''.join(b'\x00'+frame[y].tobytes() for y in range(h))
    def chunk(name,data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    payload=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+
             chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
