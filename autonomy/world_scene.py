"""Mixed-size/mass/friction multi-object MuJoCo world construction.

Hidden dynamics are only used by scene generator and evaluation. A live agent
must infer outcomes from ray scans and body contact without object mass labels.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np

from .scene import make_scene


@dataclass(frozen=True)
class PropSpec:
    label: str
    x: float
    y: float
    half_x: float
    half_y: float
    half_z: float
    mobile: bool
    mass: float
    friction: float


def sample_props(seed, count=3):
    if count!=3: raise ValueError('Current controlled environment requires 3 objects')
    rng=np.random.default_rng(seed)
    positions=[(.63,-.36),(.63,.36),(.99,.0)]
    flags=np.array([True,True,False]);rng.shuffle(flags)
    result=[]
    for idx,((x,y),mobile) in enumerate(zip(positions,flags)):
        result.append(PropSpec(
            label=f'interaction_target_{idx:02d}',
            x=float(x+rng.uniform(-.035,.035)),
            y=float(y+rng.uniform(-.025,.025)),
            half_x=float(rng.uniform(.055,.105)),
            half_y=float(rng.uniform(.065,.15)),
            half_z=float(rng.uniform(.075,.12)),
            mobile=bool(mobile),
            mass=float(rng.uniform(.025,.27)),
            friction=float(rng.uniform(.14,1.1))))
    return tuple(result)


def make_world(base_xml,seed=7):
    scene=make_scene(base_xml,seed=seed,count=0)
    root=ET.fromstring(scene.xml)
    world=root.find('worldbody')
    if world is None:raise ValueError('MuJoCo scene missing worldbody')
    specs=sample_props(seed)
    for item in specs:
        body=ET.SubElement(world,'body',{
            'name':item.label,
            'pos':f'{item.x:.6f} {item.y:.6f} {item.half_z:.6f}'})
        if item.mobile:
            ET.SubElement(body,'freejoint',{'name':f'{item.label}_free'})
        ET.SubElement(body,'geom',{
            'name':f'{item.label}_geom','type':'box',
            'size':f'{item.half_x:.6f} {item.half_y:.6f} {item.half_z:.6f}',
            'mass':f'{item.mass:.6f}',
            'friction':f'{item.friction:.6f} 0.005 0.0001',
            'rgba':'0.45 0.51 0.57 1',
            'condim':'4'})
    return ET.tostring(root,encoding='unicode'),specs,scene.obstacles
