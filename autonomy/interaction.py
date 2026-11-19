"""Sensor-based object affordance learning for *simulation-only* push probes.

The policy sees idealized range scans and a coarse visual-geometry signature. It
never sees MuJoCo body IDs, freejoint flags, mass or ground-truth displacement.
MuJoCo world positions are allowed in the sensor emulator and the evaluator.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np

from .explorer import SCAN_ANGLES

VERSION = 1
ACTIONS = ('observe', 'probe')


def visible_points(xy, yaw, distances, hits, angles=SCAN_ANGLES):
    xy = np.asarray(xy, float)
    distances = np.asarray(distances, float)
    hits = np.asarray(hits, bool)
    angles = np.asarray(angles, float)
    if (xy.shape != (2,) or not np.isfinite(xy).all() or
            not np.isfinite(yaw) or not np.isfinite(distances).all() or
            np.any(distances < 0) or distances.shape != hits.shape or
            distances.shape != angles.shape):
        raise ValueError('Invalid range observation')
    if not np.any(hits):
        return np.empty((0, 2))
    theta = yaw + angles[hits]
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    return xy[None, :] + distances[hits, None]*directions


def object_signature(xy, yaw, distances, hits, angles=SCAN_ANGLES):
    """A view-derived coarse width class; no simulator object metadata.

    A 23-beam 220-degree range scan cannot identify arbitrary objects. We use
    the number of neighboring returns on one isolated target at a fixed start
    range; do not transfer this classifier to camera data or new distances.
    """
    hits = np.asarray(hits, bool)
    distances = np.asarray(distances, float)
    angles = np.asarray(angles, float)
    if not (hits.shape == distances.shape == angles.shape):
        raise ValueError('Shape mismatch')
    if not np.any(hits):
        return None
    if not np.isfinite(distances).all() or not np.isfinite(yaw):
        raise ValueError('Bad measurements')
    front = hits & (np.abs(angles) < math.radians(65)) & (distances < 1.05)
    if not np.any(front):
        return None
    points = visible_points(xy, yaw, distances, front, angles)
    lateral = np.array([-math.sin(yaw), math.cos(yaw)])
    # Include the angular beam footprint; otherwise single-beam objects appear
    # to have zero width. This is an estimate of apparent width, not true size.
    apparent_width = (float(np.ptp(points@lateral)) +
                      2*float(np.min(distances[front]))*math.tan(math.radians(5)))
    return 'wide' if apparent_width >= 0.25 else 'narrow'


def target_surface(xy, yaw, distances, hits, angles=SCAN_ANGLES):
    """Estimate world-frame target surface from laser returns, not geom_xpos."""
    distances = np.asarray(distances, float)
    hits = np.asarray(hits, bool)
    angles = np.asarray(angles, float)
    # In the isolated object trial there is one target; map all its returns.
    selected = hits & (np.abs(angles) < math.radians(80))
    pts = visible_points(xy, yaw, distances, selected, angles)
    return None if not len(pts) else np.median(pts, axis=0)


def observed_motion(before, after, push_axis=(1, 0)):
    if before is None or after is None:
        return None
    b, a = np.asarray(before,float),np.asarray(after,float)
    axis=np.asarray(push_axis,float)
    if b.shape!=(2,) or a.shape!=(2,) or axis.shape!=(2,) or not np.isfinite(b).all() or not np.isfinite(a).all() or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('Invalid surfaces / push axis')
    return float(np.dot(a-b,axis/np.linalg.norm(axis)))


@dataclass
class OutcomeMemory:
    """A two-action experience learner with Beta-smoothed mobility estimates.

    Novel signatures get one limited physical probe. Later the agent observes
    signatures that have proven immobile; probes ones that responded. This is
    deliberately interpretable learning, not a trained neural network.
    """
    observations: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def estimate(self, signature):
        data=self.observations.get(signature, {'moved':0,'not_moved':0})
        return (data['moved']+1)/(data['moved']+data['not_moved']+2)

    def choose(self, signature):
        if signature not in ('narrow','wide'):
            return 'observe'
        data=self.observations.get(signature)
        if not data or data['moved']+data['not_moved']==0:
            return 'probe'  # explicitly bounded, one trial per new signature
        return 'probe' if self.estimate(signature)>=0.5 else 'observe'

    def learn(self, signature, action, sensor_displacement, touch, fallen, label):
        if signature not in ('narrow','wide') or action not in ACTIONS:
            raise ValueError('Unknown signature or action')
        if sensor_displacement is not None and not np.isfinite(sensor_displacement):
            raise ValueError('Invalid motion')
        moved=(sensor_displacement is not None and sensor_displacement>=0.035)
        learned=(action=='probe' and bool(touch) and not bool(fallen)
                 and sensor_displacement is not None)
        if learned:
            state=self.observations.setdefault(signature, {'moved':0,'not_moved':0})
            state['moved' if moved else 'not_moved']+=1
        outcome={
            'case':str(label), 'signature':signature, 'action':action,
            'sensed_motion_m':None if sensor_displacement is None else round(sensor_displacement,5),
            'touch':bool(touch), 'fell':bool(fallen), 'learned':bool(learned),
            'inferred_moved':bool(moved) if learned else None,
            'posterior_pushability':round(self.estimate(signature),5),
        }
        self.history.append(outcome)
        return outcome

    def save(self,filename):
        path=Path(filename); path.parent.mkdir(parents=True, exist_ok=True)
        data={'version':VERSION,'kind':'object_affordance_beta',
              'observations':self.observations,'history':self.history}
        tmp=path.with_name('.'+path.name+'.tmp')
        tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        tmp.replace(path)

    @classmethod
    def load(cls,filename):
        data=json.loads(Path(filename).read_text())
        if data.get('version')!=VERSION or data.get('kind')!='object_affordance_beta':
            raise ValueError('Incompatible interaction memory')
        obs=data.get('observations')
        if not isinstance(obs,dict) or not isinstance(data.get('history'),list):
            raise ValueError('Invalid interaction memory')
        for sign,count in obs.items():
            if sign not in ('narrow','wide') or not isinstance(count,dict) or set(count)!= {'moved','not_moved'} or any(type(x) is not int or x<0 for x in count.values()):
                raise ValueError('Corrupt interaction prior')
        return cls(observations=obs,history=data['history'])
