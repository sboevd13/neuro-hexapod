"""Small goal-conditioned neural action-value model for local navigation.

A contextual one-step value learner (not a recurrent animal brain, SLAM or PPO).
Inputs are idealized ranges and a relative waypoint; outputs are the predicted
immediate utility of forward, left turn and right turn in a kinematic world.
The global frontier planner and safety interlocks remain in charge.
"""
import json
import math
from pathlib import Path

import numpy as np

from .explorer import SENSOR_RANGE

# Nine rays are a subset of the 23 simulated range beams, covering 220 degrees.
BEAM_INDEX = np.array([0, 3, 6, 8, 11, 14, 16, 19, 22], dtype=int)
ACTIONS = ((0.85, 0.0, 0.0), (0.0, 0.0, 0.85), (0.0, 0.0, -0.85))
INPUT_SIZE = 12  # 9 ranges, sin(goal angle), cos(goal angle), goal distance
HIDDEN = 32
FORMAT_VERSION = 1


def state_features(ranges, yaw, xy, waypoint):
    rays = np.asarray(ranges, dtype=float)
    xy = np.asarray(xy, dtype=float)
    waypoint = np.asarray(waypoint, dtype=float)
    if rays.shape != (23,) or xy.shape != (2,) or waypoint.shape != (2,):
        raise ValueError('Expected 23 ranges, 2D pose and 2D waypoint')
    if not (np.all(np.isfinite(rays)) and np.all(np.isfinite(xy))
            and np.all(np.isfinite(waypoint)) and np.isfinite(yaw)):
        raise ValueError('Sensor and pose values must be finite')
    delta = waypoint - xy
    angle = math.atan2(delta[1], delta[0]) - yaw
    return np.concatenate([
        np.clip(rays[BEAM_INDEX] / SENSOR_RANGE, 0, 1),
        [math.sin(angle), math.cos(angle), min(1., np.linalg.norm(delta) / 1.5)],
    ]).astype(np.float32)


def init_weights(seed=2026):
    rng = np.random.default_rng(seed)
    return {'w1': (rng.standard_normal((INPUT_SIZE, HIDDEN)) * np.sqrt(2/INPUT_SIZE)).astype(np.float32),
            'b1': np.zeros(HIDDEN, np.float32),
            'w2': (rng.standard_normal((HIDDEN, HIDDEN)) * np.sqrt(2/HIDDEN)).astype(np.float32),
            'b2': np.zeros(HIDDEN, np.float32),
            'w3': (rng.standard_normal((HIDDEN, 3)) * np.sqrt(1/HIDDEN)).astype(np.float32),
            'b3': np.zeros(3, np.float32)}


def forward(weights, x):
    x = np.asarray(x, dtype=np.float32)
    h1 = np.tanh(x @ weights['w1'] + weights['b1'])
    h2 = np.tanh(h1 @ weights['w2'] + weights['b2'])
    return h2 @ weights['w3'] + weights['b3']


class BehaviorModel:
    def __init__(self, weights):
        self.weights = {k: np.array(v, dtype=np.float32, copy=True) for k, v in weights.items()}
        sample = init_weights()
        if set(self.weights) != set(sample):
            raise ValueError('Missing or extra neural model weights')
        for k, default in sample.items():
            if self.weights[k].shape != default.shape or not np.isfinite(self.weights[k]).all():
                raise ValueError(f'Invalid weights: {k}')

    def values(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.shape not in ((INPUT_SIZE,), (features.shape[0], INPUT_SIZE)):
            raise ValueError('Unexpected sensor feature dimension')
        if not np.isfinite(features).all():
            raise ValueError('Non-finite sensor features')
        return forward(self.weights, features)

    def choose(self, features, ranges, margin=0.37):
        """Gate forward near obstructions; a learned score never removes this."""
        q = self.values(features)
        if q.shape != (3,):
            raise ValueError('Expected a single state')
        q = q.copy()
        beams = np.asarray(ranges, dtype=float)
        if beams.shape != (23,) or not np.isfinite(beams).all():
            raise ValueError('Invalid clearance sensor')
        if min(beams[10:13]) < margin:
            q[0] = -np.inf
        return int(np.argmax(q))

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': FORMAT_VERSION, 'architecture': [INPUT_SIZE, HIDDEN, HIDDEN, 3],
                   'weights': {key: value.tolist() for key, value in self.weights.items()},
                   'metadata': metadata}
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')) + '\n')
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text())
        if (obj.get('version') != FORMAT_VERSION or
                obj.get('architecture') != [INPUT_SIZE, HIDDEN, HIDDEN, 3]):
            raise ValueError('Unsupported behavior model checkpoint')
        return cls(obj['weights'])
