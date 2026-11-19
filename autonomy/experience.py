"""Feedback learning from the *chosen* local actions in physical MuJoCo steps.

Each record contains sensor features, action and measured short-horizon reward.
Unlike the synthetic pretraining script, only executed actions are labeled.
"""
import json
import math
from pathlib import Path

import numpy as np

from .behavior import BehaviorModel, INPUT_SIZE


def experience_reward(start_xy, end_xy, start_yaw, end_yaw, waypoint, contacts):
    """Relative progress and heading improvement with a collision penalty."""
    start, end, goal = (np.asarray(a, dtype=float) for a in
                        (start_xy, end_xy, waypoint))
    if any(a.shape != (2,) or not np.isfinite(a).all() for a in (start, end, goal)):
        raise ValueError('Invalid pose or waypoint')
    before = math.atan2(goal[1]-start[1], goal[0]-start[0]) - start_yaw
    after = math.atan2(goal[1]-end[1], goal[0]-end[0]) - end_yaw
    gain = 5.0*(np.linalg.norm(goal-start)-np.linalg.norm(goal-end))
    gain += 0.32*(math.cos(after)-math.cos(before)) - 0.01
    return float(np.clip(gain-3.0*max(0, contacts), -5.0, 2.0))


def read_experiences(paths):
    rows = []
    for filename in paths:
        path = Path(filename)
        with path.open() as stream:
            for lineno, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                record = json.loads(text)
                feat = np.asarray(record['features'], dtype=np.float32)
                action = record['action']
                reward = record['reward']
                if (feat.shape != (INPUT_SIZE,) or not np.isfinite(feat).all()
                        or not isinstance(action, int) or action not in (0,1,2)
                        or not np.isfinite(reward) or abs(reward)>5):
                    raise ValueError(f'Invalid experience in {path}:{lineno}')
                rows.append((feat, int(action), float(reward)))
    if not rows:
        raise ValueError('No usable experiences')
    states, actions, rewards = zip(*rows)
    return (np.array(states, np.float32), np.array(actions, np.int64),
            np.array(rewards, np.float32))


def fit_experience(model, states, actions, rewards, epochs=25, lr=0.0005, seed=17):
    """Masked Huber regression with Adam updates for executed actions only."""
    if (states.ndim != 2 or states.shape[1] != INPUT_SIZE or
            actions.shape != (len(states),) or rewards.shape != (len(states),)):
        raise ValueError('Malformed experience batch')
    if len(states) < 4:
        raise ValueError('At least four experiences are required')
    if not (np.isfinite(states).all() and np.isfinite(rewards).all()):
        raise ValueError('Non-finite experience batch')
    weights = {k: v.copy() for k, v in model.weights.items()}
    first = {k: np.zeros_like(v) for k, v in weights.items()}
    second = {k: np.zeros_like(v) for k, v in weights.items()}
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(states)), max(1, math.ceil(len(states)/64))):
            x, act, target = states[batch], actions[batch], rewards[batch]
            h1 = np.tanh(x@weights['w1']+weights['b1'])
            h2 = np.tanh(h1@weights['w2']+weights['b2'])
            q = h2@weights['w3']+weights['b3']
            grad = np.zeros_like(q)
            grad[np.arange(len(batch)), act] = np.clip(q[np.arange(len(batch)), act]-target,-1,1)/len(batch)
            gw3=h2.T@grad
            gb3=grad.sum(axis=0)
            dh2=(grad@weights['w3'].T)*(1-h2*h2)
            gw2=h1.T@dh2
            gb2=dh2.sum(axis=0)
            dh1=(dh2@weights['w2'].T)*(1-h1*h1)
            grads={'w1': x.T@dh1,'b1':dh1.sum(axis=0),
                   'w2':gw2,'b2':gb2,'w3':gw3,'b3':gb3}
            count += 1
            for key,g in grads.items():
                first[key]=0.9*first[key]+0.1*g
                second[key]=0.999*second[key]+0.001*g*g
                weights[key] -= lr*(first[key]/(1-0.9**count))/(np.sqrt(second[key]/(1-0.999**count))+1e-8)
    return BehaviorModel(weights)


def observed_mse(model, states, actions, rewards):
    q=model.values(states)
    return float(np.mean((q[np.arange(len(states)), actions]-rewards)**2))


def save_experiences(filename, rows):
    path=Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
