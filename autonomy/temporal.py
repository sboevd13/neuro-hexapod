"""Discrete temporal-difference navigation learning on *executed* MuJoCo actions.

This is an optional local waypoint controller. It uses a replay buffer of
transitions (observation, executed action, measured reward, next observation,
terminal), learns a discounted action value via Double DQN, and retains a
non-neural frontier planner and an independent forward-clearance shield.

It does not perform image understanding, mapping, localization, or hardware
control. The simulator still provides idealized ranges and ground-truth pose.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .behavior import BehaviorModel, INPUT_SIZE, HIDDEN

VERSION = 2
ACTIONS = 3


def safe_mask(ranges, margin=0.37):
    """Always permit turning; block forward when any frontal ray is too near."""
    ranges = np.asarray(ranges, dtype=float)
    if ranges.shape != (23,) or not np.isfinite(ranges).all() or np.any(ranges < 0):
        raise ValueError('Expected 23 finite nonnegative range readings')
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('Expected a positive margin')
    return np.array([min(ranges[10:13]) >= margin, True, True], dtype=bool)


def reward_from_motion(start_xy, end_xy, start_yaw, end_yaw, waypoint,
                       coverage_gain, new_contacts=0, fallen=False):
    """Dense route reward plus *newly mapped* area, with strong safety penalties.

    Each reward belongs to one high-level 0.3-s action, not the 50-Hz servo
    actions. Only changes observed after execution are used.
    """
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    waypoint = np.asarray(waypoint, dtype=float)
    if (any(x.shape != (2,) or not np.isfinite(x).all() for x in (start, end, waypoint))
            or not all(np.isfinite(x) for x in (start_yaw, end_yaw, coverage_gain))
            or not isinstance(new_contacts, (int, np.integer)) or new_contacts < 0):
        raise ValueError('Invalid movement feedback')
    old_distance = np.linalg.norm(waypoint-start)
    new_distance = np.linalg.norm(waypoint-end)
    old_error = math.atan2(waypoint[1]-start[1], waypoint[0]-start[0])-start_yaw
    new_error = math.atan2(waypoint[1]-end[1], waypoint[0]-end[0])-end_yaw
    result = (4.0*(old_distance-new_distance)
              + 0.22*(math.cos(new_error)-math.cos(old_error))
              + 5.0*max(0.0, coverage_gain) - 0.025
              - 3.0*new_contacts - (5.0 if fallen else 0.0))
    return float(np.clip(result, -8.0, 3.0))


class TemporalModel:
    """Validated temporal Q checkpoint, distinct from one-step behavior_v1."""
    def __init__(self, weights):
        self.actor = BehaviorModel(weights)

    @property
    def weights(self):
        return self.actor.weights

    def values(self, states):
        return self.actor.values(states)

    def choose(self, state, ranges, epsilon=0.0, rng=None):
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError('Expected epsilon in [0,1]')
        mask = safe_mask(ranges)
        possible = np.flatnonzero(mask)
        if epsilon > 0:
            if rng is None:
                raise ValueError('An explicit random generator is required for exploration')
            if rng.random() < epsilon:
                return int(rng.choice(possible))
        q = self.values(state).copy()
        if q.shape != (3,):
            raise ValueError('Expected one observation')
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': VERSION, 'kind': 'temporal_double_dqn',
                   'architecture': [INPUT_SIZE, HIDDEN, HIDDEN, ACTIONS],
                   'weights': {k: value.tolist() for k, value in self.weights.items()},
                   'metadata': metadata}
        tmp = path.with_name('.'+path.name+'.tmp')
        tmp.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':'))+'\n')
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if (payload.get('version') != VERSION or payload.get('kind') != 'temporal_double_dqn'
                or payload.get('architecture') != [INPUT_SIZE, HIDDEN, HIDDEN, ACTIONS]):
            raise ValueError('Not a compatible temporal Q-learning checkpoint')
        return cls(payload['weights'])


@dataclass(frozen=True)
class Replay:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_masks: np.ndarray

    def validate(self):
        n = len(self.states)
        if (n == 0 or self.states.shape != (n, INPUT_SIZE)
                or self.actions.shape != (n,) or self.rewards.shape != (n,)
                or self.next_states.shape != (n, INPUT_SIZE)
                or self.dones.shape != (n,)
                or self.next_masks.shape != (n, ACTIONS)):
            raise ValueError('Replay shape mismatch or empty replay')
        if not (np.isfinite(self.states).all() and np.isfinite(self.rewards).all()
                and np.isfinite(self.next_states).all()):
            raise ValueError('Replay contains non-finite values')
        if not (np.isin(self.actions, [0,1,2]).all()
                and (self.rewards >= -8).all() and (self.rewards <= 3).all()
                and np.isin(self.dones, [0,1]).all()
                and np.all(self.next_masks[:, 1:])):
            raise ValueError('Invalid action, reward, terminal flag or action masks')
        return self


def read_replay(paths):
    fields = ('state', 'action', 'reward', 'next_state', 'done', 'next_mask')
    rows = []
    for filename in paths:
        with Path(filename).open() as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get('version') != VERSION or any(key not in obj for key in fields):
                    raise ValueError(f'Incompatible transition in {filename}:{lineno}')
                if type(obj['action']) is not int or type(obj['done']) is not bool:
                    raise ValueError(f'Invalid transition types in {filename}:{lineno}')
                if type(obj['reward']) not in (float,int) or any(type(z) is not bool for z in obj['next_mask']):
                    raise ValueError(f'Invalid reward/mask types in {filename}:{lineno}')
                rows.append(obj)
    if not rows:
        raise ValueError('No transitions were collected')
    replay = Replay(np.array([x['state'] for x in rows], np.float32),
                    np.array([x['action'] for x in rows], np.int64),
                    np.array([x['reward'] for x in rows], np.float32),
                    np.array([x['next_state'] for x in rows], np.float32),
                    np.array([x['done'] for x in rows], np.float32),
                    np.array([x['next_mask'] for x in rows], bool))
    return replay.validate()


def write_replay(path, rows):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a killed run never exposes a truncated replay as complete.
    tmp = dest.with_name('.'+dest.name+'.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
    tmp.replace(dest)


def double_dqn_targets(online, target, next_states, rewards, dones,
                       next_masks, gamma=0.96):
    """Online argmax; lagging target evaluates it. Never bootstrap a terminal."""
    if not 0 <= gamma < 1:
        raise ValueError('Discount factor must be in [0,1)')
    next_q = np.asarray(online.values(next_states)).copy()
    next_q[~next_masks] = -np.inf
    greedy_actions = np.argmax(next_q, axis=1)
    reference_q = target.values(next_states)
    value = reference_q[np.arange(len(rewards)), greedy_actions]
    return np.clip(rewards + gamma*(1-dones)*value, -8.0, 8.0)


def train_double_dqn(replay, initial, epochs=40, gamma=0.96, lr=0.0003,
                     batch_size=64, target_every=5, seed=19):
    """Replay-based Double DQN with frozen target copies and Huber/Adam SGD.

    Offline updates on a fixed replay buffer are an *experiment*, not an
    assurance of optimal control: coverage must be verified in new MuJoCo rooms.
    """
    replay.validate()
    if (epochs <= 0 or batch_size <= 0 or target_every <= 0 or lr <= 0
            or not np.isfinite(lr) or not 0 <= gamma < 1):
        raise ValueError('Invalid hyperparameters')
    weights = {k: v.copy() for k, v in initial.weights.items()}
    target_weights = {k: v.copy() for k, v in weights.items()}
    momentum = {k: np.zeros_like(v) for k,v in weights.items()}
    variance = {k: np.zeros_like(v) for k,v in weights.items()}
    rng = np.random.default_rng(seed)
    t = 0
    losses = []
    n = len(replay.states)
    for epoch in range(epochs):
        # Targets stay fixed for a few epochs, then synchronize.
        if epoch and epoch % target_every == 0:
            target_weights = {k: v.copy() for k,v in weights.items()}
        online = TemporalModel(weights)
        target = TemporalModel(target_weights)
        for indices in np.array_split(rng.permutation(n), math.ceil(n/batch_size)):
            x = replay.states[indices]
            a = replay.actions[indices]
            b = double_dqn_targets(online, target, replay.next_states[indices],
                                    replay.rewards[indices], replay.dones[indices],
                                    replay.next_masks[indices], gamma)
            h1 = np.tanh(x @ weights['w1'] + weights['b1'])
            h2 = np.tanh(h1 @ weights['w2'] + weights['b2'])
            pred = h2 @ weights['w3'] + weights['b3']
            residual = pred[np.arange(len(a)), a] - b
            losses.append(float(np.mean(np.where(np.abs(residual) <= 1,
                                           0.5*residual*residual, np.abs(residual)-0.5))))
            grads_out = np.zeros_like(pred)
            grads_out[np.arange(len(a)), a] = np.clip(residual, -1, 1)/len(a)
            g3 = h2.T @ grads_out
            gb3 = grads_out.sum(axis=0)
            dh2 = (grads_out @ weights['w3'].T)*(1-h2*h2)
            g2 = h1.T @ dh2
            gb2 = dh2.sum(axis=0)
            dh1 = (dh2 @ weights['w2'].T)*(1-h1*h1)
            grads = {'w1':x.T @ dh1,'b1':dh1.sum(axis=0),
                     'w2':g2,'b2':gb2,'w3':g3,'b3':gb3}
            norm = float(np.sqrt(sum(np.sum(g*g) for g in grads.values())))
            if norm > 1.0:
                grads = {key: value/norm for key,value in grads.items()}
            t += 1
            for key, g in grads.items():
                momentum[key] = 0.9*momentum[key]+0.1*g
                variance[key] = 0.999*variance[key]+0.001*g*g
                weights[key] -= lr*(momentum[key]/(1-0.9**t)) / (
                    np.sqrt(variance[key]/(1-0.999**t))+1e-8)
            # Keep target selection fresh within the epoch, but not the target network.
            online = TemporalModel(weights)
    trained = TemporalModel(weights)
    mean_td = float(np.mean(np.abs(
        trained.values(replay.states)[np.arange(n), replay.actions] -
        double_dqn_targets(trained, TemporalModel(target_weights), replay.next_states,
                           replay.rewards, replay.dones, replay.next_masks, gamma))))
    return trained, {'n':n,'epochs':epochs,'gamma':gamma,'mean_huber_train_loss':float(np.mean(losses[-max(1, math.ceil(n/batch_size)):])),'mean_td_error':mean_td}
