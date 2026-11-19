"""Uncertainty-aware, sensor-only outcomes for bounded MuJoCo push experiments.

All world-model features are obtained before the *second* physical action:
[apparent width, range, short-probe displacement, contact, visits, action length].
Mass, friction, object IDs, freejoint status and simulator truth are forbidden.
Ensemble disagreement is an *epistemic heuristic*, NOT a calibrated interval.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

VERSION = 1
BASE_FEATURES = 5
FEATURES = 6
ACTION_DURATIONS = (0.24, 0.55)  # bounded additional pushes, seconds
OUTCOME_CLIP = (-0.06, 0.30)  # signed displacement in metres
HIDDEN = 20


def experiment_features(sensor_features, duration):
    base = np.asarray(sensor_features, dtype=float)
    if base.shape != (BASE_FEATURES,) or not np.isfinite(base).all():
        raise ValueError('Expected 5 finite pre-action sensor features')
    if duration not in ACTION_DURATIONS:
        raise ValueError('Unrecognized bounded experimental push duration')
    return np.concatenate((base, [(float(duration) - 0.395) / .155]))


def sanitize(rows):
    """Limit training data to observable pre-action features and sensor outcomes."""
    cleaned = []
    for row in rows:
        if not row.get('valid') or row.get('fell'):
            continue
        value = row.get('total_sensed_displacement_m')
        if value is None or not np.isfinite(value):
            continue
        x = experiment_features(row['features'], row['second_push_seconds'])
        cleaned.append({'features': x.tolist(),
                        'displacement_m': float(np.clip(value, *OUTCOME_CLIP)),
                        'action_seconds': float(row['second_push_seconds'])})
    return cleaned


def dataset(rows):
    xs = np.asarray([x['features'] for x in rows], float)
    ys = np.asarray([x['displacement_m'] for x in rows], float)
    if len(rows) < 2 or xs.shape != (len(rows), FEATURES) or ys.shape != (len(rows),):
        raise ValueError('Not enough valid sensor-only observations')
    if not np.isfinite(xs).all() or not np.isfinite(ys).all() or np.any(ys < OUTCOME_CLIP[0]) or np.any(ys > OUTCOME_CLIP[1]):
        raise ValueError('Invalid experiment data')
    return xs, ys


@dataclass
class TinyRegressor:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def fresh(cls, seed):
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, .24, (FEATURES, HIDDEN)), np.zeros(HIDDEN),
                   rng.normal(0, .14, (HIDDEN, 1)), np.zeros(1))

    def predict(self, xs):
        x = np.asarray(xs, float)
        if x.shape[-1:] != (FEATURES,) or not np.isfinite(x).all():
            raise ValueError('Invalid observable input')
        h = np.tanh(x @ self.w1 + self.b1)
        # Keep members untruncated: clipping individual predictions erases
        # disagreement for out-of-distribution objects (false certainty).
        return (h @ self.w2 + self.b2).reshape(x.shape[:-1]) * .12

    def train(self, xs, ys, steps=100, lr=.015, seed=0):
        x, y = np.asarray(xs, float), np.asarray(ys, float)
        if x.ndim != 2 or x.shape[1] != FEATURES or y.shape != (len(x),) or len(x) < 2:
            raise ValueError('Invalid fit data')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or steps < 1 or lr <= 0:
            raise ValueError('Invalid fit parameters')
        rng = np.random.default_rng(seed)
        momentum = [np.zeros_like(v) for v in (self.w1, self.b1, self.w2, self.b2)]
        variance = [v.copy() for v in momentum]
        weights = [self.w1, self.b1, self.w2, self.b2]
        batch_size = min(32, len(x))
        count = 0
        for _ in range(steps):
            for indices in np.array_split(rng.permutation(len(x)),
                                          int(np.ceil(len(x) / batch_size))):
                xb, yb = x[indices], y[indices, None] / .12
                h = np.tanh(xb @ self.w1 + self.b1)
                pred = h @ self.w2 + self.b2
                residual = pred - yb
                d = np.clip(residual, -1.5, 1.5) / len(indices)
                gw2 = h.T @ d + .002 * self.w2
                gb2 = d.sum(axis=0)
                dh = (d @ self.w2.T) * (1 - h*h)
                gw1 = xb.T @ dh + .002 * self.w1
                gb1 = dh.sum(axis=0)
                grads = [gw1, gb1, gw2, gb2]
                mag = np.sqrt(sum(float(np.sum(g*g)) for g in grads))
                if mag > 2:
                    grads = [g * 2/mag for g in grads]
                count += 1
                for i, (weight, grad) in enumerate(zip(weights, grads)):
                    momentum[i] = .9 * momentum[i] + .1 * grad
                    variance[i] = .999 * variance[i] + .001 * grad * grad
                    weight -= lr * (momentum[i] / (1-.9**count)) / (np.sqrt(variance[i] / (1-.999**count)) + 1e-8)
        return self


class OutcomeEnsemble:
    """Bootstrap ensemble. Std is disagreement; cannot be treated as calibrated risk."""
    def __init__(self, members):
        if len(members) < 3:
            raise ValueError('Need at least 3 independent members')
        self.members = members

    @classmethod
    def fit(cls, xs, ys, seed=11, members=5, steps=90):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        if xs.ndim != 2 or xs.shape[1] != FEATURES or len(xs) < 3 or ys.shape != (len(xs),):
            raise ValueError('Invalid dataset for ensemble')
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError('Nonfinite ensemble examples')
        rng = np.random.default_rng(seed)
        nets = []
        for i in range(members):
            ids = rng.integers(0, len(xs), size=len(xs))
            net = TinyRegressor.fresh(seed * 103 + i * 31 + 7)
            net.train(xs[ids], ys[ids], steps=steps, seed=seed + i * 23)
            nets.append(net)
        return cls(nets)

    def predict(self, xs):
        x = np.asarray(xs, float)
        if x.shape[-1:] != (FEATURES,) or not np.isfinite(x).all():
            raise ValueError('Expected preaction observable features')
        pred = np.stack([net.predict(x) for net in self.members], axis=0)
        # Clamp predicted mean to the measured outcome range, not epistemic spread.
        return np.clip(pred.mean(axis=0), *OUTCOME_CLIP), pred.std(axis=0)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        params = []
        for net in self.members:
            params.append({key: getattr(net, key).tolist() for key in ('w1','b1','w2','b2')})
        payload = {'version': VERSION, 'type': 'sensor-push-regression-ensemble',
                   'architecture': [FEATURES,HIDDEN,1], 'models': params}
        tmp = path.with_name('.'+path.name+'.tmp')
        tmp.write_text(json.dumps(payload, allow_nan=False)+'\n')
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if (payload.get('version') != VERSION or payload.get('type') != 'sensor-push-regression-ensemble' or
            payload.get('architecture') != [FEATURES,HIDDEN,1] or
            not isinstance(payload.get('models'), list)):
            raise ValueError('Invalid regression checkpoint')
        nets=[]
        for obj in payload['models']:
            if not isinstance(obj, dict): raise ValueError('Malformed ensemble member')
            try: net=TinyRegressor(**{key:np.asarray(obj[key],dtype=float) for key in ('w1','b1','w2','b2')})
            except (KeyError,TypeError) as e: raise ValueError('Malformed model weights') from e
            for key,shape in zip(('w1','b1','w2','b2'),
                                 ((FEATURES,HIDDEN),(HIDDEN,),(HIDDEN,1),(1,))):
                a=getattr(net,key)
                if a.shape != shape or not np.isfinite(a).all():
                    raise ValueError('Invalid weight tensor')
            nets.append(net)
        return cls(nets)


def regression_metrics(model, xs, truth, baseline=None):
    xs = np.asarray(xs,float); truth=np.asarray(truth,float)
    if len(xs)==0 or xs.shape != (len(truth),FEATURES): raise ValueError('Bad evaluation data')
    mean, uncertainty = model.predict(xs)
    baseline = float(np.mean(truth)) if baseline is None else float(baseline)
    return {'n': len(truth),
            'mae_m': round(float(np.mean(np.abs(mean-truth))), 6),
            'rmse_m': round(float(np.sqrt(np.mean((mean-truth)**2))),6),
            'constant_mae_m':round(float(np.mean(np.abs(baseline-truth))),6),
            'constant_rmse_m':round(float(np.sqrt(np.mean((baseline-truth)**2))),6),
            'mean_ensemble_disagreement_m':round(float(np.mean(uncertainty)),6)}


def acquire(strategy, model, known_xs, candidates, count, rng):
    """Select unrevealed physical outcomes using ONLY observable inputs.

    Unknown labels must never be supplied. Short probe observations for the
    candidate pool are available before selecting longer experiments.
    """
    x = np.asarray(candidates, float)
    if x.ndim != 2 or x.shape[1] != FEATURES or count < 1 or count > len(x):
        raise ValueError('Invalid selection candidates')
    if strategy == 'random':
        return rng.permutation(len(x))[:count].tolist()
    if strategy == 'heuristic':
        # Choose contacts whose *short probe* already moved most, plus novelty.
        priority = np.abs(x[:,2]) + .05*np.abs(x[:,0])
    elif strategy == 'uncertainty':
        _, disagreement = model.predict(x)
        old = np.asarray(known_xs,float)
        nearest = np.min(np.linalg.norm(x[:,None,:]-old[None,:,:],axis=-1),axis=1)
        # Ensemble disagreement plus novelty, measured before second action.
        priority = disagreement + .012 * nearest
    else:
        raise ValueError('Unknown experimental acquisition strategy')
    # Greedy diversity term avoids choosing near-duplicates in each batch.
    chosen=[]
    for _ in range(count):
        if chosen:
            diverse = np.min(np.linalg.norm(x[:,None,:]-x[np.asarray(chosen)][None,:,:],axis=-1),axis=1)
            ranks = priority + .004*diverse
        else:
            ranks = priority.copy()
        if chosen: ranks[np.asarray(chosen)] = -np.inf
        chosen.append(int(np.argmax(ranks)))
    return chosen


def choose_action(model, sensor_features, safety_ok=True, novelty_weight=1.0):
    """Choose observe, gentle, or longer probe after one short contact.

    Returns an interpretable decision. Safety checks outside this model remain
    mandatory; no real-force or torque limits are inferred by this function.
    """
    if not safety_ok:
        return {'action':'observe','duration':0.,'reason':'safety_veto'}
    xx=np.stack([experiment_features(sensor_features,t) for t in ACTION_DURATIONS])
    mean, uncertainty=model.predict(xx)
    # Learning reward favors epistemic uncertainty but penalizes action time.
    benefit = np.maximum(mean,0.) + novelty_weight*uncertainty*.8
    utility = benefit - np.asarray(ACTION_DURATIONS)*.05
    option=int(np.argmax(utility))
    # A controlled epistemic query can be valuable even when the current
    # expected motion is small. Cap its duration to the gentle action.
    # It is an explicit heuristic for *simulator-only* experiments, not a
    # safety certificate or a general reinforcement-learning policy.
    if novelty_weight >= 1.5 and float(np.max(uncertainty)) >= .003:
        idx=0
        return {'action':'gentle_push','duration':ACTION_DURATIONS[idx],
                'reason':'epistemic_query',
                'predicted_m':round(float(mean[idx]),5),
                'disagreement_m':round(float(uncertainty[idx]),5)}
    if utility[option] < .014:
        return {'action':'observe','duration':0.,'reason':'low_predicted_benefit',
                'predicted_m':round(float(mean[option]),5),
                'disagreement_m':round(float(uncertainty[option]),5)}
    return {'action':'gentle_push' if option==0 else 'long_push',
            'duration': ACTION_DURATIONS[option],
            'predicted_m':round(float(mean[option]),5),
            'disagreement_m':round(float(uncertainty[option]),5)}
