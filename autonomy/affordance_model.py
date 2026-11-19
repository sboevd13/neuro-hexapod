"""Small inspectable outcome predictor; only range-derived features and probe outcomes.

No access to MuJoCo mass, friction, IDs or mobile flags in model features.
A first probing action is the sensor's equivalent of a physical experiment;
predicting unseen hidden properties from geometry alone is generally impossible.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

VERSION = 1
INPUTS = 5
HIDDEN = 16


def features(apparent_width, distance, first_displacement, contact, previous_observations=0):
    """Normalize observable properties for a stronger-push outcome forecast."""
    a = np.array([
        (float(apparent_width)-0.25)/0.25,
        (float(distance)-0.52)/0.30,
        np.clip(float(first_displacement)/0.15, -1, 2),
        float(bool(contact)),
        min(float(previous_observations), 10)/10,
    ], np.float64)
    if not np.isfinite(a).all() or a.shape != (INPUTS,):
        raise ValueError('Invalid observed object features')
    return np.clip(a, -3, 3)


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -25, 25)))


@dataclass
class OutcomePredictor:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def fresh(cls, seed=13):
        rng=np.random.default_rng(seed)
        return cls(rng.normal(0,.17,(INPUTS,HIDDEN)),
                   np.zeros(HIDDEN),rng.normal(0,.17,(HIDDEN,1)),
                   np.zeros(1))

    def predict(self, values):
        x=np.asarray(values,float)
        if x.shape[-1:] != (INPUTS,) or not np.isfinite(x).all():
            raise ValueError('Expected finite range-derived inputs')
        h=np.tanh(x@self.w1+self.b1)
        y=sigmoid(h@self.w2+self.b2)
        return float(y[0]) if x.ndim==1 else y[:,0]

    def fit(self, inputs, targets, epochs=100, lr=.015, seed=91):
        x=np.asarray(inputs,float)
        y=np.asarray(targets,float).reshape(-1,1)
        if x.ndim!=2 or x.shape[1]!=INPUTS or y.shape != (len(x),1) or len(x)<2:
            raise ValueError('Invalid supervised observations')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y,[0.,1.]).all():
            raise ValueError('Invalid input labels')
        if epochs<1 or lr<=0 or not np.isfinite(lr):
            raise ValueError('Invalid learning hyperparameters')
        rng=np.random.default_rng(seed)
        for _ in range(epochs):
            for ids in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/32)))):
                xb,yb=x[ids],y[ids]
                h=np.tanh(xb@self.w1+self.b1)
                p=sigmoid(h@self.w2+self.b2)
                dz=(p-yb)/len(ids)
                gw2=h.T@dz;gb2=dz.sum(axis=0)
                dh=(dz@self.w2.T)*(1-h*h)
                gw1=xb.T@dh;gb1=dh.sum(axis=0)
                # Clip to avoid exploding updates on tiny online datasets.
                n=np.sqrt(sum(np.sum(v*v) for v in (gw1,gw2,gb1,gb2)))
                scale=min(1.,1./max(n,1e-12))
                self.w2-=lr*gw2*scale;self.b2-=lr*gb2*scale
                self.w1-=lr*gw1*scale;self.b1-=lr*gb1*scale
        return self

    def save(self, path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        data={'version':VERSION,'type':'probe-conditioned-mlp','architecture':[INPUTS,HIDDEN,1],
              'weights':{k:getattr(self,k).tolist() for k in ('w1','b1','w2','b2')}}
        tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(data,allow_nan=False)+'\n');tmp.replace(path)

    @classmethod
    def load(cls, path):
        data=json.loads(Path(path).read_text())
        if (data.get('version')!=VERSION or data.get('type')!='probe-conditioned-mlp' or
                data.get('architecture')!=[INPUTS,HIDDEN,1]):
            raise ValueError('Unsupported predictor checkpoint')
        try: net=cls(**{k:np.asarray(data['weights'][k],float) for k in ('w1','b1','w2','b2')})
        except (KeyError,TypeError) as error: raise ValueError('Malformed predictor') from error
        shapes=((INPUTS,HIDDEN),(HIDDEN,),(HIDDEN,1),(1,))
        for name,shape in zip(('w1','b1','w2','b2'),shapes):
            v=getattr(net,name)
            if v.shape!=shape or not np.isfinite(v).all():
                raise ValueError('Invalid predictor matrix')
        return net


def evaluate(net, xs, ys, baseline=None):
    xs=np.asarray(xs,float);ys=np.asarray(ys,float)
    if len(xs)==0 or xs.ndim!=2 or xs.shape[1]!=INPUTS or ys.shape!=(len(xs),):
        raise ValueError('Invalid evaluation observations')
    p=np.clip(net.predict(xs),1e-5,1-1e-5)
    base=np.full(len(xs), np.mean(ys) if baseline is None else float(baseline))
    base=np.clip(base,1e-5,1-1e-5)
    cross=lambda v:float(np.mean(-ys*np.log(v)-(1-ys)*np.log(1-v)))
    return {'samples':len(xs),'positive_rate':round(float(ys.mean()),5),
            'neural_brier':round(float(np.mean((p-ys)**2)),5),
            'baseline_brier':round(float(np.mean((base-ys)**2)),5),
            'neural_cross_entropy':round(cross(p),5),
            'baseline_cross_entropy':round(cross(base),5),
            'neural_accuracy':round(float(np.mean((p>=.5)==ys)),5),
            'baseline_accuracy':round(float(np.mean((base>=.5)==ys)),5)}
