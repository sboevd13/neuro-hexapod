"""Interchangeable body-relative command sources."""
import json
from pathlib import Path
import numpy as np
from .ports import Command

class Keyboard:
    def __init__(self):
        self.held=set()

    def key(self,name,pressed):
        if pressed:
            self.held.add(name)
        else:
            self.held.discard(name)

    def read(self,time):
        def axis(pos,neg):
            return int(bool(self.held.intersection(pos)))-int(bool(self.held.intersection(neg)))
        return Command(axis(('w','up'),('s','down')),axis(('a','left'),('d','right')),axis(('q',),('e',)))

class Gamepad:
    def __init__(self,device,deadzone=.12):
        self.device,self.deadzone=device,deadzone

    def read(self,time):
        if not self.device.get_init() or self.device.get_numaxes()<2:
            return Command()
        def axis(i):
            x=self.device.get_axis(i)
            return np.sign(x)*max(0.,abs(x)-self.deadzone)/(1.-self.deadzone)
        return Command(-axis(1),-axis(0),-axis(2) if self.device.get_numaxes()>2 else 0.)

class Replay:
    def __init__(self,frames):
        if not frames:
            raise ValueError('Empty command sequence')
        self.times=np.array([f['at'] for f in frames],dtype=float)
        if not np.isfinite(self.times).all() or self.times[0]<0 or np.any(np.diff(self.times)<=0):
            raise ValueError('Replay timestamps must increase')
        self.commands=[Command(f.get('forward',0.),f.get('left',0.),f.get('turn',0.)) for f in frames]
        for cmd in self.commands:
            cmd.vector()

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text()))

    def read(self,time):
        i=np.searchsorted(self.times,time,side='right')-1
        return self.commands[i] if i>=0 else Command()
