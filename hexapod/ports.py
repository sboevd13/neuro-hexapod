"""Body frame: metres, seconds, radians. X forward, Y left, Z up."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

DT = .02
LEGS = ('FL', 'ML', 'BL', 'FR', 'MR', 'BR')

@dataclass(frozen=True)
class Command:
    forward: float = 0.
    left: float = 0.
    turn: float = 0.

    def vector(self):
        val = np.array([self.forward, self.left, self.turn], dtype=float)
        if not np.isfinite(val).all():
            raise ValueError("Non-finite operator command")
        val = np.clip(val, -1., 1.)
        val[:2] /= max(1., np.linalg.norm(val[:2]))
        return val

@dataclass(frozen=True)
class Scan:
    time: float
    valid: bool = True
    roof_m: float | None = None
    left_m: float | None = None
    right_m: float | None = None
    wall_m: float | None = None
    ground_m: tuple[float, float, float] = (0., 0., 0.)

@dataclass(frozen=True)
class Sensors:
    time: float
    rpy: tuple[float, float, float]
    gyro: tuple[float, float, float]
    scan: Scan

@dataclass(frozen=True)
class Targets:
    angles: np.ndarray
    mode: str

class CommandSource(Protocol):
    def read(self, time: float) -> Command: ...

class SensorSource(Protocol):
    def observe(self) -> Sensors: ...

class GaitController(Protocol):
    def step(self, command: Command, sensors: Sensors, dt: float) -> Targets: ...

class Actuators(Protocol):
    def apply(self, targets: Targets) -> None: ...

def tick(source: CommandSource, sensors: SensorSource, controller: GaitController,
         drives: Actuators, dt: float = DT):
    observation = sensors.observe()
    targets = controller.step(source.read(observation.time), observation, dt)
    drives.apply(targets)
    return targets
