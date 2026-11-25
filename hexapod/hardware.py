"""Host-side adapters for real data sources. Device drivers supply timestamps on one monotonic clock."""
from dataclasses import dataclass
from time import monotonic
from typing import Callable,Protocol
import numpy as np

from .geometry import firmware_degrees
from .ports import Scan,Sensors,Targets

@dataclass(frozen=True)
class ImuReading:
    acquired: float
    attitude: tuple[float,float,float]
    angular_rate: tuple[float,float,float]

@dataclass(frozen=True)
class CameraReading:
    acquired: float
    image: np.ndarray

class Perception(Protocol):
    def read(self,image,attitude,time:float) -> Scan: ...

class LiveSensors:
    """Adapt previously acquired frames and a fused IMU to the same interface as MuJoCo.

    The getters return the latest timestamped sample, or None while disconnected.
    Camera intrinsics, mounting orientation and IMU axes must match the calibrated profile.
    """
    def __init__(self,latest_imu:Callable,latest_camera:Callable,perception:Perception,clock=monotonic):
        self.latest_imu=latest_imu
        self.latest_camera=latest_camera
        self.perception=perception
        self.clock=clock
        self.start=clock()
        self.scan=Scan(0.,valid=False)
        self.frame_time=float('-inf')

    def observe(self):
        raw=self.clock()
        now=raw-self.start
        imu=self.latest_imu()
        if imu is None or not 0.<=raw-imu.acquired<=.12:
            absent=(float('nan'),)*3
            return Sensors(now,absent,absent,Scan(now,valid=False))
        frame=self.latest_camera()
        if frame is None or not 0.<=raw-frame.acquired<=.35:
            self.scan=Scan(now,valid=False)
        elif frame.acquired>self.frame_time:
            self.frame_time=frame.acquired
            self.scan=self.perception.read(frame.image,imu.attitude,frame.acquired-self.start)
        return Sensors(now,tuple(imu.attitude),tuple(imu.angular_rate),self.scan)

class CalibratedActuators:
    """Submit a complete position frame in firmware leg order to a board driver."""
    def __init__(self,write_frame:Callable[[bytes],None]):
        self.write_frame=write_frame

    def apply(self,targets:Targets):
        self.write_frame(firmware_degrees(targets.angles).tobytes())
