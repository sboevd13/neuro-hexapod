"""Shared V3 physics, directional V5 controller and deployable observations.

Native MuJoCo is used identically by training, evaluation and the viewer. The
XML's implicitfast integrator is intentional: Euler + disabled damping makes
these stiff position servos unstable. No simulator joint/velocity feedback is
exposed to the actor; orientation and angular rate stand in for an IMU.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import math

import mujoco
import numpy as np

import reference_gait as v5
from hexapod.gait import clean_command, directional_reference, rpy, wrap_angle, STAND

VERSION = "directional_v5_residual_ppo_v3"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "arena.xml"
OBS_SIZE = 65
ACTION_SIZE = 18
CONTROL_DT = 0.020
@dataclass(frozen=True)
class Config:
    residual_deg: tuple = (3.0, 4.0, 4.0)
    servo_deg_s: float = 300.0
    linear_speed: float = 0.18
    yaw_speed: float = 0.30
    episode_seconds: float = 12.0
    ramp_seconds: float = 0.8
    residual_filter: float = 0.20
    heading_kp: float = 2.0
    heading_kd: float = 0.20

    def to_dict(self):
        return asdict(self)


def load_model(xml=None):
    xml = MODEL_PATH.read_text() if xml is None else xml
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nu != 18 or model.nq != 25:
        raise ValueError("Expected the 18-servo hexapod model")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError("V3 requires the validated implicitfast integrator")
    if not np.isclose(model.opt.timestep, 0.002):
        raise ValueError("V3 requires the 2 ms physics timestep")
    return model


def model_hash(xml):
    return hashlib.sha256(xml.encode()).hexdigest()


def sample_command(rng):
    # Equal representation of translations; no forward-only learning stage.
    mode = rng.integers(0, 11)
    if mode == 10:
        return np.zeros(3)
    if mode >= 8:
        return np.array([0., 0., 1. if mode == 8 else -1.])
    angle = mode * np.pi / 4
    speed = rng.uniform(0.5, 1.0)
    return np.array([speed*np.cos(angle), speed*np.sin(angle), 0.])


class LocomotionEnv:
    def __init__(self, config=None, model=None):
        self.config = config or Config()
        self.model = model if model is not None else load_model()
        self.data = mujoco.MjData(self.model)
        self.links_cm = tuple(float(self.model.numeric(name + "_length_mm").data[0]) / 10
                              for name in ("coxa", "femur", "tibia"))
        self.stand = directional_reference(0., np.zeros(3), self.links_cm)
        self.low = self.model.actuator_ctrlrange[:, 0]
        self.high = self.model.actuator_ctrlrange[:, 1]
        self.scale = np.tile(np.deg2rad(self.config.residual_deg), 6)
        self.frame_skip = round(CONTROL_DT / self.model.opt.timestep)
        self.max_steps = round(self.config.episode_seconds / CONTROL_DT)

    def reset(self, command=(1, 0, 0), seed=0, yaw=0.0):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, 0.110]
        self.data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        self.data.qpos[7:] = self.stand + rng.uniform(-0.002, 0.002, 18)
        self.commanded_q = np.clip(self.stand.copy(), self.low, self.high)
        self.data.ctrl[:] = self.commanded_q
        mujoco.mj_forward(self.model, self.data)
        # Let gravity load the legs before the first step; identical in viewer.
        mujoco.mj_step(self.model, self.data, nstep=750)
        self.goal_command = clean_command(command)
        self.command = self.goal_command.copy()
        self.residual = np.zeros(18)
        self.phase = 0.0
        self.steps = 0
        self.heading = float(rpy(self.data.qpos[3:7])[2])
        self.velocity = np.zeros(2)
        self.yaw_rate = 0.0
        self.start_position = self.data.qpos[:3].copy()
        return self.observation()

    def set_command(self, command):
        self.goal_command = clean_command(command)

    def reference(self):
        orientation = rpy(self.data.qpos[3:7])
        c = self.command.copy()
        if np.linalg.norm(c) > 1e-5:
            # Relative heading hold from IMU orientation/gyro; no world-position
            # or true translational velocity is used by the controller.
            heading_error = wrap_angle(orientation[2] - self.heading)
            c[2] = np.clip(c[2] - self.config.heading_kp*heading_error
                           - self.config.heading_kd*self.data.qvel[5], -1, 1)
        return np.clip(directional_reference(self.phase, c, self.links_cm), self.low, self.high)

    def observation(self):
        orientation = rpy(self.data.qpos[3:7])
        # Fixed physical scales avoid mutable JIT-captured running statistics.
        # 3 command + 2 tilt + 3 gyro + 1 heading + 2 phase + 18 ref +
        # 18 previous servo commands + 18 filtered residuals = 65.
        obs = np.concatenate([
            self.command, orientation[:2]/0.5, self.data.qvel[3:6]/3.0,
            [wrap_angle(orientation[2]-self.heading)/np.pi],
            [np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase)],
            self.reference()/np.pi, self.commanded_q/np.pi, self.residual,
        ])
        return np.clip(obs, -5, 5).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (18,) or not np.all(np.isfinite(action)):
            raise ValueError("Policy produced invalid actions")
        self.command += np.clip(self.goal_command-self.command, -0.04, 0.04)
        self.heading = float(wrap_angle(self.heading +
                             self.command[2]*self.config.yaw_speed*CONTROL_DT))
        previous_residual = self.residual.copy()
        self.residual += self.config.residual_filter*(np.clip(action, -1, 1)-self.residual)
        q_ref = self.reference()
        ramp = v5.smoother(self.steps*CONTROL_DT/self.config.ramp_seconds)
        target = self.stand + ramp*(q_ref + self.scale*self.residual - self.stand)
        target = np.clip(target, self.low, self.high)
        limit = np.deg2rad(self.config.servo_deg_s)*CONTROL_DT
        self.commanded_q += np.clip(target-self.commanded_q, -limit, limit)
        old_pos = self.data.qpos[:3].copy()
        old_yaw = rpy(self.data.qpos[3:7])[2]
        self.data.ctrl[:] = self.commanded_q
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.steps += 1
        self.phase = (self.phase + CONTROL_DT/v5.CYCLE_TIME) % 1.0

        orientation = rpy(self.data.qpos[3:7])
        yaw = orientation[2]
        world_velocity = (self.data.qpos[:2]-old_pos[:2])/CONTROL_DT
        cy, sy = np.cos(yaw), np.sin(yaw)
        measured = np.array([cy*world_velocity[0]+sy*world_velocity[1],
                            -sy*world_velocity[0]+cy*world_velocity[1]])
        self.velocity += 0.2*(measured-self.velocity)
        yaw_delta = float(wrap_angle(yaw-old_yaw))
        self.yaw_rate += 0.2*(yaw_delta/CONTROL_DT-self.yaw_rate)
        finite = np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
        terminated = (not finite or self.data.qpos[2] < 0.05 or
                      np.max(np.abs(orientation[:2])) > np.deg2rad(45))
        truncated = self.steps >= self.max_steps and not terminated
        velocity_error = self.velocity-self.command[:2]*self.config.linear_speed
        yaw_error = self.yaw_rate-self.command[2]*self.config.yaw_speed
        reward = CONTROL_DT*(
            3.0*np.exp(-np.sum(velocity_error**2)/0.12**2)
            + 1.0*np.exp(-yaw_error**2/0.35**2) + 0.3
            - 2.0*np.sum(orientation[:2]**2)
            - 0.20*np.mean(self.residual**2)
            - 0.50*np.mean((self.residual-previous_residual)**2)
            - 0.30*wrap_angle(yaw-self.heading)**2
        )
        if terminated:
            reward = -5.0
        info = dict(position=self.data.qpos[:3].copy(), rpy=orientation,
                    yaw_delta=yaw_delta, velocity=self.velocity.copy(),
                    residual_deg=np.rad2deg(self.scale*self.residual),
                    fallen=bool(terminated))
        obs = self.observation() if finite else np.zeros(OBS_SIZE, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info
