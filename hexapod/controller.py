"""Portable 50 Hz control. Robot measurements enter only through the sensor port."""

from pathlib import Path
import numpy as np
from .gait import directional_reference, wrap_angle
from .geometry import JOINT_LOW, JOINT_HIGH, LINKS, SERVO_SPEED, tripod
from .ports import Targets

WEIGHTS = Path(__file__).resolve().parent / "weights" / "gait_ppo.npz"


class Actor:
    def __init__(self, path=WEIGHTS):
        with np.load(path, allow_pickle=False) as data:
            self.w1, self.b1, self.w2, self.b2, self.wm, self.bm = (
                data[k] for k in ("w1", "b1", "w2", "b2", "wm", "bm")
            )

    def __call__(self, observation):
        hidden = np.tanh(observation @ self.w1 + self.b1)
        return np.tanh(np.tanh(hidden @ self.w2 + self.b2) @ self.wm + self.bm)


class Controller:
    def __init__(self, adaptive=True, actor=None):
        self.adaptive = adaptive
        self.actor = actor if actor is not None else Actor()
        self.stand = directional_reference(0.0, np.zeros(3), LINKS * 100)
        self.angles = self.stand.copy()
        self.residual = np.zeros(18)
        self.command = np.zeros(3)
        self.phase = self.walk_phase = self.elapsed = 0.0
        self.heading = None
        self.hold = {"low": 0.0, "narrow": 0.0, "rough": 0.0}
        self.blend = self.compact = self.crouch = 0.0
        self.ground = np.zeros(3)
        self.was_blocked = False
        self.half_width = 0.215
        self.center_trim = 0.0

    def step(self, command, sensors, dt):
        if not 0 < dt <= 0.1:
            raise ValueError("Invalid control interval")
        desired = command.vector()
        scan = sensors.scan
        attitude = np.array(sensors.rpy)
        gyro = np.array(sensors.gyro)
        fresh = scan.valid and -0.001 <= sensors.time - scan.time <= 0.35
        imu_ok = np.isfinite(attitude).all() and np.isfinite(gyro).all()
        hazards = self.adaptive and desired[0] > 0.03
        blocked = hazards and (
            not fresh
            or (scan.wall_m is not None and scan.wall_m < 0.40)
            or (scan.roof_m is not None and scan.roof_m < 0.132)
            or (
                scan.left_m is not None
                and scan.right_m is not None
                and scan.left_m + scan.right_m < 0.387
            )
        )
        if not imu_ok or blocked:
            self.was_blocked = True
            self.command[:] = 0.0
            return Targets(
                self.angles.copy(), "blocked" if fresh and imu_ok else "sensor_stop"
            )
        if self.was_blocked or self.heading is None:
            self.heading = float(attitude[2])
            self.was_blocked = False
        for key in self.hold:
            self.hold[key] = max(0.0, self.hold[key] - dt * np.linalg.norm(desired[:2]))
        if hazards and fresh:
            if scan.roof_m is not None and scan.roof_m < 0.195:
                self.hold["low"] = 4.0
            if (
                scan.left_m is not None
                and scan.right_m is not None
                and scan.left_m + scan.right_m < 0.50
            ):
                self.hold["narrow"] = 7.2
                if scan.roof_m is None and self.hold["low"] < 1.05:
                    self.hold["low"] = 0.0
            if (
                scan.roof_m is None
                and not (scan.left_m and scan.right_m)
                and max(scan.ground_m) > 0.009
            ):
                self.hold["rough"] = 3.8
                self.ground = np.maximum(self.ground * 0.93, scan.ground_m)
        mode = next(
            (k for k in ("low", "narrow", "rough") if self.hold[k] > 0), "normal"
        )
        if mode == "normal":
            self.ground *= 0.97
        if self.hold["narrow"] > 0 and desired[0] > 0:
            left, right = scan.left_m, scan.right_m
            if left and right:
                self.half_width = (left + right) * 0.5
            if left or right:
                offset = (left - self.half_width) if left else (self.half_width - right)
                self.center_trim += min(1.0, 5 * dt) * (
                    np.clip(9.0 * offset, -0.24, 0.24) - self.center_trim
                )
            else:
                # Once both walls leave the lens, return to the requested heading.
                self.center_trim *= np.exp(-2.0 * dt)
            desired[1] += self.center_trim
        else:
            self.center_trim = 0.0
        self.command += np.clip(desired - self.command, -2 * dt, 2 * dt)
        self.heading = wrap_angle(self.heading + self.command[2] * 0.30 * dt)
        error = wrap_angle(attitude[2] - self.heading)
        corrected = self.command.copy()
        if np.linalg.norm(self.command) > 0.005:
            corrected[2] = np.clip(
                corrected[2] - 2.0 * error - 0.20 * gyro[2], -1.0, 1.0
            )
        corrected[1] -= self.compact * np.clip(5.0 * error * corrected[0], -0.25, 0.25)
        phase = self.phase
        ref = np.clip(
            directional_reference(phase, corrected, LINKS * 100), JOINT_LOW, JOINT_HIGH
        )
        obs = np.r_[
            self.command,
            attitude[:2] / 0.5,
            gyro / 3.0,
            error / np.pi,
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
            ref / np.pi,
            self.angles / np.pi,
            self.residual,
        ]
        action = self.actor(np.clip(obs, -5.0, 5.0).astype(np.float32))
        self.residual += 0.2 * (action - self.residual)
        reference = ref + np.tile(np.deg2rad([3.0, 4.0, 4.0]), 6) * self.residual
        self.blend += np.clip(float(mode != "normal") - self.blend, -dt * 2.5, dt * 2.5)
        self.crouch += np.clip(float(mode == "low") - self.crouch, -dt * 2.5, dt * 2.5)
        self.compact += np.clip(
            float(mode == "narrow") - self.compact, -dt * 2.5, dt * 2.5
        )
        if self.blend:
            radius = 0.113 + 0.027 * self.crouch - 0.021 * self.compact
            height = 0.105 - 0.040 * self.crouch - 0.021 * self.compact
            lift = (
                0.034
                + min(0.042, float(np.max(self.ground)))
                * (1 - self.compact - self.crouch)
                - 0.020 * self.crouch
                - 0.014 * self.compact
            )
            posture = tripod(
                self.walk_phase,
                corrected,
                radius,
                height,
                lift,
                self.ground,
                self.compact,
            )
            reference = (1 - self.blend) * reference + self.blend * posture
        ramp = min(1.0, self.elapsed / 0.8)
        ramp = ramp**3 * (10 - 15 * ramp + 6 * ramp * ramp)
        target = np.clip(
            self.stand + ramp * (reference - self.stand), JOINT_LOW, JOINT_HIGH
        )
        self.angles += np.clip(
            target - self.angles, -SERVO_SPEED * dt, SERVO_SPEED * dt
        )
        self.phase = (phase + dt / 1.22) % 1.0
        self.walk_phase = (self.walk_phase + dt / 0.69) % 1.0
        self.elapsed += dt
        return Targets(self.angles.copy(), mode)
