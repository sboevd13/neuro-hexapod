"""Kinematics of the supervisor's 18-servo hexapod."""

import numpy as np

from .gait import COS_YAW, SIN_YAW, HIP_XY_CM

LINKS = np.array([0.051, 0.065, 0.121])
HIP = HIP_XY_CM / 100.0
RADIAL = np.column_stack((COS_YAW, SIN_YAW))
TANGENTIAL = np.column_stack((-SIN_YAW, COS_YAW))
YAW_SIGN = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
FIRMWARE_ORDER = np.array([3, 4, 5, 2, 1, 0])
TRIMS = np.array(
    [[-3, 0, -1], [-2, 0, -3], [-3, -1, -2], [2, 4, 0], [-1, -2, -3], [-1, 0, -3]]
)
SERVO_ZERO = np.array([90.0, 104.0, 157.0]) + TRIMS
SERVO_SIGN = np.column_stack((YAW_SIGN, np.ones((6, 2))))
end_a = -SERVO_ZERO / SERVO_SIGN
end_b = (180.0 - SERVO_ZERO) / SERVO_SIGN
JOINT_LOW = np.deg2rad(
    np.maximum(np.minimum(end_a, end_b), [-45.0, -45.0, -178.0])
).ravel()
JOINT_HIGH = np.deg2rad(
    np.minimum(np.maximum(end_a, end_b), [45.0, 125.0, -5.0])
).ravel()
SERVO_SPEED = np.deg2rad(300.0)


def firmware_degrees(angles):
    angles = np.asarray(angles)
    if (
        angles.shape != (18,)
        or not np.isfinite(angles).all()
        or (angles < JOINT_LOW - 1e-8).any()
        or (angles > JOINT_HIGH + 1e-8).any()
    ):
        raise ValueError("Outside firmware command limits")
    values = SERVO_ZERO + np.rad2deg(angles.reshape(6, 3)) * SERVO_SIGN
    return np.floor(values[FIRMWARE_ORDER] + 0.5).astype(np.uint8)


def solve(feet):
    """Six body-relative foot targets measured from the hip pivots."""
    x = np.sum(feet[:, :2] * RADIAL, axis=1)
    y = np.sum(feet[:, :2] * TANGENTIAL, axis=1)
    z = feet[:, 2]
    a, b, c = LINKS
    planar = np.hypot(x, y) - a
    radius = np.clip(np.hypot(planar, z), abs(b - c) + 0.001, b + c - 0.001)
    hip = np.arctan2(z, planar) + np.arccos(
        np.clip((b * b + radius**2 - c * c) / (2 * b * radius), -1.0, 1.0)
    )
    knee = -np.arccos(np.clip((radius**2 - b * b - c * c) / (2 * b * c), -1.0, 1.0))
    return np.stack([YAW_SIGN * np.arctan2(y, x), hip, knee], axis=1).ravel()


def tripod(phase, command, radius, height, lift, ground, compact=0.0):
    p = (phase + np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])) % 1.0
    duty = 0.63
    swing = np.clip((p - duty) / (1 - duty), 0.0, 1.0)
    sweep = np.where(p < duty, 0.5 - p / duty, -0.5 + swing**2 * (3.0 - 2.0 * swing))
    tuck = np.deg2rad([18.0, 37.0, -18.0, -18.0, -37.0, 18.0]) * compact
    offsets = radius * (
        RADIAL * np.cos(tuck[:, None]) - TANGENTIAL * np.sin(tuck[:, None])
    )
    neutral = HIP + offsets
    tangents = np.column_stack((-neutral[:, 1], neutral[:, 0])) / 0.21
    velocity = command[:2] + command[2] * tangents
    displacement = 0.074 * (1.0 - 0.48 * compact) * sweep[:, None] * velocity
    displacement[[1, 4]] *= 1.0 - 0.64 * compact
    feet = np.empty((6, 3))
    feet[:, :2] = offsets + displacement
    movement = np.minimum(1.0, np.linalg.norm(velocity, axis=1))
    outer = np.array([ground[0]] * 3 + [ground[2]] * 3)
    feet[:, 1] -= YAW_SIGN * np.clip(outer - ground[1], -0.025, 0.025) * movement
    feet[:, 2] = -height + movement * lift * np.sin(np.pi * swing) ** 1.35
    return solve(feet)
