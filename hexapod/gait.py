"""Portable geometric reference shared with PPO training and the robot controller."""

import numpy as np

import reference_gait as v5

HIP_XY_CM = np.array(
    [
        [11.04, 5.84],
        [0, 9.08],
        [-11.04, 5.84],
        [11.04, -5.84],
        [0, -9.08],
        [-11.04, -5.84],
    ]
)
LEG_YAW = np.deg2rad(v5.LEG_YAW_DEG)
COS_YAW, SIN_YAW = np.cos(LEG_YAW), np.sin(LEG_YAW)
FOOT_XY_CM = HIP_XY_CM + v5.HOME_X_CM * np.stack([COS_YAW, SIN_YAW], axis=-1)
TANGENTS = np.stack([-FOOT_XY_CM[:, 1], FOOT_XY_CM[:, 0]], axis=-1) / 24.0
STAND_Z = np.array([v5.RAISED_Z[i] for i in range(6)])
STAND = v5.raised_stand_targets().astype(np.float64)
TABLE_SIZE = 1024
# Preserve the accepted leg-specific timing, load/push/coil/whip/catch shape.
TRAJECTORIES = np.array(
    [
        [v5.leg_trajectory(leg, (p + v5.PHASE_OFFSET[leg]) % 1.0) for leg in range(6)]
        for p in np.arange(TABLE_SIZE) / TABLE_SIZE
    ]
)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def rpy(quat):
    w, x, y, z = quat
    return np.array(
        [
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ]
    )


def clean_command(command):
    c = np.asarray(command, dtype=np.float64).copy()
    if c.shape != (3,) or not np.all(np.isfinite(c)):
        raise ValueError("Command must be three finite numbers: vx vy yaw")
    c = np.clip(c, -1.0, 1.0)
    c[:2] /= max(1.0, float(np.linalg.norm(c[:2])))
    return c


def directional_reference(phase, command, links_cm=(v5.L0, v5.L1, v5.L2)):
    """Rotate V5 foot strokes in BODY XY; yaw adds tangential foot strokes.

    Back/side/diagonal commands keep the same asymmetric six-leg choreography
    instead of losing their reference and becoming unrestricted joint control.
    """
    l0, l1, l2 = links_cm
    index = (float(phase) % 1.0) * TABLE_SIZE
    i = int(index)
    f, z, radial = (
        (1 - (index - i)) * TRAJECTORIES[i]
        + (index - i) * TRAJECTORIES[(i + 1) % TABLE_SIZE]
    ).T
    translation = np.asarray(command[:2], dtype=np.float64).copy()
    # V5 is deliberately asymmetric. Its uncalibrated left strokes undertravel
    # relative to forward strokes. A continuous feed-forward calibration keeps
    # the requested body direction (it uses no measured body velocity).
    direction_x = translation[0] / max(float(np.linalg.norm(translation)), 1e-8)
    if translation[1] > 0:
        lateral = translation[1]
        translation[0] -= lateral * (0.08 + 0.20 * (1 - abs(direction_x)))
        translation[1] *= 1.50 + 0.35 * direction_x
    velocity = translation[None, :] + command[2] * TANGENTS
    amplitude = np.minimum(1.0, np.linalg.norm(velocity, axis=1))
    shift = f[:, None] * velocity
    x = (
        v5.HOME_X_CM
        + radial * amplitude
        + shift[:, 0] * COS_YAW
        + shift[:, 1] * SIN_YAW
    )
    y = -shift[:, 0] * SIN_YAW + shift[:, 1] * COS_YAW
    z = STAND_Z + amplitude * (z - STAND_Z)

    gamma = np.arctan2(y, x)
    horizontal = np.hypot(x - l0 * np.cos(gamma), y - l0 * np.sin(gamma))
    distance = np.hypot(horizontal, z)
    clipped = np.clip(
        distance, abs(l2 - l1) + v5.IK_MARGIN_CM, l1 + l2 - v5.IK_MARGIN_CM
    )
    ratio = clipped / np.maximum(distance, 1e-9)
    horizontal, z, distance = horizontal * ratio, z * ratio, clipped
    alpha = np.arccos(
        np.clip((l1**2 + distance**2 - l2**2) / (2 * l1 * distance), -1, 1)
    ) + np.arctan2(z, horizontal)
    beta = np.arccos(np.clip((l1**2 + l2**2 - distance**2) / (2 * l1 * l2), -1, 1))
    return np.stack(
        [gamma * np.asarray(v5.Q1_SIGN), alpha, beta - np.pi], axis=-1
    ).reshape(18)
