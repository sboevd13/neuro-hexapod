"""Accepted ASYMMETRIC SPIDER GALLOP V5 TURBO reference gait.

The output is 18 pure geometric joint angles in radians in this order:
FL, ML, BL, FR, MR, BR; each leg is coxa, femur, tibia.
"""
import math
import numpy as np

L0 = 5.1077
L1 = 6.4648
L2 = 11.4615
IK_MARGIN_CM = 0.10
CYCLE_TIME = 1.22

FL, ML, BL, FR, MR, BR = range(6)
LEG_NAMES = ["FL", "ML", "BL", "FR", "MR", "BR"]
LEG_YAW_DEG = [45.0, 90.0, 135.0, -45.0, -90.0, -135.0]
Q1_SIGN = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
HOME_X_CM = 11.5

PHASE_OFFSET = {
    FL: 0.067, MR: 0.299, BL: 0.323,
    FR: 0.595, ML: 0.706, BR: 0.005,
}
ROLE = {
    FL: "front", FR: "front",
    ML: "middle", MR: "middle",
    BL: "back", BR: "back",
}

LEG_PARAMS = {
    FL: dict(front_x=8.0, back_x=-4.8, air_z=-4.0, ground_z=-10.1,
             push_z=-11.15, coil_x=-3.8, radial_ground=1.5, radial_push=3.0,
             radial_coil=-2.3, radial_reach=2.5, stance_end=0.47,
             push_start=0.73, coil_end=0.12, reach_end=0.82),
    MR: dict(front_x=5.8, back_x=-6.0, air_z=-4.3, ground_z=-10.9,
             push_z=-12.0, coil_x=-4.2, radial_ground=1.8, radial_push=3.4,
             radial_coil=-2.0, radial_reach=2.8, stance_end=0.56,
             push_start=0.71, coil_end=0.16, reach_end=0.78),
    BL: dict(front_x=5.4, back_x=-7.2, air_z=-4.6, ground_z=-11.2,
             push_z=-12.35, coil_x=-4.8, radial_ground=1.7, radial_push=3.8,
             radial_coil=-2.5, radial_reach=2.6, stance_end=0.43,
             push_start=0.69, coil_end=0.11, reach_end=0.83),
    FR: dict(front_x=7.4, back_x=-5.1, air_z=-4.2, ground_z=-10.2,
             push_z=-11.25, coil_x=-3.5, radial_ground=1.3, radial_push=2.8,
             radial_coil=-2.6, radial_reach=2.7, stance_end=0.52,
             push_start=0.75, coil_end=0.13, reach_end=0.80),
    ML: dict(front_x=6.1, back_x=-5.6, air_z=-4.1, ground_z=-10.8,
             push_z=-11.9, coil_x=-4.0, radial_ground=2.0, radial_push=3.5,
             radial_coil=-2.2, radial_reach=3.0, stance_end=0.46,
             push_start=0.68, coil_end=0.10, reach_end=0.84),
    BR: dict(front_x=5.7, back_x=-7.0, air_z=-4.5, ground_z=-11.3,
             push_z=-12.35, coil_x=-4.5, radial_ground=1.6, radial_push=4.0,
             radial_coil=-2.4, radial_reach=2.9, stance_end=0.58,
             push_start=0.74, coil_end=0.17, reach_end=0.76),
}

RAISED_Z = {
    FL: -9.7, FR: -9.8,
    ML: -10.4, MR: -10.5,
    BL: -11.0, BR: -11.1,
}


def clamp01(x):
    return max(0.0, min(1.0, x))


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a + (b - a) * t


def inverse_kinematics(x, y, z):
    gamma = math.atan2(y, x)
    x0 = L0 * math.cos(gamma)
    y0 = L0 * math.sin(gamma)
    dx, dy = x - x0, y - y0
    horizontal = math.hypot(dx, dy)
    l3 = math.hypot(horizontal, z)

    safe_min = abs(L2 - L1) + IK_MARGIN_CM
    safe_max = L1 + L2 - IK_MARGIN_CM
    if l3 < safe_min or l3 > safe_max:
        if l3 < 1e-9:
            raise ValueError("IK: L3 is almost zero")
        scale = float(np.clip(l3, safe_min, safe_max)) / l3
        dx, dy, z = dx * scale, dy * scale, z * scale
        horizontal = math.hypot(dx, dy)
        l3 = math.hypot(horizontal, z)

    cos_alpha = (L1 * L1 + l3 * l3 - L2 * L2) / (2.0 * L1 * l3)
    cos_beta = (L1 * L1 + L2 * L2 - l3 * l3) / (2.0 * L1 * L2)
    alpha = math.acos(float(np.clip(cos_alpha, -1.0, 1.0))) + math.atan2(z, horizontal)
    beta = math.acos(float(np.clip(cos_beta, -1.0, 1.0)))
    return gamma, alpha, -(math.pi - beta)


def leg_target(leg, forward_cm, z_cm, radial_cm=0.0):
    yaw = math.radians(LEG_YAW_DEG[leg])
    x = HOME_X_CM + radial_cm + forward_cm * math.cos(yaw)
    y = -forward_cm * math.sin(yaw)
    q0, q1, q2 = inverse_kinematics(x, y, z_cm)
    q0 *= Q1_SIGN[leg]
    return np.array([q0, q1, q2], dtype=np.float32)


def micro_extension(leg, phase):
    amp = 0.18 if ROLE[leg] == "front" else (0.28 if ROLE[leg] == "middle" else 0.22)
    return amp * math.sin(2.0 * math.pi * phase + 0.7 * leg)


def leg_trajectory(leg, phase):
    p = LEG_PARAMS[leg]
    if phase < p["stance_end"]:
        s = phase / p["stance_end"]
        if s < p["push_start"]:
            local = smoother(s / p["push_start"])
            forward = lerp(p["front_x"], p["back_x"] * 0.62, local)
            radial = lerp(p["radial_reach"], p["radial_ground"] * 0.35, local)
            compression = 0.95 * math.sin(math.pi * local)
            z = p["ground_z"] + compression + micro_extension(leg, phase)
            return forward, z, radial

        local = (s - p["push_start"]) / (1.0 - p["push_start"])
        burst = clamp01(local) ** 0.43
        return (
            lerp(p["back_x"] * 0.62, p["back_x"], burst),
            lerp(p["ground_z"] + 0.35, p["push_z"], burst),
            lerp(p["radial_ground"] * 0.35, p["radial_push"], burst),
        )

    s = (phase - p["stance_end"]) / (1.0 - p["stance_end"])
    if s < p["coil_end"]:
        local = smoother(s / p["coil_end"])
        return (
            lerp(p["back_x"], p["coil_x"], local),
            lerp(p["push_z"], p["air_z"], local),
            lerp(p["radial_push"], p["radial_coil"], local),
        )

    if s < p["reach_end"]:
        local = (s - p["coil_end"]) / (p["reach_end"] - p["coil_end"])
        whip = clamp01(local) ** 0.78
        extra_lift = 0.9 * math.sin(math.pi * local)
        return (
            lerp(p["coil_x"], p["front_x"], whip),
            p["air_z"] + extra_lift,
            lerp(p["radial_coil"], p["radial_reach"], whip),
        )

    local = clamp01((s - p["reach_end"]) / (1.0 - p["reach_end"]))
    catch = local ** 0.72
    return p["front_x"], lerp(p["air_z"], p["ground_z"], catch), p["radial_reach"]


def reference_targets(global_phase):
    targets = np.zeros(18, dtype=np.float32)
    global_phase = float(global_phase) % 1.0
    for leg in range(6):
        phase = (global_phase + PHASE_OFFSET[leg]) % 1.0
        forward, z, radial = leg_trajectory(leg, phase)
        targets[leg * 3:leg * 3 + 3] = leg_target(leg, forward, z, radial)
    return targets


def stand_targets(z_by_leg):
    targets = np.zeros(18, dtype=np.float32)
    for leg in range(6):
        targets[leg * 3:leg * 3 + 3] = leg_target(leg, 0.0, z_by_leg[leg], 0.0)
    return targets


def raised_stand_targets():
    return stand_targets(RAISED_Z)


def build_reference_table(samples=2048):
    phases = np.arange(samples, dtype=np.float64) / float(samples)
    return np.stack([reference_targets(p) for p in phases]).astype(np.float32)
