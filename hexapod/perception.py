"""Local passage estimates using camera intrinsics and measured attitude."""

import numpy as np
from .ports import Scan


class Pinhole:
    def __init__(self, width=61, height=41, fovy=85.0):
        self.width, self.height = width, height
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        v, u = np.indices((height, width))
        self.rays = np.stack(
            (
                np.ones_like(u),
                (width / 2 - 0.5 - u) / focal,
                (height / 2 - 0.5 - v) / focal,
            ),
            axis=-1,
        ).reshape(-1, 3)
        self.norms = np.linalg.norm(self.rays, axis=1)


def level_rotation(roll, pitch):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]])


class DepthPerception:
    def __init__(self, camera):
        self.camera = camera
        self.floor = -0.15

    def read(self, depth, rpy, time):
        depth = np.asarray(depth)
        if depth.shape != (self.camera.height, self.camera.width):
            raise ValueError("Depth image and calibration disagree")
        d = depth.ravel()
        ok = np.isfinite(d) & (d > 0.012) & (d < 1.4)
        if ok.sum() < 12:
            return Scan(time, valid=False)
        pts = (self.camera.rays[ok] * d[ok, None]) @ level_rotation(*rpy[:2]).T
        x, y, z = pts.T
        patch = (x > 0.025) & (x < 0.85) & (np.abs(y) < 0.37) & (z < -0.04)
        if patch.sum() >= 8:
            self.floor = float(np.percentile(z[patch], 8))
        h = z - self.floor
        near = (x > -0.015) & (x < 0.47)
        center = near & (np.abs(y) < 0.065)
        mid = center & (h > 0.044) & (h < 0.095)
        upper = center & (h > 0.102) & (h < 0.20)
        wall = None
        if mid.sum() >= 3 and upper.sum() >= 3:
            front = float(np.percentile(x[mid], 20))
            if (np.abs(x[upper] - front) < 0.045).sum() >= 3:
                wall = front
        roof = None
        overhead = near & (np.abs(y) < 0.145) & (h > 0.072) & (h < 0.235)
        if overhead.sum() >= 9:
            ys = y[overhead]
            if (ys > 0.045).any() and (ys < -0.045).any() and (abs(ys) < 0.045).any():
                roof = float(np.percentile(h[overhead], 8))
        sides = (
            near
            & (h > 0.042)
            & (h < (min(0.11, roof - 0.035) if roof is not None else 0.24))
        )

        def margin(sign):
            values = sign * y[sides]
            values = values[(values > 0.085) & (values < 0.39)]
            return float(np.percentile(values, 8)) if values.size >= 5 else None

        ground = near & (x > 0.06) & (x < 0.43) & (h > 0.006) & (h < 0.053)
        heights = []
        for lo, hi in ((0.075, 0.235), (-0.075, 0.075), (-0.235, -0.075)):
            part = ground & (y > lo) & (y < hi)
            heights.append(
                float(np.percentile(h[part], 80)) if part.sum() >= 3 else 0.0
            )
        return Scan(time, True, roof, margin(1), margin(-1), wall, tuple(heights))
