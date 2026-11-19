"""Range sensing, occupancy memory and frontier-based exploration.

The high-level explorer sees only simulated range measurements and a supplied
pose estimate. It never receives the obstacle list. The MuJoCo demo currently
supplies ground-truth pose; hardware requires a real localization estimator.
"""
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Rectangle:
    x: float
    y: float
    half_x: float
    half_y: float
    half_z: float = 0.16


SCAN_ANGLES = np.deg2rad(np.linspace(-110.0, 110.0, 23))
SENSOR_RANGE = 1.15


def range_scan(xy, yaw, obstacles, angles=SCAN_ANGLES, max_range=SENSOR_RANGE):
    """Idealized horizontal range sensor: intersections with static AABB boxes.

    The scene geometry is used ONLY here, as a sensor simulation. The explorer
    receives distances and hit flags, not obstacle positions.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError('Expected a finite 2D sensor position')
    angles = np.asarray(angles, dtype=float)
    if max_range <= 0:
        raise ValueError('max_range must be positive')
    distances = np.full(len(angles), float(max_range))
    hits = np.zeros(len(angles), dtype=bool)
    for i, relative_angle in enumerate(angles):
        theta = yaw + relative_angle
        direction = np.array([math.cos(theta), math.sin(theta)])
        for box in obstacles:
            minimum = np.array([box.x - box.half_x, box.y - box.half_y])
            maximum = np.array([box.x + box.half_x, box.y + box.half_y])
            near, far = -np.inf, np.inf
            for axis in range(2):
                if abs(direction[axis]) < 1e-12:
                    if xy[axis] < minimum[axis] or xy[axis] > maximum[axis]:
                        near, far = np.inf, -np.inf
                        break
                else:
                    t1 = (minimum[axis] - xy[axis]) / direction[axis]
                    t2 = (maximum[axis] - xy[axis]) / direction[axis]
                    near = max(near, min(t1, t2))
                    far = min(far, max(t1, t2))
            distance = max(near, 0.0)
            if far >= distance and distance < distances[i]:
                distances[i], hits[i] = distance, True
    return distances, hits


class OccupancyMemory:
    UNKNOWN, FREE, OCCUPIED = -1, 0, 1

    def __init__(self, size=3.0, resolution=0.075):
        if size <= 0 or resolution <= 0:
            raise ValueError('size and resolution must be positive')
        self.resolution = float(resolution)
        self.side = int(math.ceil(size / resolution))
        self.minimum = -self.side * self.resolution / 2.0
        self.cells = np.full((self.side, self.side), self.UNKNOWN, dtype=np.int8)
        self.visits = np.zeros_like(self.cells, dtype=np.int32)
        self.survey_count = 0

    def cell(self, xy):
        x, y = np.floor((np.asarray(xy) - self.minimum) / self.resolution).astype(int)
        return (int(y), int(x)) if 0 <= x < self.side and 0 <= y < self.side else None

    def center(self, cell):
        row, col = cell
        return np.array([col + 0.5, row + 0.5]) * self.resolution + self.minimum

    def observe(self, xy, yaw, distances, hits, angles=SCAN_ANGLES):
        if not (len(distances) == len(hits) == len(angles)):
            raise ValueError('Scan arrays must have equal lengths')
        origin = np.asarray(xy, dtype=float)
        for angle, raw_distance, hit in zip(angles, distances, hits):
            distance = float(raw_distance)
            if not np.isfinite(distance) or distance < 0:
                raise ValueError('Invalid range measurement')
            ray = np.array([math.cos(yaw + angle), math.sin(yaw + angle)])
            # Stop just short of the measured surface to avoid overwriting it.
            free_length = max(0.0, distance - (0.5 * self.resolution if hit else 0.0))
            for d in np.arange(0.0, free_length + 1e-9, self.resolution * 0.4):
                idx = self.cell(origin + d * ray)
                if idx is not None and self.cells[idx] != self.OCCUPIED:
                    self.cells[idx] = self.FREE
            if hit:
                # Slightly inside the box, rather than on a rounding boundary.
                idx = self.cell(origin + (distance + 0.012) * ray)
                if idx is not None:
                    self.cells[idx] = self.OCCUPIED
        here = self.cell(origin)
        if here is not None:
            self.cells[here] = self.FREE
            self.visits[here] += 1
        self.survey_count += 1

    def mark_bump(self, xy, yaw, distance=0.20):
        """Remember an unexpected contact/stall as a blocked location."""
        idx = self.cell(np.asarray(xy) + distance * np.array([math.cos(yaw), math.sin(yaw)]))
        if idx is not None:
            self.cells[idx] = self.OCCUPIED

    @property
    def coverage(self):
        return float(np.mean(self.cells != self.UNKNOWN))

    def save(self, filename, scene_seed):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=1, scene_seed=int(scene_seed), side=self.side,
                       minimum=self.minimum, resolution=self.resolution,
                       cells=self.cells.tolist(), visits=self.visits.tolist(),
                       survey_count=self.survey_count)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
        temporary.replace(path)

    @classmethod
    def load(cls, filename, scene_seed):
        payload = json.loads(Path(filename).read_text())
        if payload.get('version') != 1 or payload.get('scene_seed') != int(scene_seed):
            raise ValueError('Map format or obstacle scene does not match')
        result = cls(size=payload['side'] * payload['resolution'],
                     resolution=payload['resolution'])
        if result.side != payload['side'] or not np.isclose(result.minimum, payload['minimum']):
            raise ValueError('Map geometry mismatch')
        cells = np.asarray(payload['cells'], dtype=np.int8)
        visits = np.asarray(payload['visits'], dtype=np.int32)
        if cells.shape != result.cells.shape or visits.shape != result.visits.shape:
            raise ValueError('Invalid map dimensions')
        if not np.isin(cells, [-1, 0, 1]).all() or (visits < 0).any():
            raise ValueError('Invalid map data')
        result.cells, result.visits = cells, visits
        result.survey_count = int(payload['survey_count'])
        return result


def inflated_obstacles(cells, radius_cells):
    occupied = cells == OccupancyMemory.OCCUPIED
    inflated = occupied.copy()
    side = len(cells)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx*dx + dy*dy > radius_cells*radius_cells:
                continue
            src_y0, src_y1 = max(0, -dy), min(side, side - dy)
            src_x0, src_x1 = max(0, -dx), min(side, side - dx)
            inflated[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] |= (
                occupied[src_y0:src_y1, src_x0:src_x1])
    return inflated


class FrontierExplorer:
    """Select reachable, unvisited boundaries of known free space.

    No pre-programmed sequence of obstacle locations or routes is used. Each
    new scan changes the map and therefore the selected trajectory.
    """

    def __init__(self, memory, clearance=0.17):
        self.memory = memory
        self.clearance = float(clearance)
        self.goal = None
        self.goals_reached = 0

    def _plan(self, start):
        mem = self.memory
        blocked = inflated_obstacles(mem.cells, math.ceil(self.clearance / mem.resolution))
        traversable = (mem.cells == mem.FREE) & ~blocked
        if start is None:
            return None, None
        if not traversable[start]:
            # A hit very close to the center may make all neighboring cells
            # inflated; keep scanning instead of planning through obstacles.
            return None, None
        dist = {start: 0.0}
        parent = {}
        queue = [(0.0, start)]
        while queue:
            cost, here = heapq.heappop(queue)
            if cost > dist[here] + 1e-8:
                continue
            y, x = here
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                target = (y + dy, x + dx)
                ty, tx = target
                if not (0 <= ty < mem.side and 0 <= tx < mem.side) or not traversable[target]:
                    continue
                if dy and dx and (not traversable[y, tx] or not traversable[ty, x]):
                    continue
                new_cost = cost + (1.41421356237 if dy and dx else 1.0)
                if new_cost + 1e-8 < dist.get(target, float('inf')):
                    dist[target], parent[target] = new_cost, here
                    heapq.heappush(queue, (new_cost, target))
        return dist, parent

    def _frontiers(self, reachable):
        cells = self.memory.cells
        for cell, distance in reachable.items():
            y, x = cell
            unknown = 0
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < len(cells) and 0 <= xx < len(cells) and cells[yy, xx] == -1:
                        unknown += 1
            if unknown:
                yield cell, distance, unknown

    def command(self, xy, yaw):
        mem = self.memory
        start = mem.cell(xy)
        distance, parent = self._plan(start)
        if not distance:
            return np.array([0., 0., 0.7]), 'scan', None
        candidates = list(self._frontiers(distance))
        if not candidates:
            return np.zeros(3), 'complete', None
        available = {item[0] for item in candidates}
        if self.goal not in available:
            # Prioritize new observations, but avoid chasing a distant frontier
            # while ignoring nearby unexplored space.
            self.goal = max(candidates, key=lambda item: (
                1.5*item[2] - 0.7*item[1] - 0.5*math.sqrt(mem.visits[item[0]])))[0]
            self.goals_reached += 1
        path = [self.goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        waypoint = path[min(3, len(path) - 1)]
        point = mem.center(waypoint)
        heading = math.atan2(point[1]-xy[1], point[0]-xy[0])
        error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
        if abs(error) > 0.22:
            return np.array([0., 0., float(np.sign(error)*0.85)]), 'turn', point
        return np.array([0.85, 0., 0.]), 'move', point
