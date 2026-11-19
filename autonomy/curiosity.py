"""Range-only object discovery, curiosity and persistent episodic experience.

This is an intentionally small, interpretable cognitive layer, NOT camera vision
or a learned object-recognition network. It operates on the same 23 range beams
as the map builder and never receives the MuJoCo obstacle list or geom IDs.
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np

from .explorer import SCAN_ANGLES


@dataclass
class ObjectTrack:
    identifier: int
    position: list
    sightings: int = 1
    inspected: int = 0
    novelty: float = 1.0
    risk: float = 0.0
    mean_reward: float = 0.0
    outcomes: int = 0
    last_seen: int = 0


class ObjectMemory:
    """Persistent object hypotheses from clusters of *measured* range endpoints."""

    VERSION = 1

    def __init__(self, scene_seed):
        self.scene_seed = int(scene_seed)
        self.tracks = []
        self.next_identifier = 0
        self.surveys = 0
        self.inspections = 0
        self.contact_events = 0
        self.total_intrinsic_reward = 0.0

    @staticmethod
    def detections(xy, yaw, distances, hits, angles=SCAN_ANGLES):
        xy = np.asarray(xy, dtype=float)
        distances = np.asarray(distances, dtype=float)
        hits = np.asarray(hits, dtype=bool)
        angles = np.asarray(angles, dtype=float)
        if (xy.shape != (2,) or distances.shape != hits.shape or
                hits.shape != angles.shape or not np.isfinite(xy).all() or
                not np.isfinite(yaw) or not np.isfinite(distances).all() or
                (distances < 0).any()):
            raise ValueError('Invalid planar range observations')
        endpoints = xy + distances[:, None] * np.stack([
            np.cos(yaw + angles), np.sin(yaw + angles)], axis=1)
        # The room's enclosing wall is not a manipulable object. This is a
        # known workspace bound, not privileged obstacle or object identity.
        valid = hits & (np.abs(endpoints[:, 0]) < 1.12) & (np.abs(endpoints[:, 1]) < 1.12)
        groups, current = [], []
        for i, keep in enumerate(valid):
            if keep:
                if current and (i != current[-1] + 1 or
                                np.linalg.norm(endpoints[i]-endpoints[current[-1]]) > 0.23):
                    groups.append(current)
                    current = []
                current.append(i)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        # Isolated reflections cannot be reliably separated from sensor noise.
        # They may be observed later when the robot approaches an object.
        return [np.mean(endpoints[g], axis=0) for g in groups if len(g) >= 2]

    def observe(self, xy, yaw, distances, hits):
        self.surveys += 1
        seen_this_survey = set()
        for point in self.detections(xy, yaw, distances, hits):
            options = [(float(np.linalg.norm(np.asarray(t.position)-point)), t)
                       for t in self.tracks if t.identifier not in seen_this_survey]
            options.sort(key=lambda item: item[0])
            if options and options[0][0] < 0.29:
                track = options[0][1]
                track.position = (0.8*np.asarray(track.position) + 0.2*point).tolist()
                track.sightings += 1
            else:
                track = ObjectTrack(identifier=self.next_identifier, position=point.tolist())
                self.next_identifier += 1
                self.tracks.append(track)
            track.last_seen = self.surveys
            seen_this_survey.add(track.identifier)
        return seen_this_survey

    def get(self, identifier):
        return next((track for track in self.tracks if track.identifier == identifier), None)

    def learn(self, identifier, reached, coverage_gain, contact=False):
        track = self.get(identifier)
        if track is None:
            raise ValueError('Unknown object hypothesis')
        if contact:
            reward = -3.0
            track.risk = min(1.0, track.risk + 0.8)
            self.contact_events += 1
        elif reached:
            # A successful safe inspection gets intrinsic curiosity reward;
            # extra information gained along the approach gives a small bonus.
            reward = 1.0 + min(0.5, max(0.0, float(coverage_gain))*3.0)
            track.inspected += 1
            self.inspections += 1
        else:
            reward = -0.3
        track.mean_reward += (reward-track.mean_reward)/(track.outcomes + 1)
        track.outcomes += 1
        # Habituation: inspected or unproductive objects become less attractive.
        track.novelty *= 0.20 if reached else 0.45
        self.total_intrinsic_reward += reward
        return reward

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(version=self.VERSION, scene_seed=self.scene_seed,
                    next_identifier=self.next_identifier, surveys=self.surveys,
                    inspections=self.inspections, contact_events=self.contact_events,
                    total_intrinsic_reward=self.total_intrinsic_reward,
                    tracks=[asdict(track) for track in self.tracks])
        tmp = path.with_name('.'+path.name+'.tmp')
        tmp.write_text(json.dumps(data, allow_nan=False, indent=2)+'\n')
        tmp.replace(path)

    @classmethod
    def load(cls, path, scene_seed):
        data = json.loads(Path(path).read_text())
        if data.get('version') != cls.VERSION or data.get('scene_seed') != int(scene_seed):
            raise ValueError('Object memory version / scene does not match')
        memory = cls(scene_seed)
        memory.next_identifier = int(data['next_identifier'])
        memory.surveys = int(data['surveys'])
        memory.inspections = int(data['inspections'])
        memory.contact_events = int(data['contact_events'])
        memory.total_intrinsic_reward = float(data['total_intrinsic_reward'])
        memory.tracks = [ObjectTrack(**item) for item in data['tracks']]
        ids = [track.identifier for track in memory.tracks]
        if (len(set(ids)) != len(ids) or min(ids, default=0) < 0 or
                memory.next_identifier <= max(ids, default=-1) or
                any(len(t.position) != 2 or not np.isfinite(t.position).all() or
                    not 0 <= t.novelty <= 1 or not 0 <= t.risk <= 1 or
                    t.sightings <= 0 or t.inspected < 0 or t.outcomes < 0
                    for t in memory.tracks)):
            raise ValueError('Corrupt object memory')
        return memory


class CuriosityController:
    """Safely approach a newly discovered object, observe it, then resume mapping.

    The lower-level locomotion V5/PPO and collision recovery remain unchanged.
    Novelty and observed outcomes influence which object is approached next.
    """

    def __init__(self, explorer, objects, minimum_standoff=0.47):
        if minimum_standoff < explorer.clearance:
            raise ValueError('Inspection standoff must exceed collision clearance')
        self.explorer = explorer
        self.objects = objects
        self.minimum_standoff = float(minimum_standoff)
        self.active_id = None
        self.vantage = None
        self.start_coverage = 0.0
        self.start_survey = 0
        self.attempts = 0
        self.commands_changed = 0

    def abandon(self, contact=False):
        if self.active_id is not None:
            self.objects.learn(self.active_id, False,
                               self.explorer.memory.coverage-self.start_coverage,
                               contact=contact)
        self.active_id = None
        self.vantage = None

    def _choose_target(self, xy, distances, reachable):
        memory = self.explorer.memory
        options = []
        for obj in self.objects.tracks:
            if (obj.sightings < 2 or obj.inspected or obj.novelty < 0.40 or
                    obj.risk >= 0.5 or self.objects.surveys-obj.last_seen > 55):
                continue
            center = np.asarray(obj.position)
            choices = []
            for cell, path_cost in reachable.items():
                world = memory.center(cell)
                sep = float(np.linalg.norm(world-center))
                if self.minimum_standoff <= sep <= self.minimum_standoff+0.22:
                    choices.append((path_cost, cell))
            if not choices:
                continue
            path_cost, vantage = min(choices)
            # Habituation and experience change which candidate is selected.
            value = (2.0*obj.novelty - 1.5*obj.risk +
                     0.15*np.clip(obj.mean_reward, -3, 2) -
                     0.28*path_cost*memory.resolution)
            if value > 0.50:
                options.append((value, -path_cost, -obj.identifier, obj.identifier, vantage))
        if not options:
            return None
        best = max(options)
        return best[3], best[4]

    def command(self, xy, yaw, distances, hits, default):
        """Return (command, decision, waypoint). No simulator geometry access."""
        self.objects.observe(xy, yaw, distances, hits)
        memory = self.explorer.memory
        xy = np.asarray(xy, dtype=float)
        distance, parent = self.explorer._plan(memory.cell(xy))
        if not distance:
            self.abandon()
            return default
        if self.active_id is not None and (
                self.vantage not in distance or
                self.objects.surveys - self.start_survey > 75):
            self.abandon()
        if self.active_id is None and (memory.coverage < 0.45 or
                                       self.objects.inspections >= 2):
            return default
        if self.active_id is None:
            chosen = self._choose_target(xy, distances, distance)
            if chosen is not None:
                self.active_id, self.vantage = chosen
                self.start_coverage = memory.coverage
                self.start_survey = self.objects.surveys
                self.attempts += 1
        if self.active_id is None:
            return default
        obj = self.objects.get(self.active_id)
        if obj is None:
            self.abandon()
            return default
        target = np.asarray(obj.position)
        vantage = memory.center(self.vantage)
        if np.linalg.norm(xy-vantage) <= 0.11:
            heading = math.atan2(target[1]-xy[1], target[0]-xy[0])
            error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
            if abs(error) < 0.30:
                self.objects.learn(self.active_id, True, memory.coverage-self.start_coverage)
                self.active_id = None
                self.vantage = None
                self.commands_changed += 1
                return np.zeros(3), 'inspect_object', target
            self.commands_changed += 1
            return np.array([0., 0., math.copysign(0.65, error)]), 'orient_object', target
        path = [self.vantage]
        while path[-1] != memory.cell(xy):
            path.append(parent[path[-1]])
        path.reverse()
        point = memory.center(path[min(3, len(path)-1)])
        heading = math.atan2(point[1]-xy[1], point[0]-xy[0])
        error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
        self.commands_changed += 1
        if abs(error) > 0.22:
            return np.array([0., 0., math.copysign(0.85, error)]), 'approach_turn', point
        return np.array([0.75, 0., 0.]), 'approach_object', point
