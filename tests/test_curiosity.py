"""Novelty, tracking, persistent outcomes and safe target selection."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autonomy.curiosity import CuriosityController, ObjectMemory, ObjectTrack
from autonomy.explorer import FrontierExplorer, OccupancyMemory, Rectangle, range_scan


WALLS = [Rectangle(1.35, 0, .05, 1.43), Rectangle(-1.35, 0, .05, 1.43),
         Rectangle(0, 1.35, 1.43, .05), Rectangle(0, -1.35, 1.43, .05)]


class CuriosityTests(unittest.TestCase):
    def test_visible_object_discovered_without_geometry_identity(self):
        obj = ObjectMemory(7)
        distances, hits = range_scan((0, 0), 0, WALLS + [Rectangle(.6, .1, .12, .1)])
        obj.observe((0, 0), 0, distances, hits)
        obj.observe((.1, 0), 0, *range_scan((.1, 0), 0, WALLS + [Rectangle(.6, .1, .12, .1)]))
        self.assertEqual(len(obj.tracks), 1)
        self.assertEqual(obj.tracks[0].sightings, 2)
        self.assertAlmostEqual(obj.tracks[0].position[0], .48, delta=.06)

    def test_wall_not_recognized_as_object(self):
        obj = ObjectMemory(7)
        for yaw in (0, .5, 1.2, 2.0):
            obj.observe((.3, 0), yaw, *range_scan((.3, 0), yaw, WALLS))
        self.assertEqual(len(obj.tracks), 0)

    def test_invalid_measurement_rejected(self):
        with self.assertRaises(ValueError):
            ObjectMemory.detections((0, 0), 0, [float('nan')], [True], [0])

    def test_habituation_reward_and_risk(self):
        obj = ObjectMemory(7)
        obj.tracks.append(ObjectTrack(0, [.5, .1]))
        obj.next_identifier = 1
        reward = obj.learn(0, True, .04)
        self.assertGreater(reward, 1)
        self.assertEqual(obj.inspections, 1)
        self.assertEqual(obj.get(0).inspected, 1)
        self.assertAlmostEqual(obj.get(0).novelty, .2)
        obj.learn(0, False, 0, contact=True)
        self.assertGreaterEqual(obj.get(0).risk, .8)
        self.assertEqual(obj.contact_events, 1)
        self.assertLess(obj.total_intrinsic_reward, 0)

    def test_persistent_memory_and_seed_mismatch(self):
        obj = ObjectMemory(19)
        obj.tracks.append(ObjectTrack(0, [.4, .1], sightings=8))
        obj.next_identifier = 1
        obj.learn(0, True, .02)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'objects.json'
            obj.save(path)
            loaded = ObjectMemory.load(path, 19)
            self.assertEqual(loaded.tracks[0].inspected, 1)
            self.assertAlmostEqual(loaded.total_intrinsic_reward, obj.total_intrinsic_reward)
            with self.assertRaises(ValueError):
                ObjectMemory.load(path, 20)

    def test_safe_reachable_approach_and_inspection(self):
        mem = OccupancyMemory()
        # Synthetic map of surveyed free space with a known occupied box.
        mem.cells[:] = mem.FREE
        obj = ObjectMemory(7)
        obj.tracks.append(ObjectTrack(0, [.9, 0], sightings=5, last_seen=1))
        obj.next_identifier = 1
        controller = CuriosityController(FrontierExplorer(mem), obj)
        empty = (np.full(23, 1.15), np.zeros(23, dtype=bool))
        default = (np.zeros(3), 'complete', None)
        command, decision, point = controller.command((0, 0), 0, *empty, default)
        self.assertIn(decision, ('approach_object', 'approach_turn', 'orient_object', 'inspect_object'))
        self.assertEqual(controller.attempts, 1)
        self.assertGreaterEqual(np.linalg.norm(mem.center(controller.vantage) - np.array([.9, 0])), .47)
        # Place body at selected safe vantage, face the object, confirm inspection.
        vantage = mem.center(controller.vantage)
        yaw = np.arctan2(-vantage[1], .9-vantage[0])
        command, decision, _ = controller.command(vantage, yaw, *empty, default)
        self.assertEqual(decision, 'inspect_object')
        self.assertEqual(obj.inspections, 1)
        self.assertEqual(controller.active_id, None)
        self.assertEqual(obj.tracks[0].novelty, .2)

    def test_fallback_when_no_object_candidate(self):
        mem = OccupancyMemory()
        ctrl = CuriosityController(FrontierExplorer(mem), ObjectMemory(7))
        default = (np.zeros(3), 'scan', None)
        result = ctrl.command((0, 0), 0, np.full(23, 1.15), np.zeros(23, dtype=bool), default)
        self.assertEqual(result[1], 'scan')
        self.assertEqual(ctrl.attempts, 0)

    def test_contact_raises_risk_and_abandons_target(self):
        mem = OccupancyMemory()
        mem.cells[:] = mem.FREE
        obj = ObjectMemory(7)
        obj.tracks.append(ObjectTrack(0, [.9, 0], sightings=3, last_seen=1))
        obj.next_identifier=1
        ctrl=CuriosityController(FrontierExplorer(mem),obj)
        ctrl.command((0,0),0,np.full(23,1.15),np.zeros(23,bool),(np.zeros(3),'complete',None))
        ctrl.abandon(contact=True)
        self.assertIsNone(ctrl.active_id)
        self.assertGreater(obj.tracks[0].risk, .5)
        self.assertEqual(obj.contact_events, 1)


if __name__ == '__main__':
    unittest.main()
