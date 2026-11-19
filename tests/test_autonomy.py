"""CPU-only checks for exploration. Does not require MuJoCo/JAX."""
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from autonomy.explorer import (FrontierExplorer, OccupancyMemory, Rectangle,
                               inflated_obstacles, range_scan)
from autonomy.scene import make_scene


class RangeSensorTests(unittest.TestCase):
    def test_forward_hit_and_miss(self):
        box = Rectangle(0.60, 0.0, 0.10, 0.20)
        distances, hits = range_scan([0, 0], 0.0, [box],
                                     angles=np.array([0, np.pi/2]), max_range=1.0)
        self.assertTrue(hits[0])
        self.assertAlmostEqual(distances[0], 0.5)
        self.assertFalse(hits[1])
        self.assertEqual(distances[1], 1.0)

    def test_rotate_sensor(self):
        box = Rectangle(0.0, 0.75, 0.12, 0.10)
        distances, hits = range_scan([0, 0], np.pi/2, [box], angles=[0.0])
        self.assertTrue(hits[0])
        self.assertAlmostEqual(distances[0], 0.65)


class OccupancyTests(unittest.TestCase):
    def test_map_records_new_space_and_contact(self):
        memory = OccupancyMemory(size=2.0, resolution=0.05)
        before = memory.coverage
        memory.observe([0, 0], 0, distances=[0.5], hits=[True], angles=[0])
        self.assertGreater(memory.coverage, before)
        self.assertEqual(memory.cells[memory.cell([0.512, 0])], 1)
        self.assertEqual(memory.cells[memory.cell([0.25, 0])], 0)
        memory.mark_bump([0, 0], np.pi/2, distance=0.2)
        self.assertEqual(memory.cells[memory.cell([0, 0.2])], 1)

    def test_inflation_preserves_clearance(self):
        cells = np.full((9, 9), -1, dtype=np.int8)
        cells[4, 4] = 1
        inflated = inflated_obstacles(cells, 2)
        self.assertTrue(inflated[4, 4])
        self.assertTrue(inflated[4, 6])
        self.assertFalse(inflated[4, 7])
        self.assertFalse(inflated[6, 6])

    def test_memory_is_persistent_and_seed_bound(self):
        memory = OccupancyMemory(size=2, resolution=0.05)
        memory.observe([0, 0], 0, [0.7], [False], [0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.json'
            memory.save(path, scene_seed=7)
            restored = OccupancyMemory.load(path, scene_seed=7)
            np.testing.assert_array_equal(restored.cells, memory.cells)
            np.testing.assert_array_equal(restored.visits, memory.visits)
            self.assertEqual(restored.survey_count, 1)
            with self.assertRaises(ValueError):
                OccupancyMemory.load(path, scene_seed=8)

    def test_frontier_is_not_hard_coded(self):
        memory = OccupancyMemory(size=2.0, resolution=0.1)
        memory.observe([0, 0], 0, np.full(23, 0.75),
                       np.zeros(23, dtype=bool))
        explorer = FrontierExplorer(memory, clearance=0.15)
        action, reason, waypoint = explorer.command(np.array([0., 0.]), 0.)
        self.assertIn(reason, ('move', 'turn'))
        self.assertIsNotNone(waypoint)
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.isfinite(action).all())
        # Re-observation changes the available map; a collision is remembered.
        memory.mark_bump([0, 0], 0, distance=0.28)
        self.assertEqual(memory.cells[memory.cell([0.28, 0])], 1)
        action2, reason2, _ = explorer.command(np.array([0., 0.]), 0.)
        self.assertIn(reason2, ('move', 'turn', 'scan'))
        self.assertTrue(np.isfinite(action2).all())


class SceneTests(unittest.TestCase):
    def test_scene_adds_obstacles_without_modifying_robot(self):
        xml = '<mujoco><option timestep="0.002"/><worldbody><body name="body"/></worldbody></mujoco>'
        a = make_scene(xml, seed=3, count=4)
        b = make_scene(xml, seed=3, count=4)
        self.assertEqual(a.xml, b.xml)
        self.assertEqual(len(a.obstacles), 8)  # 4 walls + 4 obstacles
        root = ET.fromstring(a.xml)
        self.assertEqual(len(root.findall('./worldbody/geom')), 8)
        self.assertIsNotNone(root.find('./worldbody/body[@name="body"]'))
        for box in a.obstacles[4:]:
            self.assertGreater(np.linalg.norm([box.x, box.y]), 0.5)


if __name__ == '__main__':
    unittest.main()
