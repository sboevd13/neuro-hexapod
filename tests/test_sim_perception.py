import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from autonomy.sim_perception import add_sim_sensors, detect_colored_objects, save_png
from autonomy.localization import PlanarEstimator, wrap
from autonomy.explorer import OccupancyMemory, SCAN_ANGLES


class PerceptionTests(unittest.TestCase):
    def test_mount_sensors_preserves_world_props_and_actuators(self):
        xml=('''<mujoco><worldbody><body name="body"><freejoint/></body>
                <body name="interaction_target_00"><geom name="interaction_target_00_geom" type="box" size=".1 .1 .1"/></body>
                </worldbody><actuator><motor joint="q1"/></actuator></mujoco>''')
        result=ET.fromstring(add_sim_sensors(xml))
        self.assertEqual(len(result.findall(".//camera[@name='navigation_rgb']")),1)
        self.assertEqual(len(result.findall(".//gyro[@name='navigation_gyro']")),1)
        self.assertEqual(len(result.findall('.//motor')),1)
        self.assertIn('0.13 0.72',result.find('.//geom').get('rgba'))
        with self.assertRaises(ValueError):add_sim_sensors(add_sim_sensors(xml))

    def test_cyan_camera_finds_objects_without_object_ids(self):
        frame=np.zeros((80,120,3),dtype=np.uint8)
        frame[:]=[130,130,130]
        frame[20:50,10:40]=[20,165,192]
        frame[21:55,70:110]=[20,165,192]
        detection=detect_colored_objects(frame)
        self.assertEqual(len(detection),2)
        self.assertLess(detection[0]['bearing_rad'],0)
        self.assertGreater(detection[1]['bearing_rad'],0)
        self.assertEqual(detect_colored_objects(np.full((80,120,3),120,dtype=np.uint8)),[])

    def test_rgb_png_round_trip_signature(self):
        frame=np.zeros((15,18,3),dtype=np.uint8)
        frame[:,:,1]=170
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'frame.png'
            save_png(path,frame)
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))

    def test_gyro_and_planted_contact_foot_odom(self):
        loc=PlanarEstimator(matching=False)
        feet=np.array([[-.13,-.09],[-.1,.12],[.11,-.09],[.12,.12],[-.2,.0],[.22,0]])
        contact=np.ones(6,dtype=bool)
        loc.integrate(0,feet,contact,.02)
        for j in range(20):
            feet[:,0]-=.002
            pose=loc.integrate(0,feet,contact,.02)
        self.assertGreaterEqual(loc.foot_updates,20)
        self.assertAlmostEqual(pose[0],.04,places=3)
        self.assertAlmostEqual(pose[1],0,places=3)
        self.assertAlmostEqual(pose[2],0,places=3)

    def test_heading_integration_and_fallback_on_airborne(self):
        loc=PlanarEstimator(matching=True)
        f=np.zeros((6,2))
        for _ in range(50):
            pose=loc.integrate(.2,f,np.zeros(6,dtype=bool),.02,[.85,0.,0.])
        self.assertAlmostEqual(pose[2],.2,places=3)
        self.assertGreater(pose[0],.05)
        self.assertEqual(loc.foot_updates,0)
        self.assertEqual(loc.fallback_updates,50)

    def test_no_map_scan_matching_does_not_change_pose(self):
        loc=PlanarEstimator(spawn=(.12,-.03,.15))
        map_=OccupancyMemory()
        before=loc.pose.copy()
        after=loc.correct_with_map(map_,np.full(23,.8),np.ones(23,dtype=bool))
        np.testing.assert_allclose(before,after)

    def test_reject_bad_foot_packet(self):
        loc=PlanarEstimator()
        with self.assertRaises(ValueError):
            loc.integrate(0,np.zeros((6,3)),np.ones(6,dtype=bool),.02)
        with self.assertRaises(ValueError):
            detect_colored_objects(np.zeros((10,10,3)))

    def test_pose_wrap(self):
        self.assertAlmostEqual(wrap(2*math.pi+.1),.1)


if __name__=='__main__':unittest.main()
