"""Behavioral adaptation from geometry-only simulated range observations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from autonomy.explorer import Rectangle, range_scan
from autonomy.interaction import OutcomeMemory, object_signature, target_surface, observed_motion

class InteractionTests(unittest.TestCase):
    def test_distinguish_apparent_width_without_object_ids(self):
        narrow=range_scan((0,0),0,[Rectangle(0.55,0,.07,.07)],max_range=1.15)
        wide=range_scan((0,0),0,[Rectangle(0.55,0,.07,.22)],max_range=1.15)
        self.assertEqual(object_signature((0,0),0,*narrow),'narrow')
        self.assertEqual(object_signature((0,0),0,*wide),'wide')

    def test_infer_motion_from_two_sensor_scans(self):
        start=range_scan((0,0),0,[Rectangle(.55,0,.07,.07)])
        end=range_scan((.15,0),0,[Rectangle(.70,0,.07,.07)])
        p1=target_surface((0,0),0,*start)
        p2=target_surface((.15,0),0,*end)
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertAlmostEqual(observed_motion(p1,p2),.15,delta=.035)

    def test_no_sensor_measurement_cannot_count_as_mobility(self):
        agent=OutcomeMemory()
        result=agent.learn('narrow','probe',None,True,False,'occluded')
        self.assertFalse(result['learned'])
        self.assertEqual(agent.choose('narrow'),'probe')

    def test_learning_transfers_to_new_object_in_same_width_class(self):
        agent=OutcomeMemory()
        self.assertEqual(agent.choose('narrow'),'probe')
        self.assertEqual(agent.choose('wide'),'probe')
        agent.learn('narrow','probe',.12,True,False,'small-A')
        agent.learn('wide','probe',.0,True,False,'wide-A')
        self.assertEqual(agent.choose('narrow'),'probe')
        self.assertEqual(agent.choose('wide'),'observe')
        self.assertGreater(agent.estimate('narrow'),agent.estimate('wide'))
        # A physically distinct later small/large object shares the learned class.
        self.assertEqual(agent.choose('narrow'),'probe')
        self.assertEqual(agent.choose('wide'),'observe')

    def test_no_contact_does_not_count_as_failed_push(self):
        agent=OutcomeMemory()
        agent.learn('wide','probe',0.,False,False,'missed')
        self.assertEqual(agent.choose('wide'),'probe')
        self.assertEqual(len(agent.history),1)
        self.assertFalse(agent.history[0]['learned'])

    def test_fall_does_not_count_as_mobility(self):
        agent=OutcomeMemory()
        agent.learn('narrow','probe',.3,True,True,'fell')
        self.assertEqual(agent.choose('narrow'),'probe')

    def test_load_roundtrip_and_reject_corruption(self):
        agent=OutcomeMemory()
        agent.learn('wide','probe',.0,True,False,'first')
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'memory.json'
            agent.save(path)
            read=OutcomeMemory.load(path)
            self.assertEqual(read.choose('wide'),'observe')
            path.write_text('{"version":1,"kind":"object_affordance_beta","observations":{"wide":{"moved":-1,"not_moved":1}},"history":[]}')
            with self.assertRaises(ValueError): OutcomeMemory.load(path)

if __name__=='__main__':
    unittest.main()
