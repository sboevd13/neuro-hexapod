import unittest
from autonomy.world_scene import sample_props,make_world

MODEL='<mujoco><worldbody><body name="dummy"><geom type="sphere" size=".1" /></body></worldbody></mujoco>'
class WorldSceneTests(unittest.TestCase):
    def test_random_independent_object_dynamics(self):
        a=sample_props(7);b=sample_props(8)
        self.assertEqual(len(a),3)
        self.assertEqual(sum(p.mobile for p in a),2)
        self.assertNotEqual(a,b)
        self.assertEqual(a,sample_props(7))
        self.assertTrue(all(.025<=p.mass<=.27 and .14<=p.friction<=1.1 for p in a))
    def test_scene_adds_freejoints_only_for_mobile_props(self):
        xml,specs,walls=make_world(MODEL,seed=7)
        self.assertEqual(xml.count('<freejoint'),2)
        self.assertEqual(xml.count('name="interaction_target_')//2>=3,True)
        self.assertEqual(len(walls),4)
    def test_other_counts_rejected(self):
        with self.assertRaises(ValueError):sample_props(4,2)

if __name__=='__main__':unittest.main()
