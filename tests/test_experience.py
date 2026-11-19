"""Feedback adaptation is validated independently of MuJoCo/JAX."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from autonomy.behavior import BehaviorModel, init_weights, INPUT_SIZE
from autonomy.experience import (experience_reward, fit_experience,
                                 read_experiences, save_experiences, observed_mse)


class ExperienceTests(unittest.TestCase):
    def test_progress_and_collision_reward(self):
        reward=experience_reward([0,0],[0.1,0],0,0,[1,0],0)
        self.assertGreater(reward,0)
        penalty=experience_reward([0,0],[0.1,0],0,0,[1,0],1)
        self.assertLess(penalty,reward)
        self.assertLess(penalty,0)

    def test_experience_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'experience.jsonl'
            examples=[{'features':[0.3]*INPUT_SIZE,'action':1,'reward':-0.4,'scene_seed':7}]
            save_experiences(path,examples)
            x,a,r=read_experiences([path])
            self.assertEqual(x.shape,(1,INPUT_SIZE))
            self.assertEqual(a.tolist(),[1])
            self.assertAlmostEqual(float(r[0]),-0.4,places=5)

    def test_masked_experience_learning(self):
        rng=np.random.default_rng(21)
        x=rng.uniform(-1,1,(80,INPUT_SIZE)).astype(np.float32)
        a=np.zeros(80,dtype=np.int64)
        target=np.full(80,0.65,dtype=np.float32)
        initial=BehaviorModel(init_weights())
        before=observed_mse(initial,x,a,target)
        fitted=fit_experience(initial,x,a,target,epochs=30)
        after=observed_mse(fitted,x,a,target)
        self.assertLess(after,before)
        self.assertTrue(np.isfinite(fitted.values(x)).all())

    def test_bad_event_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.jsonl'
            save_experiences(p,[{'features':[1]*INPUT_SIZE,'action':99,'reward':0}])
            with self.assertRaises(ValueError):
                read_experiences([p])


if __name__=='__main__':
    unittest.main()
