"""Standalone mathematical checks for sequential RL and collision shield."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from autonomy.behavior import INPUT_SIZE, init_weights
from autonomy.temporal import (Replay, TemporalModel, safe_mask, approve_override, reward_from_motion,
                               double_dqn_targets, read_replay, write_replay,
                               train_double_dqn)


class TemporalTests(unittest.TestCase):
    def test_shield_blocks_forward_even_if_q_prefers_it(self):
        model = TemporalModel(init_weights())
        for weight in model.weights.values():
            weight.fill(0)
        model.weights['b3'][:] = [100, -1, -2]
        rays = np.full(23, 1.15)
        self.assertEqual(model.choose(np.zeros(INPUT_SIZE), rays), 0)
        rays[11] = 0.1
        self.assertEqual(model.choose(np.zeros(INPUT_SIZE), rays), 1)
        rng = np.random.default_rng(4)
        for _ in range(50):
            self.assertNotEqual(model.choose(np.zeros(INPUT_SIZE), rays, epsilon=1, rng=rng), 0)
        with self.assertRaises(ValueError):
            safe_mask(np.array([np.nan]*23))

    def test_route_arbitration_blocks_aggressive_neural_overrides(self):
        far = np.full(23, 1.15)
        close = far.copy()
        close[11] = 0.55
        self.assertFalse(approve_override(1, 0, far))
        self.assertFalse(approve_override(1, 2, close))
        self.assertFalse(approve_override(0, 2, far))
        self.assertTrue(approve_override(0, 2, close))
        self.assertTrue(approve_override(1, 1, far))
        self.assertFalse(approve_override(None, 0, far))

    def test_temporal_reward_depends_on_actual_progress_and_penalizes_contact(self):
        forward = reward_from_motion([0,0],[0.2,0],0,0,[1,0],0.02)
        backward = reward_from_motion([0,0],[-0.2,0],0,0,[1,0],0)
        collision = reward_from_motion([0,0],[0.2,0],0,0,[1,0],0.02,new_contacts=1)
        novel = reward_from_motion([0,0],[0.2,0],0,0,[1,0],0.10)
        self.assertGreater(forward, backward)
        self.assertGreater(novel, forward)
        self.assertLess(collision, forward)
        self.assertLess(reward_from_motion([0,0],[0.2,0],0,0,[1,0],0,fallen=True),0)

    def test_terminal_bootstrap_zero_and_next_safety_mask(self):
        online = TemporalModel(init_weights(3))
        target = TemporalModel(init_weights(2))
        for obj in (online,target):
            for weight in obj.weights.values(): weight.fill(0)
        online.weights['b3'][:] = [10, 4, 3]
        target.weights['b3'][:] = [10, 4, 3]
        x=np.zeros((2,INPUT_SIZE),np.float32)
        targets=double_dqn_targets(online,target,x,np.array([1.,2.]),
                 np.array([1.,0.]),np.array([[True,True,True], [False,True,True]]),gamma=0.9)
        np.testing.assert_allclose(targets, [1.,5.6], atol=1e-5)

    def test_replay_round_trip_and_bad_checkpoint_rejected(self):
        model=TemporalModel(init_weights(5))
        row={'version':2,'state':[0.]*INPUT_SIZE,'action':1,'reward':0.4,
             'next_state':[0.]*INPUT_SIZE,'done':False,
             'next_mask':[False,True,True],'scene_seed':8}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'experience.jsonl'
            write_replay(path,[row])
            data=read_replay([path]); self.assertEqual(len(data.actions),1)
            modelpath=Path(folder)/'model.json'
            model.save(modelpath, {'test':True})
            self.assertEqual(TemporalModel.load(modelpath).values(np.zeros(INPUT_SIZE)).shape,(3,))
            obj=json.loads(modelpath.read_text());obj['version']=1
            modelpath.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):TemporalModel.load(modelpath)
            row['next_mask']=[False,False,False]
            write_replay(path,[row])
            with self.assertRaises(ValueError):read_replay([path])

    def test_future_reward_propagates_back_to_previous_action(self):
        # s0 --a0 / reward=0 --> s1 --a1 / reward=1 --> terminal.
        # A one-step reward regressor leaves Q(s0,a0) near zero; TD can
        # discover that action a0 is useful due to a future outcome.
        s0=np.zeros(INPUT_SIZE,dtype=np.float32)
        s1=np.ones(INPUT_SIZE,dtype=np.float32)*0.5
        n=128
        replay=Replay(states=np.array([s0,s1]*n),
            actions=np.array([0,1]*n),
            rewards=np.array([0.,1.]*n,np.float32),
            next_states=np.array([s1,s0]*n),
            dones=np.array([0.,1.]*n,np.float32),
            next_masks=np.tile([True,True,True],(2*n,1)))
        initial=TemporalModel(init_weights(10))
        trained,info=train_double_dqn(replay,initial,epochs=55,lr=0.002,
                                      gamma=0.9,target_every=4,seed=23)
        self.assertGreater(trained.values(s0)[0],0.45)
        self.assertGreater(trained.values(s1)[1],0.75)
        self.assertEqual(info['n'],2*n)


if __name__ == '__main__':
    unittest.main()
