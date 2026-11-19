"""Model safety, data generation and training regressions. NumPy only."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from autonomy.behavior import (BehaviorModel, INPUT_SIZE, init_weights,
                               state_features, ACTIONS)
from autonomy.explorer import SENSOR_RANGE
from train_behavior import action_rewards, clearance, data_set, learn, metrics, scenario


class NeuralBehaviorTests(unittest.TestCase):
    def test_features_and_q_shape(self):
        x = state_features(np.full(23, SENSOR_RANGE), 0, [0, 0], [1, 0])
        self.assertEqual(x.shape, (INPUT_SIZE,))
        self.assertAlmostEqual(x[-2], 1.)
        model = BehaviorModel(init_weights())
        self.assertEqual(model.values(x).shape, (3,))
        self.assertEqual(model.values(np.stack([x, x])).shape, (2, 3))

    def test_forward_shield_cannot_be_bypassed_by_network(self):
        model = BehaviorModel(init_weights())
        model.weights['w1'].fill(0)
        model.weights['w2'].fill(0)
        model.weights['w3'].fill(0)
        model.weights['b3'][:] = [100, -1, -2]
        x = np.zeros(INPUT_SIZE, np.float32)
        self.assertEqual(model.choose(x, np.full(23, 1.15)), 0)
        rays = np.full(23, 1.15)
        rays[11] = 0.10
        self.assertNotEqual(model.choose(x, rays), 0)

    def test_checkpoint_round_trip(self):
        model = BehaviorModel(init_weights())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'behavior.json'
            model.save(path, {'test': True})
            copy = BehaviorModel.load(path)
            x = np.ones(INPUT_SIZE, np.float32)
            np.testing.assert_allclose(model.values(x), copy.values(x))
            path.write_text('{"version":99,"weights":{}}')
            with self.assertRaises(ValueError):
                BehaviorModel.load(path)

    def test_bad_sensor_rejected(self):
        model = BehaviorModel(init_weights())
        with self.assertRaises(ValueError):
            state_features(np.array([float('nan')]*23), 0, [0,0], [1,0])
        with self.assertRaises(ValueError):
            model.choose(np.zeros(INPUT_SIZE), np.zeros(18))

    def test_collision_penalty_and_model_learning(self):
        rng = np.random.default_rng(5)
        boxes = scenario(rng)
        x, y = data_set(256, 100)
        model = BehaviorModel(learn(init_weights(), x, y, epochs=3))
        self.assertTrue(np.isfinite(model.values(x)).all())
        self.assertEqual(len(ACTIONS), 3)

    def test_training_is_deterministic(self):
        x, y = data_set(90, 3)
        first = learn(init_weights(), x, y, epochs=2)
        second = learn(init_weights(), x, y, epochs=2)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


if __name__ == '__main__':
    unittest.main()
