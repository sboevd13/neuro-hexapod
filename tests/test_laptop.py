"""Regression tests for policy/physics contracts, not just training-script syntax."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import numpy as np

import reference_gait as v5
from laptop_policy import (NumpyPolicy, generalized_advantage, init_params,
    load_checkpoint, make_optimizer, network, save_checkpoint)
from locomotion_laptop import (Config, LocomotionEnv, MODEL_PATH,
    directional_reference, load_model)


class LaptopContracts(unittest.TestCase):
    def test_forward_reference_preserves_v5_choreography(self):
        for phase in [0., 17/1024, 313/1024, 779/1024]:
            np.testing.assert_allclose(directional_reference(phase, [1, 0, 0]),
                                       v5.reference_targets(phase), atol=1e-6)

    def test_numpy_rollout_and_jax_optimizer_use_same_network(self):
        params = init_params()
        # Nonzero trained-like head, rather than testing only the zero initializer.
        params['wm'] = params['wm'] + .017
        params['wv'] = params['wv'] + .023
        obs = np.random.default_rng(9).normal(size=(7, 65)).astype(np.float32)
        cpu_action, _, _, cpu_value = NumpyPolicy(params).act(obs)
        mean, _, value = jax.jit(network)(params, obs)
        np.testing.assert_allclose(cpu_action, np.tanh(np.asarray(mean)), atol=2e-6)
        np.testing.assert_allclose(cpu_value, value, atol=2e-6)

    def test_time_limit_bootstraps_but_does_not_leak_next_episode(self):
        reward = np.array([[1.], [100.]], np.float32)
        zero = np.zeros_like(reward)
        next_values = np.array([[5.], [7.]], np.float32)
        _, returns = generalized_advantage(reward, zero, next_values, zero,
                                           np.array([[1.], [0.]], np.float32))
        self.assertAlmostEqual(float(returns[0, 0]), 1+.99*5, places=5)
        _, returns = generalized_advantage(reward, zero, next_values,
                                           np.array([[1.], [0.]], np.float32), zero)
        self.assertAlmostEqual(float(returns[0, 0]), 1., places=5)

    def test_actor_cannot_read_simulator_joint_or_linear_velocity_feedback(self):
        env = LocomotionEnv()
        before = env.reset(seed=41)
        env.data.qpos[7:] += .02
        env.data.qvel[:3] = [10, 20, 30]
        env.data.qvel[6:] = 40
        np.testing.assert_array_equal(before, env.observation())

    def test_checkpoint_restores_optimizer_weights_and_exact_physics(self):
        params = init_params()
        optimizer = make_optimizer()
        state = optimizer.init(params)
        grads = jax.tree_util.tree_map(lambda a: a*.1+.03, params)
        _, state = optimizer.update(grads, state, params)
        xml = MODEL_PATH.read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.npz'
            save_checkpoint(path, params, state,
                            dict(config=Config().to_dict(), env_steps=2048, updates=1), xml)
            p2, s2, metadata, xml2 = load_checkpoint(path)
            self.assertEqual(xml, xml2)
            self.assertEqual(metadata['env_steps'], 2048)
            for a, b in zip(jax.tree_util.tree_leaves((params, state)),
                            jax.tree_util.tree_leaves((p2, s2))):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(load_model(xml2).nu, 18)

    def test_rate_and_residual_limits_under_extreme_policy(self):
        env = LocomotionEnv()
        env.reset(seed=1)
        for step in range(80):
            before = env.commanded_q.copy()
            _, _, fallen, _, info = env.step(np.full(18, 1 if step%2 else -1))
            self.assertLessEqual(np.max(np.abs(env.commanded_q-before)), np.deg2rad(6)+1e-10)
            self.assertTrue(np.all(np.abs(info['residual_deg']) <= np.tile([3, 4, 4], 6)+1e-7))
            self.assertFalse(fallen)


if __name__ == '__main__':
    unittest.main()
