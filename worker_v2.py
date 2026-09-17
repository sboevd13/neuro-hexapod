"""Worker fixes for command locomotion v2.

Key differences from the legacy worker:
- fresh runs never silently reuse global stale observation normalization;
- validation uses deterministic mean actions instead of exploration samples;
- resume can explicitly restore normalization from the selected checkpoint.
"""

import os

import jax
import numpy as np

from running_mean_std_jax import RunningMeanStd
from worker import WorkerThread as _LegacyWorkerThread


class WorkerThread(_LegacyWorkerThread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The legacy worker auto-loads obs-norm/obs_norm_last.npz.  That can make
        # a supposedly fresh experiment inherit statistics from an older policy.
        # V2 always starts clean unless load_obs_norm() is explicitly called.
        self.running_obs_mean_std = RunningMeanStd(
            shape=self.env.observation_space_shape
        )
        self.obs_rms_mean = jax.device_put(
            jax.numpy.zeros(self.env.observation_space_shape),
            device=self.device,
        )
        self.obs_rms_var = jax.device_put(
            jax.numpy.ones(self.env.observation_space_shape),
            device=self.device,
        )

    def load_obs_norm(self, checkpoint_dir):
        """Explicitly restore observation normalization for a resumed run."""
        norm_path = None
        for name in ("obs_norm_last.npz", "obs_norm.npz"):
            candidate = os.path.join(checkpoint_dir, name)
            if os.path.exists(candidate):
                norm_path = candidate
                break

        if norm_path is None:
            raise FileNotFoundError(
                f"No obs_norm_last.npz or obs_norm.npz in {checkpoint_dir}"
            )

        values = np.load(norm_path)
        mean = jax.numpy.asarray(values["obs_mean"])
        var = jax.numpy.asarray(values["obs_var"])
        count = float(values["obs_count"]) if "obs_count" in values else 1e-4

        if tuple(mean.shape) != tuple(self.env.observation_space_shape):
            raise ValueError(
                f"Normalization shape mismatch: {mean.shape} vs "
                f"{self.env.observation_space_shape}"
            )

        self.running_obs_mean_std.set_stats(mean, var, count)
        self.obs_rms_mean = jax.device_put(mean, device=self.device)
        self.obs_rms_var = jax.device_put(var, device=self.device)
        return norm_path

    def agent_step(self, env_state, inference_key, trainee_params, validant_params):
        train_key, val_key = jax.random.split(inference_key, 2)

        trainee_func = self.agent_config["trainee_func"]
        validation_func = self.agent_config.get(
            "validation_agent_func", trainee_func
        )

        obs_raw = env_state["obs"]
        agent_obs = self.normalize_obs_if_needed(obs_raw)

        # Training environments keep SAC exploration.
        action_trainee, _, _ = trainee_func(
            trainee_params,
            agent_obs[:self.env_batch_size],
            train_key,
        )

        # Validation evaluates the CURRENT trainee deterministically.  Do not use
        # a stochastic sample and do not evaluate an older frozen policy here.
        action_val, _, _ = validation_func(
            trainee_params,
            agent_obs[self.env_batch_size:],
            val_key,
        )

        combined_actions = jax.numpy.concatenate(
            [action_trainee, action_val], axis=0
        )

        state, true_obs, true_reward, terminated, validation_rewards = (
            self.batched_step(env_state, combined_actions)
        )

        validation_terminated_count = jax.numpy.sum(
            terminated[self.env_batch_size:]
        )
        validation_reward_sum = jax.numpy.sum(
            validation_rewards[self.env_batch_size:]
        )

        return (
            state,
            combined_actions,
            obs_raw,
            true_obs,
            true_reward,
            terminated,
            validation_terminated_count,
            validation_reward_sum,
        )
