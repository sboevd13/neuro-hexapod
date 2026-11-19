"""Worker fixes for command locomotion v2.

Key differences from the legacy worker:
- fresh runs never silently reuse global stale observation normalization;
- validation uses deterministic mean actions instead of exploration samples;
- resume can explicitly restore normalization from the selected checkpoint;
- validation/reward logs no longer pretend rewards are metres;
- normalization is checkpoint-owned instead of being silently shared globally.
"""

import os

import jax
import numpy as np

from running_mean_std_jax import RunningMeanStd
from worker import WorkerThread as _LegacyWorkerThread, logger


class WorkerThread(_LegacyWorkerThread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The legacy worker auto-loads obs-norm/obs_norm_last.npz. That can make
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

        # Validation evaluates the CURRENT trainee deterministically.
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

    def run(self):
        use_norm = "(with obs norm)" if self.normalize_obs else ""
        logger.info(f"Spider-Worker {use_norm} {self.name} alive on {self.device}!")
        self.running = True

        state = self.jit_reset(self.env_rng_key)
        state["step"] = jax.random.randint(
            self.rng_key,
            state["step"].shape,
            0,
            self.env.max_steps - 2,
        )

        logger.info(
            f"Starting {self.random_steps_count} random steps to fill buffer..."
        )
        for i in range(self.random_steps_count):
            if not self.running:
                return
            state, _, _ = self.perform_step(state, self.jit_random_step)
            if i % 100 == 0 and len(self.episode_reward) > 0:
                episode_reward = np.mean(self.episode_reward)
                logger.info(
                    f"Warm-up {i:05d}\tmean episode reward: "
                    f"{episode_reward:1.4f}"
                )

        logger.info("Random steps finished. Starting training loop.")
        self.step = 0

        trainee_name, params, qf1, qf2, global_step, opt_params = (
            self.agent_queue.get()
        )
        trainee_params = self.transfer_params_if_needed(params)
        validant_params = trainee_params

        total_validation_reward = 0.0
        total_episodes = 0.0
        validation_step = 0

        while self.running:
            if self.step % 16 == 0:
                (
                    trainee_name,
                    trainee_params,
                    qf1,
                    qf2,
                    global_step,
                    opt_params,
                ) = self.agent_queue.get()
                trainee_params = self.transfer_params_if_needed(trainee_params)

            state, val_term, val_rew = self.perform_step(
                state,
                self.jit_agent_step,
                (trainee_params, validant_params),
            )

            com_val, step_val = jax.device_get(
                (
                    state["last_com"][self.plot_env_idx],
                    state["step"][self.plot_env_idx],
                )
            )

            self.best_agent_xs.append(float(com_val[0]))
            self.best_agent_ys.append(float(com_val[1]))
            self.best_agent_heights.append(float(com_val[2]))
            is_reset = int(step_val) == 0

            if is_reset and len(self.best_agent_heights) > 100:
                role_name = (
                    "Exploration"
                    if self.plot_env_idx < self.env_batch_size
                    else "Validation"
                )
                logger.warning(
                    f"Saving trajectory plot for env #{self.plot_env_idx} "
                    f"({role_name})"
                )
                self.plot_and_save_trajectory_graphs()
                self.best_agent_heights = []
                self.best_agent_xs = []
                self.best_agent_ys = []
                self.plot_env_idx = np.random.randint(0, self.total_batch_size)

            # Do not write a global obs-norm file here. Trainer checkpoints own
            # normalization state, preventing cross-run contamination.
            self.step += 1

            if self.step >= self.validation_start:
                total_validation_reward += val_rew
                total_episodes += val_term
                validation_step += 1

            if validation_step >= self.validation_steps and total_episodes > 0:
                reward_per_episode = float(
                    total_validation_reward / total_episodes
                )
                self.validant_last_reward = reward_per_episode

                if (
                    self.validant_best_reward is None
                    or reward_per_episode > self.validant_best_reward
                ):
                    self.validant_best_reward = reward_per_episode
                    agent_path = os.path.join(
                        "checkpoints",
                        f"{trainee_name}",
                        f"best_at_{global_step}",
                    )
                    logger.info(
                        f"!!! NEW RECORD: validation reward "
                        f"{reward_per_episode:1.2f}. Saving best agent to "
                        f"'{agent_path}'"
                    )
                    self.save_state(
                        agent_path,
                        trainee_params,
                        qf1,
                        qf2,
                        *opt_params,
                    )

                    mean, var, count = self.running_obs_mean_std.get_stats()
                    np.savez(
                        os.path.join(agent_path, "obs_norm.npz"),
                        obs_mean=jax.device_get(mean),
                        obs_var=jax.device_get(var),
                        obs_count=float(count),
                    )
                    validant_params = trainee_params

                total_validation_reward = 0.0
                total_episodes = 0.0
                validation_step = 0

        logger.debug("Worker finished")
