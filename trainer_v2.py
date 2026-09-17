"""Trainer wrapper for command locomotion v2.

Keeps the existing SAC update implementation, but fixes misleading distance labels,
uses an explicit run name, saves alpha/target-Q state, and records current_step before
auto-checkpointing.
"""

import os
from queue import Full

import jax
from jax import jit
import numpy as np
from flax import serialization

from trainer import Reporter, TrainingThread as _LegacyTrainingThread


class TrainingThread(_LegacyTrainingThread):
    def save_state(self, path):
        super().save_state(path)
        self.write_params_to_file(
            os.path.join(path, "log_alpha.flax"), self.log_alpha
        )
        self.write_params_to_file(
            os.path.join(path, "q1_target.flax"), self.qf_target1_params
        )
        self.write_params_to_file(
            os.path.join(path, "q2_target.flax"), self.qf_target2_params
        )

    def load_state(self, path):
        super().load_state(path)

        alpha_path = os.path.join(path, "log_alpha.flax")
        if os.path.exists(alpha_path):
            with open(alpha_path, "rb") as checkpoint:
                self.log_alpha = serialization.from_bytes(
                    self.log_alpha, checkpoint.read()
                )
            self.alpha = jax.numpy.exp(self.log_alpha)

        q1_target_path = os.path.join(path, "q1_target.flax")
        if os.path.exists(q1_target_path):
            with open(q1_target_path, "rb") as checkpoint:
                self.qf_target1_params = serialization.from_bytes(
                    self.qf_target1_params, checkpoint.read()
                )

        q2_target_path = os.path.join(path, "q2_target.flax")
        if os.path.exists(q2_target_path):
            with open(q2_target_path, "rb") as checkpoint:
                self.qf_target2_params = serialization.from_bytes(
                    self.qf_target2_params, checkpoint.read()
                )

    def run(self):
        self.running = True

        jit_q_step = jit(self.q_step, device=self.device)
        jit_policy_step = jit(self.policy_step, device=self.device)
        jit_target_step = jit(self.target_step, device=self.device)

        step = self.current_step
        q_loss = 0.0
        p_loss = 0.0
        a_loss = 0.0

        config = dict(
            batch_size=self.batch_size,
            policy_update_interval=self.policy_update_interval,
            initial_alpha=self.initial_alpha,
            tau=self.tau,
            gamma=self.gamma,
            q_lr=self.q_lr,
            p_lr=self.p_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )
        config.update(self.config)

        reporter = Reporter(
            use_tensorboard=config["report_to_tensorboard"],
            use_wandb=config["report_to_wandb"],
            tensorboard_dir="runs/",
            wandb_project="hexapod-command-locomotion-v2",
            wandb_config=config,
        )
        if not config["report_to_wandb"]:
            reporter.run_name = config.get(
                "run_name", "command_locomotion_v2"
            )

        os.makedirs(os.path.join("checkpoints", reporter.run_name), exist_ok=True)

        # Seed the worker queue with the current policy.
        for _ in range(self.agent_queue.maxsize + 2):
            self.agent_queue.put(
                (
                    reporter.run_name,
                    self.agent_params,
                    self.qf1_params,
                    self.qf2_params,
                    step,
                    (self.q_opt_state, self.p_opt_state, self.alpha_opt_state),
                ),
                block=self.synchroneous,
            )

        try:
            while self.running:
                batch = self.buffer_thread.sample(self.batch_size)
                observations, next_observations, actions, rewards, dones = [
                    jax.device_put(data, self.device) for data in batch
                ]

                mean_reward = rewards.mean()
                inference_key0, inference_key1, self.rng_key = jax.random.split(
                    self.rng_key, 3
                )

                mean_raw, var_raw, _ = (
                    self.worker_thread.running_obs_mean_std.get_stats()
                )
                current_mean = jax.device_put(mean_raw, self.device)
                current_var = jax.device_put(var_raw, self.device)

                q_loss, self.qf1_params, self.qf2_params, self.q_opt_state = (
                    jit_q_step(
                        self.q_opt_state,
                        observations,
                        next_observations,
                        actions,
                        rewards,
                        dones,
                        self.qf1_params,
                        self.qf2_params,
                        self.qf_target1_params,
                        self.qf_target2_params,
                        self.agent_params,
                        inference_key0,
                        self.alpha,
                        current_mean,
                        current_var,
                    )
                )

                if (
                    self.policy_update_interval > 0
                    and step % self.policy_update_interval == 0
                ):
                    (
                        p_loss,
                        a_loss,
                        self.agent_params,
                        self.alpha,
                        self.log_alpha,
                        self.p_opt_state,
                        self.alpha_opt_state,
                    ) = jit_policy_step(
                        self.p_opt_state,
                        self.alpha_opt_state,
                        observations,
                        self.agent_params,
                        self.qf1_params,
                        self.qf2_params,
                        inference_key1,
                        self.alpha,
                        self.log_alpha,
                        current_mean,
                        current_var,
                    )

                self.qf_target1_params, self.qf_target2_params = jit_target_step(
                    self.qf_target1_params,
                    self.qf_target2_params,
                    self.qf1_params,
                    self.qf2_params,
                )

                if step % 25 == 0 or step >= self.total_steps:
                    episode_reward = (
                        float(np.mean(self.worker_thread.episode_reward))
                        if self.worker_thread.episode_reward
                        else 0.0
                    )
                    current_p_lr = self.p_lr_schedule(step)
                    current_q_lr = self.q_lr_schedule(step)

                    data = {
                        "q-loss": float(q_loss),
                        "p-loss": float(p_loss),
                        "a-loss": float(a_loss),
                        "alpha": float(self.alpha),
                        "reward-buffer-mean": float(mean_reward),
                        "reward-episode": float(episode_reward),
                        "p_lr": float(current_p_lr),
                        "q_lr": float(current_q_lr),
                        "env-steps": self.worker_thread.step,
                    }

                    validation_str = ""
                    if self.worker_thread.validant_last_reward is not None:
                        last_val = float(
                            self.worker_thread.validant_last_reward
                        )
                        best_val = float(
                            self.worker_thread.validant_best_reward
                        )
                        data["validation-reward-last"] = last_val
                        data["validation-reward-best"] = best_val
                        validation_str = (
                            f" | Val Reward: {last_val:8.2f}"
                            f" | Best Val: {best_val:8.2f}"
                        )

                    print(
                        f"Step: {step:05d} | Env Step: {self.worker_thread.step} | "
                        f"Q-Loss: {float(q_loss):8.4f} | "
                        f"Alpha: {float(self.alpha):6.4f} | "
                        f"Episode Reward: {episode_reward:8.2f}"
                        f"{validation_str}"
                    )
                    reporter.log(data, step=step)

                if step % 16 == 0:
                    try:
                        self.agent_queue.put(
                            (
                                reporter.run_name,
                                self.agent_params,
                                self.qf1_params,
                                self.qf2_params,
                                step,
                                (
                                    self.q_opt_state,
                                    self.p_opt_state,
                                    self.alpha_opt_state,
                                ),
                            ),
                            block=self.synchroneous,
                        )
                    except Full:
                        pass

                # Save the step that actually corresponds to these weights.
                self.current_step = step
                if step % 250 == 0 or step >= self.total_steps:
                    self.save_state(
                        os.path.join(
                            "checkpoints", reporter.run_name, "last"
                        )
                    )

                if step >= self.total_steps:
                    break
                step += 1
        finally:
            reporter.close()
