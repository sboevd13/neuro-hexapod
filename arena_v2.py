"""Safer command-conditioned locomotion environment.

V2 keeps the accepted V5 gait protected for straight-forward motion while still
allowing broad policy authority for side/back/turn commands.  It also uses a more
realistic loaded-servo command-rate limit and stronger anti-chatter regularization.
"""

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from arena import BattleArena as _LegacyBattleArena
from reference_gait import CYCLE_TIME


class BattleArena(_LegacyBattleArena):
    # MG996R datasheet speed is roughly 0.14 s / 60 deg unloaded.  The real robot
    # moves a heavy leg, so train against a deliberately more conservative loaded
    # command speed: 0.20 s / 60 deg = 300 deg/s = 6 deg per 20 ms control tick.
    SERVO_SPEED_RAD_S = np.deg2rad(60.0) / 0.20

    # Full authority remains available away from straight-forward V5 motion.
    POLICY_RANGE_DEG = (40.0, 55.0, 65.0)

    # Straight forward already has a known-good V5 gait.  The actor is therefore
    # a small residual there, not a replacement gait generator.
    FORWARD_POLICY_RANGE_DEG = (10.0, 12.0, 15.0)

    def __init__(self) -> None:
        super().__init__()
        self.full_policy_scale = self.policy_scale
        one_leg_forward = jp.deg2rad(
            jp.array(self.FORWARD_POLICY_RANGE_DEG, dtype=jp.float32)
        )
        self.forward_policy_scale = jp.tile(one_leg_forward, 6)

    def _policy_scale_for_command(self, command):
        """Small residual on V5; smoothly expand to full authority elsewhere."""
        prior = self._forward_prior_strength(command)
        return (
            self.full_policy_scale * (1.0 - prior)
            + self.forward_policy_scale * prior
        )

    def _sample_command(self, rng):
        """Sample commands while explicitly preserving enough straight-forward V5.

        10% idle, 15% pure yaw, 20% straight-forward analog, 55% general analog
        translation. General translation can still include simultaneous yaw.
        """
        (
            k_mode,
            k_angle,
            k_radius,
            k_yaw,
            k_turn,
            k_hold,
        ) = jax.random.split(rng, 6)

        mode = jax.random.uniform(k_mode)
        radius = 0.20 + 0.80 * jp.sqrt(jax.random.uniform(k_radius))
        angle = jax.random.uniform(k_angle, minval=-jp.pi, maxval=jp.pi)

        vx = radius * jp.cos(angle)
        vy = radius * jp.sin(angle)

        yaw_u = jax.random.uniform(k_yaw)
        translation_yaw = jp.where(
            yaw_u < 0.15,
            1.0,
            jp.where(yaw_u > 0.85, -1.0, 0.0),
        )
        turn_sign = jp.where(jax.random.bernoulli(k_turn), 1.0, -1.0)

        idle = jp.zeros(3, dtype=jp.float32)
        pure_turn = jp.array([0.0, 0.0, turn_sign], dtype=jp.float32)
        pure_forward = jp.array([radius, 0.0, 0.0], dtype=jp.float32)
        translation = jp.array([vx, vy, translation_yaw], dtype=jp.float32)

        command = jp.where(
            mode < 0.10,
            idle,
            jp.where(
                mode < 0.25,
                pure_turn,
                jp.where(mode < 0.45, pure_forward, translation),
            ),
        )

        hold_steps = jax.random.randint(
            k_hold,
            (),
            self.COMMAND_HOLD_MIN_STEPS,
            self.COMMAND_HOLD_MAX_STEPS + 1,
            dtype=jp.int32,
        )
        return command, hold_steps

    def compute_reward(self, old_state, new_data, q_ref, action):
        body_pos = new_data.xpos[self.body_id]
        old_body_pos = old_state["data"].xpos[self.body_id]

        roll, pitch, yaw = self._rpy_from_quat(new_data.qpos[3:7])
        _, _, old_yaw = self._rpy_from_quat(old_state["data"].qpos[3:7])

        # Privileged simulator velocity remains reward-only.
        world_velocity_xy = (
            body_pos[0:2] - old_body_pos[0:2]
        ) / self.control_dt

        cy = jp.cos(yaw)
        sy = jp.sin(yaw)
        body_velocity = jp.array([
            cy * world_velocity_xy[0] + sy * world_velocity_xy[1],
            -sy * world_velocity_xy[0] + cy * world_velocity_xy[1],
        ])
        yaw_rate = self._wrap_angle(yaw - old_yaw) / self.control_dt

        command = old_state["command"]
        target_velocity = command[0:2] * self.MAX_LINEAR_SPEED_M_S
        target_yaw_rate = command[2] * self.MAX_YAW_RATE_RAD_S

        velocity_error_sq = jp.sum((body_velocity - target_velocity) ** 2)
        yaw_error = yaw_rate - target_yaw_rate
        yaw_error_sq = yaw_error ** 2

        linear_tracking_reward = 6.0 * jp.exp(
            -velocity_error_sq / (0.20 ** 2)
        )
        yaw_tracking_reward = 2.0 * jp.exp(
            -yaw_error_sq / (0.70 ** 2)
        )

        height_penalty = 5.0 * ((body_pos[2] - 0.105) ** 2)
        tilt_penalty = 1.0 * (roll * roll + pitch * pitch)

        actual_q = new_data.qpos[7:25]
        joint_speed = new_data.qvel[6:24]
        actuator_force = new_data.actuator_force

        forward_prior = self._forward_prior_strength(command)

        # Keep the known-good V5 attractor meaningful on straight forward motion.
        reference_penalty = (
            0.80 * forward_prior * jp.mean((actual_q - q_ref) ** 2)
        )
        forward_correction_penalty = (
            0.25 * forward_prior * jp.mean(action ** 2)
        )

        # Penalize high-frequency policy chatter and excessive leg angular speed.
        action_penalty = 0.02 * jp.mean(action ** 2)
        smooth_penalty = 0.30 * jp.mean(
            (action - old_state["last_action"]) ** 2
        )
        joint_speed_penalty = 0.004 * jp.mean(joint_speed ** 2)

        # Wrong yaw should be actively bad, not merely lose the +2 yaw reward.
        yaw_error_penalty = 0.20 * jp.minimum(yaw_error_sq, 9.0)

        power = jp.sum(jp.abs(joint_speed * actuator_force))
        energy_penalty = 0.0015 * self.control_dt * power

        return (
            linear_tracking_reward
            + yaw_tracking_reward
            - height_penalty
            - tilt_penalty
            - reference_penalty
            - forward_correction_penalty
            - action_penalty
            - smooth_penalty
            - joint_speed_penalty
            - yaw_error_penalty
            - energy_penalty
        )

    def step(self, state0, control):
        action = jp.clip(control, -1.0, 1.0)

        q_ref = self.reference_at_phase(state0["phase"])
        base_q, _ = self._command_base_pose(q_ref, state0["command"])
        policy_scale = self._policy_scale_for_command(state0["command"])

        desired_q = jp.clip(
            base_q + action * policy_scale,
            self.ctrl_low,
            self.ctrl_high,
        )

        # 300 deg/s loaded-servo command rate -> max 6 deg per 20 ms tick.
        max_delta = self.SERVO_SPEED_RAD_S * self.control_dt
        commanded_q = state0["commanded_q"] + jp.clip(
            desired_q - state0["commanded_q"],
            -max_delta,
            max_delta,
        )
        commanded_q = jp.clip(commanded_q, self.ctrl_low, self.ctrl_high)

        data, _ = jax.lax.scan(
            lambda d0, _: (
                mjx.step(self.mjx_model, d0.replace(ctrl=commanded_q)),
                None,
            ),
            state0["data"],
            (),
            self.frame_skip,
        )

        new_step = state0["step"] + 1
        new_phase = jp.mod(
            state0["phase"] + self.control_dt / CYCLE_TIME,
            1.0,
        )
        body_pos = data.xpos[self.body_id]

        reward = self.compute_reward(state0, data, q_ref, action)

        roll, pitch, _ = self._rpy_from_quat(data.qpos[3:7])
        angle_limit = jp.deg2rad(50.0)
        fell = body_pos[2] < 0.045
        tipped = (
            (jp.abs(roll) > angle_limit)
            | (jp.abs(pitch) > angle_limit)
        )
        steps_limit_reached = new_step >= self.max_steps
        done = jp.where(
            fell | tipped | steps_limit_reached,
            1.0,
            0.0,
        )
        reward = reward - 20.0 * done

        rng_next, rng_command = jax.random.split(state0["rng"])
        sampled_command, sampled_hold = self._sample_command(rng_command)

        remaining = state0["command_steps_left"] - 1
        change_command = remaining <= 0
        next_command = jp.where(
            change_command,
            sampled_command,
            state0["command"],
        )
        next_hold = jp.where(
            change_command,
            sampled_hold,
            remaining,
        )

        obs = self.get_obs(
            data,
            new_phase,
            next_command,
            commanded_q,
        )

        new_state = dict(
            data=data,
            step=new_step,
            obs=obs,
            rng=rng_next,
            last_com=body_pos,
            phase=new_phase,
            command=next_command,
            command_steps_left=next_hold,
            commanded_q=commanded_q,
            last_action=action,
        )

        next_state = jax.lax.cond(
            done > 0.5,
            self.reset,
            lambda _: new_state,
            new_state["rng"],
        )

        val_reward = reward
        return (
            next_state,
            next_state["obs"],
            jp.array([reward]).reshape(-1),
            jp.array([done]).reshape(-1),
            val_reward,
        )
