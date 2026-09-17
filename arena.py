import argparse
import os
import time

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx import Data
import numpy as np

from reference_gait import CYCLE_TIME, build_reference_table, raised_stand_targets


class BattleArena:
    """Command-conditioned locomotion environment for the real hexapod.

    Control command uses the robot/body frame and matches the intended gamepad:
      command[0] = vx_cmd: +forward, -backward
      command[1] = vy_cmd: +left,    -right
      command[2] = yaw_cmd: +turn left, -turn right

    vx/vy are analog values in [-1, 1]. yaw is trained as the D-pad-like
    discrete command {-1, 0, +1}.

    Actor observation deliberately contains only signals that can exist on the
    physical robot after adding one IMU:
      command (joystick)                 3
      roll, pitch (IMU estimate)         2
      angular rate xyz (IMU gyro)        3
      sin/cos gait phase                 2
      forward-V5 reference targets      18
      last commanded servo targets      18
                                           = 46 values

    MuJoCo may use privileged state for reward only. The actor never receives
    actual joint angles, joint velocities, contacts, actuator forces, or true
    body velocity because the MG996R servos provide no feedback and the first
    real robot version has no velocity sensor.

    V5 TURBO is a prior, not a cage:
      * straight-forward commands start from the accepted V5 reference;
      * lateral/backward/turn commands progressively fall back toward the raised
        stand, giving the policy broad authority to discover its own leg motion.
    """

    OBS_SIZE = 46
    SERVO_SPEED_RAD_S = np.deg2rad(60.0) / 0.14

    # Normalized joystick command -> physical training target.
    MAX_LINEAR_SPEED_M_S = 0.30
    MAX_YAW_RATE_RAD_S = 1.20

    # Policy authority around the command-dependent base pose.
    # Repeated per leg as [coxa, femur, tibia].
    POLICY_RANGE_DEG = (40.0, 55.0, 65.0)

    # A command is held long enough to span at least about one V5 cycle.
    COMMAND_HOLD_MIN_STEPS = 75   # 1.5 s at 50 Hz
    COMMAND_HOLD_MAX_STEPS = 200  # 4.0 s at 50 Hz

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("models/arena.xml")

        # Keep the existing fast MJX settings. The XML supplies the real
        # geometry, masses, contacts, torque limits and 2 ms physics timestep.
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        self.model.opt.iterations = 4
        self.model.opt.ls_iterations = 4

        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)

        # 10 * 2 ms = 20 ms = 50 Hz, matching the real controller loop.
        self.frame_skip = 10
        self.control_dt = float(self.model.opt.timestep) * self.frame_skip

        self.agent_count = 1
        self.action_space_shape = (self.mjx_model.nu,)
        self.observation_space_shape = (self.OBS_SIZE,)
        self.ctrlrange_high = 1.0
        self.ctrlrange_low = -1.0
        self.max_steps = 1000  # 20 s episodes

        if self.mjx_model.nu != 18:
            raise RuntimeError(f"Expected 18 actuators, found {self.mjx_model.nu}")

        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "body"
        )
        if self.body_id < 0:
            raise RuntimeError("Body 'body' not found in models/arena.xml")

        # Exact accepted V5-TURBO cycle. It remains the forward-motion prior.
        self.reference_table = jp.asarray(build_reference_table(2048))
        self.reference_samples = int(self.reference_table.shape[0])
        self.raised_stand = jp.asarray(raised_stand_targets())

        self.ctrl_low = jp.asarray(self.model.actuator_ctrlrange[:, 0])
        self.ctrl_high = jp.asarray(self.model.actuator_ctrlrange[:, 1])

        one_leg_scale = jp.deg2rad(
            jp.array(self.POLICY_RANGE_DEG, dtype=jp.float32)
        )
        self.policy_scale = jp.tile(one_leg_scale, 6)

    def reference_at_phase(self, phase):
        phase = jp.mod(phase, 1.0)
        position = phase * self.reference_samples
        base = jp.floor(position)
        i0 = base.astype(jp.int32) % self.reference_samples
        i1 = (i0 + 1) % self.reference_samples
        frac = position - base
        return (
            self.reference_table[i0] * (1.0 - frac)
            + self.reference_table[i1] * frac
        )

    @staticmethod
    def _rpy_from_quat(q):
        qw, qx, qy, qz = q

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = jp.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = jp.arcsin(jp.clip(sinp, -0.999999, 0.999999))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = jp.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _wrap_angle(angle):
        return jp.arctan2(jp.sin(angle), jp.cos(angle))

    def _sample_command(self, rng):
        """Sample commands with the same semantics as the planned gamepad.

        10% idle, 15% pure in-place turn, 75% analog translation.
        Translation commands may also include a D-pad-like yaw command so the
        policy learns combined translation+rotation as well.
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
        angle = jax.random.uniform(k_angle, minval=-jp.pi, maxval=jp.pi)

        # Avoid spending most training time very close to the stick centre.
        radius = 0.20 + 0.80 * jp.sqrt(jax.random.uniform(k_radius))
        vx = radius * jp.cos(angle)
        vy = radius * jp.sin(angle)

        yaw_u = jax.random.uniform(k_yaw)
        translation_yaw = jp.where(
            yaw_u < 0.15,
            1.0,
            jp.where(yaw_u > 0.85, -1.0, 0.0),
        )
        turn_sign = jp.where(jax.random.bernoulli(k_turn), 1.0, -1.0)

        translation = jp.array([vx, vy, translation_yaw], dtype=jp.float32)
        pure_turn = jp.array([0.0, 0.0, turn_sign], dtype=jp.float32)
        idle = jp.zeros(3, dtype=jp.float32)

        command = jp.where(
            mode < 0.10,
            idle,
            jp.where(mode < 0.25, pure_turn, translation),
        )

        hold_steps = jax.random.randint(
            k_hold,
            (),
            self.COMMAND_HOLD_MIN_STEPS,
            self.COMMAND_HOLD_MAX_STEPS + 1,
            dtype=jp.int32,
        )
        return command, hold_steps

    @staticmethod
    def _forward_prior_strength(command):
        """Return how strongly the accepted forward V5 should influence a command."""
        forward = jp.clip(command[0], 0.0, 1.0)
        lateral_factor = 1.0 - 0.65 * jp.clip(jp.abs(command[1]), 0.0, 1.0)
        yaw_factor = 1.0 - 0.85 * jp.clip(jp.abs(command[2]), 0.0, 1.0)
        return jp.clip(forward * lateral_factor * yaw_factor, 0.0, 1.0)

    def get_obs(self, data: Data, phase, command, commanded_q):
        roll, pitch, _ = self._rpy_from_quat(data.qpos[3:7])

        # This is the simulation counterpart of a body-mounted IMU gyro.
        gyro_xyz = data.qvel[3:6]

        # Keep raw forward V5 in observation as a phase/motion prior. The actor
        # is not forced to follow it for non-forward commands.
        q_ref = self.reference_at_phase(phase)
        phase_features = jp.array([
            jp.sin(2.0 * jp.pi * phase),
            jp.cos(2.0 * jp.pi * phase),
        ])

        return jp.concatenate((
            command,
            jp.array([roll, pitch]),
            gyro_xyz,
            phase_features,
            q_ref,
            commanded_q,
        ))

    def reset(self, rng: jax.Array):
        data = mjx.make_data(self.mjx_model)
        rng_joint, rng_vel, rng_command, rng_next = jax.random.split(rng, 4)

        qpos = data.qpos
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, 0.110]))
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))

        # Hidden +/-1 degree shaft error improves robustness without exposing
        # impossible servo feedback to the actor.
        joint_noise = jax.random.uniform(
            rng_joint,
            (18,),
            minval=-jp.deg2rad(1.0),
            maxval=jp.deg2rad(1.0),
        )
        initial_q = jp.clip(
            self.raised_stand + joint_noise,
            self.ctrl_low,
            self.ctrl_high,
        )
        qpos = qpos.at[7:25].set(initial_q)

        qvel = jax.random.uniform(
            rng_vel,
            (self.mjx_model.nv,),
            minval=-0.005,
            maxval=0.005,
        )
        qvel = qvel.at[0:6].set(jp.zeros(6))

        commanded_q = jp.clip(
            self.raised_stand,
            self.ctrl_low,
            self.ctrl_high,
        )
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=commanded_q)
        data = mjx.forward(self.mjx_model, data)

        phase = jp.array(0.0, dtype=jp.float32)
        command, command_steps_left = self._sample_command(rng_command)
        obs = self.get_obs(data, phase, command, commanded_q)
        body_pos = data.xpos[self.body_id]

        return dict(
            data=data,
            step=jp.array(0, dtype=jp.int32),
            obs=obs,
            rng=rng_next,
            last_com=body_pos,
            phase=phase,
            command=command,
            command_steps_left=command_steps_left,
            commanded_q=commanded_q,
            last_action=jp.zeros(18, dtype=jp.float32),
        )

    def _command_base_pose(self, q_ref, command):
        prior = self._forward_prior_strength(command)
        return self.raised_stand + prior * (q_ref - self.raised_stand), prior

    def compute_reward(self, old_state, new_data, q_ref, action):
        body_pos = new_data.xpos[self.body_id]
        old_body_pos = old_state["data"].xpos[self.body_id]

        roll, pitch, yaw = self._rpy_from_quat(new_data.qpos[3:7])
        _, _, old_yaw = self._rpy_from_quat(old_state["data"].qpos[3:7])

        # Privileged true velocity is used ONLY for training reward.
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
        yaw_error_sq = (yaw_rate - target_yaw_rate) ** 2

        # Dense tracking rewards: standing still is good only when commanded.
        linear_tracking_reward = 6.0 * jp.exp(
            -velocity_error_sq / (0.20 ** 2)
        )
        yaw_tracking_reward = 2.0 * jp.exp(
            -yaw_error_sq / (0.70 ** 2)
        )

        height_penalty = 5.0 * ((body_pos[2] - 0.105) ** 2)
        tilt_penalty = 1.0 * (roll * roll + pitch * pitch)

        # Privileged joint/force state is reward-only, never actor input.
        actual_q = new_data.qpos[7:25]
        joint_speed = new_data.qvel[6:24]
        actuator_force = new_data.actuator_force

        # V5 tracking is deliberately weak and only active where V5 is relevant.
        forward_prior = self._forward_prior_strength(command)
        reference_penalty = (
            0.35 * forward_prior * jp.mean((actual_q - q_ref) ** 2)
        )

        # Small regularizers; they must not overpower command tracking.
        action_penalty = 0.015 * jp.mean(action ** 2)
        smooth_penalty = 0.06 * jp.mean(
            (action - old_state["last_action"]) ** 2
        )

        power = jp.sum(jp.abs(joint_speed * actuator_force))
        energy_penalty = 0.0015 * self.control_dt * power

        return (
            linear_tracking_reward
            + yaw_tracking_reward
            - height_penalty
            - tilt_penalty
            - reference_penalty
            - action_penalty
            - smooth_penalty
            - energy_penalty
        )

    def step(self, state0, control):
        action = jp.clip(control, -1.0, 1.0)

        q_ref = self.reference_at_phase(state0["phase"])
        base_q, _ = self._command_base_pose(q_ref, state0["command"])

        # This is no longer +/-10 degree residual control.
        # Straight forward + zero action reproduces V5. For side/back/turn the
        # base moves toward raised stand and the policy gets broad joint authority.
        desired_q = jp.clip(
            base_q + action * self.policy_scale,
            self.ctrl_low,
            self.ctrl_high,
        )

        # Software command-rate limit mirrors the MG996R speed limit.
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

        # Change joystick command every 1.5-4.0 s independently in each parallel
        # simulation. Current-transition reward uses state0["command"].
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

        # Validation measures command tracking, not raw +X distance.
        val_reward = reward

        return (
            next_state,
            next_state["obs"],
            jp.array([reward]).reshape(-1),
            jp.array([done]).reshape(-1),
            val_reward,
        )


def normalize_obs(obs, mean, var):
    return jp.clip(
        (obs - mean) / jp.sqrt(var + 1e-8),
        -10.0,
        10.0,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command-conditioned hexapod locomotion visualization"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Locomotion actor .flax. Omit to visualize the forward V5 prior.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        help="Observation normalization .npz for the actor.",
    )
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        default=None,
        help=(
            "Fixed normalized command for visualization. "
            "Body frame: +VX forward, +VY left, +YAW turn left."
        ),
    )
    return parser.parse_args()


def _load_actor(actor, dummy_obs, agent_path):
    if not agent_path:
        return None

    from flax import serialization

    try:
        with open(agent_path, "rb") as f:
            raw = f.read()
        template = actor.init(jax.random.key(0), dummy_obs)["params"]
        params = serialization.from_bytes(template, raw)
        actor.apply({"params": params}, dummy_obs)
        return jax.device_put(params)
    except Exception as exc:
        print(f"Could not load actor '{agent_path}': {exc}")
        print("Falling back to forward V5 prior.")
        return None


def _load_norm(path, obs_size):
    mean = jp.zeros(obs_size)
    var = jp.ones(obs_size)
    if not path or not os.path.exists(path):
        return mean, var

    values = np.load(path)
    if len(values["obs_mean"]) != obs_size:
        print(
            f"Norm shape mismatch: expected {obs_size}; "
            "using identity normalization."
        )
        return mean, var
    return jp.asarray(values["obs_mean"]), jp.asarray(values["obs_var"])


def _force_command(env, state, command):
    """Override command in the interactive viewer without changing training."""
    command = jp.clip(
        jp.asarray(command, dtype=jp.float32),
        -1.0,
        1.0,
    )
    state["command"] = command
    state["command_steps_left"] = jp.array(1_000_000, dtype=jp.int32)
    state["obs"] = env.get_obs(
        state["data"],
        state["phase"],
        command,
        state["commanded_q"],
    )
    return state


def main():
    import pygame
    from pygame.locals import QUIT
    from agent import ActorSimple_skip

    args = parse_args()
    env = BattleArena()

    print("Command-conditioned RL hexapod")
    print(
        f"Observation size: {env.observation_space_shape[0]} "
        "(real-robot signals only)"
    )
    print(f"Reference cycle: {CYCLE_TIME:.2f} s")
    print(
        "Policy range per joint: "
        f"coxa +/-{env.POLICY_RANGE_DEG[0]:.0f} deg, "
        f"femur +/-{env.POLICY_RANGE_DEG[1]:.0f} deg, "
        f"tibia +/-{env.POLICY_RANGE_DEG[2]:.0f} deg"
    )
    print(
        f"Command targets: {env.MAX_LINEAR_SPEED_M_S:.2f} m/s linear, "
        f"{env.MAX_YAW_RATE_RAD_S:.2f} rad/s yaw"
    )

    actor = ActorSimple_skip(
        env.action_space_shape[0],
        env.ctrlrange_high,
        env.ctrlrange_low,
        512,
        512,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.key(42))

    dummy_obs = jp.expand_dims(state["obs"], 0)
    params = _load_actor(actor, dummy_obs, args.agent)
    obs_mean, obs_var = _load_norm(
        args.norm,
        env.observation_space_shape[0],
    )

    if args.command is not None:
        fixed_command = np.clip(
            np.asarray(args.command, dtype=np.float32),
            -1.0,
            1.0,
        )
    elif params is None:
        # No actor -> preserve the old useful reference-only forward demo.
        fixed_command = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        fixed_command = None

    if params is None:
        print("Mode: FORWARD V5 PRIOR (policy action = 0)")
    else:
        print(f"Mode: COMMAND-CONDITIONED ACTOR ({args.agent})")
        jit_action = jax.jit(actor.get_action)

    if fixed_command is not None:
        print(
            "Fixed command: "
            f"vx={fixed_command[0]:+.2f}, "
            f"vy={fixed_command[1]:+.2f}, "
            f"yaw={fixed_command[2]:+.2f}"
        )

    width, height = 1280, 720
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, width=width, height=height)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.body_id
    camera.distance = 0.8
    camera.azimuth = 135
    camera.elevation = -25

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Hexapod: command-conditioned locomotion")
    clock = pygame.time.Clock()
    inference_key = jax.random.key(7)
    frame = 0

    action_scale_deg = np.tile(
        np.asarray(env.POLICY_RANGE_DEG, dtype=np.float32),
        6,
    )

    try:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    state = jit_reset(jax.random.key(time.time_ns()))

            if fixed_command is not None:
                state = _force_command(env, state, fixed_command)

            if params is None:
                action = jp.zeros(18)
            else:
                obs = jp.expand_dims(state["obs"], 0)
                obs_norm = normalize_obs(obs, obs_mean, obs_var)
                inference_key, action_key = jax.random.split(inference_key)
                # Third return is tanh(mean): deterministic visualization action.
                _, _, mean_action = jit_action(params, obs_norm, action_key)
                action = jp.squeeze(mean_action)

            state, _, reward, _, _ = jit_step(state, action)

            mjx.get_data_into(env.data, env.model, state["data"])
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=camera)
            pixels = renderer.render()
            screen.blit(
                pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1)),
                (0, 0),
            )

            if frame % 25 == 0:
                action_np = np.asarray(jax.device_get(action))
                max_policy_delta = float(
                    np.max(np.abs(action_np * action_scale_deg))
                )
                command_np = np.asarray(jax.device_get(state["command"]))
                print(
                    f"step={int(state['step']):4d} "
                    f"phase={float(state['phase']):.3f} "
                    f"cmd=({command_np[0]:+.2f},"
                    f"{command_np[1]:+.2f},"
                    f"{command_np[2]:+.0f}) "
                    f"x={float(state['last_com'][0]):+.3f} "
                    f"y={float(state['last_com'][1]):+.3f} "
                    f"reward={float(reward[0]):+.3f} "
                    f"max_policy_delta={max_policy_delta:.1f}deg"
                )

            pygame.display.flip()
            clock.tick(50)
            frame += 1
    finally:
        renderer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
