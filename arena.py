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
    """Residual-RL environment for the real six-legged robot.

    Actor observation is deliberately restricted to signals that can exist on the
    physical robot after adding one IMU:
      command (joystick)                 3
      roll, pitch (IMU estimate)         2
      angular rate xyz (IMU gyro)        3
      sin/cos gait phase                 2
      reference joint targets           18
      last commanded servo targets      18
                                           = 46 values

    Reward may use privileged MuJoCo state during training. The actor never sees
    actual joint angles, joint velocities, contacts or actuator forces because the
    MG996R servos provide no feedback to the Mega in the real robot.
    """

    OBS_SIZE = 46
    RESIDUAL_LIMIT_DEG = 10.0
    SERVO_SPEED_RAD_S = np.deg2rad(60.0) / 0.14

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("models/arena.xml")

        # Keep the existing fast MJX settings. The new XML still supplies the real
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
        self.max_steps = 1000

        if self.mjx_model.nu != 18:
            raise RuntimeError(f"Expected 18 actuators, found {self.mjx_model.nu}")

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body")
        if self.body_id < 0:
            raise RuntimeError("Body 'body' not found in models/arena.xml")

        # Precompute one exact V5-TURBO gait cycle and interpolate it inside JIT.
        self.reference_table = jp.asarray(build_reference_table(2048))
        self.reference_samples = int(self.reference_table.shape[0])
        self.raised_stand = jp.asarray(raised_stand_targets())

        # SAC output is a residual, not a direct servo target.
        self.residual_scale = jp.full((18,), jp.deg2rad(self.RESIDUAL_LIMIT_DEG))
        self.ctrl_low = jp.asarray(self.model.actuator_ctrlrange[:, 0])
        self.ctrl_high = jp.asarray(self.model.actuator_ctrlrange[:, 1])

        # Initial task is forward motion. The command is already part of observation
        # so joystick-conditioned speed/turn training can be added without changing
        # the physical inference interface later.
        self.default_command = jp.array([1.0, 0.0, 0.0], dtype=jp.float32)

    def reference_at_phase(self, phase):
        phase = jp.mod(phase, 1.0)
        position = phase * self.reference_samples
        base = jp.floor(position)
        i0 = base.astype(jp.int32) % self.reference_samples
        i1 = (i0 + 1) % self.reference_samples
        frac = position - base
        return self.reference_table[i0] * (1.0 - frac) + self.reference_table[i1] * frac

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

    def get_obs(self, data: Data, phase, command, commanded_q):
        roll, pitch, _ = self._rpy_from_quat(data.qpos[3:7])
        gyro_xyz = data.qvel[3:6]
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
        rng_joint, rng_vel, rng_next = jax.random.split(rng, 3)

        qpos = data.qpos
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, 0.110]))
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))

        # Hidden +/-1 degree shaft error improves robustness without giving the actor
        # servo feedback that the real MG996R cannot provide.
        joint_noise = jax.random.uniform(
            rng_joint,
            (18,),
            minval=-jp.deg2rad(1.0),
            maxval=jp.deg2rad(1.0),
        )
        initial_q = jp.clip(self.raised_stand + joint_noise, self.ctrl_low, self.ctrl_high)
        qpos = qpos.at[7:25].set(initial_q)

        qvel = jax.random.uniform(rng_vel, (self.mjx_model.nv,), minval=-0.005, maxval=0.005)
        qvel = qvel.at[0:6].set(jp.zeros(6))

        commanded_q = jp.clip(self.raised_stand, self.ctrl_low, self.ctrl_high)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=commanded_q)
        data = mjx.forward(self.mjx_model, data)

        phase = jp.array(0.0, dtype=jp.float32)
        command = self.default_command
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
            commanded_q=commanded_q,
            last_action=jp.zeros(18, dtype=jp.float32),
        )

    def compute_reward(self, old_state, new_data, q_ref, action):
        body_pos = new_data.xpos[self.body_id]
        old_body_pos = old_state["data"].xpos[self.body_id]
        velocity_x = (body_pos[0] - old_body_pos[0]) / self.control_dt

        roll, pitch, yaw = self._rpy_from_quat(new_data.qpos[3:7])

        # Privileged training-only state: legal for reward, forbidden from actor obs.
        actual_q = new_data.qpos[7:25]
        joint_speed = new_data.qvel[6:24]
        actuator_force = new_data.actuator_force

        forward_reward = 12.0 * velocity_x
        sideways_penalty = 15.0 * (body_pos[1] ** 2)
        height_penalty = 4.0 * ((body_pos[2] - 0.105) ** 2)
        tilt_penalty = 0.6 * (roll * roll + pitch * pitch)
        yaw_penalty = 1.5 * (yaw * yaw)

        tracking_penalty = 0.8 * jp.mean((actual_q - q_ref) ** 2)
        residual_penalty = 0.08 * jp.mean(action ** 2)
        smooth_penalty = 0.04 * jp.mean((action - old_state["last_action"]) ** 2)

        power = jp.sum(jp.abs(joint_speed * actuator_force))
        energy_penalty = 0.002 * self.control_dt * power

        return (
            forward_reward
            - sideways_penalty
            - height_penalty
            - tilt_penalty
            - yaw_penalty
            - tracking_penalty
            - residual_penalty
            - smooth_penalty
            - energy_penalty
        )

    @staticmethod
    def validation_reward(old_com, new_com):
        return new_com[0] - old_com[0]

    def step(self, state0, control):
        action = jp.clip(control, -1.0, 1.0)

        q_ref = self.reference_at_phase(state0["phase"])
        desired_q = jp.clip(
            q_ref + action * self.residual_scale,
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
            lambda d0, _: (mjx.step(self.mjx_model, d0.replace(ctrl=commanded_q)), None),
            state0["data"],
            (),
            self.frame_skip,
        )

        new_step = state0["step"] + 1
        new_phase = jp.mod(state0["phase"] + self.control_dt / CYCLE_TIME, 1.0)
        body_pos = data.xpos[self.body_id]

        reward = self.compute_reward(state0, data, q_ref, action)

        roll, pitch, _ = self._rpy_from_quat(data.qpos[3:7])
        angle_limit = jp.deg2rad(50.0)
        fell = body_pos[2] < 0.045
        tipped = (jp.abs(roll) > angle_limit) | (jp.abs(pitch) > angle_limit)
        steps_limit_reached = new_step >= self.max_steps
        done = jp.where(fell | tipped | steps_limit_reached, 1.0, 0.0)
        reward = reward - 20.0 * done

        obs = self.get_obs(data, new_phase, state0["command"], commanded_q)
        val_reward = self.validation_reward(state0["last_com"], body_pos)

        new_state = dict(
            data=data,
            step=new_step,
            obs=obs,
            rng=state0["rng"],
            last_com=body_pos,
            phase=new_phase,
            command=state0["command"],
            commanded_q=commanded_q,
            last_action=action,
        )

        next_state = jax.lax.cond(
            done > 0.5,
            self.reset,
            lambda _: new_state,
            new_state["rng"],
        )

        return (
            next_state,
            next_state["obs"],
            jp.array([reward]).reshape(-1),
            jp.array([done]).reshape(-1),
            val_reward,
        )


def normalize_obs(obs, mean, var):
    return jp.clip((obs - mean) / jp.sqrt(var + 1e-8), -10.0, 10.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Residual-RL hexapod visualization")
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Residual actor .flax. Omit to run pure reference gait.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        help="Observation normalization .npz for the residual actor.",
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
        print(f"Could not load residual actor '{agent_path}': {exc}")
        print("Falling back to reference gait with ZERO neural correction.")
        return None


def _load_norm(path, obs_size):
    mean = jp.zeros(obs_size)
    var = jp.ones(obs_size)
    if not path or not os.path.exists(path):
        return mean, var

    values = np.load(path)
    if len(values["obs_mean"]) != obs_size:
        print(f"Norm shape mismatch: expected {obs_size}; using identity normalization.")
        return mean, var
    return jp.asarray(values["obs_mean"]), jp.asarray(values["obs_var"])


def main():
    import pygame
    from pygame.locals import QUIT
    from agent import ActorSimple_skip

    args = parse_args()
    env = BattleArena()

    print("Residual-RL hexapod")
    print(f"Observation size: {env.observation_space_shape[0]} (real-robot signals only)")
    print(f"Reference cycle: {CYCLE_TIME:.2f} s")
    print(f"Residual limit: +/-{env.RESIDUAL_LIMIT_DEG:.1f} deg per joint")

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
    obs_mean, obs_var = _load_norm(args.norm, env.observation_space_shape[0])

    if params is None:
        print("Mode: REFERENCE ONLY (neural residual = 0)")
    else:
        print(f"Mode: REFERENCE + RESIDUAL ACTOR ({args.agent})")
        jit_action = jax.jit(actor.get_action)

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
    pygame.display.set_caption("Hexapod: reference gait + residual RL")
    clock = pygame.time.Clock()
    inference_key = jax.random.key(7)
    frame = 0

    try:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    state = jit_reset(jax.random.key(time.time_ns()))

            if params is None:
                residual = jp.zeros(18)
            else:
                obs = jp.expand_dims(state["obs"], 0)
                obs_norm = normalize_obs(obs, obs_mean, obs_var)
                inference_key, action_key = jax.random.split(inference_key)
                # Third return is tanh(mean): deterministic action for visualization.
                _, _, mean_action = jit_action(params, obs_norm, action_key)
                residual = jp.squeeze(mean_action)

            state, _, reward, _, _ = jit_step(state, residual)

            mjx.get_data_into(env.data, env.model, state["data"])
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=camera)
            pixels = renderer.render()
            screen.blit(pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1)), (0, 0))

            if frame % 25 == 0:
                residual_np = np.asarray(jax.device_get(residual))
                max_correction = float(np.max(np.abs(residual_np))) * env.RESIDUAL_LIMIT_DEG
                print(
                    f"step={int(state['step']):4d} "
                    f"phase={float(state['phase']):.3f} "
                    f"x={float(state['last_com'][0]):+.3f} "
                    f"reward={float(reward[0]):+.3f} "
                    f"max_residual={max_correction:.1f}deg"
                )

            pygame.display.flip()
            clock.tick(50)
            frame += 1
    finally:
        renderer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
