"""Lightweight real-time viewer for a trained command-conditioned locomotion policy.

Unlike arena.py, this viewer does NOT use MJX for simulation. Native MuJoCo runs the
physics and rendering; JAX/Flax is used only for one small actor inference at 50 Hz.
This keeps the viewer smooth while reproducing the policy observation/action path used
in training.
"""

import argparse
import os
import time

# The actor is tiny for single-robot inference. Keep it on CPU so the viewer does not
# touch CUDA/MJX and remains independent from GPU rendering/training issues.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import serialization

from agent import ActorSimple_skip
from reference_gait import CYCLE_TIME, raised_stand_targets, reference_targets


OBS_SIZE = 46
CONTROL_DT = 0.020  # 50 Hz, exactly as in training / real controller
SERVO_SPEED_RAD_S = np.deg2rad(60.0) / 0.14
POLICY_RANGE_DEG = np.array([40.0, 55.0, 65.0] * 6, dtype=np.float32)
POLICY_SCALE = np.deg2rad(POLICY_RANGE_DEG).astype(np.float32)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_RENDER_FPS = 30
DEFAULT_REALTIME_SPEED = 1.0
DEFAULT_CAMERA_DISTANCE = 1.35
MAX_WALL_LAG_S = 0.12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smooth native-MuJoCo viewer for a trained hexapod policy"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/command_locomotion_interrupted",
        help=(
            "Checkpoint directory containing agent.flax and obs_norm_last.npz "
            "or obs_norm.npz"
        ),
    )
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        default=None,
        help=(
            "Persistent normalized command. If omitted, keyboard control is used. "
            "+VX forward, +VY left, +YAW left turn."
        ),
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_RENDER_FPS)
    parser.add_argument("--speed", type=float, default=DEFAULT_REALTIME_SPEED)
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=DEFAULT_CAMERA_DISTANCE,
        help="Tracking-camera distance in metres (default: 1.35)",
    )
    return parser.parse_args()


def disable_expensive_rendering(model, renderer):
    try:
        model.vis.quality.shadowsize = 0
    except Exception:
        pass
    try:
        model.vis.quality.offsamples = 1
    except Exception:
        pass

    for flag_name in (
        "mjRND_SHADOW",
        "mjRND_REFLECTION",
        "mjRND_SKYBOX",
        "mjRND_HAZE",
    ):
        try:
            renderer.scene.flags[getattr(mujoco.mjtRndFlag, flag_name)] = 0
        except Exception:
            pass


def _rpy_from_quat(q):
    qw, qx, qy, qz = [float(v) for v in q]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -0.999999, 0.999999))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def _forward_prior_strength(command):
    forward = np.clip(command[0], 0.0, 1.0)
    lateral_factor = 1.0 - 0.65 * np.clip(abs(command[1]), 0.0, 1.0)
    yaw_factor = 1.0 - 0.85 * np.clip(abs(command[2]), 0.0, 1.0)
    return float(np.clip(forward * lateral_factor * yaw_factor, 0.0, 1.0))


def build_observation(data, phase, command, commanded_q):
    roll, pitch, _ = _rpy_from_quat(data.qpos[3:7])

    # Match arena.py exactly: free-joint angular velocity components.
    gyro_xyz = np.asarray(data.qvel[3:6], dtype=np.float32)
    q_ref = np.asarray(reference_targets(phase), dtype=np.float32)
    phase_features = np.array(
        [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
        dtype=np.float32,
    )

    obs = np.concatenate(
        (
            np.asarray(command, dtype=np.float32),
            np.array([roll, pitch], dtype=np.float32),
            gyro_xyz,
            phase_features,
            q_ref,
            np.asarray(commanded_q, dtype=np.float32),
        )
    ).astype(np.float32)

    if obs.shape != (OBS_SIZE,):
        raise RuntimeError(f"Expected observation shape {(OBS_SIZE,)}, got {obs.shape}")
    return obs


def load_policy(checkpoint_dir):
    agent_path = os.path.join(checkpoint_dir, "agent.flax")
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Actor checkpoint not found: {agent_path}")

    norm_path = None
    for name in ("obs_norm_last.npz", "obs_norm.npz"):
        candidate = os.path.join(checkpoint_dir, name)
        if os.path.exists(candidate):
            norm_path = candidate
            break
    if norm_path is None:
        raise FileNotFoundError(
            f"No obs_norm_last.npz or obs_norm.npz found in {checkpoint_dir}"
        )

    actor = ActorSimple_skip(18, 1.0, -1.0, 512, 512)
    dummy_obs = jnp.ones((1, OBS_SIZE), dtype=jnp.float32)
    template = actor.init(jax.random.key(0), dummy_obs)["params"]

    with open(agent_path, "rb") as f:
        params = serialization.from_bytes(template, f.read())

    norm = np.load(norm_path)
    obs_mean = np.asarray(norm["obs_mean"], dtype=np.float32)
    obs_var = np.asarray(norm["obs_var"], dtype=np.float32)
    if obs_mean.shape != (OBS_SIZE,) or obs_var.shape != (OBS_SIZE,):
        raise ValueError(
            f"Normalization shape mismatch: mean={obs_mean.shape}, var={obs_var.shape}"
        )

    def deterministic_action(p, obs):
        mean, _ = actor.apply({"params": p}, obs)
        return jnp.tanh(mean)

    jit_action = jax.jit(deterministic_action)
    # Compile once before opening the real-time loop.
    jit_action(params, dummy_obs).block_until_ready()

    return params, jit_action, obs_mean, obs_var, agent_path, norm_path


def reset_robot(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0.0, 0.0, 0.110])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    stand = np.asarray(raised_stand_targets(), dtype=np.float64)
    data.qpos[7:25] = stand
    commanded_q = np.clip(
        stand,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    data.ctrl[:] = commanded_q
    mujoco.mj_forward(model, data)
    return commanded_q.copy()


def policy_control_step(
    model,
    data,
    commanded_q,
    phase,
    command,
    params,
    jit_action,
    obs_mean,
    obs_var,
):
    obs = build_observation(data, phase, command, commanded_q)
    obs_norm = np.clip(
        (obs - obs_mean) / np.sqrt(obs_var + 1e-8),
        -10.0,
        10.0,
    ).astype(np.float32)

    action = np.asarray(
        jit_action(params, jnp.asarray(obs_norm[None, :])),
        dtype=np.float32,
    )[0]
    action = np.clip(action, -1.0, 1.0)

    q_ref = np.asarray(reference_targets(phase), dtype=np.float64)
    stand = np.asarray(raised_stand_targets(), dtype=np.float64)
    prior = _forward_prior_strength(command)
    base_q = stand + prior * (q_ref - stand)

    desired_q = np.clip(
        base_q + action.astype(np.float64) * POLICY_SCALE.astype(np.float64),
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )

    max_delta = SERVO_SPEED_RAD_S * CONTROL_DT
    commanded_q = commanded_q + np.clip(
        desired_q - commanded_q,
        -max_delta,
        max_delta,
    )
    commanded_q = np.clip(
        commanded_q,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    data.ctrl[:] = commanded_q
    return commanded_q, action


def keyboard_command(pygame):
    keys = pygame.key.get_pressed()

    vx = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
    vy = float(keys[pygame.K_a]) - float(keys[pygame.K_d])
    yaw = float(keys[pygame.K_LEFT]) - float(keys[pygame.K_RIGHT])

    # Match joystick geometry: diagonal translation stays inside unit circle.
    mag = np.hypot(vx, vy)
    if mag > 1.0:
        vx /= mag
        vy /= mag

    return np.array([vx, vy, yaw], dtype=np.float32)


def main():
    import pygame
    from pygame.locals import QUIT

    args = parse_args()
    width = max(320, int(args.width))
    height = max(180, int(args.height))
    render_fps = max(5, int(args.fps))
    realtime_speed = max(0.05, float(args.speed))
    camera_distance = max(0.5, float(args.camera_distance))

    params, jit_action, obs_mean, obs_var, agent_path, norm_path = load_policy(
        args.checkpoint
    )

    model = mujoco.MjModel.from_xml_path("models/arena.xml")
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    model.opt.iterations = 4
    model.opt.ls_iterations = 4

    data = mujoco.MjData(model)
    commanded_q = reset_robot(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

    fixed_command = None
    if args.command is not None:
        fixed_command = np.clip(
            np.asarray(args.command, dtype=np.float32), -1.0, 1.0
        )
        mag = np.linalg.norm(fixed_command[:2])
        if mag > 1.0:
            fixed_command[:2] /= mag

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    try:
        model.vis.quality.shadowsize = 0
        model.vis.quality.offsamples = 1
    except Exception:
        pass

    renderer = mujoco.Renderer(model, width=width, height=height)
    disable_expensive_rendering(model, renderer)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = body_id
    camera.distance = camera_distance
    camera.azimuth = 135
    camera.elevation = -24

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Hexapod POLICY VIEW")
    clock = pygame.time.Clock()

    wall_origin = time.perf_counter()
    sim_origin = float(data.time)
    next_control_time = float(data.time)
    report_wall = wall_origin
    report_sim = float(data.time)

    action = np.zeros(18, dtype=np.float32)
    command = fixed_command if fixed_command is not None else np.zeros(3, dtype=np.float32)

    print("POLICY VIEW: native MuJoCo physics + CPU JAX actor, no MJX")
    print(f"Actor: {agent_path}")
    print(f"Norm:  {norm_path}")
    print(
        f"Window {width}x{height}, target {render_fps} FPS, control 50 Hz, "
        f"camera {camera_distance:.2f} m, realtime {realtime_speed:.2f}x"
    )
    if fixed_command is None:
        print("Controls: W/S forward/back, A/D left/right, LEFT/RIGHT yaw, SPACE reset, ESC exit")
    else:
        print(
            "Fixed command: "
            f"vx={fixed_command[0]:+.2f}, vy={fixed_command[1]:+.2f}, "
            f"yaw={fixed_command[2]:+.2f}; SPACE reset, ESC exit"
        )

    try:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    commanded_q = reset_robot(model, data)
                    action[:] = 0.0
                    now = time.perf_counter()
                    wall_origin = now
                    sim_origin = float(data.time)
                    next_control_time = float(data.time)
                    report_wall = now
                    report_sim = float(data.time)

            if fixed_command is None:
                command = keyboard_command(pygame)
            else:
                command = fixed_command

            now = time.perf_counter()
            target_sim_time = sim_origin + (now - wall_origin) * realtime_speed

            lag = target_sim_time - float(data.time)
            if lag > MAX_WALL_LAG_S:
                wall_origin = now - (float(data.time) - sim_origin) / realtime_speed
                target_sim_time = float(data.time)

            while float(data.time) + 0.5 * float(model.opt.timestep) < target_sim_time:
                if float(data.time) + 1e-9 >= next_control_time:
                    phase = ((next_control_time - sim_origin) / CYCLE_TIME) % 1.0
                    commanded_q, action = policy_control_step(
                        model,
                        data,
                        commanded_q,
                        phase,
                        command,
                        params,
                        jit_action,
                        obs_mean,
                        obs_var,
                    )
                    next_control_time += CONTROL_DT

                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=camera)
            disable_expensive_rendering(model, renderer)
            pixels = renderer.render()
            surface = pygame.surfarray.make_surface(np.swapaxes(pixels, 0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(render_fps)

            now_report = time.perf_counter()
            wall_dt = now_report - report_wall
            if wall_dt >= 1.0:
                sim_dt = float(data.time) - report_sim
                measured_speed = sim_dt / wall_dt if wall_dt > 0.0 else 0.0
                pos = data.xpos[body_id]
                _, _, yaw = _rpy_from_quat(data.qpos[3:7])
                max_policy_delta = float(np.max(np.abs(action * POLICY_RANGE_DEG)))
                print(
                    f"FPS={clock.get_fps():5.1f} realtime={measured_speed:4.2f}x "
                    f"cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.0f}) "
                    f"x={pos[0]:+.3f} y={pos[1]:+.3f} yaw={np.rad2deg(yaw):+.1f}deg "
                    f"max_policy_delta={max_policy_delta:.1f}deg"
                )
                report_wall = now_report
                report_sim = float(data.time)
    finally:
        renderer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
