"""Archived V2 SAC entry point. New laptop runs use train_laptop.py."""
import argparse
from queue import Queue

import jax
import jax.numpy as jnp

from agent import ActorSimple_skip
from arena_v2 import BattleArena
from buffer import BufferThread
from trainer_v2 import TrainingThread
from worker_v2 import WorkerThread


RUN_NAME = "command_locomotion_v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train command-conditioned SAC locomotion v2 for the hexapod"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Optional V2 checkpoint directory. Omit this for a completely fresh "
            "run; old V1 checkpoints should not be resumed into V2."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    devices = jax.devices()
    print(f"Available devices: {devices}")
    worker_device = devices[0]
    trainer_device = devices[1] if len(devices) > 1 else worker_device

    config = dict(
        run_name=RUN_NAME,
        seed=42,
        worker_device=worker_device,
        trainer_device=trainer_device,
        worker_batch_size=64,
        validation_batch_size=16,
        trainer_batch_size=256,
        buffer_size=1_000_000,
        random_steps_count=250,
        initial_alpha=1.0,
        autotune_alpha=True,
        tau=0.005,
        gamma=0.985,
        q_lr=0.0003,
        p_lr=0.0003,
        total_steps=5_000_000,
        warmup_steps=10_000,
        report_to_tensorboard=False,
        report_to_wandb=False,
    )

    _, worker_key, trainer_key, init_key = jax.random.split(
        jax.random.key(config["seed"]), 4
    )

    env = BattleArena()
    print(f"Observation shape: {env.observation_space_shape}")
    print(f"Action shape: {env.action_space_shape} (18 learned joint controls)")
    print(
        "Command convention: +vx forward, +vy left, +yaw left turn; "
        "vx/vy analog, yaw in {-1, 0, +1}."
    )
    print(
        f"Training targets: +/-{env.MAX_LINEAR_SPEED_M_S:.2f} m/s linear, "
        f"+/-{env.MAX_YAW_RATE_RAD_S:.2f} rad/s yaw."
    )
    print(
        "V2 policy authority: straight-forward V5 residual "
        f"+/-{env.FORWARD_POLICY_RANGE_DEG} deg; side/back/turn expands to "
        f"+/-{env.POLICY_RANGE_DEG} deg."
    )
    print(
        f"Loaded-servo command limit: "
        f"{jnp.rad2deg(env.SERVO_SPEED_RAD_S):.1f} deg/s "
        f"({jnp.rad2deg(env.SERVO_SPEED_RAD_S * env.control_dt):.1f} deg/tick)."
    )
    print(
        f"Batches: worker={config['worker_batch_size']}, "
        f"validation={config['validation_batch_size']}, "
        f"total_envs={config['worker_batch_size'] + config['validation_batch_size']}, "
        f"trainer={config['trainer_batch_size']}."
    )

    agent = ActorSimple_skip(
        env.action_space_shape[0],
        env.ctrlrange_high,
        env.ctrlrange_low,
        512,
        512,
    )

    state0 = env.reset(jax.random.key(0))
    observations_count = int(
        jax.numpy.prod(jax.numpy.array(state0["obs"].shape)).item()
    )
    agent_params = agent.init(
        init_key,
        jax.numpy.ones((1, observations_count)),
    )["params"]

    def deterministic_validation_action(params, obs, rng_key):
        del rng_key
        mean, _ = agent.apply({"params": params}, obs)
        action = jnp.tanh(mean)
        log_prob_placeholder = jnp.zeros((obs.shape[0], 1), dtype=obs.dtype)
        return action, log_prob_placeholder, action

    buffer_thread = BufferThread(
        config["buffer_size"],
        config["worker_batch_size"],
        env.observation_space_shape,
        env.action_space_shape,
    )

    worker_agent_queue = Queue(maxsize=8)

    worker_thread = WorkerThread(
        config=config,
        env=env,
        rng_key=worker_key,
        result_queue=buffer_thread.input_queue,
        agent_queue=worker_agent_queue,
        agent_config=dict(
            trainee_func=agent.get_action,
            validation_agent_func=deterministic_validation_action,
            ref_agent_params=[],
            validation_agent_params=[],
        ),
        name="Hexapod-Command-Locomotion-V2-Worker-0",
    )

    trainer_thread = TrainingThread(
        config=config,
        buffer_thread=buffer_thread,
        agent_queue=worker_agent_queue,
        agent=agent,
        agent_params=agent_params,
        rng_key=trainer_key,
        observation_space_shape=env.observation_space_shape,
        action_space_shape=env.action_space_shape,
        worker_thread=worker_thread,
        name="Hexapod-Command-Locomotion-V2-Trainer",
    )

    if args.resume:
        print(f"Resuming V2 training from: {args.resume}")
        trainer_thread.load_state(args.resume)
        norm_path = worker_thread.load_obs_norm(args.resume)
        worker_thread.step = trainer_thread.current_step
        print(f"Restored observation normalization from: {norm_path}")
        print(
            "Note: network/optimizer/alpha/target-Q/norm/step are restored; "
            "the replay buffer itself starts fresh."
        )
    else:
        print("Starting V2 locomotion training FROM SCRATCH.")
        print("Fresh observation normalization; no V1 checkpoint is reused.")

    buffer_thread.start()
    worker_thread.start()
    trainer_thread.start()

    interrupted = False
    try:
        trainer_thread.join()
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping training...")
    finally:
        worker_thread.running = False
        trainer_thread.running = False
        buffer_thread.stop()

    save_path = (
        "checkpoints/command_locomotion_v2_interrupted"
        if interrupted
        else "checkpoints/command_locomotion_v2_final"
    )
    trainer_thread.save_state(save_path)
    print(f"V2 command-conditioned model saved to: {save_path}")


if __name__ == "__main__":
    main()
