import argparse
from queue import Queue

import jax

from agent import ActorSimple_skip
from arena import BattleArena
from buffer import BufferThread
from trainer import TrainingThread
from worker import WorkerThread


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train command-conditioned SAC locomotion for the hexapod"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Optional checkpoint directory created by THIS command-conditioned "
            "environment. Do not resume the old +/-10 degree residual model."
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
        seed=42,
        worker_device=worker_device,
        trainer_device=trainer_device,
        worker_batch_size=256,
        validation_batch_size=64,
        trainer_batch_size=1024,
        buffer_size=1_000_000,
        random_steps_count=1000,
        initial_alpha=1.0,
        autotune_alpha=True,
        tau=0.005,
        gamma=0.985,
        q_lr=0.0003,
        p_lr=0.0003,
        # Omnidirectional command tracking is a substantially broader task than
        # the previous forward-only residual problem, so give SAC more budget.
        total_steps=5_000_000,
        warmup_steps=10_000,
        report_to_tensorboard=True,
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
        "V5 TURBO remains a strong prior for straight-forward motion, "
        "but side/back/turn motion is learned with much wider joint authority."
    )

    # Keep the existing 512x512 actor as requested.
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
            validation_agent_func=agent.get_action,
            ref_agent_params=[],
            validation_agent_params=[],
        ),
        name="Hexapod-Command-Locomotion-Worker-0",
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
        name="Hexapod-Command-Locomotion-Trainer",
    )

    # Observation size is still 46, but action semantics/reward changed enough
    # that the old forward residual checkpoint should not be reused.
    if args.resume:
        print(f"Resuming command-conditioned training from: {args.resume}")
        trainer_thread.load_state(args.resume)
        worker_thread.step = trainer_thread.current_step
    else:
        print("Starting command-conditioned locomotion training from scratch.")
        print(
            "Every parallel simulation receives changing vx/vy/yaw commands; "
            "straight-forward zero action still has the accepted V5 prior."
        )

    buffer_thread.start()
    worker_thread.start()
    trainer_thread.start()

    interrupted = False
    try:
        trainer_thread.join()
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping training...")
        worker_thread.running = False
        trainer_thread.running = False

    save_path = (
        "checkpoints/command_locomotion_interrupted"
        if interrupted
        else "checkpoints/command_locomotion_final"
    )
    trainer_thread.save_state(save_path)
    print(f"Command-conditioned model saved to: {save_path}")


if __name__ == "__main__":
    main()
