"""Smooth viewer matched to command locomotion V2.

Reuses the native-MuJoCo realtime viewer, but patches its controller so the visual
policy path exactly uses V2's command-dependent residual authority and 300 deg/s
loaded-servo command limit.
"""

import numpy as np
import jax.numpy as jnp

import policy_view as pv
from arena_v2 import BattleArena


FULL_SCALE = np.deg2rad(
    np.array(BattleArena.POLICY_RANGE_DEG * 6, dtype=np.float32)
)
FORWARD_SCALE = np.deg2rad(
    np.array(BattleArena.FORWARD_POLICY_RANGE_DEG * 6, dtype=np.float32)
)

_original_parse_args = pv.parse_args


def _parse_args_v2():
    args = _original_parse_args()
    if args.checkpoint == "checkpoints/command_locomotion_interrupted":
        args.checkpoint = "checkpoints/command_locomotion_v2_interrupted"
    return args


def _policy_control_step_v2(
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
    obs = pv.build_observation(data, phase, command, commanded_q)
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

    q_ref = np.asarray(pv.reference_targets(phase), dtype=np.float64)
    stand = np.asarray(pv.raised_stand_targets(), dtype=np.float64)
    prior = pv._forward_prior_strength(command)
    base_q = stand + prior * (q_ref - stand)

    policy_scale = (
        FULL_SCALE * (1.0 - prior) + FORWARD_SCALE * prior
    ).astype(np.float64)

    desired_q = np.clip(
        base_q + action.astype(np.float64) * policy_scale,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )

    max_delta = BattleArena.SERVO_SPEED_RAD_S * pv.CONTROL_DT
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


pv.parse_args = _parse_args_v2
pv.policy_control_step = _policy_control_step_v2
pv.SERVO_SPEED_RAD_S = BattleArena.SERVO_SPEED_RAD_S


if __name__ == "__main__":
    pv.main()
