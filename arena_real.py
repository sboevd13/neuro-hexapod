"""Real-robot environment wrapper.

Uses models/arena.xml from the real-robot-match branch, but keeps the legacy
arena.py implementation intact.  The only environment-level change here is
the reset pose: the educate-center robot's neutral HOME_Z is -80 mm, so its
body/coxas start 82 mm above the floor (2 mm clearance).

Run a trained policy for visual inspection with:

    python arena_real.py --agent PATH/agent.flax --norm PATH/obs_norm_last.npz

For training on this model use sac_real.py.
"""

import jax
from jax import numpy as jp
from mujoco import mjx

import arena as _legacy_arena


REAL_BODY_START_Z = 0.082  # 80 mm HOME leg height + 2 mm initial clearance


class BattleArena(_legacy_arena.BattleArena):
    """BattleArena with the reset pose matched to the physical hexapod."""

    def __init__(self) -> None:
        super().__init__()
        # Firmware control loop is exactly 50 Hz. models/arena.xml uses a
        # 4 ms physics timestep and arena.py uses frame_skip=5 => 20 ms.
        self.frame_skip = 5
        # Keep the old forward-walking task, but use the real neutral height
        # as the nominal vertical target instead of the old robot's 0.10 m.
        self.target_position = jp.array([5.0, 0.0, 0.080])
        self.initial_dist = jp.sqrt(
            self.target_position[0] ** 2 + self.target_position[1] ** 2
        )

    def reset(self, rng: jax.Array):
        data = mjx.make_data(self.mjx_model)
        rng_pos, rng_vel, rng_next = jax.random.split(rng, 3)

        qpos = data.qpos
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, REAL_BODY_START_Z]))

        # q=0 is the physical HOME stance in models/arena.xml.  Small reset
        # noise keeps training robust without changing the nominal geometry.
        num_joints = self.mjx_model.nu
        joint_noise = jax.random.uniform(
            rng_pos,
            (num_joints,),
            minval=-jp.pi / 36.0,
            maxval=jp.pi / 36.0,
        )
        qpos = qpos.at[7 : 7 + num_joints].set(joint_noise)

        qvel = jax.random.uniform(
            rng_vel, (self.mjx_model.nv,), minval=-0.01, maxval=0.01
        )

        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        obs = self.get_obs(data)
        com_pos = data.subtree_com[1]

        return dict(
            data=data,
            step=0,
            obs=obs,
            rng=rng_next,
            last_com=com_pos,
        )


# Patch the class looked up by arena.main(), while leaving arena.py itself
# untouched.  This gives us a real-robot viewer without risking the legacy
# working path on main.
_legacy_arena.BattleArena = BattleArena


if __name__ == "__main__":
    _legacy_arena.main()
