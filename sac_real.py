"""Train SAC against the educate-center physical-robot model.

This wrapper intentionally does not modify sac_my_flax.py.  It replaces the
BattleArena class exported by arena.py with arena_real.BattleArena, then runs
the existing SAC pipeline unchanged.

Usage:
    python sac_real.py
"""

import runpy

import arena
from arena_real import BattleArena


arena.BattleArena = BattleArena
runpy.run_module("sac_my_flax", run_name="__main__")
