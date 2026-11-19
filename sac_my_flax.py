"""Historical launcher; the maintained laptop path is bounded residual PPO V3.

Use sac_legacy_v2.py only to reproduce old SAC experiments/checkpoints.
"""
from train_laptop import main

if __name__ == '__main__':
    print('Starting laptop V3 (bounded residual PPO). Legacy SAC: sac_legacy_v2.py', flush=True)
    main()
