# Learning from object interaction in MuJoCo

Experimental branch `feature/object-affordances-v1`, built on the object-curiosity experiment.

## Goal

Instead of only mapping and looking at obstacles, the simulated hexapod executes
**one bounded forward probe** against a physical MuJoCo box and measures whether
its observed surface moved. The same learned category preference is used when
faced with a new box of similar apparent width.

- `train_light_narrow`: freejoint, low-mass obstacle; the robot tries a push.
- `train_fixed_wide`: anchored obstacle; the robot tries once and learns.
- `heldout_light_narrow`: new light object, same observed category.
- `heldout_fixed_wide`: new anchored object, same observed category.

The policy sees only idealized 23-beam lidar returns and whether an experimental
contact switch was triggered. It **does not** read freejoint metadata, mass,
known labels, or true object displacement. Those are used only by the
simulator/sensor adapter or the independent evaluator. The visual width classes
(`narrow`, `wide`) are coarse range-sensor proxies and not real object recognition.

## How to run

```bash
python interact_laptop.py --seed 7 --seconds 8 --output runs/interaction
```

Outputs: `runs/interaction/affordance_memory.json` (versioned experience),
`runs/interaction/report.json` (sensor motion **and** privileged truth values
clearly separated). The model stays isolated from existing trained PPO, V5,
frontier exploration, and curiosity settings.

The learning algorithm uses Beta-smoothed estimates of pushability by observed
width category. Initial unknown objects permit a limited test, then the
posterior mobility estimate biases future `observe` versus `probe` decisions.
This is **interpretable online learning**, not a deep neural network or a
universal affordance predictor. The four-case trial is a controlled demonstration
only and does not show manipulation in arbitrary scenes.

## Limitations

- The probe uses ordinary V5 forward locomotion: there is no arm, grasping,
  precision force control, or learned 18-joint pushing skill.
- The robot's world position is obtained directly from MuJoCo, not SLAM.
- Ray intersections use simplified oriented-square geometry; rotating boxes
  may be imperfectly estimated.
- MuJoCo's contact state acts as an **idealized tactile switch**; real robots
  would need physical force/touch feedback and a watchdog.
- Only simulator trials, no physical robot actuation.
- Safety of pushing household objects cannot be inferred from sandbox trials.
