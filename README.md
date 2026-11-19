
## Experiment: embodied learning of object motion

The `feature/embodied-learning-v1` branch includes a *separate, optional* MuJoCo
research workflow with independently randomized object width, mass, friction
and mobility; a probe-conditioned neural outcome predictor; and a single-scene
room controller combining range-based mapping, curiosity, bounded physical
pushing and online experience updates. Existing V5/PPO gait and baseline
navigation remain available and unchanged by default.

```bash
python autonomous_world.py --seed 7 --seconds 90 \
  --model models/affordance_outcome_experimental.json \
  --experience models/affordance_training_experiences.json \
  --output runs/world_demo --view
```

In CI, 32 training and 16 independent test physical scenes produced a small
Brier-score improvement over a constant predictor (0.226 vs 0.243), with
unchanged classification accuracy (62.5%). Three continuous multi-object rooms
produced five real physical contacts and two incremental online model updates.
No result is asserted for hardware control, semantics, force safety or long-term
learned animal cognition. [Detailed protocol](docs/embodied_learning.md) and
[reproducible results](validation/embodied_learning_v1.json).

## Active physical experiment selection (research branch)

The optional `feature/active-experiment-learning` branch adds an ensemble
predictor of **physical push distance and model disagreement**. It compares
random, hand-crafted and information-seeking choices of longer MuJoCo push
experiments with identical label budgets on independent randomized objects.
The same model can choose bounded extra push lengths during a continuous
room simulation and update from newly observed outcomes. Physical robot
control and the original gait are unchanged.

```bash
python active_learning_benchmark.py --pool-size 200 --test-size 60 --output runs/active_learning
python autonomous_world.py --seed 11 --seconds 90 --view \
  --experiment-model runs/active_learning/active_world_model.json \
  --experiment-experience runs/active_learning/pool/sensor_labels.json \
  --output runs/active_world
```

See [docs/active_learning.md](docs/active_learning.md) and the CI report.


Verified 260-object physics benchmark (Nov 2026): at 64 revealed labels,
ensemble-disagreement acquisition scored 1.84 cm MAE vs 1.91 cm for random,
on 60 unseen objects. A model trained on 200 labels scored 1.81 cm MAE vs
3.28 cm for the constant baseline. Five gentle interactions and five online
updates were completed in 3 continuous simulation rooms. These are limited
research results, not proven real-world generalization. Ready-to-use model:
`models/active_displacement_ensemble.json`; experiences:
`models/active_push_replay.json`.


## Camera-guided simulation, IMU and estimated pose (experimental)

The `feature/visual-slam-simulation` branch mounts a **real rendered MuJoCo
RGB+depth camera**, simulated noisy gyro and range beams, and estimated
position from relative foot kinematics, contact and local map scan matching.
The navigation planner receives the estimated position instead of the
ground-truth MuJoCo robot coordinates. All changes are opt-in.

```bash
python visual_autonomy.py --seed 7 --seconds 60 --view --output runs/camera_demo

python autonomous_world.py --seed 11 --seconds 90 --vision-odometry \
  --experiment-model models/active_displacement_ensemble.json \
  --experiment-experience models/active_push_replay.json \
  --view --output runs/camera_embodied
```

Three 22-second MuJoCo rooms had ~**6.7 cm mean position RMSE** with
scan correction (versus 6.8 cm using foot/IMU dead reckoning). The
correction is **not consistently better in each room**. The camera produced
75 RGB frames with 113 coarse color-object detections. In three additional
continuous rooms, the robot made four object contacts and three extra
gentle pushes, with three online learning updates and no falls.

This is **not full visual SLAM or general object recognition**: depth and
color are still from simulated sensors, the props are deliberately cyan,
there is no optical-flow odometry or loop closure, and physical hardware
has not been connected. See
[docs/visual_navigation.md](docs/visual_navigation.md),
[validation/visual_localization_v1.json](validation/visual_localization_v1.json).
