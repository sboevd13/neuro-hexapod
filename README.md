
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
