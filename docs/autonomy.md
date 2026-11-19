# Autonomous exploration: simulation prototype

This experiment adds a high-level autonomous agent on top of the validated
V5 gait and residual PPO controller. The existing locomotion training files,
observations (65) and 18 actuator actions are unchanged.

## What is implemented

- Deterministic MuJoCo rooms with four walls and 0–12 randomized fixed obstacles.
- Idealized **horizontal depth rays** (23 rays, 220° field of view, 1.15 m range).
- Occupancy-grid memory: free, unknown, occupied, and remembered collisions.
- Safe frontier selection with clearance inflation and shortest paths through
  *observed free* cells. Unknown space is not treated as traversable.
- Automatic high-level commands (forward/turn/idle), executed by the original
  V5/PPO locomotion controller at 50 Hz.
- Persistent room memory, contact statistics and deterministic seeds.
- CPU-only unit tests for the sensor, mapping, navigation and scene generator.

This is **not** a trained visual animal brain. Learning here means accumulating
an occupancy map and adapting future routes to experience. It does not include
online neural-network weight updates, object understanding, emotional states,
SLAM, or recovery from arbitrary disturbances.

## Installation and launch

Use the established laptop environment from the root README:

```bash
python -m pip install -r requirements-laptop.txt
python -m unittest discover -s tests -p 'test_autonomy.py' -v
```

Reference gait (does not require a PPO checkpoint):

```bash
python explore_laptop.py --reference --seconds 90 --seed 7 --view
```

Already trained residual PPO (with **your local** checkpoint):

```bash
python explore_laptop.py --checkpoint checkpoints/laptop_v3/best.npz --seconds 90 --seed 7 --view
```

Headless mode: omit `--view`. Close the viewer to stop an experiment. Results
are written to `runs/autonomy/occupancy.json` and `runs/autonomy/metrics.json`.
Run again with `--continue-map` **only for the same seed and obstacle count** to
reuse the previous map. Set `--output runs/experiment_2` to compare runs without
overwriting logs. No checkpoint or weights are created by this experiment.

Map coverage is the fraction of grid cells *observed*, not necessarily safely
visited. A `contacts` event is physical contact with a scene wall/obstacle, not
a guarantee of robot damage. The simulation is not yet validated against a
physical camera or IMU.

## Measurement and research protocol

For a meaningful evaluation, repeat the runs at several unseen scene seeds
and compare V5/reference to V5+PPO using identical seeds and durations.
Measure coverage, collisions, traveled distance, falls and elapsed time.
The agent should demonstrate increasing mapped area as it explores and avoid
obstacles already recorded; this must be checked experimentally rather than
assumed from the algorithm.

## Important limitations and next steps

1. **Pose is privileged**: the demo supplies MuJoCo's true XY/yaw as an ideal
   localization estimate. On hardware, replace this with visual-inertial
   odometry or another localization method. IMU alone cannot locate the robot.
2. **Depth is synthetic**: the virtual rays read geometric intersections with
   static obstacles, not RGB camera pixels. Add a simulated camera and a depth
   or obstacle-perception model before claiming real visual navigation.
3. **The robot is physically large relative to the map**: clearance is inflated
   conservatively, but foot contacts, body turning radius and slip may still
   cause collisions. Sensor and planner are research prototypes.
4. **No neural high-level policy yet**: the navigator is an adaptive frontier
   planner, not end-to-end reinforcement learning. A separate high-level RL
   policy can be trained later using this scene/sensor interface.
5. **No real robot deployment**: add servo feedback, a hardware emergency stop,
   bounded command timeouts and calibrated perception/localization before
   sending any autonomous commands to Arduino.
6. **Locomotion checkpoint unchanged**: obstacle scenes are created at runtime
   without modifying its trained physics XML. Existing flat-ground evaluation
   results do not establish obstacle-navigation success.
