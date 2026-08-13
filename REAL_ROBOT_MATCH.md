# Real robot matching status

This branch adapts the MuJoCo environment to the physical hexapod described in the `educate-center/hexapod` repository snapshot.

## Parameters taken directly from the physical robot firmware

Leg order:

| Index | Firmware | MuJoCo names |
|---:|---|---|
| 0 | FR | Leg One |
| 1 | MR | Leg Two |
| 2 | BR | Leg Three |
| 3 | BL | Leg Four |
| 4 | ML | Leg Five |
| 5 | FL | Leg Six |

Kinematic dimensions:

- Coxa: 51 mm
- Femur: 65 mm
- Tibia: 121 mm
- Neutral foot Z: -80 mm relative to the coxa pivot

Coxa pivot positions in firmware coordinates (X forward, Y right), mm:

- FR: `(110.4, 58.4)`
- MR: `(0.0, 90.8)`
- BR: `(-110.4, 58.4)`
- BL: `(-110.4, -58.4)`
- ML: `(0.0, -90.8)`
- FL: `(110.4, -58.4)`

Neutral foot targets relative to the corresponding coxa pivot, mm:

- FR: `(82, 82, -80)`
- MR: `(0, 116, -80)`
- BR: `(-82, 82, -80)`
- BL: `(-82, -82, -80)`
- ML: `(0, -116, -80)`
- FL: `(82, -82, -80)`

The firmware control frame is 20 ms / 50 Hz. The MuJoCo model therefore uses a 4 ms timestep with `frame_skip=5`.

## Action mapping

The neural-network action remains 18-dimensional and uses the firmware leg order:

`FR coxa, FR femur, FR tibia, MR coxa, ... , FL tibia`.

The existing normalized SAC action `[-1, 1]` is retained. The current safe simulation deltas around the neutral physical pose are:

- coxa: +/-45 deg
- femur: +/-45 deg
- tibia: +/-70 deg

These are deliberately conservative. The real firmware ultimately maps IK results to 0..180 degree MG996R servo commands and applies per-leg calibration trims.

## Servo calibration from firmware

Per-leg firmware trims `(coxa, femur, tibia)` in degrees:

- FR: `(+2, +4, 0)`
- MR: `(-1, -2, -3)`
- BR: `(-1, 0, -3)`
- BL: `(-3, -1, -2)`
- ML: `(-2, 0, -3)`
- FL: `(-3, 0, -1)`

Global IK mechanical offsets:

- femur: +14 deg
- tibia: -23 deg in the firmware IK formula

These calibration values describe the real servo installation. They are not baked into the MuJoCo joint zero positions; MuJoCo joint zero is defined as the ideal neutral HOME stance. They will matter when the learned 18 joint targets are converted to commands for the real Arduino/PCA9685 robot.

## Files in this branch

- `models/arena.xml` — simplified MuJoCo rigid-body model matched to the real robot's kinematic dimensions, leg mounting points and 50 Hz control period.
- `arena_real.py` — real-robot reset pose and viewer wrapper.
- `sac_real.py` — training entry point that reuses the existing SAC implementation with the real-robot environment.

Viewer example:

```bash
python arena_real.py --agent good_models/last/agent.flax --norm good_models/last/obs_norm_last.npz
```

Training entry point:

```bash
python sac_real.py
```

## Still approximate / needs measurement on the physical robot

The following values are currently engineering approximations rather than measured robot parameters:

- assembled body mass and centre of mass;
- printed-link masses and inertias;
- joint friction/backlash;
- exact MG996R closed-loop response and deadband under the robot's supply voltage;
- foot/ground friction for the actual foot material;
- battery/electronics mass distribution.

Before final sim-to-real training these should be measured or identified experimentally. The kinematic skeleton, leg order, neutral stance and 50 Hz command timing are already matched to the firmware values.
