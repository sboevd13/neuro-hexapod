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

The firmware control frame is 20 ms / 50 Hz. The MuJoCo model uses a 4 ms physics timestep with `frame_skip=5`.

The HOME geometry has been checked numerically in MuJoCo: all six `tibia_*_tip` sites match the expected firmware foot targets to about 0.00-0.03 mm.

## Training model philosophy

The training model intentionally does **not** use STL render meshes. STL appearance is irrelevant to the policy and made the WSL viewer slower and harder to debug.

`models/arena.xml` therefore uses only primitive geometry:

- body: box based on the real body-plate outer dimensions;
- MG996R bodies: boxes using nominal servo dimensions;
- coxa/femur/tibia: capsules with the real kinematic lengths;
- foot: a small sphere approximating the physical 12 mm-thick foot bumper.

The kinematic contact point remains at 121 mm from the tibia joint. The foot sphere is centred slightly before that point so its outer surface approximately ends at the kinematic tip.

Robot geoms collide with the floor but self-collision is disabled through MuJoCo contact masks. This keeps MJX training fast.

## Action mapping

The neural-network action remains 18-dimensional and uses firmware leg order:

`FR coxa, FR femur, FR tibia, MR coxa, ... , FL tibia`.

The current simulation action ranges around HOME are:

- coxa: +/-45 deg
- femur: +/-45 deg
- tibia: +/-70 deg

These are still provisional safe ranges. The physical firmware ultimately maps commands to MG996R servo positions and applies calibration trims.

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

These calibration values are not baked into the MuJoCo ideal HOME joint zeros. They belong in the future policy-to-servo conversion layer for the physical robot.

## Files in this branch

- `models/arena.xml` — STL-free MuJoCo training model.
- `real_robot_model.py` — regenerates that primitive training model.
- `arena_real.py` — real-robot reset pose and viewer wrapper.
- `sac_real.py` — training entry point reusing the existing SAC pipeline.

Regenerate the model:

```bash
python real_robot_model.py
python check_geometry.py
```

Viewer:

```bash
python arena_real.py --agent good_models/last/agent.flax --norm good_models/last/obs_norm_last.npz
```

Training:

```bash
python sac_real.py
```

## Still approximate / needs measurement on the physical robot

The major remaining sim-to-real uncertainties are dynamic rather than visual:

- assembled body mass and centre of mass;
- printed-link masses and inertias;
- exact MG996R speed/torque response at the robot supply voltage;
- servo deadband, backlash and command delay;
- real joint limits and sign/zero mapping for all 18 servos;
- foot/ground friction and compliance;
- battery/electronics mass distribution;
- observations actually available on the physical robot (IMU, servo feedback, etc.).

These should be measured or identified before final transfer training. STL appearance is not required.
