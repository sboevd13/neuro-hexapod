# Real robot STL assets

The visual model uses exact source STL files from the supervisor robot repository snapshot `educate-center/hexapod`, path `hardware/stl/`.

Required files:

- `Body_Top_Plate.stl`
- `Body_Bottom_Plate.stl`
- `Servo_Mount.stl`
- `Femur_Bracket.stl`
- `Tibia_Base_Plate.stl`
- `Tibia_Foot_Plate.stl`
- `Tibia_Side_1.stl`
- `Tibia_Side_2.stl`
- `Tibia_Foot_Bumper.stl`

Run:

```bash
python prepare_real_robot_meshes.py /path/to/hexapod-master.zip
```

or pass the small `real_robot_stls_used.zip` bundle. The helper copies the STL bytes verbatim into:

```text
models/real_robot_meshes/source/
```

Then regenerate the MuJoCo XML:

```bash
python real_robot_model.py
```

`models/arena.xml` loads these STL files as render-only meshes. They use `contype=0` / `conaffinity=0` and are excluded from inertial inference; the already verified primitive collision/kinematic skeleton stays unchanged.
