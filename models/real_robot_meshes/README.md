# Real robot STL assets

The binary STL bundle in this directory comes from the supervisor robot repository snapshot `educate-center/hexapod`, path `hardware/stl/`.

`real_robot_stls.zip` contains the exact source STL files currently used by the MuJoCo visual model:

- `Body_Top_Plate.stl`
- `Body_Bottom_Plate.stl`
- `Servo_Mount.stl`
- `Femur_Bracket.stl`
- `Tibia_Base_Plate.stl`
- `Tibia_Foot_Plate.stl`
- `Tibia_Side_1.stl`
- `Tibia_Side_2.stl`
- `Tibia_Foot_Bumper.stl`

Run `python prepare_real_robot_meshes.py` after pulling this branch. It extracts the exact STL files into `models/real_robot_meshes/source/`, where `models/arena.xml` loads them as render-only meshes. The STL visual geometry has `contype=0` / `conaffinity=0` and is excluded from inertial inference; the already checked primitive collision skeleton remains unchanged.
