"""Generate the MuJoCo model for the educate-center physical hexapod.

The kinematic/collision skeleton is the already verified one. Visual geometry
uses the supervisor robot's actual STL triangle meshes after prepare_real_robot_meshes.py
only rigidly re-expresses their CAD coordinates in each MuJoCo joint frame.
Visual STL geoms never participate in contacts or inertial inference.
"""
from pathlib import Path

MODEL_PATH = Path("models/arena.xml")

LEGS = [
    ("One", "one", "FR",  0.1104, -0.0584, -0.7853981634, "Motor_One_Joint_One",   "Joint_One_Motor_One",   "Joint_Two_Motor_One"),
    ("Two", "two", "MR",  0.0000, -0.0908, -1.5707963270, "Motor_One_Joint_Two",   "Joint_One_Motor_Two",   "Joint_Two_Motor_Two"),
    ("Three", "three", "BR", -0.1104, -0.0584, -2.3561944900, "Motor_One_Joint_Three", "Joint_One_Motor_Three", "Joint_Two_Motor_Three"),
    ("Four", "four", "BL", -0.1104,  0.0584,  2.3561944900, "Motor_One_Joint_Four",  "Joint_One_Motor_Four",  "Joint_Two_Motor_Four"),
    ("Five", "five", "ML",  0.0000,  0.0908,  1.5707963270, "Motor_One_Joint_Five",  "Joint_One_Motor_Five",  "Joint_Two_Motor_Five"),
    ("Six", "six", "FL",  0.1104,  0.0584,  0.7853981634, "Motor_One_Joint_Six",   "Joint_One_Motor_Six",   "Joint_Two_Motor_Six"),
]

# The two tibia side plates taper from the 35.27 mm proximal/base-plate width
# toward the 25.4 mm distal/foot-plate width over roughly 100 mm.
TIBIA_SIDE_YAW = 0.0492900443


def build_xml() -> str:
    parts = [f'''<mujoco model="educate_center_hexapod_real_stl_v2">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" inertiagrouprange="0 1" meshdir="real_robot_meshes/visual"/>
  <option timestep="0.004" iterations="50" tolerance="1e-10" solver="Newton" jacobian="sparse" gravity="0 0 -9.81"/>

  <default>
    <joint armature="0.002" damping="0.08" stiffness="0" limited="true"/>
    <geom contype="1" conaffinity="2" condim="3" friction="1.4 0.08 0.02" solref="0.006 1" solimp="0.9 0.99 0.001"/>
  </default>

  <asset>
    <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0.15 0.15 0.15" rgb2="0.75 0.75 0.75" type="2d" width="100"/>
    <material name="m_arena" reflectance="0.1" texture="texplane"/>
    <material name="m_print" rgba="0.08 0.30 0.82 1"/>
    <material name="m_print_dark" rgba="0.05 0.18 0.55 1"/>
    <material name="m_servo" rgba="0.065 0.065 0.075 1"/>
    <material name="m_foot" rgba="0.035 0.035 0.040 1"/>

    <mesh name="body_top_stl" file="Body_Top_Plate.stl" scale="0.001 0.001 0.001"/>
    <mesh name="body_bottom_stl" file="Body_Bottom_Plate.stl" scale="0.001 0.001 0.001"/>
    <mesh name="servo_mount_stl" file="Servo_Mount_local.stl" scale="0.001 0.001 0.001"/>
    <mesh name="femur_bracket_stl" file="Femur_Bracket_local.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_base_stl" file="Tibia_Base_Plate_local.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_foot_stl" file="Tibia_Foot_Plate_local.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_side1_stl" file="Tibia_Side_1_local.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_side2_stl" file="Tibia_Side_2_local.stl" scale="0.001 0.001 0.001"/>
    <mesh name="foot_bumper_stl" file="Tibia_Foot_Bumper_local.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <light diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 2.5"/>
    <geom material="m_arena" name="floor" pos="0 0 0" size="20 20 0.05" type="plane" contype="2" conaffinity="1" friction="1.4 0.08 0.02"/>

    <body name="Hexapod_Body" pos="0 0 0.082">
      <camera name="track" mode="trackcom" pos="-0.8 -1.3 0.7" xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root" type="free" limited="false" armature="0" damping="0"/>

      <!-- Verified primitive physics body; invisible in the viewer. -->
      <geom name="body_main" type="box" size="0.115 0.095 0.012" pos="0 0 0.002" mass="0.45" rgba="0 0 0 0" group="0"/>

      <!-- Actual printable body plates. Their source CAD frame already is the body frame. -->
      <geom type="mesh" mesh="body_top_stl" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="mesh" mesh="body_bottom_stl" material="m_print_dark" contype="0" conaffinity="0" group="2"/>

      <!-- Electronics/battery are placeholders because the STL set has no electronics meshes. -->
      <geom type="box" size="0.045 0.026 0.009" pos="0 0 0.028" rgba="0.04 0.19 0.32 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.043 0.025 0.008" pos="0 0 -0.002" material="m_servo" contype="0" conaffinity="0" group="2"/>
''']

    for word, low, short, x, y, yaw, j1, j2, j3 in LEGS:
        parts.append(f'''
      <body name="Leg_{word}_Motor_One" pos="{x:.4f} {y:.4f} 0" euler="0 0 {yaw:.10f}">
        <joint name="{j1}" type="hinge" axis="0 0 1" range="-0.7853981634 0.7853981634"/>

        <!-- Physics: unchanged and invisible. -->
        <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
        <geom type="capsule" fromto="0 0 0 0.051 0 0" size="0.010" mass="0.020" rgba="0 0 0 0" group="0"/>

        <!-- Coxa: actual Servo_Mount STL bridges inward to the body plate. -->
        <geom type="mesh" mesh="servo_mount_stl" material="m_print" contype="0" conaffinity="0" group="2"/>
        <!-- MG996R: 40.7 x 19.7 x 42.9 mm nominal body. -->
        <geom type="box" size="0.02035 0.00985 0.02145" pos="0 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>

        <body name="Leg_{word}_Motor_Two" pos="0.051 0 0" euler="0 -0.666718034 0">
          <joint name="{j2}" type="hinge" axis="0 1 0" range="-0.7853981634 0.7853981634"/>

          <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
          <geom type="capsule" fromto="0 0 0 0.065 0 0" size="0.009" mass="0.025" rgba="0 0 0 0" group="0"/>

          <!-- Femur bracket extends 5 mm behind the servo axis, eliminating the visual gap. -->
          <geom type="mesh" mesh="femur_bracket_stl" material="m_print" contype="0" conaffinity="0" group="2"/>
          <geom type="box" size="0.02035 0.00985 0.02145" pos="0 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>

          <body name="Leg_{word}_Motor_Three" pos="0.065 0 0" euler="0 2.122223468 0">
            <joint name="{j3}" type="hinge" axis="0 1 0" range="-1.221730476 1.221730476"/>

            <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
            <geom type="capsule" fromto="0 0 0 0.121 0 0" size="0.008" mass="0.040" rgba="0 0 0 0" group="0"/>
            <geom type="sphere" pos="0.121 0 0" size="0.012" mass="0.005" friction="1.8 0.08 0.02" rgba="0 0 0 0" group="0"/>

            <!-- Tibia is an assembled truss: proximal plate + two tapered sides + distal foot plate. -->
            <geom type="mesh" mesh="tibia_base_stl" material="m_print" contype="0" conaffinity="0" group="2"/>
            <geom type="mesh" mesh="tibia_side1_stl" pos="0.012 -0.017633 0" euler="0 0 {TIBIA_SIDE_YAW:.10f}" material="m_print" contype="0" conaffinity="0" group="2"/>
            <geom type="mesh" mesh="tibia_side2_stl" pos="0.012 0.017633 0" euler="0 0 {-TIBIA_SIDE_YAW:.10f}" material="m_print" contype="0" conaffinity="0" group="2"/>
            <geom type="mesh" mesh="tibia_foot_stl" pos="0.079 0 0" material="m_print_dark" contype="0" conaffinity="0" group="2"/>

            <!-- The black bumper is the visible ground-contact end. Its distal end is x ~= 121 mm. -->
            <geom type="mesh" mesh="foot_bumper_stl" pos="0.092 0 0" material="m_foot" contype="0" conaffinity="0" group="2"/>
            <geom type="box" size="0.02035 0.00985 0.02145" pos="0 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>

            <!-- Sites are still used by reward/debug code but hidden visually. -->
            <site name="elbow_{low}" pos="0 0 0" size="0.001" rgba="0 0 0 0" type="sphere"/>
            <site name="tibia_{low}_tip" pos="0.121 0 0" size="0.001" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>
''')

    parts.append('''    </body>
  </worldbody>

  <actuator>
''')
    for _, _, short, _, _, _, j1, j2, j3 in LEGS:
        parts.append(f'''    <position name="{short}_coxa" joint="{j1}" kp="22" ctrlrange="-0.7853981634 0.7853981634" ctrllimited="true" forcerange="-1.1 1.1" forcelimited="true"/>
    <position name="{short}_femur" joint="{j2}" kp="22" ctrlrange="-0.7853981634 0.7853981634" ctrllimited="true" forcerange="-1.1 1.1" forcelimited="true"/>
    <position name="{short}_tibia" joint="{j3}" kp="22" ctrlrange="-1.221730476 1.221730476" ctrllimited="true" forcerange="-1.1 1.1" forcelimited="true"/>
''')

    parts.append('''  </actuator>
</mujoco>
''')
    return "".join(parts)


def write_model(path: Path = MODEL_PATH) -> Path:
    visual_dir = Path("models/real_robot_meshes/visual")
    required = [
        "Body_Top_Plate.stl", "Body_Bottom_Plate.stl", "Servo_Mount_local.stl",
        "Femur_Bracket_local.stl", "Tibia_Base_Plate_local.stl",
        "Tibia_Foot_Plate_local.stl", "Tibia_Side_1_local.stl",
        "Tibia_Side_2_local.stl", "Tibia_Foot_Bumper_local.stl",
    ]
    missing = [name for name in required if not (visual_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "Visual STL files are not prepared: " + ", ".join(missing) +
            "\nRun: python prepare_real_robot_meshes.py"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_xml(), encoding="utf-8")
    print(f"Wrote assembled real-STL robot model: {path}")
    return path


if __name__ == "__main__":
    write_model()
