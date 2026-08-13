"""Generate the MuJoCo model for the educate-center physical hexapod.

The verified kinematic/collision skeleton is unchanged. Visual geometry uses
actual STL files copied from educate-center/hexapod hardware/stl into
models/real_robot_meshes/source. The leg STL assets keep their original CAD
geometry; rigid geom transforms only re-express them in the local joint frames.
Visual meshes are group=2 with collisions disabled and are excluded from
inertial inference.
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

# Rigid CAD->link-frame transforms for the raw STL coordinates.
# Positions are metres after applying the STL 0.001 mm->m scale.
SERVO_MOUNT_POS = (-0.036182624, -0.062803612, -0.012647933)
SERVO_MOUNT_QUAT = (0.707106781, 0.0, 0.0, 0.707106781)
FEMUR_POS = (-0.021647749, 0.068099220, 0.038347733)
FEMUR_QUAT = (0.0, 0.707106781, -0.707106781, 0.0)
TIBIA_BASE_POS = (0.089113770, 0.002428256, -0.028366667)
TIBIA_BASE_QUAT = (0.707106781, 0.707106781, 0.0, 0.0)
TIBIA_FOOT_POS = (-0.171741379, 0.089119028, -0.015850497)
TIBIA_FOOT_QUAT = (0.0, 0.0, 0.707383670, 0.706829784)
TIBIA_SIDE1_POS = (0.292385437, -0.177457764, 0.150303963)
TIBIA_SIDE1_QUAT = (0.671882324, 0.671882324, -0.220395425, 0.220395425)
TIBIA_SIDE2_POS = (-0.082465642, -0.065561855, 0.087486024)
TIBIA_SIDE2_QUAT = (-0.118663999, -0.118663999, 0.697078801, -0.697078801)
FOOT_BUMPER_POS = (0.080772675, -0.027642534, 0.002590756)
FOOT_BUMPER_QUAT = (0.707105013, 0.707105044, 0.001567249, -0.001581522)


def v3(v):
    return " ".join(f"{x:.9f}" for x in v)


def q4(q):
    return " ".join(f"{x:.9f}" for x in q)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def build_xml() -> str:
    parts = [f'''<mujoco model="educate_center_hexapod_real_stl">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" inertiagrouprange="0 1" meshdir="real_robot_meshes/source"/>
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
    <material name="m_print2" rgba="0.05 0.18 0.55 1"/>
    <material name="m_servo" rgba="0.075 0.075 0.09 1"/>
    <material name="m_metal" rgba="0.55 0.58 0.62 1"/>

    <mesh name="body_top_stl" file="Body_Top_Plate.stl" scale="0.001 0.001 0.001"/>
    <mesh name="body_bottom_stl" file="Body_Bottom_Plate.stl" scale="0.001 0.001 0.001"/>
    <mesh name="servo_mount_stl" file="Servo_Mount.stl" scale="0.001 0.001 0.001"/>
    <mesh name="femur_bracket_stl" file="Femur_Bracket.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_base_stl" file="Tibia_Base_Plate.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_foot_stl" file="Tibia_Foot_Plate.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_side1_stl" file="Tibia_Side_1.stl" scale="0.001 0.001 0.001"/>
    <mesh name="tibia_side2_stl" file="Tibia_Side_2.stl" scale="0.001 0.001 0.001"/>
    <mesh name="foot_bumper_stl" file="Tibia_Foot_Bumper.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <light diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 2.5"/>
    <geom material="m_arena" name="floor" pos="0 0 0" size="20 20 0.05" type="plane" contype="2" conaffinity="1" friction="1.4 0.08 0.02"/>

    <body name="Hexapod_Body" pos="0 0 0.082">
      <camera name="track" mode="trackcom" pos="-0.8 -1.3 0.7" xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root" type="free" limited="false" armature="0" damping="0"/>
      <geom name="body_main" type="box" size="0.115 0.095 0.012" pos="0 0 0.002" mass="0.45" rgba="0 0 0 0" group="0"/>

      <!-- Actual printable body plates, unmodified STL geometry. -->
      <geom type="mesh" mesh="body_top_stl" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="mesh" mesh="body_bottom_stl" material="m_print2" contype="0" conaffinity="0" group="2"/>

      <!-- Electronics placeholders; the repository has no MG996R STL. -->
      <geom type="box" size="0.045 0.026 0.009" pos="0 0 0.028" rgba="0.04 0.19 0.32 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.043 0.025 0.008" pos="0 0 -0.002" material="m_servo" contype="0" conaffinity="0" group="2"/>
''']

    for word, low, short, x, y, yaw, j1, j2, j3 in LEGS:
        servo_pos = add(SERVO_MOUNT_POS, (0.0255, 0.0, 0.0))
        femur_pos = add(FEMUR_POS, (0.0034, 0.0, 0.0))
        tib_base_pos = add(TIBIA_BASE_POS, (0.012, 0.0, 0.0))
        tib_foot_pos = add(TIBIA_FOOT_POS, (0.108, 0.0, 0.0))
        side1_pos = add(TIBIA_SIDE1_POS, (0.010, -0.014, 0.0))
        side2_pos = add(TIBIA_SIDE2_POS, (0.010, 0.014, 0.0))
        bumper_pos = add(FOOT_BUMPER_POS, (0.113, 0.0, 0.0))
        parts.append(f'''
      <body name="Leg_{word}_Motor_One" pos="{x:.4f} {y:.4f} 0" euler="0 0 {yaw:.10f}">
        <joint name="{j1}" type="hinge" axis="0 0 1" range="-0.7853981634 0.7853981634"/>
        <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
        <geom type="capsule" fromto="0 0 0 0.051 0 0" size="0.010" mass="0.020" rgba="0 0 0 0" group="0"/>
        <geom type="mesh" mesh="servo_mount_stl" pos="{v3(servo_pos)}" quat="{q4(SERVO_MOUNT_QUAT)}" material="m_print" contype="0" conaffinity="0" group="2"/>
        <geom type="box" size="0.020 0.010 0.020" pos="0.018 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>

        <body name="Leg_{word}_Motor_Two" pos="0.051 0 0" euler="0 -0.666718034 0">
          <joint name="{j2}" type="hinge" axis="0 1 0" range="-0.7853981634 0.7853981634"/>
          <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
          <geom type="capsule" fromto="0 0 0 0.065 0 0" size="0.009" mass="0.025" rgba="0 0 0 0" group="0"/>
          <geom type="mesh" mesh="femur_bracket_stl" pos="{v3(femur_pos)}" quat="{q4(FEMUR_QUAT)}" material="m_print" contype="0" conaffinity="0" group="2"/>
          <geom type="box" size="0.020 0.010 0.020" pos="0.010 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>

          <body name="Leg_{word}_Motor_Three" pos="0.065 0 0" euler="0 2.122223468 0">
            <joint name="{j3}" type="hinge" axis="0 1 0" range="-1.221730476 1.221730476"/>
            <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
            <geom type="capsule" fromto="0 0 0 0.121 0 0" size="0.008" mass="0.040" rgba="0 0 0 0" group="0"/>
            <geom type="sphere" pos="0.121 0 0" size="0.012" mass="0.005" friction="1.8 0.08 0.02" rgba="0 0 0 0" group="0"/>

            <geom type="mesh" mesh="tibia_base_stl" pos="{v3(tib_base_pos)}" quat="{q4(TIBIA_BASE_QUAT)}" material="m_print" contype="0" conaffinity="0" group="2"/>
            <geom type="mesh" mesh="tibia_side1_stl" pos="{v3(side1_pos)}" quat="{q4(TIBIA_SIDE1_QUAT)}" material="m_print" contype="0" conaffinity="0" group="2"/>
            <geom type="mesh" mesh="tibia_side2_stl" pos="{v3(side2_pos)}" quat="{q4(TIBIA_SIDE2_QUAT)}" material="m_print2" contype="0" conaffinity="0" group="2"/>
            <geom type="mesh" mesh="tibia_foot_stl" pos="{v3(tib_foot_pos)}" quat="{q4(TIBIA_FOOT_QUAT)}" material="m_print" contype="0" conaffinity="0" group="2"/>
            <geom type="mesh" mesh="foot_bumper_stl" pos="{v3(bumper_pos)}" quat="{q4(FOOT_BUMPER_QUAT)}" material="m_servo" contype="0" conaffinity="0" group="2"/>
            <geom type="box" size="0.020 0.010 0.020" pos="0.010 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>

            <site name="elbow_{low}" pos="0 0 0" size="0.009" rgba="1 0 0 0.25" type="sphere"/>
            <site name="tibia_{low}_tip" pos="0.121 0 0" size="0.007" rgba="0 1 0 0.2"/>
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_xml(), encoding="utf-8")
    print(f"Wrote real-STL robot model: {path}")
    return path


if __name__ == "__main__":
    write_model()
