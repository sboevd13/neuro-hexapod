"""Generate a lightweight MuJoCo model matched to the educate-center hexapod.

The kinematic/physics skeleton stays simple for MJX. Visual geometry is kept in
render-only group 2 and does not participate in contacts or inertial inference.
The body outline comes from the outer hull of Body_Top_Plate.stl and
Body_Bottom_Plate.stl in the physical-robot repository; leg visuals are kept
low-poly/primitive on purpose so the WSL viewer does not return to ~4 FPS.

Run once after pulling this branch:
    python real_robot_model.py
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

# Outer XY hull of the actual printable body plates, metres.
BODY_HULL = [
    (-0.094, -0.020), (-0.072, -0.042), (0.072, -0.042), (0.094, -0.020),
    ( 0.094,  0.020), ( 0.072,  0.042), (-0.072, 0.042), (-0.094, 0.020),
]


def _prism_mesh(z0: float, z1: float):
    vertices = [(x, y, z) for z in (z0, z1) for x, y in BODY_HULL]
    faces = []
    for i in range(1, 7):
        faces.append((0, i + 1, i))
        faces.append((8, 8 + i, 8 + i + 1))
    for i in range(8):
        j = (i + 1) % 8
        faces.append((i, j, 8 + j))
        faces.append((i, 8 + j, 8 + i))
    vertex_attr = " ".join(f"{c:.6f}" for v in vertices for c in v)
    face_attr = " ".join(str(i) for f in faces for i in f)
    return vertex_attr, face_attr


def build_xml() -> str:
    top_v, top_f = _prism_mesh(0.0125, 0.0190)
    bot_v, bot_f = _prism_mesh(-0.0155, -0.0125)

    parts = [f'''<mujoco model="educate_center_hexapod_visual">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" inertiagrouprange="0 1"/>
  <option timestep="0.004" iterations="50" tolerance="1e-10" solver="Newton" jacobian="sparse" gravity="0 0 -9.81"/>
  <default>
    <joint armature="0.002" damping="0.08" stiffness="0" limited="true"/>
    <geom contype="1" conaffinity="2" condim="3" friction="1.4 0.08 0.02" solref="0.006 1" solimp="0.9 0.99 0.001"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0.15 0.15 0.15" rgb2="0.75 0.75 0.75" type="2d" width="100"/>
    <material name="m_arena" reflectance="0.2" texture="texplane"/>
    <material name="m_print" rgba="0.08 0.28 0.78 1"/>
    <material name="m_servo" rgba="0.10 0.10 0.12 1"/>
    <mesh name="body_top_visual" vertex="{top_v}" face="{top_f}"/>
    <mesh name="body_bottom_visual" vertex="{bot_v}" face="{bot_f}"/>
  </asset>
  <worldbody>
    <light diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 2.5"/>
    <geom material="m_arena" name="floor" pos="0 0 0" size="20 20 0.05" type="plane" contype="2" conaffinity="1" friction="1.4 0.08 0.02"/>
    <body name="Hexapod_Body" pos="0 0 0.082">
      <camera name="track" mode="trackcom" pos="-0.8 -1.3 0.7" xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root" type="free" limited="false" armature="0" damping="0"/>
      <geom name="body_main" type="box" size="0.115 0.095 0.012" pos="0 0 0.002" mass="0.45" rgba="0 0 0 0" group="0"/>
      <geom type="mesh" mesh="body_top_visual" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="mesh" mesh="body_bottom_visual" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="cylinder" size="0.004 0.014" pos="0.070 -0.031 0.0015" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="cylinder" size="0.004 0.014" pos="0 -0.040 0.0015" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="cylinder" size="0.004 0.014" pos="-0.070 -0.031 0.0015" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="cylinder" size="0.004 0.014" pos="-0.070 0.031 0.0015" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="cylinder" size="0.004 0.014" pos="0 0.040 0.0015" material="m_print" contype="0" conaffinity="0" group="2"/>
      <geom type="cylinder" size="0.004 0.014" pos="0.070 0.031 0.0015" material="m_print" contype="0" conaffinity="0" group="2"/>
''']

    for word, low, short, x, y, yaw, j1, j2, j3 in LEGS:
        parts.append(f'''
      <body name="Leg_{word}_Motor_One" pos="{x:.4f} {y:.4f} 0" euler="0 0 {yaw:.10f}">
        <joint name="{j1}" type="hinge" axis="0 0 1" range="-0.7853981634 0.7853981634"/>
        <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
        <geom type="capsule" fromto="0 0 0 0.051 0 0" size="0.010" mass="0.020" rgba="0 0 0 0" group="0"/>
        <geom type="box" size="0.020 0.010 0.018" pos="0.018 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>
        <geom type="capsule" fromto="0.005 0 0 0.051 0 0" size="0.008" material="m_print" contype="0" conaffinity="0" group="2"/>
        <body name="Leg_{word}_Motor_Two" pos="0.051 0 0" euler="0 -0.666718034 0">
          <joint name="{j2}" type="hinge" axis="0 1 0" range="-0.7853981634 0.7853981634"/>
          <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
          <geom type="capsule" fromto="0 0 0 0.065 0 0" size="0.009" mass="0.025" rgba="0 0 0 0" group="0"/>
          <geom type="box" size="0.020 0.010 0.018" pos="0.012 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>
          <geom type="capsule" fromto="0.010 -0.010 0 0.065 -0.010 0" size="0.004" material="m_print" contype="0" conaffinity="0" group="2"/>
          <geom type="capsule" fromto="0.010 0.010 0 0.065 0.010 0" size="0.004" material="m_print" contype="0" conaffinity="0" group="2"/>
          <body name="Leg_{word}_Motor_Three" pos="0.065 0 0" euler="0 2.122223468 0">
            <joint name="{j3}" type="hinge" axis="0 1 0" range="-1.221730476 1.221730476"/>
            <geom type="box" size="0.020 0.014 0.020" mass="0.055" rgba="0 0 0 0" group="0"/>
            <geom type="capsule" fromto="0 0 0 0.121 0 0" size="0.008" mass="0.040" rgba="0 0 0 0" group="0"/>
            <geom type="sphere" pos="0.121 0 0" size="0.012" mass="0.005" friction="1.8 0.08 0.02" rgba="0.08 0.08 0.09 1" group="0"/>
            <geom type="box" size="0.020 0.010 0.018" pos="0.012 0 0" material="m_servo" contype="0" conaffinity="0" group="2"/>
            <geom type="capsule" fromto="0.012 -0.008 0 0.116 -0.008 0" size="0.0035" material="m_print" contype="0" conaffinity="0" group="2"/>
            <geom type="capsule" fromto="0.012 0.008 0 0.116 0.008 0" size="0.0035" material="m_print" contype="0" conaffinity="0" group="2"/>
            <site name="elbow_{low}" pos="0 0 0" size="0.009" rgba="1 0 0 0.35" type="sphere"/>
            <site name="tibia_{low}_tip" pos="0.121 0 0" size="0.007"/>
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
    print(f"Wrote real-robot MuJoCo model: {path}")
    return path


if __name__ == "__main__":
    write_model()
