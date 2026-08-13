"""Generate the STL-free MuJoCo training model for the educate-center hexapod.

This model is intentionally primitive-only.  Sim-to-real depends on the robot's
kinematics, actuator dynamics, masses/inertias, timing and foot contact—not on
render meshes.  The geometry constants match the physical robot firmware; the
remaining dynamics are explicitly marked as approximations to refine later.
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


def build_xml() -> str:
    parts = ['''<mujoco model="educate_center_hexapod_training">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <!-- 4 ms physics x frame_skip=5 in arena.py = 20 ms / 50 Hz policy rate. -->
  <option timestep="0.004" iterations="50" tolerance="1e-10"
          solver="Newton" jacobian="sparse" gravity="0 0 -9.81"/>

  <default>
    <joint armature="0.002" damping="0.08" stiffness="0" limited="true"/>
    <!-- Robot geoms (1/2) collide with floor (2/1), but not with each other. -->
    <geom contype="1" conaffinity="2" condim="3"
          friction="1.4 0.08 0.02" solref="0.006 1" solimp="0.9 0.99 0.001"/>
  </default>

  <asset>
    <texture builtin="gradient" height="100" width="100"
             rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox"/>
    <texture builtin="checker" height="100" width="100"
             name="texplane" rgb1="0.15 0.15 0.15" rgb2="0.75 0.75 0.75" type="2d"/>
    <material name="m_arena" reflectance="0.1" texture="texplane"/>
  </asset>

  <worldbody>
    <light diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 2.5"/>
    <geom name="floor" type="plane" material="m_arena" pos="0 0 0"
          size="20 20 0.05" contype="2" conaffinity="1"
          friction="1.4 0.08 0.02"/>

    <!--
      Firmware geometry:
        coxa=51 mm, femur=65 mm, tibia=121 mm
        BODY_X=[110.4,0,-110.4,-110.4,0,110.4] mm
        MuJoCo +Y is left, so firmware Y signs are mirrored.
      HOME body height is 82 mm: the firmware -80 mm foot point starts
      about 2 mm above the floor and settles under gravity.
    -->
    <body name="Hexapod_Body" pos="0 0 0.082">
      <camera name="track" mode="trackcom" pos="-0.8 -1.3 0.7"
              xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root" type="free" limited="false" armature="0" damping="0"/>

      <!-- Approximate rigid body envelope from the real top/bottom plate bounds:
           188 x 104 x 34.5 mm. Mass still needs physical measurement. -->
      <geom name="body_main" type="box" size="0.094 0.052 0.01725"
            pos="0 0 0.00175" mass="0.45" rgba="0.72 0.12 0.12 1"/>
''']

    for word, low, short, x, y, yaw, j1, j2, j3 in LEGS:
        parts.append(f'''
      <body name="Leg_{word}_Motor_One" pos="{x:.4f} {y:.4f} 0" euler="0 0 {yaw:.10f}">
        <joint name="{j1}" type="hinge" axis="0 0 1"
               range="-0.7853981634 0.7853981634"/>
        <geom type="box" size="0.02035 0.00985 0.02145"
              mass="0.055" rgba="0.18 0.18 0.20 1"/>
        <geom type="capsule" fromto="0 0 0 0.051 0 0"
              size="0.008" mass="0.020" rgba="0.48 0.48 0.52 1"/>

        <body name="Leg_{word}_Motor_Two" pos="0.051 0 0" euler="0 -0.666718034 0">
          <joint name="{j2}" type="hinge" axis="0 1 0"
                 range="-0.7853981634 0.7853981634"/>
          <geom type="box" size="0.02035 0.00985 0.02145"
                mass="0.055" rgba="0.18 0.18 0.20 1"/>
          <geom type="capsule" fromto="0 0 0 0.065 0 0"
                size="0.007" mass="0.025" rgba="0.55 0.55 0.58 1"/>

          <body name="Leg_{word}_Motor_Three" pos="0.065 0 0" euler="0 2.122223468 0">
            <joint name="{j3}" type="hinge" axis="0 1 0"
                   range="-1.221730476 1.221730476"/>
            <geom type="box" size="0.02035 0.00985 0.02145"
                  mass="0.055" rgba="0.18 0.18 0.20 1"/>

            <!-- The real foot bumper is about 12 mm thick.  A 6 mm-radius
                 sphere is centred at 115 mm, so its distal surface is near
                 the 121 mm kinematic foot point rather than penetrating the floor. -->
            <geom type="capsule" fromto="0 0 0 0.109 0 0"
                  size="0.0065" mass="0.040" rgba="0.62 0.62 0.65 1"/>
            <geom name="{short}_foot" type="sphere" pos="0.115 0 0"
                  size="0.006" mass="0.005"
                  friction="1.8 0.08 0.02" rgba="0.05 0.05 0.05 1"/>

            <site name="elbow_{low}" pos="0 0 0"
                  size="0.001" rgba="0 0 0 0" type="sphere"/>
            <site name="tibia_{low}_tip" pos="0.121 0 0"
                  size="0.001" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>
''')

    parts.append('''    </body>
  </worldbody>

  <actuator>
''')
    for _, _, short, _, _, _, j1, j2, j3 in LEGS:
        parts.append(
            f'    <position name="{short}_coxa" joint="{j1}" kp="22" '
            'ctrlrange="-0.7853981634 0.7853981634" ctrllimited="true" '
            'forcerange="-1.1 1.1" forcelimited="true"/>\n'
        )
        parts.append(
            f'    <position name="{short}_femur" joint="{j2}" kp="22" '
            'ctrlrange="-0.7853981634 0.7853981634" ctrllimited="true" '
            'forcerange="-1.1 1.1" forcelimited="true"/>\n'
        )
        parts.append(
            f'    <position name="{short}_tibia" joint="{j3}" kp="22" '
            'ctrlrange="-1.221730476 1.221730476" ctrllimited="true" '
            'forcerange="-1.1 1.1" forcelimited="true"/>\n'
        )

    parts.append('''  </actuator>
</mujoco>
''')
    return "".join(parts)


def write_model(path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_xml(), encoding="utf-8")
    print(f"Wrote STL-free real-robot training model: {path}")
    return path


if __name__ == "__main__":
    write_model()
