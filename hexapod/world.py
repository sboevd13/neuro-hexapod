"""Calibrated rigid-body model, real contact geometry and reproducible courses."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .geometry import LINKS, JOINT_LOW, JOINT_HIGH

ROOT = Path(__file__).resolve().parents[1]
COURSES = ("flat", "low", "narrow", "rough", "blocked", "mixed")
TITLES = (
    "Ровная площадка",
    "Низкий тоннель",
    "Узкий коридор",
    "Неровный настил",
    "Глухая преграда",
    "Комбинированная трасса",
)
CAMERA = "onboard"
FOV = 85.0


@dataclass(frozen=True)
class Course:
    name: str
    finish: float
    obstacles: tuple[str, ...]
    entrance: float


def numbers(values):
    return " ".join(f"{v:.8g}" for v in values)


def robot_xml():
    root = ET.fromstring((ROOT / "models/arena.xml").read_text())
    body = root.find("./worldbody/body[@name='body']")
    for name, length in zip(("coxa", "femur", "tibia"), LINKS):
        root.find(f"./custom/numeric[@name='{name}_length_mm']").set(
            "data", str(length * 1000)
        )
    for i, joint in enumerate(body.iter("joint")):
        joint.set("range", numbers(np.rad2deg([JOINT_LOW[i], JOINT_HIGH[i]])))
    for node in body.iter("body"):
        name = node.get("name", "")
        if name.startswith(("femur_", "tibia_")):
            node.set(
                "pos", numbers([LINKS[0 if name.startswith("femur") else 1], 0, 0])
            )
    for geom in body.iter("geom"):
        name = geom.get("name", "")
        geom.set("group", "1")
        for prefix, length in zip(
            ("coxa", "femur", "tibia"), (*LINKS[:2], LINKS[2] - 0.007)
        ):
            if name.startswith(prefix + "_collision_"):
                geom.set("fromto", numbers([0, 0, 0, length, 0, 0]))
        for prefix, length in zip(
            ("mass_servo_q2_", "mass_servo_q3_", "foot_collision_"),
            (*LINKS[:2], LINKS[2] - 0.007),
        ):
            if name.startswith(prefix):
                geom.set("pos", numbers([length, 0, 0]))
    for site in body.iter("site"):
        if site.get("name", "").startswith("foot_"):
            site.set("pos", numbers([LINKS[2], 0, 0]))
    ET.SubElement(
        body,
        "geom",
        name="camera_housing",
        type="box",
        pos=numbers([0.098, 0, 0.045]),
        size=".012 .023 .010",
        mass=".026",
        group="1",
        rgba=".12 .28 .33 1",
    )
    ET.SubElement(
        body,
        "camera",
        name=CAMERA,
        pos=numbers([0.111, 0, 0.045]),
        xyaxes="0 -1 0 0 0 1",
        fovy=str(FOV),
    )
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(
        sensors, "framequat", name="imu_orientation", objtype="site", objname="imu_site"
    )
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu_site")
    return root


def build(name, seed=0, opening=None):
    if name not in COURSES:
        raise ValueError(f"Unknown course {name!r}")
    rng = np.random.default_rng(seed)
    root = robot_xml()
    visual = root.find("visual")
    ET.SubElement(visual, "map", znear=".007", zfar="24")
    ET.SubElement(visual, "global", offwidth="800", offheight="520")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="tiles",
        type="2d",
        builtin="checker",
        width="256",
        height="256",
        rgb1=".36 .43 .44",
        rgb2=".41 .48 .49",
    )
    ET.SubElement(
        assets,
        "material",
        name="ground",
        texture="tiles",
        texrepeat="4 4",
        texuniform="true",
    )
    world = root.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.attrib.update(
        size="12 6 .1",
        material="ground",
        rgba="1 1 1 1",
        friction=numbers([rng.uniform(0.80, 1.0), 0.01, 0.0001]),
    )
    obstacles = []

    def box(label, center, size, color, solid=True):
        attr = dict(
            name=label,
            type="box",
            pos=numbers(center),
            size=numbers(size),
            rgba=numbers((*color, 1.0)),
        )
        if solid:
            attr.update(friction=".85 .01 .0001", group="0")
            obstacles.append(label)
        else:
            attr.update(contype="0", conaffinity="0", group="3")
        ET.SubElement(world, "geom", **attr)

    def segment(kind, start):
        span = 0.40
        x = start + span / 2
        color = np.clip(
            np.array([0.65, 0.41, 0.23]) + rng.uniform(-0.06, 0.06, 3), 0.1, 0.9
        )
        if kind == "low":
            h = (
                opening
                if opening is not None and name == "low"
                else rng.uniform(0.143, 0.153)
            )
            box("roof_low", (x, 0, h + 0.027), (span / 2, 0.39, 0.027), color)
            for side in (-1, 1):
                box(
                    f"post_{side}",
                    (x, side * 0.40, h / 2),
                    (0.033, 0.025, h / 2),
                    color,
                )
        elif kind == "narrow":
            half = (
                opening / 2
                if opening is not None and name == "narrow"
                else rng.uniform(0.224, 0.230)
            )
            for side in (-1, 1):
                box(
                    f"wall_{side}",
                    (x, side * (half + 0.034), 0.11),
                    (span / 2, 0.034, 0.11),
                    color,
                )
        elif kind == "rough":
            for row in range(4):
                for col in range(3):
                    h = rng.uniform(0.014, 0.038) * (1 if (row + col) % 2 else 0.48)
                    box(
                        f"tile_{row}_{col}",
                        (start + 0.055 + row * 0.102, (col - 1) * 0.148, h / 2),
                        (0.047, 0.071, h / 2),
                        np.clip(color * (0.78 + 0.06 * row), 0, 1),
                    )
            span = 0.42
        elif kind == "blocked":
            box("barrier", (start + 0.04, 0, 0.18), (0.04, 0.48, 0.18), color)
            span = 0.08
        return start + span

    start = end = 0.63
    for kind in ("low", "narrow", "rough") if name == "mixed" else (name,):
        end = segment(kind, end)
        if name == "mixed" and kind != "rough":
            end += 0.60
    finish = end + 0.35 if name != "flat" else 1.1
    for sign in (-1, 1):
        box(
            f"lane_marker_{sign}",
            ((finish - 0.1) / 2, sign * 0.55, 0.001),
            ((finish + 0.8) / 2, 0.009, 0.001),
            (0.82, 0.77, 0.48),
            False,
        )
    box(
        "finish_mark",
        (finish, 0, 0.0015),
        (0.014, 0.5, 0.0015),
        (0.16, 0.65, 0.55),
        False,
    )
    return ET.tostring(root, encoding="unicode"), Course(
        name, finish, tuple(obstacles), start
    )
