"""Prepare exact robot STL assets plus rigidly re-framed visual copies.

The files in models/real_robot_meshes/source/ are copied verbatim from the
supervisor's educate-center/hexapod hardware/stl directory. The files in
models/real_robot_meshes/visual/ contain the same triangle geometry after only
rigid rotation/translation, so their local origins line up with MuJoCo joints.
No scaling, deformation, decimation or remeshing is performed here.

Usage:
    python prepare_real_robot_meshes.py real_robot_stls_used.zip
    python prepare_real_robot_meshes.py /path/to/hexapod-master.zip
    python prepare_real_robot_meshes.py /path/to/hexapod-master

If source/ is already populated, running without an argument simply rebuilds
visual/ from those exact source files.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np

SOURCE_DIR = Path("models/real_robot_meshes/source")
VISUAL_DIR = Path("models/real_robot_meshes/visual")

REQUIRED = [
    "Body_Top_Plate.stl",
    "Body_Bottom_Plate.stl",
    "Servo_Mount.stl",
    "Femur_Bracket.stl",
    "Tibia_Base_Plate.stl",
    "Tibia_Foot_Plate.stl",
    "Tibia_Side_1.stl",
    "Tibia_Side_2.stl",
    "Tibia_Foot_Bumper.stl",
]

# Source-CAD -> canonical link-frame transforms, in millimetres.
# These are proper rotations (det=+1) plus translations only.
TRANSFORMS = {
    "Servo_Mount.stl": (
        "Servo_Mount_local.stl",
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        (-27.682622910, -62.803611755, -12.647932768),
    ),
    "Femur_Bracket.stl": (
        "Femur_Bracket_local.stl",
        ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        (-26.647748947, 68.099220276, 38.347733498),
    ),
    "Tibia_Base_Plate.stl": (
        "Tibia_Base_Plate_local.stl",
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        (110.113769531, 28.366666794, 2.428256005),
    ),
    "Tibia_Foot_Plate.stl": (
        "Tibia_Foot_Plate_local.stl",
        ((-1.0, 0.0, 0.0), (0.0, -0.999999693210, 0.000783313306),
         (0.0, 0.000783313306, 0.999999693210)),
        (-150.741378784, 15.850497379, 89.119028465),
    ),
    "Tibia_Side_1.stl": (
        "Tibia_Side_1_local.stl",
        ((0.805703426740, -0.592319160706, 0.0), (0.0, 0.0, -1.0),
         (0.592319160706, 0.805703426740, 0.0)),
        (292.385437337, -177.457763672, 150.303962508),
    ),
    "Tibia_Side_2.stl": (
        "Tibia_Side_2_local.stl",
        ((-0.943675421655, -0.330872631936, 0.0), (0.0, 0.0, -1.0),
         (0.330872631936, -0.943675421655, 0.0)),
        (-82.465642490, -65.561855316, 87.486024073),
    ),
    "Tibia_Foot_Bumper.stl": (
        "Tibia_Foot_Bumper_local.stl",
        ((0.999990085039, 0.004453023249, -0.000020186022),
         (-0.000020185821, -0.000000089889, -0.999999999796),
         (-0.004453023250, 0.999990085243, 0.0)),
        (95.327886629, -27.642533718, 2.590756060),
    ),
}


def _candidate_paths() -> list[Path]:
    candidates = [
        Path("real_robot_stls_used.zip"),
        Path("hexapod-master.zip"),
        Path.home() / "Downloads" / "real_robot_stls_used.zip",
        Path.home() / "Downloads" / "hexapod-master.zip",
    ]
    win_users = Path("/mnt/c/Users")
    if win_users.exists():
        for user in win_users.iterdir():
            candidates.extend([
                user / "Downloads" / "real_robot_stls_used.zip",
                user / "Downloads" / "hexapod-master.zip",
            ])
    return candidates


def _extract_from_zip(source: Path) -> None:
    with zipfile.ZipFile(source) as zf:
        by_basename: dict[str, tuple[int, str]] = {}
        for member in zf.namelist():
            base = Path(member).name
            if base not in REQUIRED:
                continue
            score = 1 if "/hardware/stl/" in member.replace("\\", "/") else 0
            prev = by_basename.get(base)
            if prev is None or score > prev[0]:
                by_basename[base] = (score, member)

        missing = [name for name in REQUIRED if name not in by_basename]
        if missing:
            raise RuntimeError("Archive is missing: " + ", ".join(missing))

        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED:
            with zf.open(by_basename[name][1]) as src, (SOURCE_DIR / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _find_stl_dir(source: Path) -> Path:
    candidates = [source, source / "hardware" / "stl", source / "hexapod-master" / "hardware" / "stl"]
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / name).is_file() for name in REQUIRED):
            return candidate
    raise RuntimeError(f"Could not find required STL files under: {source}")


def _copy_from_dir(source: Path) -> None:
    stl_dir = _find_stl_dir(source)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(stl_dir / name, SOURCE_DIR / name)


def _transform_binary_stl(src: Path, dst: Path, rotation, translation) -> None:
    data = src.read_bytes()
    if len(data) < 84:
        raise RuntimeError(f"Invalid STL: {src}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + triangles * 50
    if len(data) != expected:
        raise RuntimeError(f"Expected binary STL, got unsupported format: {src}")

    rot = np.asarray(rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64)
    if not np.isclose(np.linalg.det(rot), 1.0, atol=1e-5):
        raise RuntimeError(f"Transform for {src.name} is not a proper rotation")

    out = bytearray(data)
    header = ("MuJoCo visual rigid-frame copy of " + src.name).encode("ascii", "replace")[:80]
    out[:80] = header.ljust(80, b" ")

    for i in range(triangles):
        offset = 84 + i * 50
        values = struct.unpack_from("<12f", data, offset)
        normal = np.asarray(values[:3], dtype=np.float64)
        vertices = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)

        normal = rot @ normal
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            normal /= norm
        vertices = vertices @ rot.T + trans

        struct.pack_into("<12f", out, offset, *normal.tolist(), *vertices.reshape(-1).tolist())

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)


def _build_visual_meshes() -> None:
    missing = [name for name in REQUIRED if not (SOURCE_DIR / name).is_file()]
    if missing:
        raise RuntimeError("Source STL set is incomplete: " + ", ".join(missing))

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    # Body plates are already authored in the robot body coordinate frame.
    shutil.copy2(SOURCE_DIR / "Body_Top_Plate.stl", VISUAL_DIR / "Body_Top_Plate.stl")
    shutil.copy2(SOURCE_DIR / "Body_Bottom_Plate.stl", VISUAL_DIR / "Body_Bottom_Plate.stl")

    for source_name, (output_name, rotation, translation) in TRANSFORMS.items():
        _transform_binary_stl(
            SOURCE_DIR / source_name,
            VISUAL_DIR / output_name,
            rotation,
            translation,
        )


def prepare(source: Path | None) -> None:
    if source is not None:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file() and zipfile.is_zipfile(source):
            _extract_from_zip(source)
        elif source.is_dir():
            _copy_from_dir(source)
        else:
            raise RuntimeError(f"Unsupported STL source: {source}")
    elif not all((SOURCE_DIR / name).is_file() for name in REQUIRED):
        raise RuntimeError("source/ is not prepared and no archive/directory was supplied")

    _build_visual_meshes()

    print(f"Exact source STL files: {SOURCE_DIR}")
    print(f"Rigidly re-framed visual STL files: {VISUAL_DIR}")
    print("No STL was scaled, deformed or remeshed.")
    print("\nNext:")
    print("  python real_robot_model.py")
    print("  python check_geometry.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="hexapod ZIP/directory; optional if source/ already exists")
    args = parser.parse_args()

    source: Path | None = None
    if args.source:
        source = Path(args.source).expanduser()
    elif not all((SOURCE_DIR / name).is_file() for name in REQUIRED):
        source = next((p for p in _candidate_paths() if p.exists()), None)
        if source is None:
            parser.error("No STL source found. Pass real_robot_stls_used.zip or hexapod-master.zip")

    if source is not None:
        print(f"STL source: {source}")
    prepare(source)


if __name__ == "__main__":
    main()
