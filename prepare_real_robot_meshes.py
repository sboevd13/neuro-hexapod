"""Extract the exact STL assets used by the real-robot MuJoCo visual model.

Usage:
    python prepare_real_robot_meshes.py /path/to/hexapod-master.zip
    python prepare_real_robot_meshes.py /path/to/hexapod-master
    python prepare_real_robot_meshes.py real_robot_stls_used.zip

The files are copied verbatim into models/real_robot_meshes/source/.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

DEST = Path("models/real_robot_meshes/source")
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


def _candidate_paths() -> list[Path]:
    candidates = [
        Path("real_robot_stls_used.zip"),
        Path("hexapod-master.zip"),
        Path("models/real_robot_meshes/real_robot_stls_used.zip"),
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
        by_basename = {}
        for member in zf.namelist():
            base = Path(member).name
            if base in REQUIRED:
                score = 1 if "/hardware/stl/" in member.replace("\\", "/") else 0
                previous = by_basename.get(base)
                if previous is None or score > previous[0]:
                    by_basename[base] = (score, member)

        missing = [name for name in REQUIRED if name not in by_basename]
        if missing:
            raise RuntimeError(
                "Archive does not contain required STL files: " + ", ".join(missing)
            )

        DEST.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED:
            member = by_basename[name][1]
            with zf.open(member) as src, (DEST / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _find_stl_dir(source: Path) -> Path:
    candidates = [
        source,
        source / "hardware" / "stl",
        source / "hexapod-master" / "hardware" / "stl",
    ]
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / name).is_file() for name in REQUIRED):
            return candidate
    raise RuntimeError(f"Could not find required STL files under directory: {source}")


def _copy_from_dir(source: Path) -> None:
    stl_dir = _find_stl_dir(source)
    DEST.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(stl_dir / name, DEST / name)


def prepare(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file() and zipfile.is_zipfile(source):
        _extract_from_zip(source)
    elif source.is_dir():
        _copy_from_dir(source)
    else:
        raise RuntimeError(f"Unsupported source: {source}")

    missing = [name for name in REQUIRED if not (DEST / name).is_file()]
    if missing:
        raise RuntimeError("Extraction incomplete: " + ", ".join(missing))

    print(f"Prepared {len(REQUIRED)} exact STL files in: {DEST}")
    for name in REQUIRED:
        print(f"  {name}")
    print("\nNext:")
    print("  python real_robot_model.py")
    print("  python check_geometry.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        help="Supervisor hexapod ZIP/repository directory or real_robot_stls_used.zip",
    )
    args = parser.parse_args()

    if args.source:
        source = Path(args.source).expanduser()
    else:
        source = next((p for p in _candidate_paths() if p.exists()), None)
        if source is None:
            parser.error(
                "No STL source found automatically. Pass /path/to/hexapod-master.zip "
                "or real_robot_stls_used.zip."
            )

    print(f"STL source: {source}")
    prepare(source)


if __name__ == "__main__":
    main()
