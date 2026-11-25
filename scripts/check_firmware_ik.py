"""Compare actual C++ firmware IK from the supervisor's ZIP with the Python bridge."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexapod.geometry import RADIAL, TRIMS, FIRMWARE_ORDER, solve, firmware_degrees

DRIVER = r"""
#include <cstdio>
#include "src/core/IKSolver.h"
int main() {
  int leg,x,y,z,a,b,c;
  while (std::scanf("%d%d%d%d%d%d%d",&leg,&x,&y,&z,&a,&b,&c)==7) {
    auto q = core_ik::solveLegIK(leg,x,y,z,a,b,c);
    std::printf("%d %d %d %d\n", q.valid, q.coxa, q.femur, q.tibia);
  }
}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--compiler", default="g++")
    args = p.parse_args()
    rng = np.random.default_rng(171)
    with tempfile.TemporaryDirectory() as directory, ZipFile(args.archive) as source:
        work = Path(directory)
        for rel in ("Config.h", "src/core/IKSolver.h", "src/core/IKSolver.cpp"):
            entry = next(
                n for n in source.namelist() if n.endswith("/firmware/hexapod/" + rel)
            )
            dest = work / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read(entry))
        (work / "verify.cpp").write_text(DRIVER)
        exe = work / "verify"
        subprocess.run(
            [
                args.compiler,
                "-std=c++17",
                str(work / "verify.cpp"),
                str(work / "src/core/IKSolver.cpp"),
                "-o",
                str(exe),
            ],
            check=True,
        )
        for i in range(10):
            feet = np.column_stack(
                (
                    RADIAL * (116 if i == 0 else rng.uniform(108, 128, (6, 1))),
                    np.full(6, -80) if i == 0 else rng.uniform(-105, -75, 6),
                )
            )
            if i:
                feet[:, :2] += rng.uniform(-14, 14, (6, 2))
            feet = np.rint(feet).astype(int)
            rows = []
            for j, leg in enumerate(FIRMWARE_ORDER):
                x, y, z = feet[leg]
                rows.append(" ".join(str(v) for v in (j, x, -y, z, *TRIMS[leg])))
            out = subprocess.check_output(
                [str(exe)], input=("\n".join(rows) + "\n").encode()
            ).decode()
            q = np.array([[int(v) for v in line.split()] for line in out.splitlines()])
            assert q[:, 0].all()
            np.testing.assert_array_equal(
                firmware_degrees(solve(feet / 1000)), q[:, 1:]
            )
    print("60 leg poses / 180 joint commands match the compiled firmware.")


if __name__ == "__main__":
    main()
