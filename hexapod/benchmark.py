"""Ground-truth scoring stays outside the control loop. Run both systems with identical seeds."""

import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .controller import Controller
from .gait import rpy
from .inputs import Replay
from .ports import DT, tick
from .simulation import Simulation
from .world import COURSES


def run(
    name,
    seed,
    adaptive=True,
    seconds=None,
    source=None,
    sim=None,
    camera="depth",
    trace=None,
):
    owned = sim is None
    sim = sim or Simulation(name, seed, camera)
    ctrl = Controller(adaptive)
    source = source or Replay([{"at": 0, "forward": 1.0}])
    kinds = Counter()
    touched = set()
    max_tilt = drift = 0.0
    min_z, max_z = 10.0, 0.0
    obstacle_ids = {sim.model.geom(n).id: n for n in sim.course.obstacles}
    new_contacts = []

    def capture_contacts(data):
        for contact in data.contact:
            a, b = contact.geom1, contact.geom2
            if a in obstacle_ids and sim.model.geom_group[b] == 1:
                obstacle, link = obstacle_ids[a], b
            elif b in obstacle_ids and sim.model.geom_group[a] == 1:
                obstacle, link = obstacle_ids[b], a
            else:
                continue
            if trace is not None and obstacle not in touched:
                new_contacts.append([obstacle, sim.model.geom(link).name])
            touched.add(obstacle)

    sim.after_substep = capture_contacts
    status = "timeout"
    halted = 0
    try:
        for _ in range(
            max(1, round((seconds or (65 if name == "mixed" else 30)) / DT))
        ):
            target = tick(source, sim, ctrl, sim)
            kinds[target.mode] += 1
            xy = sim.data.qpos[:2]
            z = float(sim.data.qpos[2])
            min_z, max_z = min(min_z, z), max(max_z, z)
            drift = max(drift, abs(float(xy[1])))
            tilt = float(np.rad2deg(np.max(np.abs(rpy(sim.data.qpos[3:7])[:2]))))
            max_tilt = max(max_tilt, tilt)
            if trace is not None and (sim.steps % 5 == 0 or new_contacts):
                scan = sim.scan
                trace.append(
                    dict(
                        t=round(sim.time, 3),
                        xy=xy.round(4).tolist(),
                        yaw_deg=round(float(np.rad2deg(rpy(sim.data.qpos[3:7])[2])), 2),
                        mode=target.mode,
                        compact=round(ctrl.compact, 3),
                        left=scan.left_m,
                        right=scan.right_m,
                        roof=scan.roof_m,
                        lateral_command=round(float(ctrl.command[1]), 3),
                        contacts=new_contacts.copy(),
                    )
                )
                new_contacts.clear()
            if not np.isfinite(sim.data.qpos).all() or tilt > 45.0 or z < 0.028:
                status = "fall"
                break
            halted = halted + 1 if target.mode in ("blocked", "sensor_stop") else 0
            if halted >= 50:
                status = target.mode
                break
            if xy[0] >= sim.course.finish:
                status = "finish"
                break
        contacts = sorted(n for n in touched if not n.startswith("tile_"))
        required = {
            "low": ("low",),
            "narrow": ("narrow",),
            "rough": ("rough",),
            "mixed": ("low", "narrow", "rough"),
        }.get(name, ())
        if name == "blocked":
            passed = (
                status == "blocked" and not contacts and xy[0] < sim.course.entrance
            )
        else:
            passed = (
                status == "finish"
                and not contacts
                and drift < 0.26
                and (not adaptive or all(kinds[k] >= 5 for k in required))
            )
        return dict(
            course=name,
            seed=seed,
            camera=sim.camera_mode,
            controller="adaptive" if adaptive else "baseline",
            accepted=bool(passed),
            outcome=status,
            seconds=round(sim.time, 2),
            x_m=round(float(xy[0]), 4),
            max_lateral_m=round(drift, 4),
            max_tilt_deg=round(max_tilt, 2),
            min_body_m=round(min_z, 4),
            max_body_m=round(max_z, 4),
            obstacle_contacts=contacts,
            modes=dict(kinds),
        )
    finally:
        sim.after_substep = None
        if owned:
            sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 97, 203])
    parser.add_argument("--courses", nargs="+", choices=COURSES, default=COURSES)
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--camera", choices=("depth", "rgb"), default="depth")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Save observed clearances, actual pose and first contacts for replay analysis.",
    )
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        for name in args.courses:
            for adaptive in (True, False) if args.both else (True,):
                trace = [] if args.trace_dir is not None else None
                row = run(name, seed, adaptive, camera=args.camera, trace=trace)
                if trace is not None:
                    args.trace_dir.mkdir(parents=True, exist_ok=True)
                    path = (
                        args.trace_dir
                        / f"{name}-{seed}-{args.camera}-{row['controller']}.json"
                    )
                    path.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
                rows.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    if not all(row["accepted"] for row in rows if row["controller"] == "adaptive"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
