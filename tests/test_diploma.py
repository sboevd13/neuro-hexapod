"""Control contracts, firmware mapping, physical contact, optics, and desktop events."""

import os
import subprocess
import sys
import unittest
import numpy as np

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import mujoco
from hexapod.benchmark import run
from hexapod.controller import Controller
from hexapod.geometry import (
    FIRMWARE_ORDER,
    HIP,
    JOINT_HIGH,
    JOINT_LOW,
    LINKS,
    RADIAL,
    SERVO_SPEED,
    TRIMS,
    firmware_degrees,
    solve,
)
from hexapod.inputs import Gamepad, Keyboard, Replay
from hexapod.perception import DepthPerception, Pinhole
from hexapod.ports import Command, DT, Scan, Sensors, tick
from hexapod.simulation import Simulation
from hexapod.world import CAMERA, ROOT


class Pad:
    def get_init(self):
        return True

    def get_numaxes(self):
        return 3

    def get_axis(self, i):
        return (0.0, -1.0, 0.0)[i]


class Contracts(unittest.TestCase):
    def test_portable_without_simulator_ui_or_trainer(self):
        cmd = "from hexapod.controller import Controller; import sys; Controller(); assert not set(('mujoco','pygame','jax','sklearn')).intersection(sys.modules)"
        subprocess.run([sys.executable, "-c", cmd], cwd=ROOT, check=True)

    def test_keyboard_gamepad_and_replay_produce_identical_motor_frames(self):
        keyboard = Keyboard()
        keyboard.key("w", True)

        class Recorder:
            def __init__(self):
                self.output = []

            def observe(self):
                return Sensors(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), Scan(0.0))

            def apply(self, targets):
                self.output.append(targets.angles)

        outputs = []
        for source in [keyboard, Gamepad(Pad()), Replay([dict(at=0, forward=1)])]:
            sink = Recorder()
            ctrl = Controller()
            for _ in range(24):
                tick(source, sink, ctrl, sink)
            outputs.append(sink.output)
        for candidate in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], candidate)
        keyboard.key("w", False)
        self.assertEqual(keyboard.read(1.0), Command())

    def test_replay_and_finite_normalized_diagonal(self):
        r = Replay([dict(at=1.0, forward=1, left=1), dict(at=2.0)])
        np.testing.assert_array_equal(r.read(0.99).vector(), [0, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(r.read(1.0).vector()), 1.0)
        self.assertEqual(r.read(2.0), Command())
        with self.assertRaises(ValueError):
            Replay([dict(at=2.0), dict(at=1.0)])
        with self.assertRaises(ValueError):
            Command(float("nan")).vector()

    def test_blocked_stale_and_retreat(self):
        ctrl = Controller()
        for scan, expected in [
            (Scan(0.0, roof_m=0.09), "blocked"),
            (Scan(0.0, left_m=0.13, right_m=0.14), "blocked"),
            (Scan(0.0, wall_m=0.18), "blocked"),
            (Scan(-1.0), "sensor_stop"),
            (Scan(0.0, False), "sensor_stop"),
        ]:
            obs = Sensors(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), scan)
            self.assertEqual(ctrl.step(Command(1.0), obs, DT).mode, expected)
            self.assertEqual(ctrl.step(Command(-1.0), obs, DT).mode, "normal")
        bad = Sensors(0.0, (float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0), Scan(0.0))
        self.assertEqual(ctrl.step(Command(1.0), bad, DT).mode, "sensor_stop")

    def test_servo_range_and_rate_for_all_legs(self):
        ctrl = Controller()
        prev = ctrl.angles.copy()
        for i in range(200):
            now = i * DT
            obs = Sensors(
                now, (0.03, 0.01, 0.01), (0.1, 0.0, 0.0), Scan(now, roof_m=0.15)
            )
            actual = ctrl.step(
                Command(1.0, 0.3, 0.5 if i < 100 else -0.5), obs, DT
            ).angles
            self.assertTrue((np.abs(actual - prev) <= SERVO_SPEED * DT + 1e-10).all())
            self.assertTrue(np.all((actual >= JOINT_LOW) & (actual <= JOINT_HIGH)))
            self.assertLessEqual(firmware_degrees(actual).max(), 180)
            prev = actual

    def test_scalar_firmware_conventions_all_legs(self):
        # Independent form of IKSolver.cpp, X forward / Y right, firmware leg order.
        rng = np.random.default_rng(171)
        for _ in range(10):
            feet = np.column_stack(
                (RADIAL * rng.uniform(108, 128, (6, 1)), rng.uniform(-105, -75, 6))
            )
            feet[:, :2] += rng.uniform(-14, 14, (6, 2))
            feet = np.rint(feet)
            reference = []
            for j, leg in enumerate(FIRMWARE_ORDER):
                x, y, z = feet[leg] * [1, -1, 1]
                d = np.hypot(x, y) - 51.0
                radius = np.hypot(d, z)
                a = np.rad2deg(np.arctan2(x, y)) + TRIMS[leg, 0]
                coxa = (
                    a
                    + [45, 90, 135, -135, -90, -45][j]
                    + (360 if j >= 3 and a < 0 else 0)
                )
                fem = (
                    np.rad2deg(
                        np.arccos((65**2 + radius**2 - 121**2) / (130 * radius))
                        + np.arctan2(z, d)
                    )
                    + 104
                    + TRIMS[leg, 1]
                )
                tib = (
                    np.rad2deg(np.arccos((65**2 + 121**2 - radius**2) / (2 * 65 * 121)))
                    - 23
                    + TRIMS[leg, 2]
                )
                reference.append(np.floor(np.clip([coxa, fem, tib], 0, 180) + 0.5))
            np.testing.assert_array_equal(
                firmware_degrees(solve(feet / 1000)), reference
            )


class HardwareAdapters(unittest.TestCase):
    def test_recorded_hardware_stream_uses_the_same_controller(self):
        from hexapod.hardware import (
            ImuReading,
            CameraReading,
            LiveSensors,
            CalibratedActuators,
        )

        class Camera:
            calls = 0

            def read(self, image, attitude, time):
                self.calls += 1
                return Scan(time)

        t = [500.0]
        imu = [ImuReading(t[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        frame = [CameraReading(t[0], np.zeros((2, 2)))]
        cam = Camera()
        live = LiveSensors(lambda: imu[0], lambda: frame[0], cam, lambda: t[0])
        wire = []
        sink = CalibratedActuators(wire.append)
        source = Replay([{"at": 0, "forward": 1}])
        ctrl = Controller()
        output = tick(source, live, ctrl, sink)
        self.assertEqual(wire[-1], firmware_degrees(output.angles).tobytes())
        self.assertEqual(len(wire[-1]), 18)
        t[0] += 0.02
        self.assertEqual(live.observe().scan.time, 0.0)
        self.assertEqual(cam.calls, 1)
        t[0] += 0.5
        self.assertEqual(tick(source, live, ctrl, sink).mode, "sensor_stop")
        imu[0] = ImuReading(t[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(tick(source, live, ctrl, sink).mode, "sensor_stop")
        frame[0] = CameraReading(t[0], np.ones((2, 2)))
        self.assertEqual(tick(source, live, ctrl, sink).mode, "normal")


class MonocularGeometry(unittest.TestCase):
    def test_corridor_fraction_from_projected_floor_edges(self):
        from hexapod.vision import CorridorEdges, RGB_WIDTH, RGB_HEIGHT
        from hexapod.perception import level_rotation

        estimator = CorridorEdges()
        for attitude in [(0.0, 0.0, 0.0), (0.06, -0.04, 0.02)]:
            rays = estimator.rays @ level_rotation(*attitude[:2]).T
            surface_y = 0.13 * rays[:, :, 1] / np.maximum(0.01, -rays[:, :, 2])
            for left, right in [(0.20, 0.25), (0.25, 0.20), (0.22, 0.22)]:
                with self.subTest(left=left, right=right, attitude=attitude):
                    image = np.full(
                        (RGB_HEIGHT, RGB_WIDTH, 3), [85, 100, 100], dtype=np.uint8
                    )
                    image[(surface_y > left) | (surface_y < -right)] = [150, 92, 46]
                    fraction = estimator.left_fraction(image, attitude)
                    self.assertIsNotNone(fraction)
                    self.assertAlmostEqual(fraction, left / (left + right), delta=0.022)
        image[:] = [100, 100, 100]
        self.assertIsNone(estimator.left_fraction(image, (0.0, 0.0, 0.0)))


class Physics(unittest.TestCase):
    def test_home_pivots_tip_positions_and_physical_limits(self):
        sim = Simulation("flat", 401)
        m, d = sim.model, sim.data
        for i, name in enumerate(("coxa", "femur", "tibia")):
            self.assertAlmostEqual(
                m.numeric(name + "_length_mm").data[0], LINKS[i] * 1000
            )
        feet = np.column_stack((RADIAL * 0.116, np.full(6, -0.08)))
        d.qpos[:3] = [0, 0, 0.12]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:] = solve(feet)
        mujoco.mj_forward(m, d)
        for i, name in enumerate(
            (
                "front_left",
                "middle_left",
                "back_left",
                "front_right",
                "middle_right",
                "back_right",
            )
        ):
            hip = d.xpos[m.body("coxa_" + name).id]
            np.testing.assert_allclose(hip[:2], HIP[i], atol=1e-8)
            np.testing.assert_allclose(
                d.site_xpos[m.site("foot_" + name).id] - hip, feet[i], atol=0.00005
            )
        np.testing.assert_allclose(m.actuator_ctrlrange[:, 0], JOINT_LOW, atol=1e-7)
        np.testing.assert_allclose(m.actuator_ctrlrange[:, 1], JOINT_HIGH, atol=1e-7)

    def test_rays_match_onscreen_camera_in_metres(self):
        sim = Simulation("low", 35)
        sampled = sim.depth()
        with mujoco.Renderer(
            sim.model, sim.camera.height, sim.camera.width
        ) as renderer:
            renderer.update_scene(sim.data, camera=CAMERA)
            renderer.enable_depth_rendering()
            optical = renderer.render()
        comparable = np.isfinite(sampled) & (optical < 1.4) & (sampled < 1.4)
        self.assertGreater(comparable.sum(), 180)
        self.assertLess(
            np.median(abs(sampled[comparable] - optical[comparable])), 0.0025
        )
        self.assertFalse(
            DepthPerception(Pinhole())
            .read(np.full_like(sampled, np.nan), [0, 0, 0], 0.0)
            .valid
        )

    def test_sensor_contract_excludes_world_and_encoders(self):
        obs = Simulation().observe()
        self.assertEqual(set(vars(obs)), {"time", "rpy", "gyro", "scan"})
        self.assertEqual(
            set(vars(obs.scan)),
            {"time", "valid", "roof_m", "left_m", "right_m", "wall_m", "ground_m"},
        )

    def test_all_challenges_and_baseline_collision(self):
        for name in ("flat", "low", "narrow", "rough", "blocked", "mixed"):
            with self.subTest(name=name):
                measured = run(name, 41)
                self.assertTrue(measured["accepted"], measured)
        self.assertFalse(run("low", 41, False)["accepted"])

    def test_stop_before_impassable_roof_and_corridor(self):
        for camera in ("depth", "rgb"):
            for name, opening in (("low", 0.105), ("narrow", 0.32)):
                with self.subTest(camera=camera, course=name):
                    sim = Simulation(name, 1609, camera, opening=opening)
                    try:
                        trial = run(name, 1609, sim=sim)
                        self.assertEqual(trial["outcome"], "blocked", trial)
                        self.assertLess(trial["x_m"], sim.course.entrance, trial)
                        self.assertEqual(trial["obstacle_contacts"], [], trial)
                    finally:
                        sim.close()

    def test_stop_without_contact_and_reverse(self):
        sim = Simulation("blocked", 97)
        ctrl = Controller()
        keyboard = Keyboard()
        keyboard.key("w", True)
        for _ in range(250):
            if tick(keyboard, sim, ctrl, sim).mode == "blocked":
                break
        else:
            self.fail("Robot never stopped")
        at = sim.data.qpos[0]
        for _ in range(100):
            self.assertEqual(tick(keyboard, sim, ctrl, sim).mode, "blocked")
        self.assertLess(abs(sim.data.qpos[0] - at), 0.035)
        keyboard.key("w", False)
        keyboard.key("s", True)
        for _ in range(170):
            tick(keyboard, sim, ctrl, sim)
        self.assertLess(sim.data.qpos[0], at - 0.12)


class Window(unittest.TestCase):
    def test_sdl_keyboard_views_reset_pause_and_exit(self):
        from hexapod.viewer import Console, pg

        app = Console("low", 41)
        try:

            def send(kind, key):
                pg.event.post(pg.event.Event(kind, key=key))
                for ev in pg.event.get():
                    app.event(ev)

            send(pg.KEYDOWN, pg.K_w)
            for _ in range(110):
                app.step()
            self.assertGreater(app.sim.data.qpos[0], 0.12)
            send(pg.KEYUP, pg.K_w)
            self.assertEqual(app.source.read(0.0), Command())
            app.draw()
            side = pg.surfarray.array3d(app.canvas).copy()
            send(pg.KEYDOWN, pg.K_TAB)
            app.draw()
            self.assertTrue(app.onboard)
            self.assertGreater(
                np.mean(abs(side.astype(float) - pg.surfarray.array3d(app.canvas))), 3.0
            )
            send(pg.KEYDOWN, pg.K_3)
            self.assertEqual(app.track, "narrow")
            self.assertEqual(app.sim.time, 0.0)
            send(pg.KEYDOWN, pg.K_SPACE)
            app.step()
            self.assertEqual(app.sim.time, 0.0)
            send(pg.KEYDOWN, pg.K_r)
            self.assertFalse(app.paused)
            send(pg.KEYDOWN, pg.K_ESCAPE)
            self.assertFalse(app.running)
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
