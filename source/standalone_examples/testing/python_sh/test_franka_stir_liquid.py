import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCENE_PATH = ROOT / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
FLOW_PATH = ROOT / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/flow_comparison_demo.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scene = _load_module("franka_stir_liquid_test_module", SCENE_PATH)
flow = _load_module("flow_comparison_demo_test_module", FLOW_PATH)


class FrankaStirLiquidTests(unittest.TestCase):
    def test_reset_consistency(self):
        env = scene.FrankaStirLiquidEnv(preset="default", backend="mock")
        env.create_scene()
        obs_a = env.reset_scene(7)
        metrics_a = env.get_metrics()
        obs_b = env.reset_scene(7)
        metrics_b = env.get_metrics()
        self.assertEqual(metrics_a["total_particle_count"], metrics_b["total_particle_count"])
        self.assertEqual(metrics_a["centroid"], metrics_b["centroid"])
        self.assertEqual(obs_a["container"]["position"], obs_b["container"]["position"])
        self.assertEqual(obs_a["tool"]["position"], obs_b["tool"]["position"])
        env.close()

    def test_phase_flow_and_events(self):
        env = scene.FrankaStirLiquidEnv(preset="default", backend="mock")
        env.create_scene()
        summary, _ = env.run_episode(seed=3, output_dir=Path(tempfile.mkdtemp()))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["phase_sequence"], scene.PHASE_ORDER)
        env.close()

    def test_observation_structure(self):
        env = scene.FrankaStirLiquidEnv(preset="default", backend="mock")
        env.create_scene()
        obs = env.reset_scene(11)
        valid, errors = scene.validate_obs(obs)
        self.assertTrue(valid, msg="; ".join(errors))
        env.close()

    def test_spill_count_and_ratio(self):
        config = scene.get_preset_config("default")
        particles = [[0.55, 0.0, 0.03], [1.2, 0.0, 0.03]]
        velocities = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        metrics = scene.compute_fluid_metrics(
            config=config,
            particle_positions=particles,
            particle_velocities=velocities,
            tool_position=[0.55, 0.0, 0.05],
            initial_centroid=[0.55, 0.0, 0.03],
            tool_path_length_in_fluid=0.0,
        )
        self.assertEqual(metrics["spill_count"], 1)
        self.assertEqual(metrics["spill_ratio"], 0.5)

    def test_tool_contact_proxy_rises_and_falls(self):
        env = scene.FrankaStirLiquidEnv(preset="default", backend="mock")
        env.create_scene()
        env.reset_scene(19)
        baseline = env.get_metrics()["tool_contact_proxy"]
        for _ in range(150):
            env.step_sim(1)
        contact = env.get_metrics()["tool_contact_proxy"]
        for _ in range(240):
            env.step_sim(1)
        lifted = env.get_metrics()["tool_contact_proxy"]
        self.assertGreater(contact, baseline)
        self.assertLess(lifted, contact)
        env.close()

    def test_stress_metrics_are_higher(self):
        default_env = scene.FrankaStirLiquidEnv(preset="default", backend="mock")
        default_env.create_scene()
        default_summary, _ = default_env.run_episode(seed=5, output_dir=Path(tempfile.mkdtemp()))
        stress_env = scene.FrankaStirLiquidEnv(preset="stress", backend="mock")
        stress_env.create_scene()
        stress_summary, _ = stress_env.run_episode(seed=5, output_dir=Path(tempfile.mkdtemp()))
        default_metrics = default_summary["metrics"]
        stress_metrics = stress_summary["metrics"]
        self.assertTrue(
            stress_metrics["spill_ratio"] > default_metrics["spill_ratio"]
            or stress_metrics["stirring_intensity_proxy"] > default_metrics["stirring_intensity_proxy"]
        )
        default_env.close()
        stress_env.close()

    def test_output_artifacts_exist(self):
        env = scene.FrankaStirLiquidEnv(preset="default", backend="mock")
        env.create_scene()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            summary, artifacts = env.run_episode(seed=29, output_dir=output_dir)
            self.assertIn(summary["status"], scene.STATUS_VALUES)
            for path in (
                artifacts.config_path,
                artifacts.metrics_path,
                artifacts.trajectory_path,
                artifacts.episode_summary_path,
                artifacts.video_path,
            ):
                self.assertTrue(path.exists(), msg=str(path))
            self.assertIn(b"ftyp", artifacts.video_path.read_bytes()[:32])
            metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("final", metrics)
        env.close()

    def test_flow_demo_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = flow.run_flow_demo(Path(tmpdir))
            self.assertTrue((output_dir / "comparison_table.json").exists())
            self.assertTrue((output_dir / "video.mp4").exists())
            self.assertIn(b"ftyp", (output_dir / "video.mp4").read_bytes()[:32])
            frames = sorted((output_dir / "frames").glob("*.ppm"))
            self.assertGreater(len(frames), 0)


if __name__ == "__main__":
    unittest.main()
