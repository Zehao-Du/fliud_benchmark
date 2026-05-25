import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCENE_PATH = ROOT / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
RENDER_PATH = ROOT / "tools/render_franka_meat_on_grill_video.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scene = _load_module("franka_meat_on_grill_test_module", SCENE_PATH)
render = _load_module("render_franka_meat_on_grill_test_module", RENDER_PATH)


class FrankaMeatOnGrillTests(unittest.TestCase):
    def test_reset_consistency(self):
        env = scene.FrankaMeatOnGrillEnv(config={"video": {"enabled": False}}, preset="default", backend="mock")
        env.create_scene()
        obs_a = env.reset_scene(5)
        obs_b = env.reset_scene(5)
        self.assertEqual(obs_a["meat"]["position"], obs_b["meat"]["position"])
        self.assertEqual(obs_a["smoke"]["centroid"], obs_b["smoke"]["centroid"])
        env.close()

    def test_phase_sequence_and_completion(self):
        env = scene.FrankaMeatOnGrillEnv(config={"video": {"enabled": False}}, preset="default", backend="mock")
        env.create_scene()
        summary, _ = env.run_episode(seed=9, output_dir=Path(tempfile.mkdtemp()))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["phase_sequence"], scene.PHASE_ORDER)
        self.assertTrue(summary["metrics"]["success"])
        self.assertTrue(summary["metrics"]["done"])
        env.close()

    def test_observation_contract(self):
        env = scene.FrankaMeatOnGrillEnv(config={"video": {"enabled": False}}, preset="cinematic_smoke", backend="mock")
        env.create_scene()
        obs = env.reset_scene(13)
        valid, errors = scene.validate_obs(obs)
        self.assertTrue(valid, msg="; ".join(errors))
        self.assertEqual(obs["task"]["phase"], "approach")
        self.assertFalse(obs["task"]["done"])
        env.close()

    def test_meat_reaches_target_zone_after_release(self):
        env = scene.FrankaMeatOnGrillEnv(config={"video": {"enabled": False}}, preset="default", backend="mock")
        env.create_scene()
        env.reset_scene(17)
        final_obs = env.get_obs()
        for _ in range(sum(scene.get_preset_config("default")["task"]["phase_steps"].values()) + 8):
            final_obs = env.get_obs()
            if final_obs["task"]["done"]:
                break
            env.step_sim(1)
        self.assertTrue(final_obs["task"]["done"])
        self.assertTrue(scene._meat_in_target_zone(env.config, final_obs["meat"]["position"]))
        self.assertTrue(final_obs["robot"]["gripper_open"])
        env.close()

    def test_render_script_fallback_outputs_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "render"
            argv_backup = sys.argv[:]
            try:
                sys.argv = [
                    "render_franka_meat_on_grill_video.py",
                    "--preset",
                    "default",
                    "--backend",
                    "mock",
                    "--frames",
                    "40",
                    "--output-dir",
                    str(output_dir),
                    "--camera-list",
                    "hero_front_diag,top_diag",
                ]
                self.assertEqual(render.main(), 0)
            finally:
                sys.argv = argv_backup
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["renders"]), 2)
            for item in summary["renders"]:
                self.assertTrue(Path(item["video_path"]).exists())


if __name__ == "__main__":
    unittest.main()
