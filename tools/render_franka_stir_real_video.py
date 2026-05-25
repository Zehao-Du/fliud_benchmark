#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


CAMERA_PRESETS = {
    "wide_front": {
        "position": (1.55, 0.00, 0.62),
        "look_at": (0.28, 0.00, 0.22),
        "focal_length": 10.0,
    },
    "wide_diag": {
        "position": (1.35, -0.92, 0.70),
        "look_at": (0.28, -0.05, 0.20),
        "focal_length": 12.0,
    },
    "wide_side": {
        "position": (0.18, -1.45, 0.58),
        "look_at": (0.28, -0.02, 0.20),
        "focal_length": 12.0,
    },
    "top_diag": {
        "position": (0.92, -0.72, 1.08),
        "look_at": (0.28, -0.02, 0.18),
        "focal_length": 11.0,
    },
    "top_diag_far": {
        "position": (1.30, -1.02, 1.42),
        "look_at": (0.26, -0.02, 0.16),
        "focal_length": 9.0,
    },
}


def _load_scene_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
    spec = importlib.util.spec_from_file_location("franka_stir_liquid_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scene module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a real Isaac Sim video for franka_stir_liquid.")
    parser.add_argument("--preset", choices=["default", "stress"], default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/franka_stir_liquid_real_video"))
    parser.add_argument("--camera-preset", choices=sorted(CAMERA_PRESETS), default="wide_diag")
    return parser.parse_args()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _encode_video(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def main() -> int:
    args = _parse_args()
    scene = _load_scene_module()

    output_dir = args.output_dir.resolve()
    frame_dir = output_dir / "frames"
    _reset_dir(output_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    camera_cfg = CAMERA_PRESETS[args.camera_preset]

    config = scene.get_preset_config(args.preset)
    config["seed"] = args.seed
    config["headless"] = True
    config["video"]["enabled"] = False

    env = scene.FrankaStirLiquidEnv(config=config, preset=args.preset, backend="isaac")
    env.create_scene()
    env.reset_scene(args.seed)

    try:
        import carb.settings
        import omni.replicator.core as rep

        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        camera = rep.create.camera(
            position=camera_cfg["position"],
            look_at=camera_cfg["look_at"],
            focal_length=camera_cfg["focal_length"],
            name=f"FrankaStirCaptureCam_{args.camera_preset}",
        )
        render_product = rep.create.render_product(camera, (args.width, args.height))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(frame_dir), rgb=True)
        writer.attach(render_product)

        rep.orchestrator.preview()

        captured = 0
        for frame_idx in range(args.frames):
            obs = env.get_obs()
            if obs["task"]["phase"] == "done":
                break
            print(f"Capturing frame {frame_idx + 1}/{args.frames}, phase={obs['task']['phase']}")
            env.step_sim(1)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            captured += 1

        rep.orchestrator.wait_until_complete()
        writer.detach()
        render_product.destroy()

        frame_paths = sorted(frame_dir.glob("rgb_*.png"))
        if not frame_paths:
            raise RuntimeError("No RGB frames were written by BasicWriter.")

        video_path = output_dir / "demo.mp4"
        _encode_video(frame_paths, video_path, args.fps)

        summary = {
            "preset": args.preset,
            "seed": args.seed,
            "frames_requested": args.frames,
            "frames_captured": captured,
            "camera_preset": args.camera_preset,
            "camera": camera_cfg,
            "video_path": str(video_path),
            "frame_dir": str(frame_dir),
            "backend": env.backend_name,
            "fluid_mode": "synthetic_fluid" if getattr(env._backend, "_uses_synthetic_fluid", False) else "physx_particles",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
