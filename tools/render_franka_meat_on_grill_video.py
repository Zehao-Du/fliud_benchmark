#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


CAMERA_PRESETS: Dict[str, Dict[str, object]] = {
    "hero_front_diag": {
        "position": (1.38, -0.88, 0.86),
        "look_at": (0.60, -0.02, 0.24),
        "focal_length": 12.0,
    },
    "front_wide": {
        "position": (1.56, -0.12, 0.72),
        "look_at": (0.62, 0.00, 0.22),
        "focal_length": 11.0,
    },
    "side_grill": {
        "position": (0.54, -1.30, 0.76),
        "look_at": (0.66, 0.04, 0.24),
        "focal_length": 13.0,
    },
    "top_diag": {
        "position": (1.02, -0.82, 1.16),
        "look_at": (0.61, 0.01, 0.19),
        "focal_length": 10.0,
    },
}


def _load_scene_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
    spec = importlib.util.spec_from_file_location("franka_meat_on_grill_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scene module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Franka meat_on_grill demo video.")
    parser.add_argument("--preset", choices=["default", "cinematic_smoke"], default="default")
    parser.add_argument("--variation", choices=["chicken", "steak"], default="chicken")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=480)
    parser.add_argument("--backend", choices=["auto", "isaac", "mock"], default="auto")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/franka_meat_on_grill_video"))
    parser.add_argument("--camera-preset", choices=sorted(CAMERA_PRESETS), default="hero_front_diag")
    parser.add_argument(
        "--camera-list",
        type=str,
        default="",
        help="Comma-separated camera preset list. When set, renders each preset into a subdirectory.",
    )
    return parser.parse_args()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _camera_names(args: argparse.Namespace) -> List[str]:
    if not args.camera_list:
        return [args.camera_preset]
    names = [name.strip() for name in args.camera_list.split(",") if name.strip()]
    if not names:
        return [args.camera_preset]
    unknown = [name for name in names if name not in CAMERA_PRESETS]
    if unknown:
        raise ValueError(f"Unknown camera presets: {unknown}")
    return names


def _summary_payload(args: argparse.Namespace, render_results: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "preset": args.preset,
        "variation": args.variation,
        "seed": args.seed,
        "backend": args.backend,
        "fps": args.fps,
        "frames_requested": args.frames,
        "renders": render_results,
    }


def _write_summary(output_dir: Path, args: argparse.Namespace, render_results: List[Dict[str, object]]) -> Dict[str, object]:
    summary = _summary_payload(args, render_results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _render_with_basic_writer(scene, env, camera_cfg: Dict[str, object], output_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    import carb.settings
    import omni.replicator.core as rep

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    camera = rep.create.camera(
        position=camera_cfg["position"],
        look_at=camera_cfg["look_at"],
        focal_length=camera_cfg["focal_length"],
        name=f"FrankaMeatOnGrillCam_{output_dir.name}",
    )
    render_product = rep.create.render_product(camera, (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frame_dir), rgb=True)
    writer.attach(render_product)
    rep.orchestrator.preview()

    captured = 0
    try:
        for _frame_index in range(args.frames):
            obs = env.get_obs()
            if obs["task"]["done"]:
                break
            env.step_sim(1)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            captured += 1
        rep.orchestrator.wait_until_complete()
    finally:
        writer.detach()
        render_product.destroy()

    frame_paths = sorted(frame_dir.glob("rgb_*.png"))
    if not frame_paths:
        raise RuntimeError("No RGB frames were written by BasicWriter.")

    video_path = output_dir / "demo.mp4"
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(video_path, fps=args.fps, codec="libx264", format="FFMPEG") as writer_obj:
            for frame_path in frame_paths:
                writer_obj.append_data(imageio.imread(frame_path))
    except Exception:
        scene.write_placeholder_video(video_path)

    final_obs = env.get_obs()
    final_metrics = env.get_metrics()
    return {
        "mode": "basic_writer",
        "backend_used": env.backend_name,
        "frames_captured": len(frame_paths),
        "video_path": str(video_path),
        "frame_dir": str(frame_dir),
        "final_task_done": bool(final_obs["task"]["done"]),
        "final_task_success": bool(final_obs["task"]["success"]),
        "step_count": int(getattr(env._backend, "_step_index", len(frame_paths))),
        "smoke_mode": getattr(env._backend, "_smoke_mode", final_obs["smoke"].get("mode", "unknown")),
        "target_error": final_metrics.get("target_error"),
    }


def _render_fallback(scene, env, output_dir: Path, args: argparse.Namespace, fallback_reason: str | None = None) -> Dict[str, object]:
    summary, artifacts = env.run_episode(seed=args.seed, output_dir=output_dir, max_steps=args.frames)
    demo_path = output_dir / "demo.mp4"
    if artifacts.video_path != demo_path:
        shutil.copy2(artifacts.video_path, demo_path)
    return {
        "mode": "synthetic_episode",
        "backend_used": env.backend_name,
        "frames_captured": int(summary.get("step_count", 0)),
        "video_path": str(demo_path),
        "episode_output_dir": str(artifacts.output_dir),
        "summary": summary,
        "fallback_reason": fallback_reason,
    }


def _render_single(scene, args: argparse.Namespace, camera_name: str, output_dir: Path) -> Tuple[Dict[str, object], object | None]:
    config = scene.get_preset_config(args.preset, {"scene": {"variation": args.variation}})
    config["seed"] = args.seed
    config["headless"] = True
    config["video"]["enabled"] = True
    config["video"]["fps"] = args.fps
    config["video"]["min_duration_seconds"] = min(
        float(config["video"].get("min_duration_seconds", 3.0)),
        max(args.frames / max(args.fps, 1), 0.5),
    )

    env = scene.FrankaMeatOnGrillEnv(config=config, preset=args.preset, backend=args.backend)
    env.create_scene()
    env.reset_scene(args.seed)
    if env.backend_name == "isaac":
        try:
            return _render_with_basic_writer(scene, env, CAMERA_PRESETS[camera_name], output_dir, args), env
        except Exception as exc:
            try:
                env.close()
            except Exception:
                pass
            env = scene.FrankaMeatOnGrillEnv(config=config, preset=args.preset, backend="mock")
            env.create_scene()
            env.reset_scene(args.seed)
            return _render_fallback(scene, env, output_dir, args, fallback_reason=str(exc)), env
    return _render_fallback(scene, env, output_dir, args), env


def _render_multiview_via_subprocess(args: argparse.Namespace, output_dir: Path, camera_names: List[str]) -> List[Dict[str, object]]:
    render_results: List[Dict[str, object]] = []
    script_path = Path(__file__).resolve()
    for camera_name in camera_names:
        camera_dir = output_dir / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script_path),
            "--preset",
            args.preset,
            "--variation",
            args.variation,
            "--seed",
            str(args.seed),
            "--frames",
            str(args.frames),
            "--backend",
            args.backend,
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--output-dir",
            str(camera_dir),
            "--camera-preset",
            camera_name,
        ]
        subprocess.run(cmd, check=True)
        summary_path = camera_dir / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Child render did not produce summary.json for {camera_name}")
        child_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not child_summary.get("renders"):
            raise RuntimeError(f"Child render summary for {camera_name} had no render results")
        render_results.append(child_summary["renders"][0])
    return render_results


def main() -> int:
    args = _parse_args()
    scene = _load_scene_module()
    output_dir = args.output_dir.resolve()
    _reset_dir(output_dir)

    camera_names = _camera_names(args)
    if len(camera_names) > 1 and args.backend != "mock":
        render_results = _render_multiview_via_subprocess(args, output_dir, camera_names)
        summary = _write_summary(output_dir, args, render_results)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    render_results: List[Dict[str, object]] = []
    envs_to_close: List[object] = []
    try:
        for camera_name in camera_names:
            camera_dir = output_dir if len(camera_names) == 1 else output_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            result, env = _render_single(scene, args, camera_name, camera_dir)
            result["camera_preset"] = camera_name
            result["camera"] = CAMERA_PRESETS[camera_name]
            render_results.append(result)
            _write_summary(output_dir, args, render_results)
            if env is not None:
                envs_to_close.append(env)
    finally:
        for env in envs_to_close:
            backend_name = getattr(env, "backend_name", None)
            if backend_name == "isaac":
                continue
            try:
                env.close()
            except Exception:
                pass

    summary = _write_summary(output_dir, args, render_results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
