#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cinematic-quality render of the franka_stir_liquid scene.

Uses RTX Path Tracing at 1920×1080 with multi-sample accumulation.
Requires an Isaac Sim Python runtime (backend=isaac).

Usage:
    python tools/render_franka_stir_cinematic.py [--preset default|stress] [--seed N]
    python tools/render_franka_stir_cinematic.py --camera hero_front_diag --spp 64
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Camera presets – tuned to show the glass container, fluid surface, and robot
# ---------------------------------------------------------------------------
CAMERA_PRESETS: Dict[str, Dict[str, object]] = {
    "hero_front_diag": {
        "position": (1.28, -0.82, 0.72),
        "look_at": (0.52, -0.02, 0.14),
        "focal_length": 28.0,
        "description": "Main hero shot – front-diagonal, shows container glass and fluid surface",
    },
    "front_close": {
        "position": (0.98, -0.38, 0.52),
        "look_at": (0.52, -0.01, 0.12),
        "focal_length": 35.0,
        "description": "Close front view – fluid surface and pick cube visible",
    },
    "top_diag": {
        "position": (0.90, -0.68, 1.02),
        "look_at": (0.52, -0.01, 0.12),
        "focal_length": 24.0,
        "description": "Top-diagonal – shows liquid surface ripples and tool path",
    },
    "side_glass": {
        "position": (0.16, -1.38, 0.60),
        "look_at": (0.52, -0.02, 0.14),
        "focal_length": 30.0,
        "description": "Side view through glass walls – shows fluid dynamics",
    },
}


def _load_scene_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("franka_stir_liquid_cinematic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scene module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cinematic-quality render of franka_stir_liquid (RTX Path Tracing).")
    parser.add_argument("--preset", choices=["default", "stress"], default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=240, help="Max simulation frames to render")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=64, help="Path tracing samples per pixel")
    parser.add_argument("--camera", choices=sorted(CAMERA_PRESETS), default="hero_front_diag")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/franka_stir_liquid_cinematic"))
    parser.add_argument("--no-denoiser", action="store_true", help="Disable OptiX denoiser")
    parser.add_argument(
        "--camera-list",
        type=str,
        default="",
        help="Comma-separated list of cameras to render (multi-view mode uses subprocesses).",
    )
    return parser.parse_args()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _encode_video(frame_paths: List[Path], output_path: Path, fps: int) -> bool:
    # Try imageio/FFMPEG first
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=9, format="FFMPEG") as writer:
            for frame_path in sorted(frame_paths):
                writer.append_data(imageio.imread(str(frame_path)))
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
    except Exception:
        pass
    # Fallback: OpenCV (mp4v codec)
    try:
        import cv2
        import numpy as np
        paths = sorted(frame_paths)
        first = cv2.imread(str(paths[0]))
        if first is None:
            return False
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is not None:
                writer.write(frame)
        writer.release()
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        print(f"[render] Video encode failed: {exc}", file=sys.stderr)
        return False


def _configure_rtx_path_tracing(spp: int, use_denoiser: bool) -> None:
    """Switch render mode to RTX Path Tracing and set quality parameters."""
    try:
        import carb.settings
        cfg = carb.settings.get_settings()
        cfg.set("/rtx/rendermode", "PathTracing")
        cfg.set("/rtx/pathtracing/spp", spp)
        cfg.set("/rtx/pathtracing/totalSpp", spp * 4)
        cfg.set("/rtx/pathtracing/clampSpp", 0)
        cfg.set("/rtx/pathtracing/maxBounces", 8)
        cfg.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
        cfg.set("/rtx/pathtracing/lightcache/cached/enabled", True)
        if use_denoiser:
            cfg.set("/rtx/pathtracing/optixDenoiser/enabled", True)
            cfg.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.1)
        cfg.set("/omni/replicator/captureOnPlay", False)
        cfg.set("rtx/post/dlss/execMode", 2)
        print(f"[render] RTX Path Tracing configured: spp={spp}, denoiser={use_denoiser}")
    except Exception as exc:
        print(f"[render] RTX config warning: {exc}", file=sys.stderr)


def _render_single(scene, args: argparse.Namespace, camera_name: str, output_dir: Path) -> Dict[str, object]:
    config = scene.get_preset_config(args.preset)
    config["seed"] = args.seed
    config["headless"] = True
    config["video"]["enabled"] = False

    env = scene.FrankaStirLiquidEnv(config=config, preset=args.preset, backend="isaac")
    env.create_scene()
    env.reset_scene(args.seed)

    # omni imports only valid after SimulationApp starts inside create_scene()
    import omni.replicator.core as rep

    _configure_rtx_path_tracing(spp=args.spp, use_denoiser=not args.no_denoiser)

    camera_cfg = CAMERA_PRESETS[camera_name]
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    camera = rep.create.camera(
        position=camera_cfg["position"],
        look_at=camera_cfg["look_at"],
        focal_length=camera_cfg["focal_length"],
        name=f"CinematicCam_{camera_name}",
    )
    render_product = rep.create.render_product(camera, (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frame_dir), rgb=True)
    writer.attach(render_product)
    rep.orchestrator.preview()

    captured = 0
    done_phase_count = 0
    try:
        for frame_idx in range(args.frames):
            obs = env.get_obs()
            phase = obs["task"]["phase"]
            if phase == "done":
                done_phase_count += 1
                if done_phase_count >= 30:
                    break
            metrics = env.get_metrics()
            print(
                f"\r[render] frame {frame_idx + 1}/{args.frames}  phase={phase:12s}"
                f"  spill={metrics.get('spill_count', 0):4d}"
                f"  fill={metrics.get('fill_ratio', 0):.3f}",
                end="",
                flush=True,
            )
            env.step_sim(1)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            captured += 1
        print()
        rep.orchestrator.wait_until_complete()
    finally:
        writer.detach()
        render_product.destroy()

    frame_paths = sorted(frame_dir.glob("rgb_*.png"))
    if not frame_paths:
        raise RuntimeError("No RGB frames produced by BasicWriter.")

    video_path = output_dir / "cinematic.mp4"
    encoded = _encode_video(frame_paths, video_path, args.fps)
    if not encoded:
        scene.write_placeholder_video(video_path)

    final_metrics = env.get_metrics()
    result = {
        "camera": camera_name,
        "camera_config": {k: v for k, v in camera_cfg.items() if k != "description"},
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "frames_captured": captured,
        "video": str(video_path),
        "frame_dir": str(frame_dir),
        "fluid_mode": "physx_particles" if not getattr(env._backend, "_uses_synthetic_fluid", True) else "synthetic",
        "final_metrics": final_metrics,
    }
    env.close()
    return result


def _render_multiview(args: argparse.Namespace, output_dir: Path, camera_names: List[str]) -> List[Dict[str, object]]:
    import subprocess
    results = []
    script_path = Path(__file__).resolve()
    for cam in camera_names:
        cam_dir = output_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(script_path),
            "--preset", args.preset,
            "--seed", str(args.seed),
            "--frames", str(args.frames),
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--spp", str(args.spp),
            "--camera", cam,
            "--output-dir", str(cam_dir),
        ]
        if args.no_denoiser:
            cmd.append("--no-denoiser")
        subprocess.run(cmd, check=True)
        summary_path = cam_dir / "summary.json"
        if summary_path.exists():
            child = json.loads(summary_path.read_text(encoding="utf-8"))
            results.append(child.get("renders", [{}])[0])
    return results


def main() -> int:
    args = _parse_args()
    scene = _load_scene_module()
    output_dir = args.output_dir.resolve()

    camera_names: List[str] = []
    if args.camera_list:
        for name in args.camera_list.split(","):
            name = name.strip()
            if name not in CAMERA_PRESETS:
                print(f"[render] Unknown camera preset: {name}", file=sys.stderr)
                return 1
            camera_names.append(name)
    if not camera_names:
        camera_names = [args.camera]

    _reset_dir(output_dir)

    if len(camera_names) > 1:
        results = _render_multiview(args, output_dir, camera_names)
    else:
        try:
            result = _render_single(scene, args, camera_names[0], output_dir)
            results = [result]
        except Exception as exc:
            print(f"[render] Isaac render failed: {exc}", file=sys.stderr)
            print("[render] Falling back to mock episode render...", file=sys.stderr)
            config = scene.get_preset_config(args.preset)
            config["video"]["enabled"] = True
            config["video"]["fps"] = args.fps
            env = scene.FrankaStirLiquidEnv(config=config, preset=args.preset, backend="mock")
            summary, artifacts = env.run_episode(seed=args.seed, output_dir=output_dir)
            env.close()
            results = [{"camera": "mock_2d", "video": str(artifacts.video_path), "summary": summary}]

    summary_payload = {
        "preset": args.preset,
        "seed": args.seed,
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "cameras": camera_names,
        "renders": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
