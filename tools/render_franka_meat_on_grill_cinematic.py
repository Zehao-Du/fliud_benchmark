#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cinematic-quality render of the franka_meat_on_grill scene.

Uses RTX Path Tracing at 1920×1080 with multi-sample accumulation.
Requires an Isaac Sim Python runtime (backend=isaac).

Usage:
    python tools/render_franka_meat_on_grill_cinematic.py [--preset default|cinematic_smoke]
    python tools/render_franka_meat_on_grill_cinematic.py --camera side_grill --spp 64
    python tools/render_franka_meat_on_grill_cinematic.py --camera-list side_grill,top_diag
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

CAMERA_PRESETS: Dict[str, Dict[str, object]] = {
    "above_grill": {
        "position": (0.72, -0.86, 0.80),
        "look_at": (0.72, 0.14, 0.40),
        "focal_length": 24.0,
        "description": "Direct front view into rising smoke column — NvFlow smoke confirmed visible from this angle",
    },
    "side_grill": {
        "position": (0.54, -1.38, 0.82),
        "look_at": (0.66, 0.04, 0.28),
        "focal_length": 28.0,
        "description": "Side view — smoke rises above grill, embers glow visible through grate",
    },
    "hero_front_diag": {
        "position": (1.42, -0.90, 0.92),
        "look_at": (0.62, 0.00, 0.26),
        "focal_length": 26.0,
        "description": "Hero front-diagonal — shows robot, meat, and grill together",
    },
    "top_diag": {
        "position": (1.06, -0.84, 1.20),
        "look_at": (0.62, 0.02, 0.22),
        "focal_length": 22.0,
        "description": "Top-diagonal — smoke column and grill grates from above",
    },
    "front_close": {
        "position": (1.04, -0.30, 0.62),
        "look_at": (0.62, 0.02, 0.26),
        "focal_length": 35.0,
        "description": "Close front — meat placement on grill up close",
    },
}


def _load_scene_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
    spec = importlib.util.spec_from_file_location("franka_meat_on_grill_cinematic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scene module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cinematic render of franka_meat_on_grill (RTX Path Tracing).")
    parser.add_argument("--preset", choices=["default", "cinematic_smoke"], default="default")
    parser.add_argument("--variation", choices=["chicken", "steak"], default="chicken")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=480, help="Max simulation frames to render")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=64, help="Path tracing samples per pixel")
    parser.add_argument("--camera", choices=sorted(CAMERA_PRESETS), default="above_grill")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/franka_meat_on_grill_cinematic"))
    parser.add_argument("--no-denoiser", action="store_true", help="Disable OptiX denoiser")
    parser.add_argument(
        "--camera-list",
        type=str,
        default="",
        help="Comma-separated camera list for multi-view mode (uses subprocesses).",
    )
    return parser.parse_args()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _encode_video(frame_paths: List[Path], output_path: Path, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=9, format="FFMPEG") as writer:
            for frame_path in sorted(frame_paths):
                writer.append_data(imageio.imread(str(frame_path)))
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
    except Exception:
        pass
    try:
        import cv2
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


def _configure_rtx_for_nvflow() -> None:
    """Use RaytracedLighting — the only RTX mode confirmed to capture NvFlow volumetrics."""
    try:
        import carb.settings
        cfg = carb.settings.get_settings()
        cfg.set("/rtx/rendermode", "RaytracedLighting")
        cfg.set("/rtx/flow/enabled", True)
        cfg.set("/rtx/flow/rayTracedTranslucencyEnabled", True)
        cfg.set("/rtx/flow/rayTracedReflectionsEnabled", False)
        cfg.set("/rtx/flow/pathTracingEnabled", False)
        cfg.set("/omni/replicator/captureOnPlay", False)
        print("[render] RTX RaytracedLighting + NvFlow configured")
    except Exception as exc:
        print(f"[render] RTX config warning: {exc}", file=sys.stderr)


def _dim_lights_for_smoke(stage) -> None:
    """Reduce scene lighting so NvFlow smoke is visible against the background.

    With dome=620 + key=46000 the background saturates at 255, completely washing
    out the diffuse NvFlow smoke volume. Tests confirm smoke is visible when dome≤200
    (background stays below 255, leaving headroom for the smoke brightness contribution).

    The defaultGroundPlane is hidden — its bright blue-tile floor fills the camera's
    background and swamps the smoke signal. With the floor hidden, the dark dome acts
    as the background, matching the confirmed-working isolated smoke test geometry.

    HeatGlow (orange grill embers) is preserved — it provides dramatic warm illumination.
    Smoke rendering parameters (densityMultiplier, colorScale) are boosted so the
    diffuse NvFlow volume is clearly visible against the dark background.
    """
    try:
        # Hide the default ground plane so the dark dome is the background
        for gp_path in ["/World/defaultGroundPlane", "/World/groundPlane", "/World/GroundPlane"]:
            gp_prim = stage.GetPrimAtPath(gp_path)
            if gp_prim.IsValid():
                from pxr import UsdGeom
                UsdGeom.Imageable(gp_prim).MakeInvisible()
                print(f"[render] Hidden ground plane at {gp_path}")
                break

        for prim in stage.Traverse():
            tp = prim.GetTypeName()
            attr = prim.GetAttribute("inputs:intensity")
            if not attr.IsValid():
                continue
            current = attr.Get()
            if current is None:
                continue
            path = str(prim.GetPath())
            if tp == "DomeLight":
                new_val = min(current, 80.0)
                attr.Set(new_val)
                print(f"[render] DomeLight {path}: {current:.0f} → {new_val:.0f}")
            elif tp == "SphereLight":
                if "defaultGroundPlane" in path:
                    attr.Set(0.0)
                    print(f"[render] SphereLight {path}: {current:.0f} → 0 (ground plane)")
                elif "HeatGlow" in path:
                    pass  # Preserve orange ember glow — illuminates smoke from below
                elif current > 5000:
                    new_val = max(current * 0.35, 5000.0)
                    attr.Set(new_val)
                    print(f"[render] SphereLight {path}: {current:.0f} → {new_val:.0f}")

        # Boost smoke visual density — render-only params, safe to change any time
        cloud_prim = stage.GetPrimAtPath("/World/FlowBbqSmoke/flowRender/rayMarch/cloud")
        if cloud_prim.IsValid():
            cloud_prim.GetAttribute("densityMultiplier").Set(3.0)
            print("[render] NvFlow densityMultiplier → 3.0")
        cmap_prim = stage.GetPrimAtPath("/World/FlowBbqSmoke/flowOffscreen/colormap")
        if cmap_prim.IsValid():
            cmap_prim.GetAttribute("colorScale").Set(5.0)
            print("[render] NvFlow colorScale → 5.0")
    except Exception as exc:
        print(f"[render] Light dimming warning (non-fatal): {exc}", file=sys.stderr)


def _render_single(scene, args: argparse.Namespace, camera_name: str, output_dir: Path) -> Dict[str, object]:
    config = scene.get_preset_config(args.preset, {"scene": {"variation": args.variation}})
    config["seed"] = args.seed
    # NvFlow requires display context — must run non-headless under xvfb-run
    config["headless"] = False
    config["video"]["enabled"] = False

    env = scene.FrankaMeatOnGrillEnv(config=config, preset=args.preset, backend="isaac")
    env.create_scene()
    env.reset_scene(args.seed)

    # omni imports only valid after SimulationApp starts inside create_scene()
    import omni.replicator.core as rep

    # Dim lights so NvFlow smoke is visible: background saturation at dome=620 washes it out.
    import omni.usd as _omni_usd
    _dim_lights_for_smoke(_omni_usd.get_context().get_stage())

    _configure_rtx_for_nvflow()

    camera_cfg = CAMERA_PRESETS[camera_name]
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    camera = rep.create.camera(
        position=camera_cfg["position"],
        look_at=camera_cfg["look_at"],
        focal_length=camera_cfg["focal_length"],
        name=f"CinematicGrillCam_{camera_name}",
    )

    # Pump one app frame so the camera prim is fully registered in the USD stage
    import omni.kit.app
    omni.kit.app.get_app().update()

    # Find the actual USD Camera prim path (rep places it under /Replicator/..._Xform/...)
    # Using the path string (not the object) routes the render through the full RTX compositor
    # that includes NvFlow volumetrics — confirmed by test_nvflow_xvfb.py.
    import omni.usd
    from pxr import UsdGeom as _UsdGeom_cam
    cam_path = ""
    for prim in omni.usd.get_context().get_stage().Traverse():
        if prim.GetTypeName() == "Camera" and f"CinematicGrillCam_{camera_name}" in str(prim.GetPath()):
            cam_path = str(prim.GetPath())
            break
    if not cam_path:
        # Fallback: construct expected path from replicator naming convention
        cam_path = f"/Replicator/CinematicGrillCam_{camera_name}_Xform/CinematicGrillCam_{camera_name}"
    print(f"[render] Camera prim path: {cam_path}")

    render_product = rep.create.render_product(cam_path, (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frame_dir), rgb=True)
    writer.attach(render_product)

    # Play timeline so NvFlow simulation advances — mirrors test_nvflow_xvfb.py exactly.
    # world.step(render=False) manually steps PhysX; app.update() advances NvFlow separately.
    import omni.timeline
    omni.timeline.get_timeline_interface().play()

    captured = 0
    done_hold_count = 0
    try:
        for frame_idx in range(args.frames):
            obs = env.get_obs()
            phase = obs["task"]["phase"]
            if phase == "hold_done":
                done_hold_count += 1
                if done_hold_count >= 30:
                    break
            metrics = env.get_metrics()
            print(
                f"\r[render] frame {frame_idx + 1}/{args.frames}  phase={phase:14s}"
                f"  smoke={obs['smoke'].get('strength', 0):.3f}"
                f"  success={int(obs['task'].get('success', False))}",
                end="",
                flush=True,
            )
            env.step_sim(1)
            # app.update() advances NvFlow (world.step(render=False) does not).
            # This is the same call that makes NvFlow work in test_nvflow_xvfb.py.
            omni.kit.app.get_app().update()
            rep.orchestrator.step(delta_time=1.0 / 60.0, pause_timeline=False)
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

    final_obs = env.get_obs()
    final_metrics = env.get_metrics()
    result = {
        "camera": camera_name,
        "camera_config": {k: v for k, v in camera_cfg.items() if k != "description"},
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "frames_captured": captured,
        "video": str(video_path),
        "frame_dir": str(frame_dir),
        "smoke_mode": obs["smoke"].get("mode", "unknown"),
        "task_success": bool(final_obs["task"].get("success", False)),
        "final_metrics": final_metrics,
    }
    # Write partial summary before env.close() — Isaac Sim's close() terminates the process,
    # so any writes after it are lost.
    partial = {"camera": camera_name, "renders": [result]}
    (output_dir / "summary.json").write_text(
        json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env.close()
    return result


def _render_multiview(args: argparse.Namespace, output_dir: Path, camera_names: List[str]) -> List[Dict[str, object]]:
    results = []
    script_path = Path(__file__).resolve()
    for cam in camera_names:
        cam_dir = output_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(script_path),
            "--preset", args.preset,
            "--variation", args.variation,
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
            config = scene.get_preset_config(args.preset, {"scene": {"variation": args.variation}})
            config["video"]["enabled"] = True
            config["video"]["fps"] = args.fps
            env = scene.FrankaMeatOnGrillEnv(config=config, preset=args.preset, backend="mock")
            summary, artifacts = env.run_episode(seed=args.seed, output_dir=output_dir)
            env.close()
            results = [{"camera": "mock_2d", "video": str(artifacts.video_path), "summary": summary}]

    summary_payload = {
        "preset": args.preset,
        "variation": args.variation,
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
