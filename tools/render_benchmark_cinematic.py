#!/usr/bin/env python3
"""Batch cinematic render for all FluidBench tasks.

Renders success and failure episodes for all 12 benchmark tasks at
1920×1080 with RTX RaytracedLighting + NvFlow smoke compositing.
Designed to run in a single Isaac Sim session in a tmux background.

Usage (in tmux):
    tmux new -s fluidbench_render
    xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" \\
      conda run -n env_isaaclab python tools/render_benchmark_cinematic.py \\
        --output-dir outputs/benchmark_videos_cinematic 2>&1 | tee /tmp/render_benchmark.log

Quick-test (2 tasks only):
    python tools/render_benchmark_cinematic.py --task grill_chicken_standard --backend mock
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.tasks import ALL_TASKS, BenchmarkTask, get_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        elif v is not None:
            base[k] = v
    return base


def _load_scene_module(scene: str):
    repo_root = Path(__file__).resolve().parents[1]
    if scene == "liquid":
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
        name = "franka_stir_liquid_cinematic"
    elif scene == "steam":
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_steam_tasks.py"
        name = "franka_steam_tasks_cinematic"
    else:
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
        name = "franka_meat_on_grill_cinematic"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _composite_smoke(grill_frames_dir: Path, smoke_frames_dir: Path, out_video: Path) -> None:
    """Additively composite NvFlow smoke frames over grill scene frames."""
    try:
        import numpy as np
        import imageio.v2 as iio
        import imageio.plugins.ffmpeg as ff_plugin

        grill_files = sorted(grill_frames_dir.glob("rgb_*.png"))
        smoke_files = sorted(smoke_frames_dir.glob("rgb_*.png"))

        if not grill_files:
            print("  [composite] No grill frames found, skipping smoke composite")
            return

        frames = []
        for i, gf in enumerate(grill_files):
            g = iio.imread(str(gf))[:, :, :3].astype(np.float32)
            if i < len(smoke_files):
                s = iio.imread(str(smoke_files[i]))[:, :, :3].astype(np.float32)
                comp = np.clip(g + s * 2.0, 0, 255).astype(np.uint8)
            else:
                comp = g.astype(np.uint8)
            frames.append(comp)

        ffmpeg_exe = ff_plugin.get_exe()
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(
            str(out_video), fps=24, codec="libx264", quality=7,
            ffmpeg_log_level="quiet", output_params=["-pix_fmt", "yuv420p"]
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  [composite] Wrote {len(frames)} frames → {out_video}")
    except Exception as exc:
        print(f"  [composite] ERROR: {exc}")


def _frames_to_video(frames_dir: Path, out_video: Path, fps: int = 24) -> None:
    """Encode a directory of PNG frames to MP4."""
    try:
        import imageio.v2 as iio
        import imageio.plugins.ffmpeg as ff_plugin

        files = sorted(frames_dir.glob("rgb_*.png"))
        if not files:
            print(f"  [encode] No frames in {frames_dir}")
            return

        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(
            str(out_video), fps=fps, codec="libx264", quality=7,
            ffmpeg_log_level="quiet", output_params=["-pix_fmt", "yuv420p"]
        )
        for f in files:
            import imageio.v2 as iio2
            frame = iio2.imread(str(f))[:, :, :3]
            writer.append_data(frame)
        writer.close()
        print(f"  [encode] Wrote {len(files)} frames → {out_video}")
    except Exception as exc:
        print(f"  [encode] ERROR: {exc}")


# ---------------------------------------------------------------------------
# Render one task/mode pair
# ---------------------------------------------------------------------------

def render_task(
    task: BenchmarkTask,
    use_failure: bool,
    out_video: Path,
    backend: str,
    frames: int,
    spp: int,
    width: int,
    height: int,
) -> Dict[str, Any]:
    mode = "failure" if use_failure else "success"
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[render] {task.task_id} / {mode}  (backend={backend})")
    print(f"{'='*60}")

    try:
        if backend == "mock":
            # Use the generate_demo_videos fast renderer for mock backend
            _render_mock(task, use_failure, out_video)
            elapsed = time.time() - t0
            print(f"[render] Done in {elapsed:.1f}s → {out_video}")
            return {"task_id": task.task_id, "mode": mode, "status": "ok", "elapsed": elapsed, "video": str(out_video)}

        # Isaac backend: use scene-specific cinematic render
        mod = _load_scene_module(task.scene)

        if task.scene == "liquid":
            _render_isaac_liquid(task, use_failure, out_video, mod, frames, spp, width, height)
        elif task.scene == "steam":
            _render_isaac_steam(task, use_failure, out_video, mod, frames, spp, width, height)
        else:
            _render_isaac_grill(task, use_failure, out_video, mod, frames, spp, width, height)

        elapsed = time.time() - t0
        print(f"[render] Done in {elapsed:.1f}s → {out_video}")
        return {"task_id": task.task_id, "mode": mode, "status": "ok", "elapsed": elapsed, "video": str(out_video)}

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[render] ERROR {task.task_id}/{mode}: {exc}")
        import traceback
        traceback.print_exc()
        return {"task_id": task.task_id, "mode": mode, "status": f"ERROR: {exc}", "elapsed": elapsed, "video": str(out_video)}


def _render_mock(task: BenchmarkTask, use_failure: bool, out_video: Path) -> None:
    """Lightweight 2D mock render (fast, for testing)."""
    import importlib as _il
    repo_root = Path(__file__).resolve().parents[1]
    spec = _il.util.spec_from_file_location(
        "generate_demo_videos_bench",
        repo_root / "tools/generate_demo_videos.py"
    )
    gmod = _il.util.module_from_spec(spec)
    spec.loader.exec_module(gmod)
    gmod._run_one(task, use_failure, out_video, backend="mock")


def _render_isaac_liquid(
    task: BenchmarkTask, use_failure: bool, out_video: Path,
    mod: Any, frames: int, spp: int, width: int, height: int,
) -> None:
    """Isaac cinematic render for liquid scene."""
    import tempfile
    config = mod.get_preset_config(task.preset)
    config["seed"] = task.seed
    config["headless"] = False
    config["video"]["enabled"] = False

    overrides = task.failure_overrides if use_failure else task.config_overrides
    _deep_update(config, overrides)

    with tempfile.TemporaryDirectory(prefix="cinematic_liquid_") as tmp:
        tmp_dir = Path(tmp)
        frames_dir = tmp_dir / "frames"
        frames_dir.mkdir()

        env = mod.FrankaStirLiquidEnv(config=config, preset=task.preset, backend="isaac")
        env.create_scene()
        env.reset_scene(seed=task.seed)

        # Configure RTX
        try:
            import omni.kit.app
            settings = omni.kit.app.get_app().get_settings()
            settings.set("/rtx/rendermode", "RaytracedLighting")
            settings.set("/rtx/raytracing/samplesPerPixel", spp)
            settings.set("/rtx/post/dlss/enabled", False)
        except Exception:
            pass

        # Set up BasicWriter
        try:
            import omni.replicator.core as rep
            cam = rep.create.camera(
                position=(0.3, -0.6, 0.7),
                look_at=(0.56, 0.0, 0.05),
                focal_length=24.0,
            )
            rp = rep.create.render_product(cam, (width, height))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(frames_dir), rgb=True)
            writer.attach([rp])
        except Exception as exc:
            print(f"  [liquid] replicator setup failed: {exc}, using placeholder")
            out_video.parent.mkdir(parents=True, exist_ok=True)
            out_video.write_bytes(b"placeholder")
            env.close()
            return

        step = 0
        while step < frames:
            env.step_sim(1)
            try:
                import omni.kit.app
                omni.kit.app.get_app().update()
            except Exception:
                pass
            step += 1

        env.close()
        _frames_to_video(frames_dir, out_video)


def _render_isaac_steam(
    task: BenchmarkTask, use_failure: bool, out_video: Path,
    mod: Any, frames: int, spp: int, width: int, height: int,
) -> None:
    """Isaac cinematic render for steam/spray scenes."""
    import tempfile
    config = mod.get_preset_config(task.task_type, task.preset)
    config["seed"] = task.seed
    config["headless"] = False
    config["video"]["enabled"] = False

    overrides = task.failure_overrides if use_failure else task.config_overrides
    _deep_update(config, overrides)

    # Camera presets per task type
    _cam_presets = {
        "steam_pot_place":       {"pos": (0.20, -0.55, 0.55), "look": (0.62, 0.05, 0.12)},
        "steam_lid_open_close":  {"pos": (0.25, -0.60, 0.58), "look": (0.68, 0.10, 0.14)},
        "spray_chamber_tray":    {"pos": (0.15, -0.60, 0.52), "look": (0.60, 0.00, 0.14)},
        "steam_valve_close":     {"pos": (0.20, -0.55, 0.48), "look": (0.60, 0.00, 0.16)},
    }
    cam_cfg = _cam_presets.get(task.task_type, {"pos": (0.20, -0.55, 0.50), "look": (0.60, 0.00, 0.12)})

    with tempfile.TemporaryDirectory(prefix="cinematic_steam_") as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir()

        env = mod.FrankaSteamTaskEnv(
            task_type=task.task_type, config=config, preset=task.preset, backend="isaac"
        )
        env.create_scene()
        env.reset_scene(seed=task.seed)

        try:
            import omni.kit.app
            settings = omni.kit.app.get_app().get_settings()
            settings.set("/rtx/rendermode", "RaytracedLighting")
            settings.set("/rtx/raytracing/samplesPerPixel", spp)
            settings.set("/rtx/post/dlss/enabled", False)
        except Exception:
            pass

        try:
            import omni.replicator.core as rep
            cam = rep.create.camera(
                position=cam_cfg["pos"],
                look_at=cam_cfg["look"],
                focal_length=28.0,
            )
            rp = rep.create.render_product(cam, (width, height))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(frames_dir), rgb=True)
            writer.attach([rp])
        except Exception as exc:
            print(f"  [steam] replicator setup failed: {exc}, using placeholder")
            out_video.parent.mkdir(parents=True, exist_ok=True)
            out_video.write_bytes(b"placeholder")
            env.close()
            return

        for _ in range(frames):
            env.step_sim(1)
            try:
                import omni.kit.app
                omni.kit.app.get_app().update()
            except Exception:
                pass

        env.close()
        _frames_to_video(frames_dir, out_video)


def _render_isaac_grill(
    task: BenchmarkTask, use_failure: bool, out_video: Path,
    mod: Any, frames: int, spp: int, width: int, height: int,
) -> None:
    """Isaac cinematic render for grill scene with NvFlow smoke compositing."""
    import tempfile
    import shutil

    config = mod.get_preset_config(task.preset, {"scene": {"variation": task.variation}})
    config["seed"] = task.seed
    config["headless"] = False
    config["video"]["enabled"] = False

    overrides = task.failure_overrides if use_failure else task.config_overrides
    _deep_update(config, overrides)

    with tempfile.TemporaryDirectory(prefix="cinematic_grill_") as tmp:
        tmp_dir = Path(tmp)
        grill_frames = tmp_dir / "grill_frames"
        smoke_frames = tmp_dir / "smoke_frames"
        grill_frames.mkdir()
        smoke_frames.mkdir()

        # --- Render grill scene frames ---
        env = mod.FrankaMeatOnGrillEnv(config=config, preset=task.preset, backend="isaac")
        env.create_scene()
        env.reset_scene(seed=task.seed)

        try:
            import omni.kit.app
            settings = omni.kit.app.get_app().get_settings()
            settings.set("/rtx/rendermode", "RaytracedLighting")
            settings.set("/rtx/raytracing/samplesPerPixel", spp)
            settings.set("/rtx/post/dlss/enabled", False)
        except Exception:
            pass

        try:
            import omni.replicator.core as rep
            cam = rep.create.camera(
                position=(0.3, -0.55, 0.55),
                look_at=(0.65, 0.14, 0.1),
                focal_length=28.0,
            )
            rp = rep.create.render_product(cam, (width, height))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(grill_frames), rgb=True)
            writer.attach([rp])
        except Exception as exc:
            print(f"  [grill] replicator setup failed: {exc}")
            out_video.parent.mkdir(parents=True, exist_ok=True)
            out_video.write_bytes(b"placeholder")
            env.close()
            return

        for _ in range(frames):
            env.step_sim(1)
            try:
                import omni.kit.app
                omni.kit.app.get_app().update()
            except Exception:
                pass
        env.close()

        # --- Render NvFlow smoke in separate cm-stage ---
        bbq_usda = Path(__file__).parent / "bbq_smoke.usda"
        if bbq_usda.exists():
            try:
                _render_nvflow_smoke(bbq_usda, smoke_frames, frames, spp, width, height)
            except Exception as exc:
                print(f"  [smoke] NvFlow render failed: {exc} — compositing without smoke")

        # --- Composite ---
        if list(smoke_frames.glob("rgb_*.png")):
            _composite_smoke(grill_frames, smoke_frames, out_video)
        else:
            _frames_to_video(grill_frames, out_video)


def _render_nvflow_smoke(
    bbq_usda: Path, out_dir: Path, frames: int, spp: int, width: int, height: int,
) -> None:
    """Render NvFlow smoke from bbq_smoke.usda (native cm stage) into out_dir."""
    try:
        import omni.kit.app
        import omni.usd
        from pxr import Usd, UsdGeom

        ctx = omni.usd.get_context()
        ctx.open_stage(str(bbq_usda))
        import omni.kit.app as _app
        for _ in range(5):
            _app.get_app().update()

        import omni.replicator.core as rep
        cam = rep.create.camera(
            position=(20, -60, 50),
            look_at=(0, 0, 20),
            focal_length=24.0,
        )
        rp = rep.create.render_product(cam, (width, height))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(out_dir), rgb=True)
        writer.attach([rp])

        for _ in range(frames):
            _app.get_app().update()

        ctx.close_stage()
    except Exception:
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch cinematic render for all FluidBench tasks (run in tmux)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark_videos_cinematic"))
    parser.add_argument("--backend", choices=["mock", "isaac"], default="isaac")
    parser.add_argument("--scene", choices=["liquid", "grill", "steam", "all"], default="all")
    parser.add_argument("--task", default=None, help="Single task_id (for quick test)")
    parser.add_argument("--frames", type=int, default=240, help="Frames per episode (240=10s@24fps)")
    parser.add_argument("--spp", type=int, default=16, help="RTX samples per pixel")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--no-failure", action="store_true", help="Skip failure episodes")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.task:
        tasks = [get_task(args.task)]
    elif args.scene == "liquid":
        from benchmark.tasks import LIQUID_TASKS
        tasks = LIQUID_TASKS
    elif args.scene == "grill":
        from benchmark.tasks import GRILL_TASKS
        tasks = GRILL_TASKS
    elif args.scene == "steam":
        from benchmark.tasks import STEAM_TASKS
        tasks = STEAM_TASKS
    else:
        tasks = ALL_TASKS

    modes = ["success"] if args.no_failure else ["success", "failure"]
    total = len(tasks) * len(modes)

    print(f"[render_benchmark] {len(tasks)} tasks × {len(modes)} modes = {total} videos")
    print(f"[render_benchmark] Output: {args.output_dir}  backend={args.backend}")
    print(f"[render_benchmark] Resolution: {args.width}×{args.height}  spp={args.spp}  frames={args.frames}")
    print()

    results: List[Dict] = []
    done = 0

    for task in tasks:
        for use_failure in ([False, True] if not args.no_failure else [False]):
            mode = "failure" if use_failure else "success"
            out_video = args.output_dir / task.scene / task.task_id / f"{mode}.mp4"
            result = render_task(
                task, use_failure, out_video,
                backend=args.backend,
                frames=args.frames,
                spp=args.spp,
                width=args.width,
                height=args.height,
            )
            results.append(result)
            done += 1
            ok_count = sum(1 for r in results if r["status"] == "ok")
            print(f"\n[progress] {done}/{total} done  ({ok_count} ok)")

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    total_time = sum(r.get("elapsed", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"[render_benchmark] Completed: {len(ok)}/{total}  Errors: {len(errors)}")
    print(f"[render_benchmark] Total time: {total_time/60:.1f} min")
    if errors:
        print("\nFailed:")
        for e in errors:
            print(f"  {e['task_id']} [{e['mode']}]: {e['status']}")

    print(f"\n  {'Task':<36} {'Mode':<10} {'Status':<10} {'Time':>6}")
    print(f"  {'-'*65}")
    for r in results:
        status = "ok" if r["status"] == "ok" else "ERR"
        t = f"{r.get('elapsed', 0):.0f}s"
        print(f"  {r['task_id']:<36} {r['mode']:<10} {status:<10} {t:>6}")

    # Save results JSON
    results_path = args.output_dir / "render_results.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n[render_benchmark] Results saved → {results_path}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
