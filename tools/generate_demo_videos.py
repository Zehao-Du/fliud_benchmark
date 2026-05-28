#!/usr/bin/env python3
"""Generate success and failure demo videos for all benchmark tasks.

Each task produces two videos:
  outputs/benchmark_videos/<task_id>/success.mp4
  outputs/benchmark_videos/<task_id>/failure.mp4

Uses mock backend (no Isaac Sim required). For cinematic quality use
tools/render_franka_meat_on_grill_cinematic.py (requires Xvfb + Isaac).

Usage:
    python tools/generate_demo_videos.py
    python tools/generate_demo_videos.py --scene grill
    python tools/generate_demo_videos.py --task grill_chicken_standard
    python tools/generate_demo_videos.py --scene liquid --workers 4
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.tasks import ALL_TASKS, GRILL_TASKS, LIQUID_TASKS, STEAM_TASKS, BenchmarkTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE_CACHE: Dict[str, Any] = {}


def _load_scene_module(scene: str):
    if scene in _MODULE_CACHE:
        return _MODULE_CACHE[scene]
    repo_root = Path(__file__).resolve().parents[1]
    if scene == "liquid":
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
        name = "franka_stir_liquid_vidgen"
    elif scene == "steam":
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_steam_tasks.py"
        name = "franka_steam_tasks_vidgen"
    else:
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
        name = "franka_meat_on_grill_vidgen"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[scene] = mod
    return mod


def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        elif v is not None:
            base[k] = v
    return base


def _render_trajectory_video(traj_csv: Path, out_video: Path, scene: str, success: bool) -> None:
    """Render a lightweight top-down 2-D replay video from trajectory CSV using imageio-ffmpeg."""
    try:
        import imageio.v2 as iio
        import numpy as np
        import csv

        rows = []
        if traj_csv.exists():
            with open(traj_csv, newline="") as f:
                rows = list(csv.DictReader(f))

        W, H = 480, 320
        fps = 24

        # Scene bounds for top-down view
        if scene == "grill":
            xmin, xmax, ymin, ymax = 0.22, 0.92, -0.38, 0.34
        elif scene == "steam":
            xmin, xmax, ymin, ymax = 0.28, 0.92, -0.32, 0.32
        else:
            xmin, xmax, ymin, ymax = 0.38, 0.74, -0.22, 0.22

        def world_to_px(x, y):
            px = int((x - xmin) / (xmax - xmin) * (W - 40) + 20)
            py = int((1 - (y - ymin) / (ymax - ymin)) * (H - 60) + 20)
            return max(0, min(W - 1, px)), max(0, min(H - 1, py))

        def draw_circle(img, cx, cy, r, color):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < W and 0 <= py < H:
                            img[py, px] = color

        # Title color: green=success, red=failure
        title_color = (80, 180, 80) if success else (200, 70, 70)

        frames = []
        # Subsample: render at most 120 frames
        step = max(1, len(rows) // 120)
        for i, row in enumerate(rows[::step]):
            img = np.full((H, W, 3), 45, dtype=np.uint8)
            # Draw grill/container target region
            if scene == "grill":
                tx, ty = world_to_px(0.72, 0.14)
                for dy in range(-18, 19):
                    for dx in range(-24, 25):
                        if 0 <= tx + dx < W and 0 <= ty + dy < H:
                            img[ty + dy, tx + dx] = (60, 40, 20)
            elif scene == "steam":
                # Draw pot/chamber target as cyan circle
                try:
                    ttx_w = float(row.get("target_x", 0.70))
                    tty_w = float(row.get("target_y", 0.10))
                    ttx, tty = world_to_px(ttx_w, tty_w)
                    for dy in range(-16, 17):
                        for dx in range(-16, 17):
                            if dx * dx + dy * dy <= 16 * 16:
                                if 0 <= ttx + dx < W and 0 <= tty + dy < H:
                                    img[tty + dy, ttx + dx] = (30, 80, 80)
                except Exception:
                    pass
            else:
                cx, cy = world_to_px(0.56, 0.0)
                for dy in range(-20, 21):
                    for dx in range(-28, 29):
                        if 0 <= cx + dx < W and 0 <= cy + dy < H:
                            img[cy + dy, cx + dx] = (30, 60, 90)

            # Tool position
            try:
                tx_w, ty_w = float(row.get("tool_x", 0.5)), float(row.get("tool_y", 0))
                tpx, tpy = world_to_px(tx_w, ty_w)
                draw_circle(img, tpx, tpy, 7, (220, 220, 220))
            except Exception:
                pass

            # Object/target
            if scene == "grill":
                try:
                    mx_w = float(row.get("target_x", 0.72))
                    my_w = float(row.get("target_y", 0.14))
                    mpx, mpy = world_to_px(mx_w, my_w)
                    draw_circle(img, mpx, mpy, 5, (220, 140, 60))
                except Exception:
                    pass
            elif scene == "steam":
                # Draw object position + steam centroid
                try:
                    ox_w = float(row.get("obj_x", row.get("target_x", 0.7)))
                    oy_w = float(row.get("obj_y", row.get("target_y", 0.1)))
                    opx, opy = world_to_px(ox_w, oy_w)
                    draw_circle(img, opx, opy, 5, (220, 160, 80))
                except Exception:
                    pass
                try:
                    sx_w = float(row.get("steam_x", 0.7))
                    sy_w = float(row.get("steam_y", 0.1))
                    ss = float(row.get("steam_strength", 0))
                    if ss > 0.05:
                        spx, spy = world_to_px(sx_w, sy_w)
                        draw_circle(img, spx, spy, max(3, int(ss * 12)), (160, 210, 220))
                except Exception:
                    pass
            else:
                try:
                    fx_w = float(row.get("centroid_x", 0.56))
                    fy_w = float(row.get("centroid_y", 0))
                    fpx, fpy = world_to_px(fx_w, fy_w)
                    draw_circle(img, fpx, fpy, 4, (80, 140, 210))
                except Exception:
                    pass

            # Status bar at bottom
            phase = row.get("phase", "")
            img[H - 28:H, :] = (30, 30, 30)
            # Color bar for phase
            phase_color = {"approach": (100, 100, 200), "grasp": (200, 150, 50),
                           "lift": (50, 200, 100), "done": title_color}.get(phase, (120, 120, 120))
            bar_w = int((i + 1) / max(len(rows[::step]), 1) * (W - 4))
            img[H - 10:H - 2, 2:2 + bar_w] = phase_color

            # Success/failure indicator dot top-right
            dot_color = (80, 200, 80) if success else (200, 80, 80)
            draw_circle(img, W - 16, 16, 10, dot_color)

            frames.append(img)

        if not frames:
            frames = [np.full((H, W, 3), 30, dtype=np.uint8)]

        # Use imageio-ffmpeg
        import imageio.plugins.ffmpeg as ff_plugin
        ffmpeg_exe = ff_plugin.get_exe()

        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(str(out_video), fps=fps, codec="libx264",
                                quality=7, ffmpeg_log_level="quiet",
                                output_params=["-pix_fmt", "yuv420p"])
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    except Exception as exc:
        # Fall back to placeholder
        out_video.parent.mkdir(parents=True, exist_ok=True)
        out_video.write_bytes(b"placeholder")
        print(f"  [vidgen] render warning: {exc}")


def _run_one(task: BenchmarkTask, use_failure: bool, out_video: Path, backend: str) -> bool:
    """Run one episode and write a video. Returns success flag."""
    mod = _load_scene_module(task.scene)

    if task.scene == "liquid":
        config = mod.get_preset_config(task.preset)
    elif task.scene == "steam":
        config = mod.get_preset_config(task.task_type, task.preset)
    else:
        config = mod.get_preset_config(task.preset, {"scene": {"variation": task.variation}})

    config["seed"] = task.seed
    config["video"]["enabled"] = False  # use our own fast renderer below
    config["headless"] = True

    overrides = task.failure_overrides if use_failure else task.config_overrides
    _deep_update(config, overrides)

    if task.scene == "liquid" and backend == "mock":
        config["scene"]["fluid"]["particle_spacing"] = 0.018

    out_video.parent.mkdir(parents=True, exist_ok=True)

    if task.scene == "liquid":
        env_cls = mod.FrankaStirLiquidEnv
    elif task.scene == "steam":
        env_cls = mod.FrankaSteamTaskEnv
    else:
        env_cls = mod.FrankaMeatOnGrillEnv

    if task.scene == "steam":
        env = env_cls(task_type=task.task_type, config=config, preset=task.preset, backend=backend)
    else:
        env = env_cls(config=config, preset=task.preset, backend=backend)

    with tempfile.TemporaryDirectory(prefix="vidgen_") as tmp:
        summary, artifacts = env.run_episode(seed=task.seed, output_dir=Path(tmp))
        env.close()
        traj_csv = artifacts.trajectory_path if hasattr(artifacts, "trajectory_path") else Path(tmp) / "trajectory.csv"
        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        if task.scene == "liquid":
            success = bool(metrics.get("object_lifted", False))
        elif task.scene == "steam":
            success = bool(metrics.get("success", False))
        else:
            success = bool(metrics.get("success", False))
        _render_trajectory_video(traj_csv, out_video, task.scene, success)
    return success


def _worker(args_tuple: Tuple) -> Dict:
    task, use_failure, out_video, backend = args_tuple
    mode = "failure" if use_failure else "success"
    try:
        success = _run_one(task, use_failure, out_video, backend)
        status = "ok"
    except Exception as exc:
        success = False
        status = f"ERROR: {exc}"
    return {
        "task_id": task.task_id,
        "mode": mode,
        "success": success,
        "video": str(out_video),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark demo videos (success + failure)")
    parser.add_argument("--scene", choices=["liquid", "grill", "steam", "all"], default="all")
    parser.add_argument("--task", default=None, help="Single task_id to render")
    parser.add_argument("--backend", choices=["mock", "isaac", "auto"], default="mock")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark_videos"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (>1 needs enough RAM)")
    parser.add_argument("--no-failure", action="store_true", help="Skip failure videos")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.task:
        from benchmark.tasks import get_task
        tasks = [get_task(args.task)]
    elif args.scene == "liquid":
        tasks = LIQUID_TASKS
    elif args.scene == "grill":
        tasks = GRILL_TASKS
    elif args.scene == "steam":
        tasks = STEAM_TASKS
    else:
        tasks = ALL_TASKS

    # Build job list
    jobs = []
    for task in tasks:
        out_dir = args.output_dir / task.scene / task.task_id
        jobs.append((task, False, out_dir / "success.mp4", args.backend))
        if not args.no_failure:
            jobs.append((task, True, out_dir / "failure.mp4", args.backend))

    print(f"[vidgen] Generating {len(jobs)} videos  (tasks={len(tasks)}, workers={args.workers})")
    print(f"[vidgen] Output: {args.output_dir}")

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    print(f"\n[vidgen] Done: {len(ok)}/{len(results)} videos generated")
    if errors:
        for e in errors:
            print(f"  ERROR {e['task_id']} [{e['mode']}]: {e['status']}")

    # Print table
    print(f"\n  {'Task':<36} {'Mode':<10} {'Outcome':<10} {'File'}")
    print(f"  {'-'*80}")
    for r in results:
        outcome = ("✓ success" if r["success"] else "✗ failed") if r["status"] == "ok" else "ERROR"
        path = Path(r["video"])
        fname = f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
        print(f"  {r['task_id']:<36} {r['mode']:<10} {outcome:<10} {fname}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
