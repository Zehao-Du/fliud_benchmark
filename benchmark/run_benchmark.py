#!/usr/bin/env python3
"""Benchmark runner: evaluate a policy (or scripted oracle) on all benchmark tasks.

Usage:
    # Oracle (scripted policy, upper bound):
    python benchmark/run_benchmark.py --policy oracle --backend mock

    # Random baseline:
    python benchmark/run_benchmark.py --policy random --backend mock

    # Trained diffusion policy:
    python benchmark/run_benchmark.py --policy diffusion \
        --liquid-ckpt policy/checkpoints/liquid/best.pt \
        --grill-ckpt  policy/checkpoints/grill/best.pt \
        --backend mock

    # Single task:
    python benchmark/run_benchmark.py --policy oracle --task grill_chicken_standard
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.tasks import ALL_TASKS, GRILL_TASKS, LIQUID_TASKS, BenchmarkTask


# ---------------------------------------------------------------------------
# Scene module loader (cached)
# ---------------------------------------------------------------------------

_MODULE_CACHE: Dict[str, Any] = {}


def _load_scene_module(scene: str):
    if scene in _MODULE_CACHE:
        return _MODULE_CACHE[scene]
    repo_root = Path(__file__).resolve().parents[1]
    if scene == "liquid":
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
        name = "franka_stir_liquid_bench"
    else:
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
        name = "franka_meat_on_grill_bench"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[scene] = mod
    return mod


# ---------------------------------------------------------------------------
# Policy interfaces
# ---------------------------------------------------------------------------

def _make_config(task: BenchmarkTask, mod, use_failure: bool, backend: str) -> Dict[str, Any]:
    """Build episode config from a task definition."""
    if task.scene == "liquid":
        config = mod.get_preset_config(task.preset)
    else:
        config = mod.get_preset_config(task.preset, {"scene": {"variation": task.variation}})
    config["seed"] = task.seed
    config["video"]["enabled"] = False
    config["headless"] = True

    overrides = task.failure_overrides if use_failure else task.config_overrides

    # Apply overrides (shallow-merge per top-level key to preserve nested defaults)
    def _deep_update(base, patch):
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _deep_update(base[k], v)
            elif v is not None:
                base[k] = v
        return base

    _deep_update(config, overrides)

    if task.scene == "liquid" and backend == "mock":
        config["scene"]["fluid"]["particle_spacing"] = 0.018

    return config


def _run_oracle(task: BenchmarkTask, backend: str, use_failure: bool, tmp_dir: Path) -> Dict[str, Any]:
    """Run the scripted oracle policy for one episode."""
    mod = _load_scene_module(task.scene)
    config = _make_config(task, mod, use_failure, backend)

    if task.scene == "liquid":
        env_cls = mod.FrankaStirLiquidEnv
    else:
        env_cls = mod.FrankaMeatOnGrillEnv

    ep_dir = tmp_dir / task.task_id / ("failure" if use_failure else "success")
    ep_dir.mkdir(parents=True, exist_ok=True)

    env = env_cls(config=config, preset=task.preset, backend=backend)
    summary, _ = env.run_episode(seed=task.seed, output_dir=ep_dir)
    env.close()

    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    if task.scene == "liquid":
        success = bool(metrics.get("object_lifted", False))
    else:
        success = bool(metrics.get("success", False))

    return {"success": success, "metrics": metrics, "steps": summary.get("step_count", 0)}


def _run_random(task: BenchmarkTask, backend: str, tmp_dir: Path) -> Dict[str, Any]:
    """Random policy: always fails (no valid actions)."""
    return {"success": False, "metrics": {}, "steps": 0, "policy": "random"}


def _run_diffusion(
    task: BenchmarkTask,
    backend: str,
    liquid_ckpt: Optional[Path],
    grill_ckpt: Optional[Path],
    tmp_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a trained diffusion policy on a single task."""
    import torch
    from policy.diffusion_policy import DiffusionPolicy

    ckpt_path = liquid_ckpt if task.scene == "liquid" else grill_ckpt
    if ckpt_path is None or not ckpt_path.exists():
        print(f"  [bench] No checkpoint for scene={task.scene}, treating as failure.")
        return {"success": False, "metrics": {}, "steps": 0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device)
    policy = DiffusionPolicy(
        obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
        obs_horizon=ckpt["obs_horizon"], pred_horizon=ckpt["pred_horizon"],
        embed_dim=ckpt["embed_dim"], unet_base_ch=ckpt["unet_base_ch"],
        num_diffusion_steps=ckpt["diffusion_steps"],
    ).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    norm_stats = {k: v.to(device) for k, v in ckpt["norm_stats"].items()}

    from collections import deque
    mod = _load_scene_module(task.scene)
    config = _make_config(task, mod, use_failure=False, backend=backend)

    if task.scene == "liquid":
        env = mod.FrankaStirLiquidEnv(config=config, preset=task.preset, backend=backend)
    else:
        env = mod.FrankaMeatOnGrillEnv(config=config, preset=task.preset, backend=backend)

    env.create_scene()
    env.reset_scene(task.seed)

    def _obs_vec(obs):
        tool = obs.get("tool", {}).get("position", [0, 0, 0])
        target = obs.get("task", {}).get("target_position", [0, 0, 0])
        if task.scene == "liquid":
            fluid = obs.get("fluid", {})
            c = fluid.get("centroid", [0, 0, 0])
            fr = float(fluid.get("fill_ratio", 1.0))
            return np.array([*tool, *target, *c, fr], dtype=np.float32)
        else:
            meat = obs.get("meat", {}).get("position", [0, 0, 0])
            smoke = obs.get("smoke", {})
            sc = smoke.get("centroid", [0, 0, 0])
            ss = float(smoke.get("strength", 0.0))
            gr = float(obs.get("robot", {}).get("gripper_open", True))
            return np.array([*tool, *meat, *target, *sc, ss, gr], dtype=np.float32)

    init_obs = env.get_obs()
    obs_buf = deque(maxlen=ckpt["obs_horizon"])
    for _ in range(ckpt["obs_horizon"]):
        obs_buf.append(_obs_vec(init_obs))

    action_queue = deque()
    success = False
    max_steps = 500

    for step in range(max_steps):
        if not action_queue:
            obs_arr = np.stack(list(obs_buf), axis=0)
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            obs_n = 2.0 * (obs_t - norm_stats["obs_min"]) / (norm_stats["obs_max"] - norm_stats["obs_min"]).clamp(min=1e-6) - 1.0
            with torch.no_grad():
                pred = policy.predict_action(obs_n, num_steps=10)[0]
            pred_dn = ((pred + 1.0) / 2.0) * (norm_stats["act_max"] - norm_stats["act_min"]).clamp(min=1e-6) + norm_stats["act_min"]
            for a in pred_dn.cpu().numpy():
                action_queue.append(a)

        action = action_queue.popleft()
        env.apply_tool_action({"target_position": action[:3].tolist(), "gripper_closed": bool(action[3] > 0.5)})
        env.step_sim(1)
        obs = env.get_obs()
        obs_buf.append(_obs_vec(obs))

        phase = obs["task"]["phase"]
        if task.scene == "liquid":
            success = phase == "done"
        else:
            success = bool(obs["task"].get("success", False))
        if success:
            break

    metrics = env.get_metrics()
    env.close()
    return {"success": success, "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, "steps": step + 1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FluidBench evaluation runner")
    parser.add_argument("--policy", choices=["oracle", "random", "diffusion"], default="oracle")
    parser.add_argument("--backend", choices=["mock", "isaac", "auto"], default="mock")
    parser.add_argument("--scene", choices=["liquid", "grill", "all"], default="all")
    parser.add_argument("--task", default=None, help="Run a single task by task_id")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--liquid-ckpt", type=Path, default=None)
    parser.add_argument("--grill-ckpt", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_results.json"))
    parser.add_argument("--include-failure", action="store_true", help="Also run failure-mode configs")
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
    else:
        tasks = ALL_TASKS

    import tempfile
    results: List[Dict[str, Any]] = []

    print(f"\n{'='*62}")
    print(f"  FluidBench  |  policy={args.policy}  backend={args.backend}")
    print(f"{'='*62}")
    print(f"  {'Task':<36} {'Diff':<8} {'Success':>8}")
    print(f"  {'-'*56}")

    with tempfile.TemporaryDirectory(prefix="fluidbench_") as tmp:
        tmp_dir = Path(tmp)

        for task in tasks:
            if args.scene != "all" and task.scene != args.scene:
                continue

            ep_results = []
            t0 = time.time()
            for ep in range(args.episodes_per_task):
                try:
                    if args.policy == "oracle":
                        r = _run_oracle(task, args.backend, use_failure=False, tmp_dir=tmp_dir)
                    elif args.policy == "random":
                        r = _run_random(task, args.backend, tmp_dir=tmp_dir)
                    else:
                        r = _run_diffusion(task, args.backend, args.liquid_ckpt, args.grill_ckpt, tmp_dir=tmp_dir)
                    ep_results.append(r)
                except Exception as exc:
                    print(f"    [bench] {task.task_id} ep={ep} ERROR: {exc}")
                    ep_results.append({"success": False, "metrics": {}, "steps": 0})

            sr = sum(1 for r in ep_results if r["success"]) / max(len(ep_results), 1)
            elapsed = time.time() - t0
            print(f"  {task.task_id:<36} {task.difficulty:<8} {sr*100:>6.0f}%  ({elapsed:.0f}s)")

            results.append({
                "task_id": task.task_id,
                "scene": task.scene,
                "difficulty": task.difficulty,
                "policy": args.policy,
                "backend": args.backend,
                "episodes": ep_results,
                "success_rate": sr,
            })

    # Summary
    overall_sr = np.mean([r["success_rate"] for r in results]) if results else 0.0
    liquid_sr  = np.mean([r["success_rate"] for r in results if r["scene"] == "liquid"]) if any(r["scene"] == "liquid" for r in results) else float("nan")
    grill_sr   = np.mean([r["success_rate"] for r in results if r["scene"] == "grill"])  if any(r["scene"] == "grill"  for r in results) else float("nan")

    print(f"\n  {'─'*56}")
    print(f"  Overall success rate : {overall_sr*100:.1f}%")
    print(f"  Liquid tasks         : {liquid_sr*100:.1f}%")
    print(f"  Grill  tasks         : {grill_sr*100:.1f}%")
    print(f"{'='*62}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"overall": overall_sr, "liquid": liquid_sr, "grill": grill_sr, "tasks": results}, f, indent=2)
    print(f"  Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
