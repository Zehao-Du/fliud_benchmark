#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch demo data collection: runs multi-seed rollouts and saves to HDF5.

Usage:
    python tools/collect_demo_data.py --scene liquid --num-episodes 50 --backend mock
    python tools/collect_demo_data.py --scene grill  --num-episodes 20 --backend isaac --seeds 0,1,2
    python tools/collect_demo_data.py --scene liquid --num-episodes 100 --output outputs/demo_liquid.h5
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


# ---------------------------------------------------------------------------
# Schema helpers  (uses actual CSV columns from trajectory.csv)
# ---------------------------------------------------------------------------
# Liquid CSV columns: step,phase,target_x,target_y,target_z,tool_x,tool_y,tool_z,
#   centroid_x,centroid_y,centroid_z,spill_count,fill_ratio,...
# Grill CSV columns: step,phase,task_done,grasped,success,tool_x,tool_y,tool_z,
#   meat_x,meat_y,meat_z,target_x,target_y,target_z,smoke_x,smoke_y,smoke_z,
#   smoke_strength,gripper_open,in_target_zone

OBS_DIM = {"liquid": 10, "grill": 14}
ACTION_DIM = 4  # 3 target_position + 1 gripper_closed


def _extract_obs_vec(row: Dict[str, str], scene: str) -> np.ndarray:
    """Extract flat float32 obs vector from a single trajectory CSV row."""
    if scene == "liquid":
        # tool position (3) + target position (3) + fluid centroid (3) + fill_ratio (1) = 10
        return np.array([
            float(row.get("tool_x", 0.0)), float(row.get("tool_y", 0.0)), float(row.get("tool_z", 0.0)),
            float(row.get("target_x", 0.0)), float(row.get("target_y", 0.0)), float(row.get("target_z", 0.0)),
            float(row.get("centroid_x", 0.0)), float(row.get("centroid_y", 0.0)), float(row.get("centroid_z", 0.0)),
            float(row.get("fill_ratio", 1.0)),
        ], dtype=np.float32)
    else:
        # tool pos (3) + meat pos (3) + target pos (3) + smoke centroid (3) + smoke_strength (1) + gripper (1) = 14
        gripper_val = row.get("gripper_open", "True")
        gripper_f = 1.0 if str(gripper_val).lower() in ("true", "1") else 0.0
        return np.array([
            float(row.get("tool_x", 0.0)), float(row.get("tool_y", 0.0)), float(row.get("tool_z", 0.0)),
            float(row.get("meat_x", 0.0)), float(row.get("meat_y", 0.0)), float(row.get("meat_z", 0.0)),
            float(row.get("target_x", 0.0)), float(row.get("target_y", 0.0)), float(row.get("target_z", 0.0)),
            float(row.get("smoke_x", 0.0)), float(row.get("smoke_y", 0.0)), float(row.get("smoke_z", 0.0)),
            float(row.get("smoke_strength", 0.0)),
            gripper_f,
        ], dtype=np.float32)


def _extract_action(row: Dict[str, str], scene: str) -> np.ndarray:
    """Return the immediate action: target EE position + gripper_closed."""
    gripper_val = row.get("gripper_open", "True")
    gripper_open = 1.0 if str(gripper_val).lower() in ("true", "1") else 0.0
    gripper_closed = 1.0 - gripper_open
    return np.array([
        float(row.get("target_x", 0.0)),
        float(row.get("target_y", 0.0)),
        float(row.get("target_z", 0.0)),
        gripper_closed,
    ], dtype=np.float32)


def _load_scene_module(scene: str):
    repo_root = Path(__file__).resolve().parents[1]
    if scene == "liquid":
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
        name = "franka_stir_liquid_collect"
    else:
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
        name = "franka_meat_on_grill_collect"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scene module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def _run_episode(
    scene_module,
    scene: str,
    seed: int,
    backend: str,
    preset: str,
    tmp_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run one episode, return raw trajectory dict or None on failure."""
    try:
        if scene == "liquid":
            config = scene_module.get_preset_config(preset)
            config["seed"] = seed
            config["headless"] = True
            config["video"]["enabled"] = False
            # Coarsen particle grid for mock backend: 0.0055→0.018 reduces ~15k particles to ~100
            # Robot kinematics data is unchanged; fluid centroid/fill_ratio remain valid approximations
            if backend == "mock":
                config["scene"]["fluid"]["particle_spacing"] = 0.018
            env_cls = scene_module.FrankaStirLiquidEnv
        else:
            config = scene_module.get_preset_config(preset, {})
            config["seed"] = seed
            config["headless"] = True
            config["video"]["enabled"] = False
            env_cls = scene_module.FrankaMeatOnGrillEnv

        ep_dir = tmp_dir / f"ep_{seed:05d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        env = env_cls(config=config, preset=preset, backend=backend)
        summary, artifacts = env.run_episode(seed=seed, output_dir=ep_dir)
        env.close()

        # Read trajectory CSV
        traj_csv = artifacts.trajectory_path if hasattr(artifacts, "trajectory_path") else ep_dir / "trajectory.csv"
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        phases: List[str] = []

        if traj_csv.exists():
            with open(traj_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    observations.append(_extract_obs_vec(row, scene))
                    actions.append(_extract_action(row, scene))
                    phases.append(row.get("phase", ""))

        # success key lives inside summary["metrics"] for both scenes
        m = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        if scene == "liquid":
            success = bool(m.get("object_lifted", False))
        else:
            success = bool(m.get("success", False))
        return {
            "observations": np.stack(observations, axis=0) if observations else np.zeros((0, 1), dtype=np.float32),
            "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 4), dtype=np.float32),
            "phases": phases,
            "success": success,
            "seed": seed,
            "step_count": len(phases),
        }
    except Exception as exc:
        print(f"  [collect] Episode seed={seed} FAILED: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def _write_hdf5(
    output_path: Path,
    episodes: List[Dict[str, Any]],
    scene: str,
    obs_dim: int,
    action_dim: int,
) -> None:
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required: pip install h5py")

    # Compute global normalization stats across all episodes
    all_obs = np.concatenate([ep["observations"] for ep in episodes if len(ep["observations"]) > 0], axis=0)
    all_act = np.concatenate([ep["actions"] for ep in episodes if len(ep["actions"]) > 0], axis=0)
    obs_min = all_obs.min(axis=0)
    obs_max = all_obs.max(axis=0)
    act_min = all_act.min(axis=0)
    act_max = all_act.max(axis=0)

    with h5py.File(str(output_path), "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["scene_type"] = scene
        meta.attrs["num_episodes"] = len(episodes)
        meta.attrs["obs_dim"] = obs_dim
        meta.attrs["action_dim"] = action_dim
        meta.attrs["obs_keys"] = "tool_pos,target_pos,fluid_or_smoke" if scene == "liquid" else "tool_pos,meat_pos,target_pos,smoke,gripper"
        meta.create_dataset("obs_min", data=obs_min)
        meta.create_dataset("obs_max", data=obs_max)
        meta.create_dataset("act_min", data=act_min)
        meta.create_dataset("act_max", data=act_max)

        for idx, ep in enumerate(episodes):
            grp = f.create_group(f"episode_{idx:04d}")
            obs_grp = grp.create_group("observations")
            obs_grp.create_dataset("obs_vector", data=ep["observations"].astype(np.float32), compression="gzip")
            act_grp = grp.create_group("actions")
            act_grp.create_dataset("action_vector", data=ep["actions"].astype(np.float32), compression="gzip")
            task_grp = grp.create_group("task_info")
            phases_encoded = np.array([p.encode("utf-8") for p in ep["phases"]], dtype=object)
            if len(phases_encoded) > 0:
                task_grp.create_dataset("phases", data=phases_encoded)
            task_grp.attrs["success"] = ep["success"]
            task_grp.attrs["seed"] = ep["seed"]
            task_grp.attrs["step_count"] = ep["step_count"]

    print(f"[collect] Saved {len(episodes)} episodes to {output_path}")
    success_count = sum(1 for ep in episodes if ep["success"])
    avg_len = np.mean([ep["step_count"] for ep in episodes]) if episodes else 0
    print(f"[collect] Success rate: {success_count}/{len(episodes)} ({100*success_count/max(len(episodes),1):.0f}%)")
    print(f"[collect] Average episode length: {avg_len:.1f} steps")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch demo data collection → HDF5")
    parser.add_argument("--scene", choices=["liquid", "grill"], required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds; if empty, uses range(num_episodes)",
    )
    parser.add_argument("--backend", choices=["auto", "mock", "isaac"], default="mock")
    parser.add_argument("--preset", default="default")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path (default: outputs/demo_<scene>.h5)",
    )
    parser.add_argument("--keep-tmp", action="store_true", help="Keep per-episode artifact directories")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not _HAS_H5PY:
        print("[collect] ERROR: h5py not installed. Run: pip install h5py", file=sys.stderr)
        return 1

    scene_module = _load_scene_module(args.scene)

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(args.num_episodes))

    output_path = args.output or Path(f"outputs/demo_{args.scene}.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[collect] Collecting {len(seeds)} episodes for scene={args.scene} backend={args.backend}")
    print(f"[collect] Seeds: {seeds[:8]}{'...' if len(seeds)>8 else ''}")
    print(f"[collect] Output: {output_path}")

    # Determine obs/action dimensions from CSV-based schema
    obs_dim = OBS_DIM[args.scene]
    action_dim = ACTION_DIM

    episodes: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="collect_demo_") as tmp_root:
        tmp_dir = Path(tmp_root)
        for i, seed in enumerate(seeds):
            print(f"[collect] Episode {i+1}/{len(seeds)} seed={seed} ...", flush=True, end=" ")
            ep = _run_episode(scene_module, args.scene, seed, args.backend, args.preset, tmp_dir)
            if ep is not None:
                episodes.append(ep)
                print(f"steps={ep['step_count']} success={ep['success']}")
            else:
                print("FAILED")

        if not episodes:
            print("[collect] ERROR: No successful episodes collected.", file=sys.stderr)
            return 1

        _write_hdf5(output_path, episodes, args.scene, obs_dim, action_dim)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
