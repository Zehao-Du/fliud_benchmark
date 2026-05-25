#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate trained diffusion policy in simulation environment.

Usage:
    python policy/eval_env.py --scene liquid --checkpoint policy/checkpoints/liquid/best.pt --num-episodes 10
    python policy/eval_env.py --scene grill  --checkpoint policy/checkpoints/grill/best.pt  --backend mock
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diffusion policy in sim environment.")
    parser.add_argument("--scene", choices=["liquid", "grill"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--backend", choices=["auto", "mock", "isaac"], default="mock")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-start", type=int, default=1000, help="Seeds offset (avoid overlap with training data)")
    parser.add_argument("--diffusion-steps", type=int, default=10, help="Denoising steps at inference (10 is fast, 100 is full quality)")
    return parser.parse_args()


def _load_scene_module(scene: str):
    repo_root = Path(__file__).resolve().parents[1]
    if scene == "liquid":
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py"
        name = "franka_stir_liquid_eval"
    else:
        path = repo_root / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py"
        name = "franka_meat_on_grill_eval"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _obs_to_vec(obs: Dict[str, Any], scene: str) -> np.ndarray:
    """Extract flat obs vector from live env.get_obs() dict (matches CSV-based training schema)."""
    tool_pos = obs.get("tool", {}).get("position", [0.0, 0.0, 0.0])
    target_pos = obs.get("task", {}).get("target_position", [0.0, 0.0, 0.0])
    if scene == "liquid":
        fluid = obs.get("fluid", {})
        centroid = fluid.get("centroid", [0.0, 0.0, 0.0])
        fill_ratio = float(fluid.get("fill_ratio", 1.0))
        return np.array([*tool_pos, *target_pos, *centroid, fill_ratio], dtype=np.float32)
    else:
        meat_pos = obs.get("meat", {}).get("position", [0.0, 0.0, 0.0])
        smoke = obs.get("smoke", {})
        smoke_centroid = smoke.get("centroid", [0.0, 0.0, 0.0])
        smoke_strength = float(smoke.get("strength", 0.0))
        gripper_open = float(obs.get("robot", {}).get("gripper_open", True))
        return np.array([*tool_pos, *meat_pos, *target_pos, *smoke_centroid, smoke_strength, gripper_open], dtype=np.float32)


def _action_to_env(action_vec: np.ndarray) -> Dict[str, Any]:
    """Convert 4-D action to environment-compatible dict."""
    return {
        "target_position": action_vec[:3].tolist(),
        "gripper_closed": bool(action_vec[3] > 0.5),
    }


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from policy.diffusion_policy import DiffusionPolicy

    ckpt = torch.load(str(args.checkpoint), map_location=args.device)
    policy = DiffusionPolicy(
        obs_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        obs_horizon=ckpt["obs_horizon"],
        pred_horizon=ckpt["pred_horizon"],
        embed_dim=ckpt["embed_dim"],
        unet_base_ch=ckpt["unet_base_ch"],
        num_diffusion_steps=ckpt["diffusion_steps"],
    ).to(args.device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    print(f"[eval] Loaded checkpoint: {args.checkpoint}  (epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss',0):.5f})")

    norm_stats = {k: v.to(args.device) for k, v in ckpt["norm_stats"].items()}
    obs_min = norm_stats["obs_min"]
    obs_max = norm_stats["obs_max"]
    act_min = norm_stats["act_min"]
    act_max = norm_stats["act_max"]

    scene_module = _load_scene_module(args.scene)

    results: List[Dict[str, Any]] = []
    for ep_idx in range(args.num_episodes):
        seed = args.seed_start + ep_idx
        print(f"[eval] Episode {ep_idx+1}/{args.num_episodes} seed={seed}", flush=True, end=" ")

        if args.scene == "liquid":
            config = scene_module.get_preset_config("default")
            if args.backend == "mock":
                config["scene"]["fluid"]["particle_spacing"] = 0.018
        else:
            config = scene_module.get_preset_config("default", {})
        config["seed"] = seed
        config["headless"] = True
        config["video"]["enabled"] = False

        if args.scene == "liquid":
            env = scene_module.FrankaStirLiquidEnv(config=config, backend=args.backend)
        else:
            env = scene_module.FrankaMeatOnGrillEnv(config=config, backend=args.backend)

        env.create_scene()
        env.reset_scene(seed)

        obs_horizon = ckpt["obs_horizon"]
        pred_horizon = ckpt["pred_horizon"]
        obs_buf: Deque[np.ndarray] = deque(maxlen=obs_horizon)

        # Fill buffer with initial obs
        init_obs = env.get_obs()
        init_vec = _obs_to_vec(init_obs, args.scene)
        for _ in range(obs_horizon):
            obs_buf.append(init_vec)

        action_queue: Deque[np.ndarray] = deque()
        success = False

        for step in range(args.max_steps):
            if not action_queue:
                obs_arr = np.stack(list(obs_buf), axis=0)  # (To, D)
                obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=args.device)
                # Normalize
                obs_t = 2.0 * (obs_t - obs_min) / (obs_max - obs_min).clamp(min=1e-6) - 1.0
                obs_batch = obs_t.unsqueeze(0)  # (1, To, D)
                pred = policy.predict_action(obs_batch, num_steps=args.diffusion_steps)[0]  # (Ta, action_dim)
                # Denormalize
                pred = ((pred + 1.0) / 2.0) * (act_max - act_min).clamp(min=1e-6) + act_min
                for a in pred.cpu().numpy():
                    action_queue.append(a)

            action = action_queue.popleft()
            env_action = _action_to_env(action)
            env.apply_tool_action(env_action)
            env.step_sim(1)

            obs = env.get_obs()
            obs_buf.append(_obs_to_vec(obs, args.scene))

            phase = obs["task"]["phase"]
            if args.scene == "liquid":
                success = phase == "done"
            else:
                success = bool(obs["task"].get("success", False))
            if success:
                break

        metrics = env.get_metrics()
        env.close()

        result = {"seed": seed, "success": success, "steps": step + 1, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        results.append(result)
        print(f"success={success}  steps={step+1}")

    success_rate = sum(1 for r in results if r["success"]) / max(len(results), 1)
    avg_steps = np.mean([r["steps"] for r in results])
    print(f"\n[eval] Success rate: {100*success_rate:.0f}%  Avg steps: {avg_steps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
