#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train diffusion policy on collected demo data.

Usage:
    python policy/train.py --data outputs/demo_liquid.h5 --scene liquid --epochs 200
    python policy/train.py --data outputs/demo_grill.h5  --scene grill  --epochs 200 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion policy on demo data.")
    parser.add_argument("--data", type=Path, required=True, help="HDF5 dataset from collect_demo_data.py")
    parser.add_argument("--scene", choices=["liquid", "grill"], required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--unet-base-ch", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("policy/checkpoints"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Lazy imports to avoid polluting module-level
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from policy.dataset import DemoDataset
    from policy.diffusion_policy import DiffusionPolicy

    print(f"[train] Loading dataset: {args.data}")
    dataset = DemoDataset(args.data, obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon)
    print(f"[train] Dataset: {len(dataset)} windows, obs_dim={dataset.obs_dim}, action_dim={dataset.action_dim}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    policy = DiffusionPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        embed_dim=args.embed_dim,
        unet_base_ch=args.unet_base_ch,
        num_diffusion_steps=args.diffusion_steps,
    ).to(args.device)

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"[train] Policy parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = args.output / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss = 0.0
        for batch in loader:
            obs = batch["obs_seq"].to(args.device)
            act = batch["actions"].to(args.device)
            loss = policy.compute_loss(obs, act)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        train_curve.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[train] epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "loss": best_loss,
                    "model_state": policy.state_dict(),
                    "obs_dim": dataset.obs_dim,
                    "action_dim": dataset.action_dim,
                    "obs_horizon": args.obs_horizon,
                    "pred_horizon": args.pred_horizon,
                    "embed_dim": args.embed_dim,
                    "unet_base_ch": args.unet_base_ch,
                    "diffusion_steps": args.diffusion_steps,
                    "norm_stats": {k: v.cpu() for k, v in dataset.norm_stats.items()},
                    "scene": args.scene,
                },
                output_dir / "best.pt",
            )

    torch.save({"model_state": policy.state_dict()}, output_dir / "last.pt")
    (output_dir / "train_curve.json").write_text(json.dumps(train_curve), encoding="utf-8")
    print(f"[train] Done. Best loss={best_loss:.5f}  Saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
