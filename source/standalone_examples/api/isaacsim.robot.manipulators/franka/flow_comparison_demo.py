# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_stir_module():
    module_path = Path(__file__).with_name("franka_stir_liquid.py")
    spec = importlib.util.spec_from_file_location("franka_stir_liquid_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_ppm(path: Path, width: int, height: int, color_shift: int) -> None:
    header = f"P3\n{width} {height}\n255\n"
    lines = [header]
    for y in range(height):
        row = []
        for x in range(width):
            r = (x * 3 + color_shift) % 256
            g = (y * 5 + color_shift * 2) % 256
            b = ((x + y) * 2 + color_shift * 3) % 256
            row.append(f"{r} {g} {b}")
        lines.append(" ".join(row) + "\n")
    path.write_text("".join(lines), encoding="ascii")


def run_flow_demo(output_root: Optional[Path] = None) -> Path:
    stir_module = _load_stir_module()
    output_dir = (output_root or Path("outputs/franka_stir_liquid")) / f"flow_comparison_demo_{_timestamp()}"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "title": "Flow vs PBD comparison for Franka stirring experiments",
        "candidates": [
            {
                "name": "PhysX PBD particles",
                "realtime": "good for the target MVP scale",
                "robot_collision_support": "direct rigid/particle interaction path",
                "api_difficulty": "medium",
                "stability": "good enough for repeatable metrics in MVP scope",
                "risk": "liquid realism is limited",
                "recommended_role": "main experiment line",
            },
            {
                "name": "Omni Flow / flowusd",
                "realtime": "good visual response for volumetric effects",
                "robot_collision_support": "weaker for direct articulated contact workflows",
                "api_difficulty": "higher for structured robotics metrics",
                "stability": "good for visuals, weaker for benchmark-style contact metrics",
                "risk": "integration cost and metric mismatch",
                "recommended_role": "comparison and visualization only",
            },
        ],
    }

    for index in range(8):
        _write_ppm(frames_dir / f"frame_{index:05d}.ppm", width=64, height=48, color_shift=index * 17)

    stir_module.encode_mp4_from_ppm_sequence(frames_dir, output_dir / "video.mp4", fps=8) or stir_module.write_placeholder_video(output_dir / "video.mp4")
    (output_dir / "comparison_table.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Flow Comparison Demo\n\n"
        "This source-only fallback writes a comparison table, a synthetic frame sequence, and encodes a playable MP4 artifact when ffmpeg is available.\n",
        encoding="utf-8",
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flow comparison demo for the Franka stirring plan.")
    parser.add_argument("--output-root", type=str, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = run_flow_demo(Path(args.output_root) if args.output_root else None)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
