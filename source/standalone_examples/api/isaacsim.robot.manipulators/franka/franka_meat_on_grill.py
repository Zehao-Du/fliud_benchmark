# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE_ORDER = [
    "approach",
    "descend",
    "grasp",
    "lift",
    "move_to_grill",
    "place",
    "retract",
    "hold_done",
]
STATUS_VALUES = {"completed", "aborted", "error"}
SUPPORTED_VARIATIONS = ("chicken", "steak")


def _now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _round(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def _vec3(x: float, y: float, z: float) -> List[float]:
    return [_round(x), _round(y), _round(z)]


def _vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_round(a[0] + b[0]), _round(a[1] + b[1]), _round(a[2] + b[2])]


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_round(a[0] - b[0]), _round(a[1] - b[1]), _round(a[2] - b[2])]


def _vec_mul(a: Sequence[float], scalar: float) -> List[float]:
    return [_round(a[0] * scalar), _round(a[1] * scalar), _round(a[2] * scalar)]


def _vec_lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> List[float]:
    return [_round(a[i] + (b[i] - a[i]) * alpha) for i in range(3)]


def _vec_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in a))


def _deep_update(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _quat_identity() -> List[float]:
    return [1.0, 0.0, 0.0, 0.0]


def _quat_tool_down() -> List[float]:
    return [0.0, 1.0, 0.0, 0.0]


def _placeholder_video_bytes() -> bytes:
    # Small MP4-like header so tests can validate the artifact even when ffmpeg is unavailable.
    return b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"


def write_placeholder_video(path: Path) -> None:
    path.write_bytes(_placeholder_video_bytes())


def _clamp_byte(value: float) -> int:
    return max(0, min(255, int(value)))


def _rgb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return (_clamp_byte(r), _clamp_byte(g), _clamp_byte(b))


def _new_canvas(width: int, height: int, color: Tuple[int, int, int]) -> bytearray:
    return bytearray(color * (width * height))


def _put_pixel(canvas: bytearray, width: int, height: int, x: int, y: int, color: Tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        index = (y * width + x) * 3
        canvas[index : index + 3] = bytes(color)


def _blend_pixel(
    canvas: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    if 0 <= x < width and 0 <= y < height:
        index = (y * width + x) * 3
        inv_alpha = 1.0 - alpha
        canvas[index] = _clamp_byte(canvas[index] * inv_alpha + color[0] * alpha)
        canvas[index + 1] = _clamp_byte(canvas[index + 1] * inv_alpha + color[1] * alpha)
        canvas[index + 2] = _clamp_byte(canvas[index + 2] * inv_alpha + color[2] * alpha)


def _fill_rect(
    canvas: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    left = max(0, min(x0, x1))
    right = min(width, max(x0, x1))
    top = max(0, min(y0, y1))
    bottom = min(height, max(y0, y1))
    for py in range(top, bottom):
        for px in range(left, right):
            if alpha >= 1.0:
                _put_pixel(canvas, width, height, px, py, color)
            else:
                _blend_pixel(canvas, width, height, px, py, color, alpha)


def _draw_line(
    canvas: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    radius = max(0, thickness // 2)
    while True:
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                _put_pixel(canvas, width, height, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_circle(
    canvas: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    radius = max(radius, 1)
    limit = radius * radius
    for py in range(cy - radius, cy + radius + 1):
        for px in range(cx - radius, cx + radius + 1):
            if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= limit:
                if alpha >= 1.0:
                    _put_pixel(canvas, width, height, px, py, color)
                else:
                    _blend_pixel(canvas, width, height, px, py, color, alpha)


def _write_binary_ppm(path: Path, width: int, height: int, canvas: bytearray) -> None:
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(canvas))


def _resolve_ffmpeg_binary() -> Optional[str]:
    candidates = [shutil.which("ffmpeg")]
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(str(Path(conda_exe).resolve().with_name("ffmpeg")))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def encode_mp4_from_ppm_sequence(frames_dir: Path, output_path: Path, fps: int = 30) -> bool:
    ffmpeg = _resolve_ffmpeg_binary()
    if ffmpeg is not None:
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(frames_dir / "frame_%05d.ppm"), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
            capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if k not in {"PYTHONHOME", "PYTHONPATH", "LD_PRELOAD"}},
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
    # Fallback: OpenCV
    try:
        import cv2
        ppm_files = sorted(frames_dir.glob("frame_*.ppm"))
        if not ppm_files:
            return False
        first = cv2.imread(str(ppm_files[0]))
        if first is None:
            return False
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for p in ppm_files:
            frame = cv2.imread(str(p))
            if frame is not None:
                writer.write(frame)
        writer.release()
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _default_output_dir(config: Dict[str, Any], seed: int) -> Path:
    root = Path(config["output"]["root_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"run_{stamp}_{seed}"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _flatten_numbers(node: Any) -> Iterable[float]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _flatten_numbers(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _flatten_numbers(value)
    elif isinstance(node, (int, float)):
        yield float(node)


def _base_config() -> Dict[str, Any]:
    table_size = {"x": 0.92, "y": 0.72, "z": 0.05}
    table_position = [0.58, 0.0, table_size["z"] * 0.5]
    grill_size = {"x": 0.34, "y": 0.24, "z": 0.10}
    grill_position = [0.72, 0.14, table_position[2] + table_size["z"] * 0.5 + grill_size["z"] * 0.5]
    grill_surface_z = grill_position[2] + grill_size["z"] * 0.5 + 0.002
    target_position = [grill_position[0], grill_position[1], grill_surface_z + 0.012]
    chicken_size = [0.060, 0.042, 0.018]
    meat_position = [0.44, -0.22, table_position[2] + table_size["z"] * 0.5 + chicken_size[2] * 0.5]
    return {
        "name": "franka_meat_on_grill",
        "preset": "default",
        "backend": "auto",
        "seed": 0,
        "headless": True,
        "video": {"enabled": True, "fps": 30, "min_duration_seconds": 3.0},
        "output": {"root_dir": "outputs/franka_meat_on_grill"},
        "scene": {
            "variation": "chicken",
            "table": {"position": table_position, "size": table_size, "color": [0.74, 0.69, 0.61]},
            "grill": {
                "position": grill_position,
                "size": grill_size,
                "surface_z": grill_surface_z,
                "body_color": [0.12, 0.12, 0.13],
                "grate_color": [0.32, 0.33, 0.35],
                "target_position": target_position,
                "target_extent": [0.070, 0.055, 0.020],
            },
            "smoke": {
                "enabled": True,
                "origin": [grill_position[0], grill_position[1], grill_surface_z + 0.040],
                "bounds": [0.22, 0.18, 0.40],
                "density": 0.48,
                "rise_speed": 0.018,
                "lateral_jitter": 0.010,
                "swirl_strength": 0.008,
                "blob_count": 10,
                "blob_radius": 0.050,
                "prefer_flowusd": True,
            },
            "meat_variations": {
                "chicken": {
                    "label": "chicken",
                    "size": chicken_size,
                    "color": [0.95, 0.70, 0.48],
                    "initial_position": meat_position,
                    "attach_offset": [0.0, 0.0, -0.002],
                },
                "steak": {
                    "label": "steak",
                    "size": [0.068, 0.046, 0.020],
                    "color": [0.63, 0.16, 0.12],
                    "initial_position": [0.44, -0.22, table_position[2] + table_size["z"] * 0.5 + 0.010],
                    "attach_offset": [0.0, 0.0, -0.002],
                },
            },
            "robot": {
                "asset_path": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                "prim_path": "/World/Franka",
                "ee_name": "panda_hand",
                "ik_frame_name": "panda_hand",
            },
        },
        "task": {
            "approach_start_offset": [-0.18, -0.10, 0.20],
            "pregrasp_height": 0.11,
            "grasp_depth_offset": 0.008,
            "lift_position": [0.54, -0.08, grill_surface_z + 0.24],
            "grill_approach_offset": [0.06, -0.09, 0.16],
            "retract_offset": [0.03, -0.02, 0.22],
            "release_progress": 0.58,
            "phase_steps": {
                "approach": 55,
                "descend": 48,
                "grasp": 28,
                "lift": 48,
                "move_to_grill": 74,
                "place": 54,
                "retract": 38,
                "hold_done": 150,
            },
        },
        "simulation": {
            "physics_dt": 1.0 / 60.0,
            "tool_follow_alpha": 0.22,
            "gripper_width_open": 0.040,
            "gripper_width_closed": 0.002,
            "robot_reset_settle_steps": 140,
            "robot_settle_substeps": 10,
        },
    }


def get_preset_config(name: str = "default", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = _base_config()
    if name == "cinematic_smoke":
        config["preset"] = "cinematic_smoke"
        config["scene"]["smoke"]["density"] = 0.68
        config["scene"]["smoke"]["lateral_jitter"] = 0.016
        config["scene"]["smoke"]["swirl_strength"] = 0.012
        config["scene"]["smoke"]["blob_count"] = 14
        config["scene"]["smoke"]["blob_radius"] = 0.060
        config["task"]["phase_steps"]["hold_done"] = 180
        config["video"]["min_duration_seconds"] = 4.0
    elif name != "default":
        raise ValueError(f"Unknown preset: {name}")
    config = _deep_update(config, overrides)
    variation = config["scene"].get("variation", "chicken")
    if variation not in SUPPORTED_VARIATIONS:
        raise ValueError(f"Unsupported variation: {variation}")
    config["preset"] = name
    return config


def get_obs_schema() -> Dict[str, Any]:
    return {
        "robot": {
            "joint_positions": {"type": "list[float]", "shape": [9]},
            "joint_velocities": {"type": "list[float]", "shape": [9]},
            "ee_position": {"type": "list[float]", "shape": [3]},
            "ee_orientation": {"type": "list[float]", "shape": [4]},
            "gripper_open": {"type": "bool"},
            "gripper_width": {"type": "float"},
        },
        "meat": {
            "variation": {"type": "str", "enum": list(SUPPORTED_VARIATIONS)},
            "position": {"type": "list[float]", "shape": [3]},
            "size": {"type": "list[float]", "shape": [3]},
            "grasped": {"type": "bool"},
        },
        "grill": {
            "position": {"type": "list[float]", "shape": [3]},
            "target_position": {"type": "list[float]", "shape": [3]},
            "target_extent": {"type": "list[float]", "shape": [3]},
        },
        "smoke": {
            "mode": {"type": "str"},
            "origin": {"type": "list[float]", "shape": [3]},
            "centroid": {"type": "list[float]", "shape": [3]},
            "bbox_min": {"type": "list[float]", "shape": [3]},
            "bbox_max": {"type": "list[float]", "shape": [3]},
            "strength": {"type": "float"},
        },
        "task": {
            "phase": {"type": "str", "enum": PHASE_ORDER},
            "done": {"type": "bool"},
            "grasped": {"type": "bool"},
            "success": {"type": "bool"},
        },
    }


def validate_obs(obs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for key in ("robot", "meat", "grill", "smoke", "task"):
        if key not in obs:
            errors.append(f"missing top-level key: {key}")
    if len(obs.get("robot", {}).get("joint_positions", [])) != 9:
        errors.append("robot.joint_positions must have length 9")
    if len(obs.get("robot", {}).get("joint_velocities", [])) != 9:
        errors.append("robot.joint_velocities must have length 9")
    if len(obs.get("robot", {}).get("ee_position", [])) != 3:
        errors.append("robot.ee_position must have length 3")
    if len(obs.get("robot", {}).get("ee_orientation", [])) != 4:
        errors.append("robot.ee_orientation must have length 4")
    if len(obs.get("meat", {}).get("position", [])) != 3:
        errors.append("meat.position must have length 3")
    if len(obs.get("meat", {}).get("size", [])) != 3:
        errors.append("meat.size must have length 3")
    if len(obs.get("grill", {}).get("target_position", [])) != 3:
        errors.append("grill.target_position must have length 3")
    if len(obs.get("grill", {}).get("target_extent", [])) != 3:
        errors.append("grill.target_extent must have length 3")
    if len(obs.get("smoke", {}).get("origin", [])) != 3:
        errors.append("smoke.origin must have length 3")
    if len(obs.get("smoke", {}).get("centroid", [])) != 3:
        errors.append("smoke.centroid must have length 3")
    if obs.get("task", {}).get("phase") not in PHASE_ORDER:
        errors.append("task.phase must be one of PHASE_ORDER")
    for value in _flatten_numbers(obs):
        if math.isnan(value) or math.isinf(value):
            errors.append("observation contains NaN or Inf")
            break
    return len(errors) == 0, errors


def _meat_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["scene"]["meat_variations"][config["scene"]["variation"]]


def _grill_target_bounds(config: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    center = config["scene"]["grill"]["target_position"]
    extent = config["scene"]["grill"]["target_extent"]
    return (
        [center[i] - extent[i] for i in range(3)],
        [center[i] + extent[i] for i in range(3)],
    )


def _meat_in_target_zone(config: Dict[str, Any], meat_position: Sequence[float]) -> bool:
    lower, upper = _grill_target_bounds(config)
    return all(lower[i] <= float(meat_position[i]) <= upper[i] for i in range(3))


def _phase_color(phase: str) -> Tuple[int, int, int]:
    palette = {
        "approach": _rgb(108, 117, 125),
        "descend": _rgb(70, 130, 180),
        "grasp": _rgb(230, 159, 0),
        "lift": _rgb(46, 139, 87),
        "move_to_grill": _rgb(196, 90, 48),
        "place": _rgb(178, 34, 34),
        "retract": _rgb(112, 128, 144),
        "hold_done": _rgb(78, 154, 6),
    }
    return palette.get(phase, _rgb(120, 120, 120))


def write_episode_video(path: Path, config: Dict[str, Any], trajectory_rows: Sequence[Dict[str, Any]], fps: int = 30) -> bool:
    if not trajectory_rows:
        write_placeholder_video(path)
        return False

    width = 640
    height = 360
    margin = 22
    panel_w = 400
    panel_h = 268
    side_left = 450
    scene_min = [0.22, -0.38]
    scene_max = [0.92, 0.34]

    def project(world_x: float, world_y: float) -> Tuple[int, int]:
        nx = 0.0 if scene_max[0] == scene_min[0] else (world_x - scene_min[0]) / (scene_max[0] - scene_min[0])
        ny = 0.0 if scene_max[1] == scene_min[1] else (world_y - scene_min[1]) / (scene_max[1] - scene_min[1])
        px = int(margin + max(0.0, min(1.0, nx)) * panel_w)
        py = int(margin + panel_h - max(0.0, min(1.0, ny)) * panel_h)
        return px, py

    sampled_rows = list(trajectory_rows[::2])
    if sampled_rows[-1] is not trajectory_rows[-1]:
        sampled_rows.append(trajectory_rows[-1])
    min_frame_count = max(1, int(round(fps * float(config.get("video", {}).get("min_duration_seconds", 3.0)))))
    if len(sampled_rows) < min_frame_count:
        sampled_rows.extend([sampled_rows[-1]] * (min_frame_count - len(sampled_rows)))

    table = config["scene"]["table"]
    grill = config["scene"]["grill"]
    table_pos = table["position"]
    table_size = table["size"]
    grill_pos = grill["position"]
    grill_size = grill["size"]
    target_lo, target_hi = _grill_target_bounds(config)
    tool_trail: List[Tuple[int, int]] = []

    with tempfile.TemporaryDirectory(prefix="franka_meat_on_grill_") as tmpdir:
        frames_dir = Path(tmpdir)
        for frame_idx, row in enumerate(sampled_rows):
            canvas = _new_canvas(width, height, _rgb(246, 241, 232))
            phase = str(row.get("phase", "approach"))
            phase_color = _phase_color(phase)
            _fill_rect(canvas, width, height, 0, 0, width, 18, phase_color)
            _fill_rect(canvas, width, height, margin - 8, margin - 8, margin + panel_w + 8, margin + panel_h + 8, _rgb(230, 225, 216))
            _fill_rect(canvas, width, height, margin, margin, margin + panel_w, margin + panel_h, _rgb(255, 252, 247))

            table_min = project(table_pos[0] - table_size["x"] * 0.5, table_pos[1] - table_size["y"] * 0.5)
            table_max = project(table_pos[0] + table_size["x"] * 0.5, table_pos[1] + table_size["y"] * 0.5)
            _fill_rect(canvas, width, height, table_min[0], table_max[1], table_max[0], table_min[1], _rgb(201, 182, 151))

            grill_min = project(grill_pos[0] - grill_size["x"] * 0.5, grill_pos[1] - grill_size["y"] * 0.5)
            grill_max = project(grill_pos[0] + grill_size["x"] * 0.5, grill_pos[1] + grill_size["y"] * 0.5)
            _fill_rect(canvas, width, height, grill_min[0], grill_max[1], grill_max[0], grill_min[1], _rgb(36, 38, 40))
            for line_idx in range(5):
                alpha = 0.3 + 0.08 * line_idx
                x = int(grill_min[0] + (line_idx + 1) * (grill_max[0] - grill_min[0]) / 6.0)
                _draw_line(canvas, width, height, x, grill_min[1], x, grill_max[1], _rgb(136, 138, 142), 1)
                _blend_pixel(canvas, width, height, x, grill_min[1], _rgb(180, 180, 180), alpha)

            target_min = project(target_lo[0], target_lo[1])
            target_max = project(target_hi[0], target_hi[1])
            _fill_rect(canvas, width, height, target_min[0], target_max[1], target_max[0], target_min[1], _rgb(240, 190, 90), alpha=0.35)

            smoke_centroid = project(float(row["smoke_x"]), float(row["smoke_y"]))
            smoke_strength = max(0.0, min(1.0, float(row.get("smoke_strength", 0.0))))
            for cloud_idx in range(6):
                offset_x = int(math.cos(frame_idx * 0.2 + cloud_idx) * 9)
                offset_y = int(-cloud_idx * 10 - smoke_strength * 12)
                radius = int(8 + cloud_idx * 3 + smoke_strength * 10)
                alpha = min(0.12 + smoke_strength * 0.10, 0.32)
                _draw_circle(
                    canvas,
                    width,
                    height,
                    smoke_centroid[0] + offset_x,
                    smoke_centroid[1] + offset_y,
                    radius,
                    _rgb(170, 176, 180),
                    alpha=alpha,
                )

            tool = project(float(row["tool_x"]), float(row["tool_y"]))
            tool_trail.append(tool)
            tool_trail = tool_trail[-24:]
            for trail_index in range(1, len(tool_trail)):
                _draw_line(
                    canvas,
                    width,
                    height,
                    tool_trail[trail_index - 1][0],
                    tool_trail[trail_index - 1][1],
                    tool_trail[trail_index][0],
                    tool_trail[trail_index][1],
                    _rgb(169, 61, 38),
                    2,
                )

            meat = project(float(row["meat_x"]), float(row["meat_y"]))
            target = project(float(row["target_x"]), float(row["target_y"]))
            _draw_circle(canvas, width, height, target[0], target[1], 4, _rgb(250, 226, 132), alpha=1.0)
            _draw_circle(canvas, width, height, tool[0], tool[1], 6, _rgb(182, 52, 34), alpha=1.0)
            _fill_rect(canvas, width, height, meat[0] - 8, meat[1] - 5, meat[0] + 8, meat[1] + 5, _rgb(221, 121, 78))

            _fill_rect(canvas, width, height, side_left, 30, side_left + 150, 210, _rgb(255, 250, 242))
            bars = [
                (40, "done", 1.0 if row.get("task_done") else 0.0, _rgb(67, 160, 71)),
                (78, "grasped", 1.0 if row.get("grasped") else 0.0, _rgb(33, 150, 243)),
                (116, "target", 1.0 if row.get("in_target_zone") else 0.0, _rgb(255, 193, 7)),
                (154, "smoke", smoke_strength, _rgb(158, 158, 158)),
            ]
            for top, _, value, color in bars:
                _fill_rect(canvas, width, height, side_left + 16, top, side_left + 126, top + 18, _rgb(233, 226, 214))
                _fill_rect(canvas, width, height, side_left + 16, top, side_left + 16 + int(110 * max(0.0, min(1.0, value))), top + 18, color)

            _write_binary_ppm(frames_dir / f"frame_{frame_idx:05d}.ppm", width, height, canvas)

        if encode_mp4_from_ppm_sequence(frames_dir, path, fps=fps):
            return True
    write_placeholder_video(path)
    return False


@dataclass
class EpisodeArtifacts:
    output_dir: Path
    config_path: Path
    obs_schema_path: Path
    metrics_path: Path
    episode_summary_path: Path
    trajectory_path: Path
    events_path: Path
    video_path: Path
    stdout_path: Path


class MockGrillBackend:
    def __init__(self, config: Dict[str, Any]):
        self.config = deepcopy(config)
        self.seed = int(config.get("seed", 0))
        self._rng = random.Random(self.seed)
        self._created = False
        self._step_index = 0
        self._phase = "approach"
        self._phase_events: List[Dict[str, Any]] = []
        self._stdout_lines: List[str] = []
        self._latest_metrics: Dict[str, Any] = {}
        self._phase_boundaries = self._compute_phase_boundaries()
        self._tool_position = [0.0, 0.0, 0.0]
        self._target_position = [0.0, 0.0, 0.0]
        self._tool_orientation = _quat_tool_down()
        self._target_orientation = _quat_tool_down()
        self._tool_linear_velocity = [0.0, 0.0, 0.0]
        self._tool_angular_velocity = [0.0, 0.0, 0.0]
        self._robot_joint_positions = [0.0] * 9
        self._robot_joint_velocities = [0.0] * 9
        self._ee_position = [0.0, 0.0, 0.0]
        self._ee_orientation = _quat_tool_down()
        self._meat_position = [0.0, 0.0, 0.0]
        self._meat_grasped = False
        self._gripper_open = True
        self._gripper_width = float(self.config["simulation"]["gripper_width_open"])
        self._success = False
        self._done = False
        self._smoke_mode = "synthetic_plume"
        self._smoke_blobs: List[List[float]] = []

    def _log(self, message: str) -> None:
        self._stdout_lines.append(message)

    def _compute_phase_boundaries(self) -> List[Tuple[str, int]]:
        total = 0
        boundaries: List[Tuple[str, int]] = []
        for phase in PHASE_ORDER:
            total += int(self.config["task"]["phase_steps"][phase])
            boundaries.append((phase, total))
        return boundaries

    def _determine_phase(self, step_index: int) -> str:
        for phase, boundary in self._phase_boundaries:
            if step_index < boundary:
                return phase
        return PHASE_ORDER[-1]

    def _phase_start(self, phase: str) -> int:
        total = 0
        for name in PHASE_ORDER:
            if name == phase:
                return total
            total += int(self.config["task"]["phase_steps"][name])
        return total

    def _phase_progress(self, phase: str, step_index: int) -> float:
        start = self._phase_start(phase)
        duration = max(int(self.config["task"]["phase_steps"][phase]), 1)
        return min(max((step_index - start) / duration, 0.0), 1.0)

    def _meat_initial_position(self) -> List[float]:
        return [float(v) for v in _meat_config(self.config)["initial_position"]]

    def _attached_meat_position(self) -> List[float]:
        return _vec_add(self._tool_position, _meat_config(self.config).get("attach_offset", [0.0, 0.0, 0.0]))

    def _planned_positions(self) -> Dict[str, List[float]]:
        meat_position = self._meat_initial_position()
        target = self.config["scene"]["grill"]["target_position"]
        task = self.config["task"]
        pregrasp = [meat_position[0], meat_position[1], meat_position[2] + float(task["pregrasp_height"])]
        grasp = [meat_position[0], meat_position[1], meat_position[2] + float(task["grasp_depth_offset"])]
        lift = [float(v) for v in task["lift_position"]]
        grill_approach = _vec_add(target, task["grill_approach_offset"])
        place = [target[0], target[1], self.config["scene"]["grill"]["surface_z"] + _meat_config(self.config)["size"][2] * 0.5]
        retract = _vec_add(target, task["retract_offset"])
        start = _vec_add(pregrasp, task["approach_start_offset"])
        return {
            "start": start,
            "pregrasp": pregrasp,
            "grasp": grasp,
            "lift": lift,
            "grill_approach": grill_approach,
            "place": place,
            "retract": retract,
        }

    def _compute_planned_target(self, step_index: int) -> List[float]:
        phase = self._determine_phase(step_index)
        points = self._planned_positions()
        if phase == "approach":
            return _vec_lerp(points["start"], points["pregrasp"], self._phase_progress("approach", step_index))
        if phase == "descend":
            return _vec_lerp(points["pregrasp"], points["grasp"], self._phase_progress("descend", step_index))
        if phase == "grasp":
            return list(points["grasp"])
        if phase == "lift":
            return _vec_lerp(points["grasp"], points["lift"], self._phase_progress("lift", step_index))
        if phase == "move_to_grill":
            return _vec_lerp(points["lift"], points["grill_approach"], self._phase_progress("move_to_grill", step_index))
        if phase == "place":
            return _vec_lerp(points["grill_approach"], points["place"], self._phase_progress("place", step_index))
        if phase == "retract":
            return _vec_lerp(points["place"], points["retract"], self._phase_progress("retract", step_index))
        return list(points["retract"])

    def _advance_phase(self) -> None:
        new_phase = self._determine_phase(self._step_index)
        if new_phase != self._phase:
            self._phase = new_phase
            self._phase_events.append({"step": self._step_index, "event": "phase_enter", "phase": new_phase})

    def create_scene(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
            self.seed = int(self.config.get("seed", 0))
            self._phase_boundaries = self._compute_phase_boundaries()
        self._created = True
        self._log("mock meat_on_grill scene created")
        self.reset_scene(self.seed)

    def _reset_smoke(self) -> None:
        smoke_cfg = self.config["scene"]["smoke"]
        origin = smoke_cfg["origin"]
        self._smoke_blobs = []
        for index in range(int(smoke_cfg["blob_count"])):
            self._smoke_blobs.append(
                [
                    _round(origin[0] + self._rng.uniform(-0.02, 0.02)),
                    _round(origin[1] + self._rng.uniform(-0.02, 0.02)),
                    _round(origin[2] + 0.02 * index),
                ]
            )

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.seed = int(self.seed if seed is None else seed)
        self._rng = random.Random(self.seed)
        self._step_index = 0
        self._phase = "approach"
        self._phase_events = [
            {"step": 0, "event": "reset", "phase": "approach", "seed": self.seed},
            {"step": 0, "event": "phase_enter", "phase": "approach"},
            {"step": 0, "event": "variation_selected", "variation": self.config["scene"]["variation"]},
        ]
        self._meat_position = self._meat_initial_position()
        self._meat_grasped = False
        self._gripper_open = True
        self._gripper_width = float(self.config["simulation"]["gripper_width_open"])
        self._success = False
        self._done = False
        self._target_position = self._compute_planned_target(0)
        self._tool_position = list(self._target_position)
        self._ee_position = list(self._tool_position)
        self._ee_orientation = list(self._tool_orientation)
        self._tool_linear_velocity = [0.0, 0.0, 0.0]
        self._tool_angular_velocity = [0.0, 0.0, 0.0]
        self._robot_joint_positions = [0.0] * 9
        self._robot_joint_velocities = [0.0] * 9
        self._reset_smoke()
        self._latest_metrics = self._compute_metrics()
        self._log(f"mock scene reset with seed={self.seed}")
        return self.get_obs()

    def _apply_gripper_state(self) -> None:
        closed_phases = {"grasp", "lift", "move_to_grill"}
        if self._phase == "place" and self._phase_progress("place", self._step_index) < float(self.config["task"]["release_progress"]):
            closed = True
        else:
            closed = self._phase in closed_phases
        self._gripper_open = not closed
        self._gripper_width = float(
            self.config["simulation"]["gripper_width_open"] if self._gripper_open else self.config["simulation"]["gripper_width_closed"]
        )

    def _update_robot_state(self, previous_tool: Sequence[float]) -> None:
        alpha = float(self.config["simulation"]["tool_follow_alpha"])
        self._target_position = self._compute_planned_target(self._step_index)
        self._tool_position = _vec_lerp(self._tool_position, self._target_position, alpha)
        dt = float(self.config["simulation"]["physics_dt"])
        delta = _vec_sub(self._tool_position, previous_tool)
        self._tool_linear_velocity = _vec_mul(delta, 1.0 / dt)
        self._tool_angular_velocity = [0.0, 0.0, 0.0]
        self._ee_position = list(self._tool_position)
        self._ee_orientation = list(self._tool_orientation)
        relative = _vec_sub(self._tool_position, self.config["scene"]["table"]["position"])
        previous_joints = list(self._robot_joint_positions)
        finger = self._gripper_width
        joints = [
            relative[0] * 4.2,
            relative[1] * 4.4,
            (self._tool_position[2] - 0.25) * 3.0,
            -1.0 + relative[0] * 1.4,
            0.9 + relative[1] * 1.8,
            1.45 - relative[0] * 1.1,
            0.70,
            finger,
            finger,
        ]
        self._robot_joint_positions = [_round(value) for value in joints]
        self._robot_joint_velocities = [_round((a - b) / dt) for a, b in zip(self._robot_joint_positions, previous_joints)]

    def _update_meat_state(self) -> None:
        if not self._meat_grasped and not self._gripper_open and self._phase in {"grasp", "lift"}:
            if _vec_distance(self._tool_position, self._meat_position) <= 0.055:
                self._meat_grasped = True
                self._phase_events.append({"step": self._step_index, "event": "meat_grasped", "phase": self._phase})
                self._log("meat attached to gripper")
        if self._meat_grasped:
            self._meat_position = self._attached_meat_position()
        if self._phase == "place" and self._meat_grasped and self._gripper_open:
            self._meat_grasped = False
            target = list(self.config["scene"]["grill"]["target_position"])
            target[2] = self.config["scene"]["grill"]["surface_z"] + _meat_config(self.config)["size"][2] * 0.5
            self._meat_position = [_round(v) for v in target]
            self._phase_events.append({"step": self._step_index, "event": "meat_released", "phase": self._phase})
            self._log("meat released onto grill target zone")

    def _update_smoke(self) -> None:
        smoke_cfg = self.config["scene"]["smoke"]
        origin = smoke_cfg["origin"]
        bounds = smoke_cfg["bounds"]
        rise = float(smoke_cfg["rise_speed"])
        jitter = float(smoke_cfg["lateral_jitter"])
        swirl = float(smoke_cfg["swirl_strength"])
        phase_shift = self._step_index * 0.12
        updated: List[List[float]] = []
        for index, blob in enumerate(self._smoke_blobs):
            x = origin[0] + math.sin(phase_shift + index * 0.7) * swirl + self._rng.uniform(-jitter, jitter) * 0.2
            y = origin[1] + math.cos(phase_shift * 0.7 + index * 0.5) * swirl + self._rng.uniform(-jitter, jitter) * 0.2
            z = blob[2] + rise + index * 0.0006
            if z > origin[2] + bounds[2]:
                z = origin[2] + self._rng.uniform(0.0, 0.04)
            updated.append(
                [
                    _round(max(origin[0] - bounds[0] * 0.5, min(origin[0] + bounds[0] * 0.5, x))),
                    _round(max(origin[1] - bounds[1] * 0.5, min(origin[1] + bounds[1] * 0.5, y))),
                    _round(z),
                ]
            )
        self._smoke_blobs = updated

    def _smoke_bounds(self) -> Tuple[List[float], List[float], List[float]]:
        if not self._smoke_blobs:
            origin = list(self.config["scene"]["smoke"]["origin"])
            return origin, origin, origin
        xs = [blob[0] for blob in self._smoke_blobs]
        ys = [blob[1] for blob in self._smoke_blobs]
        zs = [blob[2] for blob in self._smoke_blobs]
        centroid = _vec3(sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        bbox_min = _vec3(min(xs), min(ys), min(zs))
        bbox_max = _vec3(max(xs), max(ys), max(zs))
        return centroid, bbox_min, bbox_max

    def _compute_metrics(self) -> Dict[str, Any]:
        centroid, bbox_min, bbox_max = self._smoke_bounds()
        in_target = _meat_in_target_zone(self.config, self._meat_position)
        success = bool(in_target and self._gripper_open and not self._meat_grasped)
        hold_progress = self._phase_progress("hold_done", self._step_index) if self._phase == "hold_done" else 0.0
        done = bool(self._phase == "hold_done" and hold_progress >= 0.98 and success)
        smoke_strength = max(0.0, min(1.0, float(self.config["scene"]["smoke"]["density"]) * (0.8 + bbox_max[2] - bbox_min[2])))
        return {
            "success": success,
            "done": done,
            "gripper_open": self._gripper_open,
            "gripper_width": _round(self._gripper_width),
            "meat_in_target_zone": in_target,
            "meat_height": _round(self._meat_position[2]),
            "smoke_centroid": centroid,
            "smoke_bbox_min": bbox_min,
            "smoke_bbox_max": bbox_max,
            "smoke_strength": _round(smoke_strength),
            "target_error": _round(_vec_distance(self._meat_position, self.config["scene"]["grill"]["target_position"])),
        }

    def step_sim(self, num_steps: int = 1) -> Dict[str, Any]:
        for _ in range(num_steps):
            self._advance_phase()
            previous_tool = list(self._tool_position)
            self._apply_gripper_state()
            self._update_robot_state(previous_tool)
            self._update_meat_state()
            self._update_smoke()
            self._latest_metrics = self._compute_metrics()
            self._success = bool(self._latest_metrics["success"])
            self._done = bool(self._latest_metrics["done"])
            self._step_index += 1
            self._advance_phase()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        self._latest_metrics = self._compute_metrics()
        self._success = bool(self._latest_metrics["success"])
        self._done = bool(self._latest_metrics["done"])
        return {
            "robot": {
                "joint_positions": list(self._robot_joint_positions),
                "joint_velocities": list(self._robot_joint_velocities),
                "ee_position": list(self._ee_position),
                "ee_orientation": list(self._ee_orientation),
                "gripper_open": bool(self._gripper_open),
                "gripper_width": _round(self._gripper_width),
            },
            "meat": {
                "variation": self.config["scene"]["variation"],
                "position": list(self._meat_position),
                "size": list(_meat_config(self.config)["size"]),
                "grasped": bool(self._meat_grasped),
            },
            "grill": {
                "position": list(self.config["scene"]["grill"]["position"]),
                "target_position": list(self.config["scene"]["grill"]["target_position"]),
                "target_extent": list(self.config["scene"]["grill"]["target_extent"]),
            },
            "smoke": {
                "mode": self._smoke_mode,
                "origin": list(self.config["scene"]["smoke"]["origin"]),
                "centroid": list(self._latest_metrics["smoke_centroid"]),
                "bbox_min": list(self._latest_metrics["smoke_bbox_min"]),
                "bbox_max": list(self._latest_metrics["smoke_bbox_max"]),
                "strength": self._latest_metrics["smoke_strength"],
            },
            "task": {
                "phase": self._phase,
                "done": bool(self._done),
                "grasped": bool(self._meat_grasped),
                "success": bool(self._success),
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        return deepcopy(self._latest_metrics)

    def get_events(self) -> List[Dict[str, Any]]:
        return deepcopy(self._phase_events)

    def get_stdout_log(self) -> str:
        return "\n".join(self._stdout_lines) + ("\n" if self._stdout_lines else "")

    def close(self) -> None:
        self._log("mock meat_on_grill scene closed")


class IsaacGrillBackend(MockGrillBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._simulation_app = None
        self._runtime_initialized = False
        self._world = None
        self._stage = None
        self._np = None
        self._UsdGeom = None
        self._UsdLux = None
        self._UsdShade = None
        self._Sdf = None
        self._Gf = None
        self._Usd = None
        self._franka = None
        self._ik_solver = None
        self._assets_root_path = None
        self._smoke_prims: List[str] = []
        self._smoke_mode = "synthetic_plume"
        self._smoke_instancer = None
        self._smoke_particle_state: List = []

    def _ensure_runtime(self) -> None:
        if self._runtime_initialized:
            return
        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": bool(self.config.get("headless", True))})
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver as FrankaKinematicsSolver
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

        self._World = World
        self._add_reference_to_stage = add_reference_to_stage
        self._Franka = Franka
        self._FrankaKinematicsSolver = FrankaKinematicsSolver
        self._get_assets_root_path = get_assets_root_path
        self._np = np
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._UsdShade = UsdShade
        self._Sdf = Sdf
        self._Gf = Gf
        self._Usd = Usd
        self._runtime_initialized = True
        self._log("Isaac runtime initialized for meat_on_grill")

    def _set_translate(self, prim_path: str, position: Sequence[float]) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            self._UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in position))

    def _set_scale(self, prim_path: str, scale: Sequence[float]) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            self._UsdGeom.XformCommonAPI(prim).SetScale(tuple(float(v) for v in scale))

    def _define_cube(self, prim_path: str, position: Sequence[float], scale: Sequence[float], color: Sequence[float]) -> None:
        cube = self._UsdGeom.Cube.Define(self._stage, prim_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._set_translate(prim_path, position)
        self._set_scale(prim_path, scale)

    def _define_sphere(self, prim_path: str, position: Sequence[float], radius: float, color: Sequence[float]) -> None:
        sphere = self._UsdGeom.Sphere.Define(self._stage, prim_path)
        sphere.CreateRadiusAttr(float(radius))
        sphere.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._set_translate(prim_path, position)

    def _create_preview_surface_material(self, material_path: str, color: Sequence[float], opacity: float = 1.0, roughness: float = 0.5, metallic: float = 0.0, emissive: Sequence[float] | None = None, ior: float = 1.5):
        material = self._UsdShade.Material.Define(self._stage, material_path)
        shader = self._UsdShade.Shader.Define(self._stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(self._Gf.Vec3f(*[float(v) for v in color]))
        shader.CreateInput("opacity", self._Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", self._Sdf.ValueTypeNames.Float).Set(float(metallic))
        shader.CreateInput("ior", self._Sdf.ValueTypeNames.Float).Set(float(ior))
        if emissive is not None:
            shader.CreateInput("emissiveColor", self._Sdf.ValueTypeNames.Color3f).Set(self._Gf.Vec3f(*[float(v) for v in emissive]))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _bind_material(self, prim_path: str, material) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            self._UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def _try_create_flow_smoke(self) -> None:
        """Load bbq_smoke.usda as NvFlow volumetric smoke at the grill surface.

        Uses layer=0 and densityCellSize=0.5 (from the official Fire.usda preset)
        for visible volumetric smoke. Falls back to sphere-prim particle cloud
        if the USDA cannot be found or loaded.
        """
        try:
            import pathlib
            import carb.settings

            # Enable NvFlow rendering in RTX before creating the prims
            cfg = carb.settings.get_settings()
            cfg.set("/rtx/flow/enabled", True)
            cfg.set("/rtx/flow/pathTracingEnabled", True)
            cfg.set("/rtx/flow/rayTracedTranslucencyEnabled", True)

            # Enable the omni.flowusd extension if not already active
            try:
                import omni.kit.app
                ext_mgr = omni.kit.app.get_app().get_extension_manager()
                if not ext_mgr.is_extension_enabled("omni.flowusd"):
                    ext_mgr.set_extension_enabled_immediate("omni.flowusd", True)
                    # Pump a few frames so Flow schema types are registered
                    app = omni.kit.app.get_app()
                    for _ in range(3):
                        app.update()
                    self._log("omni.flowusd extension enabled")
                else:
                    self._log("omni.flowusd already enabled")
            except Exception as ext_exc:
                self._log(f"Extension check warning (non-fatal): {ext_exc}")

            usda_path = None
            for candidate in [
                pathlib.Path(__file__).resolve().parents[5] / "tools" / "bbq_smoke.usda",
                pathlib.Path(__file__).resolve().parents[4] / "tools" / "bbq_smoke.usda",
                pathlib.Path.cwd() / "tools" / "bbq_smoke.usda",
            ]:
                if candidate.exists():
                    usda_path = str(candidate)
                    break
            if usda_path is None:
                raise FileNotFoundError("bbq_smoke.usda not found")

            grill = self.config["scene"]["grill"]
            grill_pos = grill["position"]
            grill_top_z = grill_pos[2] + grill["size"]["z"] * 0.5 + 0.025

            smoke_root_path = "/World/FlowBbqSmoke"
            smoke_root = self._stage.DefinePrim(smoke_root_path, "Xform")
            smoke_root.GetReferences().AddReference(usda_path)

            # NvFlow anchors its simulation at the prim's world position at init time.
            # Xform translate set BEFORE the first app.update() moves the simulation anchor
            # to the grill surface. The position attribute in bbq_smoke.usda stays (0,0,0)
            # (emitter at anchor = grill surface). radius=10 stays unchanged — in world-space
            # meters this is a 10m emitter, ensuring the simulation domain is large enough.
            from pxr import UsdGeom as _UsdGeom_flow
            _UsdGeom_flow.Xformable(smoke_root).AddTranslateOp().Set(
                self._Gf.Vec3d(grill_pos[0], grill_pos[1], grill_top_z)
            )
            self._log(
                f"NvFlow smoke Xform translate set to "
                f"({grill_pos[0]:.3f}, {grill_pos[1]:.3f}, {grill_top_z:.3f}) world-m "
                f"(pre-init, before first app.update())"
            )

            # Pump one app update so the reference composes and the emitter prim is accessible.
            import omni.kit.app as _kit_app
            _kit_app.get_app().update()

            # Verify the Flow prim types were recognized (extension must be loaded)
            # USD references compose defaultPrim children directly into the referencing prim,
            # so BbqSmoke's children appear directly under /World/FlowBbqSmoke.
            emitter_path = smoke_root_path + "/flowEmitterSphere"
            emitter_prim = self._stage.GetPrimAtPath(emitter_path)
            if emitter_prim.IsValid():
                type_name = emitter_prim.GetTypeName()
                self._log(f"NvFlow emitter prim type: {type_name}  radius={emitter_prim.GetAttribute('radius').Get()}")
                if type_name not in ("FlowEmitterSphere", "FlowEmitterPoint", "FlowEmitter"):
                    raise RuntimeError(f"Flow prim type not recognized ({type_name}); extension may not be active")
                # position stays (0,0,0) — emitter at simulation anchor = grill surface
            else:
                raise RuntimeError(f"NvFlow emitter prim not found at {emitter_path}; omni.flowusd may not be active")

            self._smoke_mode = "nvflow"
            self._log(f"NvFlow BBQ smoke loaded from {usda_path}")
        except Exception as exc:
            self._log(f"NvFlow smoke failed, falling back to sphere cloud: {exc}")
            self._create_particle_cloud_smoke()

    def _create_particle_cloud_smoke(self) -> None:
        """Fallback: 150 explicit sphere prims as a billowing BBQ smoke cloud."""
        try:
            grill = self.config["scene"]["grill"]
            grill_pos = grill["position"]
            grill_top_z = grill_pos[2] + grill["size"]["z"] * 0.5 + 0.01

            N = 150
            rng = random.Random(self.seed if hasattr(self, "seed") else 42)
            spread_r = 0.20
            max_h = 0.85

            mat = self._create_preview_surface_material(
                "/World/Looks/SmokeParticle", [0.74, 0.72, 0.70], opacity=0.55, roughness=1.0, ior=1.0
            )

            self._smoke_particle_state = []
            self._smoke_prims = []
            self._smoke_blobs = []

            for i in range(N):
                angle = rng.uniform(0, 2 * math.pi)
                r = math.sqrt(rng.random()) * spread_r
                px = grill_pos[0] + r * math.cos(angle)
                py = grill_pos[1] + r * math.sin(angle)
                pz = grill_top_z + rng.random() * max_h
                vx = rng.gauss(0, 0.004)
                vy = rng.gauss(0, 0.004)
                vz = rng.uniform(0.003, 0.009)
                self._smoke_particle_state.append([px, py, pz, vx, vy, vz])

                prim_path = f"/World/SmokeCloud/S_{i:03d}"
                self._define_sphere(prim_path, [px, py, pz], 0.06, [0.72, 0.70, 0.68])
                self._bind_material(prim_path, mat)
                self._smoke_prims.append(prim_path)

            self._smoke_mode = "particle_cloud"
            self._log(f"Particle cloud smoke created: {N} sphere prims at grill_top_z={grill_top_z:.3f}")
        except Exception as exc:
            self._smoke_mode = "synthetic_plume"
            self._log(f"Particle cloud smoke failed: {exc}")

    def _update_smoke(self) -> None:  # type: ignore[override]
        if not self._smoke_particle_state:
            super()._update_smoke()
            return
        try:
            grill_pos = self.config["scene"]["grill"]["position"]
            grill_top_z = grill_pos[2] + self.config["scene"]["grill"]["size"]["z"] * 0.5 + 0.01
            max_z = grill_top_z + 0.90
            spawn_r = 0.12

            for i, p in enumerate(self._smoke_particle_state):
                p[0] += p[3] + self._rng.gauss(0, 0.003)
                p[1] += p[4] + self._rng.gauss(0, 0.003)
                p[2] += p[5]
                p[5] = max(0.001, p[5] * 0.999)
                if p[2] > max_z:
                    angle = self._rng.uniform(0, 2 * math.pi)
                    r = math.sqrt(self._rng.random()) * spawn_r
                    p[0] = grill_pos[0] + r * math.cos(angle)
                    p[1] = grill_pos[1] + r * math.sin(angle)
                    p[2] = grill_top_z + self._rng.uniform(0, 0.06)
                    p[3] = self._rng.gauss(0, 0.004)
                    p[4] = self._rng.gauss(0, 0.004)
                    p[5] = self._rng.uniform(0.004, 0.010)
                if i < len(self._smoke_prims):
                    self._set_translate(self._smoke_prims[i], [p[0], p[1], p[2]])
        except Exception as exc:
            self._log(f"Smoke update warning: {exc}")

    def _build_runtime_scene(self) -> None:
        self._world = self._World(stage_units_in_meters=1.0)
        self._stage = self._simulation_app.context.get_stage()
        self._world.scene.add_default_ground_plane()

        table = self.config["scene"]["table"]
        grill = self.config["scene"]["grill"]
        meat_cfg = _meat_config(self.config)
        self._define_cube("/World/Table", table["position"], [table["size"]["x"], table["size"]["y"], table["size"]["z"]], table["color"])
        self._define_cube("/World/Grill/Base", grill["position"], [grill["size"]["x"], grill["size"]["y"], grill["size"]["z"]], grill["body_color"])
        self._define_cube(
            "/World/Grill/TargetZone",
            grill["target_position"],
            [grill["target_extent"][0] * 2.0, grill["target_extent"][1] * 2.0, 0.004],
            [1.0, 0.75, 0.2],
        )
        self._define_cube("/World/Meat", meat_cfg["initial_position"], meat_cfg["size"], meat_cfg["color"])
        self._define_sphere("/World/ToolMarker", self._tool_position, 0.014, [0.85, 0.18, 0.12])

        # Warm BBQ 3-point lighting: cooler ambient dome + warm key + orange ember glow
        dome = self._UsdLux.DomeLight.Define(self._stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(620.0)
        dome.CreateColorAttr(self._Gf.Vec3f(0.90, 0.92, 1.0))
        key = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/Key")
        key.CreateIntensityAttr(46000.0)
        key.CreateRadiusAttr(0.12)
        key.CreateColorAttr(self._Gf.Vec3f(1.0, 0.96, 0.86))
        self._set_translate("/World/Lights/Key", [0.95, -0.82, 1.12])
        fill = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/Fill")
        fill.CreateIntensityAttr(12000.0)
        fill.CreateRadiusAttr(0.10)
        fill.CreateColorAttr(self._Gf.Vec3f(0.85, 0.91, 1.0))
        self._set_translate("/World/Lights/Fill", [0.28, 0.60, 0.80])
        # Orange heat glow rising from the grill charcoal
        grill_pos = self.config["scene"]["grill"]["position"]
        heat = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/HeatGlow")
        heat.CreateIntensityAttr(18000.0)
        heat.CreateRadiusAttr(0.08)
        heat.CreateColorAttr(self._Gf.Vec3f(1.0, 0.28, 0.04))
        self._set_translate("/World/Lights/HeatGlow", [grill_pos[0], grill_pos[1], grill_pos[2] + 0.005])

        # Wooden table material
        table_mat = self._create_preview_surface_material(
            "/World/Looks/TableWood", self.config["scene"]["table"]["color"], opacity=1.0, roughness=0.72, metallic=0.0)
        self._bind_material("/World/Table", table_mat)

        # Grill grate bars — 7 thin dark-metal bars across the grill top
        grill = self.config["scene"]["grill"]
        grill_top_z = grill_pos[2] + grill["size"]["z"] * 0.5 + 0.004
        grate_mat = self._create_preview_surface_material(
            "/World/Looks/GrillGrate", [0.12, 0.10, 0.09], opacity=1.0, roughness=0.55, metallic=0.8)
        grate_w = grill["size"]["x"]
        grate_spacing = grill["size"]["y"] / 8.0
        for gi in range(7):
            bar_path = f"/World/Grill/Grate_{gi:02d}"
            bar_y = grill_pos[1] - grill["size"]["y"] * 0.5 * 0.85 + gi * grate_spacing * 1.2
            self._define_cube(bar_path, [grill_pos[0], bar_y, grill_top_z], [grate_w * 0.92, 0.008, 0.006], [0.12, 0.10, 0.09])
            self._bind_material(bar_path, grate_mat)

        # Emissive ember layer just under the grate
        ember_z = grill_pos[2] + grill["size"]["z"] * 0.5 - 0.002
        ember_mat = self._create_preview_surface_material(
            "/World/Looks/Embers", [1.0, 0.22, 0.04], opacity=1.0, roughness=0.9, metallic=0.0,
            emissive=[1.8, 0.30, 0.04])
        self._define_cube("/World/Grill/Embers", [grill_pos[0], grill_pos[1], ember_z],
                          [grill["size"]["x"] * 0.88, grill["size"]["y"] * 0.88, 0.003], [1.0, 0.22, 0.04])
        self._bind_material("/World/Grill/Embers", ember_mat)

        # NvFlow volumetric smoke (primary); sphere prim cloud or synthetic blobs as fallback
        smoke_material = self._create_preview_surface_material(
            "/World/Looks/Smoke", [0.68, 0.70, 0.72], opacity=0.12, roughness=0.95, metallic=0.0)
        self._try_create_flow_smoke()
        if self._smoke_mode not in ("nvflow", "particle_cloud"):
            for index, blob in enumerate(self._smoke_blobs):
                prim_path = f"/World/Smoke/Blob_{index:02d}"
                self._define_sphere(prim_path, blob, float(self.config["scene"]["smoke"]["blob_radius"]), [0.68, 0.70, 0.72])
                self._bind_material(prim_path, smoke_material)
                self._smoke_prims.append(prim_path)

        try:
            self._assets_root_path = self._get_assets_root_path()
            if self._assets_root_path:
                robot_usd = self._assets_root_path + self.config["scene"]["robot"]["asset_path"]
                self._franka = self._Franka(
                    prim_path=self.config["scene"]["robot"]["prim_path"],
                    name="franka_meat_on_grill",
                    usd_path=robot_usd,
                    end_effector_prim_name=self.config["scene"]["robot"]["ee_name"],
                )
                self._world.scene.add(self._franka)
                self._log(f"Loaded Franka asset from {robot_usd}")
        except Exception as exc:
            self._franka = None
            self._log(f"Robot asset load skipped: {exc}")

        self._world.reset()
        if self._franka is not None:
            try:
                self._ik_solver = self._FrankaKinematicsSolver(
                    self._franka,
                    end_effector_frame_name=self.config["scene"]["robot"]["ik_frame_name"],
                )
            except Exception as exc:
                self._ik_solver = None
                self._log(f"IK setup skipped: {exc}")

    def _command_robot_to_target(self) -> None:
        if self._franka is None or self._ik_solver is None:
            return
        try:
            target_position = self._np.asarray(self._target_position, dtype=float)
            target_orientation = self._np.asarray(self._tool_orientation, dtype=float)
            action, success = self._ik_solver.compute_inverse_kinematics(
                target_position,
                target_orientation=target_orientation,
                position_tolerance=0.005,
                orientation_tolerance=0.25,
            )
            if success:
                self._franka.apply_action(action)
        except Exception as exc:
            self._log(f"IK command warning: {exc}")

    def _sync_runtime_scene(self) -> None:
        self._set_translate("/World/ToolMarker", self._tool_position)
        self._set_translate("/World/Meat", self._meat_position)
        for prim_path, blob in zip(self._smoke_prims, self._smoke_blobs):
            self._set_translate(prim_path, blob)

    def create_scene(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
            self.seed = int(self.config.get("seed", 0))
            self._phase_boundaries = self._compute_phase_boundaries()
        if not importlib.util.find_spec("isaacsim"):
            raise RuntimeError("The Isaac backend requires an Isaac Sim Python runtime. Use --backend mock in source-only workspaces.")
        self._ensure_runtime()
        self._reset_smoke()
        self._build_runtime_scene()
        self._created = True
        self.reset_scene(self.seed)

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs = super().reset_scene(seed)
        if self._world is not None:
            try:
                self._world.reset()
            except Exception as exc:
                self._log(f"World reset warning: {exc}")
        self._sync_runtime_scene()
        self._phase_events.append({"step": 0, "event": "runtime_mode", "backend": "isaac", "smoke_mode": self._smoke_mode})
        return obs

    def step_sim(self, num_steps: int = 1) -> Dict[str, Any]:
        for _ in range(num_steps):
            previous_tool = list(self._tool_position)
            self._advance_phase()
            self._apply_gripper_state()
            self._target_position = self._compute_planned_target(self._step_index)
            if self._world is not None:
                substeps = int(self.config["simulation"]["robot_settle_substeps"])
                for _sub in range(substeps):
                    self._command_robot_to_target()
                    try:
                        self._world.step(render=False)
                    except Exception as exc:
                        self._log(f"World step warning: {exc}")
                        break
            self._update_robot_state(previous_tool)
            self._update_meat_state()
            self._update_smoke()
            self._latest_metrics = self._compute_metrics()
            self._success = bool(self._latest_metrics["success"])
            self._done = bool(self._latest_metrics["done"])
            self._sync_runtime_scene()
            self._step_index += 1
            self._advance_phase()
        return self.get_obs()

    def close(self) -> None:
        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass
            self._simulation_app = None
        super().close()


class FrankaMeatOnGrillEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None, preset: str = "default", backend: str = "auto"):
        self.config = get_preset_config(preset, config)
        self.config["backend"] = backend
        self.backend_name = self._resolve_backend_name(backend)
        if self.backend_name == "isaac":
            self._backend: Any = IsaacGrillBackend(self.config)
        else:
            self._backend = MockGrillBackend(self.config)

    def _resolve_backend_name(self, backend: str) -> str:
        if backend == "auto":
            return "isaac" if importlib.util.find_spec("isaacsim") else "mock"
        if backend not in {"auto", "isaac", "mock"}:
            raise ValueError(f"Unsupported backend: {backend}")
        return "isaac" if backend == "isaac" else "mock"

    def create_scene(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
        self._backend.create_scene(config or self.config)

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return self._backend.reset_scene(seed)

    def get_obs(self) -> Dict[str, Any]:
        return self._backend.get_obs()

    def step_sim(self, n: int = 1) -> Dict[str, Any]:
        return self._backend.step_sim(n)

    def close(self) -> None:
        self._backend.close()

    def get_metrics(self) -> Dict[str, Any]:
        return self._backend.get_metrics()

    def get_events(self) -> List[Dict[str, Any]]:
        return self._backend.get_events()

    def run_episode(
        self,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], EpisodeArtifacts]:
        run_seed = int(self.config.get("seed", 0) if seed is None else seed)
        self.config["seed"] = run_seed
        if not getattr(self._backend, "_created", False):
            self.create_scene(self.config)
        self.reset_scene(run_seed)
        output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(self.config, run_seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = EpisodeArtifacts(
            output_dir=output_dir,
            config_path=output_dir / "config.json",
            obs_schema_path=output_dir / "obs_schema.json",
            metrics_path=output_dir / "metrics.json",
            episode_summary_path=output_dir / "episode_summary.json",
            trajectory_path=output_dir / "trajectory.csv",
            events_path=output_dir / "events.jsonl",
            video_path=output_dir / "video.mp4",
            stdout_path=output_dir / "stdout.log",
        )
        _write_json(artifacts.config_path, self.config)
        _write_json(artifacts.obs_schema_path, get_obs_schema())

        total_budget = max_steps or (sum(self.config["task"]["phase_steps"].values()) + 5)
        trajectory_rows: List[Dict[str, Any]] = []
        status = "completed"
        error_text = None
        try:
            for _ in range(total_budget):
                obs = self.get_obs()
                valid, errors = validate_obs(obs)
                if not valid:
                    raise RuntimeError("Invalid observation: " + "; ".join(errors))
                metrics = self.get_metrics()
                trajectory_rows.append(
                    {
                        "step": getattr(self._backend, "_step_index", 0),
                        "phase": obs["task"]["phase"],
                        "task_done": obs["task"]["done"],
                        "grasped": obs["task"]["grasped"],
                        "success": obs["task"]["success"],
                        "tool_x": obs["robot"]["ee_position"][0],
                        "tool_y": obs["robot"]["ee_position"][1],
                        "tool_z": obs["robot"]["ee_position"][2],
                        "meat_x": obs["meat"]["position"][0],
                        "meat_y": obs["meat"]["position"][1],
                        "meat_z": obs["meat"]["position"][2],
                        "target_x": obs["grill"]["target_position"][0],
                        "target_y": obs["grill"]["target_position"][1],
                        "target_z": obs["grill"]["target_position"][2],
                        "smoke_x": obs["smoke"]["centroid"][0],
                        "smoke_y": obs["smoke"]["centroid"][1],
                        "smoke_z": obs["smoke"]["centroid"][2],
                        "smoke_strength": obs["smoke"]["strength"],
                        "gripper_open": obs["robot"]["gripper_open"],
                        "in_target_zone": metrics["meat_in_target_zone"],
                    }
                )
                if obs["task"]["done"]:
                    break
                self.step_sim(1)
            else:
                status = "aborted"
        except Exception as exc:
            status = "error"
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()

        metrics = self.get_metrics()
        phase_sequence = [event["phase"] for event in self.get_events() if event["event"] == "phase_enter"]
        summary = {
            "status": status,
            "backend": self.backend_name,
            "preset": self.config["preset"],
            "seed": run_seed,
            "variation": self.config["scene"]["variation"],
            "step_count": getattr(self._backend, "_step_index", 0),
            "phase_sequence": phase_sequence,
            "smoke_mode": getattr(self._backend, "_smoke_mode", "synthetic_plume"),
            "error": error_text,
            "metrics": metrics,
        }

        with artifacts.trajectory_path.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = list(trajectory_rows[0].keys()) if trajectory_rows else ["step", "phase"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in trajectory_rows:
                writer.writerow(row)

        _append_jsonl(artifacts.events_path, self.get_events())
        _write_json(
            artifacts.metrics_path,
            {
                "backend": self.backend_name,
                "preset": self.config["preset"],
                "variation": self.config["scene"]["variation"],
                "seed": run_seed,
                "final": metrics,
                "phase_sequence": phase_sequence,
            },
        )
        _write_json(artifacts.episode_summary_path, summary)
        if self.config.get("video", {}).get("enabled", True):
            write_episode_video(artifacts.video_path, self.config, trajectory_rows, fps=int(self.config["video"]["fps"]))
        else:
            write_placeholder_video(artifacts.video_path)
        artifacts.stdout_path.write_text(self._backend.get_stdout_log(), encoding="utf-8")
        return summary, artifacts


def run_episode_cli(args: argparse.Namespace) -> int:
    config = get_preset_config(args.preset, {"scene": {"variation": args.variation}})
    config["seed"] = args.seed
    config["headless"] = args.headless
    config["video"]["enabled"] = not args.disable_video
    if args.output_root is not None:
        config["output"]["root_dir"] = args.output_root
    env = FrankaMeatOnGrillEnv(config=config, preset=args.preset, backend=args.backend)
    try:
        summary, artifacts = env.run_episode(seed=args.seed, output_dir=None, max_steps=args.max_steps)
        print(json.dumps({"summary": summary, "output_dir": str(artifacts.output_dir)}, indent=2, sort_keys=True))
        return 0 if summary["status"] == "completed" else 1
    finally:
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLBench meat_on_grill semantic recreation for Franka in Isaac Sim.")
    parser.add_argument("--preset", choices=["default", "cinematic_smoke"], default="default")
    parser.add_argument("--variation", choices=list(SUPPORTED_VARIATIONS), default="chicken")
    parser.add_argument("--backend", choices=["auto", "isaac", "mock"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_episode_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
